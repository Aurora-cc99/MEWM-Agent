"""CLIP dual-tower localiser training (修改方案 §1 + §2 + §3 的落地).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ClipConfig, MEWMConfig, PACKAGE_ROOT, load_config
from ..engines.clip_motion_engine import (
    CLIPSpotterModel, head_feature_block, info_nce,
)
from ..engines.motion_description import frame_motion_description
from ..knowledge.au_anatomy import K_SLOTS
from .localiser_supervised import _frame_auc, _masked_focal_bce, frame_labels

LOGGER = logging.getLogger(__name__)

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class ClipVideoSample:
    """One video's CLIP-trainable material, all arrays aligned on the frame axis."""

    video_key: str
    subject: str
    frames: List[int]
    frame_paths: List[str]
    descriptions: List[str]
    labels: np.ndarray            # (T,)
    ignore: np.ndarray            # (T,) True where the loss is masked
    head_block: np.ndarray        # (T, 7) explicit head-motion input
    analytic: np.ndarray          # (T, K) analytic activations (distill target)

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())

    @property
    def head_speed(self) -> np.ndarray:
        return self.head_block[:, -1]


def build_clip_dataset(
    videos: Sequence[Any],
    config: Optional[MEWMConfig] = None,
    stride: int = 1,
    max_frames: int = 0,
    require_events: bool = True,
) -> List[ClipVideoSample]:
    """Stage I per video, plus descriptions, labels and the head-motion block.

    ``videos`` must already be the fold's training pool (same contract as
    ``build_frame_dataset``); the trainer records the subjects it actually saw.
    """
    from ..pipeline import run_representation

    config = config or load_config()
    samples: List[ClipVideoSample] = []

    for video in videos:
        if require_events and not video.micro_events():
            continue
        try:
            representation = run_representation(
                video, config, stride=stride, max_frames=max_frames)
        except Exception as exc:  # noqa: BLE001 - one broken video must not stop a fold
            LOGGER.warning("clip dataset: skipping %s: %s", video.video_id, exc)
            continue
        if representation.slot_activations is None or len(representation) < 32:
            LOGGER.warning("clip dataset: %s too short, skipping", video.video_id)
            continue

        frames, paths, descriptions, keep = [], [], [], []
        for i, t in enumerate(representation.frames):
            frame_path = video.paths.frame(t)
            if not frame_path.is_file():
                continue
            state = representation.stream.get(t)
            if state is None:
                continue
            frames.append(t)
            paths.append(str(frame_path))
            descriptions.append(frame_motion_description(state.measurements))
            keep.append(i)

        if len(frames) < 32:
            LOGGER.warning("clip dataset: %s has %d usable RGB frames, skipping",
                           video.video_id, len(frames))
            continue

        keep_idx = np.asarray(keep, dtype=np.int64)
        labels, ignore = frame_labels(video, frames)
        if require_events and labels.sum() <= 0:
            LOGGER.warning("clip dataset: %s has no positive frames in range",
                           video.video_id)
            continue

        head_block = head_feature_block(
            representation.head_motion, len(representation.frames))[keep_idx]
        analytic = np.asarray(
            representation.slot_activations, dtype=np.float32)[keep_idx]

        samples.append(ClipVideoSample(
            video_key=video.video_key, subject=str(video.subject),
            frames=frames, frame_paths=paths, descriptions=descriptions,
            labels=labels, ignore=ignore, head_block=head_block, analytic=analytic))
        LOGGER.info("clip dataset: %s -> %d frames, %d positive",
                    video.video_id, len(frames), int(labels.sum()))
    return samples


# ---------------------------------------------------------------------------
# Losses beyond the shared focal BCE
# ---------------------------------------------------------------------------


def soft_iou_loss(probs: "torch.Tensor", labels: "torch.Tensor",
                  mask: "torch.Tensor") -> "torch.Tensor":
    """``1 − soft-IoU`` between the per-frame probabilities and the GT interval mask.

    方案 §3 的连续监督: the discrete IoU of eq. (2) with the indicator replaced by the
    sigmoid probability, so the interval boundaries receive a *dense* gradient rather
    than only the per-frame BCE one. Ignored (macro) frames contribute to neither side.
    """
    p = probs * mask
    y = labels * mask
    intersection = (p * y).sum()
    union = (p + y - p * y).sum().clamp_min(1e-6)
    return 1.0 - intersection / union


def head_motion_triplet(
    u: "torch.Tensor",
    labels: "torch.Tensor",
    negatives: "torch.Tensor",
    margin: float,
) -> Optional["torch.Tensor"]:
    """方案 §2.2 分量二 -- the micro-expression vs head-motion discriminative triplet.
    """
    pos_idx = torch.nonzero(labels > 0.5, as_tuple=False).flatten()
    neg_idx = torch.nonzero(negatives > 0.5, as_tuple=False).flatten()
    if pos_idx.numel() < 2 or neg_idx.numel() == 0:
        return None
    import torch.nn.functional as F
    anchor = F.normalize(u[pos_idx].mean(dim=0, keepdim=True), dim=-1)
    positive = F.normalize(u[pos_idx], dim=-1)
    negative = F.normalize(u[neg_idx], dim=-1)
    sim_pos = (anchor @ positive.t()).mean()
    sim_neg = (anchor @ negative.t()).max()
    return torch.clamp(margin + sim_neg - sim_pos, min=0.0)


def negative_mask(sample: ClipVideoSample, percentile: float) -> np.ndarray:
    """Head-motion negative frames: fast head, outside every annotated event."""
    speed = sample.head_speed
    if speed.size == 0:
        return np.zeros(0, dtype=np.float32)
    threshold = np.percentile(speed, percentile)
    negatives = (speed >= threshold) & (sample.labels <= 0.5) & (~sample.ignore)
    return negatives.astype(np.float32)


# ---------------------------------------------------------------------------
# Windows (event-biased, CLIP-affordable)
# ---------------------------------------------------------------------------


def _event_windows(sample: ClipVideoSample, window: int, rng,
                   n_uniform: int) -> List[Tuple[int, int]]:
    n = len(sample.labels)
    if n <= window:
        return [(0, n)]
    spans: List[Tuple[int, int]] = []
    positive = np.flatnonzero(sample.labels > 0)
    if positive.size:
        breaks = np.flatnonzero(np.diff(positive) > 1)
        for group in np.split(positive, breaks + 1):
            centre = int(group.mean())
            start = int(np.clip(centre - window // 2, 0, n - window))
            spans.append((start, start + window))
    for start in rng.integers(0, n - window, size=max(0, n_uniform)):
        spans.append((int(start), int(start) + window))
    return spans


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass
class CLIPLocaliserCheckpoint:
    """Trainable-only weights plus fold provenance (same guarantees as the MLP one)."""

    trainable_state: Dict[str, Any]
    clip_config: Dict[str, Any]
    n_slots: int = K_SLOTS
    train_subjects: List[str] = field(default_factory=list)
    val_subjects: List[str] = field(default_factory=list)
    fold_name: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def assert_excludes(self, subjects: Sequence[str]) -> None:
        seen = set(self.train_subjects) | set(self.val_subjects)
        leaked = sorted(seen & {str(s) for s in subjects})
        if leaked:
            raise RuntimeError(
                f"CLIP localiser checkpoint for fold {self.fold_name!r} was fitted on "
                f"subject(s) {leaked} that it is now being asked to predict on")

    def save(self, path: Path | str) -> Path:
        if not _TORCH:
            raise ImportError("saving a CLIP localiser checkpoint needs PyTorch")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trainable_state": self.trainable_state,
            "clip_config": self.clip_config,
            "n_slots": self.n_slots,
            "train_subjects": self.train_subjects,
            "val_subjects": self.val_subjects,
            "fold_name": self.fold_name,
            "metrics": self.metrics,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CLIPLocaliserCheckpoint":
        if not _TORCH:
            raise ImportError("loading a CLIP localiser checkpoint needs PyTorch")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            trainable_state=blob["trainable_state"],
            clip_config=dict(blob.get("clip_config", {})),
            n_slots=int(blob.get("n_slots", K_SLOTS)),
            train_subjects=list(blob.get("train_subjects", [])),
            val_subjects=list(blob.get("val_subjects", [])),
            fold_name=blob.get("fold_name", ""),
            metrics=blob.get("metrics", {}))


def checkpoint_path(dataset: str, fold_name: str,
                    config: Optional[ClipConfig] = None) -> Path:
    config = config or load_config().clip
    root = Path(config.checkpoint_root)
    if not root.is_absolute():
        root = PACKAGE_ROOT / root
    return root / dataset / f"fold_{fold_name}" / "clip_localiser.pt"


def find_checkpoint(dataset: str, subject: str,
                    config: Optional[ClipConfig] = None) -> Optional[Path]:
    """The fold checkpoint whose held-out subject is ``subject``, if trained."""
    path = checkpoint_path(dataset, str(subject), config)
    return path if path.is_file() else None


def train_state_path(dataset: str, fold_name: str,
                     config: Optional[ClipConfig] = None) -> Path:
    """Epoch-level resume checkpoint for a fold that is still training.

    Lives next to the fold's final ``clip_localiser.pt`` so a kill mid-fold loses at
    most one epoch of that fold instead of the whole fold (断点续训, user directive).
    """
    return checkpoint_path(dataset, fold_name, config).parent / "train_state.pt"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _split_subjects(samples: Sequence[ClipVideoSample],
                    config: ClipConfig) -> Tuple[List[str], List[str]]:
    """Subject-disjoint train/val split (same rule as localiser_supervised)."""
    rng = np.random.default_rng(config.seed)
    with_pos = sorted({s.subject for s in samples if s.n_positive > 0})
    without = sorted({s.subject for s in samples} - set(with_pos))
    if len(with_pos) <= 1:
        return sorted({s.subject for s in samples}), []
    n_val = max(config.min_val_subjects,
                int(round(len(with_pos) * config.val_subject_fraction)))
    n_val = min(n_val, len(with_pos) - 1)
    order = rng.permutation(len(with_pos))
    val = sorted(with_pos[i] for i in order[:n_val])
    train = sorted(set(with_pos) - set(val)) + without
    return sorted(train), val


class _WindowEncoder:
    """Runs one window through the towers, chunked so 8 GB survives it."""

    def __init__(self, model: CLIPSpotterModel, config: ClipConfig, device: str):
        self.model = model
        self.config = config
        self.device = device

    def _autocast(self):
        import contextlib
        if self.config.autocast_bf16 and self.device.startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def encode(self, sample: ClipVideoSample, a: int, b: int,
               grad: bool = True) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Frames ``[a, b)`` -> (v, m) each (T, d). Chunked; graphs kept when ``grad``."""
        import contextlib
        chunks_v, chunks_m = [], []
        step = max(1, self.config.batch_frames)
        no_grad = contextlib.nullcontext() if grad else torch.no_grad()
        with no_grad, self._autocast():
            for start in range(a, b, step):
                stop = min(b, start + step)
                pixels = self.model.towers.preprocess_images(
                    sample.frame_paths[start:stop]).to(self.device)
                chunks_v.append(self.model.towers.encode_images(pixels))
                tokens = self.model.towers.tokenize(
                    sample.descriptions[start:stop])
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                chunks_m.append(self.model.towers.encode_texts(
                    tokens["input_ids"], tokens["attention_mask"]))
        return (torch.cat(chunks_v, dim=0).float(),
                torch.cat(chunks_m, dim=0).float())


def _save_train_state(
    path: Path, *, epoch: int, model: CLIPSpotterModel,
    optimiser: "torch.optim.Optimizer", best_auc: float,
    best_state: Optional[Dict[str, "torch.Tensor"]], best_epoch: int, stale: int,
    history: List[Dict[str, float]], rng: np.random.Generator,
) -> None:
    """Write the epoch-granularity resume blob (model/optimiser/RNG/best-so-far).

    Written after every epoch, atomically (tmp file + rename) so a kill mid-write never
    leaves a corrupt resume file -- the whole point is surviving an interruption.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model_state": model.trainable_state_dict(),
        "optimiser_state": optimiser.state_dict(),
        "best_auc": best_auc,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "stale": stale,
        "history": history,
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": cuda_state,
    }, tmp)
    tmp.replace(path)


def train_clip_localiser(
    samples: Sequence[ClipVideoSample],
    config: Optional[ClipConfig] = None,
    fold_name: str = "",
    state_path: Optional[Path] = None,
) -> CLIPLocaliserCheckpoint:
    """Fit the dual tower + heads on a fold's pool; select on validation frame AUC.

    ``state_path``, if given, is an epoch-level resume checkpoint: written after every
    epoch and read back at the top of this call if it already exists, so an interrupted
    fold picks up at ``last_epoch + 1`` instead of restarting (断点续训, user directive).
    It is deleted once the fold finishes normally.
    """
    if not _TORCH:
        raise ImportError("train_clip_localiser needs PyTorch installed.")
    if not samples:
        raise ValueError("train_clip_localiser got no samples")

    config = config or load_config().clip
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = config.device if (config.device == "cpu"
                               or torch.cuda.is_available()) else "cpu"

    model = CLIPSpotterModel(config, n_slots=K_SLOTS).to(device)
    counts = model.towers.unfrozen_layer_counts()
    LOGGER.info("clip localiser fold %s: fine-tuning %d vision / %d text layer(s)",
                fold_name, counts["vision"], counts["text"])
    encoder = _WindowEncoder(model, config, device)

    train_subjects, val_subjects = _split_subjects(samples, config)
    train = [s for s in samples if s.subject in set(train_subjects)]
    val = [s for s in samples if s.subject in set(val_subjects)]
    LOGGER.info("clip localiser fold %s: %d train / %d val video(s); val subjects %s",
                fold_name, len(train), len(val), val_subjects)

    n_pos = sum(int(s.labels.sum()) for s in train)
    n_neg = sum(int((~s.ignore).sum() - s.labels.sum()) for s in train)
    pos_weight = min(config.pos_weight_cap, max(1.0, n_neg / max(1, n_pos)))

    tower_params = [p for p in model.towers.parameters() if p.requires_grad]
    head_params = (list(model.fusion.parameters())
                   + list(model.transition.parameters())
                   + list(model.stem.parameters())
                   + list(model.blocks.parameters())
                   + list(model.head.parameters()))
    optimiser = torch.optim.AdamW([
        {"params": tower_params, "lr": config.learning_rate_towers},
        {"params": head_params, "lr": config.learning_rate_head},
    ], weight_decay=config.weight_decay)

    negatives_by_video = {s.video_key: negative_mask(s, config.head_speed_percentile)
                          for s in samples}

    def _evaluate(pool: Sequence[ClipVideoSample]) -> float:
        model.eval()
        aucs = []
        for sample in pool:
            scores = _score_sample(model, encoder, sample, device)
            auc = _frame_auc(scores, sample.labels, ~sample.ignore)
            if np.isfinite(auc):
                aucs.append(auc)
        return float(np.mean(aucs)) if aucs else float("nan")

    best_auc, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: List[Dict[str, float]] = []
    start_epoch = 0

    if state_path is not None and Path(state_path).is_file():
        LOGGER.info("clip localiser fold %s: resume file found at %s, loading",
                    fold_name, state_path)
        # map_location="cpu": the RNG-state tensors *must* stay CPU ByteTensors for
        # torch.set_rng_state / cuda.set_rng_state_all -- load_state_dict below casts
        # the model/optimiser tensors onto `device` on its own.
        blob = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(blob["model_state"])
        optimiser.load_state_dict(blob["optimiser_state"])
        start_epoch = int(blob["epoch"]) + 1
        best_auc = blob["best_auc"]
        best_state = blob["best_state"]
        best_epoch = blob["best_epoch"]
        stale = blob["stale"]
        history = blob["history"]
        rng.bit_generator.state = blob["rng_state"]
        torch.set_rng_state(blob["torch_rng_state"])
        if device.startswith("cuda") and blob.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(blob["cuda_rng_state"])
        LOGGER.info("clip localiser fold %s: resuming at epoch %d (best epoch %d, "
                    "auc %.4f, stale %d)", fold_name, start_epoch, best_epoch,
                    best_auc, stale)

    for epoch in range(start_epoch, config.epochs):
        model.train()
        spans: List[Tuple[ClipVideoSample, int, int]] = []
        for sample in train:
            window = min(config.window, len(sample.labels))
            for a, b in _event_windows(sample, window, rng,
                                       config.uniform_windows_per_video):
                spans.append((sample, a, b))
        rng.shuffle(spans)

        totals: Dict[str, float] = {}
        n_steps = 0
        for sample, a, b in spans:
            v, m = encoder.encode(sample, a, b, grad=True)
            u = model.fuse(v, m)

            labels = torch.from_numpy(sample.labels[a:b]).float().to(device)
            mask = torch.from_numpy(
                (~sample.ignore[a:b]).astype(np.float32)).to(device)
            head = torch.from_numpy(sample.head_block[a:b]).float().to(device)
            analytic = torch.from_numpy(sample.analytic[a:b]).float().to(device)
            negatives = torch.from_numpy(
                negatives_by_video[sample.video_key][a:b]).float().to(device)

            logits = model.localise(u[None], head[None])[0]
            probs = torch.sigmoid(logits)

            loss_align = info_nce(v, m, config.temperature)
            loss_loc = _masked_focal_bce(logits, labels, mask, pos_weight,
                                         config.focal_gamma)
            loss_distill = torch.nn.functional.mse_loss(model.transition(u), analytic)
            loss = (config.lambda_align * loss_align
                    + config.lambda_loc * loss_loc
                    + config.lambda_distill * loss_distill)
            parts = {"align": float(loss_align.detach()),
                     "loc": float(loss_loc.detach()),
                     "distill": float(loss_distill.detach())}

            if labels.sum() > 0:
                loss_prop = soft_iou_loss(probs, labels, mask)
                loss = loss + config.lambda_prop * loss_prop
                parts["prop"] = float(loss_prop.detach())
            loss_cont = head_motion_triplet(u, labels, negatives,
                                            config.contrastive_margin)
            if loss_cont is not None:
                loss = loss + config.lambda_cont * loss_cont
                parts["cont"] = float(loss_cont.detach())

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config.grad_clip)
            optimiser.step()

            parts["total"] = float(loss.detach())
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            n_steps += 1

        means = {k: round(v / max(1, n_steps), 5) for k, v in totals.items()}
        val_auc = _evaluate(val or train)
        history.append({"epoch": epoch, **means, "val_auc": val_auc})
        LOGGER.info("clip localiser fold %s epoch %d: %s val_auc %.4f",
                    fold_name, epoch, means, val_auc)

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = model.trainable_state_dict()
        else:
            stale += 1

        if state_path is not None:
            _save_train_state(
                state_path, epoch=epoch, model=model, optimiser=optimiser,
                best_auc=best_auc, best_state=best_state, best_epoch=best_epoch,
                stale=stale, history=history, rng=rng)

        if stale >= config.patience:
            LOGGER.info("clip localiser fold %s: early stop at epoch %d "
                        "(best %d, auc %.4f)", fold_name, epoch, best_epoch, best_auc)
            break

    if best_state is None:
        best_state = model.trainable_state_dict()

    if state_path is not None:
        Path(state_path).unlink(missing_ok=True)

    return CLIPLocaliserCheckpoint(
        trainable_state=best_state,
        clip_config={k: getattr(config, k) for k in vars(config)},
        n_slots=K_SLOTS,
        train_subjects=sorted({s.subject for s in train}),
        val_subjects=sorted({s.subject for s in val}),
        fold_name=fold_name,
        metrics={"best_val_auc": best_auc, "best_epoch": best_epoch,
                 "pos_weight": pos_weight, "n_positive_frames": n_pos,
                 "unfrozen_layers": counts, "history": history})


def _score_sample(model: CLIPSpotterModel, encoder: _WindowEncoder,
                  sample: ClipVideoSample, device: str) -> np.ndarray:
    """Full-video forward (no grad) -> raw per-frame detection scores."""
    with torch.no_grad():
        v, m = encoder.encode(sample, 0, len(sample.frames), grad=False)
        u = model.fuse(v, m)
        head = torch.from_numpy(sample.head_block).float().to(device)
        logits = model.localise(u[None], head[None])[0]
    return logits.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class TrainedCLIPSpotter:
    """Loaded fold checkpoint: representation -> detection curve + (T, K) activations.

    The curve is a raw score, exactly like ``TrainedLocaliser.curve``: the Spotter
    applies its own robust normalisation and hysteresis downstream (方案 §2.3).
    """

    def __init__(self, checkpoint: CLIPLocaliserCheckpoint,
                 config: Optional[ClipConfig] = None, device: str = "cuda"):
        if not _TORCH:
            raise ImportError("TrainedCLIPSpotter needs PyTorch installed.")
        self.checkpoint = checkpoint
        base = config or load_config().clip
        # Architecture fields come from the checkpoint; paths stay overridable.
        merged = ClipConfig(**{**vars(base), **{
            k: v for k, v in checkpoint.clip_config.items()
            if k in {"vision_unfreeze_layers", "text_unfreeze_layers",
                     "head_channels", "head_dropout", "temperature"}}})
        merged.weights_path = base.weights_path
        self.config = merged
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.model = CLIPSpotterModel(merged, n_slots=checkpoint.n_slots)
        self.model.load_trainable_state_dict(checkpoint.trainable_state)
        self.model.to(self.device).eval()
        self._encoder = _WindowEncoder(self.model, merged, self.device)

    @classmethod
    def from_path(cls, path: Path | str, config: Optional[ClipConfig] = None,
                  device: str = "cuda") -> "TrainedCLIPSpotter":
        return cls(CLIPLocaliserCheckpoint.load(path), config=config, device=device)

    def _sample_for(self, video: Any, representation: Any) -> ClipVideoSample:
        frames, paths, descriptions, keep = [], [], [], []
        for i, t in enumerate(representation.frames):
            frame_path = video.paths.frame(t)
            state = representation.stream.get(t)
            if state is None or not frame_path.is_file():
                continue
            frames.append(t)
            paths.append(str(frame_path))
            descriptions.append(frame_motion_description(state.measurements))
            keep.append(i)
        keep_idx = np.asarray(keep, dtype=np.int64)
        n = len(representation.frames)
        head_block = head_feature_block(representation.head_motion, n)[keep_idx]
        analytic = (np.asarray(representation.slot_activations,
                               dtype=np.float32)[keep_idx]
                    if representation.slot_activations is not None
                    else np.zeros((len(frames), self.checkpoint.n_slots),
                                  dtype=np.float32))
        return ClipVideoSample(
            video_key=video.video_key, subject=str(video.subject),
            frames=frames, frame_paths=paths, descriptions=descriptions,
            labels=np.zeros(len(frames), dtype=np.float32),
            ignore=np.zeros(len(frames), dtype=bool),
            head_block=head_block, analytic=analytic)

    def infer(self, video: Any, representation: Any) -> Dict[str, np.ndarray]:
        """Curve on the representation's frame grid + transition-head activations.

        Frames without an RGB file keep the analytic activation row and get a curve
        value of the finite minimum, so the output grids stay aligned with
        ``representation.frames``.
        """
        self.checkpoint.assert_excludes([str(video.subject)])
        sample = self._sample_for(video, representation)
        with torch.no_grad():
            v, m = self._encoder.encode(sample, 0, len(sample.frames), grad=False)
            u = self.model.fuse(v, m)
            head = torch.from_numpy(sample.head_block).float().to(self.device)
            logits = self.model.localise(u[None], head[None])[0]
            activations = self.model.transition(u)
        scores = logits.detach().cpu().numpy().astype(np.float64)
        acts = activations.detach().cpu().numpy().astype(np.float64)

        index = {t: i for i, t in enumerate(sample.frames)}
        n = len(representation.frames)
        curve = np.full(n, float(scores.min()) if scores.size else 0.0)
        full_acts = (np.asarray(representation.slot_activations, dtype=np.float64)
                     if representation.slot_activations is not None
                     else np.zeros((n, self.checkpoint.n_slots)))
        full_acts = full_acts.copy()
        for i, t in enumerate(representation.frames):
            j = index.get(t)
            if j is not None:
                curve[i] = scores[j]
                full_acts[i] = acts[j]
        return {"curve": curve, "activations": full_acts}


def evaluate_clip_localiser(
    checkpoint: CLIPLocaliserCheckpoint,
    samples: Sequence[ClipVideoSample],
    config: Optional[ClipConfig] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Frame-level AUC on held-out samples, per video and pooled."""
    spotter = TrainedCLIPSpotter(checkpoint, config=config, device=device)
    spotter.checkpoint.assert_excludes([s.subject for s in samples])
    per_video, aucs = [], []
    for sample in samples:
        scores = _score_sample(spotter.model, spotter._encoder, sample, spotter.device)
        auc = _frame_auc(scores, sample.labels, ~sample.ignore)
        per_video.append({"video": sample.video_key, "subject": sample.subject,
                          "auc": auc, "n_positive": sample.n_positive})
        if np.isfinite(auc):
            aucs.append(auc)
    return {
        "pooled_mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_videos": len(per_video),
        "n_above_0.5": int(sum(1 for a in aucs if a > 0.5)),
        "n_above_0.7": int(sum(1 for a in aucs if a > 0.7)),
        "per_video": per_video,
    }


def write_fold_report(checkpoint: CLIPLocaliserCheckpoint, report: Dict[str, Any],
                      dataset: str, fold_name: str,
                      config: Optional[ClipConfig] = None) -> Path:
    target = checkpoint_path(dataset, fold_name, config)
    checkpoint.save(target)
    report_path = target.parent / "test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    return target


__all__ = [
    "ClipVideoSample", "build_clip_dataset", "soft_iou_loss", "head_motion_triplet",
    "negative_mask", "CLIPLocaliserCheckpoint", "checkpoint_path", "find_checkpoint",
    "train_state_path", "train_clip_localiser", "TrainedCLIPSpotter",
    "evaluate_clip_localiser", "write_fold_report",
]
