"""Supervised localiser fine-tuning using manually annotated ME intervals."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, load_config

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    torch = None
    nn = None
    _TORCH = False


@dataclass
class LocaliserTrainConfig:

    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16)
    channels: int = 64
    kernel_size: int = 3
    dropout: float = 0.1

    epochs: int = 300
    batch_windows: int = 8
    window: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 12

    pos_weight_cap: float = 20.0
    focal_gamma: float = 1.0

    val_subject_fraction: float = 0.25
    min_val_subjects: int = 1

    device: str = "cuda"
    seed: int = 20260828


def _robust_z(x: np.ndarray, axis: int = 0) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (x - med) / scale


def frame_features(
    representation: Any,
    slot_error: Optional[np.ndarray] = None,
) -> np.ndarray:
    activations = np.asarray(representation.slot_activations, dtype=np.float64)
    if activations.ndim != 2 or activations.shape[0] < 3:
        raise ValueError("frame_features needs (T, K) slot activations with T >= 3")
    n_frames, n_slots = activations.shape

    a = _robust_z(activations)
    velocity = np.diff(a, axis=0, prepend=a[:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1])

    blocks: List[np.ndarray] = [a, velocity, acceleration]

    if slot_error is not None:
        err = np.asarray(slot_error, dtype=np.float64)
        if err.shape == activations.shape:
            blocks.append(_robust_z(err))
            blocks.append(_robust_z(err.sum(axis=1, keepdims=True)))

    head = getattr(representation, "head_motion", None)
    if head is not None:
        head = np.asarray(head, dtype=np.float64)
        if head.ndim == 2 and head.shape[0] == n_frames:
            head_z = _robust_z(head)
            head_v = np.diff(head_z, axis=0, prepend=head_z[:1])
            speed = np.linalg.norm(head_v, axis=1, keepdims=True)
            blocks.extend([head_z, head_v, _robust_z(speed)])

    gate = getattr(representation, "coherence_gate", None)
    if gate is not None:
        gate = np.asarray(gate, dtype=np.float64)
        if gate.ndim == 2 and gate.shape[0] == n_frames:
            blocks.append(gate)

    features = np.concatenate(blocks, axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def frame_labels(
    video: Any,
    frames: Sequence[int],
    micro_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = list(frames)
    index = {f: i for i, f in enumerate(frames)}
    labels = np.zeros(len(frames), dtype=np.float32)
    ignore = np.zeros(len(frames), dtype=bool)

    for onset, offset in video.ground_truth_intervals(micro_only=True):
        for t in range(int(onset), int(offset) + 1):
            i = index.get(t)
            if i is not None:
                labels[i] = 1.0

    return labels, ignore


@dataclass
class VideoSample:

    video_key: str
    subject: str
    features: np.ndarray
    labels: np.ndarray
    ignore: np.ndarray
    frames: List[int] = field(default_factory=list)

    @property
    def n_positive(self) -> int:
        return int(self.labels.sum())


def build_frame_dataset(
    videos: Sequence[Any],
    config: Optional[MEWMConfig] = None,
    max_frames: int = 0,
    stride: int = 1,
    require_events: bool = True,
) -> List[VideoSample]:
    from ..pipeline import run_representation
    from ..engines.m1_dynamics import AnalyticDynamics

    config = config or load_config()
    dynamics = AnalyticDynamics(config.dynamics)
    samples: List[VideoSample] = []

    for video in videos:
        if require_events and not video.micro_events():
            continue
        try:
            representation = run_representation(
                video, config, stride=stride, max_frames=max_frames)
        except Exception as exc:
            LOGGER.warning("localiser dataset: skipping %s: %s", video.video_id, exc)
            continue

        activations = representation.slot_activations
        if activations is None or activations.shape[0] < 32:
            LOGGER.warning("localiser dataset: %s too short, skipping", video.video_id)
            continue

        n, k = activations.shape
        slot_error = np.zeros((n, k), dtype=np.float64)
        velocity = np.zeros(k)
        for t in range(1, n):
            predicted, _var = dynamics.step(activations[t - 1], momentum=velocity)
            slot_error[t] = np.abs(activations[t] - predicted)
            velocity = 0.6 * velocity + 0.4 * (activations[t] - activations[t - 1])

        features = frame_features(representation, slot_error=slot_error)
        labels, ignore = frame_labels(video, representation.frames)
        if require_events and labels.sum() <= 0:
            LOGGER.warning("localiser dataset: %s has no positive frames in range",
                           video.video_id)
            continue

        samples.append(VideoSample(
            video_key=video.video_key, subject=str(video.subject),
            features=features, labels=labels, ignore=ignore,
            frames=list(representation.frames)))
        LOGGER.info("localiser dataset: %s -> %d frames, %d positive",
                    video.video_id, len(labels), int(labels.sum()))

    return samples


if _TORCH:

    class _ResidualBlock(nn.Module):

        def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
            super().__init__()
            padding = dilation * (kernel - 1) // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel,
                                   padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel,
                                   padding=padding, dilation=dilation)
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.drop(F.gelu(self.norm1(self.conv1(x))))
            h = self.drop(F.gelu(self.norm2(self.conv2(h))))
            return x + h

    class SupervisedLocaliser(nn.Module):

        def __init__(self, n_features: int, config: Optional[LocaliserTrainConfig] = None):
            super().__init__()
            config = config or LocaliserTrainConfig()
            self.n_features = int(n_features)
            self.config = config
            c = config.channels
            self.stem = nn.Conv1d(self.n_features, c, 1)
            self.blocks = nn.ModuleList([
                _ResidualBlock(c, config.kernel_size, d, config.dropout)
                for d in config.dilations
            ])
            self.head = nn.Conv1d(c, 1, 1)

        @property
        def receptive_field(self) -> int:
            return 1 + 2 * (self.config.kernel_size - 1) * sum(self.config.dilations)

        def forward(self, x):
            h = x.transpose(1, 2)
            h = self.stem(h)
            for block in self.blocks:
                h = block(h)
            return self.head(h).squeeze(1)

else:

    class SupervisedLocaliser:
        def __init__(self, *a, **kw):
            raise ImportError("SupervisedLocaliser needs PyTorch installed.")


@dataclass
class LocaliserCheckpoint:

    state_dict: Dict[str, Any]
    n_features: int
    train_config: LocaliserTrainConfig
    train_subjects: List[str] = field(default_factory=list)
    val_subjects: List[str] = field(default_factory=list)
    fold_name: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def assert_excludes(self, subjects: Sequence[str]) -> None:
        seen = set(self.train_subjects) | set(self.val_subjects)
        leaked = sorted(seen & {str(s) for s in subjects})
        if leaked:
            raise RuntimeError(
                f"localiser checkpoint for fold {self.fold_name!r} was fitted on "
                f"subject(s) {leaked} that it is now being asked to predict on")

    def save(self, path: Path | str) -> Path:
        if not _TORCH:
            raise ImportError("saving a localiser checkpoint needs PyTorch")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict,
            "n_features": self.n_features,
            "train_config": self.train_config.__dict__,
            "train_subjects": self.train_subjects,
            "val_subjects": self.val_subjects,
            "fold_name": self.fold_name,
            "metrics": self.metrics,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "LocaliserCheckpoint":
        if not _TORCH:
            raise ImportError("loading a localiser checkpoint needs PyTorch")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = LocaliserTrainConfig(**blob.get("train_config", {}))
        return cls(
            state_dict=blob["state_dict"], n_features=int(blob["n_features"]),
            train_config=cfg, train_subjects=list(blob.get("train_subjects", [])),
            val_subjects=list(blob.get("val_subjects", [])),
            fold_name=blob.get("fold_name", ""), metrics=blob.get("metrics", {}))


def _split_subjects(
    samples: Sequence[VideoSample], config: LocaliserTrainConfig,
) -> Tuple[List[str], List[str]]:
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


def _windows(sample: VideoSample, window: int, rng) -> List[Tuple[int, int]]:
    n = len(sample.labels)
    if n <= window:
        return [(0, n)]
    spans: List[Tuple[int, int]] = []

    positive = np.flatnonzero(sample.labels > 0)
    if positive.size:
        breaks = np.flatnonzero(np.diff(positive) > 1)
        groups = np.split(positive, breaks + 1)
        for group in groups:
            centre = int(group.mean())
            start = int(np.clip(centre - window // 2, 0, n - window))
            spans.append((start, start + window))

    n_uniform = max(1, n // window)
    for start in rng.integers(0, n - window, size=n_uniform):
        spans.append((int(start), int(start) + window))
    return spans


def _batch(
    samples: Sequence[VideoSample], spans: Sequence[Tuple[int, VideoSample, int, int]],
):
    x = np.stack([s.features[a:b] for _, s, a, b in spans])
    y = np.stack([s.labels[a:b] for _, s, a, b in spans])
    m = np.stack([~s.ignore[a:b] for _, s, a, b in spans])
    return (torch.from_numpy(x).float(), torch.from_numpy(y).float(),
            torch.from_numpy(m.astype(np.float32)))


def _masked_focal_bce(logits, targets, mask, pos_weight: float, gamma: float):
    weight = torch.where(targets > 0.5,
                         torch.full_like(targets, pos_weight),
                         torch.ones_like(targets))
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none")
    if gamma > 0:
        prob = torch.sigmoid(logits)
        p_t = torch.where(targets > 0.5, prob, 1.0 - prob)
        bce = bce * (1.0 - p_t).clamp_min(1e-6).pow(gamma)
    bce = bce * weight * mask
    denom = (weight * mask).sum().clamp_min(1.0)
    return bce.sum() / denom


def _frame_auc(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    keep = mask.astype(bool)
    pos = scores[keep & (labels > 0.5)]
    neg = scores[keep & (labels <= 0.5)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    values = np.concatenate([pos, neg])[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[j + 1] == values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    rank_sum = ranks[:pos.size].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def train_localiser(
    samples: Sequence[VideoSample],
    config: Optional[LocaliserTrainConfig] = None,
    fold_name: str = "",
) -> LocaliserCheckpoint:
    if not _TORCH:
        raise ImportError("train_localiser needs PyTorch installed.")
    if not samples:
        raise ValueError("train_localiser got no samples")

    config = config or LocaliserTrainConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    train_subjects, val_subjects = _split_subjects(samples, config)
    train = [s for s in samples if s.subject in set(train_subjects)]
    val = [s for s in samples if s.subject in set(val_subjects)]
    LOGGER.info("localiser fold %s: %d train video(s) / %d val video(s); "
                "val subjects %s", fold_name, len(train), len(val), val_subjects)

    n_features = int(samples[0].features.shape[1])
    device = config.device if (config.device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = SupervisedLocaliser(n_features, config).to(device)

    n_pos = sum(int(s.labels.sum()) for s in train)
    n_neg = sum(int((~s.ignore).sum() - s.labels.sum()) for s in train)
    pos_weight = min(config.pos_weight_cap, max(1.0, n_neg / max(1, n_pos)))
    LOGGER.info("localiser fold %s: %d positive / %d negative frames, pos_weight %.2f",
                fold_name, n_pos, n_neg, pos_weight)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_auc, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history: List[Dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        spans: List[Tuple[int, VideoSample, int, int]] = []
        for i, sample in enumerate(train):
            for a, b in _windows(sample, min(config.window, len(sample.labels)), rng):
                spans.append((i, sample, a, b))
        rng.shuffle(spans)

        target_len = min(config.window, min(len(s.labels) for s in train))
        spans = [(i, s, a, min(a + target_len, len(s.labels))) for i, s, a, b in spans]
        spans = [sp for sp in spans if sp[3] - sp[2] == target_len]

        total, n_batches = 0.0, 0
        for start in range(0, len(spans), config.batch_windows):
            chunk = spans[start:start + config.batch_windows]
            if not chunk:
                continue
            x, y, m = _batch(train, chunk)
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            loss = _masked_focal_bce(logits, y, m, pos_weight, config.focal_gamma)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            total += float(loss.detach())
            n_batches += 1

        train_loss = total / max(1, n_batches)
        evaluation = val or train
        aucs = []
        model.eval()
        with torch.no_grad():
            for sample in evaluation:
                x = torch.from_numpy(sample.features[None]).float().to(device)
                scores = model(x)[0].cpu().numpy()
                auc = _frame_auc(scores, sample.labels, ~sample.ignore)
                if np.isfinite(auc):
                    aucs.append(auc)
        val_auc = float(np.mean(aucs)) if aucs else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
        LOGGER.info("localiser fold %s epoch %d: loss %.4f val_auc %.4f",
                    fold_name, epoch, train_loss, val_auc)

        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return LocaliserCheckpoint(
        state_dict=best_state, n_features=n_features, train_config=config,
        train_subjects=sorted({s.subject for s in train}),
        val_subjects=sorted({s.subject for s in val}),
        fold_name=fold_name,
        metrics={"best_val_auc": best_auc, "best_epoch": best_epoch,
                 "pos_weight": pos_weight, "n_positive_frames": n_pos,
                 "history": history})


class TrainedLocaliser:

    def __init__(self, checkpoint: LocaliserCheckpoint, device: str = "cuda"):
        if not _TORCH:
            raise ImportError("TrainedLocaliser needs PyTorch installed.")
        self.checkpoint = checkpoint
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.model = SupervisedLocaliser(checkpoint.n_features, checkpoint.train_config)
        self.model.load_state_dict(checkpoint.state_dict)
        self.model.to(self.device).eval()

    @classmethod
    def from_path(cls, path: Path | str, device: str = "cuda") -> "TrainedLocaliser":
        return cls(LocaliserCheckpoint.load(path), device=device)

    def curve(self, representation: Any, slot_error: Optional[np.ndarray] = None) -> np.ndarray:
        features = frame_features(representation, slot_error=slot_error)
        if features.shape[1] != self.checkpoint.n_features:
            raise ValueError(
                f"feature width {features.shape[1]} does not match the checkpoint's "
                f"{self.checkpoint.n_features}; the representation config changed since "
                f"fold {self.checkpoint.fold_name!r} was trained")
        with torch.no_grad():
            x = torch.from_numpy(features[None]).float().to(self.device)
            return self.model(x)[0].cpu().numpy().astype(np.float64)


def evaluate_localiser(
    checkpoint: LocaliserCheckpoint,
    samples: Sequence[VideoSample],
    device: str = "cuda",
) -> Dict[str, Any]:
    localiser = TrainedLocaliser(checkpoint, device=device)
    localiser.checkpoint.assert_excludes([s.subject for s in samples])

    per_video, aucs = [], []
    for sample in samples:
        with torch.no_grad():
            x = torch.from_numpy(sample.features[None]).float().to(localiser.device)
            scores = localiser.model(x)[0].cpu().numpy()
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
