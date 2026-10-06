"""SoftNet-based ME spotter: lightweight localisation prior for the pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

_FEATURE_ROOT = "softnet_features"
_CHECKPOINT_ROOT = "softnet_spotter"
_IMAGE_SIZE = 42

def extract_flow_tensor(flow: np.ndarray,
                        landmarks: Optional[np.ndarray] = None) -> np.ndarray:
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"expected an HxWx2 flow field, got {flow.shape}")
    import cv2
    u = flow[..., 0]
    v = flow[..., 1]

    if landmarks is not None and np.asarray(landmarks).shape == (68, 2):
        pts = np.asarray(landmarks, dtype=np.int64)
        nose = u[max(0, pts[28, 1] - 5):pts[28, 1] + 6,
                  max(0, pts[28, 0] - 5):pts[28, 0] + 6]
        u = np.abs(u - float(nose.mean()))
        nose_v = v[max(0, pts[28, 1] - 5):pts[28, 1] + 6,
                   max(0, pts[28, 0] - 5):pts[28, 0] + 6]
        v = np.abs(v - float(nose_v.mean()))

        def grad_x(a: np.ndarray) -> np.ndarray:
            g = np.zeros_like(a)
            g[:, :-1] = a[:, 1:] - a[:, :-1]
            return g

        def grad_y(a: np.ndarray) -> np.ndarray:
            g = np.zeros_like(a)
            g[:-1, :] = a[1:, :] - a[:-1, :]
            return g

        epsilon = np.sqrt(grad_x(u) ** 2 + grad_y(v) ** 2
                          + 0.5 * (grad_y(u) + grad_x(v)) ** 2)
        epsilon = epsilon - float(epsilon[max(0, pts[28, 1] - 5):pts[28, 1] + 6,
                                           max(0, pts[28, 0] - 5):pts[28, 0] + 6].mean())

        full = np.stack([u, v, epsilon], axis=-1).astype(np.float32)
        left_eye = [(pts[36 + i, 0] + (15 if i == 0 else 0),
                     pts[36 + i, 1] + (-15 if i == 1 else 0)) for i in range(3)]
        left_eye += [(pts[39, 0] + 15, pts[39, 1]), (pts[40, 0], pts[40, 1] + 15),
                     (pts[41, 0], pts[41, 1] + 15)]
        right_eye = [(pts[42 + i, 0] - (15 if i == 0 else 0),
                      pts[42 + i, 1] + (-15 if i == 1 else 0)) for i in range(3)]
        right_eye += [(pts[45, 0] + 15, pts[45, 1]), (pts[46, 0], pts[46, 1] + 15),
                      (pts[47, 0], pts[47, 1] + 15)]
        h, w = u.shape
        for eye in (left_eye, right_eye):
            polygon = np.array([[min(w - 1, max(0, x)), min(h - 1, max(0, y))]
                                for x, y in eye], dtype=np.int32)
            cv2.fillPoly(full, [polygon], 0)

        x_lo = max(0, int(pts[17, 0]) - 12)
        x_hi = min(w, int(pts[26, 0]) + 12)
        y_lo = max(0, min(int(pts[19, 1]), int(pts[24, 1])) - 12)
        y_hi = min(h, max(int(pts[41, 1]), int(pts[46, 1])) + 12)
        m_lo = max(0, int(pts[50, 1]) - 12)
        m_hi = min(h, int(pts[57, 1]) + 12)
        m_x_lo = max(0, int(pts[60, 0]) - 12)
        m_x_hi = min(w, int(pts[64, 0]) + 12)
        tensor = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.float32)
        tensor[:21] = cv2.resize(full[y_lo:y_hi, x_lo:x_hi], (_IMAGE_SIZE, 21))
        tensor[21:] = cv2.resize(full[m_lo:m_hi, m_x_lo:m_x_hi], (_IMAGE_SIZE, 21))
    else:
        def grad_x(a: np.ndarray) -> np.ndarray:
            g = np.zeros_like(a)
            g[:, :-1] = a[:, 1:] - a[:, :-1]
            return g

        def grad_y(a: np.ndarray) -> np.ndarray:
            g = np.zeros_like(a)
            g[:-1, :] = a[1:, :] - a[:-1, :]
            return g

        u = u - float(np.median(u))
        v = v - float(np.median(v))
        epsilon = np.sqrt(grad_x(u) ** 2 + grad_y(v) ** 2
                          + 0.5 * (grad_y(u) + grad_x(v)) ** 2)
        tensor = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.float32)
        tensor[..., 0] = cv2.resize(u, (_IMAGE_SIZE, _IMAGE_SIZE))
        tensor[..., 1] = cv2.resize(v, (_IMAGE_SIZE, _IMAGE_SIZE))
        tensor[..., 2] = cv2.resize(epsilon, (_IMAGE_SIZE, _IMAGE_SIZE))

    for channel in range(3):
        lo, hi = float(tensor[..., channel].min()), float(tensor[..., channel].max())
        if hi - lo > 1e-6:
            tensor[..., channel] = (tensor[..., channel] - lo) / (hi - lo)
    return tensor

def _torch() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False

class SoftNetModel:

    def __init__(self) -> None:
        import torch
        import torch.nn as nn
        self._torch = torch
        self._nn = nn
        self.model = self._build()

    def _build(self):
        nn = self._nn
        inputs = [nn.Sequential(
            nn.Conv2d(1, filters, (5, 5), padding="same"),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), stride=(3, 3)),
        ) for filters in (3, 5, 8)]
        merged = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear((3 + 5 + 8) * 7 * 7, 400),
            nn.ReLU(),
            nn.Linear(400, 1),
        )
        return nn.ModuleList(inputs + [merged])

    def forward(self, u, v, eps):
        torch = self._torch
        streams = [self.model[0](u), self.model[1](v), self.model[2](eps)]
        merged = torch.cat(streams, dim=1)
        return self.model[3](merged).squeeze(-1)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        converted = {}
        for name, value in state.items():
            if not hasattr(value, "shape") or isinstance(value, np.ndarray):
                import torch
                converted[name] = torch.from_numpy(np.asarray(value))
            else:
                converted[name] = value
        return self.model.load_state_dict(converted)

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

def pseudo_labels(intervals: Sequence[Tuple[int, int]], n_frames: int,
                  k: int) -> np.ndarray:
    labels = np.zeros(max(0, n_frames - k), dtype=np.float32)
    for onset, offset in intervals:
        for index in range(len(labels)):
            lo = max(index, onset)
            hi = min(index + k, offset)
            if hi > lo:
                labels[index] = 1.0
    return labels

def spot_peaks(scores: np.ndarray, k: int, p: float = 0.55,
               fps: float = 30.0) -> List[Tuple[int, int, int, float]]:
    from scipy.signal import find_peaks

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size <= 2 * k:
        return []
    aggregated = scores.copy()
    for x in range(k, len(scores) - k):
        aggregated[x] = scores[x - k: x + k].mean()
    aggregated = aggregated[k:-k]
    if aggregated.size == 0 or float(aggregated.max()) <= 1e-9:
        return []
    threshold = float(aggregated.mean()
                      + p * (aggregated.max() - aggregated.mean()))
    peaks, _ = find_peaks(aggregated, height=threshold, distance=max(1, k))
    proposals = []
    for peak in peaks:
        frame = int(peak) + k
        proposals.append((frame - k, frame + k, frame,
                          float(aggregated[peak])))
    return proposals

@dataclass
class SoftNetFeatureCache:

    video: str
    frames: np.ndarray
    tensors: np.ndarray
    k: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frames=self.frames, tensors=self.tensors, k=self.k)

    @classmethod
    def load(cls, path: Path) -> "SoftNetFeatureCache":
        data = np.load(path)
        return cls(video=path.stem, frames=data["frames"],
                   tensors=data["tensors"], k=int(data["k"]))

def build_feature_caches(videos, feature_root, force: bool = False):
    from .v1_motion import MotionFrontEnd, detect_landmarks

    caches: List[SoftNetFeatureCache] = []
    for video in videos:
        target = feature_root / f"{video.video_key}.npz"
        if target.is_file() and not force:
            caches.append(SoftNetFeatureCache.load(target))
            continue
        pairs = video.paths.aligned_pairs()
        if not pairs:
            LOGGER.warning("%s: no flow pairs; skipping softnet feature build",
                          video.video_id)
            continue
        first = video.paths.frame(video.frame_lo)
        if not first.is_file():
            for index in range(video.frame_lo,
                              min(video.frame_hi + 1, video.frame_lo + 30)):
                candidate = video.paths.frame(index)
                if candidate.is_file():
                    first = candidate
                    break
        landmarks = None
        if first.is_file():
            try:
                landmarks = detect_landmarks(first)
            except Exception:
                landmarks = None
        tensors, frames = [], []
        for pair in pairs:
            flow = MotionFrontEnd.load_flow(
                MotionFrontEnd.prefer_raw_flow(pair.flow_path))
            if flow is None:
                continue
            tensors.append(extract_flow_tensor(flow, landmarks))
            frames.append(pair.t)
        if not tensors:
            LOGGER.warning("%s: no decodable flow for softnet features",
                          video.video_id)
            continue
        cache = SoftNetFeatureCache(video=video.video_key,
                                    frames=np.asarray(frames, dtype=np.int64),
                                    tensors=np.asarray(tensors, dtype=np.float16),
                                    k=video.flow_gap)
        cache.save(target)
        caches.append(cache)
    return caches

@dataclass
class SoftNetCheckpoint:
    fold: str
    k: int
    p: float
    state: Dict[str, np.ndarray]
    final_mse: float = float("nan")
    trivial_mse: float = float("nan")
    val_auc: float = float("nan")
    best_epoch: int = -1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, fold=self.fold, k=self.k, p=self.p,
                 final_mse=self.final_mse, trivial_mse=self.trivial_mse,
                 val_auc=self.val_auc, best_epoch=self.best_epoch,
                 **self.state)

    @classmethod
    def load(cls, path: Path) -> "SoftNetCheckpoint":
        data = np.load(path)
        meta = ("fold", "k", "p", "final_mse", "trivial_mse", "val_auc", "best_epoch")
        state = {name: data[name] for name in data.files
                 if name not in meta}
        return cls(fold=str(data["fold"]), k=int(data["k"]),
                   p=float(data["p"]),
                   final_mse=float(data["final_mse"]) if "final_mse" in data.files else float("nan"),
                   trivial_mse=float(data["trivial_mse"]) if "trivial_mse" in data.files else float("nan"),
                   val_auc=float(data["val_auc"]) if "val_auc" in data.files else float("nan"),
                   best_epoch=int(data["best_epoch"]) if "best_epoch" in data.files else -1,
                   state=state)

class SoftNetSpotter:

    def __init__(self, checkpoint: SoftNetCheckpoint, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.model = SoftNetModel()
        self.model.load_state_dict(checkpoint.state)
        import torch
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def score(self, tensors: np.ndarray, batch: int = 128) -> np.ndarray:
        import torch
        x = torch.from_numpy(np.asarray(tensors, dtype=np.float32)).to(self.device)
        scores = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch):
                chunk = x[start:start + batch]
                out = self.model.forward(
                    chunk[..., 0:1].permute(0, 3, 1, 2),
                    chunk[..., 1:2].permute(0, 3, 1, 2),
                    chunk[..., 2:3].permute(0, 3, 1, 2),
                )
                scores.append(out.cpu().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)

    def propose(self, cache: SoftNetFeatureCache,
                frame_offset: int = 0,
                top_m: int = 10) -> List[Tuple[int, int, int, float]]:
        scores = self.score(cache.tensors)
        proposals = spot_peaks(scores, self.checkpoint.k, self.checkpoint.p)
        proposals.sort(key=lambda item: -item[3])
        proposals = proposals[: max(1, top_m)]
        shifted = [(t_on + frame_offset, t_off + frame_offset,
                    apex + frame_offset, peak)
                   for t_on, t_off, apex, peak in proposals]
        return shifted

def _split_softnet_subjects(
    caches: Sequence[SoftNetFeatureCache],
    intervals_by_video: Dict[str, Sequence[Tuple[int, int]]],
    subjects_by_video: Dict[str, str],
    val_subject_fraction: float,
    min_val_subjects: int,
    seed: int = 1,
) -> Tuple[List[str], List[str]]:
    subject_has_pos: Dict[str, bool] = {}
    for cache in caches:
        subject = subjects_by_video.get(cache.video, cache.video)
        has_pos = bool(intervals_by_video.get(cache.video))
        subject_has_pos[subject] = subject_has_pos.get(subject, False) or has_pos
    with_pos = sorted(s for s, has in subject_has_pos.items() if has)
    without = sorted(s for s in subject_has_pos if s not in with_pos)
    if len(with_pos) <= 1:
        return sorted(subject_has_pos), []
    rng = np.random.default_rng(seed)
    n_val = max(min_val_subjects, int(round(len(with_pos) * val_subject_fraction)))
    n_val = min(n_val, len(with_pos) - 1)
    order = rng.permutation(len(with_pos))
    val = sorted(with_pos[i] for i in order[:n_val])
    train = sorted(set(with_pos) - set(val)) + without
    return sorted(train), val

def _softnet_frame_auc(
    model: "SoftNetModel",
    val_caches: Sequence[SoftNetFeatureCache],
    intervals_by_video: Dict[str, Sequence[Tuple[int, int]]],
    k: int,
    device: str,
) -> float:
    import torch
    from ..training.localiser_supervised import _frame_auc

    model.eval()
    aucs: List[float] = []
    with torch.no_grad():
        for cache in val_caches:
            labels = pseudo_labels(intervals_by_video.get(cache.video, []),
                                   cache.tensors.shape[0], k)
            if labels.size == 0:
                continue
            tensors = cache.tensors[: len(labels)].astype(np.float32)
            xb = torch.from_numpy(tensors).to(device)
            scores = model.forward(
                xb[..., 0:1].permute(0, 3, 1, 2),
                xb[..., 1:2].permute(0, 3, 1, 2),
                xb[..., 2:3].permute(0, 3, 1, 2),
            ).detach().cpu().numpy()
            auc = _frame_auc(scores, labels, np.ones_like(labels, dtype=bool))
            if np.isfinite(auc):
                aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")

def train_fold(
    train_caches: Sequence[SoftNetFeatureCache],
    intervals_by_video: Dict[str, Sequence[Tuple[int, int]]],
    fold_name: str,
    k: int,
    epochs: int = 50,
    batch: int = 128,
    p: float = 0.55,
    device: str = "cuda",
    state_path: Optional[Path] = None,
    negative_ratio: float = 4.0,
    subjects_by_video: Optional[Dict[str, str]] = None,
    val_subject_fraction: float = 0.25,
    min_val_subjects: int = 1,
) -> SoftNetCheckpoint:
    import torch
    import torch.nn as nn

    model = SoftNetModel()
    if state_path is not None and Path(state_path).is_file():
        model.load_state_dict({k2: v2 for k2, v2 in
                               np.load(state_path).items()})
    model.to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    val_caches: List[SoftNetFeatureCache] = []
    fit_caches: Sequence[SoftNetFeatureCache] = train_caches
    if subjects_by_video:
        train_subjects, val_subjects = _split_softnet_subjects(
            train_caches, intervals_by_video, subjects_by_video,
            val_subject_fraction, min_val_subjects)
        if val_subjects:
            val_set = set(val_subjects)
            fit_caches = [c for c in train_caches
                         if subjects_by_video.get(c.video, c.video) not in val_set]
            val_caches = [c for c in train_caches
                         if subjects_by_video.get(c.video, c.video) in val_set]
            LOGGER.info("softnet fold %s: held-in val subjects %s (%d/%d videos)",
                        fold_name, val_subjects, len(val_caches), len(train_caches))
        else:
            LOGGER.info("softnet fold %s: too few positive-bearing subjects for a "
                        "held-in val split, falling back to no-validation training",
                        fold_name)

    rng = np.random.default_rng(1)
    tensor_shape: Optional[Tuple[int, ...]] = None
    pos_tensors: List[np.ndarray] = []
    neg_tensors: List[np.ndarray] = []
    for cache in fit_caches:
        labels = pseudo_labels(intervals_by_video.get(cache.video, []),
                               cache.tensors.shape[0], k)
        tensors = cache.tensors[: len(labels)].astype(np.float32)
        if tensor_shape is None and len(tensors):
            tensor_shape = tensors.shape[1:]
        negatives = np.where(labels == 0)[0]
        positives = np.where(labels == 1)[0]
        if len(positives):
            pos_tensors.append(tensors[positives])
        if len(negatives):
            neg_tensors.append(tensors[negatives])
    tensor_shape = tensor_shape or (_IMAGE_SIZE, _IMAGE_SIZE, 3)
    pos_all = (np.concatenate(pos_tensors, axis=0) if pos_tensors
              else np.zeros((0,) + tensor_shape, dtype=np.float32))
    neg_all = (np.concatenate(neg_tensors, axis=0) if neg_tensors
              else np.zeros((0,) + tensor_shape, dtype=np.float32))
    n_pos_raw = len(pos_all)
    cap = (max(1, int(round(negative_ratio * n_pos_raw))) if n_pos_raw
          else min(len(neg_all), 200))
    if len(neg_all) > cap:
        keep_idx = rng.choice(len(neg_all), size=cap, replace=False)
        neg_all = neg_all[keep_idx]
    def augment(tensor: np.ndarray) -> List[np.ndarray]:
        import cv2
        out = []
        out.append(np.fliplr(tensor))
        blurred = np.stack([cv2.GaussianBlur(tensor[..., c], (7, 7), 0)
                            for c in range(3)], axis=-1)
        out.append(blurred)
        noisy = tensor + rng.normal(0, 0.02, tensor.shape).astype(np.float32)
        out.append(noisy)
        return out

    if n_pos_raw:
        augmented = [a for tensor in pos_all for a in augment(tensor)]
        pos_all = np.concatenate([pos_all, np.stack(augmented, axis=0)], axis=0)

    x_all = np.concatenate([pos_all, neg_all], axis=0)
    y_all = np.concatenate([np.ones(len(pos_all), dtype=np.float32),
                            np.zeros(len(neg_all), dtype=np.float32)])
    order = rng.permutation(len(x_all))
    x_all, y_all = x_all[order], y_all[order]
    LOGGER.info("softnet fold %s: %d samples, %d positives (%d raw + augmented)",
                fold_name, len(x_all), int(y_all.sum()), n_pos_raw)

    y_float = y_all.astype(np.float32)
    final_epoch_mse = float("nan")
    best_auc = -np.inf
    best_epoch = -1
    best_state = None
    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        perm = rng.permutation(len(x_all))
        for start_idx in range(0, len(x_all), batch):
            indices = perm[start_idx:start_idx + batch]
            xb = torch.from_numpy(x_all[indices]).to(device)
            yb = torch.from_numpy(y_float[indices]).to(device)
            pred = model.forward(xb[..., 0:1].permute(0, 3, 1, 2),
                                 xb[..., 1:2].permute(0, 3, 1, 2),
                                 xb[..., 2:3].permute(0, 3, 1, 2))
            loss = loss_fn(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        final_epoch_mse = float(np.mean(losses))
        if val_caches:
            val_auc = _softnet_frame_auc(model, val_caches, intervals_by_video, k, device)
            LOGGER.info("softnet fold %s epoch %d: mse %.4f val_auc %.4f (n=%d)",
                        fold_name, epoch, final_epoch_mse, val_auc, len(x_all))
            if np.isfinite(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                best_state = {name: tensor.detach().clone()
                              for name, tensor in model.state_dict().items()}
        else:
            LOGGER.info("softnet fold %s epoch %d: mse %.4f (n=%d)",
                        fold_name, epoch, final_epoch_mse, len(x_all))
    label_rate = float(y_float.mean()) if len(y_float) else 0.0
    trivial_mse = label_rate * (1.0 - label_rate)
    if trivial_mse > 0 and final_epoch_mse >= 0.9 * trivial_mse:
        LOGGER.warning(
            "",
            fold_name, final_epoch_mse, trivial_mse)

    if best_state is not None:
        model.load_state_dict(best_state)
        LOGGER.info("", fold_name, best_epoch, best_auc)

    state = {}
    for name, tensor in model.state_dict().items():
        state[name] = tensor.detach().cpu().numpy()
    return SoftNetCheckpoint(fold=fold_name, k=k, p=p, state=state,
                             final_mse=final_epoch_mse, trivial_mse=trivial_mse,
                             val_auc=float(best_auc) if val_caches else float("nan"),
                             best_epoch=best_epoch if val_caches else -1)

__all__ = [
    "extract_flow_tensor", "pseudo_labels", "spot_peaks", "SoftNetModel",
    "SoftNetFeatureCache", "SoftNetCheckpoint", "SoftNetSpotter", "train_fold",
]
