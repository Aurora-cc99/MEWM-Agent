"""SOFTNet-style peak spotting (blueprint phase 3, 2026-09-02).
"""

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


# ---------------------------------------------------------------------------
# Features: (u, v, epsilon) from one cached flow field
# ---------------------------------------------------------------------------


def extract_flow_tensor(flow: np.ndarray,
                        landmarks: Optional[np.ndarray] = None) -> np.ndarray:
    """One 42x42x3 (u, v, epsilon) tensor from a flow field, SoftNet-style.
    """
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"expected an HxWx2 flow field, got {flow.shape}")
    import cv2
    u = flow[..., 0]
    v = flow[..., 1]

    if landmarks is not None and np.asarray(landmarks).shape == (68, 2):
        pts = np.asarray(landmarks, dtype=np.int64)
        # Nose region (part 28) -- global head motion reference, SoftNet's choice.
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

        # Eye masking (SoftNet: hexagons over both eyes, zeroed before ROIs).
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

        # ROI crops: eyebrow band (parts 17-26) and mouth band (parts 50-64).
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

    # SoftNet's normalize(): per-channel min-max per image.
    for channel in range(3):
        lo, hi = float(tensor[..., channel].min()), float(tensor[..., channel].max())
        if hi - lo > 1e-6:
            tensor[..., channel] = (tensor[..., channel] - lo) / (hi - lo)
    return tensor


# ---------------------------------------------------------------------------
# Model: shallow three-stream CNN (mirrors SOFTNet)
# ---------------------------------------------------------------------------


def _torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class SoftNetModel:
    """Lazy torch wrapper so the module imports without torch."""

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


# ---------------------------------------------------------------------------
# Pseudo-labels: window IoU against the annotated intervals (SoftNet loss)
# ---------------------------------------------------------------------------


def pseudo_labels(intervals: Sequence[Tuple[int, int]], n_frames: int,
                  k: int) -> np.ndarray:
    """``y[i] = 1`` iff the window ``[i, i + k]`` overlaps an annotated interval.

    SoftNet's construction: for every frame index the k-window's IoU against the
    ground truth is checked and any positive overlap yields label 1. Frames beyond
    ``n_frames - k`` are dropped (the window would run past the video).
    """
    labels = np.zeros(max(0, n_frames - k), dtype=np.float32)
    for onset, offset in intervals:
        for index in range(len(labels)):
            lo = max(index, onset)
            hi = min(index + k, offset)
            if hi > lo:
                labels[index] = 1.0
    return labels


# ---------------------------------------------------------------------------
# Spotting: smoothing + adaptive threshold + peaks + [peak-k, peak+k]
# ---------------------------------------------------------------------------


def spot_peaks(scores: np.ndarray, k: int, p: float = 0.55,
               fps: float = 30.0) -> List[Tuple[int, int, int, float]]:
    """Convert a per-frame score sequence into proposals.

    Returns ``(t_on, t_off, apex, peak_score)`` per peak. ``k`` is the dataset's
    average micro-expression half-length in frames; the score curve is truncated by
    ``k`` on both ends by the smoothing, so proposals are shifted by ``+k`` back onto
    the original frame axis.
    """
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
        frame = int(peak) + k  # aggregate index k maps back to frame index k
        proposals.append((frame - k, frame + k, frame,
                          float(aggregated[peak])))
    return proposals


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------


@dataclass
class SoftNetFeatureCache:
    """Per-video (u, v, epsilon) tensors aligned with the flow-pair frame axis."""

    video: str
    frames: np.ndarray            # frame indices (the flow target frame t)
    tensors: np.ndarray           # (N, 42, 42, 3) float16
    k: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, frames=self.frames, tensors=self.tensors, k=self.k)

    @classmethod
    def load(cls, path: Path) -> "SoftNetFeatureCache":
        data = np.load(path)
        return cls(video=path.stem, frames=data["frames"],
                   tensors=data["tensors"], k=int(data["k"]))


# ---------------------------------------------------------------------------
# Spotter (inference) + fold training
# ---------------------------------------------------------------------------


@dataclass
class SoftNetCheckpoint:
    fold: str
    k: int
    p: float
    state: Dict[str, np.ndarray]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, fold=self.fold, k=self.k, p=self.p, **self.state)

    @classmethod
    def load(cls, path: Path) -> "SoftNetCheckpoint":
        data = np.load(path)
        state = {name: data[name] for name in data.files
                 if name not in ("fold", "k", "p")}
        return cls(fold=str(data["fold"]), k=int(data["k"]),
                   p=float(data["p"]), state=state)


class SoftNetSpotter:
    """Loaded fold checkpoint: per-frame score curve + peak proposals."""

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
        """Peak proposals on the video's own frame axis.

        Capped at the ``top_m`` strongest peaks (by smoothed height) so the union
        route never floods the proposal pool -- the prediction system already
        contributes its own spans, and the SOFTNet route is there to rescue missed
        events, not to re-describe the whole curve.
        """
        scores = self.score(cache.tensors)
        proposals = spot_peaks(scores, self.checkpoint.k, self.checkpoint.p)
        proposals.sort(key=lambda item: -item[3])
        proposals = proposals[: max(1, top_m)]
        shifted = [(t_on + frame_offset, t_off + frame_offset,
                    apex + frame_offset, peak)
                   for t_on, t_off, apex, peak in proposals]
        return shifted


def train_fold(
    train_caches: Sequence[SoftNetFeatureCache],
    intervals_by_video: Dict[str, Sequence[Tuple[int, int]]],
    fold_name: str,
    k: int,
    epochs: int = 10,
    batch: int = 128,
    p: float = 0.55,
    device: str = "cuda",
    state_path: Optional[Path] = None,
) -> SoftNetCheckpoint:
    """LOSO fold training: pool = every video but the held-out subject's."""
    import torch
    import torch.nn as nn

    model = SoftNetModel()
    if state_path is not None and Path(state_path).is_file():
        model.load_state_dict({k2: v2 for k2, v2 in
                               np.load(state_path).items()})
    model.to(device)
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.0005)

    rng = np.random.default_rng(1)
    # -- assemble the frame pool once --------------------------------------
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for cache in train_caches:
        labels = pseudo_labels(intervals_by_video.get(cache.video, []),
                               cache.tensors.shape[0], k)
        # SoftNet drops the last k frames: the k-window would run past the video,
        # so the labels are n-k long and the tensors are trimmed to match.
        tensors = cache.tensors[: len(labels)]
        negatives = np.where(labels == 0)[0]
        positives = np.where(labels == 1)[0]
        # Cap negatives at ~4x the positives (SoftNet halves them; our pool is far
        # sparser -- 0.8% positives -- so a fixed halving still leaves a 64:1
        # imbalance and the model collapses to an all-zero predictor, MSE ~ the
        # label prior. A 4:1 cap keeps the baseline without drowning the signal.)
        if len(positives):
            cap = 4 * len(positives)
            keep_neg = (rng.choice(negatives, size=min(len(negatives), cap),
                                   replace=False)
                        if len(negatives) > cap else negatives)
        else:
            keep_neg = (rng.choice(negatives, size=min(len(negatives), 200),
                                   replace=False)
                        if len(negatives) else negatives)
        chosen = np.sort(np.concatenate([keep_neg, positives]))
        xs.append(tensors[chosen].astype(np.float32))
        ys.append(labels[chosen])
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    order = rng.permutation(len(x_all))
    x_all, y_all = x_all[order], y_all[order]
    LOGGER.info("softnet fold %s: %d samples, %d positives",
                fold_name, len(x_all), int(y_all.sum()))

    # -- augmentation on positives (flip / blur / noise) -------------------
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

    y_float = y_all.astype(np.float32)
    for epoch in range(epochs):
        model.train()
        losses: List[float] = []
        perm = rng.permutation(len(x_all))
        for start in range(0, len(x_all), batch):
            indices = perm[start:start + batch]
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
        LOGGER.info("softnet fold %s epoch %d: mse %.4f (n=%d)",
                    fold_name, epoch, float(np.mean(losses)), len(x_all))

    state = {}
    for name, tensor in model.state_dict().items():
        state[name] = tensor.detach().cpu().numpy()
    return SoftNetCheckpoint(fold=fold_name, k=k, p=p, state=state)


__all__ = [
    "extract_flow_tensor", "pseudo_labels", "spot_peaks", "SoftNetModel",
    "SoftNetFeatureCache", "SoftNetCheckpoint", "SoftNetSpotter", "train_fold",
]
