"""V2 slot encoder: AU-centered object-slot representation of facial regions."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RepresentationConfig
from ..knowledge.au_anatomy import (
    K_SLOTS, N_ROI, ROI_INDEX, SLOT_AUS, SLOT_INDEX, regions_of,
)
from ..schemas import ROIMeasurement

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    torch = None
    nn = object
    _TORCH = False

def build_routing_mask() -> np.ndarray:
    mask = np.zeros((K_SLOTS, N_ROI), dtype=np.float32)
    for au in SLOT_AUS:
        for roi in regions_of(au):
            mask[SLOT_INDEX[au], ROI_INDEX[roi] - 1] = 1.0
    return mask

ROUTING_MASK: np.ndarray = build_routing_mask()

def slot_region_counts() -> Dict[str, int]:
    return {au: int(ROUTING_MASK[SLOT_INDEX[au]].sum()) for au in SLOT_AUS}

@dataclass
class SlotReadout:

    au: str
    activation: float
    magnitude: float
    coherence: float
    fit_score: float
    n_salient: int = 0

    def to_dict(self) -> Dict[str, float | str | int]:
        return {
            "au": self.au, "activation": round(self.activation, 4),
            "magnitude": round(self.magnitude, 4), "coherence": round(self.coherence, 4),
            "fit_score": round(self.fit_score, 4), "n_salient": self.n_salient,
        }

def analytic_slot_readout(
    measurements: Sequence[ROIMeasurement],
    magnitude_scale: float = 0.25,
) -> Dict[str, SlotReadout]:
    from ..knowledge.au_anatomy import AU_ROI_PRIOR, direction_fit_score

    by_roi = {m.roi_name: m for m in measurements}
    out: Dict[str, SlotReadout] = {}
    for au in SLOT_AUS:
        rois = regions_of(au)
        if not rois:
            out[au] = SlotReadout(au, 0.0, 0.0, 0.0, 0.0)
            continue
        magnitudes, coherences, fits, salient = [], [], [], 0
        for roi in rois:
            measurement = by_roi.get(roi)
            if measurement is None:
                continue
            prior = AU_ROI_PRIOR.get((au, roi))
            expected = prior[1] if prior else []
            magnitudes.append(measurement.magnitude_px)
            coherences.append(measurement.coherence)
            fits.append(direction_fit_score(measurement.direction_deg, expected))
            salient += int(measurement.salient)
        if not magnitudes:
            out[au] = SlotReadout(au, 0.0, 0.0, 0.0, 0.0)
            continue
        magnitude = float(np.mean(magnitudes))
        coherence = float(np.mean(coherences))
        fit = float(np.mean(fits))
        strength = (1.0 - math.exp(-magnitude / magnitude_scale)) * coherence * fit
        out[au] = SlotReadout(au, round(float(np.clip(strength, 0.0, 1.0)), 5),
                              magnitude, coherence, fit, salient)
    return out

def coherence_is_saturated(
    measurements: Sequence[ROIMeasurement], median_floor: float = 0.90,
    spread_floor: float = 0.15,
) -> bool:
    if len(measurements) < 4:
        return False
    values = np.array([m.coherence for m in measurements], dtype=np.float64)
    return bool(np.median(values) >= median_floor
                and (values.max() - values.min()) <= spread_floor + (1.0 - median_floor))

def select_active_slots(
    readout: Dict[str, SlotReadout],
    threshold: float = 0.35,
    max_active: int = 5,
    relative_margin: float = 0.75,
    weak_threshold: float = 0.20,
) -> Tuple[List[str], List[str]]:
    scored = sorted(((r.activation, au) for au, r in readout.items()), reverse=True)
    if not scored:
        return [], []
    best = scored[0][0]
    if best <= 1e-6:
        return [], []

    active = [
        au for score, au in scored[:max_active]
        if score >= threshold and score >= relative_margin * best
    ]
    weak = [
        au for score, au in scored
        if au not in active and score >= weak_threshold
    ][:max_active]
    return (sorted(active, key=lambda a: int(a[2:])),
            sorted(weak, key=lambda a: int(a[2:])))

if _TORCH:

    class SlotEncoder(nn.Module):

        def __init__(self, config: Optional[RepresentationConfig] = None) -> None:
            super().__init__()
            self.config = config or RepresentationConfig()
            self.slot_dim = self.config.slot_dim
            self.appearance_dim = self.config.appearance_dim

            mask = torch.from_numpy(ROUTING_MASK)
            self.register_buffer("routing_mask", mask)
            self.register_buffer("region_counts", mask.sum(dim=1).clamp(min=1.0))

            in_dim = N_ROI * 4 + self.appearance_dim
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(in_dim, self.slot_dim),
                    nn.LayerNorm(self.slot_dim),
                    nn.GELU(),
                    nn.Linear(self.slot_dim, self.slot_dim),
                )
                for _ in range(K_SLOTS)
            ])
            self.readout = nn.Parameter(torch.zeros(K_SLOTS, self.slot_dim))
            nn.init.normal_(self.readout, std=0.02)
            self.readout_bias = nn.Parameter(torch.zeros(K_SLOTS))

        def forward(
            self,
            measurements: "torch.Tensor",
            appearance: Optional["torch.Tensor"] = None,
            slot_dropout: float = 0.0,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            batch = measurements.shape[0]
            flat = measurements.reshape(batch, -1)
            if appearance is None:
                appearance = measurements.new_zeros((batch, K_SLOTS, self.appearance_dim))

            observed = measurements.new_ones((batch, K_SLOTS))
            if slot_dropout > 0.0:
                observed = (torch.rand_like(observed) >= slot_dropout).float()

            slots, activations = [], []
            for k in range(K_SLOTS):
                gate = self.routing_mask[k].repeat_interleave(4).unsqueeze(0)
                routed = flat * gate
                encoded = self.encoders[k](torch.cat([routed, appearance[:, k]], dim=-1))
                encoded = encoded * observed[:, k : k + 1]
                slots.append(encoded)
                activations.append(
                    (encoded * self.readout[k]).sum(-1) + self.readout_bias[k]
                )

            slot_tensor = torch.stack(slots, dim=1)
            activation = torch.sigmoid(torch.stack(activations, dim=1))
            return slot_tensor, activation * observed, observed

        @torch.no_grad()
        def encode_frame(
            self, measurement_matrix: np.ndarray, appearance: Optional[np.ndarray] = None
        ) -> Tuple[np.ndarray, Dict[str, float]]:
            self.eval()
            tensor = torch.from_numpy(np.asarray(measurement_matrix, dtype=np.float32))[None]
            appearance_tensor = (
                torch.from_numpy(np.asarray(appearance, dtype=np.float32))[None]
                if appearance is not None else None
            )
            slots, activation, _ = self.forward(tensor, appearance_tensor)
            return (
                slots[0].cpu().numpy(),
                {au: float(activation[0, SLOT_INDEX[au]]) for au in SLOT_AUS},
            )

else:

    class SlotEncoder:

        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError(
                "SlotEncoder needs PyTorch. The analytic path "
                "(analytic_slot_readout / SlotBank) works without it."
            )

class SlotBank:

    def __init__(self, video_id: str, fps: float = 30.0) -> None:
        self.video_id = video_id
        self.fps = fps
        self.frames: List[int] = []
        self._activations: Dict[str, List[float]] = {au: [] for au in SLOT_AUS}
        self._vectors: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frames)

    def append(self, t: int, activations: Dict[str, float],
               slots: Optional[np.ndarray] = None) -> None:
        self.frames.append(t)
        for au in SLOT_AUS:
            self._activations[au].append(float(activations.get(au, 0.0)))
        if slots is not None:
            self._vectors[t] = np.asarray(slots, dtype=np.float32)

    def trajectory(self, au: str, t_on: Optional[int] = None,
                   t_off: Optional[int] = None) -> np.ndarray:
        series = self._activations.get(au, [])
        if not series:
            return np.zeros(0, dtype=np.float32)
        if t_on is None and t_off is None:
            return np.asarray(series, dtype=np.float32)
        lo = self._locate(t_on) if t_on is not None else 0
        hi = (self._locate(t_off) + 1) if t_off is not None else len(series)
        return np.asarray(series[lo:hi], dtype=np.float32)

    def matrix(self, t_on: Optional[int] = None, t_off: Optional[int] = None) -> np.ndarray:
        return np.stack(
            [self.trajectory(au, t_on, t_off) for au in SLOT_AUS], axis=-1
        ) if self.frames else np.zeros((0, K_SLOTS), dtype=np.float32)

    def vectors(self, t_on: int, t_off: int) -> np.ndarray:
        rows = [self._vectors[t] for t in range(t_on, t_off + 1) if t in self._vectors]
        return np.stack(rows) if rows else np.zeros((0, K_SLOTS, 0), dtype=np.float32)

    def peak(self, au: str, t_on: Optional[int] = None, t_off: Optional[int] = None) -> float:
        traj = self.trajectory(au, t_on, t_off)
        return float(traj.max()) if traj.size else 0.0

    def active_at(self, t: int, threshold: float = 0.35) -> List[str]:
        idx = self._locate(t)
        return [au for au in SLOT_AUS
                if idx < len(self._activations[au]) and self._activations[au][idx] >= threshold]

    def _locate(self, t: int) -> int:
        if not self.frames:
            return 0
        import bisect
        idx = bisect.bisect_left(self.frames, t)
        if idx < len(self.frames) and self.frames[idx] == t:
            return idx
        return max(0, min(len(self.frames) - 1, idx - 1))

    def to_dict(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id, "fps": self.fps, "n_frames": len(self.frames),
            "frame_range": [self.frames[0], self.frames[-1]] if self.frames else [],
            "peaks": {au: round(self.peak(au), 4) for au in SLOT_AUS},
        }

def phase_profile(
    trajectory: np.ndarray,
    frames: Sequence[int],
    hi: float = 0.5,
    lo: float = 0.25,
) -> Optional[Tuple[int, int, int, float, float, float]]:
    if trajectory.size == 0 or len(frames) != trajectory.size:
        return None
    peak_idx = int(np.argmax(trajectory))
    peak = float(trajectory[peak_idx])
    if peak < hi:
        return None

    start = peak_idx
    while start > 0 and trajectory[start - 1] >= lo:
        start -= 1
    end = peak_idx
    while end < trajectory.size - 1 and trajectory[end + 1] >= lo:
        end += 1

    def _slope(segment: np.ndarray) -> float:
        if segment.size < 2:
            return 0.0
        x = np.arange(segment.size, dtype=np.float64)
        return float(np.polyfit(x, segment.astype(np.float64), 1)[0])

    return (
        int(frames[start]), int(frames[peak_idx]), int(frames[end]), peak,
        _slope(trajectory[start : peak_idx + 1]),
        _slope(trajectory[peak_idx : end + 1]),
    )

__all__ = [
    "build_routing_mask", "ROUTING_MASK", "slot_region_counts", "SlotReadout",
    "analytic_slot_readout", "coherence_is_saturated", "select_active_slots",
    "SlotEncoder", "SlotBank", "phase_profile",
]
