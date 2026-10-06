"""M2 localiser: refines and confirms ME interval boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SpottingConfig
from .m2_spotting import CandidateInterval, PhysioEvent

__all__ = ["MicroLocaliser", "double_burst_kernel", "localiser_diagnostics"]


def double_burst_kernel(
    duration: int,
    flank_fraction: float = 0.3,
    trough_weight: float = 1.0,
) -> np.ndarray:
    # zero-mean so a constant offset contributes nothing to the correlation score
    duration = max(3, int(duration))
    flank = max(1, int(round(flank_fraction * duration)))
    flank = min(flank, (duration - 1) // 2)
    kernel = np.full(duration, -float(trough_weight), dtype=np.float64)
    kernel[:flank] = 1.0
    kernel[duration - flank:] = 1.0
    kernel -= kernel.mean()
    norm = np.linalg.norm(kernel)
    if norm <= 0:
        kernel = np.zeros(duration)
        kernel[0] = 1.0
        return kernel
    return kernel / norm


def _moving_stats(x: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    width = max(1, int(width))
    n = x.size
    pad = width // 2
    padded = np.pad(x, (pad, width - 1 - pad), mode="edge")
    c1 = np.concatenate(([0.0], np.cumsum(padded)))
    c2 = np.concatenate(([0.0], np.cumsum(padded * padded)))
    total = c1[width:width + n] - c1[:n]
    total_sq = c2[width:width + n] - c2[:n]
    mean = total / width
    var = np.maximum(total_sq / width - mean * mean, 0.0)
    return mean, np.sqrt(var)


def _correlate_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.size
    pad_lo = k // 2
    pad_hi = k - 1 - pad_lo
    padded = np.pad(x, (pad_lo, pad_hi), mode="edge")
    return np.correlate(padded, kernel, mode="valid")


@dataclass
class LocaliserResponse:
    response: np.ndarray
    best_duration: np.ndarray
    durations: List[int]


class MicroLocaliser:

    def __init__(self, config: Optional[SpottingConfig] = None, fps: float = 30.0) -> None:
        self.config = config or SpottingConfig()
        self.fps = float(fps) if fps and fps > 0 else 30.0

    MIN_DECODABLE_FRAMES = 3

    def duration_bank_for(self, n_frames: int = 0) -> List[int]:
        lo = self.MIN_DECODABLE_FRAMES
        if self.config.min_micro_seconds > 0:
            lo = max(lo, int(round(self.config.min_micro_seconds * self.fps)))

        if self.config.max_micro_seconds > 0:
            hi = int(round(self.config.max_micro_seconds * self.fps))
        elif n_frames and n_frames > 0:
            hi = int(n_frames) // 4
        else:
            hi = int(round(4.0 * self.fps))
        hi = max(lo + 1, hi)

        size = max(2, int(self.config.localiser_bank_size))
        ratio = (hi / lo) ** (1.0 / (size - 1))
        bank = {lo, hi}
        for i in range(size):
            bank.add(int(round(lo * ratio ** i)))
        return sorted(d for d in bank if lo <= d <= hi)

    @property
    def duration_bank(self) -> List[int]:
        return self.duration_bank_for(0)

    def response(self, s_curve: np.ndarray) -> LocaliserResponse:
        s = np.asarray(s_curve, dtype=np.float64).reshape(-1)
        bank = self.duration_bank_for(s.size)
        if s.size < 3:
            zeros = np.zeros(s.size)
            return LocaliserResponse(zeros, zeros.astype(int), bank)

        scale = float(s.std())
        if not np.isfinite(scale) or scale <= 0.0:
            zeros = np.zeros(s.size)
            return LocaliserResponse(zeros, zeros.astype(int), bank)
        std_floor = 1e-6 * scale

        best = np.full(s.size, -np.inf)
        best_d = np.zeros(s.size, dtype=int)
        for duration in bank:
            if duration >= s.size:
                continue
            kernel = double_burst_kernel(
                duration, self.config.localiser_flank_fraction,
                self.config.localiser_trough_weight)
            raw = _correlate_same(s, kernel)
            _, std = _moving_stats(s, duration)
            shape = np.where(std > std_floor,
                             raw / (std * np.sqrt(duration) + 1e-12), 0.0)

            flank = max(1, int(round(self.config.localiser_flank_fraction * duration)))
            energy_kernel = np.zeros(duration)
            energy_kernel[:flank] = 1.0 / (2 * flank)
            energy_kernel[duration - flank:] = 1.0 / (2 * flank)
            energy = _correlate_same(s, energy_kernel)

            score = np.maximum(shape, 0.0) * np.maximum(energy, 0.0)
            improved = score > best
            best_d[improved] = duration
            best = np.where(improved, score, best)

        best[~np.isfinite(best)] = 0.0
        return LocaliserResponse(best, best_d, bank)

    def localise(
        self,
        s_curve: np.ndarray,
        t_start: int = 0,
        per_slot: Optional[np.ndarray] = None,
        physio_events: Optional[Sequence[PhysioEvent]] = None,
    ) -> List[CandidateInterval]:
        s = np.asarray(s_curve, dtype=np.float64).reshape(-1)
        out = self.response(s)
        response, best_d = out.response, out.best_duration
        if response.size == 0:
            return []

        order = np.argsort(-response)
        min_sep = max(1, int(round(self.config.localiser_min_separation_seconds * self.fps)))
        picked: List[Tuple[int, int, int, float]] = []
        for centre in order:
            score = float(response[centre])
            if score <= self.config.localiser_min_score:
                break
            duration = int(best_d[centre]) or out.durations[0]
            lo = int(centre) - duration // 2
            hi = lo + duration - 1
            lo = max(0, lo)
            hi = min(s.size - 1, hi)
            if hi - lo + 1 < 3:
                continue
            if any(abs(int(centre) - c) < min_sep or _iou((lo, hi), (a, b))
                   > self.config.localiser_nms_iou
                   for c, a, b, _ in picked):
                continue
            picked.append((int(centre), lo, hi, score))
            if len(picked) >= self.config.localiser_max_per_video:
                break

        intervals: List[CandidateInterval] = []
        for order_i, (centre, lo, hi, score) in enumerate(picked):
            intervals.append(CandidateInterval(
                cid=f"L{order_i + 1:02d}",
                t_on=t_start + lo,
                t_off=t_start + hi,
                apex=t_start + centre,
                peak_S=round(score, 4),
                attribution=_attribute(
                    per_slot, lo, hi, self.config.attribution_sharpen_temperature),
                physio_overlap=any(e.overlaps(lo, hi) for e in (physio_events or [])),
                channel="micro",
                notes="extent decoded by double-burst matched filter (M2b)",
            ))
        return intervals


def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union if union else 0.0


def _attribute(per_slot: Optional[np.ndarray], lo: int, hi: int,
               temperature: Optional[float] = None) -> Dict[str, float]:
    if per_slot is None:
        return {}
    arr = np.asarray(per_slot, dtype=np.float64)
    if arr.size == 0 or arr.ndim != 2:
        return {}
    from ..engines.m2_spotting import ProposalGenerator
    return ProposalGenerator._attribute(arr, lo, hi, temperature)


def localiser_diagnostics(
    intervals: Sequence[CandidateInterval],
    truth: Sequence[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    best = []
    for t in truth:
        best.append(max((_iou((i.t_on, i.t_off), t) for i in intervals), default=0.0))
    best_arr = np.array(best) if best else np.zeros(0)
    return {
        "n_intervals": len(intervals),
        "n_truth": len(truth),
        "n_found_any": int((best_arr > 0).sum()),
        "n_hit": int((best_arr >= iou_threshold).sum()),
        "mean_best_iou": float(best_arr.mean()) if best_arr.size else 0.0,
    }
