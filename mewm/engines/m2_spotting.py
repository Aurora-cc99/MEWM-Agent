"""M2 -- prediction-error decomposition and candidate proposal generation.

    1. scene   -- explainable by the slow variable's one-step forecast and the head
                  motion projection.  The slow time constant is far longer than a
                  micro-expression, so the expressive transition cannot hide here.
    2. physio  -- matching-pursuit against a template dictionary of blink / swallow /
                  speech error shapes clustered from neutral long video.
    3. expr    -- what is left, restricted to AU anatomical regions that pass the
                  coherence gate, then decomposed per slot.  The attribution vector
                  falls straight out of this step, so a detection arrives already
                  carrying AU-level provenance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import SpottingConfig
from ..data.paths import MICRO_CEILING_FRAMES
from ..knowledge.au_anatomy import K_SLOTS, SLOT_AUS, SLOT_INDEX, regions_of
from ..schemas import CandidateInterval, ErrorRecord, PhysioEvent

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Physiological template dictionary
# ---------------------------------------------------------------------------


@dataclass
class PhysioTemplate:
    """One prototypical non-expressive error shape."""

    template_id: str
    label: str
    shape: np.ndarray               # unit-norm error profile
    duration: int

    @property
    def length(self) -> int:
        return int(self.shape.size)


def default_physio_templates(fps: float = 30.0) -> List[PhysioTemplate]:
    """Analytic blink / swallow / speech shapes, scaled to the capture rate.

    Stand-ins until :func:`fit_physio_templates` clusters real neutral-video residuals;
    the durations follow the physiology (a blink is ~100-150 ms, a swallow ~500 ms,
    speech bursts are longer and multi-modal).
    """
    def _pulse(duration: int, rise: float) -> np.ndarray:
        n = max(3, duration)
        peak = max(1, int(n * rise))
        curve = np.concatenate([
            np.linspace(0.0, 1.0, peak, endpoint=False),
            np.linspace(1.0, 0.0, n - peak),
        ])
        return curve / (np.linalg.norm(curve) + 1e-8)

    def _oscillation(duration: int, cycles: float) -> np.ndarray:
        n = max(4, duration)
        curve = np.abs(np.sin(np.linspace(0, cycles * np.pi, n)))
        return curve / (np.linalg.norm(curve) + 1e-8)

    scale = max(1.0, fps / 30.0)
    specs = [
        ("blink_fast", "blink", _pulse(int(3 * scale), 0.4), int(3 * scale)),
        ("blink_slow", "blink", _pulse(int(5 * scale), 0.35), int(5 * scale)),
        ("blink_double", "blink", _oscillation(int(8 * scale), 2.0), int(8 * scale)),
        ("swallow", "swallow", _pulse(int(15 * scale), 0.5), int(15 * scale)),
        ("swallow_long", "swallow", _pulse(int(22 * scale), 0.45), int(22 * scale)),
        ("speech_short", "speech", _oscillation(int(20 * scale), 3.0), int(20 * scale)),
        ("speech_long", "speech", _oscillation(int(45 * scale), 6.0), int(45 * scale)),
        ("head_jerk", "head", _pulse(int(6 * scale), 0.25), int(6 * scale)),
    ]
    return [PhysioTemplate(tid, label, shape, duration) for tid, label, shape, duration in specs]


def fit_physio_templates(
    residuals: Sequence[np.ndarray], n_clusters: int = 8, fps: float = 30.0,
) -> List[PhysioTemplate]:
    """Cluster error shapes harvested from neutral long video (appendix B.5 step 2).

    ``residuals`` are fixed-length error windows taken from stretches with no annotated
    expression.  k-means over L2-normalised windows; falls back to the analytic
    dictionary when there is too little material to cluster.
    """
    windows = [np.asarray(r, dtype=np.float64).reshape(-1) for r in residuals if np.size(r) > 2]
    if len(windows) < n_clusters * 3:
        LOGGER.info("only %d residual windows; keeping the analytic physio dictionary",
                    len(windows))
        return default_physio_templates(fps)

    length = int(np.median([w.size for w in windows]))
    stacked = np.stack([_resample(w, length) for w in windows])
    stacked /= (np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-8)

    rng = np.random.default_rng(0)
    centres = stacked[rng.choice(len(stacked), size=n_clusters, replace=False)]
    for _ in range(30):
        distances = ((stacked[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        assignment = distances.argmin(axis=1)
        moved = 0.0
        for k in range(n_clusters):
            members = stacked[assignment == k]
            if len(members):
                updated = members.mean(axis=0)
                updated /= (np.linalg.norm(updated) + 1e-8)
                moved = max(moved, float(np.linalg.norm(updated - centres[k])))
                centres[k] = updated
        if moved < 1e-4:
            break

    return [
        PhysioTemplate(f"cluster_{k}", "learned", centres[k], length)
        for k in range(n_clusters)
    ]


def _resample(vector: np.ndarray, length: int) -> np.ndarray:
    if vector.size == length:
        return vector
    source = np.linspace(0.0, 1.0, vector.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, vector)


# ---------------------------------------------------------------------------
# Three-way decomposition
# ---------------------------------------------------------------------------


@dataclass
class DecompositionResult:
    """Full output of the sequential projection over one video."""

    delta_total: np.ndarray
    delta_scene: np.ndarray
    delta_physio: np.ndarray
    delta_expr: np.ndarray
    per_slot: np.ndarray                       # (T, K) expressive error by AU slot
    physio_events: List[PhysioEvent] = field(default_factory=list)
    scene_r2: float = 0.0                      # how much of delta the scene term explained

    def shares(self) -> Dict[str, float]:
        total = float(np.abs(self.delta_total).sum()) + 1e-8
        return {
            "scene": round(float(np.abs(self.delta_scene).sum()) / total, 4),
            "physio": round(float(np.abs(self.delta_physio).sum()) / total, 4),
            "expr": round(float(np.abs(self.delta_expr).sum()) / total, 4),
        }


class ErrorDecomposer:
    """Sequential projection of the raw prediction error (eq. 6, appendix B.5)."""

    def __init__(self, config: Optional[SpottingConfig] = None, fps: float = 30.0) -> None:
        self.config = config or SpottingConfig()
        self.fps = fps
        self.templates = default_physio_templates(fps)

    def set_templates(self, templates: Sequence[PhysioTemplate]) -> None:
        self.templates = list(templates)

    def decompose(
        self,
        delta: np.ndarray,
        head_motion: Optional[np.ndarray] = None,
        slow_prediction: Optional[np.ndarray] = None,
        slot_errors: Optional[np.ndarray] = None,
        coherence_gate: Optional[np.ndarray] = None,
    ) -> DecompositionResult:
        """Split ``delta`` into scene + physio + expression.

        ``coherence_gate`` is the ``(T, K)`` boolean of
        ``c_{r(k),t} >= c_min`` -- the indicator in the per-slot sum of paper 3.3.2.
        Without it, incoherent noise inside an AU region would be booked as expressive.
        """
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        n = delta.size

        scene, r2 = self._scene_term(delta, head_motion, slow_prediction)
        residual = delta - scene
        # Windows already explained by coherent motion inside AU regions are withheld
        # from the physiological pass -- see _physio_term for why this is required.
        protected = self._expressive_protection(slot_errors, coherence_gate, n)
        physio, events = self._physio_term(residual, protected)
        expr = np.clip(residual - physio, 0.0, None)

        per_slot = self._attribute(expr, slot_errors, coherence_gate, n)
        # Attribution defines the expressive term: only mass that lands in an AU region
        # and passes the coherence gate survives as expressive error.
        if slot_errors is not None:
            expr = per_slot.sum(axis=1)

        return DecompositionResult(delta, scene, physio, expr, per_slot, events, r2)

    def _scene_term(
        self, delta: np.ndarray,
        head_motion: Optional[np.ndarray],
        slow_prediction: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """Least-squares projection of ``delta`` onto the head motion + slow forecast."""
        n = delta.size
        columns = [np.ones(n)]
        if head_motion is not None:
            head = np.atleast_2d(np.asarray(head_motion, dtype=np.float64))
            if head.shape[0] != n:
                head = head.T
            if head.shape[0] == n:
                columns.extend(head[:, j] for j in range(head.shape[1]))
        if slow_prediction is not None:
            slow = np.asarray(slow_prediction, dtype=np.float64).reshape(n, -1)
            columns.extend(slow[:, j] for j in range(slow.shape[1]))

        if len(columns) == 1:
            # No regressors at all -- the external-detector-curve path withholds head
            # motion and the slow forecast on purpose (run_spotting). A flat median
            # would book the curve's whole baseline as "scene": the shifted logit
            # curve's median sits at ~97% of its mean (measured 2026-08-31 across 92
            # casme_sq videos), so every candidate window read "energy mostly scene"
            # and P.scan rule 2 rejected the batch -- one of the three compounding
            # causes of that run's 0 TP. The baseline is already removed by the
            # curve's own minimum shift, so the honest scene term here is zero.
            return np.zeros(n), 0.0

        design = np.stack(columns, axis=1)
        try:
            coefficients, *_ = np.linalg.lstsq(design, delta, rcond=None)
        except np.linalg.LinAlgError:
            return np.full(n, float(np.median(delta))), 0.0

        fitted = design @ coefficients
        # The scene term must not go negative or exceed the error it explains.
        fitted = np.clip(fitted, 0.0, np.maximum(delta, 0.0))
        variance = float(np.var(delta))
        r2 = float(1.0 - np.var(delta - fitted) / variance) if variance > 1e-12 else 0.0
        return fitted, round(max(0.0, r2), 4)

    @staticmethod
    def _expressive_protection(
        slot_errors: Optional[np.ndarray],
        coherence_gate: Optional[np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Frames carrying coherent motion inside AU regions; ``True`` = off limits.
        """
        if slot_errors is None or coherence_gate is None:
            return np.zeros(n, dtype=bool)
        slots = np.clip(np.asarray(slot_errors, dtype=np.float64), 0.0, None)
        gate = np.asarray(coherence_gate, dtype=np.float64)
        if slots.shape != gate.shape or slots.shape[0] != n:
            return np.zeros(n, dtype=bool)
        gated = (slots * gate).sum(axis=1)
        return gated > 1e-9

    def _physio_term(
        self, residual: np.ndarray, protected: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[PhysioEvent]]:
        """Matching pursuit against the template dictionary.
        """
        working = residual.copy()
        explained = np.zeros_like(residual)
        events: List[PhysioEvent] = []
        n = residual.size
        if n < 3 or not self.templates:
            return explained, events
        if protected is None:
            protected = np.zeros(n, dtype=bool)

        baseline = float(np.median(residual))
        spread = 1.4826 * float(np.median(np.abs(residual - baseline)))
        energy_floor = baseline + 3.0 * max(spread, 1e-9)

        # Searching on a masked copy keeps protected frames from attracting a match in
        # the first place, rather than rejecting it after the fact.
        searchable = working.copy()
        searchable[protected] = baseline

        for _ in range(self.config.n_physio_templates):
            best = None
            for template in self.templates:
                length = min(template.length, n)
                if length < 2:
                    continue
                shape = _resample(template.shape, length)
                shape = shape / (np.linalg.norm(shape) + 1e-8)
                correlation = np.correlate(searchable, shape, mode="valid")
                if correlation.size == 0:
                    continue
                position = int(np.argmax(correlation))
                energy = float(correlation[position])
                if protected[position:position + length].any():
                    continue
                window = working[position:position + length]
                norm = float(np.linalg.norm(window))
                score = energy / (norm + 1e-8)             # shape agreement in [0, 1]
                if score < self.config.physio_match_thresh:
                    continue
                if float(window.max()) < energy_floor:      # amplitude admission test
                    continue
                if best is None or energy > best[0]:
                    best = (energy, position, length, shape, template, score)

            if best is None:
                break
            energy, position, length, shape, template, score = best
            contribution = np.clip(energy, 0.0, None) * shape
            segment = working[position:position + length]
            contribution = np.minimum(contribution, np.clip(segment, 0.0, None))
            if float(contribution.sum()) <= 1e-9:
                break
            working[position:position + length] -= contribution
            searchable[position:position + length] = baseline
            explained[position:position + length] += contribution
            events.append(PhysioEvent(
                t_start=position, t_end=position + length - 1,
                template_id=template.template_id, match_energy=round(score, 4),
                label=template.label,
            ))

        return explained, events

    def _attribute(
        self,
        expr: np.ndarray,
        slot_errors: Optional[np.ndarray],
        coherence_gate: Optional[np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Per-slot decomposition with the coherence indicator applied."""
        if slot_errors is None:
            per_slot = np.zeros((n, K_SLOTS), dtype=np.float64)
            per_slot[:, 0] = expr
            return per_slot

        slots = np.asarray(slot_errors, dtype=np.float64)
        if slots.shape[0] != n:
            slots = np.resize(slots, (n, K_SLOTS))
        slots = np.clip(slots, 0.0, None)
        if coherence_gate is not None:
            gate = np.asarray(coherence_gate, dtype=np.float64)
            if gate.shape == slots.shape:
                slots = slots * gate
        # Rescale so the per-slot mass matches the expressive residual it came from.
        row_sum = slots.sum(axis=1, keepdims=True)
        scale = np.divide(expr.reshape(-1, 1), row_sum, out=np.zeros_like(row_sum),
                          where=row_sum > 1e-9)
        return slots * scale


# ---------------------------------------------------------------------------
# Detection statistic + proposals
# ---------------------------------------------------------------------------


def robust_normalise(
    values: np.ndarray,
    window: int,
    eps: float = 1e-6,
    scale_floor_ratio: float = 0.05,
    min_window: int = 16,
    max_score: float = 1000.0,
) -> np.ndarray:
    """``S_t`` of eq. (7): trailing-window median/MAD standardisation.

    *Causality.* Eq. (7) is defined on ``[t - W, t]``. A streaming detector cannot see
    the future, so nothing computed at time ``t`` may depend on a sample after ``t``.
    Deriving the scale floor from whole-series statistics would violate this silently:
    a spike at frame 3000 would shift the score at frame 100, and the resulting
    localisation accuracy would be optimistic in a way no downstream metric reveals.
    Every statistic here -- window, fallback and floor alike -- comes from the
    *expanding prefix* ``[0, t]``.

    *Boundedness.* The expressive residual is sparse by construction, near zero except
    inside events, so a trailing MAD can legitimately collapse to zero and turn eq. (7)
    into a division by ``eps``, producing scores in the thousands that no threshold can
    be set against. The scale is floored at ``scale_floor_ratio`` of the prefix's own
    robust scale, which bounds the statistic on flat stretches while keeping it
    dimensionless and comparable across subjects, with a final clip at ``max_score``
    as a backstop.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    for t in range(n):
        prefix = values[: t + 1]
        prefix_median = float(np.median(prefix))
        prefix_scale = 1.4826 * float(np.median(np.abs(prefix - prefix_median)))
        if prefix_scale <= eps:
            # Degenerate prefix (constant so far): fall back to the spread of whatever
            # departs from the median.
            departures = prefix[np.abs(prefix - prefix_median) > eps]
            prefix_scale = float(departures.std()) if departures.size > 1 else 0.0

        segment = values[max(0, t - window + 1): t + 1]
        if segment.size < min_window:
            median, scale = prefix_median, prefix_scale
        else:
            median = float(np.median(segment))
            scale = 1.4826 * float(np.median(np.abs(segment - median)))

        deviation = values[t] - median
        floor = scale_floor_ratio * prefix_scale
        if floor <= eps:
            # No baseline variability has been observed yet, so there is no scale to
            # measure this sample against. Using eps here would emit 1e6 and make the
            # statistic unthresholdable; using the sample's own deviation says the
            # honest thing instead -- "this is the first departure, one unit of a scale
            # we cannot yet estimate" -- and stays causal.
            floor = max(abs(deviation), eps)
        out[t] = deviation / max(scale, floor)
    return np.clip(out, -max_score, max_score)


@dataclass
class SpottingResult:
    """Everything M2 produces for one video."""

    error_record: ErrorRecord
    proposals: List[CandidateInterval]
    decomposition: DecompositionResult
    macro_intervals: List[CandidateInterval] = field(default_factory=list)
    # Extents decoded by M2b's matched filter. Kept beside ``proposals`` rather
    # than replacing them: the hysteresis output still feeds the macro channel
    # and the existing reward path, and a caller that distrusts the decoder can
    # compare the two on the same curve.
    localised: List[CandidateInterval] = field(default_factory=list)

    @property
    def micro_intervals(self) -> List[CandidateInterval]:
        """The micro-expression interval set a consumer should use.
        """
        return self.localised if self.localised else self.proposals

    def summary(self) -> Dict[str, object]:
        return {
            "n_proposals": len(self.proposals),
            "n_localised": len(self.localised),
            "n_macro": len(self.macro_intervals),
            "n_physio": len(self.decomposition.physio_events),
            "error_shares": self.decomposition.shares(),
            "scene_r2": self.decomposition.scene_r2,
            "peak_S": round(float(max(self.error_record.s_curve, default=0.0)), 3),
        }


class ProposalGenerator:
    """Two-threshold hysteresis over ``S_t`` (paper 3.3.2)."""

    def __init__(self, config: Optional[SpottingConfig] = None, fps: float = 30.0) -> None:
        self.config = config or SpottingConfig()
        self.fps = fps

    @property
    def max_micro_frames(self) -> int:
        """The micro/macro routing ceiling in frames.
        """
        if self.config.max_micro_seconds > 0:
            return max(2, int(round(self.config.max_micro_seconds * self.fps)))
        return MICRO_CEILING_FRAMES

    def generate(
        self,
        s_curve: np.ndarray,
        t_start: int,
        per_slot: Optional[np.ndarray] = None,
        physio_events: Optional[Sequence[PhysioEvent]] = None,
    ) -> Tuple[List[CandidateInterval], List[CandidateInterval]]:
        """Hysteresis segmentation; returns ``(micro proposals, macro intervals)``.
        """
        s_curve = np.asarray(s_curve, dtype=np.float64).reshape(-1)
        spans = self._hysteresis_spans(s_curve)
        spans = self._merge_close(spans)

        floor = max(0, int(self.config.min_duration_frames))
        ceiling = self.max_micro_frames
        micro: List[CandidateInterval] = []
        macro: List[CandidateInterval] = []
        for order, (lo, hi) in enumerate(spans):
            duration = hi - lo + 1
            if floor and duration < floor:
                continue
            segment = s_curve[lo:hi + 1]
            apex_offset = int(np.argmax(segment))
            attribution = self._attribute(
                per_slot, lo, hi, self.config.attribution_sharpen_temperature)
            overlap = any(
                event.overlaps(lo, hi) for event in (physio_events or [])
            )
            channel = "macro" if ceiling and duration > ceiling else "micro"
            interval = CandidateInterval(
                cid=f"p{order + 1:02d}",
                t_on=t_start + lo,
                t_off=t_start + hi,
                apex=t_start + lo + apex_offset,
                peak_S=round(float(segment.max()), 4),
                attribution=attribution,
                physio_overlap=overlap,
                channel=channel,
                notes=("duration exceeds the micro-expression ceiling; "
                       "handed to the macro channel" if channel == "macro" else ""),
            )
            (micro if channel == "micro" else macro).append(interval)

        for order, interval in enumerate(micro):
            interval.cid = f"p{order + 1:02d}"
        for order, interval in enumerate(macro):
            interval.cid = f"M{order + 1:02d}"
        return micro, macro

    def _hysteresis_spans(self, s_curve: np.ndarray) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        inside, start = False, 0
        for t, value in enumerate(s_curve):
            if not inside and value >= self.config.tau_hi:
                inside, start = True, t
                # Walk the onset back to where the curve first left the low threshold,
                # so the reported boundary is the true departure, not the trigger point.
                while start > 0 and s_curve[start - 1] >= self.config.tau_lo:
                    start -= 1
            elif inside and value < self.config.tau_lo:
                spans.append((start, t - 1))
                inside = False
        if inside:
            spans.append((start, len(s_curve) - 1))
        return spans

    def _merge_close(self, spans: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Join spans separated by less than ``merge_gap_frames`` (one flickering event)."""
        if not spans:
            return []
        merged = [list(spans[0])]
        for lo, hi in spans[1:]:
            if lo - merged[-1][1] - 1 <= self.config.merge_gap_frames:
                merged[-1][1] = hi
            else:
                merged.append([lo, hi])
        return [(int(lo), int(hi)) for lo, hi in merged]

    @staticmethod
    def _attribute(
        per_slot: Optional[np.ndarray], lo: int, hi: int,
        temperature: Optional[float] = None,
    ) -> Dict[str, float]:
        """``pi_{k,j}`` -- normalised share of expressive error mass per AU.
        """
        if per_slot is None or per_slot.size == 0:
            return {}
        window = np.asarray(per_slot)[lo:hi + 1]
        if window.size == 0:
            return {}
        mass = window.sum(axis=0)
        total = float(mass.sum())
        if total <= 1e-9:
            return {}
        n = min(K_SLOTS, mass.size)
        shares = np.array([mass[k] / total for k in range(n)], dtype=np.float64)
        if temperature and temperature > 0:
            scaled = shares / float(temperature)
            scaled -= scaled.max()  # shift-invariance guard against overflow
            weights = np.exp(scaled)
            weights_total = float(weights.sum())
            if weights_total > 1e-12:
                shares = weights / weights_total
        return {
            SLOT_AUS[k]: round(float(shares[k]), 4)
            for k in range(n) if shares[k] >= 0.02
        }


# ---------------------------------------------------------------------------
# End-to-end spotter
# ---------------------------------------------------------------------------


class Spotter:
    """Decompose the error, standardise it, and emit proposals."""

    def __init__(self, config: Optional[SpottingConfig] = None, fps: float = 30.0) -> None:
        self.config = config or SpottingConfig()
        self.fps = fps
        self.decomposer = ErrorDecomposer(self.config, fps)
        self.generator = ProposalGenerator(self.config, fps)
        # Imported here rather than at module scope: m2_localiser imports
        # CandidateInterval and PhysioEvent from this module.
        from .m2_localiser import MicroLocaliser
        self.localiser = MicroLocaliser(self.config, fps)

    def run(
        self,
        video_id: str,
        delta: np.ndarray,
        t_start: int = 0,
        head_motion: Optional[np.ndarray] = None,
        slow_prediction: Optional[np.ndarray] = None,
        slot_errors: Optional[np.ndarray] = None,
        coherence_gate: Optional[np.ndarray] = None,
        low_confidence_spans: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> SpottingResult:
        decomposition = self.decomposer.decompose(
            delta, head_motion, slow_prediction, slot_errors, coherence_gate
        )
        s_curve = robust_normalise(decomposition.delta_expr, self.config.window)

        events = [
            PhysioEvent(e.t_start + t_start, e.t_end + t_start, e.template_id,
                        e.match_energy, e.label)
            for e in decomposition.physio_events
        ]
        micro, macro = self.generator.generate(
            s_curve, t_start, decomposition.per_slot,
            [PhysioEvent(e.t_start - t_start, e.t_end - t_start, e.template_id,
                         e.match_energy, e.label) for e in events],
        )
        if self.config.proposal_raw_curve_frac > 0:
            micro = self._raw_curve_spans(micro, decomposition.delta_expr, t_start,
                                          self.config.proposal_raw_curve_frac)
        if (self.config.proposal_max_per_video > 0
                or self.config.proposal_trim_fraction > 0
                or self.config.proposal_soft_k_margin > 0):
            micro = self._refine_proposals(micro, decomposition.delta_expr, t_start)

        record = ErrorRecord(
            video_id=video_id,
            t_start=t_start,
            s_curve=[round(float(v), 4) for v in s_curve],
            delta_total=[round(float(v), 5) for v in decomposition.delta_total],
            delta_scene=[round(float(v), 5) for v in decomposition.delta_scene],
            delta_physio=[round(float(v), 5) for v in decomposition.delta_physio],
            delta_expr=[round(float(v), 5) for v in decomposition.delta_expr],
            au_attribution={
                SLOT_AUS[k]: [round(float(v), 5) for v in decomposition.per_slot[:, k]]
                for k in range(min(K_SLOTS, decomposition.per_slot.shape[1]))
                if decomposition.per_slot[:, k].any()
            },
            physio_events=events,
            low_confidence_spans=list(low_confidence_spans or []),
        )
        localised: List[CandidateInterval] = []
        if getattr(self.config, "localiser_enabled", False):
            localised = self.localiser.localise(
                s_curve, t_start, decomposition.per_slot,
                [PhysioEvent(e.t_start - t_start, e.t_end - t_start, e.template_id,
                             e.match_energy, e.label) for e in events],
            )
        return SpottingResult(record, micro, decomposition, macro, localised)

    def _refine_proposals(
        self,
        micro: Sequence[CandidateInterval],
        delta_expr: np.ndarray,
        t_start: int,
    ) -> List[CandidateInterval]:
        """Rank hysteresis spans by RAW expressive peak and trim each to its energy core.
        """
        expr = np.asarray(delta_expr, dtype=np.float64).reshape(-1)
        scored = []
        for proposal in micro:
            lo = max(0, proposal.t_on - t_start)
            hi = min(len(expr), proposal.t_off - t_start + 1)
            peak = float(expr[lo:hi].max()) if hi > lo else 0.0
            scored.append((peak, proposal))
        scored.sort(key=lambda item: -item[0])

        cap = self.config.proposal_max_per_video
        if cap > 0 and len(scored) > cap:
            kept = scored[:cap]
            margin = self.config.proposal_soft_k_margin
            if margin > 0:
                floor = margin * scored[cap - 1][0]
                kept.extend((peak, proposal)
                            for peak, proposal in scored[cap:]
                            if peak >= floor)
            scored = kept

        fraction = self.config.proposal_trim_fraction
        refined: List[CandidateInterval] = []
        for peak, proposal in scored:
            if fraction <= 0:
                refined.append(proposal)
                continue
            lo = max(0, proposal.t_on - t_start)
            hi = min(len(expr), proposal.t_off - t_start + 1)
            if hi <= lo:
                refined.append(proposal)
                continue
            mask = expr[lo:hi] >= peak * fraction
            runs: List[Tuple[int, int]] = []
            run_start = -1
            for i, hot in enumerate(mask):
                if hot and run_start < 0:
                    run_start = i
                elif not hot and run_start >= 0:
                    runs.append((run_start, i - 1))
                    run_start = -1
            if run_start >= 0:
                runs.append((run_start, len(mask) - 1))
            if not runs:
                # The contour is empty (degenerate peak shape): keep the span as the
                # hysteresis drew it rather than inventing an extent.
                refined.append(proposal)
                continue
            # Keep the hot run that contains the apex -- a double-burst event has two
            # lobes and the naive first-to-last trim bridges the trough between them,
            # which is exactly the over-wide extent the C2 attribution bucket shows.
            apex_rel = min(max(0, proposal.apex - t_start - lo), max(0, len(mask) - 1))
            chosen = None
            for run in runs:
                if run[0] <= apex_rel <= run[1]:
                    chosen = run
                    break
            if chosen is None:
                chosen = max(runs, key=lambda r: r[1] - r[0])
            core_lo, core_hi = chosen
            apex = lo + core_lo + int(np.argmax(expr[lo + core_lo: lo + core_hi + 1]))
            refined.append(replace(
                proposal,
                t_on=t_start + lo + core_lo,
                t_off=t_start + lo + core_hi,
                apex=t_start + apex,
            ))
        refined.sort(key=lambda proposal: proposal.t_on)
        for order, proposal in enumerate(refined):
            proposal.cid = f"p{order + 1:02d}"
        return refined

    def _raw_curve_spans(
        self,
        micro: Sequence[CandidateInterval],
        delta_expr: np.ndarray,
        t_start: int,
        fraction: float,
    ) -> List[CandidateInterval]:
        """Merge threshold spans of the RAW expressive residual into the pool.
        """
        expr = np.asarray(delta_expr, dtype=np.float64).reshape(-1)
        if expr.size == 0 or float(expr.max()) <= 1e-9:
            return list(micro)
        threshold = fraction * float(expr.max())
        spans: List[Tuple[int, int]] = []
        inside, start = False, 0
        for i, value in enumerate(expr):
            if not inside and value >= threshold:
                inside, start = True, i
            elif inside and value < threshold:
                spans.append((start, i - 1))
                inside = False
        if inside:
            spans.append((start, len(expr) - 1))
        merged = list(micro)
        for lo, hi in spans:
            segment = expr[lo:hi + 1]
            merged.append(CandidateInterval(
                cid="", t_on=t_start + lo, t_off=t_start + hi,
                apex=t_start + lo + int(np.argmax(segment)),
                peak_S=round(float(segment.max()), 4),
                attribution={}, physio_overlap=False, channel="micro",
                notes="raw-curve threshold span",
            ))
        return merged


def alignment_auc(s_curve: Sequence[float], intervals: Sequence[Tuple[int, int]],
                  t_start: int = 0) -> float:
    """Frame-level ROC-AUC of ``S_t`` against the annotated intervals.

    The threshold-free core metric of the rollout-quality layer (corollary B.4): it
    simultaneously measures how flat the baseline is on stationary stretches and how
    exposed the transitions are.  Computed via the rank-sum identity, so it needs no
    threshold sweep.
    """
    scores = np.asarray(s_curve, dtype=np.float64)
    if scores.size == 0 or not intervals:
        return 0.0
    labels = np.zeros(scores.size, dtype=bool)
    for onset, offset in intervals:
        lo = max(0, onset - t_start)
        hi = min(scores.size - 1, offset - t_start)
        if lo <= hi:
            labels[lo:hi + 1] = True

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # Average ranks within tied score groups, or ties bias the statistic.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return round(float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)), 4)


def calibrate_thresholds(
    s_curves: Sequence[Sequence[float]],
    interval_sets: Sequence[Sequence[Tuple[int, int]]],
    grid_hi: Sequence[float] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0),
    grid_lo_ratio: Sequence[float] = (0.3, 0.4, 0.5, 0.6),
) -> Dict[str, float]:
    """Pick ``(tau_hi, tau_lo)`` on a calibration fold by Youden's J (appendix F.1)."""
    best = {"tau_hi": 3.5, "tau_lo": 1.5, "youden": -1.0}
    for tau_hi in grid_hi:
        for ratio in grid_lo_ratio:
            tau_lo = tau_hi * ratio
            tpr_all, fpr_all = [], []
            for curve, intervals in zip(s_curves, interval_sets):
                scores = np.asarray(curve, dtype=np.float64)
                if scores.size == 0:
                    continue
                labels = np.zeros(scores.size, dtype=bool)
                for onset, offset in intervals:
                    lo, hi = max(0, onset), min(scores.size - 1, offset)
                    if lo <= hi:
                        labels[lo:hi + 1] = True
                predicted = scores >= tau_hi
                positives, negatives = int(labels.sum()), int((~labels).sum())
                if positives == 0 or negatives == 0:
                    continue
                tpr_all.append(float((predicted & labels).sum()) / positives)
                fpr_all.append(float((predicted & ~labels).sum()) / negatives)
            if not tpr_all:
                continue
            youden = float(np.mean(tpr_all) - np.mean(fpr_all))
            if youden > best["youden"]:
                best = {"tau_hi": float(tau_hi), "tau_lo": round(float(tau_lo), 3),
                        "youden": round(youden, 4)}
    return best


__all__ = [
    "PhysioTemplate", "default_physio_templates", "fit_physio_templates",
    "DecompositionResult", "ErrorDecomposer", "robust_normalise", "SpottingResult",
    "ProposalGenerator", "Spotter", "alignment_auc", "calibrate_thresholds",
]
