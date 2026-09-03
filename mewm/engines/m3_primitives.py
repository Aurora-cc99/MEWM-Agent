"""M3 -- the four rollout service primitives (paper 3.3.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import DynamicsConfig
from ..knowledge.au_anatomy import K_SLOTS, SLOT_AUS, SLOT_INDEX
from ..knowledge.emotion_prototypes import (
    DEFAULT_LIBRARY, FINE_EMOTIONS, PrototypeLibrary, TEMPLATE_LENGTH, normalise_scores,
)
from ..engines.m1_dynamics import AnalyticDynamics
from ..engines.v3_latent import BeliefState
from ..schemas import RolloutRecord

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------


def dtw_distance(a: np.ndarray, b: np.ndarray, band_ratio: float = 0.2) -> float:
    """Sakoe-Chiba-banded DTW between two activation sequences (appendix C.3).

    The band both bounds the cost at ``O(n * band)`` and rules out degenerate alignments
    that would warp a slow ramp onto a sharp transient -- a distinction this system
    depends on when separating a social smile from a micro-expression.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("inf")
    if a.shape[1] != b.shape[1]:
        width = min(a.shape[1], b.shape[1])
        a, b = a[:, :width], b[:, :width]

    n, m = a.shape[0], b.shape[0]
    band = max(1, int(round(band_ratio * max(n, m))))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            local = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    result = cost[n, m]
    # Normalise by path length so sequences of different duration stay comparable.
    return float(result / (n + m)) if np.isfinite(result) else float("inf")


def cosine_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """``1 - mean cosine similarity`` between two aligned latent trajectories."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    steps = min(a.shape[0], b.shape[0])
    if steps == 0:
        return 1.0
    similarities = []
    for i in range(steps):
        na, nb = np.linalg.norm(a[i]), np.linalg.norm(b[i])
        if na < 1e-9 or nb < 1e-9:
            continue
        similarities.append(float(a[i] @ b[i] / (na * nb)))
    return round(1.0 - float(np.mean(similarities)), 5) if similarities else 1.0


def resample_template(curves: Dict[str, Sequence[float]], length: int) -> np.ndarray:
    """Render a prototype template as a ``(length, K)`` activation matrix."""
    out = np.zeros((max(1, length), K_SLOTS), dtype=np.float64)
    source = np.linspace(0.0, 1.0, TEMPLATE_LENGTH)
    target = np.linspace(0.0, 1.0, max(1, length))
    for au, curve in curves.items():
        if au not in SLOT_INDEX:
            continue
        out[:, SLOT_INDEX[au]] = np.interp(target, source, np.asarray(curve, dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    trajectory: np.ndarray                       # (S, K) expected activations
    variance: np.ndarray                         # (S,) or (S, K)
    emotion: Optional[str] = None
    record: Optional[RolloutRecord] = None

    def digest(self) -> Dict[str, Any]:
        return {
            "steps": int(self.trajectory.shape[0]) if self.trajectory.size else 0,
            "emotion": self.emotion,
            "peak_aus": _top_aus(self.trajectory),
            "mean_variance": round(float(np.mean(self.variance)), 5) if self.variance.size else 0.0,
        }


@dataclass
class ScoreResult:
    log_likelihood: Dict[str, float]
    normalised: Dict[str, float]                  # DC(e), min-max within the candidate set
    record: Optional[RolloutRecord] = None

    def ranked(self) -> List[Tuple[str, float]]:
        return sorted(self.log_likelihood.items(), key=lambda kv: -kv[1])

    def likelihood_ratio(self, first: str, second: str) -> float:
        """``Lambda = l(e1) - l(e2)``."""
        return round(self.log_likelihood.get(first, 0.0) - self.log_likelihood.get(second, 0.0), 5)

    def digest(self) -> Dict[str, Any]:
        return {"ranked": [(e, round(v, 4)) for e, v in self.ranked()[:4]],
                "dc": {k: v for k, v in list(self.normalised.items())[:6]}}


@dataclass
class MaskResult:
    belief_full: BeliefState
    belief_masked: BeliefState
    masked_slots: List[str]
    mni: float                                    # D_KL(q(z^e|A) || q(z^e|A_-k))
    flipped: bool                                 # did the argmax hypothesis change?
    entropy_delta: float
    record: Optional[RolloutRecord] = None

    def digest(self) -> Dict[str, Any]:
        return {
            "masked": self.masked_slots, "MNI": round(self.mni, 5),
            "flipped": self.flipped, "entropy_delta": round(self.entropy_delta, 5),
            "full_top": self.belief_full.top(2), "masked_top": self.belief_masked.top(2),
        }


@dataclass
class CompareResult:
    cosine_divergence: float
    dtw_distance: float
    per_au_delta: Dict[str, float] = field(default_factory=dict)
    record: Optional[RolloutRecord] = None

    def digest(self) -> Dict[str, Any]:
        return {
            "cosine_divergence": round(self.cosine_divergence, 5),
            "dtw": round(self.dtw_distance, 5),
            "largest_gaps": sorted(self.per_au_delta.items(), key=lambda kv: -abs(kv[1]))[:4],
        }


def _top_aus(trajectory: np.ndarray, n: int = 4) -> List[Tuple[str, float]]:
    if trajectory.size == 0:
        return []
    peaks = np.atleast_2d(trajectory).max(axis=0)
    order = np.argsort(peaks)[::-1][:n]
    return [(SLOT_AUS[i], round(float(peaks[i]), 4)) for i in order
            if i < len(SLOT_AUS) and peaks[i] > 0.01]


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class RolloutService:
    """The four primitives, with provenance recording.

    Works with a trained :class:`~mewm.engines.m1_dynamics.AUDynamicsModel` or with the
    analytic stand-in; ``model_version`` always reflects which one answered.
    """

    def __init__(
        self,
        dynamics: Optional[Any] = None,
        config: Optional[DynamicsConfig] = None,
        library: Optional[PrototypeLibrary] = None,
        emotions: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or DynamicsConfig()
        self.dynamics = dynamics or AnalyticDynamics(self.config)
        self.library = library or DEFAULT_LIBRARY
        self.emotions = list(emotions or FINE_EMOTIONS)
        self.records: List[RolloutRecord] = []

    @property
    def model_version(self) -> str:
        getter = getattr(self.dynamics, "version_hash", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:  # noqa: BLE001
                pass
        return str(getattr(self.dynamics, "version", "m1-unknown"))

    def _record(self, caller: str, primitive: str, params: Dict[str, Any],
                digest: Dict[str, Any], cid: str) -> RolloutRecord:
        record = RolloutRecord.create(caller, primitive, params, digest,
                                      self.model_version, cid)
        self.records.append(record)
        return record

    # -- 1. rollout ---------------------------------------------------------

    def rollout(
        self,
        context: np.ndarray,
        emotion: Optional[str] = None,
        steps: Optional[int] = None,
        caller: str = "R",
        cid: str = "",
    ) -> RolloutResult:
        """Roll forward from ``context``; with ``emotion``, under that hypothesis."""
        steps = steps or self.config.rollout_steps
        context = np.atleast_2d(np.asarray(context, dtype=np.float64))
        last = context[-1] if context.shape[0] else np.zeros(K_SLOTS)
        momentum = (context[-1] - context[-2]) if context.shape[0] >= 2 else None

        out = self.dynamics.rollout(last, steps=steps, emotion=emotion, momentum=momentum)
        result = RolloutResult(
            trajectory=np.asarray(out["trajectory"], dtype=np.float64),
            variance=np.asarray(out.get("variance", np.zeros(steps)), dtype=np.float64),
            emotion=emotion,
        )
        result.record = self._record(
            caller, "rollout",
            {"steps": steps, "emotion": emotion, "context_len": int(context.shape[0])},
            result.digest(), cid,
        )
        return result

    def expected_trajectory(self, emotion: str, length: int) -> np.ndarray:
        """``tilde A^(e)`` -- the prototype-shaped trajectory the hypothesis predicts.

        Sourced from the prototype library rather than the transition model, so that the
        CFS comparison of eq. (9) has a reference that is statistically independent of
        the model producing the likelihoods (appendix C.3).
        """
        template = self.library.get(emotion, "full")
        return resample_template(template.curves, length)

    # -- 2. score -----------------------------------------------------------

    def score(
        self,
        trajectory: np.ndarray,
        emotions: Optional[Sequence[str]] = None,
        caller: str = "R",
        cid: str = "",
    ) -> ScoreResult:
        """Conditional log-likelihood of the observed trajectory per hypothesis."""
        trajectory = np.atleast_2d(np.asarray(trajectory, dtype=np.float64))
        candidates = list(emotions or self.emotions)
        likelihood = {
            emotion: round(float(self.dynamics.log_likelihood(trajectory, emotion)), 5)
            for emotion in candidates
        }
        result = ScoreResult(likelihood, normalise_scores(likelihood))
        result.record = self._record(
            caller, "score",
            {"emotions": candidates, "n_frames": int(trajectory.shape[0])},
            result.digest(), cid,
        )
        return result

    # -- 3. mask ------------------------------------------------------------

    def mask(
        self,
        trajectory: np.ndarray,
        masked_slots: Sequence[str],
        emotions: Optional[Sequence[str]] = None,
        caller: str = "C",
        cid: str = "",
    ) -> MaskResult:
        """Mask slots out of the observation set and re-infer the belief (eq. 10).

        Masked slots are marked *unobserved* rather than set to zero.  Zeroing would
        assert "this AU was measured and found inactive", which is a different and much
        stronger claim than "this AU was not measured" -- and it is the latter that slot
        dropout made an in-distribution input.
        """
        trajectory = np.atleast_2d(np.asarray(trajectory, dtype=np.float64))
        candidates = list(emotions or self.emotions)

        belief_full = self._infer_belief(trajectory, candidates)
        indices = [SLOT_INDEX[au] for au in masked_slots if au in SLOT_INDEX]
        observed = np.ones(K_SLOTS, dtype=bool)
        for index in indices:
            observed[index] = False
        belief_masked = self._infer_belief(trajectory, candidates, observed)

        mni = belief_full.kl_to(belief_masked)
        result = MaskResult(
            belief_full=belief_full,
            belief_masked=belief_masked,
            masked_slots=[au for au in masked_slots if au in SLOT_INDEX],
            mni=round(float(mni), 5),
            flipped=belief_full.mean_label != belief_masked.mean_label,
            entropy_delta=round(belief_masked.entropy - belief_full.entropy, 5),
        )
        result.record = self._record(
            caller, "mask",
            {"masked": result.masked_slots, "n_frames": int(trajectory.shape[0])},
            result.digest(), cid,
        )
        return result

    def _infer_belief(
        self, trajectory: np.ndarray, emotions: Sequence[str],
        observed: Optional[np.ndarray] = None,
    ) -> BeliefState:
        """``q(z^e | A)`` -- posterior from per-hypothesis conditional likelihood."""
        working = trajectory.copy()
        if observed is not None:
            # Unobserved slots are dropped from the evidence, not asserted to be zero.
            working = working[:, observed] if working.shape[1] == observed.size else working
            padded = np.zeros_like(trajectory)
            if working.shape[1] == int(observed.sum()):
                padded[:, observed] = working
                working = padded

        belief = BeliefState.uniform(list(emotions))
        scores = {
            emotion: float(self.dynamics.log_likelihood(working, emotion))
            for emotion in emotions
        }
        if observed is not None:
            # Re-weight by how much of each prototype survived the mask, so removing a
            # hypothesis's core evidence actually costs that hypothesis.
            from ..knowledge.emotion_prototypes import core_aus
            for emotion in emotions:
                core = [au for au in core_aus(emotion) if au in SLOT_INDEX]
                if not core:
                    continue
                kept = sum(1 for au in core if observed[SLOT_INDEX[au]])
                present = sum(
                    float(working[:, SLOT_INDEX[au]].max()) for au in core
                    if observed[SLOT_INDEX[au]]
                )
                coverage = kept / len(core)
                scores[emotion] = scores[emotion] + 2.0 * (present * coverage)
        else:
            from ..knowledge.emotion_prototypes import core_aus
            for emotion in emotions:
                core = [au for au in core_aus(emotion) if au in SLOT_INDEX]
                if not core:
                    continue
                present = sum(float(working[:, SLOT_INDEX[au]].max()) for au in core)
                scores[emotion] = scores[emotion] + 2.0 * present

        centre = float(np.mean(list(scores.values()))) if scores else 0.0
        belief.update({e: v - centre for e, v in scores.items()}, weight=1.0)
        return belief

    # -- 4. compare ---------------------------------------------------------

    def compare(
        self,
        first: np.ndarray,
        second: np.ndarray,
        caller: str = "C",
        cid: str = "",
    ) -> CompareResult:
        """Structured difference report between two trajectories."""
        first = np.atleast_2d(np.asarray(first, dtype=np.float64))
        second = np.atleast_2d(np.asarray(second, dtype=np.float64))
        per_au: Dict[str, float] = {}
        width = min(first.shape[1], second.shape[1], len(SLOT_AUS))
        for k in range(width):
            delta = float(first[:, k].max() - second[:, k].max())
            if abs(delta) > 0.01:
                per_au[SLOT_AUS[k]] = round(delta, 4)

        result = CompareResult(
            cosine_divergence=cosine_divergence(first, second),
            dtw_distance=round(dtw_distance(first, second), 5),
            per_au_delta=per_au,
        )
        result.record = self._record(
            caller, "compare",
            {"len_1": int(first.shape[0]), "len_2": int(second.shape[0])},
            result.digest(), cid,
        )
        return result

    # -- derived quantities the agents ask for ------------------------------

    def counterfactual_consistency(
        self,
        observed: np.ndarray,
        emotions: Sequence[str],
        caller: str = "C",
        cid: str = "",
    ) -> Dict[str, float]:
        """``CFS(e)`` of eq. (9): ``1 - DTW(A_obs, tilde A^(e)) / max_e' DTW``."""
        observed = np.atleast_2d(np.asarray(observed, dtype=np.float64))
        length = max(2, observed.shape[0])
        distances = {
            emotion: dtw_distance(observed, self.expected_trajectory(emotion, length))
            for emotion in emotions
        }
        finite = [d for d in distances.values() if np.isfinite(d)]
        worst = max(finite) if finite else 1.0
        if worst <= 1e-9:
            return {emotion: 1.0 for emotion in emotions}
        result = {
            emotion: round(1.0 - min(distance, worst) / worst, 4)
            for emotion, distance in distances.items()
        }
        self._record(caller, "compare",
                     {"primitive": "CFS", "emotions": list(emotions)},
                     {"cfs": result}, cid)
        return result

    def necessity_indices(
        self,
        observed: np.ndarray,
        critical_aus: Sequence[str],
        emotions: Optional[Sequence[str]] = None,
        caller: str = "C",
        cid: str = "",
    ) -> Dict[str, MaskResult]:
        """Leave-one-out ``MNI_k`` for each claimed critical AU."""
        return {
            au: self.mask(observed, [au], emotions, caller=caller, cid=cid)
            for au in critical_aus
        }

    def template_comparison(
        self,
        observed: np.ndarray,
        emotion: str,
        caller: str = "C",
        cid: str = "",
    ) -> Dict[str, float]:
        """Distances to the full / neutralised / masked templates (appendix C.4).

        The third-party reference: likelihood and rollout both come from the transition
        model, so a systematic model bias shifts them together.  These templates are
        counted from data, so agreement across all three is strong evidence and
        disagreement localises the fault.
        """
        observed = np.atleast_2d(np.asarray(observed, dtype=np.float64))
        length = max(2, observed.shape[0])
        out: Dict[str, float] = {}
        for variant in ("full", "neutralised", "masked"):
            try:
                template = self.library.get(emotion, variant)
            except KeyError:
                continue
            reference = resample_template(template.curves, length)
            out[variant] = round(dtw_distance(observed, reference), 5)
        self._record(caller, "compare",
                     {"primitive": "template", "emotion": emotion}, out, cid)
        return out

    # -- provenance ---------------------------------------------------------

    def records_for(self, cid: str) -> List[RolloutRecord]:
        return [r for r in self.records if r.cid == cid]

    def stats(self) -> Dict[str, Any]:
        by_primitive: Dict[str, int] = {}
        for record in self.records:
            by_primitive[record.primitive] = by_primitive.get(record.primitive, 0) + 1
        return {"model_version": self.model_version, "calls": len(self.records),
                "by_primitive": by_primitive}


__all__ = [
    "dtw_distance", "cosine_divergence", "resample_template", "RolloutResult",
    "ScoreResult", "MaskResult", "CompareResult", "RolloutService",
]
