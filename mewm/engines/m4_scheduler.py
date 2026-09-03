"""M4 -- the scheduling signal (paper 3.3.4, appendix F.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..config import SchedulerConfig

LOGGER = logging.getLogger(__name__)

PATH_FAST = "fast"
PATH_STANDARD = "standard"
PATH_DEEP = "deep"
PATHS = (PATH_FAST, PATH_STANDARD, PATH_DEEP)


@dataclass
class ScheduleSignal:
    """Per-proposal routing inputs and the decision they produce."""

    cid: str
    evidence_margin: float = 0.0        # Delta: ES gap between the top two hypotheses
    likelihood_ratio: float = 0.0       # Lambda: l(e1) - l(e2)
    belief_variance: float = 1.0        # Var[z^e], normalised entropy in [0, 1]
    detection_margin: float = 0.0       # S_apex - tau_hi
    predictive_variance: float = 0.0    # transition-model uncertainty
    open_questions: int = 0
    confidence: Optional[float] = None  # available only after adjudication
    suppression: str = "none"
    challenge_upheld: bool = False
    path: str = PATH_STANDARD
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid, "path": self.path,
            "evidence_margin": round(self.evidence_margin, 4),
            "likelihood_ratio": round(self.likelihood_ratio, 4),
            "belief_variance": round(self.belief_variance, 4),
            "detection_margin": round(self.detection_margin, 4),
            "open_questions": self.open_questions,
            "confidence": (round(self.confidence, 4) if self.confidence is not None else None),
            "suppression": self.suppression,
            "reasons": list(self.reasons),
        }


class Scheduler:
    """Deterministic path selection.  No LLM is involved in this decision."""

    def __init__(self, config: Optional[SchedulerConfig] = None) -> None:
        self.config = config or SchedulerConfig()
        self.history: List[ScheduleSignal] = []

    def route(self, signal: ScheduleSignal) -> ScheduleSignal:
        """Assign ``signal.path`` from the F.4 conditions; deep wins over fast."""
        reasons: List[str] = []

        deep = False
        if signal.confidence is not None and signal.confidence < self.config.theta_deep:
            deep, _ = True, reasons.append(
                f"confidence {signal.confidence:.3f} < theta_deep {self.config.theta_deep}")
        if signal.suppression in {"neutralised", "masked"}:
            deep, _ = True, reasons.append(f"{signal.suppression} pattern detected")
        if signal.challenge_upheld:
            deep, _ = True, reasons.append("a challenge was upheld")
        if signal.belief_variance > self.config.theta_var:
            deep, _ = True, reasons.append(
                f"belief variance {signal.belief_variance:.3f} > theta_var {self.config.theta_var}")

        if deep:
            signal.path, signal.reasons = PATH_DEEP, reasons
            self.history.append(signal)
            return signal

        fast = (
            signal.evidence_margin >= self.config.theta_fast
            and signal.likelihood_ratio >= self.config.eta_fast
            and signal.open_questions == 0
        )
        if fast:
            signal.path = PATH_FAST
            signal.reasons = [
                f"evidence margin {signal.evidence_margin:.3f} >= {self.config.theta_fast}",
                f"likelihood ratio {signal.likelihood_ratio:.3f} >= {self.config.eta_fast}",
                "no open questions",
            ]
        else:
            signal.path = PATH_STANDARD
            if signal.evidence_margin < self.config.theta_fast:
                reasons.append(f"evidence margin {signal.evidence_margin:.3f} below fast threshold")
            if signal.likelihood_ratio < self.config.eta_fast:
                reasons.append(f"likelihood ratio {signal.likelihood_ratio:.3f} below fast threshold")
            if signal.open_questions:
                reasons.append(f"{signal.open_questions} open question(s)")
            signal.reasons = reasons

        self.history.append(signal)
        return signal

    def reroute_after_adjudication(
        self, signal: ScheduleSignal, confidence: float,
        suppression: str = "none", challenge_upheld: bool = False,
    ) -> ScheduleSignal:
        """Re-evaluate once the verdict exists; may escalate a proposal to deep."""
        signal.confidence = confidence
        signal.suppression = suppression
        signal.challenge_upheld = challenge_upheld
        previous = signal.path
        rerouted = self.route(signal)
        if previous != rerouted.path:
            LOGGER.info("proposal %s rerouted %s -> %s", signal.cid, previous, rerouted.path)
        return rerouted

    def distribution(self) -> Dict[str, int]:
        counts = {path: 0 for path in PATHS}
        for signal in self.history:
            counts[signal.path] = counts.get(signal.path, 0) + 1
        return counts

    def stats(self) -> Dict[str, Any]:
        return {"routed": len(self.history), "distribution": self.distribution()}


def build_signal(
    cid: str,
    es_scores: Optional[Dict[str, float]] = None,
    likelihood: Optional[Dict[str, float]] = None,
    belief_variance: float = 1.0,
    detection_margin: float = 0.0,
    predictive_variance: float = 0.0,
    open_questions: int = 0,
) -> ScheduleSignal:
    """Assemble a routing signal from the engine and agent outputs."""
    margin = 0.0
    if es_scores and len(es_scores) >= 2:
        ordered = sorted(es_scores.values(), reverse=True)
        margin = float(ordered[0] - ordered[1])
    elif es_scores:
        margin = float(next(iter(es_scores.values())))

    ratio = 0.0
    if likelihood and len(likelihood) >= 2:
        ordered = sorted(likelihood.values(), reverse=True)
        ratio = float(ordered[0] - ordered[1])

    return ScheduleSignal(
        cid=cid, evidence_margin=margin, likelihood_ratio=ratio,
        belief_variance=belief_variance, detection_margin=detection_margin,
        predictive_variance=predictive_variance, open_questions=open_questions,
    )


__all__ = [
    "PATH_FAST", "PATH_STANDARD", "PATH_DEEP", "PATHS", "ScheduleSignal", "Scheduler",
    "build_signal",
]
