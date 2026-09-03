"""What counts as a *passing* sample.

* **format** -- the output parses as the declared contract and, where it carries a
  composed answer, satisfies :mod:`mewm.eval.answer_format`;
* **temporal** -- the proposal clears the eq. (2) criterion at the *configured*
  ``iou_threshold``, including the affective-rescue clause when it is enabled;
* **semantic** -- the fine label matches, with the coarse label consistent with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import EvaluationConfig
from ..knowledge.emotion_prototypes import canonical_fine_label, coarse_of, labels_consistent
from .answer_format import validate_answer
from .metrics import iou as interval_iou, tp_decision


@dataclass
class PassOutcome:
    """Per-sample verdict, with the reason it failed when it did."""

    passed: bool = False
    format_ok: bool = False
    temporal_ok: bool = False
    label_ok: bool = False
    iou: float = 0.0
    rescued: bool = False
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed, "format_ok": self.format_ok,
            "temporal_ok": self.temporal_ok, "label_ok": self.label_ok,
            "iou": round(self.iou, 4), "rescued": self.rescued,
            "reasons": list(self.reasons),
        }


def _as_interval(value: Any) -> Optional[Tuple[int, int]]:
    """Coerce a proposal to ``(onset, offset)``; ``None`` when it is not one."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "onset" not in value or "offset" not in value:
            return None
        value = (value["onset"], value["offset"])
    try:
        pair = tuple(int(round(float(v))) for v in tuple(value)[:2])
    except (TypeError, ValueError):
        return None
    if len(pair) != 2:
        return None
    lo, hi = pair
    return (lo, hi) if hi >= lo else (hi, lo)


def check_format(product: Dict[str, Any], required_fields: Sequence[str] = ()) -> Tuple[bool, List[str]]:
    """Contract fields present, and any composed answer clean.

    The two halves are distinct failures: a model can emit perfectly valid JSON whose
    narrative still cites the paper, and it can write clean prose inside a malformed
    object. Both are format failures here.
    """
    reasons: List[str] = []
    if not isinstance(product, dict) or not product:
        return False, ["output did not parse as a JSON object"]

    for name in required_fields:
        value = product.get(name)
        if value is None or (isinstance(value, (str, list, dict)) and not value):
            reasons.append(f"missing field: {name}")

    answer = product.get("answer") or product.get("global_narrative") or ""
    if isinstance(answer, str) and answer.strip():
        report = validate_answer(answer, require_sections=False)
        reasons.extend(report.problems())

    return (not reasons), reasons


def check_temporal(
    proposal: Any, truth: Any, fine_pred: str = "", fine_true: str = "",
    evaluation: Optional[EvaluationConfig] = None,
) -> Tuple[bool, float, bool, List[str]]:
    """The eq. (2) criterion at the configured threshold.

    Returns ``(ok, iou, rescued, reasons)``. The verdict itself comes from
    :func:`~mewm.eval.metrics.tp_decision`, which the training reward and the final
    evaluation also call -- this function's own job is only to turn the two inputs into
    an IoU and the verdict into this module's tuple.
    """
    evaluation = evaluation or EvaluationConfig()
    predicted = _as_interval(proposal)
    reference = _as_interval(truth)
    if predicted is None:
        return False, 0.0, False, ["no interval proposed"]
    if reference is None:
        return False, 0.0, False, ["no reference interval to score against"]

    overlap = float(interval_iou(predicted, reference))
    verdict = tp_decision(overlap, fine_pred, fine_true,
                          evaluation.iou_threshold, evaluation.affective_rescue,
                          evaluation.rescue_min_iou)
    return verdict.is_tp, verdict.iou, verdict.rescued, list(verdict.reasons)


def check_label(fine_pred: str, fine_true: str,
                coarse_pred: str = "") -> Tuple[bool, List[str]]:
    """Fine label exact after canonicalisation, coarse label consistent with it.

    Canonicalisation first, because an out-of-vocabulary string is a different failure
    from a wrong-but-real emotion and the two should not be scored alike.
    """
    reasons: List[str] = []
    predicted, recognised = canonical_fine_label(fine_pred)
    reference, _ = canonical_fine_label(fine_true)

    if not recognised:
        reasons.append(f"fine label {fine_pred!r} is outside the label set")
    elif predicted != reference:
        reasons.append(f"fine label {predicted!r} != {reference!r}")

    if coarse_pred:
        if not labels_consistent(predicted, coarse_pred):
            reasons.append(
                f"coarse label {coarse_pred!r} inconsistent with fine {predicted!r} "
                f"(expected {coarse_of(predicted)!r})"
            )
    return (not reasons), reasons


def evaluate_sample(
    product: Dict[str, Any],
    truth: Dict[str, Any],
    evaluation: Optional[EvaluationConfig] = None,
    required_fields: Sequence[str] = (),
) -> PassOutcome:
    """Full verdict for one sampled output."""
    outcome = PassOutcome()

    outcome.format_ok, format_reasons = check_format(product, required_fields)
    outcome.reasons.extend(format_reasons)

    fine_pred = str(product.get("fine_label", ""))
    fine_true = str(truth.get("fine", ""))

    outcome.temporal_ok, outcome.iou, outcome.rescued, temporal_reasons = check_temporal(
        product.get("interval"), truth.get("interval"), fine_pred, fine_true, evaluation,
    )
    outcome.reasons.extend(temporal_reasons)

    outcome.label_ok, label_reasons = check_label(
        fine_pred, fine_true, str(product.get("coarse_label", "")))
    outcome.reasons.extend(label_reasons)

    outcome.passed = outcome.format_ok and outcome.temporal_ok and outcome.label_ok
    return outcome


def evaluate_video_sample(
    product: Dict[str, Any],
    truth: Dict[str, Any],
    evaluation: Optional[EvaluationConfig] = None,
    required_fields: Sequence[str] = (),
) -> PassOutcome:
    """Verdict for a *whole-video* answer rather than a single proposal.
    """
    evaluation = evaluation or EvaluationConfig()
    outcome = PassOutcome()

    outcome.format_ok, format_reasons = check_format(product, required_fields)
    outcome.reasons.extend(format_reasons)

    reference = [e for e in (truth.get("events") or []) if e.get("interval")]
    claimed = list(product.get("events") or [])

    declared = product.get("n_micro")
    count = int(declared) if isinstance(declared, (int, float)) else len(claimed)
    if count != len(reference):
        outcome.reasons.append(
            f"claims {count} micro-expression event(s), annotation has {len(reference)}")

    if not reference and not claimed:
        # Nothing to localise and nothing claimed: the count is the whole criterion.
        outcome.temporal_ok = count == 0
        outcome.label_ok = count == 0
        outcome.passed = outcome.format_ok and outcome.temporal_ok
        return outcome

    unmatched = list(range(len(reference)))
    ious: List[float] = []
    rescued_any = False
    matched = 0

    for index, claim in enumerate(claimed):
        span = _as_interval(claim.get("interval") if isinstance(claim, dict) else claim)
        if span is None:
            outcome.reasons.append(f"claimed event {index + 1} has no usable interval")
            continue
        fine_pred = str(claim.get("fine_label", "") if isinstance(claim, dict) else "")

        best_j, best_iou = None, -1.0
        for j in unmatched:
            overlap = float(interval_iou(span, tuple(reference[j]["interval"])))
            if overlap > best_iou:
                best_j, best_iou = j, overlap
        if best_j is None:
            outcome.reasons.append(
                f"claimed event {index + 1} has no annotated event left to match")
            continue

        verdict = tp_decision(
            best_iou, fine_pred, str(reference[best_j].get("fine", "")),
            evaluation.iou_threshold, evaluation.affective_rescue,
            evaluation.rescue_min_iou)
        ious.append(verdict.iou)
        if verdict.is_tp:
            matched += 1
            rescued_any = rescued_any or verdict.rescued
            unmatched.remove(best_j)
        else:
            outcome.reasons.append(
                f"claimed event {index + 1} at {span} is not a true positive "
                f"(IoU {verdict.iou:.3f})")

    outcome.iou = sum(ious) / len(ious) if ious else 0.0
    outcome.rescued = rescued_any
    outcome.temporal_ok = (matched == len(reference) == count)
    if not outcome.temporal_ok and not any("true positive" in r for r in outcome.reasons):
        outcome.reasons.append(
            f"{matched}/{len(reference)} annotated event(s) were localised")

    label_reasons: List[str] = []
    for index, claim in enumerate(claimed):
        if not isinstance(claim, dict):
            continue
        coarse = str(claim.get("coarse_label", ""))
        fine = str(claim.get("fine_label", ""))
        if fine or coarse:
            ok, reasons = check_label(fine, fine, coarse)
            if not ok:
                # Self-consistency only: whether the fine label is *correct* is already
                # decided by the rescue clause above, so re-scoring it here would count
                # the same error twice.
                label_reasons.extend(f"claimed event {index + 1}: {r}" for r in reasons
                                     if "inconsistent" in r or "outside the label set" in r)
    outcome.label_ok = not label_reasons
    outcome.reasons.extend(label_reasons)

    outcome.passed = outcome.format_ok and outcome.temporal_ok and outcome.label_ok
    return outcome


def stage_rates(outcomes: Sequence[PassOutcome]) -> Dict[str, float]:
    """Per-stage pass rates, so the bottleneck is visible rather than aggregated away."""
    total = len(outcomes)
    if not total:
        return {"n": 0, "format": 0.0, "temporal": 0.0, "label": 0.0, "joint": 0.0}
    return {
        "n": total,
        "format": round(sum(o.format_ok for o in outcomes) / total, 4),
        "temporal": round(sum(o.temporal_ok for o in outcomes) / total, 4),
        "label": round(sum(o.label_ok for o in outcomes) / total, 4),
        "joint": round(sum(o.passed for o in outcomes) / total, 4),
        "rescued": round(sum(o.rescued for o in outcomes) / total, 4),
    }


__all__ = [
    "PassOutcome", "check_format", "check_temporal", "check_label",
    "evaluate_sample", "evaluate_video_sample", "stage_rates",
]
