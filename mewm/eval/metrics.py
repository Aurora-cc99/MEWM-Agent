"""Evaluation metrics for the four protocols of paper 4.1-4.2.

**``theta`` is a hyper-parameter, not the constant 0.5.** The paper fixes it at 0.5 and
the ME-LVQA baselines report there, so that is the default -- but the entire
precision/recall trade-off pivots on it, and a framework that hard-codes it cannot run
the sensitivity sweep its own protocol asks for. It is threaded through every function
here and driven from ``EvaluationConfig.iou_threshold``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..knowledge.emotion_prototypes import canonical_fine_label

Interval = Tuple[int, int]

#: Paper default; override through ``EvaluationConfig.iou_threshold``.
DEFAULT_IOU_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# The eq. (2) criterion -- one implementation, three callers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TPDecision:
    """The verdict of eq. (2) on one proposal, with why it came out that way."""

    is_tp: bool
    rescued: bool
    iou: float
    #: Empty when the proposal passed; otherwise a human-readable failure reason.
    reasons: Tuple[str, ...] = ()

    @property
    def is_tp_strict(self) -> bool:
        """True only on the IoU clause -- the rescue-free figure."""
        return self.is_tp and not self.rescued


def tp_decision(
    overlap: float,
    fine_pred: str = "",
    fine_true: str = "",
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    affective_rescue: bool = True,
    rescue_min_iou: float = 0.0,
) -> TPDecision:
    """The single implementation of eq. (2). Everything that judges a proposal calls it.

    *Scoring* is deliberately not here. This answers "does it count"; the reward's
    partial-credit curve is a separate question layered on top of the answer.
    """
    overlap = float(overlap)
    if overlap > iou_threshold:
        return TPDecision(True, False, overlap)

    if not affective_rescue:
        return TPDecision(False, False, overlap,
                          (f"IoU {overlap:.3f} below threshold {iou_threshold:.2f} "
                           f"and affective rescue is disabled",))
    if overlap < rescue_min_iou:
        # Without this floor a proposal that merely grazes the event -- or, at the 0.0
        # default, misses it outright -- passes on its label alone.
        return TPDecision(False, False, overlap,
                          (f"IoU {overlap:.3f} below the rescue floor "
                           f"{rescue_min_iou:.2f}",))

    # Canonicalise before comparing: "happiness" and "HAPPINESS " are the same emotion,
    # and whether a run credits them alike must not depend on which caller is asking.
    predicted, pred_known = canonical_fine_label(fine_pred)
    reference, true_known = canonical_fine_label(fine_true)
    if pred_known and true_known:
        matched = predicted == reference
    else:
        # Outside the canonical vocabulary every unknown string collapses to "other",
        # so comparing canonical forms there would rescue "banana" against "unicorn".
        # Fall back to exact equality of what was actually written -- which is also what
        # lets a dataset-specific label match itself.
        written = fine_pred.strip().casefold()
        matched = bool(written) and written == fine_true.strip().casefold()
    if matched:
        return TPDecision(True, True, overlap)
    return TPDecision(False, False, overlap,
                      (f"IoU {overlap:.3f} below threshold {iou_threshold:.2f} and "
                       f"label {fine_pred!r} does not match {fine_true!r}",))


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------


def iou(a: Interval, b: Interval) -> float:
    """Temporal intersection over union of two inclusive frame intervals."""
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    intersection = max(0, hi - lo + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Match:
    """One proposal paired with the ground-truth event it claimed."""

    proposal_index: int
    truth_index: Optional[int]
    iou: float
    predicted_label: str = ""
    true_label: str = ""
    rescued: bool = False           # counted only via the affective-rescue clause
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    affective_rescue: bool = True
    rescue_min_iou: float = 0.0

    @property
    def is_tp_strict(self) -> bool:
        return self.truth_index is not None and self.iou > self.iou_threshold

    @property
    def decision(self) -> TPDecision:
        """This match under eq. (2). An unmatched proposal is a FP by construction."""
        if self.truth_index is None:
            return TPDecision(False, False, self.iou,
                              ("no ground-truth event was claimed",))
        return tp_decision(self.iou, self.predicted_label, self.true_label,
                           self.iou_threshold, self.affective_rescue,
                           self.rescue_min_iou)

    @property
    def is_tp(self) -> bool:
        return self.decision.is_tp


def greedy_match(
    proposals: Sequence[Interval],
    truths: Sequence[Interval],
    predicted_labels: Sequence[str] = (),
    true_labels: Sequence[str] = (),
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    affective_rescue: bool = True,
    rescue_min_iou: float = 0.0,
) -> List[Match]:
    """Greedy IoU matching; each ground-truth event is claimed at most once."""
    pairs = sorted(
        (
            (iou(p, g), pi, gi)
            for pi, p in enumerate(proposals)
            for gi, g in enumerate(truths)
        ),
        key=lambda item: -item[0],
    )
    taken_truth: set[int] = set()
    taken_proposal: set[int] = set()
    assigned: Dict[int, Tuple[int, float]] = {}

    for score, pi, gi in pairs:
        if score <= 0.0 or pi in taken_proposal or gi in taken_truth:
            continue
        assigned[pi] = (gi, score)
        taken_proposal.add(pi)
        taken_truth.add(gi)

    matches: List[Match] = []
    for pi in range(len(proposals)):
        gi, score = assigned.get(pi, (None, 0.0))
        predicted = predicted_labels[pi] if pi < len(predicted_labels) else ""
        truth = true_labels[gi] if (gi is not None and gi < len(true_labels)) else ""
        match = Match(pi, gi, round(score, 4), predicted, truth,
                      iou_threshold=iou_threshold, affective_rescue=affective_rescue,
                      rescue_min_iou=rescue_min_iou)
        match.rescued = bool(match.is_tp and not match.is_tp_strict)
        matches.append(match)
    return matches


# ---------------------------------------------------------------------------
# P1 -- proposal-level localisation + analysis
# ---------------------------------------------------------------------------


@dataclass
class ProposalMetrics:
    """Precision / recall / F1 under both criteria, plus boundary error."""

    n_proposals: int = 0
    n_truths: int = 0
    tp: int = 0
    tp_strict: int = 0
    n_rescued: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    precision_strict: float = 0.0
    recall_strict: float = 0.0
    f1_strict: float = 0.0
    onset_error_median: float = 0.0
    offset_error_median: float = 0.0
    onset_error_iqr: float = 0.0
    offset_error_iqr: float = 0.0
    mean_iou: float = 0.0
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    affective_rescue: bool = True

    @property
    def rescue_gain(self) -> float:
        """F1 attributable to the affective-rescue clause alone."""
        return round(self.f1 - self.f1_strict, 4)

    def to_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()}
        data["rescue_gain"] = self.rescue_gain
        return data


def evaluate_proposals(
    proposals: Sequence[Interval],
    truths: Sequence[Interval],
    predicted_labels: Sequence[str] = (),
    true_labels: Sequence[str] = (),
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    affective_rescue: bool = True,
    rescue_min_iou: float = 0.0,
) -> ProposalMetrics:
    """P1: proposal-level metrics under both the eq. (2) and the strict-IoU criteria."""
    matches = greedy_match(proposals, truths, predicted_labels, true_labels,
                           iou_threshold, affective_rescue, rescue_min_iou)
    metrics = ProposalMetrics(n_proposals=len(proposals), n_truths=len(truths),
                              iou_threshold=iou_threshold,
                              affective_rescue=affective_rescue)
    if not proposals and not truths:
        return metrics

    metrics.tp = sum(1 for m in matches if m.is_tp)
    metrics.tp_strict = sum(1 for m in matches if m.is_tp_strict)
    metrics.n_rescued = sum(1 for m in matches if m.rescued)

    metrics.precision, metrics.recall, metrics.f1 = _prf(
        metrics.tp, len(proposals), len(truths))
    metrics.precision_strict, metrics.recall_strict, metrics.f1_strict = _prf(
        metrics.tp_strict, len(proposals), len(truths))

    onset_errors, offset_errors, ious = [], [], []
    for match in matches:
        if match.truth_index is None:
            continue
        proposal = proposals[match.proposal_index]
        truth = truths[match.truth_index]
        onset_errors.append(abs(proposal[0] - truth[0]))
        offset_errors.append(abs(proposal[1] - truth[1]))
        ious.append(match.iou)

    if onset_errors:
        metrics.onset_error_median = float(np.median(onset_errors))
        metrics.onset_error_iqr = float(np.subtract(*np.percentile(onset_errors, [75, 25])))
    if offset_errors:
        metrics.offset_error_median = float(np.median(offset_errors))
        metrics.offset_error_iqr = float(np.subtract(*np.percentile(offset_errors, [75, 25])))
    if ious:
        metrics.mean_iou = round(float(np.mean(ious)), 4)
    return metrics


def _prf(tp: int, n_pred: int, n_true: int) -> Tuple[float, float, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def aggregate_proposal_metrics(per_video: Sequence[ProposalMetrics]) -> ProposalMetrics:
    """Micro-average over videos: pool the counts, then compute the rates once.

    Micro rather than macro because a per-video average would give a 40-frame clip with
    one event the same weight as a 9000-frame video with eight.
    """
    total = ProposalMetrics()
    if per_video:
        total.iou_threshold = per_video[0].iou_threshold
        total.affective_rescue = per_video[0].affective_rescue
    for metrics in per_video:
        total.n_proposals += metrics.n_proposals
        total.n_truths += metrics.n_truths
        total.tp += metrics.tp
        total.tp_strict += metrics.tp_strict
        total.n_rescued += metrics.n_rescued
    total.precision, total.recall, total.f1 = _prf(
        total.tp, total.n_proposals, total.n_truths)
    total.precision_strict, total.recall_strict, total.f1_strict = _prf(
        total.tp_strict, total.n_proposals, total.n_truths)
    onsets = [m.onset_error_median for m in per_video if m.onset_error_median]
    offsets = [m.offset_error_median for m in per_video if m.offset_error_median]
    ious = [m.mean_iou for m in per_video if m.mean_iou]
    total.onset_error_median = round(float(np.median(onsets)), 3) if onsets else 0.0
    total.offset_error_median = round(float(np.median(offsets)), 3) if offsets else 0.0
    total.mean_iou = round(float(np.mean(ious)), 4) if ious else 0.0
    return total


def sweep_iou_threshold(
    per_video_inputs: Sequence[Tuple[Sequence[Interval], Sequence[Interval],
                                     Sequence[str], Sequence[str]]],
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
    affective_rescue: bool = True,
) -> List[Dict[str, Any]]:
    """Recompute P1 across a grid of IoU thresholds (paper 4.6(2)).

    Reports both criteria at every point, so the precision/recall Pareto front and the
    rescue clause's contribution can be read off together rather than confounded.
    """
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        per_video = [
            evaluate_proposals(proposals, truths, predicted, true,
                               iou_threshold=threshold,
                               affective_rescue=affective_rescue)
            for proposals, truths, predicted, true in per_video_inputs
        ]
        aggregate = aggregate_proposal_metrics(per_video)
        rows.append({
            "iou_threshold": threshold,
            "precision": aggregate.precision, "recall": aggregate.recall,
            "f1": aggregate.f1,
            "precision_strict": aggregate.precision_strict,
            "recall_strict": aggregate.recall_strict, "f1_strict": aggregate.f1_strict,
            "rescue_gain": aggregate.rescue_gain, "n_rescued": aggregate.n_rescued,
            "tp": aggregate.tp, "n_proposals": aggregate.n_proposals,
            "n_truths": aggregate.n_truths,
        })
    return rows


# ---------------------------------------------------------------------------
# P2 -- interval understanding: recognition and AU detection
# ---------------------------------------------------------------------------


def unweighted_f1(y_true: Sequence[str], y_pred: Sequence[str],
                  labels: Optional[Sequence[str]] = None) -> Tuple[float, Dict[str, float]]:
    """UF1: macro F1 over classes, unweighted by support.

    Unweighted because micro-expression datasets are heavily imbalanced; a
    support-weighted score would mostly report performance on the majority class.
    """
    labels = list(labels or sorted(set(y_true) | set(y_pred)))
    per_class: Dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        denominator = 2 * tp + fp + fn
        per_class[label] = round(2 * tp / denominator, 4) if denominator else 0.0
    macro = round(float(np.mean(list(per_class.values()))), 4) if per_class else 0.0
    return macro, per_class


def unweighted_average_recall(y_true: Sequence[str], y_pred: Sequence[str],
                              labels: Optional[Sequence[str]] = None) -> float:
    """UAR: mean per-class recall."""
    labels = list(labels or sorted(set(y_true)))
    recalls = []
    for label in labels:
        support = sum(1 for t in y_true if t == label)
        if not support:
            continue
        hits = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        recalls.append(hits / support)
    return round(float(np.mean(recalls)), 4) if recalls else 0.0


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true), 4)


def au_set_metrics(predicted: Sequence[Sequence[str]],
                   truth: Sequence[Sequence[str]]) -> Dict[str, float]:
    """AU set F1 and Jaccard, averaged over samples."""
    f1_scores, jaccards = [], []
    for pred, true in zip(predicted, truth):
        p, t = set(pred), set(true)
        if not p and not t:
            f1_scores.append(1.0)
            jaccards.append(1.0)
            continue
        intersection = len(p & t)
        f1_scores.append(2 * intersection / (len(p) + len(t)) if (p or t) else 0.0)
        jaccards.append(intersection / len(p | t) if (p | t) else 0.0)
    return {
        "au_f1": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "au_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0.0,
    }


# ---------------------------------------------------------------------------
# P3 -- ME-LVQA official metrics
# ---------------------------------------------------------------------------


def count_errors(predicted: Sequence[int], truth: Sequence[int]) -> Dict[str, Any]:
    """Event-count MAE and RMSE.
    """
    if not predicted or not truth:
        return {"status": "unavailable",
                "reason": "no (predicted, truth) count pair was supplied"}
    errors = np.array([p - t for p, t in zip(predicted, truth)], dtype=np.float64)
    return {
        "status": "ok",
        "mae": round(float(np.abs(errors).mean()), 4),
        "rmse": round(float(np.sqrt((errors ** 2).mean())), 4),
    }


def _ngrams(tokens: Sequence[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Sentence BLEU with the standard brevity penalty."""
    cand = candidate.split()
    ref = reference.split()
    if not cand or not ref:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_grams = _ngrams(cand, n)
        ref_grams = _ngrams(ref, n)
        if not cand_grams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_grams.get(gram, 0))
                      for gram, count in cand_grams.items())
        precisions.append(overlap / sum(cand_grams.values()))
    if min(precisions) <= 0:
        # Smooth rather than collapse to zero: a single missing 4-gram should not erase
        # a otherwise-good short answer.
        precisions = [max(p, 1e-9) for p in precisions]
    geometric = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / max(1, len(cand)))
    return round(brevity * geometric, 4)


def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    """ROUGE-N recall."""
    cand_grams = _ngrams(candidate.split(), n)
    ref_grams = _ngrams(reference.split(), n)
    if not ref_grams:
        return 0.0
    overlap = sum(min(count, cand_grams.get(gram, 0)) for gram, count in ref_grams.items())
    return round(overlap / sum(ref_grams.values()), 4)


def rouge_l(candidate: str, reference: str, beta: float = 1.2) -> float:
    """ROUGE-L F-measure over the longest common subsequence."""
    cand, ref = candidate.split(), reference.split()
    if not cand or not ref:
        return 0.0
    table = [[0] * (len(ref) + 1) for _ in range(len(cand) + 1)]
    for i, a in enumerate(cand, start=1):
        for j, b in enumerate(ref, start=1):
            table[i][j] = table[i - 1][j - 1] + 1 if a == b else max(table[i - 1][j], table[i][j - 1])
    lcs = table[len(cand)][len(ref)]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    denominator = recall + beta ** 2 * precision
    return round((1 + beta ** 2) * precision * recall / denominator, 4) if denominator else 0.0


# ---------------------------------------------------------------------------
# Calibration and causal reliability
# ---------------------------------------------------------------------------


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    """ECE with equal-width bins."""
    if not confidences:
        return 0.0
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return round(float(total), 4)


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    if not confidences:
        return 0.0
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    return round(float(((confidences - correct) ** 2).mean()), 4)


@dataclass
class CausalReliability:
    """The causal-reliability block of the second evaluation layer."""

    challenge_pass_rate: float = 0.0      # rho_pass
    hallucination_rate: float = 0.0       # rho_hall: claimed AUs with no evidence
    mean_mni: float = 0.0
    flip_rate: float = 0.0                # rho_flip
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def causal_reliability(
    challenge_finals: Sequence[str],
    claimed_aus: Sequence[Sequence[str]],
    evidenced_aus: Sequence[Sequence[str]],
    mni_values: Sequence[Dict[str, float]],
    flips: Sequence[Dict[str, bool]],
) -> CausalReliability:
    """Aggregate the four causal-reliability indicators."""
    metrics = CausalReliability(n_samples=len(claimed_aus))
    if challenge_finals:
        metrics.challenge_pass_rate = round(
            sum(1 for f in challenge_finals if f == "rejected") / len(challenge_finals), 4)

    hallucinated, claimed_total = 0, 0
    for claimed, evidenced in zip(claimed_aus, evidenced_aus):
        evidence_set = set(evidenced)
        claimed_total += len(claimed)
        hallucinated += sum(1 for au in claimed if au not in evidence_set)
    metrics.hallucination_rate = round(hallucinated / claimed_total, 4) if claimed_total else 0.0

    all_mni = [v for entry in mni_values for v in entry.values()]
    metrics.mean_mni = round(float(np.mean(all_mni)), 5) if all_mni else 0.0

    all_flips = [v for entry in flips for v in entry.values()]
    metrics.flip_rate = round(sum(1 for f in all_flips if f) / len(all_flips), 4) if all_flips else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Trajectory-level indicators
# ---------------------------------------------------------------------------


def trajectory_metrics(states: Sequence[Any]) -> Dict[str, Any]:
    """Gate pass rate, revision effectiveness, degradation rate over a run set."""
    first_pass, total_gates, revisions, effective = 0, 0, 0, 0
    degraded_videos, n_calls = 0, 0
    for state in states:
        n_calls += state.budget.llm_calls_used
        if state.budget.degradations:
            degraded_videos += 1
        for records in state.gate_records.values():
            for record in records:
                total_gates += 1
                if record.retry_idx == 0 and record.passed:
                    first_pass += 1
                if record.retry_idx > 0:
                    revisions += 1
                    if record.passed:
                        effective += 1
    return {
        "gate_first_pass_rate": round(first_pass / total_gates, 4) if total_gates else 0.0,
        "revision_effectiveness": round(effective / revisions, 4) if revisions else 0.0,
        "degraded_video_rate": round(degraded_videos / len(states), 4) if states else 0.0,
        "mean_llm_calls": round(n_calls / len(states), 2) if states else 0.0,
        "n_videos": len(states),
    }


__all__ = [
    "Interval", "iou", "TPDecision", "tp_decision", "Match", "greedy_match",
    "ProposalMetrics",
    "evaluate_proposals", "aggregate_proposal_metrics", "sweep_iou_threshold",
    "DEFAULT_IOU_THRESHOLD", "unweighted_f1",
    "unweighted_average_recall", "accuracy", "au_set_metrics", "count_errors",
    "bleu", "rouge_n", "rouge_l", "expected_calibration_error", "brier_score",
    "CausalReliability", "causal_reliability", "trajectory_metrics",
]
