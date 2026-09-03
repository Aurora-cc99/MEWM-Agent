"""Assemble the paper's 4.2 evaluation report from finished runs.

**What this reads.** A finished ``run`` leaves four artefacts per video under
``runs/<video_id>/``: ``answer.json`` (the composed answer, its proposals and per-proposal
analyses), ``state.json`` (the full working state, including gate records, challenges and
the critic's counterfactual analysis), ``summary.json`` and ``episodic.json``. The report
is computed from the first two. Nothing is re-run, so a report is cheap and can be
regenerated at a different IoU threshold without touching a GPU.

**Unavailable is a first-class answer.** A section that cannot be computed says so, with
the reason and what would fix it. It does not fall back to a default that happens to look
good. This matters more than it sounds: a narrative section silently reporting BLEU 0.0
against absent references, or a causal-reliability block reporting a perfect 0.0
hallucination rate because no AU was ever claimed, is worse than no number at all --
both read as results, and both are artefacts of missing input. Every such section here
carries ``status: "unavailable"`` and the reader can tell the difference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import MEWMConfig, load_config
from ..knowledge.au_anatomy import SLOT_AUS
from ..knowledge.emotion_prototypes import FINE_EMOTIONS, canonical_fine_label
from .metrics import (
    accuracy, aggregate_proposal_metrics, au_set_metrics, bleu, brier_score,
    causal_reliability, count_errors, evaluate_proposals, expected_calibration_error,
    greedy_match, rouge_l, rouge_n, trajectory_metrics, unweighted_average_recall,
    unweighted_f1,
)

LOGGER = logging.getLogger(__name__)


def _unavailable(reason: str, remedy: str = "") -> Dict[str, Any]:
    """A section that could not be computed, and why. Never a zero pretending to be one."""
    out: Dict[str, Any] = {"status": "unavailable", "reason": reason}
    if remedy:
        out["remedy"] = remedy
    return out


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------


@dataclass
class RunArtefacts:
    """One finished video run, as it sits on disk."""

    video_id: str
    path: Path
    answer: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def from_annotation(self) -> bool:
        """True when this run was given the ground-truth intervals -- the P4 upper bound."""
        return bool(self.answer.get("proposals_from_annotation"))

    def proposals(self) -> List[Tuple[int, int]]:
        return [(int(p["onset"]), int(p["offset"]))
                for p in self.answer.get("part1_proposals", [])]

    def analyses(self) -> List[Dict[str, Any]]:
        return list(self.answer.get("part2_analysis", []))


def load_runs(root: Path | str, video_ids: Optional[Sequence[str]] = None) -> List[RunArtefacts]:
    """Load every run under ``root``, or only the named ones.

    A directory missing ``answer.json`` is skipped with a warning rather than failing the
    report: a partially-completed sweep should still produce numbers for what finished,
    and the count of what was loaded is reported alongside them.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    wanted = set(video_ids) if video_ids else None
    runs: List[RunArtefacts] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if wanted is not None and child.name not in wanted:
            continue
        answer_path = child / "answer.json"
        if not answer_path.is_file():
            continue
        run = RunArtefacts(video_id=child.name, path=child)
        try:
            run.answer = json.loads(answer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("unreadable answer.json in %s: %s", child, exc)
            continue
        for attribute, name in (("state", "state.json"), ("summary", "summary.json")):
            path = child / name
            if path.is_file():
                try:
                    setattr(run, attribute, json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError) as exc:
                    LOGGER.warning("unreadable %s in %s: %s", name, child, exc)
        runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Layer 2 -- P1 localisation, P2 AU detection, P3 emotion recognition
# ---------------------------------------------------------------------------


def _paired_events(run: RunArtefacts, events: Sequence[Any], config: MEWMConfig):
    """Pair each analysed proposal with the ground-truth event it claimed.
    """
    analyses = run.analyses()
    proposals = run.proposals()
    labels = [str(a.get("fine_label", "")) for a in analyses]
    matches = greedy_match(
        proposals, [e.interval for e in events], labels,
        [e.fine_label for e in events],
        iou_threshold=config.evaluation.iou_threshold,
        affective_rescue=config.evaluation.affective_rescue,
        rescue_min_iou=config.evaluation.rescue_min_iou,
    )
    for match in matches:
        if match.truth_index is None or not match.is_tp:
            continue
        analysis = analyses[match.proposal_index] if match.proposal_index < len(analyses) else {}
        yield analysis, events[match.truth_index], match


def _p1(runs: Sequence[RunArtefacts], events_by_video: Dict[str, List[Any]],
        config: MEWMConfig) -> Dict[str, Any]:
    """Proposal-level localisation under eq. (2) and under strict IoU."""
    per_video, details = [], []
    for run in runs:
        events = events_by_video.get(run.video_id)
        if events is None:
            continue
        metrics = evaluate_proposals(
            run.proposals(), [e.interval for e in events],
            [str(a.get("fine_label", "")) for a in run.analyses()],
            [e.fine_label for e in events],
            iou_threshold=config.evaluation.iou_threshold,
            affective_rescue=config.evaluation.affective_rescue,
            rescue_min_iou=config.evaluation.rescue_min_iou,
        )
        per_video.append(metrics)
        details.append({"video": run.video_id, "from_annotation": run.from_annotation,
                        **metrics.to_dict()})
    if not per_video:
        return _unavailable("no run could be paired with annotations",
                            "check that --dataset matches the runs directory")
    return {"status": "ok", "aggregate": aggregate_proposal_metrics(per_video).to_dict(),
            "per_video": details}


def _p2(runs: Sequence[RunArtefacts], events_by_video: Dict[str, List[Any]],
        config: MEWMConfig) -> Dict[str, Any]:
    """AU detection over the slot AUs: per-unit F1, macro UF1, and set-level agreement.

    Reported over the *matched* proposals only. Scoring AU sets on proposals that located
    nothing would measure localisation a second time under an AU-shaped name.
    """
    predicted_sets: List[List[str]] = []
    truth_sets: List[List[str]] = []
    for run in runs:
        events = events_by_video.get(run.video_id)
        if events is None:
            continue
        for analysis, event, _ in _paired_events(run, events, config):
            predicted_sets.append([str(a) for a in (analysis.get("active_aus") or [])])
            truth_sets.append([str(a) for a in (event.aus or [])])

    if not predicted_sets:
        return _unavailable("no matched proposal to score AUs on",
                            "P2 is conditioned on P1: fix localisation first")
    if not any(truth_sets):
        return _unavailable(
            "the annotations carry no AU labels for the matched events",
            "P2 needs an AU-annotated corpus; CAS(ME)^2/CAS(ME)^3 carry them, a "
            "spotting-only annotation file does not")

    # Per-unit: each slot AU becomes a binary problem over the matched events.
    per_au: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for au in SLOT_AUS:
        tp = sum(1 for p, t in zip(predicted_sets, truth_sets) if au in p and au in t)
        fp = sum(1 for p, t in zip(predicted_sets, truth_sets) if au in p and au not in t)
        fn = sum(1 for p, t in zip(predicted_sets, truth_sets) if au not in p and au in t)
        support = tp + fn
        if not support:
            # An AU that never occurs in the reference has no F1 to speak of; including a
            # 0.0 for it would drag the macro average by an amount that depends only on
            # how many units the corpus happens not to annotate.
            per_au[au] = {"f1": None, "support": 0}
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / support
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_au[au] = {"f1": round(f1, 4), "precision": round(precision, 4),
                      "recall": round(recall, 4), "support": support}
        f1s.append(f1)

    return {
        "status": "ok",
        "protocol": "LOSO",
        "n_matched_events": len(predicted_sets),
        "per_au": per_au,
        "uf1_macro": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "n_units_scored": len(f1s),
        "set_level": au_set_metrics(predicted_sets, truth_sets),
        "note": ("LOSO only. The paper also asks for LODO; build the folds with "
                 "mewm.data.datasets.lodo_folds and re-run this report per held-out "
                 "corpus to produce that column."),
    }


def _p3(runs: Sequence[RunArtefacts], events_by_video: Dict[str, List[Any]],
        config: MEWMConfig) -> Dict[str, Any]:
    """Emotion recognition on the matched events, plus event-count error per video."""
    y_true: List[str] = []
    y_pred: List[str] = []
    predicted_counts: List[int] = []
    truth_counts: List[int] = []

    for run in runs:
        events = events_by_video.get(run.video_id)
        if events is None:
            continue
        predicted_counts.append(len(run.proposals()))
        truth_counts.append(len(events))
        for analysis, event, _ in _paired_events(run, events, config):
            pred, _ = canonical_fine_label(str(analysis.get("fine_label", "")))
            true, _ = canonical_fine_label(str(event.fine_label))
            y_true.append(true)
            y_pred.append(pred)

    counts = (count_errors(predicted_counts, truth_counts)
              if predicted_counts else _unavailable("no runs"))
    if not y_true:
        return {"status": "partial", "recognition": _unavailable(
            "no matched proposal carries an emotion label"), "event_counts": counts}

    labels = [e for e in FINE_EMOTIONS if e in set(y_true) | set(y_pred)]
    uf1, per_class = unweighted_f1(y_true, y_pred, labels)
    return {
        "status": "ok",
        "n_events": len(y_true),
        "uf1": uf1,
        "uar": unweighted_average_recall(y_true, y_pred, labels),
        "accuracy": accuracy(y_true, y_pred),
        "per_class_f1": per_class,
        "event_counts": counts,
    }


# ---------------------------------------------------------------------------
# Calibration, causal reliability, trajectory
# ---------------------------------------------------------------------------


def _calibration(runs: Sequence[RunArtefacts], events_by_video: Dict[str, List[Any]],
                 config: MEWMConfig) -> Dict[str, Any]:
    """ECE and Brier over the confidence attached to each matched analysis.

    "Correct" here means the fine label was right, which is the event the confidence is
    a statement about.
    """
    confidences: List[float] = []
    correct: List[bool] = []
    for run in runs:
        events = events_by_video.get(run.video_id)
        if events is None:
            continue
        for analysis, event, _ in _paired_events(run, events, config):
            raw = analysis.get("confidence")
            if raw is None:
                continue
            pred, _ = canonical_fine_label(str(analysis.get("fine_label", "")))
            true, _ = canonical_fine_label(str(event.fine_label))
            confidences.append(float(raw))
            correct.append(pred == true)
    if not confidences:
        return _unavailable("no matched analysis carries a confidence")
    return {
        "status": "ok",
        "n": len(confidences),
        "ece": expected_calibration_error(confidences, correct),
        "brier": brier_score(confidences, correct),
        "mean_confidence": round(sum(confidences) / len(confidences), 4),
        "empirical_accuracy": round(sum(1 for c in correct if c) / len(correct), 4),
    }


def _causal(runs: Sequence[RunArtefacts]) -> Dict[str, Any]:
    """rho_pass / rho_hall / mean MNI / rho_flip, read out of the saved states.

    ``rho_flip`` is the one that used to be unrecoverable: the critic computed the
    per-AU belief flips, printed them into its own prompt, and dropped them. They are now
    persisted on the causal chain, so this reads them.
    """
    challenge_finals: List[str] = []
    claimed: List[List[str]] = []
    evidenced: List[List[str]] = []
    mni_values: List[Dict[str, float]] = []
    flips: List[Dict[str, bool]] = []
    missing_flips = 0

    for run in runs:
        state = run.state
        if not state:
            continue
        for records in (state.get("challenges") or {}).values():
            for record in records:
                final = str(record.get("final", ""))
                if final:
                    challenge_finals.append(final)
        for analysis in run.analyses():
            claimed.append([str(a) for a in (analysis.get("active_aus") or [])])
            # Evidence-backed units: the k_crit set is what the agent could defend.
            evidenced.append([str(a) for a in (analysis.get("k_crit") or [])])
        for cot in (state.get("causal_cots") or {}).values():
            cf = cot.get("cf_mhv") or {}
            if cf.get("mni"):
                mni_values.append({k: float(v) for k, v in cf["mni"].items()})
            if "flips" in cf:
                flips.append({k: bool(v) for k, v in (cf["flips"] or {}).items()})
            else:
                missing_flips += 1

    if not claimed and not challenge_finals:
        return _unavailable("no saved state carries challenges or analyses",
                            "run the pipeline with checkpointing so state.json is written")

    report = causal_reliability(challenge_finals, claimed, evidenced, mni_values, flips)
    out = {"status": "ok", "n_challenges": len(challenge_finals), **report.to_dict()}
    if missing_flips:
        # Runs produced before the critic persisted its flips cannot contribute here,
        # and a rho_flip averaged over only the newer half is not the run's rho_flip.
        out["flip_rate"] = None
        out["flip_rate_note"] = (
            f"{missing_flips} proposal(s) predate the fix that persists the critic's "
            f"belief-flip test; rho_flip is withheld rather than computed on the "
            f"subset that happens to carry it. Re-run those videos to fill it in.")
    return out


class _RecordView(SimpleNamespace):
    """Attribute access over a saved gate record, for :func:`trajectory_metrics`."""


def _trajectory(runs: Sequence[RunArtefacts]) -> Dict[str, Any]:
    """Gate first-pass rate, revision effectiveness, degradation rate.

    Adapts the saved JSON to the attribute shape :func:`trajectory_metrics` expects
    rather than reimplementing it -- the point of this module is to give the existing
    metric a caller, not to grow a second copy of it that can drift.
    """
    views = []
    for run in runs:
        state = run.state
        if not state:
            continue
        budget = state.get("budget") or {}
        gate_records = {
            cid: [_RecordView(passed=bool(r.get("passed")),
                              retry_idx=int(r.get("retry_idx", 0)))
                  for r in records]
            for cid, records in (state.get("gate_records") or {}).items()
        }
        views.append(SimpleNamespace(
            budget=SimpleNamespace(
                llm_calls_used=int(budget.get("llm_calls_used", 0)),
                degradations=list(budget.get("degradations") or []),
            ),
            gate_records=gate_records,
        ))
    if not views:
        return _unavailable("no saved state to read gate records from")
    return {"status": "ok", **trajectory_metrics(views)}


def _narrative(runs: Sequence[RunArtefacts],
               references: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """BLEU / ROUGE against reference narratives, when there are any.

    Without references this is *not* zero -- it is unmeasured, and says so.
    """
    if not references:
        return _unavailable(
            "no reference narratives supplied",
            "pass --references <file.json> mapping video_id -> reference answer text")
    scored = []
    for run in runs:
        reference = references.get(run.video_id)
        candidate = str(run.answer.get("answer", ""))
        if not reference or not candidate:
            continue
        scored.append({
            "video": run.video_id,
            "bleu": bleu(candidate, reference),
            "rouge_1": rouge_n(candidate, reference, 1),
            "rouge_2": rouge_n(candidate, reference, 2),
            "rouge_l": rouge_l(candidate, reference),
        })
    if not scored:
        return _unavailable("no run matched a supplied reference by video_id")
    keys = ("bleu", "rouge_1", "rouge_2", "rouge_l")
    return {
        "status": "ok", "n": len(scored),
        "mean": {k: round(sum(s[k] for s in scored) / len(scored), 4) for k in keys},
        "per_video": scored,
    }


# ---------------------------------------------------------------------------
# P4 -- localisation error propagating through the pipeline
# ---------------------------------------------------------------------------


def compare_p4(self_runs: Sequence[RunArtefacts], upper_bound_runs: Sequence[RunArtefacts],
               events_by_video: Dict[str, List[Any]], config: MEWMConfig) -> Dict[str, Any]:
    """How much of the end-to-end error is localisation error, stratified by how far off.
    """
    by_id = {run.video_id: run for run in upper_bound_runs}
    paired = [(run, by_id[run.video_id]) for run in self_runs if run.video_id in by_id]
    if not paired:
        return _unavailable(
            "no video has both a self-proposed and an upper-bound run",
            "produce the second half with: run --use-gt-proposals --output <dir>_gt")

    strata = {"0-5": [], "5-10": [], ">10": [], "unmatched": []}
    rows = []
    for own, upper in paired:
        events = events_by_video.get(own.video_id)
        if events is None:
            continue
        truths = [e.interval for e in events]
        own_metrics = evaluate_proposals(
            own.proposals(), truths,
            [str(a.get("fine_label", "")) for a in own.analyses()],
            [e.fine_label for e in events],
            iou_threshold=config.evaluation.iou_threshold,
            affective_rescue=config.evaluation.affective_rescue,
            rescue_min_iou=config.evaluation.rescue_min_iou)
        upper_metrics = evaluate_proposals(
            upper.proposals(), truths,
            [str(a.get("fine_label", "")) for a in upper.analyses()],
            [e.fine_label for e in events],
            iou_threshold=config.evaluation.iou_threshold,
            affective_rescue=config.evaluation.affective_rescue,
            rescue_min_iou=config.evaluation.rescue_min_iou)

        boundary = max(own_metrics.onset_error_median, own_metrics.offset_error_median)
        if not own_metrics.tp:
            bucket = "unmatched"
        elif boundary <= 5:
            bucket = "0-5"
        elif boundary <= 10:
            bucket = "5-10"
        else:
            bucket = ">10"
        delta = round(upper_metrics.f1 - own_metrics.f1, 4)
        strata[bucket].append(delta)
        rows.append({
            "video": own.video_id, "stratum": bucket,
            "boundary_error_frames": round(float(boundary), 2),
            "f1_self": own_metrics.f1, "f1_upper_bound": upper_metrics.f1,
            "delta_f1": delta,
        })

    return {
        "status": "ok",
        "n_paired": len(rows),
        "by_stratum": {
            name: {"n": len(values),
                   "mean_delta_f1": round(sum(values) / len(values), 4) if values else None}
            for name, values in strata.items()
        },
        "per_video": rows,
        "reading": ("delta_f1 is upper bound minus self-produced: large and positive "
                    "means localisation error is what is costing the answer."),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_report(
    dataset: str,
    runs: Sequence[RunArtefacts],
    videos: Sequence[Any],
    config: Optional[MEWMConfig] = None,
    references: Optional[Dict[str, str]] = None,
    layer1: Optional[Dict[str, Any]] = None,
    upper_bound_runs: Sequence[RunArtefacts] = (),
) -> Dict[str, Any]:
    """The whole 4.2 table for one dataset, from artefacts already on disk."""
    config = config or load_config()
    events_by_video = {v.video_id: v.micro_events() for v in videos}
    self_runs = [r for r in runs if not r.from_annotation]
    upper = list(upper_bound_runs) or [r for r in runs if r.from_annotation]

    report: Dict[str, Any] = {
        "dataset": dataset,
        "n_runs": len(runs),
        "n_self_proposed": len(self_runs),
        "n_upper_bound": len(upper),
        "evaluation": {
            "iou_threshold": config.evaluation.iou_threshold,
            "affective_rescue": config.evaluation.affective_rescue,
            "rescue_min_iou": config.evaluation.rescue_min_iou,
        },
        "layer1_rollout_quality": layer1 or _unavailable(
            "not requested",
            "pass --layer1 to run the frozen dynamics over the pool; it needs the "
            "representation engine and so costs real time"),
        "layer2_p1_localisation": _p1(self_runs or list(runs), events_by_video, config),
        "layer2_p2_au_detection": _p2(self_runs or list(runs), events_by_video, config),
        "layer2_p3_emotion": _p3(self_runs or list(runs), events_by_video, config),
        "calibration": _calibration(self_runs or list(runs), events_by_video, config),
        "causal_reliability": _causal(runs),
        "trajectory": _trajectory(runs),
        "narrative": _narrative(self_runs or list(runs), references),
    }
    report["p4_propagation"] = (
        compare_p4(self_runs, upper, events_by_video, config) if self_runs and upper
        else _unavailable(
            "needs both a self-proposed and an upper-bound run set",
            "produce the second half with: run --use-gt-proposals --output <dir>_gt"))
    return report


def format_report(report: Dict[str, Any]) -> str:
    """A flat text rendering, so the command is readable without a JSON viewer."""
    lines: List[str] = []
    lines.append(f"{report.get('dataset', '?')}  --  {report.get('n_runs', 0)} run(s), "
                 f"IoU>{report.get('evaluation', {}).get('iou_threshold')}")
    lines.append("=" * 72)

    def section(title: str, body: Dict[str, Any], fields: Sequence[str]) -> None:
        lines.append("")
        lines.append(title)
        if body.get("status") == "unavailable":
            lines.append(f"  unavailable: {body.get('reason')}")
            if body.get("remedy"):
                lines.append(f"  -> {body['remedy']}")
            return
        for name in fields:
            if name in body and body[name] is not None:
                lines.append(f"  {name:<24} {body[name]}")

    p1 = report.get("layer2_p1_localisation", {})
    aggregate = p1.get("aggregate", {}) if p1.get("status") == "ok" else {}
    section("P1  proposal localisation", {**p1, **aggregate},
            ("tp", "tp_strict", "n_rescued", "precision", "recall", "f1",
             "f1_strict", "rescue_gain", "onset_error_median", "offset_error_median"))
    section("P2  AU detection", report.get("layer2_p2_au_detection", {}),
            ("n_matched_events", "uf1_macro", "n_units_scored", "set_level", "note"))
    section("P3  emotion recognition", report.get("layer2_p3_emotion", {}),
            ("n_events", "uf1", "uar", "accuracy", "event_counts"))
    section("calibration", report.get("calibration", {}),
            ("n", "ece", "brier", "mean_confidence", "empirical_accuracy"))
    section("causal reliability", report.get("causal_reliability", {}),
            ("n_challenges", "challenge_pass_rate", "hallucination_rate", "mean_mni",
             "flip_rate", "flip_rate_note", "n_samples"))
    section("trajectory", report.get("trajectory", {}),
            ("n_videos", "gate_first_pass_rate", "revision_effectiveness",
             "degraded_video_rate", "mean_llm_calls"))
    section("layer 1  rollout quality", report.get("layer1_rollout_quality", {}),
            ("alignment_auc", "prediction_error", "counterfactual_structure"))
    section("P4  localisation propagation", report.get("p4_propagation", {}),
            ("n_paired", "by_stratum", "reading"))
    section("narrative", report.get("narrative", {}), ("n", "mean"))
    return "\n".join(lines)


__all__ = [
    "RunArtefacts", "load_runs", "build_report", "format_report", "compare_p4",
]
