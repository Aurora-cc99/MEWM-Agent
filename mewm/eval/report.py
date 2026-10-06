"""Aggregate evaluation report builder across subjects and datasets."""
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
from . import megc_metrics as mm
from .metrics import (
    accuracy, aggregate_proposal_metrics, au_set_metrics, bleu, brier_score,
    causal_reliability, count_errors, evaluate_proposals, expected_calibration_error,
    greedy_match, rouge_l, rouge_n, trajectory_metrics, unweighted_average_recall,
    unweighted_f1,
)

LOGGER = logging.getLogger(__name__)


def _unavailable(reason: str, remedy: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "unavailable", "reason": reason}
    if remedy:
        out["remedy"] = remedy
    return out


@dataclass
class RunArtefacts:

    video_id: str
    path: Path
    answer: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def from_annotation(self) -> bool:
        return bool(self.answer.get("proposals_from_annotation"))

    def proposals(self) -> List[Tuple[int, int]]:
        return [(int(p["onset"]), int(p["offset"]))
                for p in self.answer.get("part1_proposals", [])]

    def analyses(self) -> List[Dict[str, Any]]:
        return list(self.answer.get("part2_analysis", []))


def load_runs(root: Path | str, video_ids: Optional[Sequence[str]] = None) -> List[RunArtefacts]:
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


def _paired_events(run: RunArtefacts, events: Sequence[Any], config: MEWMConfig):
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
        return _unavailable("",
                            "")
    if not any(truth_sets):
        return _unavailable(
            "",
            "")

    per_au: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for au in SLOT_AUS:
        tp = sum(1 for p, t in zip(predicted_sets, truth_sets) if au in p and au in t)
        fp = sum(1 for p, t in zip(predicted_sets, truth_sets) if au in p and au not in t)
        fn = sum(1 for p, t in zip(predicted_sets, truth_sets) if au not in p and au in t)
        support = tp + fn
        if not support:
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
        "note": (""),
    }


def _p3(runs: Sequence[RunArtefacts], events_by_video: Dict[str, List[Any]],
        config: MEWMConfig) -> Dict[str, Any]:

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
    megc = mm.recognition_scores(y_true, y_pred)
    return {
        "status": "ok",
        "n_events": len(y_true),
        "uf1": uf1,
        "uar": unweighted_average_recall(y_true, y_pred, labels),
        "accuracy": accuracy(y_true, y_pred),
        "per_class_f1": per_class,
        "coarse": megc["coarse"],
        "fine": megc["fine"],
        "headline": "",
        "event_counts": counts,
    }


def _canonical_or_empty(label: Any) -> str:
    text, recognised = canonical_fine_label(str(label or ""))
    return text if recognised else ""


def _qa_segment_reference(interval: Tuple[int, int], items: Sequence[Any]) -> Optional[str]:
    from ..data.qa_loader import parse_segment_question

    best: Optional[Tuple[float, str]] = None
    for item in items:
        if item.qtype != "segment_analysis":
            continue
        span = parse_segment_question(item.question)
        if not span:
            continue
        from .metrics import iou as _iou
        overlap = _iou(tuple(interval), (int(span["onset"]), int(span["offset"])))
        if best is None or overlap > best[0]:
            best = (overlap, item.answer_text)
    return best[1] if best and best[0] > 0.0 else None


def _megc_suite(runs: Sequence[RunArtefacts],
                events_by_video: Dict[str, List[Any]],
                all_events_by_video: Dict[str, List[Any]],
                config: MEWMConfig,
                qa: Optional[Dict[str, Sequence[Any]]] = None) -> Dict[str, Any]:
    threshold = config.evaluation.iou_threshold
    rescue_min_iou = config.evaluation.rescue_min_iou

    spotting_rows: List[Dict[str, Any]] = []
    count_pairs: Dict[str, List[Tuple[int, int]]] = {
        quantity: [] for quantity in mm.COUNT_QUANTITIES}
    au_pred: List[List[str]] = []
    au_true: List[List[str]] = []
    emo_true: List[str] = []
    emo_pred: List[str] = []
    event_candidates: List[str] = []
    event_references: List[str] = []
    whole_candidates: List[str] = []
    whole_references: List[str] = []

    n_pred_labels = 0
    n_true_labels = 0
    tp_correct = 0
    tp_fine_true: List[str] = []
    tp_fine_pred: List[str] = []

    for run in runs:
        micro_events = events_by_video.get(run.video_id)
        if micro_events is None:
            continue
        all_events = all_events_by_video.get(run.video_id, micro_events)
        proposals = run.proposals()
        analyses = run.analyses()
        labels = [_canonical_or_empty(a.get("fine_label", "")) for a in analyses]
        truths = [e.interval for e in micro_events]

        spotting_rows.append({
            "proposals": proposals,
            "proposal_types": ["micro-expression"] * len(proposals),
            "proposal_labels": labels,
            "truth": truths,
            "truth_types": ["micro-expression"] * len(truths),
            "truth_labels": [str(e.fine_label or "") for e in micro_events],
        })

        n_pred_labels += len(proposals)
        n_true_labels += len(truths)
        for pred_i, truth_i, overlap in mm._greedy_pairs(proposals, truths, threshold):
            if overlap < threshold:
                continue
            gold = _canonical_or_empty(micro_events[truth_i].fine_label)
            tp_fine_true.append(gold)
            tp_fine_pred.append(labels[pred_i])
            if labels[pred_i] and gold and labels[pred_i] == gold:
                tp_correct += 1

        for analysis, event, _ in _paired_events(run, micro_events, config):
            au_pred.append([str(a) for a in (analysis.get("active_aus") or [])])
            au_true.append([str(a) for a in (event.aus or [])])
            emo_pred.append(_canonical_or_empty(analysis.get("fine_label", "")))
            emo_true.append(_canonical_or_empty(event.fine_label))

        count_pairs["micro"].append((len(proposals), len(micro_events)))
        count_pairs["expression"].append((
            int(run.answer.get("n_detected", len(proposals))),
            int(run.answer.get("n_annotated", len(all_events)))))
        spotting_summary = run.summary.get("spotting") or {}
        n_macro_pred = int(spotting_summary.get("n_macro", 0))
        n_macro_true = len(all_events) - len(micro_events)
        count_pairs["macro"].append((n_macro_pred, n_macro_true))

        if qa is not None:
            items = list(qa.get(run.video_id) or [])
            whole = next((it for it in items if it.qtype == "reason_full"), None)
            if whole is not None:
                whole_candidates.append(str(run.answer.get("answer", "")))
                whole_references.append(whole.answer_text)
            for analysis in analyses:
                interval = analysis.get("interval")
                if not (isinstance(interval, (list, tuple)) and len(interval) >= 2):
                    continue
                reference = _qa_segment_reference(
                    (int(interval[0]), int(interval[1])), items)
                if reference is None:
                    continue
                event_candidates.append(str(analysis.get("au_cot") or ""))
                event_references.append(reference)

    spotting = mm.spotting_scores(spotting_rows, iou_threshold=threshold,
                                  rescue_min_iou=rescue_min_iou)
    f1_spot = (spotting.get("headline_f1")
               if spotting.get("status") == "ok" else None)
    analysis = mm._prf(tp_correct, n_pred_labels, n_true_labels)
    f1a_block = mm.recognition_scores(tp_fine_true, tp_fine_pred)
    f1a_fine = f1a_block.get("fine", {}).get("megc", {})
    f1_analysis = (f1a_fine.get("reg_uf1")
                   if f1a_fine.get("status") == "ok" else 0.0)

    if f1_spot is None:
        strs_block = _unavailable(
            "no spotting F1 was computed, so STRS has no F1_s to multiply")
    else:
        strs_block = {
            "status": "ok",
            "strs": mm.strs(f1_spot, f1_analysis),
            "f1_spotting": round(f1_spot, 4),
            "f1_analysis": round(f1_analysis, 4),
            "f1_analysis_basis": (""),
            "f1_analysis_micro": round(float(analysis.get("f1", 0.0) or 0.0), 4),
            "analysis": dict(analysis, note=(
                "")),
            "definition": "",
        }

    return {
        "spotting": {
            "interval": dict(spotting, n_videos=len(spotting_rows)),
            "counting": mm.count_scores(count_pairs),
        },
        "recognition": {
            "action_units": mm.au_scores(au_pred, au_true),
            "emotion": mm.recognition_scores(emo_true, emo_pred),
            "text": mm.text_scores(event_candidates, event_references),
        },
        "strs": {
            "score": strs_block,
            "text": mm.text_scores(whole_candidates, whole_references),
        },
        "provenance": {
            "spotting": "",
            "counting": "",
            "recognition": "",
            "strs": "",
        },
    }


def _calibration(runs: Sequence[RunArtefacts], events_by_video: Dict[str, List[Any]],
                 config: MEWMConfig) -> Dict[str, Any]:
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


def _runtime(runs: Sequence[RunArtefacts]) -> Dict[str, Any]:
    per_video: List[Dict[str, Any]] = []
    times: List[float] = []
    fps_values: List[float] = []
    for run in runs:
        elapsed = run.summary.get("elapsed_s")
        if elapsed is None:
            continue
        elapsed = float(elapsed)
        frames = int(run.summary.get("frames_processed", 0) or 0)
        fps = round(frames / elapsed, 4) if elapsed > 0 and frames else None
        per_video.append({
            "video": run.video_id, "elapsed_s": round(elapsed, 2),
            "frames_processed": frames, "frames_per_s": fps,
        })
        times.append(elapsed)
        if fps is not None:
            fps_values.append(fps)

    if not times:
        return _unavailable(
            "no run carried a summary.json with elapsed_s",
            "the pipeline writes summary.json (with elapsed_s) at the end of every "
            "video automatically -- check that runs were not copied without it")

    times_sorted = sorted(times)
    n = len(times_sorted)
    median = (times_sorted[n // 2] if n % 2 else
              (times_sorted[n // 2 - 1] + times_sorted[n // 2]) / 2)
    total_wall_s = sum(times)
    return {
        "status": "ok",
        "n_videos": n,
        "total_wall_s": round(total_wall_s, 2),
        "total_wall_hms": _format_hms(total_wall_s),
        "throughput_videos_per_hour": round(3600.0 * n / total_wall_s, 4) if total_wall_s else 0.0,
        "throughput_frames_per_s_pooled": (
            round(sum(v["frames_processed"] for v in per_video) / total_wall_s, 4)
            if total_wall_s else 0.0),
        "per_video_seconds": {
            "mean": round(total_wall_s / n, 2),
            "median": round(median, 2),
            "min": round(min(times_sorted), 2),
            "max": round(max(times_sorted), 2),
        },
        "per_video_frames_per_s": {
            "mean": round(sum(fps_values) / len(fps_values), 4) if fps_values else None,
        },
        "per_video": per_video,
    }


def _format_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _causal(runs: Sequence[RunArtefacts]) -> Dict[str, Any]:
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
        out["flip_rate"] = None
        out["flip_rate_note"] = (
            f"{missing_flips} proposal(s) predate the fix that persists the critic's "
            f"belief-flip test; rho_flip is withheld rather than computed on the "
            f"subset that happens to carry it. Re-run those videos to fill it in.")
    return out


class _RecordView(SimpleNamespace):
    pass

def _trajectory(runs: Sequence[RunArtefacts]) -> Dict[str, Any]:
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


def compare_p4(self_runs: Sequence[RunArtefacts], upper_bound_runs: Sequence[RunArtefacts],
               events_by_video: Dict[str, List[Any]], config: MEWMConfig) -> Dict[str, Any]:
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
        "reading": (""),
    }


def build_report(
    dataset: str,
    runs: Sequence[RunArtefacts],
    videos: Sequence[Any],
    config: Optional[MEWMConfig] = None,
    references: Optional[Dict[str, str]] = None,
    layer1: Optional[Dict[str, Any]] = None,
    upper_bound_runs: Sequence[RunArtefacts] = (),
    qa: Optional[Dict[str, Sequence[Any]]] = None,
) -> Dict[str, Any]:
    config = config or load_config()
    events_by_video = {v.video_id: v.micro_events() for v in videos}
    all_events_by_video = {v.video_id: v.events for v in videos}
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
        "megc": _megc_suite(self_runs or list(runs), events_by_video,
                            all_events_by_video, config, qa),
        "calibration": _calibration(self_runs or list(runs), events_by_video, config),
        "causal_reliability": _causal(runs),
        "trajectory": _trajectory(runs),
        "runtime": _runtime(runs),
        "narrative": _narrative(self_runs or list(runs), references),
    }
    report["p4_propagation"] = (
        compare_p4(self_runs, upper, events_by_video, config) if self_runs and upper
        else _unavailable(
            "needs both a self-proposed and an upper-bound run set",
            "produce the second half with: run --use-gt-proposals --output <dir>_gt"))
    return report


def format_report(report: Dict[str, Any]) -> str:
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
            ("n_events", "uf1", "uar", "accuracy", "coarse", "fine", "event_counts"))
    megc = report.get("megc", {})
    megc_spot = megc.get("spotting", {})
    megc_rec = megc.get("recognition", {})
    megc_strs = megc.get("strs", {})
    section("",
            megc_spot.get("interval", {}),
            (""))
    section("",
            megc_spot.get("counting", {}),
            ("expression", "micro", "macro"))
    section("",
            megc_rec.get("action_units", {}),
            ("f1_au", "jaccard_au", "n"))
    section("",
            megc_rec.get("emotion", {}),
            ("fine", "coarse"))
    section("",
            megc_rec.get("text", {}),
            ("bleu", "rouge_1", "n"))
    section("", megc_strs.get("score", {}),
            ("strs", "f1_spotting", "f1_analysis", "definition"))
    section("",
            megc_strs.get("text", {}),
            ("bleu", "rouge_1", "n"))
    section("calibration", report.get("calibration", {}),
            ("n", "ece", "brier", "mean_confidence", "empirical_accuracy"))
    section("causal reliability", report.get("causal_reliability", {}),
            ("n_challenges", "challenge_pass_rate", "hallucination_rate", "mean_mni",
             "flip_rate", "flip_rate_note", "n_samples"))
    section("trajectory", report.get("trajectory", {}),
            ("n_videos", "gate_first_pass_rate", "revision_effectiveness",
             "degraded_video_rate", "mean_llm_calls"))
    section("runtime", report.get("runtime", {}),
            ("n_videos", "total_wall_s", "total_wall_hms",
             "throughput_videos_per_hour", "throughput_frames_per_s_pooled",
             "per_video_seconds", "per_video_frames_per_s"))
    section("layer 1  rollout quality", report.get("layer1_rollout_quality", {}),
            ("alignment_auc", "prediction_error", "counterfactual_structure"))
    section("P4  localisation propagation", report.get("p4_propagation", {}),
            ("n_paired", "by_stratum", "reading"))
    section("narrative", report.get("narrative", {}), ("n", "mean"))
    return "\n".join(lines)


__all__ = [
    "RunArtefacts", "load_runs", "build_report", "format_report", "compare_p4",
]
