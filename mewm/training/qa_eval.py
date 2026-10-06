"""QA evaluation loop: answer correctness scoring for training feedback."""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import MEWMConfig, load_config
from ..data.datasets import LongVideo
from ..eval import megc_metrics as mm
from ..eval.megc_questions import (
    ALL_GROUPS, GROUP_AU_SET, GROUP_COUNT_EXPRESSION, GROUP_COUNT_MACRO,
    GROUP_COUNT_MICRO, GROUP_EVENT_RECOGNITION, GROUP_SPOT_INTERVAL, GROUP_TYPE_BINARY,
    GROUP_UNROUTED, GROUP_VIDEO_STRS, event_anchor, localisation_scope,
    normalise_au_set, parse_count, parse_expression_type, parse_intervals,
    route_question,
)
from ..llm.client import call_model
from ..knowledge.emotion_prototypes import canonical_fine_label, coarse_of
from .rl_prompts import VideoEvidence, _match_event, _render_evidence, unavailable_evidence

LOGGER = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = (
    "You are analysing facial movement in a long video. You are given the output of a "
    "frozen perceptual front end -- slot activations and interval proposals -- and one "
    "question about the video. Answer the question from that evidence.\n\n"
    "Reply with a single JSON object and nothing else, using exactly the keys the user "
    "message shows. Do not add, rename or omit top-level keys. Never mention this "
    "system's internals: no stage names, no agent names, no pipeline vocabulary, no "
    "document or equation references. Write about brows, lids, lips, timing, and what "
    "they indicate.\n\n"
    "Answering \"none\" or \"0\" is legitimate and common: most long videos in this "
    "corpus contain no micro-expression at all. Do not invent an event to fill a field."
)

_SKELETONS: Dict[str, str] = {
    GROUP_COUNT_EXPRESSION: '{"count": <integer>, "answer": "<one sentence>"}',
    GROUP_COUNT_MICRO: '{"count": <integer>, "answer": "<one sentence>"}',
    GROUP_COUNT_MACRO: '{"count": <integer>, "answer": "<one sentence>"}',
    GROUP_SPOT_INTERVAL: (
        '{"count": <integer>, "events": [{"interval": [<onset_frame>, <offset_frame>], '
        '"type": "micro-expression" | "macro-expression"}], "answer": "<one sentence>"}'),
    GROUP_AU_SET: (
        '{"action_units": ["<AU code or FACS name>", ...], "answer": "<one sentence>"}'),
    GROUP_TYPE_BINARY: (
        '{"expression_type": "micro-expression" | "macro-expression", '
        '"answer": "<one sentence>"}'),
    GROUP_EVENT_RECOGNITION: (
        '{"fine_label": "<happiness|surprise|fear|disgust|anger|sadness|contempt|other>", '
        '"coarse_label": "<positive|negative|surprise|other>", '
        '"action_units": ["<AU code>", ...], '
        '"interval": [<onset_frame>, <offset_frame>], '
        '"answer": "<the prose answer to the question>"}'),
    GROUP_VIDEO_STRS: (
        '{"n_micro": <integer>, "events": [{"interval": [<onset_frame>, <offset_frame>], '
        '"fine_label": "<emotion>", "coarse_label": "<coarse emotion>"}], '
        '"answer": "<the prose analysis of the whole video>"}'),
}

_COUNT_HINT = (
    "The proposal counts in the evidence block are unfiltered candidate windows from a "
    "low-level detector, typically one or two orders of magnitude more numerous than the "
    "real events; they are not the answer. Judge how many genuine expression events the "
    "evidence supports and report that number."
)

_GROUP_HINTS: Dict[str, str] = {
    GROUP_COUNT_EXPRESSION: _COUNT_HINT,
    GROUP_COUNT_MICRO: _COUNT_HINT,
    GROUP_COUNT_MACRO: _COUNT_HINT,
    GROUP_SPOT_INTERVAL: (
        "List one entry in \"events\" for every event you localise, and set \"count\" to "
        "exactly that many. Frame numbers are absolute frame indices in this video. "
        + _COUNT_HINT),
    GROUP_AU_SET: (
        "List the distinct action units visible anywhere in the video. AU codes such as "
        "\"AU4\" and official FACS names such as \"brow lowerer\" are both accepted."),
    GROUP_TYPE_BINARY: (
        "A micro-expression is brief (roughly under half a second); a macro-expression "
        "lasts longer. Decide which the named window is."),
    GROUP_VIDEO_STRS: (
        "\"events\" lists exactly \"n_micro\" entries, one per micro-expression event you "
        "claim, each with the emotion you read from it. Use 0 and [] when the evidence "
        "does not support one."),
}


@dataclass
class EvalItem:

    question_id: str
    question: str
    group: str
    route_detail: str
    video: str
    subject: str
    reference_answer: str
    evidence_text: str
    evidence_status: str
    anchor: Dict[str, int] = field(default_factory=dict)
    truth: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    product: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def parsed(self) -> bool:
        return bool(self.product)


def build_eval_items(
    dataset: str,
    videos: Sequence[LongVideo],
    qa_rows: Iterable[Dict[str, Any]],
    evidence_by_video: Optional[Dict[str, VideoEvidence]] = None,
) -> Tuple[List[EvalItem], Dict[str, Any]]:
    evidence_by_video = evidence_by_video or {}
    by_key = {v.video_key: v for v in videos}

    items: List[EvalItem] = []
    counts: Dict[str, int] = {group: 0 for group in ALL_GROUPS}
    counts[GROUP_UNROUTED] = 0
    skipped: List[Dict[str, str]] = []
    unrouted_examples: List[str] = []

    for row in qa_rows:
        question = str(row.get("question", ""))
        video_key = str(row.get("video", ""))
        group, detail = route_question(question)
        counts[group] = counts.get(group, 0) + 1

        if group == GROUP_UNROUTED:
            if len(unrouted_examples) < 10:
                unrouted_examples.append(question[:160])
            skipped.append({"video": video_key, "reason": "unrecognised template",
                            "question": question[:160]})
            continue

        video = by_key.get(video_key)
        if video is None:
            skipped.append({"video": video_key, "reason": "video not in this pool",
                            "question": question[:160]})
            continue

        anchor = event_anchor(question) or {}
        truth: Dict[str, Any] = {}
        if group in (GROUP_EVENT_RECOGNITION, GROUP_TYPE_BINARY):
            probe = anchor or _anchor_from_type_question(video, question)
            event = _match_event(video, probe) if probe else None
            if event is not None:
                truth = {
                    "fine_label": event.fine_label, "coarse_label": event.coarse_label,
                    "interval": list(event.interval),
                    "expression_type": _type_of_event(event, video),
                }

        evidence = evidence_by_video.get(
            video_key, unavailable_evidence(video_key, "video not spotted in this run"))

        items.append(EvalItem(
            question_id=str(row.get("video_id", "")) or f"{dataset}_{len(items)}",
            question=question, group=group, route_detail=detail, video=video_key,
            subject=video.subject, reference_answer=str(row.get("answer", "")),
            evidence_text=_render_evidence(evidence, anchor or None),
            evidence_status=evidence.status, anchor=anchor, truth=truth,
        ))

    report = {
        "n_reference_rows": sum(counts.values()),
        "n_evaluated": len(items),
        "by_group": {k: v for k, v in counts.items() if v},
        "n_skipped": len(skipped),
        "skipped": skipped[:50],
        "unrouted_examples": unrouted_examples,
    }
    LOGGER.info("%s: routed %d/%d reference instruction(s) for evaluation",
                dataset, len(items), report["n_reference_rows"])
    return items, report


def _anchor_from_type_question(video: LongVideo, question: str) -> Dict[str, int]:
    from ..eval.megc_questions import event_ordinal

    ordinal = event_ordinal(question)
    if not ordinal or ordinal > len(video.events):
        return {}
    event = video.events[ordinal - 1]
    onset, offset = int(event.interval[0]), int(event.interval[1])
    apex = int(getattr(event, "apex", (onset + offset) // 2) or (onset + offset) // 2)
    return {"ordinal": ordinal, "onset": onset, "offset": offset, "apex": apex}


def _type_of_event(event: Any, video: LongVideo) -> str:
    micro = {tuple(e.interval) for e in video.micro_events()}
    return "micro-expression" if tuple(event.interval) in micro else "macro-expression"


def build_user_prompt(item: EvalItem) -> str:
    hint = _GROUP_HINTS.get(item.group, "")
    hint_block = f"\n{hint}\n" if hint else "\n"
    return (
        f"{item.evidence_text}\n\n"
        f"QUESTION: {item.question}\n{hint_block}\n"
        f"Reply with exactly this JSON object, with these key names and no others:\n"
        f"{_SKELETONS[item.group]}\n\n"
        "Output the JSON and nothing else."
    )


@dataclass
class EvalSamplingStats:
    n_calls: int = 0
    n_failed_calls: int = 0
    n_unparsable: int = 0
    total_latency_s: float = 0.0
    errors: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_calls": self.n_calls, "n_failed_calls": self.n_failed_calls,
            "n_unparsable": self.n_unparsable,
            "mean_latency_s": round(self.total_latency_s / self.n_calls, 2)
            if self.n_calls else 0.0,
            "errors": dict(self.errors),
        }


def sample_answers(
    items: Sequence[EvalItem],
    model: str = "claude-sonnet-5",
    max_tokens: int = 1024,
    timeout: int = 300,
    retries: int = 2,
    reasoning_effort: str = "",
    max_workers: int = 8,
    caller: Optional[Callable[..., Any]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> EvalSamplingStats:
    from .api_sampler import extract_product

    stats = EvalSamplingStats()
    lock = threading.Lock()
    call = caller or call_model
    done = 0

    def run(item: EvalItem) -> None:
        nonlocal done
        try:
            response = call(EVAL_SYSTEM_PROMPT, build_user_prompt(item), model=model,
                            max_tokens=max_tokens, timeout=timeout, retries=retries,
                            reasoning_effort=reasoning_effort or None)
            text = getattr(response, "text", str(response))
            latency = float(getattr(response, "latency_s", 0.0) or 0.0)
        except Exception as exc:
            with lock:
                stats.n_calls += 1
                stats.n_failed_calls += 1
                key = type(exc).__name__
                stats.errors[key] = stats.errors.get(key, 0) + 1
                done += 1
            item.error = f"{type(exc).__name__}: {exc}"
            return

        product = extract_product(text)
        with lock:
            stats.n_calls += 1
            stats.total_latency_s += latency
            if product is None:
                stats.n_unparsable += 1
            done += 1
            position = done
        item.raw_text = text
        if product is None:
            item.error = "no JSON object in the completion"
        else:
            item.product = product
        if progress:
            progress(position, len(items))

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        list(pool.map(run, items))
    return stats


def _pred_text(item: EvalItem) -> str:
    answer = item.product.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return item.raw_text.strip()


def _score_counts(items: Sequence[EvalItem]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    quantity_of = {GROUP_COUNT_EXPRESSION: "expression", GROUP_COUNT_MICRO: "micro",
                   GROUP_COUNT_MACRO: "macro"}
    pairs: Dict[str, List[Tuple[float, float]]] = {q: [] for q in mm.COUNT_QUANTITIES}
    failures: Dict[str, int] = {q: 0 for q in mm.COUNT_QUANTITIES}

    for item in items:
        quantity = quantity_of.get(item.group)
        if quantity is None:
            continue
        truth = parse_count(item.reference_answer)
        if truth is None:
            continue
        predicted = parse_count(item.product.get("count")) if item.parsed else None
        if predicted is None:
            predicted = parse_count(_pred_text(item)) if item.parsed else None
        if predicted is None:
            failures[quantity] += 1
            continue
        pairs[quantity].append((float(predicted), float(truth)))

    return mm.count_scores(pairs), {"unparsed_predictions": failures}


def _score_spotting(
    items: Sequence[EvalItem],
    evidence_by_video: Dict[str, VideoEvidence],
    videos: Sequence[LongVideo],
    iou_threshold: float,
) -> Dict[str, Any]:
    by_key = {v.video_key: v for v in videos}

    micro_rows: List[Dict[str, Any]] = []
    macro_rows: List[Dict[str, Any]] = []
    combined_rows: List[Dict[str, Any]] = []
    for video_key, evidence in sorted(evidence_by_video.items()):
        video = by_key.get(video_key)
        if video is None:
            continue
        micro_truth = [e for e in video.events
                       if _type_of_event(e, video) == "micro-expression"]
        macro_truth = [e for e in video.events
                       if _type_of_event(e, video) == "macro-expression"]
        micro_props = [(int(t_on), int(t_off))
                       for t_on, t_off, _apex, _peak in evidence.proposals]
        macro_props = [(int(a), int(b)) for a, b in evidence.macro_intervals]

        def row(proposals, p_type, truth_events):
            return {
                "video": video_key,
                "proposals": proposals,
                "proposal_types": [p_type] * len(proposals),
                "truth": [tuple(int(x) for x in e.interval) for e in truth_events],
                "truth_types": [_type_of_event(e, video) for e in truth_events],
                "truth_labels": [str(e.fine_label or "") for e in truth_events],
            }

        micro_rows.append(row(micro_props, "micro-expression", micro_truth))
        macro_rows.append(row(macro_props, "macro-expression", macro_truth))
        combined_rows.append({
            "video": video_key,
            "proposals": micro_props + macro_props,
            "proposal_types": (["micro-expression"] * len(micro_props)
                               + ["macro-expression"] * len(macro_props)),
            "truth": [tuple(int(x) for x in e.interval) for e in video.events],
            "truth_types": [_type_of_event(e, video) for e in video.events],
            "truth_labels": [str(e.fine_label or "") for e in video.events],
        })

    policy_by_scope: Dict[str, List[Dict[str, Any]]] = {}
    parse_failures: Dict[str, int] = {}
    for item in items:
        if item.group != GROUP_SPOT_INTERVAL:
            continue
        scope = localisation_scope(item.question)
        policy_by_scope.setdefault(scope, [])
        parse_failures.setdefault(scope, 0)

        truth = parse_intervals(item.reference_answer)
        if not item.parsed:
            parse_failures[scope] += 1
            predicted: List[Dict[str, Any]] = []
        else:
            predicted = parse_intervals(item.product.get("events")) or []
            if not predicted:
                predicted = parse_intervals(_pred_text(item))
        policy_by_scope[scope].append({
            "video": item.video,
            "proposals": [p["interval"] for p in predicted],
            "proposal_types": [p["type"] or "micro-expression" for p in predicted],
            "truth": [t["interval"] for t in truth],
            "truth_types": [t["type"] for t in truth],
        })

    n_micro_truth = sum(len(r["truth"]) for r in micro_rows)
    n_macro_truth = sum(len(r["truth"]) for r in macro_rows)
    engine = {
        "micro_expression": dict(
            mm.spotting_scores(micro_rows, iou_threshold=iou_threshold),
            note=("the spotter's micro-scale proposal set against micro-expression "
                  "ground truth -- the comparison the task statement defines"),
            n_proposals=sum(len(r["proposals"]) for r in micro_rows),
            n_truth=n_micro_truth),
        "macro_expression": dict(
            mm.spotting_scores(macro_rows, iou_threshold=iou_threshold),
            note="the engine's macro interval set against macro-expression ground truth",
            n_proposals=sum(len(r["proposals"]) for r in macro_rows),
            n_truth=n_macro_truth),
        "pooled_both_types": dict(
            mm.spotting_scores(combined_rows, iou_threshold=iou_threshold),
            note=("both proposal sets against every annotated event; reported for "
                  "completeness, but a type mismatch shows up here as a false positive"),
            n_truth=n_micro_truth + n_macro_truth),
        "headline": "micro_expression",
    }
    if not n_micro_truth:
        engine["micro_expression"]["coverage_warning"] = (
            "this video pool contains no annotated micro-expression event, so precision "
            "is charged against an empty truth set and recall is undefined")

    return {
        "engine_proposals": dict(
            engine,
            note=("the frozen stage-I/II proposal set scored against ground truth; no "
                  "language model is involved, and this is the number the task "
                  "statement defines"),
            n_videos=len(micro_rows)),
        "policy_localisation": {
            scope: dict(mm.spotting_scores(rows, iou_threshold=iou_threshold),
                        n_questions=len(rows),
                        unparsed_predictions=parse_failures.get(scope, 0))
            for scope, rows in sorted(policy_by_scope.items())
        },
        "note": ("engine_proposals and policy_localisation measure different objects and "
                 "are not interchangeable; see this module's docstring"),
    }


def _score_au(items: Sequence[EvalItem]) -> Dict[str, Any]:
    predicted: List[Sequence[str]] = []
    truth: List[Sequence[str]] = []
    unmapped_pred: Dict[str, int] = {}
    unmapped_truth: Dict[str, int] = {}
    parse_failures = 0

    for item in items:
        if item.group != GROUP_AU_SET:
            continue
        gold, gold_unmapped = normalise_au_set(item.reference_answer)
        for name in gold_unmapped:
            unmapped_truth[name] = unmapped_truth.get(name, 0) + 1
        if not item.parsed:
            parse_failures += 1
            predicted.append([])
            truth.append(gold)
            continue
        raw = item.product.get("action_units")
        if raw in (None, ""):
            raw = _pred_text(item)
        pred, pred_unmapped = normalise_au_set(raw)
        for name in pred_unmapped:
            unmapped_pred[name] = unmapped_pred.get(name, 0) + 1
        predicted.append(pred)
        truth.append(gold)

    scores = mm.au_scores(predicted, truth)
    scores["n_questions"] = len(truth)
    scores["unparsed_predictions"] = parse_failures
    scores["unmapped_names"] = {
        "in_predictions": dict(sorted(unmapped_pred.items(), key=lambda kv: -kv[1])),
        "in_reference": dict(sorted(unmapped_truth.items(), key=lambda kv: -kv[1])),
    }
    return scores


def _score_type_binary(items: Sequence[EvalItem]) -> Dict[str, Any]:
    pairs: List[Tuple[str, str]] = []
    parse_failures = 0
    no_truth = 0

    for item in items:
        if item.group != GROUP_TYPE_BINARY:
            continue
        gold = parse_expression_type(item.reference_answer)
        if not gold:
            no_truth += 1
            continue
        if not item.parsed:
            parse_failures += 1
            pairs.append((gold, ""))
            continue
        predicted = parse_expression_type(item.product.get("expression_type"))
        if not predicted:
            predicted = parse_expression_type(_pred_text(item))
        pairs.append((gold, predicted))

    if not pairs:
        return mm._unavailable("no expression-type question had a parseable reference",
                               n_questions=0)
    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    scores = dict(mm._emotion_pair(y_true, y_pred, mm.EXPRESSION_TYPES, "expression_type"))
    if "reg_uf1" in scores:
        scores["spot_uf1"] = scores.pop("reg_uf1")
        scores["spot_uar"] = scores.pop("reg_uar")
    scores["n_questions"] = len(pairs)
    scores["unparsed_predictions"] = parse_failures
    scores["n_without_reference_type"] = no_truth
    scores["classes"] = list(mm.EXPRESSION_TYPES)
    return scores


def _score_event_recognition(items: Sequence[EvalItem]) -> Dict[str, Any]:
    fine_true: List[str] = []
    fine_pred: List[str] = []
    candidates: List[str] = []
    references: List[str] = []
    parse_failures = 0
    no_truth = 0

    for item in items:
        if item.group != GROUP_EVENT_RECOGNITION:
            continue
        candidates.append(_pred_text(item))
        references.append(item.reference_answer)

        gold = str(item.truth.get("fine_label") or "")
        if not gold:
            no_truth += 1
            continue
        if not item.parsed:
            parse_failures += 1
            fine_true.append(gold)
            fine_pred.append("")
            continue
        fine_true.append(gold)
        fine_pred.append(str(item.product.get("fine_label") or ""))

    emotion = mm.recognition_scores(fine_true, fine_pred)
    text = mm.text_scores(candidates, references)
    return {
        "emotion": dict(emotion, n_scored=len(fine_true),
                        unparsed_predictions=parse_failures,
                        n_without_reference_label=no_truth),
        "text": dict(text, n_scored=len(candidates)),
        "n_questions": len(candidates),
    }


def _score_video_strs(
    items: Sequence[EvalItem],
    videos: Sequence[LongVideo],
    iou_threshold: float,
) -> Dict[str, Any]:
    by_key = {v.video_key: v for v in videos}
    rows: List[Dict[str, Any]] = []
    candidates: List[str] = []
    references: List[str] = []
    parse_failures = 0

    tp_correct = 0
    n_pred_labels = 0
    n_true_labels = 0

    for item in items:
        if item.group != GROUP_VIDEO_STRS:
            continue
        video = by_key.get(item.video)
        if video is None:
            continue
        candidates.append(_pred_text(item))
        references.append(item.reference_answer)

        truth_events = list(video.micro_events())
        truth = [tuple(int(x) for x in e.interval) for e in truth_events]
        n_true_labels += len(truth_events)

        claimed = item.product.get("events") if item.parsed else None
        if not item.parsed:
            parse_failures += 1
        claimed = claimed if isinstance(claimed, list) else []

        proposals: List[Tuple[int, int]] = []
        claimed_labels: List[str] = []
        for entry in claimed:
            if not isinstance(entry, dict):
                continue
            span = entry.get("interval") or entry.get("frames") or []
            if not (isinstance(span, (list, tuple)) and len(span) >= 2):
                continue
            try:
                interval = (int(span[0]), int(span[1]))
            except (TypeError, ValueError):
                continue
            proposals.append(interval)
            label, recognised = canonical_fine_label(
                str(entry.get("fine_label") or ""))
            claimed_labels.append(label if recognised else "")
        n_pred_labels += len(proposals)
        rows.append({
            "video": item.video,
            "proposals": proposals,
            "proposal_types": ["micro-expression"] * len(proposals),
            "proposal_labels": claimed_labels,
            "truth": truth,
            "truth_types": ["micro-expression"] * len(truth),
            "truth_labels": [str(e.fine_label or "") for e in truth_events],
        })

        for pred_i, truth_i, overlap in mm._greedy_pairs(proposals, truth,
                                                         iou_threshold):
            if overlap < iou_threshold:
                continue
            gold, _recognised = canonical_fine_label(
                str(truth_events[truth_i].fine_label or ""))
            if claimed_labels[pred_i] and gold and claimed_labels[pred_i] == gold:
                tp_correct += 1

    spotting = mm.spotting_scores(rows, iou_threshold=iou_threshold)
    f1_spot = float(spotting.get("headline_f1", 0.0) or 0.0)
    analysis = mm._prf(tp_correct, n_pred_labels, n_true_labels)
    f1_analysis = float(analysis.get("f1", 0.0) or 0.0)

    return {
        "spotting": dict(spotting, n_videos=len(rows)),
        "analysis": dict(analysis, note=("emotion correctness on the spotting true "
                                         "positives only (MEGC2025 sec. 2.5)")),
        "strs": {
            "score": round(mm.strs(f1_spot, f1_analysis), 6),
            "f1_spotting": round(f1_spot, 6),
            "f1_analysis": round(f1_analysis, 6),
            "formula": "STRS = F1_s * F1_a  (MEGC2025 eq. 1)",
        },
        "text": dict(mm.text_scores(candidates, references), n_scored=len(candidates)),
        "n_questions": len(candidates),
        "unparsed_predictions": parse_failures,
    }


def score_items(
    items: Sequence[EvalItem],
    videos: Sequence[LongVideo],
    evidence_by_video: Dict[str, VideoEvidence],
    iou_threshold: float = mm.DEFAULT_IOU_THRESHOLD,
) -> Dict[str, Any]:
    counting, counting_failures = _score_counts(items)
    return {
        "localisation": {
            "note": ("spotting and counting: TP at IoU >= %.2f against ground truth"
                     % iou_threshold),
            "interval": _score_spotting(items, evidence_by_video, videos, iou_threshold),
            "expression_type_unweighted": _score_type_binary(items),
            "counting": dict(counting, **counting_failures),
        },
        "recognition": {
            "note": ("description of the proposed segments: action units and the "
                     "coarse/fine emotion classes"),
            "action_units": _score_au(items),
            "event": _score_event_recognition(items),
        },
        "whole_video": {
            "note": ("spot-then-recognise over the entire video, including every "
                     "localised event and its emotion analysis"),
            **_score_video_strs(items, videos, iou_threshold),
        },
        "iou_threshold": iou_threshold,
    }


def by_subject(
    items: Sequence[EvalItem],
    videos: Sequence[LongVideo],
    evidence_by_video: Dict[str, VideoEvidence],
    iou_threshold: float = mm.DEFAULT_IOU_THRESHOLD,
) -> Dict[str, Any]:
    by_key = {v.video_key: v for v in videos}
    subjects: Dict[str, List[EvalItem]] = {}
    for item in items:
        subjects.setdefault(item.subject, []).append(item)

    out: Dict[str, Any] = {}
    for subject, subject_items in sorted(subjects.items()):
        keys = {i.video for i in subject_items}
        subject_videos = [by_key[k] for k in sorted(keys) if k in by_key]
        subject_evidence = {k: v for k, v in evidence_by_video.items() if k in keys}
        groups: Dict[str, int] = {}
        for item in subject_items:
            groups[item.group] = groups.get(item.group, 0) + 1
        out[subject] = {
            "n_questions": len(subject_items),
            "n_videos": len(subject_videos),
            "by_group": dict(sorted(groups.items())),
            "n_unparsed": sum(1 for i in subject_items if not i.parsed),
            "metrics": score_items(subject_items, subject_videos, subject_evidence,
                                  iou_threshold),
        }
    return out


def build_report(
    dataset: str,
    model: str,
    items: Sequence[EvalItem],
    videos: Sequence[LongVideo],
    evidence_by_video: Dict[str, VideoEvidence],
    routing: Dict[str, Any],
    perception: Dict[str, Any],
    sampling: Dict[str, Any],
    iou_threshold: float = mm.DEFAULT_IOU_THRESHOLD,
    include_per_subject: bool = True,
) -> Dict[str, Any]:
    parse_by_group: Dict[str, Dict[str, int]] = {}
    for item in items:
        row = parse_by_group.setdefault(item.group, {"n": 0, "unparsed": 0})
        row["n"] += 1
        if not item.parsed:
            row["unparsed"] += 1

    report: Dict[str, Any] = {
        "dataset": dataset,
        "policy_model": model,
        "protocol": {
            "channel": "megc-evaluation",
            "draws_per_question": 1,
            "iou_threshold": iou_threshold,
            "pool": ("every reference instruction, independent of RL admission -- the "
                     "counting, AU-inventory and event-type metrics are defined on the "
                     "questions augmentation excludes"),
            "sources": ["MEGC2025.md sec. 2.3/2.5 and eq. (1)-(3)",
                        "MEGC2026 sec. C", "MEGC2024.md"],
        },
        "routing": routing,
        "perception": perception,
        "sampling": dict(sampling, parse_rate_by_group=parse_by_group),
        "metrics": score_items(items, videos, evidence_by_video, iou_threshold),
    }
    if include_per_subject:
        report["per_subject"] = by_subject(items, videos, evidence_by_video,
                                          iou_threshold)
    return report


def write_predictions(items: Sequence[EvalItem], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({
                "question_id": item.question_id,
                "video": item.video,
                "subject": item.subject,
                "group": item.group,
                "route_detail": item.route_detail,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "predicted_product": item.product,
                "predicted_text": _pred_text(item) if item.parsed else "",
                "raw_completion": item.raw_text,
                "parsed": item.parsed,
                "error": item.error,
                "evidence_status": item.evidence_status,
                "truth": item.truth,
            }, ensure_ascii=False, default=str) + "\n")
    LOGGER.info("wrote %d prediction record(s) to %s", len(items), path)
    return path


def write_report(report: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    LOGGER.info("wrote MEGC metrics to %s", path)
    return path


def run_evaluation(
    dataset: str,
    videos: Sequence[LongVideo],
    qa_rows: Sequence[Dict[str, Any]],
    output: Path,
    config: Optional[MEWMConfig] = None,
    model: str = "claude-sonnet-5",
    evidence_by_video: Optional[Dict[str, VideoEvidence]] = None,
    cache_root: Optional[Path] = None,
    max_frames: int = 0,
    stride: int = 1,
    max_workers: int = 8,
    reasoning_effort: str = "",
    iou_threshold: float = mm.DEFAULT_IOU_THRESHOLD,
    max_questions: int = 0,
    caller: Optional[Callable[..., Any]] = None,
    include_per_subject: bool = True,
) -> Dict[str, Any]:
    config = config or load_config()
    started = time.time()

    if evidence_by_video is None:
        from .perception_cache import perceive_cached

        def perception_progress(key: str, position: int, total: int) -> None:
            if position == 1 or position % 10 == 0 or position == total:
                LOGGER.info("perceiving %d/%d (%s)", position, total, key)

        evidence_by_video, perception = perceive_cached(
            videos, cache_root, config, max_frames, stride, perception_progress)
    else:
        perception = {
            "n_videos": len(videos),
            "n_with_evidence": sum(1 for e in evidence_by_video.values() if e.available),
            "reused": "handed over by the caller; not recomputed",
        }

    items, routing = build_eval_items(dataset, videos, qa_rows, evidence_by_video)
    if max_questions and len(items) > max_questions:
        routing["truncated_to"] = max_questions
        routing["truncation_note"] = (
            "max_questions was set: this report covers a prefix of the routed pool and "
            "its metrics are NOT comparable with a full run")
        items = list(items[:max_questions])

    def sampling_progress(position: int, total: int) -> None:
        if position == 1 or position % 25 == 0 or position == total:
            LOGGER.info("sampled %d/%d answers", position, total)

    stats = sample_answers(items, model=model, max_workers=max_workers,
                           reasoning_effort=reasoning_effort, caller=caller,
                           progress=sampling_progress)

    report = build_report(dataset, model, items, videos, evidence_by_video, routing,
                          perception, stats.to_dict(), iou_threshold,
                          include_per_subject)
    report["elapsed_s"] = round(time.time() - started, 1)
    output = Path(output)
    predictions = output.with_name(output.stem + "_predictions.jsonl")
    write_predictions(items, predictions)
    report["predictions_file"] = str(predictions)
    write_report(report, output)
    return report


__all__ = [
    "EVAL_SYSTEM_PROMPT", "EvalItem", "EvalSamplingStats", "build_eval_items",
    "build_user_prompt", "sample_answers", "score_items", "by_subject", "build_report",
    "write_predictions", "write_report", "run_evaluation",
]
