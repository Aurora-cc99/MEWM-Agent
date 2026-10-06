"""Interactive QA interrogation utility for inspecting model answers per clip."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.qa_loader import QAItem, classify_question, parse_segment_question
from ..knowledge.emotion_prototypes import canonical_fine_label
from ..orchestration.existence_gate import (
    ExistenceDecision, decide_existence, find_existence_question,
    split_items_by_gate,
)

LOGGER = logging.getLogger(__name__)

_COUNT_RE = re.compile(r"\b(\d+)\b")
_AU_CODE_RE = re.compile(r"\bAU\s*(\d{1,2})\b", re.IGNORECASE)
_AU_NAMES = {
    "inner brow raiser": "AU1", "outer brow raiser": "AU2",
    "brow lowerer": "AU4", "upper lid raiser": "AU5",
    "cheek raiser": "AU6", "lid tightener": "AU7",
    "nose wrinkler": "AU9", "upper lip raiser": "AU10",
    "lip corner puller": "AU12", "dimpler": "AU14",
    "lip corner depressor": "AU15", "lip presser": "AU24",
    "lip tightener": "AU23", "lip suck": "AU28",
    "jaw drop": "AU26", "mouth stretch": "AU27",
}


def load_jsonl_qa(path: Path | str) -> Dict[str, List[QAItem]]:
    by_video: Dict[str, List[QAItem]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        item = QAItem(
            video_id=str(row.get("video_id", "")),
            video=str(row.get("video", "")),
            question=str(row.get("question", "")),
            answer=row.get("answer"),
            qtype=classify_question(str(row.get("question", ""))),
        )
        by_video.setdefault(item.video, []).append(item)
    return by_video


def extract_count(text: str) -> Optional[int]:
    match = _COUNT_RE.search(text or "")
    return int(match.group(1)) if match else None


def extract_aus(text: str) -> List[str]:
    codes = [f"AU{m.group(1)}" for m in _AU_CODE_RE.finditer(text or "")]
    lowered = (text or "").lower()
    for name, code in _AU_NAMES.items():
        if name in lowered and code not in codes:
            codes.append(code)
    return codes


def extract_emotion(text: str) -> Optional[str]:
    emotion, recognised = canonical_fine_label(text or "")
    return emotion if recognised else None


@dataclass
class Prediction:
    question: str
    qtype: str
    predicted: str
    latency_s: float
    segment: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qtype": self.qtype, "question": self.question,
            "predicted": self.predicted, "latency_s": round(self.latency_s, 2),
            "segment": self.segment,
        }


def interrogate_video(
    video: Any,
    items: Sequence[QAItem],
    call: Any,
    model: str,
    reasoning_effort: str = "high",
    attach_frames: bool = True,
) -> List[Prediction]:
    predictions: List[Prediction] = []
    for item in items:
        segment = parse_segment_question(item.question)
        image_paths: List[str] = []
        if attach_frames and segment:
            for t in (segment["onset"], segment["apex"], segment["offset"]):
                path = video.paths.frame(t)
                if path.is_file():
                    image_paths.append(str(path))
        user_prompt = (
            f"Video: {video.video_key}\n\n"
            f"Question: {item.question}\n\n"
            "Answer the question directly, based on the video (frames are attached "
            "where the question names an event interval)."
        )
        started = time.time()
        try:
            response = call(
                "You are an attentive micro-expression analyst. Answer briefly and "
                "precisely; state numbers and AU codes explicitly.",
                user_prompt,
                image_paths=image_paths or None,
                model=model,
                max_tokens=1024,
                reasoning_effort=reasoning_effort,
            )
            text = response.text
        except Exception as exc:
            LOGGER.warning("question failed for %s: %s", video.video_key, exc)
            text = f"__FAILED__: {type(exc).__name__}"
        predictions.append(Prediction(
            question=item.question, qtype=item.qtype, predicted=text,
            latency_s=time.time() - started, segment=segment,
        ))
    return predictions


def integrate(
    items: Sequence[QAItem],
    predictions: Sequence[Prediction],
) -> Dict[str, Any]:
    count_answers = []
    count_preds = []
    au_refs: Dict[str, List[str]] = {}
    au_preds: Dict[str, List[str]] = {}
    emotion_refs: Dict[str, List[Optional[str]]] = {}
    emotion_preds: Dict[str, List[Optional[str]]] = {}

    for item, pred in zip(items, predictions):
        key = str(pred.segment["index"]) if pred.segment else item.video_id
        if item.qtype in ("count_micro", "count_expression", "count_macro"):
            ref = item.answer
            try:
                ref = int(ref)
            except (TypeError, ValueError):
                ref = None
            count_answers.append(ref)
            count_preds.append(extract_count(pred.predicted))
        elif item.qtype == "au_set":
            au_refs.setdefault(key, []).extend(extract_aus(str(item.answer)))
            au_preds.setdefault(key, []).extend(extract_aus(pred.predicted))
        elif item.qtype in ("event_type", "segment_analysis", "reason_full",
                            "localize_micro", "localize_expression"):
            au_refs.setdefault(key, []).extend(extract_aus(str(item.answer)))
            au_preds.setdefault(key, []).extend(extract_aus(pred.predicted))
            emotion_refs.setdefault(key, []).append(
                extract_emotion(str(item.answer)))
            emotion_preds.setdefault(key, []).append(
                extract_emotion(pred.predicted))

    def vote(candidates: Sequence[Optional[int]]) -> Optional[int]:
        clean = [c for c in candidates if c is not None]
        if not clean:
            return None
        return max(set(clean), key=clean.count)

    def vote_emotion(candidates: Sequence[Optional[str]]) -> Optional[str]:
        clean = [c for c in candidates if c]
        if not clean:
            return None
        return max(set(clean), key=clean.count)

    integrated: Dict[str, Any] = {
        "n_questions": len(predictions),
        "n_failed": sum(1 for p in predictions if p.predicted.startswith("__FAILED__")),
    }

    count_pred_vote = vote(count_preds)
    if count_answers:
        ref_vote = vote(count_answers)
        integrated["count"] = {
            "predicted": count_pred_vote,
            "reference": ref_vote,
            "per_question_predicted": count_preds,
            "per_question_reference": count_answers,
        }
        integrated["count_error"] = (
            abs(count_pred_vote - ref_vote)
            if count_pred_vote is not None and ref_vote is not None else None)

    au_rows = []
    for key in sorted(set(au_refs) | set(au_preds)):
        refs = sorted(set(au_refs.get(key, [])))
        preds = sorted(set(au_preds.get(key, [])))
        inter = len(set(refs) & set(preds))
        f1 = (2 * inter / (len(refs) + len(preds))) if (refs or preds) else 1.0
        au_rows.append({"key": str(key), "reference_aus": refs,
                        "predicted_aus": preds, "au_f1": round(f1, 4)})
    if au_rows:
        integrated["au"] = au_rows
        integrated["au_f1_mean"] = round(
            sum(r["au_f1"] for r in au_rows) / len(au_rows), 4)

    emo_rows = []
    for key in sorted(set(emotion_refs) | set(emotion_preds)):
        refs = emotion_refs.get(key, [])
        preds = emotion_preds.get(key, [])
        pred_vote = vote_emotion(preds)
        ref_vote = vote_emotion(refs)
        emo_rows.append({
            "key": str(key), "predicted": pred_vote, "reference": ref_vote,
            "correct": bool(pred_vote) and pred_vote == ref_vote,
            "per_question_predicted": preds, "per_question_reference": refs,
        })
    if emo_rows:
        integrated["emotion"] = emo_rows
        integrated["emotion_accuracy"] = round(
            sum(1 for r in emo_rows if r["correct"]) / len(emo_rows), 4)

    return integrated


@dataclass
class QARecord:

    question_index: int
    question: str
    qtype: str
    gated: bool
    predicted: Optional[str]
    reference: Any
    latency_s: Optional[float] = None
    segment: Optional[Dict[str, int]] = None
    skipped_reason: Optional[str] = None
    existence_decision: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_index": self.question_index, "question": self.question,
            "question_type": self.qtype, "gated": self.gated,
            "answer": self.predicted, "reference_answer": self.reference,
            "latency_s": (round(self.latency_s, 2) if self.latency_s is not None else None),
            "segment": self.segment, "skipped_reason": self.skipped_reason,
            "existence_decision": self.existence_decision,
        }


def interrogate_video_gated(
    video: Any,
    items: Sequence[QAItem],
    call: Any,
    model: str,
    n_tp_proposals: int,
    reasoning_effort: str = "high",
    attach_frames: bool = True,
    gate_on_existence: bool = True,
) -> Tuple[List[QARecord], ExistenceDecision, List[QAItem], List[Prediction]]:
    items = list(items)
    existence_item = find_existence_question(items) if gate_on_existence else None

    if existence_item is None:
        predictions = interrogate_video(video, items, call, model, reasoning_effort,
                                        attach_frames)
        decision = ExistenceDecision(exists=True, source="ungated",
                                     n_tp_proposals=n_tp_proposals,
                                     model_answer_count=None, existence_item=None)
        records = [
            QARecord(i, it.question, it.qtype, False, p.predicted, it.answer,
                     p.latency_s, p.segment)
            for i, (it, p) in enumerate(zip(items, predictions))
        ]
        return records, decision, items, predictions

    existence_prediction = interrogate_video(
        video, [existence_item], call, model, reasoning_effort, attach_frames)[0]
    decision = decide_existence(
        n_tp_proposals=n_tp_proposals,
        model_answer_text=existence_prediction.predicted,
        existence_item=existence_item,
        extract_count_fn=extract_count,
    )

    to_ask, to_skip = split_items_by_gate(items, decision)
    to_ask_rest = [it for it in to_ask if it is not existence_item]
    rest_predictions = (
        interrogate_video(video, to_ask_rest, call, model, reasoning_effort,
                          attach_frames)
        if to_ask_rest else []
    )
    predicted_by_id = {id(it): p for it, p in zip(to_ask_rest, rest_predictions)}

    records: List[QARecord] = []
    asked_items: List[QAItem] = [existence_item]
    asked_predictions: List[Prediction] = [existence_prediction]
    for index, item in enumerate(items):
        if item is existence_item:
            records.append(QARecord(
                index, item.question, item.qtype, False,
                existence_prediction.predicted, item.answer,
                existence_prediction.latency_s, existence_prediction.segment,
                existence_decision=decision.to_dict(),
            ))
        elif id(item) in predicted_by_id:
            pred = predicted_by_id[id(item)]
            records.append(QARecord(
                index, item.question, item.qtype, False, pred.predicted,
                item.answer, pred.latency_s, pred.segment,
            ))
            asked_items.append(item)
            asked_predictions.append(pred)
        else:
            records.append(QARecord(
                index, item.question, item.qtype, True, None, item.answer,
                skipped_reason="no_micro_expression_detected",
                existence_decision=decision.to_dict(),
            ))
    return records, decision, asked_items, asked_predictions


__all__ = ["load_jsonl_qa", "extract_count", "extract_aus", "extract_emotion",
           "Prediction", "interrogate_video", "integrate",
           "QARecord", "interrogate_video_gated"]
