"""Existence gate: filters ME candidates before multi-agent reasoning."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.qa_loader import QAItem

EXISTENCE_QTYPES: Tuple[str, ...] = ("count_micro", "count_expression", "count_macro")

SKIP_REASON_NOT_FOUND = "no_micro_expression_detected"


@dataclass
class ExistenceDecision:
    exists: bool
    source: str
    n_tp_proposals: int
    model_answer_count: Optional[int]
    existence_item: Optional[QAItem]
    existence_predicted_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exists": self.exists,
            "source": self.source,
            "n_tp_proposals": self.n_tp_proposals,
            "model_answer_count": self.model_answer_count,
            "existence_question": self.existence_item.question if self.existence_item else None,
        }


def find_existence_question(items: Sequence[QAItem]) -> Optional[QAItem]:
    for item in items:
        if item.qtype in EXISTENCE_QTYPES:
            return item
    return None


def decide_existence(
    n_tp_proposals: int,
    model_answer_text: Optional[str] = None,
    existence_item: Optional[QAItem] = None,
    extract_count_fn: Optional[Any] = None,
) -> ExistenceDecision:
    model_count: Optional[int] = None
    if model_answer_text is not None and extract_count_fn is not None:
        model_count = extract_count_fn(model_answer_text)

    tp_says_exists = n_tp_proposals > 0
    model_says_exists = bool(model_count) and model_count > 0

    if tp_says_exists and model_says_exists:
        exists, source = True, "both"
    elif tp_says_exists and not model_says_exists:
        exists, source = True, "tp_proposals"
    elif model_says_exists and not tp_says_exists:
        exists, source = True, "model_answer"
    else:
        exists, source = False, "neither"

    return ExistenceDecision(
        exists=exists, source=source, n_tp_proposals=n_tp_proposals,
        model_answer_count=model_count, existence_item=existence_item,
        existence_predicted_text=model_answer_text,
    )


def split_items_by_gate(
    items: Sequence[QAItem],
    decision: ExistenceDecision,
) -> Tuple[List[QAItem], List[QAItem]]:
    to_ask: List[QAItem] = []
    to_skip: List[QAItem] = []
    for item in items:
        if decision.existence_item is not None and item is decision.existence_item:
            to_ask.append(item)
            continue
        if decision.exists:
            to_ask.append(item)
        else:
            to_skip.append(item)
    return to_ask, to_skip


def skipped_record(item: QAItem, decision: ExistenceDecision) -> Dict[str, Any]:
    return {
        "question": item.question,
        "question_type": item.qtype,
        "gated": True,
        "answer": None,
        "reference_answer": item.answer,
        "skipped_reason": SKIP_REASON_NOT_FOUND,
        "existence_decision": decision.to_dict(),
    }


__all__ = [
    "EXISTENCE_QTYPES", "SKIP_REASON_NOT_FOUND", "ExistenceDecision",
    "find_existence_question", "decide_existence", "split_items_by_gate",
    "skipped_record",
]
