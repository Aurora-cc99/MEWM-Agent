"""QA "存在性优先"门控（formwork.md 第 IV 条，`MEWM-Agent_完整执行方案.md` 第 4.2 节）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..data.qa_loader import QAItem

#: 计数类问题都可以充当"是否存在微表情"的代理问题；QA 构建脚本本身已经把
#: ``count_micro``（"How many micro-expression events appear in this video?"）放在每
#: 个视频问题列表的第一条（见 ``build_me_lvqa_dataset.py`` 的写出顺序），这里不假设顺
#: 序、而是显式按类型查找，防止某个数据集/某次构建改变了顺序时门控静默失效。
EXISTENCE_QTYPES: Tuple[str, ...] = ("count_micro", "count_expression", "count_macro")

#: 记录在逐题落盘 jsonl 里的跳过原因（`mewm/eval/subject_report.py` 直接消费这个字符串）。
SKIP_REASON_NOT_FOUND = "no_micro_expression_detected"


@dataclass
class ExistenceDecision:
    """第一类问题的门控裁决结果，供 :func:`split_items_by_gate` 与落盘记录复用。"""

    exists: bool
    #: 判据来源："tp_proposals"（双分支并集有 TP）｜"model_answer"（模型自己数出 >0）｜
    #: "both"（两者一致为存在）｜"neither"（两者都判不存在）。
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
    """QA 列表里第一条计数类问题；找不到则返回 ``None``（视频的 QA 集合不含计数问题）。"""
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
    """合并两路存在性信号（第 3.2.6 节并集 TP、模型自己的计数回答）给出裁决。

    ``extract_count_fn`` 复用 ``mewm.qa.interrogate.extract_count``（避免重复实现同一个
    "从自然语言里抠出一个整数"的解析逻辑——这是 formwork.md 反复强调的"同一把尺子"原则）。
    """
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
    """把问题列表拆成"要问的"和"因门控被跳过的"两组（存在性问题本身永远在"要问"组）。"""
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
    """一条"因门控被跳过"的问答记录，字段与 ``subject_report`` 的 jsonl schema 对齐。"""
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
