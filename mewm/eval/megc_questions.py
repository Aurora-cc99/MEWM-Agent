"""MEGC-style QA question templates and label-mapping utilities."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..knowledge.au_anatomy import AU_ANATOMY

GROUP_COUNT_EXPRESSION = "count_expression"
GROUP_COUNT_MICRO = "count_micro"
GROUP_COUNT_MACRO = "count_macro"
GROUP_SPOT_INTERVAL = "spot_interval"
GROUP_AU_SET = "au_set"
GROUP_TYPE_BINARY = "type_binary"
GROUP_EVENT_RECOGNITION = "event_recognition"
GROUP_VIDEO_STRS = "video_strs"
GROUP_UNROUTED = "unrouted"

ALL_GROUPS: Tuple[str, ...] = (
    GROUP_COUNT_EXPRESSION, GROUP_COUNT_MICRO, GROUP_COUNT_MACRO,
    GROUP_SPOT_INTERVAL, GROUP_AU_SET, GROUP_TYPE_BINARY,
    GROUP_EVENT_RECOGNITION, GROUP_VIDEO_STRS,
)

_EVENT_ANCHOR = re.compile(
    r"In the (\d+)-th expression event of this video "
    r"\(frames (\d+)-(\d+), apex (\d+)\)", re.IGNORECASE)
_TYPE_QUESTION = re.compile(
    r"What is the expression type of the (\d+)-th expression event", re.IGNORECASE)


def route_question(question: str) -> Tuple[str, str]:
    text = (question or "").strip()
    if not text:
        return GROUP_UNROUTED, "empty question"

    if "Localize every" in text:
        if "micro-expression event" in text:
            scope = "micro"
        elif "macro-expression event" in text:
            scope = "macro"
        else:
            scope = "expression"
        return GROUP_SPOT_INTERVAL, "localisation, scope=%s" % scope

    if text.startswith("How many expression events"):
        return GROUP_COUNT_EXPRESSION, "count of all expression events"
    if text.startswith("How many micro-expression events"):
        return GROUP_COUNT_MICRO, "count of micro-expression events"
    if text.startswith("How many macro-expression events"):
        return GROUP_COUNT_MACRO, "count of macro-expression events"
    if text.startswith("What distinct action units"):
        return GROUP_AU_SET, "AU inventory of the whole video"
    if _TYPE_QUESTION.search(text):
        return GROUP_TYPE_BINARY, "ME vs MaE for one named event"
    if _EVENT_ANCHOR.search(text):
        return GROUP_EVENT_RECOGNITION, "free-form description of one named event"
    if text.startswith("Reason over the whole video"):
        return GROUP_VIDEO_STRS, "whole-video spot-then-recognise narrative"

    return GROUP_UNROUTED, "unrecognised template"


def localisation_scope(question: str) -> str:
    text = question or ""
    if "micro-expression event" in text:
        return "micro"
    if "macro-expression event" in text:
        return "macro"
    return "expression"


def event_ordinal(question: str) -> Optional[int]:
    match = _TYPE_QUESTION.search(question or "") or _EVENT_ANCHOR.search(question or "")
    return int(match.group(1)) if match else None


def event_anchor(question: str) -> Optional[Dict[str, int]]:
    match = _EVENT_ANCHOR.search(question or "")
    if not match:
        return None
    return {"ordinal": int(match.group(1)), "onset": int(match.group(2)),
            "offset": int(match.group(3)), "apex": int(match.group(4))}


_FACS_NAMES: Dict[str, str] = {
    "inner brow raiser": "AU1",
    "outer brow raiser": "AU2",
    "brow lowerer": "AU4",
    "upper lid raiser": "AU5",
    "cheek raiser": "AU6",
    "lid tightener": "AU7",
    "nose wrinkler": "AU9",
    "upper lip raiser": "AU10",
    "nasolabial deepener": "AU11",
    "lip corner puller": "AU12",
    "sharp lip puller": "AU13",
    "dimpler": "AU14",
    "lip corner depressor": "AU15",
    "lower lip depressor": "AU16",
    "chin raiser": "AU17",
    "lip pucker": "AU18",
    "lip stretcher": "AU20",
    "lip funneler": "AU22",
    "lip tightener": "AU23",
    "lip pressor": "AU24",
    "lips part": "AU25",
    "jaw drop": "AU26",
    "mouth stretch": "AU27",
    "lip suck": "AU28",
    "jaw sideways": "AU30",
    "cheek puff": "AU33",
    "nostril dilator": "AU38",
    "nostril compressor": "AU39",
    "eye closure": "AU43",
    "blink": "AU45",
}

_CODEBASE_NAMES: Dict[str, str] = {
    str(name).strip().lower(): code for code, name in AU_ANATOMY.items()
}

_AU_CODE = re.compile(r"^au[\s_-]?(\d{1,2})[a-z]?$", re.IGNORECASE)


def _normalise_one_au(token: str) -> Optional[str]:
    text = (token or "").strip().lower().strip(".;:")
    if not text:
        return None
    match = _AU_CODE.match(text)
    if match:
        return "AU%d" % int(match.group(1))
    if text in _FACS_NAMES:
        return _FACS_NAMES[text]
    if text in _CODEBASE_NAMES:
        return _CODEBASE_NAMES[text]
    inner = re.match(r"^(au[\s_-]?\d{1,2})\b", text)
    if inner:
        return _normalise_one_au(inner.group(1))
    for name, code in _FACS_NAMES.items():
        if name in text:
            return code
    return None


def normalise_au_set(value: Any) -> Tuple[List[str], List[str]]:
    if value is None:
        return [], []
    if isinstance(value, (list, tuple, set)):
        tokens = [str(v) for v in value]
    else:
        tokens = re.split(r"[,;/]|\band\b", str(value))
    codes: Set[str] = set()
    unmapped: List[str] = []
    for token in tokens:
        text = token.strip()
        if not text:
            continue
        code = _normalise_one_au(text)
        if code:
            codes.add(code)
        else:
            unmapped.append(text)
    return sorted(codes, key=lambda c: int(c[2:])), unmapped


_LEADING_INT = re.compile(r"-?\d+")
_INTERVAL = re.compile(
    r"(\d+)-th\s+(micro-expression|macro-expression)\s*:\s*frames\s+(\d+)-(\d+)",
    re.IGNORECASE)


def parse_count(answer: Any) -> Optional[int]:
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return int(answer)
    match = _LEADING_INT.search(str(answer or ""))
    return int(match.group()) if match else None


def parse_intervals(answer: Any) -> List[Dict[str, Any]]:
    if isinstance(answer, (list, tuple)):
        out: List[Dict[str, Any]] = []
        for item in answer:
            if isinstance(item, dict):
                span = item.get("interval") or item.get("frames") or []
                if isinstance(span, (list, tuple)) and len(span) >= 2:
                    out.append({
                        "interval": (int(span[0]), int(span[1])),
                        "type": str(item.get("type") or item.get("expression_type")
                                    or "").strip().lower(),
                    })
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append({"interval": (int(item[0]), int(item[1])), "type": ""})
        return out
    return [
        {"interval": (int(m.group(3)), int(m.group(4))), "type": m.group(2).lower()}
        for m in _INTERVAL.finditer(str(answer or ""))
    ]


def parse_expression_type(answer: Any) -> str:
    text = str(answer or "").strip().lower()
    if not text:
        return ""
    if "micro-expression" in text or re.search(r"\bmicro\b", text):
        return "micro-expression"
    if "macro-expression" in text or re.search(r"\bmacro\b", text):
        return "macro-expression"
    return ""


__all__ = [
    "GROUP_COUNT_EXPRESSION", "GROUP_COUNT_MICRO", "GROUP_COUNT_MACRO",
    "GROUP_SPOT_INTERVAL", "GROUP_AU_SET", "GROUP_TYPE_BINARY",
    "GROUP_EVENT_RECOGNITION", "GROUP_VIDEO_STRS", "GROUP_UNROUTED", "ALL_GROUPS",
    "route_question", "localisation_scope", "event_ordinal", "event_anchor",
    "normalise_au_set", "parse_count", "parse_intervals", "parse_expression_type",
]
