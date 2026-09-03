"""Routing reference questions to MEGC metric groups, and parsing both sides.

**The RL-eligible pool and the metric pool are different sets, and the difference is the
majority.** ``mewm.training.rl_prompts`` deliberately refuses counting, AU-inventory and
event-type questions: their answers are fixed by the annotation, so there is no reasoning
to reward. But MAE/RMSE, F1_AU/Jaccard_AU and the ME/MaE unweighted pair are defined on
exactly those questions. Evaluating only the RL-eligible pool would leave most of the
required table unavailable, so routing here covers every reference question and is
independent of admission.

**The reference answers name action units in FACS prose, not in codes.** A reference AU
inventory reads ``"brow lowerer, upper lip raiser, dimpler"``; this codebase's
``AU_ANATOMY`` calls the same units ``"brow draw-down"`` and ``"upper-lip raise"``; a
policy will produce ``"AU4"`` or an invented paraphrase. Comparing raw strings would score
F1_AU at approximately zero and the number would look like a model failure. The mapping
below therefore carries the official FACS names, the codebase's paraphrases and the bare
codes, and -- crucially -- ``normalise_au_set`` returns the names it could *not* map so an
unmapped vocabulary shows up as a reported count instead of a silently deflated score.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..knowledge.au_anatomy import AU_ANATOMY

# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------

#: MAE / RMSE over the total number of expression events.
GROUP_COUNT_EXPRESSION = "count_expression"
#: MAE / RMSE over the number of micro-expression events.
GROUP_COUNT_MICRO = "count_micro"
#: MAE / RMSE over the number of macro-expression events.
GROUP_COUNT_MACRO = "count_macro"
#: TP/FP/FN at IoU >= 0.5, precision/recall/F1, and the ME/MaE unweighted pair.
GROUP_SPOT_INTERVAL = "spot_interval"
#: F1_AU and Jaccard_AU over the video's AU inventory.
GROUP_AU_SET = "au_set"
#: MEGC2026 sec. C's binary ME-vs-MaE task: UF1 and UAR.
GROUP_TYPE_BINARY = "type_binary"
#: RegUF1 / RegUAR over emotion classes, plus BLEU / ROUGE-1 on the description.
GROUP_EVENT_RECOGNITION = "event_recognition"
#: STRS, plus BLEU / ROUGE-1 on the whole-video narrative.
GROUP_VIDEO_STRS = "video_strs"
#: A question whose template this module does not recognise.
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
    """Return ``(group, detail)`` for one reference question.

    Order matters: the localisation questions share the counting stem
    ("How many ... ? Localize every ..."), so the localisation test must come first or
    every one of them would be filed as a pure count.
    """
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
    """``"micro"`` / ``"macro"`` / ``"expression"`` for a localisation question."""
    text = question or ""
    if "micro-expression event" in text:
        return "micro"
    if "macro-expression event" in text:
        return "macro"
    return "expression"


def event_ordinal(question: str) -> Optional[int]:
    """The 1-based event ordinal a type or event question names."""
    match = _TYPE_QUESTION.search(question or "") or _EVENT_ANCHOR.search(question or "")
    return int(match.group(1)) if match else None


def event_anchor(question: str) -> Optional[Dict[str, int]]:
    """``{"ordinal", "onset", "offset", "apex"}`` for an event-anchored question."""
    match = _EVENT_ANCHOR.search(question or "")
    if not match:
        return None
    return {"ordinal": int(match.group(1)), "onset": int(match.group(2)),
            "offset": int(match.group(3)), "apex": int(match.group(4))}


# ---------------------------------------------------------------------------
# Action units
# ---------------------------------------------------------------------------

#: Official FACS names, as the reference corpus writes them. Every one of the 26 distinct
#: names in the casme_sq reference AU inventories is here; the table is grounded in that
#: inventory rather than guessed, and ``normalise_au_set`` reports anything outside it.
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

#: The paraphrases this codebase uses for the same units. Kept separate from the FACS
#: table so a future edit to ``AU_ANATOMY`` cannot silently shadow an official name.
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
    # A parenthesised gloss is common on the policy side: "AU4 (brow lowerer)".
    inner = re.match(r"^(au[\s_-]?\d{1,2})\b", text)
    if inner:
        return _normalise_one_au(inner.group(1))
    for name, code in _FACS_NAMES.items():
        if name in text:
            return code
    return None


def normalise_au_set(value: Any) -> Tuple[List[str], List[str]]:
    """Return ``(codes, unmapped)`` for an AU inventory given as text or a list.

    ``unmapped`` is returned rather than discarded because an unmapped vocabulary and a
    genuinely wrong prediction are different failures, and a set metric cannot tell them
    apart on its own.
    """
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


# ---------------------------------------------------------------------------
# Parsing answers
# ---------------------------------------------------------------------------

_LEADING_INT = re.compile(r"-?\d+")
_INTERVAL = re.compile(
    r"(\d+)-th\s+(micro-expression|macro-expression)\s*:\s*frames\s+(\d+)-(\d+)",
    re.IGNORECASE)


def parse_count(answer: Any) -> Optional[int]:
    """The first integer in an answer, or ``None``.

    ``None`` and ``0`` must stay distinct: "0 micro-expression events" is a correct claim
    that contributes an error of 0, while an unparseable answer contributes nothing and is
    counted as a parse failure. Collapsing them would reward silence.
    """
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return int(answer)
    match = _LEADING_INT.search(str(answer or ""))
    return int(match.group()) if match else None


def parse_intervals(answer: Any) -> List[Dict[str, Any]]:
    """Frame intervals and their ME/MaE type from a localisation answer.

    Accepts the reference prose form
    ``"2 expression events. 1-th macro-expression: frames 557-608 (apex 572), ..."``
    and a JSON-ish list of ``{"interval": [a, b], "type": ...}`` from a policy.
    """
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
    """``"micro-expression"`` / ``"macro-expression"`` / ``""``.

    Micro is tested first: "macro-expression" does not contain "micro-expression", but a
    sentence naming both is answered by whichever appears -- and a policy hedging with
    "not a macro-expression but a micro-expression" should read as micro.
    """
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
