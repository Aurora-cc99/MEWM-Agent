"""Memory retrieval: token-budget-aware query and re-ranking over stored states."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import MemoryConfig
from ..schemas import AU_PATTERN, EMOTION_LEXICON, EvidenceLevel
from .store import CaseEntry, PrecedentEntry, SemanticMemory, jaccard

LOGGER = logging.getLogger(__name__)

QUERY_TYPES = ("support", "confusion", "counterexample", "precedent")


@dataclass
class RetrievalRequest:
    query_type: str
    au_signature: Sequence[str]
    level: EvidenceLevel
    dataset: str = ""
    emotion: str = ""
    limit: int = 2
    exclude: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.query_type not in QUERY_TYPES:
            raise ValueError(f"unknown query type {self.query_type!r}; expected {QUERY_TYPES}")


@dataclass
class RetrievalResult:
    entries: List[CaseEntry] = field(default_factory=list)
    precedents: List[PrecedentEntry] = field(default_factory=list)
    filtered_counts: Dict[str, int] = field(default_factory=dict)
    redactions: int = 0

    def __len__(self) -> int:
        return len(self.entries) + len(self.precedents)

    @property
    def empty(self) -> bool:
        return not self.entries and not self.precedents

    def to_prompt(self, lang: str = "en") -> str:
        if self.empty:
            return ""
        header = "【检索到的参考案例】" if lang == "zh" else "[Retrieved reference cases]"
        lines = [header]
        for entry in self.entries:
            signature = ", ".join(entry.au_signature) if entry.au_signature else "-"
            lines.append(f"- ({entry.kind}, signature [{signature}]) {entry.content}")
        for precedent in self.precedents:
            lines.append(
                f"- (precedent) check '{precedent.check_type}' on "
                f"[{', '.join(precedent.au_signature)}]: upheld "
                f"{precedent.n_upheld}/{precedent.n_used} "
                f"(discriminative power {precedent.discriminative_power})"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_entries": len(self.entries), "n_precedents": len(self.precedents),
            "filtered": self.filtered_counts, "redactions": self.redactions,
        }


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def redact_above_level(text: str, level: EvidenceLevel) -> Tuple[str, int]:
    if not text:
        return "", 0
    sentences = _SENTENCE_SPLIT.split(text)
    kept, dropped = [], 0
    for sentence in sentences:
        lowered = sentence.lower()
        mentions_emotion = any(word in lowered for word in EMOTION_LEXICON)
        mentions_au = bool(AU_PATTERN.search(sentence))
        if level <= EvidenceLevel.MOTION and (mentions_emotion or mentions_au):
            dropped += 1
            continue
        if level <= EvidenceLevel.AU and mentions_emotion:
            dropped += 1
            continue
        kept.append(sentence)
    return " ".join(kept).strip(), dropped


def truncate_tokens(text: str, max_tokens: int) -> str:
    parts = text.split()
    if len(parts) <= max_tokens:
        return text
    return " ".join(parts[:max_tokens]) + " ..."


class CaseRetriever:

    def __init__(self, memory: SemanticMemory, config: Optional[MemoryConfig] = None) -> None:
        self.memory = memory
        self.config = config or MemoryConfig()
        self._session_seen: set[str] = set()

    def reset_session(self) -> None:
        self._session_seen.clear()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        result = RetrievalResult()
        if not self.config.retrieval_enabled:
            result.filtered_counts["disabled"] = 1
            return result

        if request.query_type == "precedent":
            result.precedents = self.memory.best_checks(
                request.au_signature, n=min(request.limit, self.config.retrieval_max_per_level)
            )
            return result

        candidates = self._candidates(request)
        result.filtered_counts["initial"] = len(candidates)

        candidates = [
            c for c in candidates
            if jaccard(c.au_signature, request.au_signature) >= self.config.retrieval_jaccard
        ]
        result.filtered_counts["after_consistency"] = len(candidates)

        visible: List[CaseEntry] = []
        for case in candidates:
            if case.evidence_level > int(request.level):
                result.redactions += 1
                continue
            content, dropped = redact_above_level(case.content, request.level)
            result.redactions += dropped
            if not content:
                continue
            visible.append(CaseEntry(
                case_id=case.case_id, dataset=case.dataset,
                au_signature=list(case.au_signature), evidence_level=case.evidence_level,
                emotion="" if request.level <= EvidenceLevel.AU else case.emotion,
                content=content, quality=case.quality, kind=case.kind, outcome=case.outcome,
            ))
        result.filtered_counts["after_visibility"] = len(visible)

        fresh = [c for c in visible
                 if c.case_id not in self._session_seen and c.case_id not in request.exclude]
        fresh.sort(key=lambda c: (-c.quality,
                                  -jaccard(c.au_signature, request.au_signature)))
        result.filtered_counts["after_freshness"] = len(fresh)

        limit = min(request.limit, self.config.retrieval_max_per_level)
        selected = fresh[:limit]
        for case in selected:
            case.content = truncate_tokens(case.content, self.config.retrieval_max_tokens)
            self._session_seen.add(case.case_id)
        result.entries = selected
        result.filtered_counts["final"] = len(selected)
        return result

    def _candidates(self, request: RetrievalRequest) -> List[CaseEntry]:
        pool = [
            c for c in self.memory.cases
            if not request.dataset or c.dataset == request.dataset
        ]
        if request.query_type == "support":
            return [c for c in pool if c.kind == "support"]
        if request.query_type == "confusion":
            return [c for c in pool
                    if c.kind in {"support", "confusion"}
                    and (not request.emotion or c.emotion != request.emotion)]
        if request.query_type == "counterexample":
            return [c for c in pool if c.kind == "counterexample"]
        return pool

    def support(self, au_signature: Sequence[str], level: EvidenceLevel,
                dataset: str = "") -> RetrievalResult:
        return self.retrieve(RetrievalRequest("support", au_signature, level, dataset))

    def confusion(self, au_signature: Sequence[str], emotion: str,
                  dataset: str = "") -> RetrievalResult:
        return self.retrieve(RetrievalRequest(
            "confusion", au_signature, EvidenceLevel.EMOTION, dataset, emotion))

    def counterexample(self, au_signature: Sequence[str], level: EvidenceLevel,
                       dataset: str = "") -> RetrievalResult:
        return self.retrieve(RetrievalRequest("counterexample", au_signature, level, dataset))

    def precedent(self, au_signature: Sequence[str]) -> RetrievalResult:
        return self.retrieve(RetrievalRequest(
            "precedent", au_signature, EvidenceLevel.VERIFICATION))


__all__ = [
    "QUERY_TYPES", "RetrievalRequest", "RetrievalResult", "redact_above_level",
    "truncate_tokens", "CaseRetriever",
]
