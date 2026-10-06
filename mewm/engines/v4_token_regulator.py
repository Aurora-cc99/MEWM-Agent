"""V4 token regulator: evidence token budget management across segments."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TokenRegulatorConfig

LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class QueryPair:

    phase: str
    positive: str
    negative: str
    positive_zh: str = ""
    negative_zh: str = ""

    def pos(self, lang: str = "en") -> str:
        return self.positive_zh if (lang == "zh" and self.positive_zh) else self.positive

    def neg(self, lang: str = "en") -> str:
        return self.negative_zh if (lang == "zh" and self.negative_zh) else self.negative

QUERY_PAIRS: Dict[str, QueryPair] = {
    "P.verify": QueryPair(
        "P.verify",
        "Which facial regions show weak but directionally coherent motion?",
        "What are the overall environment and shooting conditions of the scene?",
        "哪些面部区域存在方向一致的微弱运动？",
        "画面的整体环境与拍摄条件是怎样的？",
    ),
    "A.encode": QueryPair(
        "A.encode",
        "Which regional motion patterns correspond to specific facial muscle actions?",
        "What are the resolution and lighting of the video?",
        "哪些区域运动模式对应特定面部肌肉动作？",
        "视频的分辨率与光照如何？",
    ),
    "A.graph": QueryPair(
        "A.graph",
        "What are the activation order, phase and mutual influence of the muscle actions?",
        "What static appearance features does the person have?",
        "各肌肉动作的激活次序、相位与相互影响是什么？",
        "人物的静态外观特征有哪些？",
    ),
    "R.reason": QueryPair(
        "R.reason",
        "Which action-unit combinations and temporal structures support a suppressed emotion?",
        "What is the scene topic of the video?",
        "哪些动作单元组合及其时序结构支持某种被压抑的情绪？",
        "视频的场景主题是什么？",
    ),
    "R.adjudicate": QueryPair(
        "R.adjudicate",
        "Which signals support or weaken the confidence of the current emotion conclusion?",
        "What format does the report use?",
        "哪些信号支持或削弱当前情绪结论的置信度？",
        "报告采用了怎样的格式？",
    ),
    "R.narrate": QueryPair(
        "R.narrate",
        "When and through what process did the affective state change?",
        "Which objects unrelated to the person appear in the video?",
        "情感状态在何时、以何种过程发生了变化？",
        "视频中有哪些与人物无关的物体？",
    ),
    "C.critic": QueryPair(
        "C.critic",
        "Which evidence may contradict, be missing from, or be dynamically inconsistent "
        "with the main emotion hypothesis?",
        "Which sentences in the text read fluently?",
        "哪些证据可能与主假设情绪矛盾、缺失或动力学不符？",
        "文本中哪些句子表达通顺？",
    ),
}

#: Phases that never see frame content and therefore bypass the regulator.
BYPASS_PHASES = frozenset({"P.scan"})

MODALITY_VISUAL = "visual"
MODALITY_TEXT = "text"
MODALITY_INSTRUCTION = "instruction"

@dataclass
class Token:

    index: int
    modality: str
    content: str = ""
    source: str = ""
    is_anchor: bool = False
    admitted: bool = True
    utility: float = 0.0
    gamma: float = 1.0

    @property
    def weight(self) -> float:
        return (1.0 if self.admitted else 0.0) * self.gamma

@dataclass
class RegulationReport:

    phase: str
    n_total: int = 0
    n_admitted: int = 0
    n_visual_admitted: int = 0
    n_text_admitted: int = 0
    n_instruction: int = 0
    n_anchor_protected: int = 0
    mean_gamma: float = 1.0
    dropped_sources: List[str] = field(default_factory=list)

    @property
    def admission_ratio(self) -> float:
        return round(self.n_admitted / self.n_total, 4) if self.n_total else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "phase": self.phase, "n_total": self.n_total, "n_admitted": self.n_admitted,
            "admission_ratio": self.admission_ratio,
            "visual_admitted": self.n_visual_admitted,
            "text_admitted": self.n_text_admitted,
            "instruction_kept": self.n_instruction,
            "anchor_protected": self.n_anchor_protected,
            "mean_gamma": round(self.mean_gamma, 4),
        }

_ANCHOR_PATTERNS = (
    re.compile(r"\d+\.\d+\s*px"), re.compile(r"\bcoherence\s*[:=]\s*\d"),
    re.compile(r"\bmagnitude\b", re.IGNORECASE), re.compile(r"\bS_t\b"),
    re.compile(r"\bES\(|\bDC\(|\bMNI\b|\bLambda\b|\bCFS\b"),
)

def is_anchor_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in _ANCHOR_PATTERNS)

class TokenRegulator:

    def __init__(self, config: Optional[TokenRegulatorConfig] = None) -> None:
        self.config = config or TokenRegulatorConfig()
        self.history: List[RegulationReport] = []

    def regulate(
        self,
        tokens: Sequence[Token],
        phase: str,
        attention_pos: Optional[np.ndarray] = None,
        attention_neg: Optional[np.ndarray] = None,
        cross_modal_predictability: Optional[np.ndarray] = None,
        affect_relevance: Optional[np.ndarray] = None,
        lang: str = "en",
    ) -> Tuple[List[Token], RegulationReport]:

        tokens = list(tokens)
        report = RegulationReport(phase=phase, n_total=len(tokens))
        if not tokens:
            return tokens, report

        if not self.config.enabled or phase in BYPASS_PHASES:
            for token in tokens:
                token.admitted, token.gamma = True, 1.0
            report.n_admitted = len(tokens)
            self.history.append(report)
            return tokens, report

        pair = QUERY_PAIRS.get(phase)
        if pair is None:
            LOGGER.debug("no query pair registered for phase %s; passing through", phase)
            for token in tokens:
                token.admitted, token.gamma = True, 1.0
            report.n_admitted = len(tokens)
            self.history.append(report)
            return tokens, report

        utility = self._contrastive_utility(tokens, pair, attention_pos, attention_neg, lang)
        for token, value in zip(tokens, utility):
            token.utility = float(value)

        self._admit(tokens, utility, report)
        self._reweight(tokens, cross_modal_predictability, affect_relevance, report)

        admitted = [t for t in tokens if t.admitted]
        report.n_admitted = len(admitted)
        report.mean_gamma = (
            float(np.mean([t.gamma for t in admitted])) if admitted else 1.0
        )
        self.history.append(report)
        return tokens, report

    def _contrastive_utility(
        self,
        tokens: Sequence[Token],
        pair: QueryPair,
        attention_pos: Optional[np.ndarray],
        attention_neg: Optional[np.ndarray],
        lang: str,
    ) -> np.ndarray:
        if attention_pos is not None and attention_neg is not None:
            positive = _normalise(np.asarray(attention_pos, dtype=np.float64))
            negative = _normalise(np.asarray(attention_neg, dtype=np.float64))
            if positive.shape == negative.shape == (len(tokens),):
                return positive - negative
            LOGGER.warning("attention shape mismatch; falling back to the proxy utility")

        positive = _normalise(np.array(
            [_affinity(t, pair.pos(lang)) for t in tokens], dtype=np.float64))
        negative = _normalise(np.array(
            [_affinity(t, pair.neg(lang)) for t in tokens], dtype=np.float64))
        return positive - negative

    def _admit(self, tokens: List[Token], utility: np.ndarray, report: RegulationReport) -> None:
        by_modality: Dict[str, List[int]] = {}
        for i, token in enumerate(tokens):
            by_modality.setdefault(token.modality, []).append(i)

        for i, token in enumerate(tokens):
            token.admitted = False

        for modality, indices in by_modality.items():
            if modality == MODALITY_INSTRUCTION:
                for i in indices:
                    tokens[i].admitted = True
                report.n_instruction = len(indices)
                continue

            ratio = self.config.p_v if modality == MODALITY_VISUAL else self.config.p_t
            keep = max(1, int(math.ceil(len(indices) * float(ratio))))
            ranked = sorted(indices, key=lambda i: -utility[i])
            # Anchors are admitted regardless of rank: dropping a quantified measurement
            anchors = [i for i in indices if tokens[i].is_anchor]
            selected = set(ranked[:keep]) | set(anchors)
            for i in selected:
                tokens[i].admitted = True
            for i in indices:
                if not tokens[i].admitted and tokens[i].source:
                    report.dropped_sources.append(tokens[i].source)
            if modality == MODALITY_VISUAL:
                report.n_visual_admitted = len(selected)
            else:
                report.n_text_admitted = len(selected)
            report.n_anchor_protected += len([i for i in anchors if i not in set(ranked[:keep])])

    def _reweight(
        self,
        tokens: List[Token],
        cross_modal: Optional[np.ndarray],
        affect_relevance: Optional[np.ndarray],
        report: RegulationReport,
    ) -> None:
        delta = float(self.config.delta)
        n = len(tokens)
        complement = (np.asarray(cross_modal, dtype=np.float64) if cross_modal is not None
                      else np.array([_complementarity(t) for t in tokens], dtype=np.float64))
        redundancy = 1.0 - complement
        gate = (np.asarray(affect_relevance, dtype=np.float64) if affect_relevance is not None
                else np.array([_affect_gate(t) for t in tokens], dtype=np.float64))

        if complement.shape != (n,):
            complement = np.full(n, 0.5)
            redundancy = 1.0 - complement
        if gate.shape != (n,):
            gate = np.full(n, 0.5)

        raw = 1.0 + delta * gate * (complement - redundancy)
        for i, token in enumerate(tokens):
            if not token.admitted:
                token.gamma = 0.0
                continue
            value = float(np.clip(raw[i], 0.05, 1.0 + delta))
            if token.is_anchor or token.modality == MODALITY_INSTRUCTION:
                value = max(value, self.config.gamma_anchor_min)
            token.gamma = round(value, 4)

    def apply_to_text(self, tokens: Sequence[Token], separator: str = "\n") -> str:
        admitted = [t for t in tokens if t.admitted and t.modality != MODALITY_INSTRUCTION]
        instructions = [t for t in tokens if t.modality == MODALITY_INSTRUCTION]
        admitted.sort(key=lambda t: -t.gamma)
        parts = [t.content for t in instructions]
        parts.extend(t.content for t in admitted if t.content)
        return separator.join(p for p in parts if p)

    def stats(self) -> Dict[str, object]:
        if not self.history:
            return {"calls": 0}
        return {
            "calls": len(self.history),
            "mean_admission_ratio": round(
                float(np.mean([r.admission_ratio for r in self.history])), 4),
            "mean_gamma": round(float(np.mean([r.mean_gamma for r in self.history])), 4),
            "by_phase": {
                phase: sum(1 for r in self.history if r.phase == phase)
                for phase in {r.phase for r in self.history}
            },
        }

_WORD_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")

def _tokenise(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "and", "or", "is", "are", "what", "which",
    "does", "do", "to", "for", "with", "that", "this", "it", "its", "their",
})

def _affinity(token: Token, query: str) -> float:
    query_terms = {w for w in _tokenise(query) if w not in _STOPWORDS}
    if not query_terms:
        return 0.0
    if token.modality == MODALITY_VISUAL:
        content_terms = {w for w in _tokenise(token.source) if w not in _STOPWORDS}
    else:
        content_terms = {w for w in _tokenise(token.content) if w not in _STOPWORDS}
    if not content_terms:
        return 0.0
    overlap = len(query_terms & content_terms)
    return overlap / math.sqrt(len(content_terms) * len(query_terms))

def _complementarity(token: Token) -> float:
    if token.is_anchor:
        return 0.9
    if token.modality == MODALITY_VISUAL:
        return 0.7
    if token.modality == MODALITY_INSTRUCTION:
        return 0.5
    return 0.35

def _affect_gate(token: Token) -> float:
    text = f"{token.content} {token.source}".lower()
    cues = ("motion", "direction", "coherence", "magnitude", "activation", "phase",
            "onset", "apex", "offset", "au", "slot", "belief")
    hits = sum(1 for cue in cues if cue in text)
    return min(1.0, 0.25 + 0.15 * hits)

def _normalise(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)

def text_token(index: int, content: str, source: str = "") -> Token:
    return Token(index=index, modality=MODALITY_TEXT, content=content, source=source,
                 is_anchor=is_anchor_text(content))

def instruction_token(index: int, content: str) -> Token:
    return Token(index=index, modality=MODALITY_INSTRUCTION, content=content,
                 source="instruction", is_anchor=True)

def visual_token(index: int, frame: int, region: str = "", content: str = "") -> Token:
    return Token(index=index, modality=MODALITY_VISUAL, content=content,
                 source=f"frame:{frame}@{region}" if region else f"frame:{frame}")

def build_context_tokens(
    instructions: Sequence[str],
    evidence_lines: Sequence[str],
    frames: Sequence[Tuple[int, str]] = (),
) -> List[Token]:
    tokens: List[Token] = []
    for text in instructions:
        tokens.append(instruction_token(len(tokens), text))
    for line in evidence_lines:
        tokens.append(text_token(len(tokens), line))
    for frame, region in frames:
        tokens.append(visual_token(len(tokens), frame, region))
    return tokens

__all__ = [
    "QueryPair", "QUERY_PAIRS", "BYPASS_PHASES", "MODALITY_VISUAL", "MODALITY_TEXT",
    "MODALITY_INSTRUCTION", "Token", "RegulationReport", "TokenRegulator",
    "is_anchor_text", "text_token", "instruction_token", "visual_token",
    "build_context_tokens",
]
