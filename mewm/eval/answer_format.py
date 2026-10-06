"""Answer formatting and serialisation helpers for eval output."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


CITATION_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bappendix\s+[A-Z](?:\.\d+)*", re.IGNORECASE),
    re.compile(r"\beq(?:uation)?\.?\s*\(\s*\d+\s*\)", re.IGNORECASE),
    re.compile(r"\bgate\s+rule\s+R\d\b", re.IGNORECASE),
    re.compile(r"\brule\s+R\d\b", re.IGNORECASE),
    re.compile(r"\bsection\s+\d+(?:\.\d+)+", re.IGNORECASE),
    re.compile(r"\bpaper\s+\d+(?:\.\d+)+", re.IGNORECASE),
    re.compile(r"\bproposition\s+[A-Z]\.\d", re.IGNORECASE),
)

INTERNAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"HSV[- ]rendered|HSV image|raw vector field", re.IGNORECASE),
    re.compile(r"\boptical flow (?:is|was) read back", re.IGNORECASE),
    re.compile(r"\bdirectional-coherence channel\b", re.IGNORECASE),
    re.compile(r"\bnarration phase did not complete\b", re.IGNORECASE),
    re.compile(r"\b(?:P|A|R|C)[-.]Agent\b"),
    re.compile(r"\b(?:R\.reason|R\.narrate|A\.encode|A\.graph|P\.scan|P\.verify|"
               r"C\.critic|R\.adjudicate|R\.respond)\b"),
    re.compile(r"\bstage\(s\) degraded\b", re.IGNORECASE),
    re.compile(r"\bdeterministic fallback\b", re.IGNORECASE),
    re.compile(r"\btoken regulator\b|\bslot dropout\b|\brollout primitive\b", re.IGNORECASE),
    re.compile(r"\bmasking necessity index\b", re.IGNORECASE),
    re.compile(r"\bsufficiency graded\b|\bunparsable model output\b", re.IGNORECASE),
    re.compile(r"\bcompetitive ranking\b|\bpre-selection\b", re.IGNORECASE),
)

REQUIRED_SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("count", "This video contains"),
    ("localisation", "micro-expression: frames"),
    ("w_matrix", "W-matrix summary:"),
    ("label_consistency", "Coarse/fine label consistency:"),
    ("main_path", "Global main path:"),
    ("dag", "DAG validation:"),
    ("key_node", "Key causal node:"),
    ("authenticity", "Authenticity score:"),
    ("gcn", "GCN-style validation:"),
    ("cfi", "Counterfactual feature intervention (CFI):"),
    ("label_recheck", "Label recheck:"),
    ("au_cot", "AU-change CoT:"),
)

OPTIONAL_SECTIONS: Tuple[str, ...] = (
    "Confidence band", "Suppression", "Reliability",
)


@dataclass
class FormatReport:
    ok: bool = True
    citations: List[str] = field(default_factory=list)
    internals: List[str] = field(default_factory=list)
    missing_sections: List[str] = field(default_factory=list)
    out_of_order: List[str] = field(default_factory=list)

    def problems(self) -> List[str]:
        out: List[str] = []
        out += [f"paper-internal citation: {c!r}" for c in self.citations]
        out += [f"implementation detail: {i!r}" for i in self.internals]
        out += [f"missing section: {s}" for s in self.missing_sections]
        out += [f"section out of order: {s}" for s in self.out_of_order]
        return out

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok, "citations": self.citations, "internals": self.internals,
            "missing_sections": self.missing_sections,
            "out_of_order": self.out_of_order,
        }


def validate_answer(text: str, require_sections: bool = True) -> FormatReport:
    report = FormatReport()
    if not text:
        report.ok = False
        report.missing_sections = [name for name, _ in REQUIRED_SECTIONS]
        return report

    for pattern in CITATION_PATTERNS:
        report.citations.extend(sorted(set(pattern.findall(text)))[:3])
    for pattern in INTERNAL_PATTERNS:
        found = pattern.findall(text)
        if found:
            report.internals.extend(sorted({str(f) for f in found})[:3])

    if require_sections:
        detected = "This video contains 0 micro-expression" not in text
        positions: List[Tuple[str, int]] = []
        for name, marker in REQUIRED_SECTIONS:
            index = text.find(marker)
            if index < 0:
                if detected:
                    report.missing_sections.append(name)
                continue
            positions.append((name, index))
        ordered = [name for name, _ in sorted(positions, key=lambda kv: kv[1])]
        expected = [name for name, _ in REQUIRED_SECTIONS if name in ordered]
        if ordered != expected:
            report.out_of_order = [
                name for name, want in zip(ordered, expected) if name != want
            ]

    report.ok = not (report.citations or report.internals
                     or report.missing_sections or report.out_of_order)
    return report


def scrub(text: str) -> str:
    if not text:
        return text
    cleaned = text
    for pattern in CITATION_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    for pattern in INTERNAL_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\(\s*[,;]?\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = re.sub(r"([,;])\s*([,.;])", r"\2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\.\s*\.", ".", cleaned)
    return cleaned.strip()


__all__ = [
    "CITATION_PATTERNS", "INTERNAL_PATTERNS", "REQUIRED_SECTIONS", "OPTIONAL_SECTIONS",
    "FormatReport", "validate_answer", "scrub",
]
