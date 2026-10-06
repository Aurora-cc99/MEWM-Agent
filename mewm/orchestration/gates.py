"""Additional evidence gates used by the orchestrator between pipeline stages."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..knowledge.au_anatomy import hard_conflicts, is_antagonistic, prior_polarity
from ..knowledge.emotion_prototypes import (
    FINE_EMOTIONS, canonical_fine_label, labels_consistent,
)
from ..schemas import (
    AUDynGraph, CausalCoT, Evidence, EvidenceChain, EvidenceLevel, GateRecord,
    OpenQuestion, Verdict, scan_forbidden_vocabulary,
)

LOGGER = logging.getLogger(__name__)

NUMERIC_TOLERANCE = 1e-3

GRADE_SUFFICIENT = "sufficient"
GRADE_GAP = "specific_gap"
GRADE_INSUFFICIENT = "clearly_insufficient"


@dataclass
class GateOutcome:
    passed: bool
    consistency: GateRecord
    sufficiency: GateRecord
    revision_instruction: str = ""

    @property
    def failed_rules(self) -> List[str]:
        return list(self.consistency.failed_rules)

    @property
    def questions(self) -> List[str]:
        return list(self.sufficiency.questions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "consistency": self.consistency.to_dict(),
            "sufficiency": self.sufficiency.to_dict(),
            "revision_instruction": self.revision_instruction,
        }


class ConsistencyGate:

    def __init__(self, tolerance: float = NUMERIC_TOLERANCE) -> None:
        self.tolerance = tolerance

    def check(
        self,
        phase: str,
        product: Dict[str, Any],
        chain: EvidenceChain,
        *,
        ban_au: bool = False,
        ban_emotion: bool = False,
        recompute: Optional[Dict[str, float]] = None,
        au_graph: Optional[AUDynGraph] = None,
        active_aus: Sequence[str] = (),
        open_questions: Sequence[OpenQuestion] = (),
        causal_cot: Optional[CausalCoT] = None,
        verdict: Optional[Verdict] = None,
        retry_idx: int = 0,
    ) -> GateRecord:
        failed: List[str] = []
        details: List[str] = []

        for rule, problems in (
            ("R1", self._r1_references(product, chain)),
            ("R2", self._r2_boundary(product, ban_au, ban_emotion)),
            ("R3", self._r3_numeric(product, recompute)),
            ("R4", self._r4_priors(active_aus, au_graph, open_questions)),
            ("R5", self._r5_phase(causal_cot, verdict)),
        ):
            if problems:
                failed.append(rule)
                details.extend(f"{rule}: {p}" for p in problems)

        return GateRecord(
            gate="consistency", node=phase, passed=not failed,
            failed_rules=failed, questions=details, retry_idx=retry_idx,
        )

    @staticmethod
    def _r1_references(product: Dict[str, Any], chain: EvidenceChain) -> List[str]:
        problems: List[str] = []
        entries = product.get("_entries") or []
        for entry in entries:
            if not isinstance(entry, Evidence):
                continue
            for ref in entry.refs:
                target = chain.get(ref)
                if target is None:
                    problems.append(f"entry {entry.eid} cites unknown {ref}")
                elif target.status == "rejected":
                    problems.append(f"entry {entry.eid} cites rejected {ref}")
                elif target.level >= entry.level:
                    problems.append(
                        f"entry {entry.eid} (level {int(entry.level)}) cites {ref} at "
                        f"level {int(target.level)}; citations must be strictly lower"
                    )
        for ref in product.get("refs", []) or []:
            if isinstance(ref, str) and ref not in chain:
                problems.append(f"product cites unknown entry {ref}")
        return problems

    R2_EXEMPT_FIELDS = frozenset({"attribution"})

    @classmethod
    def _r2_boundary(cls, product: Dict[str, Any], ban_au: bool, ban_emotion: bool) -> List[str]:
        if not (ban_au or ban_emotion):
            return []
        scannable = cls._strip_exempt(
            {k: v for k, v in product.items() if not k.startswith("_")}
        )
        return scan_forbidden_vocabulary(scannable, ban_au=ban_au, ban_emotion=ban_emotion)

    @classmethod
    def _strip_exempt(cls, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {
                key: cls._strip_exempt(value)
                for key, value in payload.items()
                if key not in cls.R2_EXEMPT_FIELDS
            }
        if isinstance(payload, list):
            return [cls._strip_exempt(item) for item in payload]
        return payload

    def _r3_numeric(
        self, product: Dict[str, Any], recompute: Optional[Dict[str, float]],
    ) -> List[str]:
        if not recompute:
            return []
        problems: List[str] = []
        claimed = _flatten_numbers(product)
        for key, expected in recompute.items():
            if key not in claimed:
                problems.append(f"quantity '{key}' is missing from the product")
                continue
            actual = claimed[key]
            if expected is None or actual is None:
                continue
            if isinstance(expected, float) and math.isnan(expected):
                continue
            if abs(float(actual) - float(expected)) > self.tolerance:
                problems.append(
                    f"quantity '{key}' claimed {actual} but recomputes to "
                    f"{round(float(expected), 6)} (tolerance {self.tolerance})"
                )
        return problems

    @staticmethod
    def _r4_priors(
        active_aus: Sequence[str],
        au_graph: Optional[AUDynGraph],
        open_questions: Sequence[OpenQuestion],
    ) -> List[str]:
        problems: List[str] = []
        registered = " ".join(q.detail for q in open_questions)

        for a, b in hard_conflicts(active_aus):
            if a not in registered or b not in registered:
                problems.append(
                    f"{a} and {b} are antagonistic but both reported active, and the "
                    f"conflict is not registered as an open question"
                )

        if au_graph is not None:
            for edge in au_graph.edges:
                if edge.conflict or (edge.weight != edge.weight):
                    marker = f"{edge.source}->{edge.target}"
                    if marker not in registered:
                        problems.append(
                            f"edge {marker} has a model/observation sign conflict that "
                            f"is not registered as an open question"
                        )
                expected = prior_polarity(edge.source, edge.target)
                if expected and edge.polarity != expected and not edge.conflict:
                    marker = f"{edge.source}->{edge.target}"
                    if marker not in registered:
                        problems.append(
                            f"edge {marker} polarity {edge.polarity} contradicts the "
                            f"anatomical prior {expected} without registration"
                        )
        return problems

    @staticmethod
    def _r5_phase(causal_cot: Optional[CausalCoT], verdict: Optional[Verdict]) -> List[str]:
        problems: List[str] = []
        for source, label in (("verdict", verdict.e_fine if verdict else ""),
                              ("CoT", causal_cot.fine_label if causal_cot else "")):
            if label and label not in FINE_EMOTIONS:
                problems.append(
                    f"{source} fine label {label!r} is not in the emotion vocabulary "
                    f"({', '.join(FINE_EMOTIONS)})"
                )
        if verdict is not None:
            if verdict.e_fine and verdict.e_coarse and not labels_consistent(
                verdict.e_fine, verdict.e_coarse
            ):
                problems.append(
                    f"fine label '{verdict.e_fine}' does not map to coarse label "
                    f"'{verdict.e_coarse}'"
                )
            if causal_cot is not None and causal_cot.fine_label:
                if verdict.e_fine and verdict.e_fine != causal_cot.fine_label:
                    if not verdict.rationale:
                        problems.append(
                            f"adjudicated emotion '{verdict.e_fine}' differs from the "
                            f"argued '{causal_cot.fine_label}' with no traceable rationale"
                        )
        if causal_cot is not None and causal_cot.fine_label and causal_cot.coarse_label:
            if not labels_consistent(causal_cot.fine_label, causal_cot.coarse_label):
                problems.append(
                    f"CoT fine label '{causal_cot.fine_label}' does not map to coarse "
                    f"label '{causal_cot.coarse_label}'"
                )
        return problems


def _flatten_numbers(payload: Any, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).startswith("_"):
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out[path] = float(value)
                out.setdefault(str(key), float(value))
            else:
                out.update(_flatten_numbers(value, path))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            out.update(_flatten_numbers(value, f"{prefix}[{index}]"))
    return out


@dataclass
class CheckItem:
    key: str
    description: str
    test: Callable[[Dict[str, Any]], bool]
    question: str
    critical: bool = False


def _non_empty(name: str) -> Callable[[Dict[str, Any]], bool]:
    return lambda product: bool(product.get(name))


CHECKLISTS: Dict[str, List[CheckItem]] = {
    "P.scan": [
        CheckItem("proposals", "the confirmed proposal set is present",
                  lambda p: "proposals" in p,
                  "Which intervals did the hysteresis rule produce, and which did you confirm?",
                  critical=True),
        CheckItem("curve_summary", "a summary of the error curve is present",
                  _non_empty("curve_summary"),
                  "Summarise the S_t curve and its three components over the scanned span."),
        CheckItem("physio_check", "physiological overlap was checked per proposal",
                  lambda p: all("physio_overlap" in item
                                for item in (p.get("proposals") or [])),
                  "State for each proposal whether it overlaps a physiological event."),
    ],
    "P.verify": [
        CheckItem("motion_evidence", "per-region measurements are reported",
                  _non_empty("motion_evidence"),
                  "Report the measurement triple for every anatomical region in the interval.",
                  critical=True),
        CheckItem("triples", "each record carries all three measured quantities",
                  lambda p: all(
                      all(k in item for k in ("magnitude_px", "direction_deg", "coherence"))
                      for item in (p.get("motion_evidence") or [])),
                  "Every record must carry magnitude, principal direction and coherence.",
                  critical=True),
        CheckItem("salience", "the salience verdict is stated per region",
                  lambda p: all("salient" in item for item in (p.get("motion_evidence") or [])),
                  "State the salience verdict for each region."),
    ],
    "A.encode": [
        CheckItem("active_aus", "the activation set is adjudicated",
                  lambda p: "active_aus" in p, "Which AUs are active and which are weak?",
                  critical=True),
        CheckItem("fits", "direction-fit evidence is given per candidate",
                  _non_empty("fits"),
                  "Give the direction fit, magnitude score and symmetry for each candidate.",
                  critical=True),
        CheckItem("cross_validation", "the slot read-out cross-check is reported",
                  lambda p: "slot_agreement" in p,
                  "Report agreement with the object-slot activation read-out, and register "
                  "any disagreement as an open question."),
    ],
    "A.graph": [
        CheckItem("nodes", "graph nodes carry phase triples",
                  lambda p: bool((p.get("au_graph") or {}).get("nodes")),
                  "Give onset/apex/offset and the rise and decay slopes for each node.",
                  critical=True),
        CheckItem("edges", "edges are present with polarity and lag",
                  lambda p: "edges" in (p.get("au_graph") or {}),
                  "Give polarity, phase lag and weight for each edge."),
        CheckItem("narrative", "the graph is rendered as a motion account",
                  _non_empty("graph_narrative"),
                  "Translate the graph into a motion description with node or edge citations."),
    ],
    "R.reason": [
        CheckItem("layers", "all five CoT layers are present",
                  lambda p: all(k in p for k in ("P", "M", "C", "MC")),
                  "The chain must contain the P, M, C, CF+MHV and MC layers.",
                  critical=True),
        CheckItem("scores", "ES and DC are computed for the candidates",
                  lambda p: bool(p.get("es")) and bool(p.get("dc")),
                  "Compute ES and DC for the top hypotheses and list them.",
                  critical=True),
        CheckItem("k_crit", "the critical evidence set is derived",
                  _non_empty("k_crit"),
                  "Run the leave-one-out test and report K_crit."),
        CheckItem("labels", "fine and coarse labels are given",
                  lambda p: bool(p.get("fine_label")) and bool(p.get("coarse_label")),
                  "Give both the fine-grained and the coarse-grained label.", critical=True),
        CheckItem("exclusions", "competing hypotheses are explicitly excluded",
                  _non_empty("exclusions"),
                  "State, hypothesis by hypothesis, why each competitor was excluded."),
    ],
    "C.critic": [
        CheckItem("challenges", "challenges are present",
                  lambda p: "challenges" in p, "Issue your challenges, or state none apply.",
                  critical=True),
        CheckItem("reports", "each challenge cites a rollout analysis report",
                  lambda p: all(item.get("analysis_report_id")
                                for item in (p.get("challenges") or [])),
                  "Every challenge must cite a rollout analysis report id.", critical=True),
        CheckItem("lambda", "the likelihood ratio was computed",
                  lambda p: "lambda" in p, "Compute Lambda between the top two hypotheses.",
                  critical=True),
        CheckItem("mni", "necessity indices were computed for K_crit",
                  _non_empty("mni"), "Report MNI for every AU in K_crit.", critical=True),
    ],
    "R.adjudicate": [
        CheckItem("confidence", "a calibrated confidence is produced",
                  lambda p: "confidence" in p, "Fuse the four signals into a confidence.",
                  critical=True),
        CheckItem("fusion", "the confidence decomposition is given",
                  _non_empty("fusion_terms"), "Decompose the confidence into its four terms."),
        CheckItem("suppression", "the suppression / masquerade verdict is stated",
                  lambda p: "suppression" in p,
                  "State the suppression or masquerade verdict and its basis."),
    ],
    "R.narrate": [
        CheckItem("text", "the narrative text is present",
                  _non_empty("text"), "Produce the global narrative.", critical=True),
        CheckItem("assertions", "assertions carry citations",
                  _non_empty("assertions"),
                  "Every temporal assertion must carry an entry citation.", critical=True),
        CheckItem("outside", "time outside the proposals is described",
                  lambda p: bool(p.get("baseline_covered")),
                  "Describe the baseline outside the proposals from the curve and the slow log."),
    ],
}


class SufficiencyGate:

    def check(self, phase: str, product: Dict[str, Any], retry_idx: int = 0) -> GateRecord:
        items = CHECKLISTS.get(phase, [])
        if not items:
            return GateRecord("sufficiency", phase, True, grade=GRADE_SUFFICIENT,
                              retry_idx=retry_idx)

        missing: List[CheckItem] = []
        for item in items:
            try:
                ok = bool(item.test(product))
            except Exception:
                ok = False
            if not ok:
                missing.append(item)

        if not missing:
            grade = GRADE_SUFFICIENT
        elif any(item.critical for item in missing) and len(missing) >= 2:
            grade = GRADE_INSUFFICIENT
        else:
            grade = GRADE_GAP

        return GateRecord(
            gate="sufficiency", node=phase, passed=grade == GRADE_SUFFICIENT,
            grade=grade, failed_rules=[item.key for item in missing],
            questions=[item.question for item in missing], retry_idx=retry_idx,
        )


REVISION_TEMPLATE = """[Revision instruction]
Target: {phase}
Failed gate: {gate}
Violated rules / gaps: {items}
Offending entries: {refs}
Targeted follow-up:
{questions}
{exemplars}Retries remaining: {remaining}"""

REVISION_TEMPLATE_ZH = """【修订指令】
目标执行体·相位：{phase}
失败门控：{gate}
违规规则/缺口：{items}
违规条目引用：{refs}
针对性追问：
{questions}
{exemplars}剩余重试余额：{remaining}"""


class DualGate:

    def __init__(self, tolerance: float = NUMERIC_TOLERANCE, lang: str = "en") -> None:
        self.consistency = ConsistencyGate(tolerance)
        self.sufficiency = SufficiencyGate()
        self.lang = lang

    def evaluate(
        self, phase: str, product: Dict[str, Any], chain: EvidenceChain,
        retries_remaining: int = 0, exemplars: str = "", **kwargs: Any,
    ) -> GateOutcome:
        consistency = self.consistency.check(phase, product, chain,
                                             retry_idx=kwargs.pop("retry_idx", 0), **kwargs)
        sufficiency = self.sufficiency.check(phase, product)
        passed = consistency.passed and sufficiency.passed

        instruction = ""
        if not passed:
            failing = consistency if not consistency.passed else sufficiency
            template = REVISION_TEMPLATE_ZH if self.lang == "zh" else REVISION_TEMPLATE
            refs = _offending_refs(product)
            instruction = template.format(
                phase=phase,
                gate=failing.gate,
                items=", ".join(failing.failed_rules) or "-",
                refs=", ".join(refs) or "-",
                questions="\n".join(f"  {i + 1}. {q}"
                                    for i, q in enumerate(failing.questions)) or "  -",
                exemplars=(f"Repair exemplars:\n{exemplars}\n" if exemplars else ""),
                remaining=retries_remaining,
            )

        return GateOutcome(passed, consistency, sufficiency, instruction)


def _offending_refs(product: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for entry in product.get("_entries", []) or []:
        if isinstance(entry, Evidence):
            refs.append(entry.eid)
    refs.extend(str(r) for r in (product.get("refs") or []))
    return refs[:8]


__all__ = [
    "NUMERIC_TOLERANCE", "GRADE_SUFFICIENT", "GRADE_GAP", "GRADE_INSUFFICIENT",
    "GateOutcome", "ConsistencyGate", "CheckItem", "CHECKLISTS", "SufficiencyGate",
    "DualGate", "REVISION_TEMPLATE", "REVISION_TEMPLATE_ZH",
]
