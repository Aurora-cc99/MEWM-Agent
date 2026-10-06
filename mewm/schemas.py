"""Typed interface objects shared across agents and engines."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


class EvidenceLevel(IntEnum):
    MOTION = 0
    AU = 1
    EMOTION = 2
    VERIFICATION = 3

    @classmethod
    def of_agent(cls, agent: str) -> "EvidenceLevel":
        mapping = {
            "P": cls.MOTION, "P-Agent": cls.MOTION, "perception": cls.MOTION,
            "A": cls.AU, "A-Agent": cls.AU, "structure": cls.AU,
            "R": cls.EMOTION, "R-Agent": cls.EMOTION, "reasoning": cls.EMOTION,
            "C": cls.VERIFICATION, "C-Agent": cls.VERIFICATION, "critic": cls.VERIFICATION,
            "M2": cls.MOTION, "M3": cls.MOTION, "V1": cls.MOTION, "V2": cls.AU,
            "orchestrator": cls.VERIFICATION,
        }
        if agent not in mapping:
            raise KeyError(f"unknown evidence source: {agent!r}")
        return mapping[agent]


class ContractError(ValueError):
    pass


def _new_id(prefix: str, payload: Any = None) -> str:
    seed = f"{prefix}|{time.time_ns()}|{payload!r}"
    return f"{prefix}#{hashlib.blake2s(seed.encode('utf-8'), digest_size=5).hexdigest()}"


@dataclass
class ROIMeasurement:
    roi_index: int
    roi_name: str
    roi_label: str
    magnitude_px: float
    direction_deg: float
    coherence: float
    salient: bool = False
    magnitude_class: str = ""
    direction_label: str = ""
    coherence_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameState:
    t: int
    measurements: List[ROIMeasurement] = field(default_factory=list)
    head_motion: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    slot_activations: Dict[str, float] = field(default_factory=dict)
    slots: Optional[Any] = None
    z_slow: Optional[Any] = None
    z_fast: Optional[Any] = None
    z_belief: Optional[Any] = None
    available: bool = True

    def digest(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "available": self.available,
            "head_motion": list(self.head_motion),
            "slot_activations": {k: round(float(v), 4) for k, v in self.slot_activations.items()},
            "salient_rois": [m.roi_name for m in self.measurements if m.salient],
        }


@dataclass
class LatentStream:
    video_id: str
    fps: float
    frames: Dict[int, FrameState] = field(default_factory=dict)
    slot_order: List[str] = field(default_factory=list)
    frame_lo: int = 0
    frame_hi: int = 0

    def __len__(self) -> int:
        return len(self.frames)

    def indices(self) -> List[int]:
        return sorted(self.frames)

    def get(self, t: int) -> Optional[FrameState]:
        return self.frames.get(t)

    def slot_trajectory(self, au: str, t_on: int, t_off: int) -> List[float]:
        return [
            float(self.frames[t].slot_activations.get(au, 0.0))
            for t in range(t_on, t_off + 1) if t in self.frames
        ]

    def availability_gaps(self) -> List[Tuple[int, int]]:
        gaps: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for t in self.indices():
            if not self.frames[t].available:
                start = t if start is None else start
            elif start is not None:
                gaps.append((start, t - 1))
                start = None
        if start is not None:
            gaps.append((start, self.indices()[-1]))
        return gaps


@dataclass
class PhysioEvent:
    t_start: int
    t_end: int
    template_id: str
    match_energy: float
    label: str = ""

    def overlaps(self, t_on: int, t_off: int) -> bool:
        return not (self.t_end < t_on or self.t_start > t_off)


@dataclass
class ErrorRecord:
    video_id: str
    t_start: int
    s_curve: List[float] = field(default_factory=list)
    delta_total: List[float] = field(default_factory=list)
    delta_scene: List[float] = field(default_factory=list)
    delta_physio: List[float] = field(default_factory=list)
    delta_expr: List[float] = field(default_factory=list)
    au_attribution: Dict[str, List[float]] = field(default_factory=dict)
    physio_events: List[PhysioEvent] = field(default_factory=list)
    low_confidence_spans: List[Tuple[int, int]] = field(default_factory=list)

    def s_at(self, t: int) -> float:
        idx = t - self.t_start
        return float(self.s_curve[idx]) if 0 <= idx < len(self.s_curve) else 0.0

    def window(self, t_on: int, t_off: int) -> Dict[str, List[float]]:
        lo = max(0, t_on - self.t_start)
        hi = min(len(self.s_curve), t_off - self.t_start + 1)
        return {
            "S": self.s_curve[lo:hi],
            "scene": self.delta_scene[lo:hi],
            "physio": self.delta_physio[lo:hi],
            "expr": self.delta_expr[lo:hi],
        }

    def summary(self, t_on: int, t_off: int) -> Dict[str, Any]:
        win = self.window(t_on, t_off)
        s = win["S"] or [0.0]
        return {
            "interval": [t_on, t_off],
            "S_peak": round(max(s), 3),
            "S_mean": round(sum(s) / len(s), 3),
            "expr_share": round(
                sum(win["expr"]) / (sum(win["expr"]) + sum(win["scene"]) + sum(win["physio"]) + 1e-8), 3
            ),
            "physio_overlap": any(e.overlaps(t_on, t_off) for e in self.physio_events),
        }


@dataclass
class CandidateInterval:
    cid: str
    t_on: int
    t_off: int
    apex: int
    peak_S: float
    attribution: Dict[str, float] = field(default_factory=dict)
    physio_overlap: bool = False
    channel: str = "micro"
    confirmed: bool = False
    notes: str = ""

    @property
    def duration(self) -> int:
        return self.t_off - self.t_on + 1

    def iou(self, other: Tuple[int, int]) -> float:
        lo = max(self.t_on, other[0])
        hi = min(self.t_off, other[1])
        inter = max(0, hi - lo + 1)
        union = self.duration + (other[1] - other[0] + 1) - inter
        return inter / union if union > 0 else 0.0

    def top_aus(self, n: int = 3) -> List[str]:
        return [au for au, _ in sorted(self.attribution.items(), key=lambda kv: -kv[1])[:n]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AU_PATTERN = re.compile(r"\bAU\s?\d{1,2}\b", re.IGNORECASE)
EMOTION_LEXICON = (
    "happiness", "happy", "joy", "surprise", "surprised", "disgust", "disgusted",
    "anger", "angry", "fear", "fearful", "afraid", "sadness", "sad", "contempt",
    "repression", "repressed", "positive", "negative", "emotion", "emotional",
    "affect", "feeling", "mood",
    "开心", "快乐", "惊讶", "厌恶", "愤怒",
    "生气", "恐惧", "害怕", "悲伤", "轻蔑",
    "压抑", "积极", "消极", "情绪", "情感",
)
MUSCLE_LEXICON = (
    "brow lowerer", "lid tighten", "nose wrinkl", "lip press", "cheek rais",
    "muscle action", "action unit", "动作单元", "肌肉动作",
)


@dataclass
class Evidence:
    eid: str
    source: str
    claim: str
    payload: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    status: str = "active"
    level: EvidenceLevel = EvidenceLevel.MOTION
    refs: List[str] = field(default_factory=list)
    cid: str = ""
    supersedes: str = ""
    created_ns: int = field(default_factory=time.time_ns)

    @classmethod
    def create(
        cls,
        source: str,
        claim: str,
        payload: Optional[Dict[str, Any]] = None,
        refs: Optional[Sequence[str]] = None,
        confidence: float = 1.0,
        cid: str = "",
        uncertainty: Optional[Dict[str, float]] = None,
    ) -> "Evidence":
        return cls(
            eid=_new_id("ev", claim),
            source=source,
            claim=claim,
            payload=dict(payload or {}),
            uncertainty=dict(uncertainty or {}),
            confidence=confidence,
            level=EvidenceLevel.of_agent(source),
            refs=list(refs or []),
            cid=cid,
        )

    def revise(self, claim: str, payload: Optional[Dict[str, Any]] = None) -> "Evidence":
        successor = Evidence.create(
            self.source, claim, payload if payload is not None else self.payload,
            refs=self.refs, confidence=self.confidence, cid=self.cid,
        )
        successor.supersedes = self.eid
        return successor

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["level"] = int(self.level)
        return data


class EvidenceChain:
    def __init__(self, cid: str = "") -> None:
        self.cid = cid
        self._entries: Dict[str, Evidence] = {}
        self._order: List[str] = []

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, eid: str) -> bool:
        return eid in self._entries

    def __iter__(self):
        return (self._entries[eid] for eid in self._order)

    def get(self, eid: str) -> Optional[Evidence]:
        return self._entries.get(eid)

    def active(self, level: Optional[EvidenceLevel] = None) -> List[Evidence]:
        return [
            e for e in self
            if e.status == "active" and (level is None or e.level == level)
        ]

    def add(self, entry: Evidence, *, enforce_layering: bool = True) -> Evidence:
        # R1: refs must point strictly lower in the evidence hierarchy
        if entry.eid in self._entries:
            raise ContractError(f"duplicate evidence id {entry.eid}")
        if enforce_layering:
            for ref in entry.refs:
                target = self._entries.get(ref)
                if target is None:
                    raise ContractError(f"{entry.eid}: dangling reference {ref}")
                if target.status == "rejected":
                    raise ContractError(f"{entry.eid}: references rejected entry {ref}")
                if target.level >= entry.level:
                    raise ContractError(
                        f"{entry.eid} (level {int(entry.level)}) references {ref} at "
                        f"level {int(target.level)}; references must be strictly lower"
                    )
        if entry.supersedes and entry.supersedes in self._entries:
            self._entries[entry.supersedes].status = "revised"
        self._entries[entry.eid] = entry
        self._order.append(entry.eid)
        return entry

    def to_list(self) -> List[Dict[str, Any]]:
        return [self._entries[eid].to_dict() for eid in self._order]

    @classmethod
    def from_list(cls, items: Sequence[Dict[str, Any]], cid: str = "") -> "EvidenceChain":
        chain = cls(cid)
        for item in items:
            payload = dict(item)
            payload["level"] = EvidenceLevel(int(payload.get("level", 0)))
            chain.add(Evidence(**payload), enforce_layering=False)
        return chain


PRIMITIVES = ("rollout", "score", "mask", "compare")


@dataclass
class RolloutRecord:
    req_id: str
    caller: str
    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)
    result_digest: Dict[str, Any] = field(default_factory=dict)
    model_version: str = "uninitialised"
    cid: str = ""
    created_ns: int = field(default_factory=time.time_ns)

    @classmethod
    def create(
        cls, caller: str, primitive: str, params: Dict[str, Any],
        result_digest: Dict[str, Any], model_version: str, cid: str = "",
    ) -> "RolloutRecord":
        if primitive not in PRIMITIVES:
            raise ContractError(f"unknown primitive {primitive!r}; expected one of {PRIMITIVES}")
        return cls(
            req_id=_new_id("roll", (caller, primitive)),
            caller=caller, primitive=primitive, params=dict(params),
            result_digest=dict(result_digest), model_version=model_version, cid=cid,
        )

    def as_evidence(self) -> Evidence:
        entry = Evidence.create(
            "M3",
            f"{self.primitive}({', '.join(f'{k}={v}' for k, v in list(self.params.items())[:3])})",
            payload={"result": self.result_digest, "model_version": self.model_version},
            cid=self.cid,
        )
        entry.eid = self.req_id
        return entry

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CHALLENGE_TYPES = ("evidence_gap", "dynamics_inconsistency", "insufficient_necessity")
CHALLENGE_TYPES_ZH = {
    "evidence_gap": "证据缺口",
    "dynamics_inconsistency": "动力学不一致",
    "insufficient_necessity": "必要性不足",
}
FINAL_VERDICTS = ("rejected", "partial", "upheld")
FINAL_VERDICT_STEP = {"rejected": 0.0, "partial": -0.08, "upheld": -0.20}


@dataclass
class ChallengeRecord:
    ch_id: str
    ch_type: str
    refs: List[str] = field(default_factory=list)
    analysis_report_id: str = ""
    statement: str = ""
    response: str = ""
    final: str = ""
    round_index: int = 0
    cid: str = ""

    @classmethod
    def create(
        cls, ch_type: str, statement: str, refs: Sequence[str],
        analysis_report_id: str, round_index: int = 0, cid: str = "",
    ) -> "ChallengeRecord":
        if ch_type not in CHALLENGE_TYPES:
            raise ContractError(f"unknown challenge type {ch_type!r}")
        if not analysis_report_id:
            raise ContractError("a challenge must cite a rollout analysis report")
        if not refs:
            raise ContractError("a challenge must cite at least one evidence entry")
        return cls(
            ch_id=_new_id("ch", statement), ch_type=ch_type, refs=list(refs),
            analysis_report_id=analysis_report_id, statement=statement,
            round_index=round_index, cid=cid,
        )

    def set_final(self, verdict: str) -> None:
        if verdict not in FINAL_VERDICTS:
            raise ContractError(f"final verdict must be one of {FINAL_VERDICTS}")
        self.final = verdict

    @property
    def confidence_step(self) -> float:
        return FINAL_VERDICT_STEP.get(self.final, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AUNode:
    au: str
    t_on: int
    t_apex: int
    t_off: int
    peak: float
    rise_slope: float
    decay_slope: float
    activation: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AUEdge:
    source: str
    target: str
    polarity: str
    lag_frames: int
    lag_ms: float
    weight: float
    w_model: float = 0.0
    w_obs: float = 0.0
    conflict: bool = False
    p_value: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.weight != self.weight:
            data["weight"] = None
        return data


@dataclass
class AUDynGraph:
    cid: str
    nodes: Dict[str, AUNode] = field(default_factory=dict)
    edges: List[AUEdge] = field(default_factory=list)
    narrative: str = ""

    @property
    def active_aus(self) -> List[str]:
        return sorted([k for k, n in self.nodes.items() if n.activation == "active"])

    @property
    def weak_aus(self) -> List[str]:
        return sorted([k for k, n in self.nodes.items() if n.activation == "weak"])

    def onset_order(self) -> List[str]:
        return [k for k, _ in sorted(self.nodes.items(), key=lambda kv: (kv[1].t_on, kv[0]))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid,
            "nodes": {k: n.to_dict() for k, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "narrative": self.narrative,
        }


@dataclass
class CausalCoT:
    cid: str
    P: Dict[str, str] = field(default_factory=dict)
    M: Dict[str, str] = field(default_factory=dict)
    C: Dict[str, Any] = field(default_factory=dict)
    cf_mhv: Dict[str, Any] = field(default_factory=dict)
    MC: Dict[str, Any] = field(default_factory=dict)

    es: Dict[str, float] = field(default_factory=dict)
    dc: Dict[str, float] = field(default_factory=dict)
    joint: Dict[str, float] = field(default_factory=dict)
    k_crit: List[str] = field(default_factory=list)
    fine_label: str = ""
    coarse_label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SUPPRESSION_STATES = ("none", "neutralised", "masked")


def coerce_suppression(value: Any) -> Tuple[str, str]:
    text = str(value or "").strip()
    if not text:
        return "none", ""
    lowered = text.lower()
    if lowered in SUPPRESSION_STATES:
        return lowered, ""
    if "masquerad" in lowered or "masked" in lowered or "掩饰" in text:
        return "masked", text
    if "neutralis" in lowered or "neutraliz" in lowered or "抑制" in text:
        return "neutralised", text
    if "suppress" in lowered:
        return "none", text
    return "none", text


@dataclass
class Verdict:
    cid: str
    e_coarse: str = ""
    e_fine: str = ""
    confidence: float = 0.0
    suppression: str = "none"
    fusion_terms: Dict[str, float] = field(default_factory=dict)
    prototype_completeness: float = 0.0
    degraded: bool = False
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NarrativeAssertion:
    text: str
    refs: List[str] = field(default_factory=list)
    t_span: Optional[Tuple[int, int]] = None


@dataclass
class Narrative:
    video_id: str
    text: str = ""
    assertions: List[NarrativeAssertion] = field(default_factory=list)
    consistency_checked: bool = False
    revisions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "text": self.text,
            "assertions": [
                {"text": a.text, "refs": a.refs,
                 "t_span": list(a.t_span) if a.t_span else None}
                for a in self.assertions
            ],
            "consistency_checked": self.consistency_checked,
            "revisions": self.revisions,
        }


@dataclass
class OpenQuestion:
    qid: str
    kind: str
    detail: str
    refs: List[str] = field(default_factory=list)
    cid: str = ""
    resolved: bool = False

    @classmethod
    def create(cls, kind: str, detail: str, refs: Sequence[str] = (), cid: str = "") -> "OpenQuestion":
        return cls(qid=_new_id("oq", detail), kind=kind, detail=detail,
                   refs=list(refs), cid=cid)

    @classmethod
    def coerce(cls, item: Any, cid: str = "", default_kind: str = "unspecified") -> "OpenQuestion":
        # open questions must not be lost to a parsing quibble
        if isinstance(item, cls):
            return item
        if isinstance(item, dict):
            return cls.create(
                str(item.get("kind", default_kind) or default_kind),
                str(item.get("detail", item.get("text", "")) or ""),
                [str(r) for r in (item.get("refs") or [])], cid,
            )
        return cls.create(default_kind, str(item), (), cid)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlowDigest:
    t: int
    summary: List[float] = field(default_factory=list)
    kalman_gain: float = 0.0
    reset: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateRecord:
    gate: str
    node: str
    passed: bool
    failed_rules: List[str] = field(default_factory=list)
    grade: str = ""
    questions: List[str] = field(default_factory=list)
    retry_idx: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetState:
    llm_calls_used: int = 0
    max_llm_calls: int = 400
    path_choices: Dict[str, str] = field(default_factory=dict)
    gate_retries: Dict[str, int] = field(default_factory=dict)
    challenge_rounds: Dict[str, int] = field(default_factory=dict)
    cascade_rollbacks: Dict[str, int] = field(default_factory=dict)
    degradations: List[str] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.llm_calls_used >= self.max_llm_calls

    def spend(self, n: int = 1) -> None:
        self.llm_calls_used += n

    def degrade(self, reason: str) -> None:
        # every fallback is recorded; silent degradation is what we avoid
        self.degradations.append(reason)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoMeta:
    video_id: str
    dataset: str
    path: str
    fps: float
    n_frames: int
    subject_id: str = ""
    frame_lo: int = 0
    frame_hi: int = 0
    frame_prefix: str = ""
    frame_ext: str = ".jpg"
    frame_digits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "proposal": {
        "required": ["interval", "apex", "peak_S", "attribution", "physio_overlap"],
        "types": {"interval": list, "apex": int, "peak_S": float,
                  "attribution": dict, "physio_overlap": bool},
    },
    "motion_evidence": {
        "required": ["roi", "magnitude_px", "direction_deg", "coherence", "salient"],
        "types": {"magnitude_px": float, "direction_deg": float,
                  "coherence": float, "salient": bool},
    },
    "au_activation": {
        "required": ["active_aus", "weak_aus", "fits"],
        "types": {"active_aus": list, "weak_aus": list, "fits": list},
    },
    "au_graph": {
        "required": ["nodes", "edges"],
        "types": {"nodes": dict, "edges": list},
    },
    "causal_cot": {
        "required": ["P", "M", "C", "MC", "fine_label", "coarse_label"],
        "types": {"P": dict, "M": dict, "C": dict, "MC": dict,
                  "fine_label": str, "coarse_label": str},
    },
    "challenges": {
        "required": ["challenges"],
        "types": {"challenges": list},
    },
    "verdict": {
        "required": ["e_coarse", "e_fine", "confidence"],
        "types": {"e_coarse": str, "e_fine": str, "confidence": float},
    },
}


def validate(payload: Dict[str, Any], contract: str) -> List[str]:
    spec = _CONTRACTS.get(contract)
    if spec is None:
        raise KeyError(f"no contract named {contract!r}")
    problems: List[str] = []
    for key in spec["required"]:
        if key not in payload:
            problems.append(f"missing required field '{key}'")
    for key, expected in spec["types"].items():
        if key not in payload:
            continue
        value = payload[key]
        if expected is float and isinstance(value, int) and not isinstance(value, bool):
            continue
        if expected is int and isinstance(value, bool):
            problems.append(f"field '{key}' must be int, got bool")
            continue
        if not isinstance(value, expected):
            problems.append(
                f"field '{key}' must be {expected.__name__}, got {type(value).__name__}"
            )
    return problems


def _text_leaves(payload: Any, path: str = "") -> Iterator[Tuple[str, str]]:
    if isinstance(payload, str):
        yield path or "<root>", payload
    elif isinstance(payload, dict):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            if isinstance(key, str):
                yield here, key
            yield from _text_leaves(value, here)
    elif isinstance(payload, (list, tuple)):
        for i, item in enumerate(payload):
            yield from _text_leaves(item, f"{path}[{i}]")


def scan_forbidden_vocabulary(payload: Any, *, ban_au: bool, ban_emotion: bool) -> List[str]:
    leaves = list(_text_leaves(payload))
    hits: List[str] = []

    def report(label: str, matches: Dict[str, List[str]]) -> None:
        if not matches:
            return
        shown = sorted(matches)[:5]
        where = sorted({p for token in shown for p in matches[token]})[:4]
        hits.append(f"{label}: {shown} in {where}")

    if ban_au:
        au: Dict[str, List[str]] = {}
        muscle: Dict[str, List[str]] = {}
        for where, text in leaves:
            for token in AU_PATTERN.findall(text):
                au.setdefault(token, []).append(where)
            lowered = text.lower()
            for word in MUSCLE_LEXICON:
                if word in lowered:
                    muscle.setdefault(word, []).append(where)
        report("AU naming in a motion-layer product", au)
        report("muscle-action naming in a motion-layer product", muscle)
    if ban_emotion:
        emotion: Dict[str, List[str]] = {}
        for where, text in leaves:
            lowered = text.lower()
            for word in EMOTION_LEXICON:
                if word in lowered:
                    emotion.setdefault(word, []).append(where)
        report("emotion vocabulary below the emotion layer", emotion)
    return hits


def json_schema_for(contract: str) -> Dict[str, Any]:
    spec = _CONTRACTS[contract]
    py_to_json = {list: "array", dict: "object", str: "string",
                  int: "integer", float: "number", bool: "boolean"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": contract,
        "type": "object",
        "required": list(spec["required"]),
        "properties": {
            key: {"type": py_to_json[value]} for key, value in spec["types"].items()
        },
    }


__all__ = [
    "EvidenceLevel", "ContractError", "ROIMeasurement", "FrameState", "LatentStream",
    "PhysioEvent", "ErrorRecord", "CandidateInterval", "Evidence", "EvidenceChain",
    "RolloutRecord", "PRIMITIVES", "ChallengeRecord", "CHALLENGE_TYPES",
    "CHALLENGE_TYPES_ZH", "FINAL_VERDICTS", "FINAL_VERDICT_STEP", "AUNode", "AUEdge",
    "AUDynGraph", "CausalCoT", "Verdict", "NarrativeAssertion", "Narrative",
    "OpenQuestion", "SlowDigest", "GateRecord", "BudgetState", "VideoMeta",
    "SUPPRESSION_STATES", "coerce_suppression",
    "validate", "scan_forbidden_vocabulary", "json_schema_for",
    "AU_PATTERN", "EMOTION_LEXICON",
]
