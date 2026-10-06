"""Abstract base class shared by all four MEWM-Agent role agents."""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import MEWMConfig, load_config
from ..engines.v4_token_regulator import (
    QUERY_PAIRS, TokenRegulator, build_context_tokens, instruction_token, text_token,
)
from ..llm.client import LLMError, LLMResponse, call_model
from ..memory.retrieval import CaseRetriever, RetrievalResult
from ..orchestration.state import (
    MEWMState, PHASE_AGENT, PHASE_BANS, Projection, project,
)
from ..schemas import (
    ContractError, Evidence, EvidenceLevel, scan_forbidden_vocabulary, validate,
)

LOGGER = logging.getLogger(__name__)

ROLES_DIR = Path(__file__).resolve().parent / "roles"

JSON_ONLY_DIRECTIVE = """"""

CONTRACT_SCHEMAS: Dict[str, str] = {
    "proposal": '{"proposals": [{"cid": str, "interval": [int, int], "apex": int, '
                '"peak_S": float, "attribution": {}, "physio_overlap": bool, '
                '"confirmed": bool, "channel": str, "notes": str}], '
                '"curve_summary": str, "rejected": []}',
    "motion_evidence": '{"motion_evidence": [{"roi": str, "roi_index": int, '
                       '"magnitude_px": float, "direction_deg": float, '
                       '"direction_label": str, "coherence": float, "salient": bool}], '
                       '"interval": [int, int], "summary": str}',
    "au_activation": '{"active_aus": [str], "weak_aus": [str], "fits": [{"au": str, '
                     '"roi": str, "direction_fit": str, "fit_score": float, '
                     '"magnitude_px": float, "symmetry": float, "refs": [str]}], '
                     '"slot_agreement": float, "open_questions": [{"kind": str, '
                     '"detail": str}], "summary": str}',
    "au_graph": '{"au_graph": {"nodes": {"AU4": {"phase": [int, int, int], '
                '"peak": float, "rise_slope": float, "decay_slope": float, '
                '"activation": str}}, "edges": [{"source": str, "target": str, '
                '"polarity": str, "lag_frames": int, "lag_ms": float, '
                '"weight": float, "w_model": float, "w_obs": float, '
                '"conflict": bool}]}, "graph_narrative": str, "open_questions": []}',
    "causal_cot": '{"P": {"q": str, "t": str, "a": str}, "M": {"q": str, "t": str, '
                  '"a": str}, "C": {"q": str, "t": str, "a": str}, '
                  '"MC": {"confidence_sources": str, "open_questions": [str]}, '
                  '"es": {}, "dc": {}, "joint": {}, "k_crit": [str], '
                  '"exclusions": [{"emotion": str, "reason": str}], '
                  '"fine_label": str, "coarse_label": str, "refs": [str], '
                  '"responses": [{"ch_id": str, "mode": str, "text": str, '
                  '"refs": [str]}], "revised_labels": null, '
                  '"new_open_questions": []}',
    "challenges": '{"challenges": [{"type": str, "statement": str, "refs": [str], '
                  '"analysis_report_id": str, "final": ""}], "lambda": float, '
                  '"cfs": {}, "cfs_margin": float, "mni": {}, '
                  '"template_distances": {}, "counterfactual_statement": str, '
                  '"suppression_signal": str}',
    "verdict": '{"e_fine": str, "e_coarse": str, "confidence": float, '
               '"suppression": str, "fusion_terms": {"q_ev": float, "margin": float, '
               '"gamma_chal": float, "s_proto": float}, '
               '"prototype_completeness": float, "rationale": str, "refs": [str]}',
    "narrative": '{"text": str, "assertions": [{"text": str, "refs": [str], '
                 '"t_span": [int, int]}], "part1_proposals": [{"proposal_id": int, '
                 '"onset": int, "offset": int}], "part2_analysis": '
                 '[{"proposal_id": int, "localisation_basis": str, '
                 '"static_description": str, "dynamic_description": str, '
                 '"au_cot": str, "coarse_label": str, "fine_label": str, '
                 '"relation": str}], "baseline_covered": bool, '
                 '"consistency_check": {"passed": bool, "revised": []}}',
}

def schema_reminder(contract: str) -> str:
    schema = CONTRACT_SCHEMAS.get(contract)
    header = (
        "\n\n===\n"
        "Respond with ONE JSON object only. Begin your reply with { and end it with }."
    )
    footer = (
        "\nNo prose. No markdown. No code fences. Do not copy the schema back -- "
        "write your own values into every field."
    )
    if not schema:
        return header + footer
    return f"{header}\nSchema: {schema}{footer}"

@dataclass
class RoleSpec:

    name: str
    agent: str
    phase: str
    description: str = ""
    temperature: float = 0.2
    tools: List[str] = field(default_factory=list)
    output_contract: str = ""
    body: str = ""
    body_zh: str = ""

    def system_prompt(self, lang: str = "en") -> str:
        if lang == "zh" and self.body_zh:
            return self.body_zh
        return self.body

    @classmethod
    def load(cls, path: Path | str) -> "RoleSpec":
        raw = Path(path).read_text(encoding="utf-8")
        meta, body = _split_frontmatter(raw)
        english, chinese = _split_languages(body)
        return cls(
            name=str(meta.get("name", Path(path).stem)),
            agent=str(meta.get("agent", "")),
            phase=str(meta.get("phase", "")),
            description=str(meta.get("description", "")),
            temperature=float(meta.get("temperature", 0.2)),
            tools=list(meta.get("tools", []) or []),
            output_contract=str(meta.get("output_contract", "")),
            body=english.strip(),
            body_zh=chinese.strip(),
        )

def _split_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        import yaml
        meta = yaml.safe_load(parts[1]) or {}
    except Exception:
        meta = {}
        for line in parts[1].splitlines():
            if ":" in line and not line.strip().startswith("#"):
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
    return (meta if isinstance(meta, dict) else {}), parts[2]

_ZH_MARKER = re.compile(r"^##\s*中文\s*$", re.MULTILINE)

def _split_languages(body: str) -> Tuple[str, str]:
    match = _ZH_MARKER.search(body)
    if not match:
        return body, ""
    return body[:match.start()], body[match.end():]

_ROLE_CACHE: Dict[str, RoleSpec] = {}

def load_role(name: str) -> RoleSpec:
    if name in _ROLE_CACHE:
        return _ROLE_CACHE[name]
    path = ROLES_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"role file not found: {path}")
    spec = RoleSpec.load(path)
    _ROLE_CACHE[name] = spec
    return spec

def available_roles() -> List[str]:
    return sorted(p.stem for p in ROLES_DIR.glob("*.md"))

@dataclass
class AgentResult:

    phase: str
    product: Dict[str, Any] = field(default_factory=dict)
    entries: List[Evidence] = field(default_factory=list)
    raw_text: str = ""
    parsed: bool = True
    degraded: bool = False
    notes: List[str] = field(default_factory=list)
    llm: Optional[LLMResponse] = None
    regulation: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None

    def gate_payload(self) -> Dict[str, Any]:
        payload = dict(self.product)
        payload["_entries"] = self.entries
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase, "product": self.product,
            "entries": [e.to_dict() for e in self.entries],
            "parsed": self.parsed, "degraded": self.degraded, "notes": self.notes,
            "llm": self.llm.to_dict() if self.llm else None,
            "regulation": self.regulation, "retrieval": self.retrieval,
        }

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

def json_reject_reason(text: str) -> str:
    if not text or not text.strip():
        return "empty reply"
    block = _first_brace_block(text) or text
    try:
        payload = json.loads(block, strict=False)
    except json.JSONDecodeError as exc:
        near = block[max(0, exc.pos - 60):exc.pos + 60]
        return f"{exc.msg} at char {exc.pos} of {len(block)}, near {near!r}"
    return (f"decoded to {type(payload).__name__}, not an object"
            if not isinstance(payload, dict) else "decoded cleanly on re-check")

def _loads_object(text: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = json.loads(text, strict=False)
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates: List[str] = []
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        payload = _loads_object(candidate)
        if payload is not None:
            return payload
        block = _first_brace_block(candidate)
        if block:
            payload = _loads_object(block)
            if payload is not None:
                return payload
    return None

def salvage_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates: List[str] = []
    fence_open = re.match(r"```(?:json)?\s*", text)
    if fence_open:
        rest = text[fence_open.end():]
        closing = rest.rfind("```")
        candidates.append(rest if closing < 0 else rest[:closing])
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        start = candidate.find("{")
        if start < 0:
            continue
        repaired = _close_truncated(candidate[start:])
        if repaired is None:
            continue
        payload = _loads_object(repaired)
        if payload is not None:
            return payload
    return None

def _close_truncated(block: str) -> Optional[str]:
    stack: List[str] = []
    in_string, escaped = False, False
    for char in block:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]"):
            if not stack or stack[-1] != char:
                return None
            stack.pop()
    if not stack and not in_string:
        return None
    repaired = block.rstrip()
    if in_string:
        repaired += '"'
    if repaired.endswith(","):
        repaired = repaired[:-1]
    while stack:
        repaired += stack.pop()
    return repaired

def _first_brace_block(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None

class BaseAgent(ABC):

    role_files: Dict[str, str] = {}

    def __init__(
        self,
        agent_id: str,
        model: str,
        config: Optional[MEWMConfig] = None,
        client: Optional[Callable[..., LLMResponse]] = None,
        regulator: Optional[TokenRegulator] = None,
        retriever: Optional[CaseRetriever] = None,
        lang: str = "en",
    ) -> None:
        self.agent_id = agent_id
        self.model = model
        self.config = config or load_config()
        self.client = client or call_model
        self.regulator = regulator or TokenRegulator(self.config.token_regulator)
        self.retriever = retriever
        self.lang = lang
        self.level = EvidenceLevel.of_agent(agent_id)
        self.call_log: List[Dict[str, Any]] = []

    @abstractmethod
    def phases(self) -> Tuple[str, ...]:
        pass

    @abstractmethod
    def build_user_prompt(self, phase: str, projection: Projection,
                          state: MEWMState, **kwargs: Any) -> str:
        pass

    @abstractmethod
    def parse(self, phase: str, payload: Dict[str, Any], projection: Projection,
              state: MEWMState, **kwargs: Any) -> AgentResult:
        pass

    def fallback(self, phase: str, projection: Projection, state: MEWMState,
                 reason: str, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=phase, degraded=True, parsed=False)
        result.notes.append(f"degraded: {reason}")
        return result

    def run(
        self,
        phase: str,
        state: MEWMState,
        cid: str = "",
        revision: str = "",
        slot_trajectory: Optional[Any] = None,
        measurements: Optional[Any] = None,
        **kwargs: Any,
    ) -> AgentResult:
        if phase not in self.phases():
            raise ValueError(f"{self.agent_id} does not implement phase {phase!r}")

        projection = project(state, phase, cid,
                             slot_trajectory=slot_trajectory, measurements=measurements)

        if state.budget.exhausted:
            state.budget.degrade(f"{phase}: LLM call budget exhausted")
            return self.fallback(phase, projection, state, "budget exhausted", **kwargs)

        role = self.role_for(phase)
        system_prompt = role.system_prompt(self.lang) + JSON_ONLY_DIRECTIVE
        user_prompt = self.build_user_prompt(phase, projection, state, **kwargs)
        if revision:
            user_prompt = f"{user_prompt}\n\n{revision}"

        retrieval = self._retrieve(phase, state, cid, **kwargs)
        if retrieval is not None and not retrieval.empty:
            user_prompt = f"{user_prompt}\n\n{retrieval.to_prompt(self.lang)}"

        regulated_prompt, regulation = self._regulate(phase, system_prompt, user_prompt,
                                                      kwargs.get("frames", ()))
        regulated_prompt += schema_reminder(role.output_contract)

        started = time.time()
        try:
            response = self.client(
                system_prompt=system_prompt,
                user_prompt=regulated_prompt,
                image_paths=kwargs.get("image_paths"),
                model=self.model,
                max_tokens=self.config.llm.max_tokens_for(phase),
                timeout=self.config.llm.timeout,
                retries=self.config.llm.retries,
                reasoning_effort=self.config.llm.effort_for(self.agent_id) or None,
                backend=self.config.llm.backend_for_role(self.agent_id),
                fallback_to_api=self.config.backends.fallback_to_api,
                role=self.agent_id,
            )
        except (LLMError, TypeError) as exc:
            if isinstance(exc, TypeError):
                try:
                    response = self.client(system_prompt, regulated_prompt,
                                           kwargs.get("image_paths"), self.model)
                except Exception as inner:
                    LOGGER.warning("%s %s: client failed (%s)", self.agent_id, phase, inner)
                    state.budget.degrade(f"{phase}: client error {inner}")
                    return self.fallback(phase, projection, state, str(inner), **kwargs)
            else:
                LOGGER.warning("%s %s: LLM failed (%s)", self.agent_id, phase, exc)
                state.budget.degrade(f"{phase}: {exc}")
                return self.fallback(phase, projection, state, str(exc), **kwargs)

        state.budget.spend()
        self.call_log.append({
            "phase": phase, "cid": cid, "model": self.model,
            "latency_s": round(time.time() - started, 3),
        })

        payload = extract_json(response.text)
        salvaged = False
        if payload is None:
            payload = salvage_json(response.text)
            salvaged = payload is not None
        if payload is None:
            raw = response.text or ""
            if not raw.strip():
                shape = "empty reply"
            elif len(raw) <= 400:
                shape = f"{len(raw)} chars: {raw!r}"
            else:
                shape = f"{len(raw)} chars: {raw[:240]!r} ... {raw[-120:]!r}"
            LOGGER.warning("%s %s: unparsable response -- %s -- %s",
                           self.agent_id, phase, json_reject_reason(raw), shape)
            state.budget.degrade(f"{phase}: unparsable model output")
            result = self.fallback(phase, projection, state, "unparsable output", **kwargs)
            result.raw_text = response.text
            result.llm = response
            return result

        if salvaged:
            LOGGER.warning("%s %s: truncated response salvaged (%d chars)",
                           self.agent_id, phase, len(response.text or ""))
            state.budget.degrade(f"{phase}: truncated response salvaged")

        try:
            result = self.parse(phase, payload, projection, state, **kwargs)
        except Exception as exc:
            LOGGER.warning(
                "%s %s: parse raised %s on schema-valid JSON -- %s",
                self.agent_id, phase, type(exc).__name__, str(exc)[:400])
            state.budget.degrade(
                f"{phase}: parse error ({type(exc).__name__}) on schema-valid JSON")
            result = self.fallback(phase, projection, state,
                                   f"parse error: {type(exc).__name__}", **kwargs)
            result.raw_text = response.text
            result.llm = response
            result.regulation = regulation
            result.retrieval = retrieval.to_dict() if retrieval else None
            self._enforce_output_bans(phase, result)
            return result

        result.raw_text = response.text
        result.llm = response
        result.regulation = regulation
        result.retrieval = retrieval.to_dict() if retrieval else None
        if salvaged:
            result.notes.append("payload recovered from a truncated response")
        self._enforce_output_bans(phase, result)
        return result

    def role_for(self, phase: str) -> RoleSpec:
        name = self.role_files.get(phase)
        if not name:
            raise KeyError(f"no role file registered for phase {phase!r}")
        return load_role(name)

    def temperature_for(self, phase: str) -> float:
        return self.role_for(phase).temperature

    def _regulate(
        self, phase: str, system_prompt: str, user_prompt: str,
        frames: Sequence[Tuple[int, str]] = (),
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if phase not in QUERY_PAIRS or not self.config.token_regulator.enabled:
            return user_prompt, None
        lines = [line for line in user_prompt.split("\n") if line.strip()]
        tokens = build_context_tokens([system_prompt], lines, frames)
        tokens, report = self.regulator.regulate(tokens, phase, lang=self.lang)
        body = self.regulator.apply_to_text(
            [t for t in tokens if t.source != "instruction"]
        )
        return (body or user_prompt), report.to_dict()

    def _retrieve(self, phase: str, state: MEWMState, cid: str,
                  **kwargs: Any) -> Optional[RetrievalResult]:
        if self.retriever is None:
            return None
        signature = kwargs.get("au_signature") or []
        if not signature:
            proposal = state.proposal(cid)
            signature = proposal.top_aus() if proposal else []
        if not signature:
            return None
        from ..orchestration.state import PHASE_MAX_EVIDENCE_LEVEL
        level = PHASE_MAX_EVIDENCE_LEVEL.get(phase, EvidenceLevel.MOTION)
        dataset = state.video_meta.dataset if state.video_meta else ""
        try:
            return self.retriever.support(signature, level, dataset)
        except Exception as exc:
            LOGGER.debug("retrieval failed for %s: %s", phase, exc)
            return None

    def _enforce_output_bans(self, phase: str, result: AgentResult) -> None:
        bans = PHASE_BANS.get(phase, {})
        if not (bans.get("ban_au") or bans.get("ban_emotion")):
            return
        hits = scan_forbidden_vocabulary(
            result.product, ban_au=bans.get("ban_au", False),
            ban_emotion=bans.get("ban_emotion", False),
        )
        for hit in hits:
            result.notes.append(f"R2 risk: {hit}")

    def emit(
        self, claim: str, payload: Optional[Dict[str, Any]] = None,
        refs: Sequence[str] = (), confidence: float = 1.0, cid: str = "",
        uncertainty: Optional[Dict[str, float]] = None,
    ) -> Evidence:
        return Evidence.create(
            self.agent_id, claim, payload, refs, confidence, cid, uncertainty
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_id, "model": self.model, "calls": len(self.call_log),
            "mean_latency_s": round(
                sum(c["latency_s"] for c in self.call_log) / len(self.call_log), 3
            ) if self.call_log else 0.0,
        }

def coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re as _re
        match = _re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return default
    return default

def coerce_float_map(payload: Any, keys: Optional[Sequence[str]] = None) -> Dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in payload.items():
        if keys is not None and key not in keys:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = float(value)
        elif isinstance(value, str):
            import re as _re
            match = _re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                try:
                    out[str(key)] = float(match.group())
                except ValueError:
                    continue
    return out

def as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}

def format_evidence_lines(entries: Sequence[Evidence], limit: int = 20) -> str:
    lines = []
    for entry in list(entries)[:limit]:
        payload = json.dumps(entry.payload, ensure_ascii=False) if entry.payload else ""
        lines.append(f"[{entry.eid}] {entry.claim}" + (f"  {payload}" if payload else ""))
    return "\n".join(lines)

__all__ = [
    "ROLES_DIR", "JSON_ONLY_DIRECTIVE", "RoleSpec", "load_role", "available_roles",
    "AgentResult", "CONTRACT_SCHEMAS", "schema_reminder", "coerce_float",
    "coerce_float_map", "as_dict", "salvage_json",
    "extract_json", "BaseAgent", "format_evidence_lines",
]
