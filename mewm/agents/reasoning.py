"""Reasoning agent: generates causal chain-of-thought and emotion labels."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..knowledge.au_anatomy import SLOT_INDEX, au_label
from ..knowledge.emotion_prototypes import (
    CONTRADICTORY_AUS, FINE_EMOTIONS, canonical_fine_label, coarse_of,
    competing_hypotheses, core_aus, evidence_sufficiency, labels_consistent,
    normalise_scores, prototype_completeness,
)
from ..orchestration.state import (
    MEWMState, PHASE_R_ADJUDICATE, PHASE_R_NARRATE, PHASE_R_REASON, PHASE_R_RESPOND,
    Projection,
)
from ..schemas import (
    CausalCoT, ChallengeRecord, Evidence, EvidenceLevel, FINAL_VERDICT_STEP, Narrative,
    NarrativeAssertion, OpenQuestion, Verdict, coerce_suppression,
)
from .base import (
    AgentResult, BaseAgent, coerce_float, coerce_float_map, as_dict, format_evidence_lines,
)

LOGGER = logging.getLogger(__name__)

def compute_es(active: Sequence[str], weak: Sequence[str],
               candidates: Sequence[str],
               intensities: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    return {
        emotion: evidence_sufficiency(emotion, active, weak, intensities)
        for emotion in candidates
    }

def joint_score(es: Dict[str, float], dc: Dict[str, float], alpha: float = 0.5) -> Dict[str, float]:
    return {
        emotion: round(alpha * es.get(emotion, 0.0) + (1.0 - alpha) * dc.get(emotion, 0.0), 4)
        for emotion in set(es) | set(dc)
    }

def leave_one_out_critical(
    active: Sequence[str], weak: Sequence[str], candidates: Sequence[str],
    dc: Dict[str, float], alpha: float = 0.5,
) -> List[str]:
    base = joint_score(compute_es(active, weak, candidates), dc, alpha)
    if not base:
        return []
    leader = max(base, key=lambda e: base[e])
    critical: List[str] = []
    for au in active:
        reduced = [a for a in active if a != au]
        trial = joint_score(compute_es(reduced, weak, candidates), dc, alpha)
        if trial and max(trial, key=lambda e: trial[e]) != leader:
            critical.append(au)
    return critical

def fuse_confidence(
    evidence_quality: float, margin: float, challenge_grade: float,
    prototype_score: float, weights: Sequence[float] = (0.35, 0.25, 0.20, 0.20),
) -> Tuple[float, Dict[str, float]]:
    terms = {
        "q_ev": round(float(evidence_quality), 4),
        "margin": round(float(margin), 4),
        "gamma_chal": round(float(challenge_grade), 4),
        "s_proto": round(float(prototype_score), 4),
    }
    total = sum(w * v for w, v in zip(weights, terms.values()))
    return round(float(np.clip(total, 0.0, 1.0)), 4), terms

def challenge_grade(challenges: Sequence[ChallengeRecord]) -> float:
    factor = 1.0
    for challenge in challenges:
        factor += FINAL_VERDICT_STEP.get(challenge.final, 0.0)
    return round(max(0.0, factor), 4)

class ReasoningAgent(BaseAgent):

    role_files = {
        PHASE_R_REASON: "r_agent_reason",
        PHASE_R_RESPOND: "r_agent_respond",
        PHASE_R_ADJUDICATE: "r_agent_adjudicate",
        PHASE_R_NARRATE: "r_agent_narrate",
    }

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__("R", model, **kwargs)

    def phases(self) -> Tuple[str, ...]:
        return (PHASE_R_REASON, PHASE_R_RESPOND, PHASE_R_ADJUDICATE, PHASE_R_NARRATE)

    def build_user_prompt(self, phase: str, projection: Projection,
                          state: MEWMState, **kwargs: Any) -> str:
        builders = {
            PHASE_R_REASON: self._reason_prompt,
            PHASE_R_RESPOND: self._respond_prompt,
            PHASE_R_ADJUDICATE: self._adjudicate_prompt,
            PHASE_R_NARRATE: self._narrate_prompt,
        }
        return builders[phase](projection, state, **kwargs)

    def _reason_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        cid = projection.cid
        graph = state.au_graphs.get(cid)
        active: Sequence[str] = kwargs.get("active_aus") or []
        weak: Sequence[str] = kwargs.get("weak_aus") or []
        es: Dict[str, float] = kwargs.get("es") or {}
        dc: Dict[str, float] = kwargs.get("dc") or {}
        proposal = state.proposal(cid)

        lines = [
            f"Proposal {cid}, interval "
            f"[{proposal.t_on if proposal else '?'}, {proposal.t_off if proposal else '?'}], "
            f"apex {proposal.apex if proposal else '?'}",
            "",
            "Prior evidence (cite these ids):",
            format_evidence_lines(projection.get("prior_evidence") or []),
            "",
            f"Active AUs: {list(active)}    Weak AUs: {list(weak)}",
        ]
        if graph is not None:
            lines.extend([
                "",
                "AU dynamic graph:",
                f"  onset order: {graph.onset_order()}",
                "  nodes: " + json.dumps(
                    {au: {"phase": [n.t_on, n.t_apex, n.t_off], "peak": n.peak,
                          "rise": n.rise_slope} for au, n in graph.nodes.items()},
                    ensure_ascii=False),
                "  edges: " + json.dumps([
                    {"e": f"{e.source}->{e.target}", "polarity": e.polarity,
                     "lag_ms": e.lag_ms,
                     "weight": (None if e.weight != e.weight else round(e.weight, 4))}
                    for e in graph.edges], ensure_ascii=False),
            ])
        if es:
            lines.append("")
            lines.append("Tool-computed ES (static evidence sufficiency): "
                         + json.dumps(es, ensure_ascii=False))
        if dc:
            lines.append("Tool-computed DC (dynamics consistency, from score): "
                         + json.dumps(dc, ensure_ascii=False))
        if kwargs.get("k_crit"):
            lines.append(f"Tool-computed K_crit (leave-one-out): {kwargs['k_crit']}")

        questions = projection.get("open_questions") or []
        if questions:
            lines.append("")
            lines.append("Open questions carried forward:")
            lines.extend(f"- [{q.kind}] {q.detail}" for q in questions)

        lines.append("")
        lines.append(
            "Build the five-layer chain. Report ES and DC separately, never merged. "
            "Leave the CF+MHV layer empty for the critic."
        )
        return "\n".join(lines)

    def _respond_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        cid = projection.cid
        challenges = [c for c in state.challenges.get(cid, []) if not c.final]
        cot = state.causal_cots.get(cid)
        lines = [f"Proposal {cid}: the critic has raised {len(challenges)} challenge(s)."]
        if cot is not None:
            lines.extend([
                "", "Your current chain:",
                f"  fine {cot.fine_label} / coarse {cot.coarse_label}",
                f"  ES {json.dumps(cot.es, ensure_ascii=False)}",
                f"  DC {json.dumps(cot.dc, ensure_ascii=False)}",
                f"  K_crit {cot.k_crit}",
            ])
        lines.append("")
        lines.append("Challenges:")
        for challenge in challenges:
            lines.append(
                f"- [{challenge.ch_id}] ({challenge.ch_type}) {challenge.statement} "
                f"(refs {challenge.refs}, report {challenge.analysis_report_id})"
            )
        lines.append("")
        lines.append("Answer each one. Choose your mode honestly.")
        return "\n".join(lines)

    def _adjudicate_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        cid = projection.cid
        cot = state.causal_cots.get(cid)
        challenges = state.challenges.get(cid, [])
        lines = [
            f"Proposal {cid} adjudication.",
            "",
            f"Argued conclusion: fine {cot.fine_label if cot else '?'} / "
            f"coarse {cot.coarse_label if cot else '?'}",
            f"Prototype completeness (tool): {kwargs.get('prototype_completeness', 0.0)}",
            f"Evidence quality (tool): {kwargs.get('evidence_quality', 0.0)}",
            f"Hypothesis margin (tool): {kwargs.get('margin', 0.0)}",
            f"Challenge factor (tool): {kwargs.get('challenge_factor', 1.0)}",
        ]
        if challenges:
            lines.append("")
            lines.append("Challenge outcomes:")
            lines.extend(
                f"- [{c.ch_id}] {c.ch_type}: {c.final or 'pending'}" for c in challenges
            )
        if kwargs.get("template_distances"):
            lines.append("")
            lines.append(f"Template distances: {kwargs['template_distances']}")
        if kwargs.get("mni"):
            lines.append(f"Necessity indices: {kwargs['mni']}")
        if kwargs.get("baseline_context"):
            lines.append("")
            lines.append(f"Out-of-proposal baseline context: {kwargs['baseline_context']}")
        else:
            lines.append("")
            lines.append(
                "Out-of-proposal baseline context is NOT available; do not assert "
                "masquerade on clause (iii)."
            )
        lines.append("")
        lines.append("Produce the verdict.")
        return "\n".join(lines)

    def _narrate_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        lines = [
            f"Video {state.video_id}. Question: {state.question}",
            "",
            f"Proposals ({len(state.proposals)}):",
        ]
        for proposal in state.proposals:
            verdict = state.verdicts.get(proposal.cid)
            graph = state.au_graphs.get(proposal.cid)
            lines.append(
                f"- {proposal.cid} [{proposal.t_on}, {proposal.t_off}] apex {proposal.apex} "
                f"peak_S {proposal.peak_S} -> "
                f"{verdict.e_fine if verdict else '?'} / "
                f"{verdict.e_coarse if verdict else '?'} "
                f"(confidence {verdict.confidence if verdict else 0.0}, "
                f"suppression {verdict.suppression if verdict else 'none'})"
            )
            if graph is not None:
                lines.append(f"    AU order: {graph.onset_order()}; "
                             f"active {graph.active_aus}, weak {graph.weak_aus}")

        baseline = kwargs.get("baseline_segments") or []
        if baseline:
            lines.append("")
            lines.append("Time outside every proposal (from the whole-curve index):")
            for segment in baseline[:12]:
                lines.append(
                    f"- frames {segment['interval'][0]}-{segment['interval'][1]}: "
                    f"S mean {segment.get('mean', 0.0)}, max {segment.get('max', 0.0)}"
                )
        links = kwargs.get("cross_links") or []
        if links:
            lines.append("")
            lines.append("Cross-proposal links:")
            lines.extend(f"- {l.source} -> {l.target}: {l.relation} ({l.detail})"
                         for l in links)
        if not state.proposals:
            lines.append("")
            lines.append(
                "No proposal was detected. Still produce the narrative and cite the "
                "curve evidence supporting the absence."
            )
        lines.append("")
        lines.append("Produce the two-part answer.")
        return "\n".join(lines)

    def parse(self, phase: str, payload: Dict[str, Any], projection: Projection,
              state: MEWMState, **kwargs: Any) -> AgentResult:
        parsers = {
            PHASE_R_REASON: self._parse_reason,
            PHASE_R_RESPOND: self._parse_respond,
            PHASE_R_ADJUDICATE: self._parse_adjudicate,
            PHASE_R_NARRATE: self._parse_narrate,
        }
        return parsers[phase](payload, projection, state, **kwargs)

    def _parse_reason(self, payload: Dict[str, Any], projection: Projection,
                      state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_R_REASON)
        cid = projection.cid
        es = coerce_float_map(payload.get("es")) or dict(kwargs.get("es") or {})
        dc = coerce_float_map(payload.get("dc")) or dict(kwargs.get("dc") or {})
        alpha = self.config.es_dc_alpha
        joint = coerce_float_map(payload.get("joint")) or joint_score(es, dc, alpha)

        raw_fine = str(payload.get("fine_label", "") or "")
        fine, recognised = canonical_fine_label(raw_fine)
        if not recognised and joint:
            fine = max(joint, key=lambda e: joint[e])
        if raw_fine and not recognised:
            result.notes.append(
                f"fine label {raw_fine!r} is outside the emotion vocabulary; "
                f"resolved to {fine!r}")
            state.register_question(OpenQuestion.create(
                "indeterminate_label",
                f"the reasoning declined to commit to a listed emotion "
                f"(it answered {raw_fine!r}); the reading is indeterminate",
                cid=cid))
        coarse = str(payload.get("coarse_label", "") or "")
        if not coarse or not labels_consistent(fine, coarse):
            coarse = coarse_of(fine)

        cot = state.causal_cots.get(cid)
        k_crit_claimed = [a for a in (payload.get("k_crit") or []) if a in SLOT_INDEX]
        if not k_crit_claimed:
            k_crit_claimed = list(kwargs.get("k_crit") or [])

        cot = CausalCoT(
            cid=cid,
            P=as_dict(payload.get("P")),
            M=as_dict(payload.get("M")),
            C=as_dict(payload.get("C")),
            cf_mhv={},
            MC=as_dict(payload.get("MC")),
            es=es, dc=dc, joint=dict(joint),
            k_crit=k_crit_claimed,
            fine_label=fine, coarse_label=coarse,
        )
        state.causal_cots[cid] = cot

        for question in (cot.MC.get("open_questions") or []):
            state.register_question(OpenQuestion.coerce(question, cid, "reasoning_open"))

        chain = state.chain(cid)
        refs = [r for r in (payload.get("refs") or [])
                if r in chain and chain.get(r).level < EvidenceLevel.EMOTION]
        result.entries.append(self.emit(
            f"causal chain concludes {fine} ({coarse}); "
            f"ES={es.get(fine, 0.0)}, DC={dc.get(fine, 0.0)}, K_crit={cot.k_crit}",
            payload={"es": es, "dc": dc, "joint": cot.joint, "k_crit": cot.k_crit,
                     "fine_label": fine, "coarse_label": coarse},
            refs=refs, cid=cid,
        ))

        result.product = {
            "P": cot.P, "M": cot.M, "C": cot.C, "MC": cot.MC,
            "es": es, "dc": dc, "joint": cot.joint, "k_crit": cot.k_crit,
            "exclusions": list(payload.get("exclusions") or []),
            "fine_label": fine, "coarse_label": coarse, "refs": refs,
        }
        return result

    def _parse_respond(self, payload: Dict[str, Any], projection: Projection,
                       state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_R_RESPOND)
        cid = projection.cid
        responses = list(payload.get("responses") or [])
        by_id = {c.ch_id: c for c in state.challenges.get(cid, [])}

        for response in responses:
            challenge = by_id.get(str(response.get("ch_id", "")))
            if challenge is not None:
                challenge.response = str(response.get("text", ""))
            chain = state.chain(cid)
            legal_refs = [
                ref for ref in (response.get("refs") or [])
                if ref in chain and chain.get(ref).level < EvidenceLevel.EMOTION
            ]
            result.entries.append(self.emit(
                f"response to {response.get('ch_id', '?')} "
                f"({response.get('mode', 'unspecified')}): "
                f"{str(response.get('text', ''))[:160]}",
                payload={"mode": response.get("mode"),
                         "new_quantities": response.get("new_quantities") or {}},
                refs=legal_refs,
                cid=cid,
            ))

        revised_labels = payload.get("revised_labels") or None
        cot = state.causal_cots.get(cid)
        if revised_labels and cot is not None:
            raw = str(revised_labels.get("fine_label", cot.fine_label))
            fine, recognised = canonical_fine_label(raw)
            if raw and not recognised:
                result.notes.append(
                    f"revised label {raw!r} is outside the emotion vocabulary; "
                    f"resolved to {fine!r}")
            cot.fine_label = fine
            cot.coarse_label = coarse_of(fine)
            result.entries.append(self.emit(
                f"main hypothesis revised to {fine} in response to a challenge",
                payload=dict(revised_labels), cid=cid,
            ))

        for question in (payload.get("new_open_questions") or []):
            state.register_question(OpenQuestion.coerce(question, cid, "response"))

        result.product = {
            "responses": responses,
            "revised_labels": revised_labels,
            "new_open_questions": list(payload.get("new_open_questions") or []),
        }
        return result

    def _parse_adjudicate(self, payload: Dict[str, Any], projection: Projection,
                          state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_R_ADJUDICATE)
        cid = projection.cid
        cot = state.causal_cots.get(cid)

        raw_fine = str(payload.get("e_fine", "") or (cot.fine_label if cot else ""))
        fine, recognised = canonical_fine_label(raw_fine)
        if raw_fine and not recognised:
            result.notes.append(
                f"verdict label {raw_fine!r} is outside the emotion vocabulary; "
                f"resolved to {fine!r}")
        coarse = str(payload.get("e_coarse", "") or "")
        if not coarse or not labels_consistent(fine, coarse):
            coarse = coarse_of(fine)

        confidence = coerce_float(payload.get("confidence"), 0.0)
        terms = coerce_float_map(payload.get("fusion_terms"))
        if not terms:
            confidence, terms = fuse_confidence(
                kwargs.get("evidence_quality", 0.5), kwargs.get("margin", 0.0),
                kwargs.get("challenge_factor", 1.0),
                kwargs.get("prototype_completeness", 0.0),
                self.config.confidence_weights,
            )

        suppression_state, suppression_prose = coerce_suppression(
            payload.get("suppression"))
        rationale = str(payload.get("rationale", ""))
        if suppression_prose and suppression_prose not in rationale:
            rationale = (rationale + " " + suppression_prose).strip()

        verdict = Verdict(
            cid=cid, e_coarse=coarse, e_fine=fine, confidence=round(confidence, 4),
            suppression=suppression_state,
            fusion_terms=dict(terms),
            prototype_completeness=coerce_float(
                payload.get("prototype_completeness"),
                float(kwargs.get("prototype_completeness", 0.0))),
            rationale=rationale,
        )
        state.verdicts[cid] = verdict

        result.entries.append(self.emit(
            f"verdict {fine} ({coarse}) at confidence {verdict.confidence}, "
            f"suppression {verdict.suppression}",
            payload=verdict.to_dict(),
            refs=[r for r in (payload.get("refs") or []) if r in state.chain(cid)],
            cid=cid,
        ))
        result.product = {
            "e_fine": fine, "e_coarse": coarse, "confidence": verdict.confidence,
            "suppression": verdict.suppression, "fusion_terms": verdict.fusion_terms,
            "prototype_completeness": verdict.prototype_completeness,
            "rationale": verdict.rationale,
        }
        return result

    def _parse_narrate(self, payload: Dict[str, Any], projection: Projection,
                       state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_R_NARRATE)
        assertions = [
            NarrativeAssertion(
                text=str(item.get("text", "")),
                refs=[str(r) for r in (item.get("refs") or [])],
                t_span=(tuple(item["t_span"]) if item.get("t_span") else None),
            )
            for item in (payload.get("assertions") or [])
        ]
        narrative = Narrative(
            video_id=state.video_id,
            text=str(payload.get("text", "")),
            assertions=assertions,
            consistency_checked=bool(
                (payload.get("consistency_check") or {}).get("passed", False)),
        )
        state.narrative = narrative
        result.product = {
            "text": narrative.text,
            "assertions": [{"text": a.text, "refs": a.refs,
                            "t_span": list(a.t_span) if a.t_span else None}
                           for a in assertions],
            "part1_proposals": list(payload.get("part1_proposals") or []),
            "part2_analysis": list(payload.get("part2_analysis") or []),
            "baseline_covered": bool(payload.get("baseline_covered", False)),
            "consistency_check": as_dict(payload.get("consistency_check")),
        }
        return result

    def fallback(self, phase: str, projection: Projection, state: MEWMState,
                 reason: str, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=phase, degraded=True, parsed=False)
        result.notes.append(f"degraded: {reason}")
        cid = projection.cid

        if phase == PHASE_R_REASON:
            es: Dict[str, float] = kwargs.get("es") or {}
            dc: Dict[str, float] = kwargs.get("dc") or {}
            joint = joint_score(es, dc, self.config.es_dc_alpha)
            fine = max(joint, key=lambda e: joint[e]) if joint else "other"
            cot = CausalCoT(
                cid=cid, es=es, dc=dc, joint=joint,
                k_crit=list(kwargs.get("k_crit") or []),
                fine_label=fine, coarse_label=coarse_of(fine),
                MC={"confidence_sources": "tool scores only; no model argumentation",
                    "open_questions": ["argumentation degraded"]},
            )
            state.causal_cots[cid] = cot
            result.product = {
                "P": {}, "M": {}, "C": {}, "MC": cot.MC, "es": es, "dc": dc,
                "joint": joint, "k_crit": cot.k_crit, "exclusions": [],
                "fine_label": fine, "coarse_label": cot.coarse_label, "refs": [],
            }
            result.entries.append(self.emit(
                f"degraded conclusion {fine} from tool scores alone",
                payload={"joint": joint}, confidence=0.4, cid=cid))
            return result

        if phase == PHASE_R_ADJUDICATE:
            cot = state.causal_cots.get(cid)
            fine = cot.fine_label if cot else "other"
            confidence, terms = fuse_confidence(
                kwargs.get("evidence_quality", 0.4), kwargs.get("margin", 0.0),
                kwargs.get("challenge_factor", 1.0),
                kwargs.get("prototype_completeness", 0.0),
                self.config.confidence_weights,
            )
            verdict = Verdict(cid=cid, e_fine=fine, e_coarse=coarse_of(fine),
                              confidence=confidence, fusion_terms=terms, degraded=True,
                              rationale="calibration incomplete: adjudication degraded")
            state.verdicts[cid] = verdict
            result.product = {
                "e_fine": fine, "e_coarse": verdict.e_coarse,
                "confidence": confidence, "suppression": "none",
                "fusion_terms": terms, "prototype_completeness":
                    kwargs.get("prototype_completeness", 0.0),
                "rationale": verdict.rationale,
            }
            return result

        if phase == PHASE_R_NARRATE:
            text = _template_narrative(state, kwargs.get("baseline_segments") or [])
            narrative = Narrative(video_id=state.video_id, text=text)
            state.narrative = narrative
            result.product = {
                "text": text, "assertions": [],
                "part1_proposals": [
                    {"proposal_id": i + 1, "onset": p.t_on, "offset": p.t_off}
                    for i, p in enumerate(state.proposals)],
                "part2_analysis": [], "baseline_covered": True,
                "consistency_check": {"passed": False, "revised": []},
            }
            return result

        result.product = {"responses": [], "revised_labels": None,
                          "new_open_questions": []}
        return result

def _template_narrative(state: MEWMState, baseline: Sequence[Dict[str, Any]]) -> str:
    if not state.proposals:
        return (f"Across video {state.video_id} the detection statistic stayed at "
                f"baseline and no micro-expression proposal was raised.")
    parts = [f"Video {state.video_id} yielded {len(state.proposals)} proposal(s)."]
    for index, proposal in enumerate(state.proposals, start=1):
        verdict = state.verdicts.get(proposal.cid)
        parts.append(
            f"Proposal {index}: frames {proposal.t_on}-{proposal.t_off} "
            f"(apex {proposal.apex}), detection peak {proposal.peak_S}; "
            f"adjudicated {verdict.e_fine if verdict else 'undetermined'} "
            f"({verdict.e_coarse if verdict else '-'}) at confidence "
            f"{verdict.confidence if verdict else 0.0}."
        )
    if baseline:
        segment = baseline[0]
        mean = float(segment.get("mean", 0.0))
        character = ("a quiet baseline" if mean < 1.0 else
                     "an elevated, unsettled baseline" if mean < 5.0 else
                     "a baseline too noisy to call neutral (which on an untrained "
                     "dynamics model usually means the residual is dominated by "
                     "unmodelled ordinary motion)")
        parts.append(
            f"Outside the proposals the error curve shows {character}; for example "
            f"frames {segment['interval'][0]}-{segment['interval'][1]} average "
            f"S = {mean:.2f}."
        )
    return " ".join(parts)

__all__ = [
    "compute_es", "joint_score", "leave_one_out_critical", "fuse_confidence",
    "challenge_grade", "ReasoningAgent",
]
