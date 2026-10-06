"""Critic-verification agent: checks hypotheses and returns evidence reports."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..knowledge.au_anatomy import SLOT_INDEX, au_label
from ..knowledge.emotion_prototypes import CONTRADICTORY_AUS, core_aus
from ..orchestration.state import MEWMState, PHASE_C_CRITIC, Projection
from ..schemas import (
    CHALLENGE_TYPES, ChallengeRecord, ContractError, Evidence, RolloutRecord,
)
from .base import AgentResult, BaseAgent, coerce_float, format_evidence_lines

LOGGER = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    role_files = {PHASE_C_CRITIC: "c_agent_critic"}

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__("C", model, **kwargs)

    def phases(self) -> Tuple[str, ...]:
        return (PHASE_C_CRITIC,)

    def build_user_prompt(self, phase: str, projection: Projection,
                          state: MEWMState, **kwargs: Any) -> str:
        cid = projection.cid
        cot = projection.get("causal_cot")
        analysis: Dict[str, Any] = kwargs.get("analysis") or {}

        lines = [f"Proposal {cid}."]
        if cot is not None:
            lines.extend([
                "",
                "The chain under review:",
                f"  main hypothesis: {cot.fine_label} ({cot.coarse_label})",
                f"  ES: {json.dumps(cot.es, ensure_ascii=False)}",
                f"  DC: {json.dumps(cot.dc, ensure_ascii=False)}",
                f"  K_crit (claimed critical): {cot.k_crit}",
                f"  C layer: {json.dumps(cot.C, ensure_ascii=False)[:600]}",
            ])
        lines.extend([
            "",
            "Prior evidence (cite these ids):",
            format_evidence_lines(projection.get("prior_evidence") or []),
        ])

        if analysis:
            lines.extend([
                "",
                "Tool results already computed for you (cite the report ids):",
                f"  likelihood ratio Lambda = {analysis.get('lambda')}",
                f"  log-likelihood per hypothesis: "
                f"{json.dumps(analysis.get('log_likelihood', {}), ensure_ascii=False)}",
                f"  CFS per hypothesis: "
                f"{json.dumps(analysis.get('cfs', {}), ensure_ascii=False)}",
                f"  CFS margin = {analysis.get('cfs_margin')}",
                f"  MNI per claimed critical AU: "
                f"{json.dumps(analysis.get('mni', {}), ensure_ascii=False)}",
                f"  belief flips on masking: "
                f"{json.dumps(analysis.get('flips', {}), ensure_ascii=False)}",
                f"  template distances: "
                f"{json.dumps(analysis.get('template_distances', {}), ensure_ascii=False)}",
                f"  analysis report ids: {analysis.get('report_ids', [])}",
            ])
            thresholds = analysis.get("thresholds", {})
            lines.append(
                f"  thresholds: Lambda >= {thresholds.get('eta_lambda')}, "
                f"CFS margin >= {thresholds.get('eta_cfs')}, "
                f"MNI hallucination floor {thresholds.get('mni_floor')}"
            )

        if kwargs.get("precedents"):
            lines.append("")
            lines.append(f"Historically discriminative checks here: {kwargs['precedents']}")

        lines.extend([
            "",
            "Issue at most three challenges. Every one must cite an analysis report id. "
            "Do not propose an alternative emotion of your own.",
        ])
        return "\n".join(lines)

    def parse(self, phase: str, payload: Dict[str, Any], projection: Projection,
              state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_C_CRITIC)
        cid = projection.cid
        analysis: Dict[str, Any] = kwargs.get("analysis") or {}
        report_ids: List[str] = list(analysis.get("report_ids") or [])
        round_index = int(kwargs.get("round_index", 0))
        chain = state.chain(cid)

        accepted: List[Dict[str, Any]] = []
        for item in list(payload.get("challenges") or [])[
            : self.config.critic.max_challenges_per_round
        ]:
            ch_type = str(item.get("type", ""))
            if ch_type not in CHALLENGE_TYPES:
                result.notes.append(f"challenge dropped: unknown type {ch_type!r}")
                continue
            refs = [r for r in (item.get("refs") or []) if r in chain]
            report_id = str(item.get("analysis_report_id", "")) or (
                report_ids[0] if report_ids else "")
            try:
                record = ChallengeRecord.create(
                    ch_type=ch_type, statement=str(item.get("statement", "")),
                    refs=refs or report_ids[:1], analysis_report_id=report_id,
                    round_index=round_index, cid=cid,
                )
            except ContractError as exc:
                result.notes.append(f"challenge rejected by the arbitration rule: {exc}")
                continue
            state.add_challenge(cid, record)
            accepted.append({
                "ch_id": record.ch_id, "type": ch_type,
                "statement": record.statement, "refs": record.refs,
                "analysis_report_id": record.analysis_report_id, "final": "",
            })
            result.entries.append(self.emit(
                f"challenge ({ch_type}): {record.statement[:180]}",
                payload={"ch_id": record.ch_id, "type": ch_type,
                         "report": record.analysis_report_id},
                refs=refs, cid=cid,
            ))

        cot = state.causal_cots.get(cid)
        if cot is not None:
            cot.cf_mhv = {
                "lambda": analysis.get("lambda"),
                "cfs": analysis.get("cfs", {}),
                "cfs_margin": analysis.get("cfs_margin"),
                "mni": analysis.get("mni", {}),
                "flips": analysis.get("flips", {}),
                "template_distances": analysis.get("template_distances", {}),
                "counterfactual_statement": str(
                    payload.get("counterfactual_statement", "")),
                "challenges": [c["ch_id"] for c in accepted],
            }

        result.product = {
            "challenges": accepted,
            "lambda": analysis.get("lambda", payload.get("lambda")),
            "cfs": analysis.get("cfs", payload.get("cfs", {})),
            "cfs_margin": analysis.get("cfs_margin", payload.get("cfs_margin")),
            "mni": analysis.get("mni", payload.get("mni", {})),
            "flips": analysis.get("flips", payload.get("flips", {})),
            "template_distances": analysis.get(
                "template_distances", payload.get("template_distances", {})),
            "counterfactual_statement": str(payload.get("counterfactual_statement", "")),
            "suppression_signal": str(payload.get("suppression_signal", "none")),
        }
        return result

    def run_analysis(
        self,
        service: Any,
        observed: np.ndarray,
        main_hypothesis: str,
        candidates: Sequence[str],
        k_crit: Sequence[str],
        cid: str = "",
    ) -> Dict[str, Any]:
        critic = self.config.critic
        scores = service.score(observed, list(candidates), caller="C", cid=cid)
        ranked = scores.ranked()
        competitor = next((e for e, _ in ranked if e != main_hypothesis),
                          (ranked[0][0] if ranked else main_hypothesis))
        lambda_value = scores.likelihood_ratio(main_hypothesis, competitor)

        cfs = service.counterfactual_consistency(observed, list(candidates),
                                                 caller="C", cid=cid)
        cfs_margin = round(cfs.get(main_hypothesis, 0.0) - cfs.get(competitor, 0.0), 4)

        mni: Dict[str, float] = {}
        flips: Dict[str, bool] = {}
        for au in k_crit:
            if au not in SLOT_INDEX:
                continue
            masked = service.mask(observed, [au], list(candidates), caller="C", cid=cid)
            mni[au] = masked.mni
            flips[au] = masked.flipped

        templates = service.template_comparison(observed, main_hypothesis,
                                                caller="C", cid=cid)

        report_ids = [record.req_id for record in service.records_for(cid)]
        analysis = {
            "lambda": lambda_value,
            "log_likelihood": scores.log_likelihood,
            "dc": scores.normalised,
            "competitor": competitor,
            "cfs": cfs, "cfs_margin": cfs_margin,
            "mni": mni, "flips": flips,
            "template_distances": templates,
            "report_ids": report_ids[-8:],
            "thresholds": {
                "eta_lambda": critic.eta_lambda, "eta_cfs": critic.eta_cfs,
                "mni_floor": critic.mni_hallucination,
            },
        }
        analysis["auto_flags"] = self._auto_flags(analysis, k_crit)
        return analysis

    def _auto_flags(self, analysis: Dict[str, Any], k_crit: Sequence[str]) -> List[Dict[str, str]]:
        critic = self.config.critic
        flags: List[Dict[str, str]] = []
        if float(analysis.get("lambda", 0.0)) < critic.eta_lambda:
            flags.append({
                "type": "dynamics_inconsistency",
                "detail": f"Lambda = {analysis.get('lambda')} is below the threshold "
                          f"{critic.eta_lambda}: the main hypothesis is dynamically "
                          f"indistinguishable from {analysis.get('competitor')}",
            })
        if float(analysis.get("cfs_margin", 0.0)) < critic.eta_cfs:
            flags.append({
                "type": "dynamics_inconsistency",
                "detail": f"CFS margin = {analysis.get('cfs_margin')} is below "
                          f"{critic.eta_cfs}: the expected trajectories of the two "
                          f"hypotheses fit the measurement about equally well",
            })
        for au, value in (analysis.get("mni") or {}).items():
            if float(value) <= critic.mni_hallucination:
                flags.append({
                    "type": "insufficient_necessity",
                    "detail": f"{au} is declared critical but MNI = {value}: masking it "
                              f"leaves the belief essentially unchanged, so the claim "
                              f"has no influence on inference",
                })
        if not k_crit:
            flags.append({
                "type": "evidence_gap",
                "detail": "no critical evidence set was declared, so the leave-one-out "
                          "test cannot be reproduced",
            })
        return flags

    def grade_responses(
        self, state: MEWMState, cid: str, responses: Sequence[Dict[str, Any]],
    ) -> List[ChallengeRecord]:
        by_id = {c.ch_id: c for c in state.challenges.get(cid, [])}
        graded: List[ChallengeRecord] = []
        answered = {str(r.get("ch_id", "")): r for r in responses}
        pending = [c for c in by_id.values() if not c.final]

        unlabelled = [r for r in responses if not str(r.get("ch_id", ""))]
        if len(pending) == 1 and len(unlabelled) == 1:
            answered[pending[0].ch_id] = unlabelled[0]
        elif unlabelled and len(unlabelled) == len(pending):
            for challenge, response in zip(pending, unlabelled):
                answered[challenge.ch_id] = response

        for ch_id, challenge in by_id.items():
            if challenge.final:
                continue
            response = answered.get(ch_id)
            if response is None:
                challenge.set_final("upheld")
            else:
                mode = str(response.get("mode", "")).lower()
                if mode in {"new_evidence", "re_reasoning"}:
                    challenge.set_final("rejected")
                elif mode == "concession":
                    challenge.set_final("upheld")
                else:
                    challenge.set_final("partial")
            graded.append(challenge)
        return graded

    def fallback(self, phase: str, projection: Projection, state: MEWMState,
                 reason: str, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_C_CRITIC, degraded=True, parsed=False)
        result.notes.append(f"degraded: {reason}; raising the tool-established flags only")
        cid = projection.cid
        analysis: Dict[str, Any] = kwargs.get("analysis") or {}
        report_ids = list(analysis.get("report_ids") or [])
        accepted: List[Dict[str, Any]] = []

        for flag in (analysis.get("auto_flags") or [])[
            : self.config.critic.max_challenges_per_round
        ]:
            if not report_ids:
                break
            try:
                record = ChallengeRecord.create(
                    ch_type=flag["type"], statement=flag["detail"],
                    refs=report_ids[:1], analysis_report_id=report_ids[0],
                    round_index=int(kwargs.get("round_index", 0)), cid=cid,
                )
            except ContractError:
                continue
            state.add_challenge(cid, record)
            accepted.append({
                "ch_id": record.ch_id, "type": record.ch_type,
                "statement": record.statement, "refs": record.refs,
                "analysis_report_id": record.analysis_report_id, "final": "",
            })

        result.product = {
            "challenges": accepted,
            "lambda": analysis.get("lambda"),
            "cfs": analysis.get("cfs", {}),
            "cfs_margin": analysis.get("cfs_margin"),
            "mni": analysis.get("mni", {}),
            "flips": analysis.get("flips", {}),
            "template_distances": analysis.get("template_distances", {}),
            "counterfactual_statement": "",
            "suppression_signal": "none",
        }
        return result


__all__ = ["CriticAgent"]
