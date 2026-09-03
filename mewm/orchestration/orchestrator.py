"""The deterministic orchestrator (paper 3.4.1, appendix D.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, load_config
from ..engines.m4_scheduler import (
    PATH_DEEP, PATH_FAST, PATH_STANDARD, ScheduleSignal, Scheduler, build_signal,
)
from ..knowledge.emotion_prototypes import (
    coarse_of, competing_hypotheses, core_aus, prototype_completeness,
)
from ..memory.store import CheckpointStore, EpisodicMemory, WorkingMemory
from ..schemas import CandidateInterval, GateRecord, OpenQuestion, Verdict
from .gates import DualGate, GateOutcome
from .state import (
    MEWMState, PHASE_A_ENCODE, PHASE_A_GRAPH, PHASE_BANS, PHASE_C_CRITIC,
    PHASE_P_SCAN, PHASE_P_VERIFY, PHASE_R_ADJUDICATE, PHASE_R_NARRATE, PHASE_R_REASON,
    PHASE_R_RESPOND,
)

LOGGER = logging.getLogger(__name__)

# Main-graph states
STATE_SCAN = "SCAN"
STATE_FANOUT = "FANOUT"
STATE_SUBGRAPH = "PROPOSAL_SUBGRAPH"
STATE_JOIN = "JOIN"
STATE_NARRATE = "NARRATE"
STATE_ASSEMBLE = "ASSEMBLE"
STATE_END = "END"

#: Cascade target when a level's failures point at the previous level.
CASCADE_TARGET: Dict[str, str] = {
    PHASE_A_ENCODE: PHASE_P_VERIFY,
    PHASE_A_GRAPH: PHASE_A_ENCODE,
    PHASE_R_REASON: PHASE_A_GRAPH,
    PHASE_C_CRITIC: PHASE_R_REASON,
}


@dataclass
class NodeOutcome:
    """Result of running one node, gate included."""

    phase: str
    cid: str
    result: Any
    gate: Optional[GateOutcome] = None
    attempts: int = 1
    cascaded: bool = False

    @property
    def passed(self) -> bool:
        return self.gate.passed if self.gate else True


@dataclass
class ProposalContext:
    """Everything the subgraph needs for one proposal, assembled once."""

    cid: str
    interval: Tuple[int, int]
    measurements: Sequence[Any] = ()
    slot_trajectory: Optional[np.ndarray] = None
    slot_readout: Dict[str, float] = field(default_factory=dict)
    frames: Sequence[int] = ()
    reference_graph: Optional[Any] = None
    image_paths: Sequence[str] = ()
    baseline_context: str = ""
    #: Competitive activation pre-selection, passed to A-Agent as the candidate set.
    preselected_active: Sequence[str] = ()
    preselected_weak: Sequence[str] = ()
    #: True when the coherence channel carried no discriminative information.
    coherence_saturated: bool = False


class Orchestrator:
    """Executes the graph over one video."""

    def __init__(
        self,
        agents: Dict[str, Any],
        service: Any,
        config: Optional[MEWMConfig] = None,
        episodic: Optional[EpisodicMemory] = None,
        checkpoint: Optional[CheckpointStore] = None,
        lang: str = "en",
    ) -> None:
        self.agents = agents               # {"P": ..., "A": ..., "R": ..., "C": ...}
        self.service = service             # RolloutService
        self.config = config or load_config()
        self.episodic = episodic
        self.checkpoint = checkpoint
        self.gate = DualGate(lang=lang)
        self.scheduler = Scheduler(self.config.scheduler)
        self.lang = lang
        self._step = 0

    # -- entry point --------------------------------------------------------

    def run(
        self,
        state: MEWMState,
        contexts: Dict[str, ProposalContext],
        candidates: Optional[Sequence[CandidateInterval]] = None,
        max_micro_frames: int = 0,
    ) -> MEWMState:
        """Run the whole graph for one video."""
        state.budget.max_llm_calls = self.config.orchestrator.max_llm_calls_per_video

        self._scan(state, candidates or state.proposals, max_micro_frames)

        if not state.proposals:
            state.log(STATE_FANOUT, "no proposals; narrating the absence")
            self._narrate(state)
            self._assemble(state)
            return state

        for proposal in list(state.proposals):
            context = contexts.get(proposal.cid)
            if context is None:
                state.budget.degrade(f"{proposal.cid}: no context assembled; skipped")
                continue
            self._proposal_subgraph(state, context)

        self._join(state)
        self._narrate(state)
        self._assemble(state)
        return state

    # -- main graph ---------------------------------------------------------

    def _scan(self, state: MEWMState, candidates: Sequence[CandidateInterval],
              max_micro_frames: int) -> None:
        all_candidates = list(candidates)
        state.log(STATE_SCAN, "start", n_candidates=len(all_candidates))
        # One call per whole video used to review every duration-eligible candidate at
        # once, each carrying a prose "notes" field -- long enough on a video with many
        # candidates to truncate mid-response (observed: gemini-3-flash cut off
        # mid-string on a 29-candidate call), which invalidates that whole response's
        # JSON. Chunking bounds each call's expected length to one batch's worth of
        # candidates regardless of how long the full list gets, and confines a
        # truncated response's fallback to just the batch that produced it.
        batch_size = max(1, self.config.orchestrator.p_scan_batch_size)
        micro_ids: set = set()
        for start in range(0, len(all_candidates), batch_size):
            batch = all_candidates[start:start + batch_size]
            outcome = self._run_node(
                state, PHASE_P_SCAN, cid="",
                candidates=batch, max_micro_frames=max_micro_frames,
            )
            products = outcome.result.product.get("proposals") or []
            # P-Agent's own "channel" call has to actually gate the subgraph, not just
            # get logged: a proposal it re-routes to "macro" (p_agent_scan.md rule 4)
            # must leave the micro-expression pipeline the same way an unconfirmed one
            # does, or the call has no effect and an over-long interval still gets
            # narrated as a micro-expression no matter what P-Agent decided.
            if products:
                micro_ids |= {
                    str(item.get("cid"))
                    for item in products
                    if item.get("confirmed", True)
                    and str(item.get("channel", "micro")) == "micro"
                }
            else:
                # This batch's call degraded or came back empty; trust the
                # deterministic engine's own channel tag for just these candidates,
                # the same fallback the single-call design used across the whole
                # video, now scoped to the batch that actually needs it.
                micro_ids |= {p.cid for p in batch if p.channel == "micro"}
        state.proposals = [p for p in all_candidates if p.cid in micro_ids]
        state.log(STATE_SCAN, "done", n_confirmed=len(state.proposals))
        self._save(state, STATE_SCAN)

    def _join(self, state: MEWMState) -> None:
        pending = [p.cid for p in state.proposals if p.cid not in state.verdicts]
        if pending:
            state.budget.degrade(f"JOIN: proposals without a verdict: {pending}")
        if self.episodic is not None:
            links = self.episodic.link_adjacent()
            state.log(STATE_JOIN, "cross-links built", n_links=len(links))
        self._save(state, STATE_JOIN)

    def _narrate(self, state: MEWMState) -> None:
        baseline = self.episodic.outside_proposal_summary() if self.episodic else []
        links = self.episodic.cross_links if self.episodic else []
        attempts = 0
        while True:
            attempts += 1
            outcome = self._run_node(
                state, PHASE_R_NARRATE, cid="",
                baseline_segments=baseline, cross_links=links,
            )
            check = outcome.result.product.get("consistency_check") or {}
            if check.get("passed", True) or attempts > self.config.orchestrator.max_narrative_revisions:
                break
            state.log(STATE_NARRATE, "self-check failed; revising", attempt=attempts)
        if state.narrative is not None and self.episodic is not None:
            state.narrative.revisions = attempts - 1
            self.episodic.narrative = state.narrative
        self._save(state, STATE_NARRATE)

    def _assemble(self, state: MEWMState) -> None:
        state.log(STATE_ASSEMBLE, "done",
                  n_proposals=len(state.proposals),
                  n_verdicts=len(state.verdicts),
                  degradations=len(state.budget.degradations))
        self._save(state, STATE_ASSEMBLE)

    # -- proposal subgraph --------------------------------------------------

    def _proposal_subgraph(self, state: MEWMState, context: ProposalContext) -> None:
        cid = context.cid
        state.log(STATE_SUBGRAPH, "start", cid=cid)

        verify = self._run_node(state, PHASE_P_VERIFY, cid,
                                measurements=context.measurements,
                                image_paths=context.image_paths)
        encode = self._run_node(state, PHASE_A_ENCODE, cid,
                                measurements=context.measurements,
                                slot_readout=context.slot_readout,
                                preselected_active=context.preselected_active,
                                preselected_weak=context.preselected_weak,
                                coherence_saturated=context.coherence_saturated)
        if not encode.passed and self._should_cascade(state, cid, PHASE_A_ENCODE):
            verify = self._run_node(state, PHASE_P_VERIFY, cid,
                                    measurements=context.measurements,
                                    revision=encode.gate.revision_instruction if encode.gate else "")
            encode = self._run_node(state, PHASE_A_ENCODE, cid,
                                    measurements=context.measurements,
                                    slot_readout=context.slot_readout,
                                    preselected_active=context.preselected_active,
                                    preselected_weak=context.preselected_weak,
                                    coherence_saturated=context.coherence_saturated)

        active = list(encode.result.product.get("active_aus") or [])
        weak = list(encode.result.product.get("weak_aus") or [])

        self._run_node(state, PHASE_A_GRAPH, cid,
                       reference_graph=context.reference_graph,
                       slot_trajectory=context.slot_trajectory)

        # -- reasoning: tool scores first, then the argument -----------------
        observed = context.slot_trajectory
        candidates = self._candidate_set(active, weak)
        es, dc, k_crit, scores = self._score_hypotheses(observed, active, weak,
                                                        candidates, cid)

        # ``measurements`` has to be handed over explicitly. The visibility matrix grants
        # both R phases a digest of it, but a digest of ``None`` is ``None``, and because
        # the field is granted rather than hidden it is never listed as withheld either --
        # so the agent sees a null it was told it could rely on, and adjudicates every
        # proposal at confidence 0 with the rationale "no measurements were supplied".
        reason = self._run_node(state, PHASE_R_REASON, cid,
                                measurements=context.measurements,
                                active_aus=active, weak_aus=weak, es=es, dc=dc,
                                k_crit=k_crit, slot_trajectory=observed)

        cot = state.causal_cots.get(cid)
        main = cot.fine_label if cot else (max(es, key=lambda e: es[e]) if es else "other")

        # -- routing ---------------------------------------------------------
        signal = build_signal(
            cid, es_scores=es, likelihood=(scores.log_likelihood if scores else None),
            belief_variance=float(kwargs_get(context, "belief_variance", 0.5)),
            open_questions=len(state.open_for(cid)),
        )
        signal = self.scheduler.route(signal)
        state.path_choices[cid] = signal.path
        state.log(STATE_SUBGRAPH, "routed", cid=cid, path=signal.path,
                  reasons=signal.reasons)

        challenge_factor = 1.0
        if signal.path in (PATH_STANDARD, PATH_DEEP):
            challenge_factor = self._challenge_loop(
                state, context, main, candidates,
                (cot.k_crit if cot else k_crit), observed,
                rounds=(self.config.orchestrator.max_challenge_rounds
                        if signal.path == PATH_DEEP else 1),
            )

        # -- adjudication -----------------------------------------------------
        completeness = prototype_completeness(main, active)
        margin = 0.0
        if len(es) >= 2:
            ordered = sorted(es.values(), reverse=True)
            margin = float(ordered[0] - ordered[1])
        quality = float(np.mean([e.confidence for e in state.chain(cid).active()])
                        if len(state.chain(cid)) else 0.5)

        self._run_node(
            state, PHASE_R_ADJUDICATE, cid,
            measurements=context.measurements,
            prototype_completeness=completeness, evidence_quality=quality,
            margin=margin, challenge_factor=challenge_factor,
            baseline_context=context.baseline_context,
            slot_trajectory=observed,
        )

        verdict = state.verdicts.get(cid)
        if verdict is not None:
            self.scheduler.reroute_after_adjudication(
                signal, verdict.confidence, verdict.suppression,
                challenge_upheld=any(c.final == "upheld"
                                     for c in state.challenges.get(cid, [])),
            )
            state.path_choices[cid] = signal.path

        if self.episodic is not None:
            node = self.episodic.node(cid)
            if node is not None:
                node.path_choice = signal.path
                node.evidence = state.chain(cid)
                node.au_graph = state.au_graphs.get(cid)
                node.causal_cot = state.causal_cots.get(cid)
                node.challenges = state.challenges.get(cid, [])
                node.rollouts = state.rollout_requests.get(cid, [])
                node.gate_records = state.gate_records.get(cid, [])
                node.verdict = verdict
                node.open_questions = state.open_for(cid)

        self._save(state, f"{STATE_SUBGRAPH}:{cid}")

    # -- challenge loop -----------------------------------------------------

    def _challenge_loop(
        self, state: MEWMState, context: ProposalContext, main: str,
        candidates: Sequence[str], k_crit: Sequence[str],
        observed: Optional[np.ndarray], rounds: int,
    ) -> float:
        """Bounded C <-> R loop; returns the monotone challenge factor."""
        from ..agents.reasoning import challenge_grade

        critic = self.agents.get("C")
        cid = context.cid
        if critic is None or observed is None:
            return 1.0

        for round_index in range(max(1, rounds)):
            analysis = critic.run_analysis(
                self.service, observed, main, candidates, k_crit, cid=cid)
            for record in self.service.records_for(cid):
                if record not in state.rollout_requests.get(cid, []):
                    state.add_rollout(cid, record)

            self._run_node(state, PHASE_C_CRITIC, cid,
                           analysis=analysis, round_index=round_index,
                           slot_trajectory=observed)

            pending = [c for c in state.challenges.get(cid, []) if not c.final]
            if not pending:
                break

            respond = self._run_node(state, PHASE_R_RESPOND, cid,
                                     slot_trajectory=observed)
            critic.grade_responses(state, cid,
                                   respond.result.product.get("responses") or [])

            upheld = [c for c in state.challenges.get(cid, []) if c.final == "upheld"]
            if not upheld:
                break
            state.log(PHASE_C_CRITIC, "challenge upheld", cid=cid,
                      round=round_index, n_upheld=len(upheld))

        return challenge_grade(state.challenges.get(cid, []))

    # -- node execution -----------------------------------------------------

    def _run_node(self, state: MEWMState, phase: str, cid: str, **kwargs: Any) -> NodeOutcome:
        """Run one node with gating and bounded revision."""
        agent = self.agents.get(_agent_of(phase))
        if agent is None:
            raise KeyError(f"no agent registered for phase {phase!r}")

        max_retries = self.config.orchestrator.max_gate_retries
        revision = kwargs.pop("revision", "")
        outcome: Optional[NodeOutcome] = None

        for attempt in range(max_retries + 1):
            result = agent.run(phase, state, cid=cid, revision=revision, **kwargs)
            bans = PHASE_BANS.get(phase, {})
            gate = self.gate.evaluate(
                phase, result.gate_payload(), state.chain(cid) if cid else state.chain("_"),
                retries_remaining=max_retries - attempt,
                retry_idx=attempt,
                ban_au=bans.get("ban_au", False),
                ban_emotion=bans.get("ban_emotion", False),
                recompute=kwargs.get("recompute"),
                au_graph=state.au_graphs.get(cid),
                active_aus=result.product.get("active_aus", []) or [],
                open_questions=state.open_for(cid),
                causal_cot=state.causal_cots.get(cid),
                verdict=state.verdicts.get(cid) if phase == PHASE_R_ADJUDICATE else None,
            )
            if cid:
                state.add_gate_record(cid, gate.consistency)
                state.add_gate_record(cid, gate.sufficiency)

            outcome = NodeOutcome(phase, cid, result, gate, attempt + 1)
            if gate.passed or result.degraded:
                break
            revision = gate.revision_instruction
            state.log(phase, "gate failed; revising", cid=cid, attempt=attempt + 1,
                      failed=gate.failed_rules)

        assert outcome is not None
        if not outcome.passed and outcome.gate is not None:
            gate = outcome.gate
            # Name the failing gate and its reason: "gate never passed ([])" is useless
            # when the consistency rules passed and it was sufficiency that blocked.
            if not gate.consistency.passed:
                reason = f"consistency rules {gate.consistency.failed_rules}"
            else:
                reason = (f"sufficiency graded '{gate.sufficiency.grade}', "
                          f"missing {gate.sufficiency.failed_rules}")
            state.budget.degrade(
                f"{phase}{'@' + cid if cid else ''}: gate never passed -- {reason}")
        self._commit(state, cid, outcome)
        return outcome

    def _commit(self, state: MEWMState, cid: str, outcome: NodeOutcome) -> None:
        """Write an accepted node's evidence into the chain."""
        chain = state.chain(cid) if cid else state.chain("_")
        for entry in outcome.result.entries:
            if entry.eid in chain:
                continue
            entry.cid = cid or entry.cid
            try:
                chain.add(entry)
            except Exception as exc:  # noqa: BLE001 - an illegal entry is dropped, not fatal
                state.log(outcome.phase, "evidence rejected", cid=cid, reason=str(exc))

    def _should_cascade(self, state: MEWMState, cid: str, phase: str) -> bool:
        used = state.budget.cascade_rollbacks.get(cid, 0)
        if used >= self.config.orchestrator.max_cascade_rollbacks:
            return False
        if phase not in CASCADE_TARGET:
            return False
        state.budget.cascade_rollbacks[cid] = used + 1
        state.log(phase, "cascading back", cid=cid, target=CASCADE_TARGET[phase])
        return True

    # -- scoring ------------------------------------------------------------

    def _candidate_set(self, active: Sequence[str], weak: Sequence[str]) -> List[str]:
        """Hypotheses worth scoring: those sharing an AU with the observation."""
        from ..knowledge.emotion_prototypes import FINE_EMOTIONS
        observed = set(active) | set(weak)
        scored = [
            (len(set(core_aus(emotion)) & observed), emotion)
            for emotion in FINE_EMOTIONS
        ]
        scored.sort(key=lambda pair: -pair[0])
        candidates = [emotion for overlap, emotion in scored if overlap > 0][:4]
        if "other" not in candidates:
            candidates.append("other")
        return candidates or list(FINE_EMOTIONS[:4])

    def _score_hypotheses(
        self, observed: Optional[np.ndarray], active: Sequence[str],
        weak: Sequence[str], candidates: Sequence[str], cid: str,
    ) -> Tuple[Dict[str, float], Dict[str, float], List[str], Any]:
        """Tool-side ES / DC / K_crit, computed before the agent argues."""
        from ..agents.reasoning import compute_es, leave_one_out_critical

        es = compute_es(active, weak, candidates)
        scores = None
        dc: Dict[str, float] = {}
        if observed is not None and np.size(observed):
            scores = self.service.score(observed, list(candidates), caller="R", cid=cid)
            dc = scores.normalised
        k_crit = leave_one_out_critical(active, weak, candidates, dc,
                                        self.config.es_dc_alpha)
        return es, dc, k_crit, scores

    # -- checkpointing ------------------------------------------------------

    def _save(self, state: MEWMState, node: str) -> None:
        if self.checkpoint is None:
            return
        self._step += 1
        try:
            self.checkpoint.save(state.video_id, self._step, node, state.to_dict())
        except Exception as exc:  # noqa: BLE001 - checkpointing must not break the run
            LOGGER.warning("checkpoint failed at %s: %s", node, exc)


def _agent_of(phase: str) -> str:
    from .state import PHASE_AGENT
    return PHASE_AGENT[phase]


def kwargs_get(context: ProposalContext, name: str, default: Any) -> Any:
    return getattr(context, name, default)


__all__ = [
    "STATE_SCAN", "STATE_FANOUT", "STATE_SUBGRAPH", "STATE_JOIN", "STATE_NARRATE",
    "STATE_ASSEMBLE", "STATE_END", "CASCADE_TARGET", "NodeOutcome", "ProposalContext",
    "Orchestrator",
]
