"""Global state and the visibility projection matrix (appendix D.2, D.4).

1. **P's verification phase and all of A are blind to emotion** (and P is additionally
   blind to AU fields).  Together with the R2 vocabulary scan on the output side, this
   blocks emotion conclusions from contaminating motion observation and AU judgement in
   both directions -- what goes in and what comes out.
2. **C cannot see the verdict.**  A critic that knows the final answer constructs
   challenges that lead to it, which is not adversarial verification.
3. **Whole-curve read access is granted only to P's scan phase and R's narration phase.**
   The former localises, the latter describes the baseline outside proposals.  Every
   phase in between sees only a within-proposal digest, so global information cannot leak
   into mid-chain reasoning as an implicit prior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..schemas import (
    AUDynGraph, BudgetState, CandidateInterval, CausalCoT, ChallengeRecord, ErrorRecord,
    Evidence, EvidenceChain, EvidenceLevel, GateRecord, Narrative, OpenQuestion,
    RolloutRecord, SlowDigest, Verdict, VideoMeta,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

PHASE_P_SCAN = "P.scan"
PHASE_P_VERIFY = "P.verify"
PHASE_A_ENCODE = "A.encode"
PHASE_A_GRAPH = "A.graph"
PHASE_R_REASON = "R.reason"
PHASE_R_RESPOND = "R.respond"
PHASE_C_CRITIC = "C.critic"
PHASE_R_ADJUDICATE = "R.adjudicate"
PHASE_R_NARRATE = "R.narrate"

ALL_PHASES: Tuple[str, ...] = (
    PHASE_P_SCAN, PHASE_P_VERIFY, PHASE_A_ENCODE, PHASE_A_GRAPH, PHASE_R_REASON,
    PHASE_R_RESPOND, PHASE_C_CRITIC, PHASE_R_ADJUDICATE, PHASE_R_NARRATE,
)

PHASE_AGENT: Dict[str, str] = {
    PHASE_P_SCAN: "P", PHASE_P_VERIFY: "P",
    PHASE_A_ENCODE: "A", PHASE_A_GRAPH: "A",
    PHASE_R_REASON: "R", PHASE_R_RESPOND: "R", PHASE_R_ADJUDICATE: "R",
    PHASE_R_NARRATE: "R", PHASE_C_CRITIC: "C",
}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------


@dataclass
class MEWMState:
    """The typed global state of appendix D.2."""

    # -- long-video level
    video_meta: Optional[VideoMeta] = None
    question: str = ""
    error_record: Optional[ErrorRecord] = None
    proposals: List[CandidateInterval] = field(default_factory=list)
    macro_intervals: List[CandidateInterval] = field(default_factory=list)
    slow_state_log: List[SlowDigest] = field(default_factory=list)

    # -- proposal level, keyed by cid
    evidence_chain: Dict[str, EvidenceChain] = field(default_factory=dict)
    rollout_requests: Dict[str, List[RolloutRecord]] = field(default_factory=dict)
    au_graphs: Dict[str, AUDynGraph] = field(default_factory=dict)
    causal_cots: Dict[str, CausalCoT] = field(default_factory=dict)
    challenges: Dict[str, List[ChallengeRecord]] = field(default_factory=dict)
    verdicts: Dict[str, Verdict] = field(default_factory=dict)
    gate_records: Dict[str, List[GateRecord]] = field(default_factory=dict)
    path_choices: Dict[str, str] = field(default_factory=dict)

    # -- global output and runtime
    narrative: Optional[Narrative] = None
    open_questions: List[OpenQuestion] = field(default_factory=list)
    budget: BudgetState = field(default_factory=BudgetState)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    # -- accessors ----------------------------------------------------------

    @property
    def video_id(self) -> str:
        return self.video_meta.video_id if self.video_meta else "unknown"

    def chain(self, cid: str) -> EvidenceChain:
        if cid not in self.evidence_chain:
            self.evidence_chain[cid] = EvidenceChain(cid)
        return self.evidence_chain[cid]

    def proposal(self, cid: str) -> Optional[CandidateInterval]:
        return next((p for p in self.proposals if p.cid == cid), None)

    def add_evidence(self, cid: str, entry: Evidence) -> Evidence:
        entry.cid = cid
        return self.chain(cid).add(entry)

    def add_rollout(self, cid: str, record: RolloutRecord) -> RolloutRecord:
        self.rollout_requests.setdefault(cid, []).append(record)
        return record

    def add_challenge(self, cid: str, record: ChallengeRecord) -> ChallengeRecord:
        self.challenges.setdefault(cid, []).append(record)
        return record

    def add_gate_record(self, cid: str, record: GateRecord) -> GateRecord:
        self.gate_records.setdefault(cid, []).append(record)
        return record

    def register_question(self, question: OpenQuestion) -> OpenQuestion:
        self.open_questions.append(question)
        return question

    def open_for(self, cid: str) -> List[OpenQuestion]:
        return [q for q in self.open_questions if q.cid == cid and not q.resolved]

    def log(self, node: str, event: str, **details: Any) -> None:
        self.trace.append({"node": node, "event": event, **details})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_meta": self.video_meta.to_dict() if self.video_meta else None,
            "question": self.question,
            "proposals": [p.to_dict() for p in self.proposals],
            "macro_intervals": [m.to_dict() for m in self.macro_intervals],
            "evidence_chain": {k: v.to_list() for k, v in self.evidence_chain.items()},
            "rollout_requests": {k: [r.to_dict() for r in v]
                                 for k, v in self.rollout_requests.items()},
            "au_graphs": {k: v.to_dict() for k, v in self.au_graphs.items()},
            "causal_cots": {k: v.to_dict() for k, v in self.causal_cots.items()},
            "challenges": {k: [c.to_dict() for c in v] for k, v in self.challenges.items()},
            "verdicts": {k: v.to_dict() for k, v in self.verdicts.items()},
            "gate_records": {k: [g.to_dict() for g in v]
                             for k, v in self.gate_records.items()},
            "path_choices": dict(self.path_choices),
            "narrative": self.narrative.to_dict() if self.narrative else None,
            "open_questions": [q.to_dict() for q in self.open_questions],
            "budget": self.budget.to_dict(),
            "trace": list(self.trace),
        }


# ---------------------------------------------------------------------------
# Visibility matrix
# ---------------------------------------------------------------------------

VISIBLE = "full"
DIGEST = "digest"
HIDDEN = "hidden"

FIELDS: Tuple[str, ...] = (
    "v1_measurements", "slot_trajectory", "error_curve", "prior_evidence",
    "emotion_hypotheses", "challenges", "verdict", "other_proposals",
)

#: Table D.2, transcribed.  ``error_curve = full`` means whole-video access; ``digest``
#: means the within-proposal summary only.
VISIBILITY_MATRIX: Dict[str, Dict[str, str]] = {
    PHASE_P_SCAN: {
        "v1_measurements": VISIBLE, "slot_trajectory": VISIBLE, "error_curve": VISIBLE,
        "prior_evidence": HIDDEN, "emotion_hypotheses": HIDDEN, "challenges": HIDDEN,
        "verdict": HIDDEN, "other_proposals": HIDDEN,
    },
    PHASE_P_VERIFY: {
        "v1_measurements": VISIBLE, "slot_trajectory": HIDDEN, "error_curve": DIGEST,
        "prior_evidence": HIDDEN, "emotion_hypotheses": HIDDEN, "challenges": HIDDEN,
        "verdict": HIDDEN, "other_proposals": HIDDEN,
    },
    PHASE_A_ENCODE: {
        "v1_measurements": VISIBLE, "slot_trajectory": VISIBLE, "error_curve": HIDDEN,
        "prior_evidence": VISIBLE, "emotion_hypotheses": HIDDEN, "challenges": HIDDEN,
        "verdict": HIDDEN, "other_proposals": HIDDEN,
    },
    PHASE_A_GRAPH: {
        "v1_measurements": VISIBLE, "slot_trajectory": VISIBLE, "error_curve": DIGEST,
        "prior_evidence": VISIBLE, "emotion_hypotheses": HIDDEN, "challenges": HIDDEN,
        "verdict": HIDDEN, "other_proposals": HIDDEN,
    },
    PHASE_R_REASON: {
        "v1_measurements": DIGEST, "slot_trajectory": VISIBLE, "error_curve": HIDDEN,
        "prior_evidence": VISIBLE, "emotion_hypotheses": VISIBLE, "challenges": VISIBLE,
        "verdict": HIDDEN, "other_proposals": HIDDEN,
    },
    PHASE_R_RESPOND: {
        "v1_measurements": DIGEST, "slot_trajectory": VISIBLE, "error_curve": HIDDEN,
        "prior_evidence": VISIBLE, "emotion_hypotheses": VISIBLE, "challenges": VISIBLE,
        "verdict": HIDDEN, "other_proposals": HIDDEN,
    },
    PHASE_C_CRITIC: {
        "v1_measurements": DIGEST, "slot_trajectory": VISIBLE, "error_curve": DIGEST,
        "prior_evidence": VISIBLE, "emotion_hypotheses": VISIBLE, "challenges": VISIBLE,
        "verdict": HIDDEN,              # the critic must not see the adjudication
        "other_proposals": HIDDEN,
    },
    PHASE_R_ADJUDICATE: {
        "v1_measurements": DIGEST, "slot_trajectory": VISIBLE, "error_curve": DIGEST,
        "prior_evidence": VISIBLE, "emotion_hypotheses": VISIBLE, "challenges": VISIBLE,
        "verdict": VISIBLE, "other_proposals": HIDDEN,
    },
    PHASE_R_NARRATE: {
        "v1_measurements": DIGEST, "slot_trajectory": DIGEST, "error_curve": VISIBLE,
        "prior_evidence": VISIBLE, "emotion_hypotheses": VISIBLE, "challenges": VISIBLE,
        "verdict": VISIBLE, "other_proposals": VISIBLE,
    },
}

#: Maximum evidence level a phase may read (enforced alongside the field matrix).
PHASE_MAX_EVIDENCE_LEVEL: Dict[str, EvidenceLevel] = {
    PHASE_P_SCAN: EvidenceLevel.MOTION,
    PHASE_P_VERIFY: EvidenceLevel.MOTION,
    PHASE_A_ENCODE: EvidenceLevel.MOTION,      # "<= P"
    PHASE_A_GRAPH: EvidenceLevel.AU,           # "<= A activation"
    PHASE_R_REASON: EvidenceLevel.AU,          # "<= A"
    PHASE_R_RESPOND: EvidenceLevel.VERIFICATION,
    PHASE_C_CRITIC: EvidenceLevel.EMOTION,     # "<= R argumentation"
    PHASE_R_ADJUDICATE: EvidenceLevel.VERIFICATION,
    PHASE_R_NARRATE: EvidenceLevel.VERIFICATION,
}

#: Vocabulary bans applied to a phase's *output* (gate rule R2).
PHASE_BANS: Dict[str, Dict[str, bool]] = {
    PHASE_P_SCAN: {"ban_au": True, "ban_emotion": True},
    PHASE_P_VERIFY: {"ban_au": True, "ban_emotion": True},
    PHASE_A_ENCODE: {"ban_au": False, "ban_emotion": True},
    PHASE_A_GRAPH: {"ban_au": False, "ban_emotion": True},
    PHASE_R_REASON: {"ban_au": False, "ban_emotion": False},
    PHASE_R_RESPOND: {"ban_au": False, "ban_emotion": False},
    PHASE_C_CRITIC: {"ban_au": False, "ban_emotion": False},
    PHASE_R_ADJUDICATE: {"ban_au": False, "ban_emotion": False},
    PHASE_R_NARRATE: {"ban_au": False, "ban_emotion": False},
}


def visibility_of(phase: str, field_name: str) -> str:
    return VISIBILITY_MATRIX.get(phase, {}).get(field_name, HIDDEN)


def can_see(phase: str, field_name: str) -> bool:
    return visibility_of(phase, field_name) != HIDDEN


@dataclass
class Projection:
    """What one phase is allowed to read, already reduced to that view."""

    phase: str
    cid: str
    fields: Dict[str, Any] = field(default_factory=dict)
    withheld: List[str] = field(default_factory=list)

    def get(self, name: str, default: Any = None) -> Any:
        return self.fields.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "cid": self.cid,
                "visible": sorted(self.fields), "withheld": sorted(self.withheld)}


def project(
    state: MEWMState,
    phase: str,
    cid: str = "",
    slot_trajectory: Optional[Any] = None,
    measurements: Optional[Any] = None,
) -> Projection:
    """Reduce the global state to the view ``phase`` is permitted.

    Pure function of its inputs -- no state mutation, no hidden context -- so the
    isolation properties are decidable by inspection and by test.
    """
    if phase not in VISIBILITY_MATRIX:
        raise KeyError(f"unknown phase {phase!r}; expected one of {ALL_PHASES}")

    row = VISIBILITY_MATRIX[phase]
    proposal = state.proposal(cid) if cid else None
    out = Projection(phase=phase, cid=cid)

    def _withhold(name: str) -> None:
        out.withheld.append(name)

    def _grant(name: str, value: Any) -> None:
        """Publish a granted field, or record it as withheld when there is nothing.
        """
        if value is None or (isinstance(value, (list, tuple, dict, str)) and not value):
            _withhold(name)
        else:
            out.fields[name] = value

    # v1 measurements
    if row["v1_measurements"] == VISIBLE:
        _grant("v1_measurements", measurements)
    elif row["v1_measurements"] == DIGEST:
        _grant("v1_measurements", _measurement_digest(measurements))
    else:
        _withhold("v1_measurements")

    # slot trajectory
    if row["slot_trajectory"] == VISIBLE:
        _grant("slot_trajectory", slot_trajectory)
    elif row["slot_trajectory"] == DIGEST:
        _grant("slot_trajectory", _trajectory_digest(slot_trajectory))
    else:
        _withhold("slot_trajectory")

    # error curve: whole-video vs within-proposal digest
    if state.error_record is not None:
        if row["error_curve"] == VISIBLE:
            out.fields["error_record"] = state.error_record
            out.fields["slow_state_log"] = state.slow_state_log
        elif row["error_curve"] == DIGEST and proposal is not None:
            out.fields["error_summary"] = state.error_record.summary(
                proposal.t_on, proposal.t_off)
        else:
            _withhold("error_curve")
    else:
        _withhold("error_curve")

    # prior evidence, clipped to the phase's maximum readable level
    if row["prior_evidence"] != HIDDEN and cid:
        ceiling = PHASE_MAX_EVIDENCE_LEVEL.get(phase, EvidenceLevel.MOTION)
        # Read without creating: projection is a pure function of the state, and
        # instantiating a chain here would make it mutate what it is meant to observe.
        existing = state.evidence_chain.get(cid)
        out.fields["prior_evidence"] = [
            entry for entry in (existing.active() if existing else [])
            if entry.level <= ceiling
        ]
    else:
        _withhold("prior_evidence")

    # emotion hypotheses (the causal CoT)
    if row["emotion_hypotheses"] != HIDDEN and cid:
        cot = state.causal_cots.get(cid)
        if cot is not None:
            out.fields["causal_cot"] = cot
    else:
        _withhold("emotion_hypotheses")

    # challenges
    if row["challenges"] != HIDDEN and cid:
        out.fields["challenges"] = list(state.challenges.get(cid, []))
    else:
        _withhold("challenges")

    # verdict
    if row["verdict"] != HIDDEN and cid:
        verdict = state.verdicts.get(cid)
        if verdict is not None:
            out.fields["verdict"] = verdict
    else:
        _withhold("verdict")

    # other proposals -- only after adjudication, only for narration
    if row["other_proposals"] != HIDDEN:
        out.fields["other_verdicts"] = {
            other: verdict for other, verdict in state.verdicts.items() if other != cid
        }
        out.fields["all_proposals"] = list(state.proposals)
    else:
        _withhold("other_proposals")

    out.fields["open_questions"] = state.open_for(cid) if cid else []
    return out


def _measurement_digest(measurements: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Salient regions only, without the full per-ROI table."""
    if not measurements:
        return None
    try:
        salient = [m for m in measurements if getattr(m, "salient", False)]
        return {
            "n_regions": len(measurements),
            "n_salient": len(salient),
            "top": [
                {"roi": m.roi_label, "magnitude_px": round(m.magnitude_px, 3),
                 "direction_deg": round(m.direction_deg, 1),
                 "coherence": round(m.coherence, 3)}
                for m in sorted(salient, key=lambda x: -x.magnitude_px)[:5]
            ],
        }
    except (AttributeError, TypeError):
        return None


def _trajectory_digest(trajectory: Optional[Any]) -> Optional[Dict[str, Any]]:
    if trajectory is None:
        return None
    try:
        import numpy as np
        array = np.atleast_2d(np.asarray(trajectory, dtype=float))
        from ..knowledge.au_anatomy import SLOT_AUS
        peaks = array.max(axis=0)
        return {
            "n_frames": int(array.shape[0]),
            "peaks": {SLOT_AUS[k]: round(float(peaks[k]), 3)
                      for k in range(min(len(SLOT_AUS), peaks.size))
                      if peaks[k] > 0.05},
        }
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Leakage probe (appendix D.6)
# ---------------------------------------------------------------------------


@dataclass
class LeakageProbe:
    """Inject a marked pseudo-evidence upstream and look for it downstream.

    Reports an (injection level x detection level) matrix.  This measures whether the
    isolation actually holds in the assembled prompts, rather than only in the matrix
    that is supposed to produce it.
    """

    marker: str = "ZQX-PROBE-7734"
    injections: List[Tuple[str, str]] = field(default_factory=list)
    detections: List[Tuple[str, str]] = field(default_factory=list)

    def inject(self, phase: str, text: str) -> str:
        self.injections.append((phase, self.marker))
        return f"{text}\n[{self.marker}]"

    def check(self, phase: str, output: str) -> bool:
        found = self.marker in (output or "")
        if found:
            self.detections.append((phase, self.marker))
        return found

    def leakage_rate(self) -> float:
        return round(len(self.detections) / max(1, len(self.injections)), 4)

    def matrix(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for injected_phase, _ in self.injections:
            row = out.setdefault(injected_phase, {})
            for detected_phase, _ in self.detections:
                row[detected_phase] = row.get(detected_phase, 0) + 1
        return out


__all__ = [
    "PHASE_P_SCAN", "PHASE_P_VERIFY", "PHASE_A_ENCODE", "PHASE_A_GRAPH",
    "PHASE_R_REASON", "PHASE_R_RESPOND", "PHASE_C_CRITIC", "PHASE_R_ADJUDICATE",
    "PHASE_R_NARRATE", "ALL_PHASES", "PHASE_AGENT", "MEWMState", "VISIBLE", "DIGEST",
    "HIDDEN", "FIELDS", "VISIBILITY_MATRIX", "PHASE_MAX_EVIDENCE_LEVEL", "PHASE_BANS",
    "visibility_of", "can_see", "Projection", "project", "LeakageProbe",
]
