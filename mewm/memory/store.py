"""The three-layer memory of paper 3.5 and appendix E.

**Working memory W** (within a call / within a proposal)
    Latent and slot trajectories for the current proposal, the local error-curve
    segment, evidence entries under construction.  Isolated per proposal.  Overflow is
    event-compressed into episodic memory rather than truncated.

**Episodic memory E** (within a video, across proposals)
    A per-sample tree: the root holds video metadata, the slow-variable log and a
    piecewise-linear index of the *whole* ``S_t`` curve; level one holds proposals; leaves
    hold evidence entries.  Adjacent proposals are linked by baseline continuity or
    affect shift, and the belief at one proposal's offset primes the next.

**Semantic memory S** (across videos)
    Static: ``K_AU`` and ``K_E``, exact look-up, versioned, read-only.  Dynamic: a case
    library and a rollout-precedent library, indexed so the critic can prefer check types
    that discriminated well historically.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MemoryConfig
from ..schemas import (
    AUDynGraph, CandidateInterval, CausalCoT, ChallengeRecord, ErrorRecord, Evidence,
    EvidenceChain, EvidenceLevel, GateRecord, Narrative, OpenQuestion, RolloutRecord,
    SlowDigest, Verdict, VideoMeta,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------


@dataclass
class SegmentSummary:
    """A stationary run folded into one token (appendix E.2 rule i)."""

    t_start: int
    t_end: int
    mean_state: List[float]
    variance: float
    n_frames: int

    def to_text(self) -> str:
        return (f"[frames {self.t_start}-{self.t_end}] stationary, {self.n_frames} frames, "
                f"variance {self.variance:.4f}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkingMemory:
    """Per-proposal scratch space with event-based compression on overflow."""

    def __init__(self, cid: str, config: Optional[MemoryConfig] = None) -> None:
        self.cid = cid
        self.config = config or MemoryConfig()
        self.slot_trajectory: Optional[np.ndarray] = None
        self.latent_window: Dict[str, np.ndarray] = {}
        self.error_segment: Dict[str, List[float]] = {}
        self.pending: List[Evidence] = []
        self.compressed: List[SegmentSummary] = []
        self.notes: List[str] = []

    def stage(self, entry: Evidence) -> Evidence:
        self.pending.append(entry)
        return entry

    def flush(self, chain: EvidenceChain) -> List[Evidence]:
        """Commit staged entries into the proposal's evidence chain."""
        committed = []
        for entry in self.pending:
            chain.add(entry)
            committed.append(entry)
        self.pending.clear()
        return committed

    def compress_stationary(
        self, s_curve: Sequence[float], t_start: int, tau_lo: float,
    ) -> List[SegmentSummary]:
        """Fold runs of ``>= L`` frames below ``tau_lo`` into segment summaries.

        Compression is lossy for representation detail but never for *time coverage* --
        the raw curve stays in episodic memory as a piecewise-linear index, so any moment
        remains queryable.
        """
        summaries: List[SegmentSummary] = []
        run_start: Optional[int] = None
        values = list(s_curve)
        for i, value in enumerate(values + [float("inf")]):
            if value < tau_lo:
                run_start = i if run_start is None else run_start
                continue
            if run_start is not None and (i - run_start) >= self.config.stationary_run_length:
                window = values[run_start:i]
                summaries.append(SegmentSummary(
                    t_start=t_start + run_start, t_end=t_start + i - 1,
                    mean_state=[round(float(np.mean(window)), 5)],
                    variance=round(float(np.var(window)), 6), n_frames=len(window),
                ))
            run_start = None
        self.compressed.extend(summaries)
        return summaries

    def belief_tokens(self, n_active_aus: int) -> int:
        """``N_q`` -- more query tokens when more AUs are in play (appendix E.2 rule ii)."""
        return (self.config.n_query_tokens_multi_au if n_active_aus > 1
                else self.config.n_query_tokens_single_au)

    def aggregate_trajectory(self, trajectory: np.ndarray, n_queries: int) -> np.ndarray:
        """Compress a multi-step slot trajectory into ``n_queries`` belief tokens.

        Boundary frames are always retained (residual boundary injection) so the segment
        edges -- where onset and offset live -- survive the compression.
        """
        trajectory = np.atleast_2d(np.asarray(trajectory, dtype=np.float64))
        steps = trajectory.shape[0]
        if steps <= n_queries:
            return trajectory
        edges = np.linspace(0, steps, n_queries + 1).astype(int)
        pooled = np.stack([
            trajectory[edges[i]:max(edges[i] + 1, edges[i + 1])].mean(axis=0)
            for i in range(n_queries)
        ])
        pooled[0] = trajectory[0]
        pooled[-1] = trajectory[-1]
        return pooled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid, "pending": len(self.pending),
            "compressed_segments": [s.to_dict() for s in self.compressed],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------


@dataclass
class PiecewiseLinear:
    """Compressed but fully queryable index of the ``S_t`` curve (appendix E.1)."""

    knots_t: List[int] = field(default_factory=list)
    knots_v: List[float] = field(default_factory=list)
    tolerance: float = 0.05

    @classmethod
    def fit(cls, values: Sequence[float], t_start: int = 0, tolerance: float = 0.05) -> "PiecewiseLinear":
        """Ramer-Douglas-Peucker style knot selection under an L-infinity tolerance."""
        values = list(values)
        if not values:
            return cls([], [], tolerance)
        knots_t, knots_v = [t_start], [float(values[0])]
        anchor = 0
        for i in range(1, len(values)):
            # Keep a knot as soon as the straight line from the last one drifts too far.
            span = i - anchor
            if span < 2:
                continue
            worst = 0.0
            for j in range(anchor + 1, i):
                interpolated = values[anchor] + (values[i] - values[anchor]) * (j - anchor) / span
                worst = max(worst, abs(values[j] - interpolated))
            if worst > tolerance:
                knots_t.append(t_start + i - 1)
                knots_v.append(float(values[i - 1]))
                anchor = i - 1
        knots_t.append(t_start + len(values) - 1)
        knots_v.append(float(values[-1]))
        return cls(knots_t, knots_v, tolerance)

    def at(self, t: int) -> float:
        """Interpolated value at any frame -- the "look up any moment" guarantee."""
        if not self.knots_t:
            return 0.0
        if t <= self.knots_t[0]:
            return self.knots_v[0]
        if t >= self.knots_t[-1]:
            return self.knots_v[-1]
        index = int(np.searchsorted(self.knots_t, t))
        t0, t1 = self.knots_t[index - 1], self.knots_t[index]
        v0, v1 = self.knots_v[index - 1], self.knots_v[index]
        if t1 == t0:
            return v1
        return float(v0 + (v1 - v0) * (t - t0) / (t1 - t0))

    def window_summary(self, t_on: int, t_off: int) -> Dict[str, float]:
        samples = [self.at(t) for t in range(t_on, t_off + 1)] or [0.0]
        return {"mean": round(float(np.mean(samples)), 4),
                "max": round(float(np.max(samples)), 4),
                "min": round(float(np.min(samples)), 4)}

    @property
    def compression_ratio(self) -> float:
        span = (self.knots_t[-1] - self.knots_t[0] + 1) if self.knots_t else 1
        return round(len(self.knots_t) / max(1, span), 5)

    def to_dict(self) -> Dict[str, Any]:
        return {"knots_t": self.knots_t, "knots_v": [round(v, 4) for v in self.knots_v],
                "tolerance": self.tolerance, "compression_ratio": self.compression_ratio}


@dataclass
class ProposalNode:
    """Level-one node: everything produced for one proposal."""

    cid: str
    interval: Tuple[int, int]
    apex: int = 0
    attribution: Dict[str, float] = field(default_factory=dict)
    path_choice: str = "standard"
    evidence: EvidenceChain = field(default_factory=EvidenceChain)
    au_graph: Optional[AUDynGraph] = None
    causal_cot: Optional[CausalCoT] = None
    challenges: List[ChallengeRecord] = field(default_factory=list)
    rollouts: List[RolloutRecord] = field(default_factory=list)
    gate_records: List[GateRecord] = field(default_factory=list)
    revision_chain: List[Tuple[str, str, str]] = field(default_factory=list)
    verdict: Optional[Verdict] = None
    open_questions: List[OpenQuestion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cid": self.cid, "interval": list(self.interval), "apex": self.apex,
            "attribution": self.attribution, "path_choice": self.path_choice,
            "evidence": self.evidence.to_list(),
            "au_graph": self.au_graph.to_dict() if self.au_graph else None,
            "causal_cot": self.causal_cot.to_dict() if self.causal_cot else None,
            "challenges": [c.to_dict() for c in self.challenges],
            "rollouts": [r.to_dict() for r in self.rollouts],
            "gate_records": [g.to_dict() for g in self.gate_records],
            "revision_chain": [list(r) for r in self.revision_chain],
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "open_questions": [q.to_dict() for q in self.open_questions],
        }


CROSS_LINK_KINDS = ("baseline_continuity", "affect_shift", "mask_context")


@dataclass
class CrossLink:
    """Relation between two proposals -- the global evolution line."""

    source: str
    target: str
    relation: str
    evidence_refs: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpisodicMemory:
    """The per-video tree of appendix E.1."""

    def __init__(self, video_meta: VideoMeta, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        self.video_meta = video_meta
        self.slow_state_log: List[SlowDigest] = []
        self.error_curve: Optional[PiecewiseLinear] = None
        self.error_components: Dict[str, PiecewiseLinear] = {}
        self.error_record: Optional[ErrorRecord] = None
        self.proposals: Dict[str, ProposalNode] = {}
        self.cross_links: List[CrossLink] = []
        self.narrative: Optional[Narrative] = None

    # -- root ---------------------------------------------------------------

    def index_error_record(self, record: ErrorRecord) -> None:
        """Store the full-video curve as a piecewise-linear index."""
        self.error_record = record
        tolerance = self.config.piecewise_linear_tol
        self.error_curve = PiecewiseLinear.fit(record.s_curve, record.t_start, tolerance)
        self.error_components = {
            name: PiecewiseLinear.fit(values, record.t_start, tolerance)
            for name, values in (
                ("scene", record.delta_scene),
                ("physio", record.delta_physio),
                ("expr", record.delta_expr),
            ) if values
        }

    def baseline_at(self, t: int) -> float:
        """``S_t`` anywhere in the video, including outside every proposal."""
        return self.error_curve.at(t) if self.error_curve else 0.0

    def outside_proposal_summary(self, padding: int = 0) -> List[Dict[str, Any]]:
        """Segments not covered by any proposal -- the narrative's baseline material."""
        if not self.error_curve or not self.error_curve.knots_t:
            return []
        lo, hi = self.error_curve.knots_t[0], self.error_curve.knots_t[-1]
        occupied = sorted(
            (max(lo, n.interval[0] - padding), min(hi, n.interval[1] + padding))
            for n in self.proposals.values()
        )
        gaps, cursor = [], lo
        for start, end in occupied:
            if start > cursor:
                gaps.append((cursor, start - 1))
            cursor = max(cursor, end + 1)
        if cursor <= hi:
            gaps.append((cursor, hi))
        return [
            {"interval": [a, b], **self.error_curve.window_summary(a, b)}
            for a, b in gaps if b >= a
        ]

    # -- proposals ----------------------------------------------------------

    def add_proposal(self, proposal: CandidateInterval) -> ProposalNode:
        node = ProposalNode(
            cid=proposal.cid, interval=(proposal.t_on, proposal.t_off),
            apex=proposal.apex, attribution=dict(proposal.attribution),
        )
        node.evidence = EvidenceChain(proposal.cid)
        self.proposals[proposal.cid] = node
        return node

    def node(self, cid: str) -> Optional[ProposalNode]:
        return self.proposals.get(cid)

    def ordered_nodes(self) -> List[ProposalNode]:
        return sorted(self.proposals.values(), key=lambda n: n.interval[0])

    # -- cross links --------------------------------------------------------

    def link_adjacent(self, shift_threshold: float = 0.5) -> List[CrossLink]:
        """Connect consecutive proposals by baseline continuity or affect shift.

        The offset belief of one proposal becomes the prior of the next, which is what
        turns a list of independent detections into a single affective evolution line.
        """
        nodes = self.ordered_nodes()
        links: List[CrossLink] = []
        for previous, current in zip(nodes, nodes[1:]):
            gap = current.interval[0] - previous.interval[1] - 1
            baseline = self.error_curve.window_summary(
                previous.interval[1] + 1, max(previous.interval[1] + 1, current.interval[0] - 1)
            ) if (self.error_curve and gap > 0) else {"mean": 0.0}

            previous_label = previous.verdict.e_fine if previous.verdict else ""
            current_label = current.verdict.e_fine if current.verdict else ""
            if previous_label and current_label and previous_label != current_label:
                relation, detail = "affect_shift", f"{previous_label} -> {current_label}"
            elif baseline.get("mean", 0.0) < shift_threshold:
                relation, detail = "baseline_continuity", (
                    f"{gap} frames of neutral baseline between the proposals "
                    f"(mean S = {baseline.get('mean', 0.0):.3f})"
                )
            else:
                relation, detail = "mask_context", (
                    f"elevated baseline between the proposals "
                    f"(mean S = {baseline.get('mean', 0.0):.3f})"
                )
            links.append(CrossLink(previous.cid, current.cid, relation, detail=detail))
        self.cross_links = links
        return links

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": {
                "video_meta": self.video_meta.to_dict(),
                "slow_state_log": [d.to_dict() for d in self.slow_state_log],
                "error_curve": self.error_curve.to_dict() if self.error_curve else None,
                "error_components": {k: v.to_dict() for k, v in self.error_components.items()},
            },
            "proposals": {cid: node.to_dict() for cid, node in self.proposals.items()},
            "cross_links": [l.to_dict() for l in self.cross_links],
            "narrative": self.narrative.to_dict() if self.narrative else None,
        }

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                          encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# Semantic memory
# ---------------------------------------------------------------------------


@dataclass
class CaseEntry:
    """One archived reasoning fragment in the case library."""

    case_id: str
    dataset: str
    au_signature: List[str]
    evidence_level: int
    emotion: str = ""
    content: str = ""
    quality: float = 0.5
    kind: str = "support"           # support | confusion | counterexample | precedent
    outcome: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrecedentEntry:
    """Historical discriminative power of one check type, keyed by AU signature."""

    check_type: str
    au_signature: List[str]
    n_used: int = 0
    n_upheld: int = 0
    mean_margin: float = 0.0

    @property
    def discriminative_power(self) -> float:
        """Share of uses where the check actually surfaced a real problem."""
        return round(self.n_upheld / self.n_used, 4) if self.n_used else 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["discriminative_power"] = self.discriminative_power
        return data


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return round(len(sa & sb) / len(union), 4) if union else 0.0


class SemanticMemory:
    """Static knowledge bases plus the case and precedent libraries.

    Case libraries are isolated *per dataset* and never contain reference ground truth
    (appendix G.1), so retrieval cannot leak test-fold answers into a prediction.
    """

    def __init__(self, config: Optional[MemoryConfig] = None,
                 store_path: Optional[Path | str] = None) -> None:
        self.config = config or MemoryConfig()
        self.cases: List[CaseEntry] = []
        self.precedents: Dict[Tuple[str, Tuple[str, ...]], PrecedentEntry] = {}
        self.store_path = Path(store_path) if store_path else None
        if self.store_path and self.store_path.is_file():
            self.load(self.store_path)

    # -- static -------------------------------------------------------------

    @staticmethod
    def au_knowledge() -> Dict[str, Any]:
        from ..knowledge.au_anatomy import knowledge_digest
        return knowledge_digest()

    @staticmethod
    def emotion_knowledge(lang: str = "en") -> Dict[str, Any]:
        from ..knowledge.emotion_prototypes import knowledge_digest
        return knowledge_digest(lang)

    # -- dynamic ------------------------------------------------------------

    def add_case(self, case: CaseEntry) -> CaseEntry:
        self.cases.append(case)
        return case

    def record_precedent(self, check_type: str, au_signature: Sequence[str],
                         upheld: bool, margin: float = 0.0) -> PrecedentEntry:
        key = (check_type, tuple(sorted(au_signature)))
        entry = self.precedents.get(key) or PrecedentEntry(check_type, sorted(au_signature))
        entry.n_used += 1
        entry.n_upheld += int(upheld)
        entry.mean_margin = round(
            (entry.mean_margin * (entry.n_used - 1) + margin) / entry.n_used, 5
        )
        self.precedents[key] = entry
        return entry

    def best_checks(self, au_signature: Sequence[str], n: int = 3) -> List[PrecedentEntry]:
        """Check types that historically discriminated best for a similar AU signature."""
        scored = [
            (jaccard(entry.au_signature, au_signature) * entry.discriminative_power, entry)
            for entry in self.precedents.values()
        ]
        scored = [(score, entry) for score, entry in scored if score > 0]
        scored.sort(key=lambda pair: -pair[0])
        return [entry for _score, entry in scored[:n]]

    # -- persistence --------------------------------------------------------

    def save(self, path: Optional[Path | str] = None) -> Optional[Path]:
        target = Path(path) if path else self.store_path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "cases": [c.to_dict() for c in self.cases],
            "precedents": [p.to_dict() for p in self.precedents.values()],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return target

    def load(self, path: Path | str) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.cases = [CaseEntry(**c) for c in payload.get("cases", [])]
        self.precedents = {}
        for item in payload.get("precedents", []):
            item.pop("discriminative_power", None)
            entry = PrecedentEntry(**item)
            self.precedents[(entry.check_type, tuple(entry.au_signature))] = entry


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class CheckpointStore:
    """SQLite checkpoint keyed by ``video_id`` (appendix D.2).

    Persisting after every node execution buys three things the paper asks for: resume
    after interruption, step-by-step replay for audit, and a human-in-the-loop pause.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path))
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                step      INTEGER NOT NULL,
                node      TEXT NOT NULL,
                created   REAL NOT NULL,
                payload   TEXT NOT NULL,
                PRIMARY KEY (thread_id, step)
            )
        """)
        self._connection.commit()

    def save(self, thread_id: str, step: int, node: str, payload: Dict[str, Any]) -> None:
        import time
        self._connection.execute(
            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?)",
            (thread_id, step, node, time.time(),
             json.dumps(payload, ensure_ascii=False, default=str)),
        )
        self._connection.commit()

    def latest(self, thread_id: str) -> Optional[Tuple[int, str, Dict[str, Any]]]:
        row = self._connection.execute(
            "SELECT step, node, payload FROM checkpoints WHERE thread_id = ? "
            "ORDER BY step DESC LIMIT 1", (thread_id,),
        ).fetchone()
        return (row[0], row[1], json.loads(row[2])) if row else None

    def history(self, thread_id: str) -> List[Tuple[int, str]]:
        rows = self._connection.execute(
            "SELECT step, node FROM checkpoints WHERE thread_id = ? ORDER BY step",
            (thread_id,),
        ).fetchall()
        return [(int(step), str(node)) for step, node in rows]

    def replay(self, thread_id: str, step: int) -> Optional[Dict[str, Any]]:
        row = self._connection.execute(
            "SELECT payload FROM checkpoints WHERE thread_id = ? AND step = ?",
            (thread_id, step),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self._connection.close()


__all__ = [
    "SegmentSummary", "WorkingMemory", "PiecewiseLinear", "ProposalNode", "CrossLink",
    "CROSS_LINK_KINDS", "EpisodicMemory", "CaseEntry", "PrecedentEntry", "jaccard",
    "SemanticMemory", "CheckpointStore",
]
