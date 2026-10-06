"""Structuring agent: builds AU sets and temporal dynamic graphs."""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..knowledge.au_anatomy import (
    AU_ROI_PRIOR, SLOT_AUS, SLOT_INDEX, au_label, candidate_entries, fit_au_at_region,
    hard_conflicts, prior_polarity, regions_of,
)
from ..orchestration.state import MEWMState, PHASE_A_ENCODE, PHASE_A_GRAPH, Projection
from ..schemas import AUDynGraph, AUEdge, AUNode, Evidence, OpenQuestion, ROIMeasurement
from .base import AgentResult, BaseAgent, coerce_float, format_evidence_lines

LOGGER = logging.getLogger(__name__)

def lagged_cross_correlation(
    a: np.ndarray, b: np.ndarray, max_lag: int = 10,
) -> Tuple[str, int, float]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 3:
        return "+", 0, 0.0
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        return "+", 0, 0.0

    best_lag, best_value = 0, 0.0
    for lag in range(-min(max_lag, n - 1), min(max_lag, n - 1) + 1):
        if lag >= 0:
            value = float(np.dot(a[: n - lag], b[lag:]))
        else:
            value = float(np.dot(a[-lag:], b[: n + lag]))
        value /= denominator
        if abs(value) > abs(best_value):
            best_lag, best_value = lag, value
    return ("+" if best_value >= 0 else "-"), best_lag, abs(best_value)

def permutation_test(a: np.ndarray, b: np.ndarray, observed: float,
                     n_permutations: int = 200, seed: int = 0) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 8:
        return 1.0
    a, b = a[:n], b[:n]
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return 1.0

    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(b - b.mean())
    magnitude = np.abs(spectrum)

    exceed = 0
    for _ in range(n_permutations):
        phases = rng.uniform(0.0, 2.0 * np.pi, magnitude.shape)
        phases[0] = 0.0
        if n % 2 == 0 and magnitude.size:
            phases[-1] = 0.0
        surrogate = np.fft.irfft(magnitude * np.exp(1j * phases), n=n)
        _polarity, _lag, value = lagged_cross_correlation(a, surrogate)
        exceed += int(value >= observed)
    return round((exceed + 1) / (n_permutations + 1), 4)

def harmonic_mean(a: float, b: float) -> float:
    a, b = abs(float(a)), abs(float(b))
    if a <= 1e-9 or b <= 1e-9:
        return 0.0
    return round(2.0 * a * b / (a + b), 5)

def build_reference_graph(
    slot_trajectories: Dict[str, np.ndarray],
    frames: Sequence[int],
    cid: str = "",
    activation_threshold: float = 0.35,
    weak_threshold: float = 0.20,
    fps: float = 30.0,
    interaction: Optional[Any] = None,
    n_permutations: int = 200,
    active_aus: Optional[Sequence[str]] = None,
    weak_aus: Optional[Sequence[str]] = None,
) -> Tuple[AUDynGraph, List[OpenQuestion]]:
    from ..engines.v2_slots import phase_profile

    graph = AUDynGraph(cid=cid)
    questions: List[OpenQuestion] = []

    for au, trajectory in slot_trajectories.items():
        trajectory = np.asarray(trajectory, dtype=np.float64).reshape(-1)
        if trajectory.size == 0 or au not in SLOT_INDEX:
            continue
        peak = float(trajectory.max())
        if peak < weak_threshold:
            continue
        if active_aus is not None and au not in set(active_aus) | set(weak_aus or ()):
            continue
        profile = phase_profile(trajectory, list(frames)[:trajectory.size],
                                hi=min(activation_threshold, peak), lo=weak_threshold)
        if profile is None:
            continue
        t_on, t_apex, t_off, peak_value, rise, decay = profile
        if active_aus is not None:
            label = "active" if au in set(active_aus) else "weak"
        else:
            label = "active" if peak_value >= activation_threshold else "weak"
        graph.nodes[au] = AUNode(
            au=au, t_on=t_on, t_apex=t_apex, t_off=t_off, peak=round(peak_value, 4),
            rise_slope=round(rise, 5), decay_slope=round(decay, 5),
            activation=label,
        )

    members = sorted(graph.nodes, key=lambda a: int(a[2:]))
    for i, source in enumerate(members):
        for target in members:
            if source == target:
                continue
            a = np.asarray(slot_trajectories[source], dtype=np.float64).reshape(-1)
            b = np.asarray(slot_trajectories[target], dtype=np.float64).reshape(-1)
            polarity, lag, peak = lagged_cross_correlation(a, b)
            if lag < 0:
                continue
            if lag == 0 and source > target:
                continue
            p_value = permutation_test(a, b, peak, n_permutations, seed=i)
            if p_value >= 0.05:
                continue

            model_weight = 0.0
            if interaction is not None:
                getter = getattr(interaction, "interaction_weight", None)
                if callable(getter):
                    model_weight = float(getter(source, target))
            if model_weight == 0.0:
                expected = prior_polarity(source, target)
                model_weight = 0.6 if expected == "+" else (-0.6 if expected == "-" else 0.0)

            model_sign = "+" if model_weight >= 0 else "-"
            conflict = bool(model_weight != 0.0 and model_sign != polarity)
            weight = float("nan") if conflict else harmonic_mean(model_weight, peak)

            graph.edges.append(AUEdge(
                source=source, target=target, polarity=polarity, lag_frames=lag,
                lag_ms=round(1000.0 * lag / max(1.0, fps), 2), weight=weight,
                w_model=round(model_weight, 4), w_obs=round(peak, 4),
                conflict=conflict, p_value=p_value,
            ))
            if conflict:
                questions.append(OpenQuestion.create(
                    "edge_sign_conflict",
                    f"edge {source}->{target}: the interaction prior expects "
                    f"{model_sign} but the measurement gives {polarity}; weight undefined",
                    cid=cid,
                ))
    return graph, questions

def graph_edit_distance(
    reference: AUDynGraph, predicted: AUDynGraph,
    node_cost: float = 1.0, edge_cost: float = 1.0, polarity_cost: float = 0.5,
) -> float:
    reference_nodes, predicted_nodes = set(reference.nodes), set(predicted.nodes)
    cost = node_cost * len(reference_nodes ^ predicted_nodes)

    def _edge_map(graph: AUDynGraph) -> Dict[Tuple[str, str], str]:
        return {(e.source, e.target): e.polarity for e in graph.edges}

    reference_edges, predicted_edges = _edge_map(reference), _edge_map(predicted)
    for key in set(reference_edges) ^ set(predicted_edges):
        cost += edge_cost
    for key in set(reference_edges) & set(predicted_edges):
        if reference_edges[key] != predicted_edges[key]:
            cost += polarity_cost
    return round(cost, 4)

class StructureAgent(BaseAgent):

    role_files = {
        PHASE_A_ENCODE: "a_agent_encode",
        PHASE_A_GRAPH: "a_agent_graph",
    }

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__("A", model, **kwargs)

    def phases(self) -> Tuple[str, ...]:
        return (PHASE_A_ENCODE, PHASE_A_GRAPH)

    def build_user_prompt(self, phase: str, projection: Projection,
                          state: MEWMState, **kwargs: Any) -> str:
        if phase == PHASE_A_ENCODE:
            return self._encode_prompt(projection, state, **kwargs)
        return self._graph_prompt(projection, state, **kwargs)

    def _encode_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        measurements: Sequence[ROIMeasurement] = kwargs.get("measurements") or []
        slot_readout: Dict[str, float] = kwargs.get("slot_readout") or {}
        proposal = state.proposal(projection.cid)
        prior = proposal.attribution if proposal else {}

        lines = [
            f"Proposal {projection.cid}, interval "
            f"[{proposal.t_on if proposal else '?'}, {proposal.t_off if proposal else '?'}]",
            "",
            "Prior evidence from the perception agent (cite these ids):",
            format_evidence_lines(projection.get("prior_evidence") or []),
            "",
            "Candidate fits per salient region (anatomy knowledge base):",
        ]
        for measurement in measurements:
            if not measurement.salient:
                continue
            candidates = candidate_entries(measurement.roi_name, measurement.direction_deg)
            rendered = ", ".join(
                f"{c['au']}({c['significance']}, {c['direction_fit']}, "
                f"fit={c['fit_score']})" for c in candidates
            )
            lines.append(
                f"- {measurement.roi_label}: magnitude {measurement.magnitude_px:.3f}px, "
                f"direction {measurement.direction_deg:.1f} deg, "
                f"coherence {measurement.coherence:.3f} -> {rendered}"
            )

        if slot_readout:
            lines.append("")
            lines.append("Object-slot activation read-out (the independent second path):")
            lines.append(", ".join(
                f"{au}={value:.3f}" for au, value in
                sorted(slot_readout.items(), key=lambda kv: -kv[1])[:8]
            ))
        preselected = list(kwargs.get("preselected_active") or [])
        if preselected:
            lines.append(
                f"Competitive pre-selection (top responses within margin of the "
                f"strongest): {preselected}; weak: "
                f"{list(kwargs.get('preselected_weak') or [])}. Treat this as the "
                f"candidate set. A micro-expression involves a handful of units, so an "
                f"activation set much larger than this indicates a threshold problem, "
                f"not a rich expression."
            )
        if kwargs.get("coherence_saturated"):
            lines.append(
                "CAVEAT: the coherence channel is saturated across regions (every ROI "
                "reports near-maximal directional agreement), so coherence carries no "
                "discriminative information here. Weight direction fit and magnitude "
                "instead, and keep the activation set conservative."
            )
        if prior:
            lines.append("")
            lines.append(f"Error attribution prior (narrows candidates only): {prior}")
        lines.append("")
        lines.append("Adjudicate the active and weak sets and return the JSON object.")
        return "\n".join(lines)

    def _graph_prompt(self, projection: Projection, state: MEWMState, **kwargs: Any) -> str:
        reference: Optional[AUDynGraph] = kwargs.get("reference_graph")
        proposal = state.proposal(projection.cid)
        lines = [
            f"Proposal {projection.cid}, interval "
            f"[{proposal.t_on if proposal else '?'}, {proposal.t_off if proposal else '?'}]",
            "",
            "Prior AU activation evidence (cite these ids):",
            format_evidence_lines(projection.get("prior_evidence") or []),
        ]
        if reference is not None:
            lines.extend([
                "",
                "Tool-computed graph (your reported parameters must match these; R3):",
                json.dumps(reference.to_dict(), ensure_ascii=False)[:3000],
            ])
        lines.append("")
        lines.append(
            "Produce the dynamic graph and translate it into a motion account with "
            "node or edge citations."
        )
        return "\n".join(lines)

    def parse(self, phase: str, payload: Dict[str, Any], projection: Projection,
              state: MEWMState, **kwargs: Any) -> AgentResult:
        if phase == PHASE_A_ENCODE:
            return self._parse_encode(payload, projection, state, **kwargs)
        return self._parse_graph(payload, projection, state, **kwargs)

    def _parse_encode(self, payload: Dict[str, Any], projection: Projection,
                      state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_A_ENCODE)
        cid = projection.cid
        active = [str(a) for a in (payload.get("active_aus") or []) if a in SLOT_INDEX]
        weak = [str(a) for a in (payload.get("weak_aus") or []) if a in SLOT_INDEX]
        fits = list(payload.get("fits") or [])

        cap = self.config.representation.max_active_aus
        if len(active) > cap:
            readout: Dict[str, float] = kwargs.get("slot_readout") or {}
            ranked = sorted(active, key=lambda au: -readout.get(au, 0.0))
            dropped = ranked[cap:]
            active = sorted(ranked[:cap], key=lambda a: int(a[2:]))
            weak = sorted(set(weak) | set(dropped), key=lambda a: int(a[2:]))
            result.notes.append(
                f"activation set trimmed from {len(ranked)} to {cap} by response rank; "
                f"demoted to weak: {dropped}"
            )

        for fit in fits:
            au = str(fit.get("au", ""))
            if au not in SLOT_INDEX:
                continue
            result.entries.append(self.emit(
                f"{au} fits the motion at {fit.get('roi', '?')} "
                f"({fit.get('direction_fit', '?')}, score {fit.get('fit_score', 0)})",
                payload=dict(fit), refs=[r for r in (fit.get("refs") or [])
                                         if r in state.chain(cid)],
                cid=cid,
            ))

        questions = list(payload.get("open_questions") or [])
        readout: Dict[str, float] = kwargs.get("slot_readout") or {}
        if readout:
            for au in active:
                if readout.get(au, 0.0) < 0.15:
                    questions.append({
                        "kind": "slot_rule_mismatch",
                        "detail": f"{au} adjudicated active by rule fitting but the slot "
                                  f"read-out gives {readout.get(au, 0.0):.3f}",
                    })
            for au, value in readout.items():
                if value >= 0.5 and au not in active and au not in weak:
                    questions.append({
                        "kind": "slot_rule_mismatch",
                        "detail": f"the slot read-out gives {au}={value:.3f} but rule "
                                  f"fitting placed it in neither set",
                    })
        for a, b in hard_conflicts(active):
            questions.append({
                "kind": "antagonist_conflict",
                "detail": f"{a} and {b} are antagonistic yet both adjudicated active",
            })

        for question in questions:
            state.register_question(OpenQuestion.coerce(question, cid,
                                                        "slot_rule_mismatch"))

        result.product = {
            "active_aus": active, "weak_aus": weak, "fits": fits,
            "slot_agreement": coerce_float(payload.get("slot_agreement"), 0.0),
            "open_questions": questions,
            "summary": str(payload.get("summary", "")),
        }
        result.entries.append(self.emit(
            f"activation set adjudicated: active {active}, weak {weak}",
            payload={"active_aus": active, "weak_aus": weak}, cid=cid,
        ))
        return result

    def _parse_graph(self, payload: Dict[str, Any], projection: Projection,
                     state: MEWMState, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=PHASE_A_GRAPH)
        cid = projection.cid
        reference: Optional[AUDynGraph] = kwargs.get("reference_graph")
        graph = reference if reference is not None else _graph_from_payload(payload, cid)
        graph.narrative = str(payload.get("graph_narrative", ""))
        state.au_graphs[cid] = graph

        for question in (payload.get("open_questions") or []):
            state.register_question(OpenQuestion.coerce(question, cid, "graph"))

        result.product = {
            "au_graph": graph.to_dict(),
            "graph_narrative": graph.narrative,
            "open_questions": list(payload.get("open_questions") or []),
            "nodes": graph.to_dict()["nodes"],
            "edges": graph.to_dict()["edges"],
        }
        result.entries.append(self.emit(
            f"AU dynamic graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges; "
            f"onset order {graph.onset_order()}",
            payload={"nodes": list(graph.nodes), "onset_order": graph.onset_order()},
            cid=cid,
        ))
        return result

    def fallback(self, phase: str, projection: Projection, state: MEWMState,
                 reason: str, **kwargs: Any) -> AgentResult:
        result = AgentResult(phase=phase, degraded=True, parsed=False)
        result.notes.append(f"degraded: {reason}; using the deterministic tool output")
        cid = projection.cid

        if phase == PHASE_A_ENCODE:
            readout: Dict[str, float] = kwargs.get("slot_readout") or {}
            active = list(kwargs.get("preselected_active") or [])
            weak = list(kwargs.get("preselected_weak") or [])
            if not active and readout:
                from ..engines.v2_slots import SlotReadout, select_active_slots
                active, weak = select_active_slots(
                    {au: SlotReadout(au, v, v, 0.0, 0.0) for au, v in readout.items()})
            fits = [
                {"au": au, "roi": (regions_of(au) or ["-"])[0],
                 "direction_fit": "PARTIAL", "fit_score": round(readout.get(au, 0.0), 4),
                 "magnitude_px": 0.0, "symmetry": 0.0, "refs": []}
                for au in active + weak
            ]
            result.product = {
                "active_aus": active, "weak_aus": weak, "fits": fits,
                "slot_agreement": 1.0,
                "open_questions": [{"kind": "degraded",
                                    "detail": "adjudicated from the slot read-out alone"}],
                "summary": "activation taken from the object-slot read-out",
            }
            result.entries.append(self.emit(
                f"activation set from the slot read-out: active {active}, weak {weak}",
                payload={"active_aus": active, "weak_aus": weak},
                confidence=0.6, cid=cid,
            ))
            return result

        reference: Optional[AUDynGraph] = kwargs.get("reference_graph")
        graph = reference if reference is not None else AUDynGraph(cid=cid)
        graph.narrative = "graph produced by the deterministic parser (agent unavailable)"
        state.au_graphs[cid] = graph
        result.product = {
            "au_graph": graph.to_dict(), "graph_narrative": graph.narrative,
            "open_questions": [], "nodes": graph.to_dict()["nodes"],
            "edges": graph.to_dict()["edges"],
        }
        result.entries.append(self.emit(
            f"AU dynamic graph from the reference parser: {len(graph.nodes)} nodes",
            payload={"nodes": list(graph.nodes)}, confidence=0.6, cid=cid,
        ))
        return result

def _graph_from_payload(payload: Dict[str, Any], cid: str) -> AUDynGraph:
    graph = AUDynGraph(cid=cid)
    raw = payload.get("au_graph") or {}
    for au, node in (raw.get("nodes") or {}).items():
        if au not in SLOT_INDEX:
            continue
        phase = node.get("phase") or [0, 0, 0]
        while len(phase) < 3:
            phase.append(phase[-1] if phase else 0)
        graph.nodes[au] = AUNode(
            au=au, t_on=int(phase[0]), t_apex=int(phase[1]), t_off=int(phase[2]),
            peak=coerce_float(node.get("peak")),
            rise_slope=coerce_float(node.get("rise_slope")),
            decay_slope=coerce_float(node.get("decay_slope")),
            activation=str(node.get("activation", "active")),
        )
    for edge in (raw.get("edges") or []):
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in SLOT_INDEX or target not in SLOT_INDEX:
            continue
        weight = edge.get("weight")
        graph.edges.append(AUEdge(
            source=source, target=target, polarity=str(edge.get("polarity", "+")),
            lag_frames=int(edge.get("lag_frames", 0) or 0),
            lag_ms=float(edge.get("lag_ms", 0.0) or 0.0),
            weight=float("nan") if weight is None else float(weight),
            w_model=float(edge.get("w_model", 0.0) or 0.0),
            w_obs=float(edge.get("w_obs", 0.0) or 0.0),
            conflict=bool(edge.get("conflict", False)),
            p_value=float(edge.get("p_value", 1.0) or 1.0),
        ))
    return graph

__all__ = [
    "lagged_cross_correlation", "permutation_test", "harmonic_mean",
    "build_reference_graph", "graph_edit_distance", "StructureAgent",
]
