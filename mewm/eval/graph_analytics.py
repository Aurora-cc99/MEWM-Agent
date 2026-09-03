"""Graph analytics over the AU dynamic graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..knowledge.au_anatomy import SLOT_INDEX
from ..schemas import AUDynGraph


def _edge_weight(edge: Any) -> float:
    weight = getattr(edge, "weight", 0.0)
    if weight != weight:            # NaN: a registered model/observation sign conflict
        return 0.0
    return abs(float(weight))


def adjacency(graph: AUDynGraph) -> Tuple[List[str], np.ndarray]:
    """``(node order, weighted adjacency)`` with edges directed source -> target."""
    nodes = sorted(graph.nodes, key=lambda a: int(a[2:]))
    index = {au: i for i, au in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for edge in graph.edges:
        if edge.source in index and edge.target in index:
            matrix[index[edge.source], index[edge.target]] = _edge_weight(edge)
    return nodes, matrix


def betweenness_centrality(graph: AUDynGraph) -> Dict[str, float]:
    """Brandes betweenness on the unweighted directed graph, normalised to ``[0, 1]``.

    Identifies the AU that most shortest paths run through -- the node whose removal
    would most disconnect the causal reading, which is what "key causal node" means in
    the answer.
    """
    nodes, matrix = adjacency(graph)
    n = len(nodes)
    scores = {au: 0.0 for au in nodes}
    if n < 3:
        return scores

    successors = {i: [j for j in range(n) if matrix[i, j] > 0] for i in range(n)}
    for source in range(n):
        stack: List[int] = []
        predecessors: Dict[int, List[int]] = {i: [] for i in range(n)}
        sigma = [0.0] * n
        distance = [-1] * n
        sigma[source] = 1.0
        distance[source] = 0
        queue = [source]
        while queue:
            current = queue.pop(0)
            stack.append(current)
            for neighbour in successors[current]:
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[current] + 1
                    queue.append(neighbour)
                if distance[neighbour] == distance[current] + 1:
                    sigma[neighbour] += sigma[current]
                    predecessors[neighbour].append(current)
        delta = [0.0] * n
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                if sigma[node] > 0:
                    delta[predecessor] += (sigma[predecessor] / sigma[node]) * (1 + delta[node])
            if node != source:
                scores[nodes[node]] += delta[node]

    # Normalise by the number of ordered pairs excluding the node itself.
    scale = (n - 1) * (n - 2)
    if scale > 0:
        scores = {au: round(value / scale, 4) for au, value in scores.items()}
    return scores


def acyclicity_score(graph: AUDynGraph) -> float:
    """``tr(exp(A ∘ A)) - n`` -- the NOTEARS acyclicity functional.

    Exactly zero for a DAG and positive when cycles carry weight, so a small value is a
    quantitative statement that the causal reading is near-acyclic rather than an
    assertion that it is.
    """
    _nodes, matrix = adjacency(graph)
    n = matrix.shape[0]
    if n == 0:
        return 0.0
    try:
        from scipy.linalg import expm
        value = float(np.trace(expm(matrix * matrix)) - n)
    except Exception:  # noqa: BLE001 - scipy optional; series expansion is enough here
        hadamard = matrix * matrix
        total = np.eye(n)
        term = np.eye(n)
        for k in range(1, 12):
            term = term @ hadamard / k
            total = total + term
        value = float(np.trace(total) - n)
    return round(max(0.0, value), 6)


def gcn_propagation(
    graph: AUDynGraph, node_features: Optional[Dict[str, float]] = None, steps: int = 2,
) -> Dict[str, Any]:
    """Symmetric-normalised message passing over the AU graph.

    A structural cross-check on the emotion reading: if the activation mass concentrates,
    after propagation, on the AUs the causal path names, the graph and the conclusion
    agree. ``consistency_gap`` measures how far apart they are.
    """
    nodes, matrix = adjacency(graph)
    if not nodes:
        return {"propagated_response": 0.0, "top_nodes": [], "consistency_gap": 0.0,
                "per_node": {}}

    features = np.array(
        [float((node_features or {}).get(au, graph.nodes[au].peak)) for au in nodes],
        dtype=np.float64,
    )
    # Symmetric normalisation with self-loops, as in a standard GCN layer.
    adjusted = matrix + matrix.T + np.eye(len(nodes))
    degree = adjusted.sum(axis=1)
    inverse_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-9)))
    normalised = inverse_sqrt @ adjusted @ inverse_sqrt

    state = features.copy()
    for _ in range(max(1, steps)):
        state = np.tanh(normalised @ state)

    order = np.argsort(state)[::-1]
    top = [nodes[i] for i in order[:2]]
    response = float(np.mean(np.abs(state)))
    # Gap between the propagated ranking and the raw activation ranking: large means the
    # graph structure disagrees with the measured intensities.
    raw_order = list(np.argsort(features)[::-1])
    propagated_order = list(order)
    gap = sum(abs(raw_order.index(i) - propagated_order.index(i))
              for i in range(len(nodes))) / max(1, len(nodes) ** 2 / 2)

    return {
        "propagated_response": round(response, 4),
        "top_nodes": top,
        "consistency_gap": round(float(gap), 4),
        "per_node": {nodes[i]: round(float(state[i]), 4) for i in range(len(nodes))},
    }


def main_path(graph: AUDynGraph, emotion: str = "", max_len: int = 4) -> List[str]:
    """Dominant activation path: follow onset order along the strongest edges."""
    if not graph.nodes:
        return []
    order = graph.onset_order()
    weights = {(e.source, e.target): _edge_weight(e) for e in graph.edges}

    path = [order[0]]
    while len(path) < max_len:
        current = path[-1]
        candidates = [
            (weights.get((current, au), 0.0), au)
            for au in order if au not in path
        ]
        candidates = [(w, au) for w, au in candidates if w > 0]
        if not candidates:
            remaining = [au for au in order if au not in path]
            if not remaining:
                break
            path.append(remaining[0])
            continue
        path.append(max(candidates)[1])
    if emotion:
        path.append(emotion.capitalize())
    return path


def strongest_links(graph: AUDynGraph, n: int = 3) -> List[Tuple[str, str, float]]:
    ranked = sorted(graph.edges, key=lambda e: -_edge_weight(e))[:n]
    return [(e.source, e.target, round(_edge_weight(e), 3)) for e in ranked]


def authenticity_score(
    es: Dict[str, float], dc: Dict[str, float], emotion: str,
    prototype_completeness: float = 0.0,
) -> Tuple[float, float]:
    """``(Score_auth, normalised confidence)`` for the named emotion.

    The product form is deliberate: a hypothesis needs static evidence, dynamics
    agreement *and* prototype coverage together. A sum would let one strong term carry a
    reading that the other two contradict.
    """
    static = float(es.get(emotion, 0.0))
    dynamic = float(dc.get(emotion, 0.0))
    raw = static * dynamic * max(prototype_completeness, 1e-3)
    total = sum(
        float(es.get(e, 0.0)) * float(dc.get(e, 0.0))
        for e in set(es) | set(dc)
    )
    normalised = (static * dynamic / total) if total > 1e-9 else 0.0
    return round(raw, 4), round(normalised, 4)


def confidence_bands(
    cfi: Dict[str, float], high: float = 0.25, low: float = 0.02,
) -> Dict[str, List[str]]:
    """Split the claimed AUs by masking necessity into high / medium / low bands."""
    bands: Dict[str, List[str]] = {"high": [], "medium": [], "low": []}
    for au, value in sorted(cfi.items(), key=lambda kv: -kv[1]):
        if value >= high:
            bands["high"].append(au)
        elif value >= low:
            bands["medium"].append(au)
        else:
            bands["low"].append(au)
    return bands


__all__ = [
    "adjacency", "betweenness_centrality", "acyclicity_score", "gcn_propagation",
    "main_path", "strongest_links", "authenticity_score", "confidence_bands",
]
