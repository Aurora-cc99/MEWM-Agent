"""The first evaluation layer's rollout-quality metrics (paper 4.2, layer 1).

* ``alignment_auc`` -- does the spotting curve rank event frames above non-event frames.
  Already wired, in :mod:`mewm.engines.m2_spotting`.
* **rollout prediction error** -- roll the frozen dynamics forward over a held-out future
  segment and measure how fast the prediction decays. This module.
* **counterfactual structure** -- condition the rollout on each of the eight emotion
  hypotheses in turn and measure how far apart the resulting trajectories are, plus
  whether that separation is ordered the way arousal is. This module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..knowledge.emotion_prototypes import FINE_EMOTIONS, VALENCE_AROUSAL

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Held-out rollout prediction error
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, with the all-zero case defined rather than NaN.

    Two flat slot vectors are a perfectly-predicted quiet stretch, not an undefined
    comparison; scoring that 1.0 keeps a neutral segment from poisoning the mean.
    """
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class RolloutErrorCurve:
    """Prediction error as a function of how far ahead the model was asked to see."""

    #: Mean squared error at horizon 1..k, in slot-activation units.
    mse: List[float] = field(default_factory=list)
    #: Mean cosine similarity between predicted and observed slot vectors at 1..k.
    cosine: List[float] = field(default_factory=list)
    #: Windows that contributed at each horizon. Falls off near the end of a video,
    #: where a k-step future does not exist; reported so a rising tail can be read as
    #: thin evidence rather than as degradation.
    n_windows: List[int] = field(default_factory=list)
    k_steps: int = 0
    n_sequences: int = 0

    @property
    def decay(self) -> float:
        """MSE at the last horizon minus MSE at the first -- how fast it comes apart."""
        if len(self.mse) < 2:
            return 0.0
        return round(self.mse[-1] - self.mse[0], 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mse_by_step": self.mse, "cosine_by_step": self.cosine,
            "n_windows_by_step": self.n_windows, "k_steps": self.k_steps,
            "n_sequences": self.n_sequences, "decay": self.decay,
        }


def rollout_prediction_error(
    sequences: Sequence[np.ndarray],
    dynamics: Any,
    k_steps: int = 5,
    emotion: Optional[str] = None,
    context: int = 2,
    stride: int = 1,
) -> RolloutErrorCurve:
    """Roll the frozen dynamics over held-out futures and report error by horizon.
    """
    curve = RolloutErrorCurve(k_steps=int(k_steps))
    if k_steps < 1:
        return curve

    squared: List[List[float]] = [[] for _ in range(k_steps)]
    cosines: List[List[float]] = [[] for _ in range(k_steps)]

    for sequence in sequences:
        observed = np.atleast_2d(np.asarray(sequence, dtype=np.float64))
        if observed.shape[0] < context + 1:
            continue
        curve.n_sequences += 1
        for start in range(context, observed.shape[0], max(1, stride)):
            prefix = observed[max(0, start - context):start]
            horizon = min(k_steps, observed.shape[0] - start)
            if horizon < 1:
                continue
            momentum = (prefix[-1] - prefix[-2]) if prefix.shape[0] >= 2 else None
            try:
                out = dynamics.rollout(prefix[-1], steps=k_steps, emotion=emotion,
                                       momentum=momentum)
            except Exception as exc:  # noqa: BLE001 - one bad window must not kill the sweep
                LOGGER.warning("rollout failed at frame %d: %s", start, exc)
                continue
            predicted = np.asarray(out["trajectory"], dtype=np.float64)
            for step in range(min(horizon, predicted.shape[0])):
                truth = observed[start + step]
                guess = predicted[step]
                squared[step].append(float(np.mean((guess - truth) ** 2)))
                cosines[step].append(_cosine(guess, truth))

    curve.mse = [round(float(np.mean(v)), 6) if v else 0.0 for v in squared]
    curve.cosine = [round(float(np.mean(v)), 5) if v else 0.0 for v in cosines]
    curve.n_windows = [len(v) for v in squared]
    return curve


# ---------------------------------------------------------------------------
# Counterfactual structure
# ---------------------------------------------------------------------------


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation, ties averaged. Zero when either side is constant."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.size < 2 or x.size != y.size:
        return 0.0

    def ranks(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        out = np.empty_like(values)
        out[order] = np.arange(values.size, dtype=np.float64)
        # Average the ranks of tied values, otherwise the correlation depends on the
        # arbitrary order argsort happened to break the tie in.
        for value in np.unique(values):
            mask = values == value
            if mask.sum() > 1:
                out[mask] = out[mask].mean()
        return out

    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


@dataclass
class CounterfactualStructure:
    """How far apart the eight emotion hypotheses push the same imagined future."""

    emotions: List[str] = field(default_factory=list)
    #: Symmetric ``(E, E)`` matrix of mean cosine distance between conditioned rollouts.
    divergence: List[List[float]] = field(default_factory=list)
    #: Spearman correlation between each hypothesis's mean divergence from the others and
    #: its arousal coordinate. The paper's claim is that high-arousal hypotheses imagine
    #: more extreme futures; this is the number that claim lives or dies by.
    arousal_rank_correlation: float = 0.0
    #: Mean off-diagonal divergence. Near zero means conditioning did nothing -- the
    #: model imagines the same future whatever it is told, and every downstream
    #: counterfactual comparison is measuring noise.
    mean_divergence: float = 0.0
    n_contexts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotions": self.emotions, "divergence": self.divergence,
            "arousal_rank_correlation": self.arousal_rank_correlation,
            "mean_divergence": self.mean_divergence, "n_contexts": self.n_contexts,
        }


def counterfactual_structure(
    contexts: Sequence[np.ndarray],
    dynamics: Any,
    steps: int = 8,
    emotions: Optional[Sequence[str]] = None,
) -> CounterfactualStructure:
    """The ``E x E`` trajectory-divergence matrix and its arousal rank correlation.
    """
    labels = [e for e in (emotions or FINE_EMOTIONS) if e != "other"]
    structure = CounterfactualStructure(emotions=list(labels))
    if len(labels) < 2:
        return structure

    n = len(labels)
    totals = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.float64)

    for raw in contexts:
        prefix = np.atleast_2d(np.asarray(raw, dtype=np.float64))
        if prefix.size == 0:
            continue
        momentum = (prefix[-1] - prefix[-2]) if prefix.shape[0] >= 2 else None
        trajectories: Dict[str, np.ndarray] = {}
        for emotion in labels:
            try:
                out = dynamics.rollout(prefix[-1], steps=steps, emotion=emotion,
                                       momentum=momentum)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("counterfactual rollout failed for %s: %s", emotion, exc)
                continue
            trajectories[emotion] = np.asarray(out["trajectory"], dtype=np.float64).ravel()
        if len(trajectories) < 2:
            continue
        structure.n_contexts += 1
        for i, first in enumerate(labels):
            for j, second in enumerate(labels):
                if first not in trajectories or second not in trajectories:
                    continue
                totals[i, j] += 1.0 - _cosine(trajectories[first], trajectories[second])
                counts[i, j] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.where(counts > 0, totals / np.maximum(counts, 1.0), 0.0)
    np.fill_diagonal(matrix, 0.0)
    structure.divergence = [[round(float(v), 5) for v in row] for row in matrix]

    off_diagonal = matrix[~np.eye(n, dtype=bool)]
    structure.mean_divergence = round(float(off_diagonal.mean()), 5) if off_diagonal.size else 0.0

    # Each hypothesis's distinctiveness is its mean distance from the other seven.
    distinctiveness = [float(matrix[i].sum() / max(1, n - 1)) for i in range(n)]
    arousal = [VALENCE_AROUSAL.get(e, (0.0, 0.0))[1] for e in labels]
    structure.arousal_rank_correlation = round(_spearman(distinctiveness, arousal), 5)
    return structure


__all__ = [
    "RolloutErrorCurve", "rollout_prediction_error",
    "CounterfactualStructure", "counterfactual_structure",
]
