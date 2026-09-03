"""V3 -- two-timescale latent state decomposition (paper 3.2.3, appendix B.3).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import RepresentationConfig
from ..schemas import SlowDigest

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
    _TORCH = False


# ---------------------------------------------------------------------------
# Slow variable: random walk + Kalman correction
# ---------------------------------------------------------------------------


class SlowStateTracker:
    """Scalar-covariance Kalman filter over the slow latent (appendix B.3).

    Cost is ``O(d_s)`` per update and storage does not grow with video length, which is
    what lets this run over tens of thousands of frames.  A change point resets the
    covariance and flags the following window as low-confidence recovery rather than
    letting the filter quietly mis-track a scene cut.
    """

    def __init__(self, dim: int = 256, config: Optional[RepresentationConfig] = None) -> None:
        self.config = config or RepresentationConfig()
        self.dim = dim
        self.mu = np.zeros(dim, dtype=np.float64)
        self.sigma = 1.0
        self.sigma0 = 1.0
        self.Q = float(self.config.slow_process_noise)
        self.obs_noise = float(self.config.slow_obs_noise)
        self.initialised = False
        self.log: List[SlowDigest] = []
        self._nll_run = 0
        self._last_gain = 0.0

    def update(self, observation: np.ndarray, t: int, n_samples: int = 1) -> Tuple[np.ndarray, float, bool]:
        """One discrete update; returns ``(mu, kalman_gain, reset_flag)``.

        ``n_samples`` is the number of frames the observation summarises: a small
        segment yields a conservative update, which is the point of quantifying scene
        drift uncertainty explicitly.
        """
        observation = np.asarray(observation, dtype=np.float64).reshape(-1)
        if observation.shape[0] != self.dim:
            observation = _resize(observation, self.dim)

        if not self.initialised:
            self.mu = observation.copy()
            self.initialised = True
            self.log.append(SlowDigest(t=t, summary=_digest(self.mu), kalman_gain=1.0))
            return self.mu.copy(), 1.0, False

        prior_var = self.sigma + self.Q
        obs_var = self.obs_noise / max(1, n_samples)
        gain = prior_var / (prior_var + obs_var)          # H_k
        innovation = observation - self.mu

        reset = self._check_changepoint(innovation, prior_var + obs_var, t)
        self.mu = self.mu + gain * innovation             # mu_k
        self.sigma = (1.0 - gain) * prior_var             # sigma_k
        self._last_gain = float(gain)

        self.log.append(SlowDigest(t=t, summary=_digest(self.mu),
                                   kalman_gain=round(float(gain), 5), reset=reset))
        return self.mu.copy(), float(gain), reset

    def predict(self) -> np.ndarray:
        """One-step forecast ``z^s_{t|t-1}``; the random walk leaves the mean unchanged."""
        return self.mu.copy()

    def _check_changepoint(self, innovation: np.ndarray, variance: float, t: int) -> bool:
        """Sustained observation-likelihood collapse means the scene actually changed."""
        nll = 0.5 * float(innovation @ innovation) / max(variance, 1e-8)
        nll = nll / max(1, self.dim)
        if nll > self.config.changepoint_nll:
            self._nll_run += 1
        else:
            self._nll_run = 0
        if self._nll_run >= self.config.changepoint_frames:
            self.sigma = self.sigma0
            self._nll_run = 0
            LOGGER.info("slow-state change point at frame %d; covariance reset", t)
            return True
        return False

    @property
    def kalman_gain(self) -> float:
        return self._last_gain

    def absorption_ratio(self) -> float:
        """``h`` of proposition B.3 -- how much of a step the slow term can absorb.

        In a stationary stretch the gain converges to a small value, so ``h << 1`` and a
        micro-expression transition survives into the expression residual.
        """
        return self._last_gain

    def reset_spans(self, window: int) -> List[Tuple[int, int]]:
        return [(entry.t, entry.t + window) for entry in self.log if entry.reset]


def _digest(vector: np.ndarray, size: int = 8) -> List[float]:
    """Small fixed-width summary of a latent vector, for the slow-variable log."""
    if vector.size == 0:
        return [0.0] * size
    chunks = np.array_split(vector, min(size, vector.size))
    return [round(float(c.mean()), 5) for c in chunks] + [0.0] * max(0, size - len(chunks))


def _resize(vector: np.ndarray, dim: int) -> np.ndarray:
    if vector.size == dim:
        return vector
    out = np.zeros(dim, dtype=np.float64)
    n = min(dim, vector.size)
    out[:n] = vector[:n]
    return out


# ---------------------------------------------------------------------------
# Belief variable: rolling posterior over emotions
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """``q(z^e_t | o_{<=t})`` as a categorical posterior over the fine emotion set."""

    labels: List[str]
    logits: np.ndarray
    decay: float = 0.92                # evidence half-life; prevents unbounded certainty

    @classmethod
    def uniform(cls, labels: Sequence[str], decay: float = 0.92) -> "BeliefState":
        labels = list(labels)
        return cls(labels, np.zeros(len(labels), dtype=np.float64), decay)

    @property
    def probabilities(self) -> np.ndarray:
        shifted = self.logits - self.logits.max()
        exponentiated = np.exp(shifted)
        return exponentiated / max(exponentiated.sum(), 1e-12)

    @property
    def mean_label(self) -> str:
        return self.labels[int(np.argmax(self.probabilities))]

    @property
    def entropy(self) -> float:
        p = self.probabilities
        return float(-(p * np.log(p + 1e-12)).sum())

    @property
    def variance(self) -> float:
        """Normalised entropy in ``[0, 1]`` -- ``Var[z^e]`` for the M4 routing signal."""
        return round(self.entropy / math.log(max(2, len(self.labels))), 5)

    @property
    def margin(self) -> float:
        """Gap between the top two hypotheses -- the ``Delta`` of the fast-path test."""
        ordered = np.sort(self.probabilities)[::-1]
        return float(ordered[0] - ordered[1]) if ordered.size >= 2 else 1.0

    def update(self, log_evidence: Dict[str, float], weight: float = 1.0) -> "BeliefState":
        """Accumulate per-emotion log evidence with geometric forgetting."""
        self.logits *= self.decay
        for i, label in enumerate(self.labels):
            self.logits[i] += weight * float(log_evidence.get(label, 0.0))
        return self

    def kl_to(self, other: "BeliefState") -> float:
        """``D_KL(self || other)`` -- the quantity ``MNI_k`` reports (eq. 10)."""
        p, q = self.probabilities, other.probabilities
        return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())

    def top(self, n: int = 3) -> List[Tuple[str, float]]:
        p = self.probabilities
        order = np.argsort(p)[::-1][:n]
        return [(self.labels[i], round(float(p[i]), 4)) for i in order]

    def copy(self) -> "BeliefState":
        return BeliefState(list(self.labels), self.logits.copy(), self.decay)

    def to_dict(self) -> Dict[str, object]:
        return {
            "top": self.top(), "entropy": round(self.entropy, 4),
            "variance": self.variance, "margin": round(self.margin, 4),
            "argmax": self.mean_label,
        }


# ---------------------------------------------------------------------------
# Learned encoder
# ---------------------------------------------------------------------------

if _TORCH:

    class LatentEncoder(nn.Module):
        """Maps measurements + flow summary to ``(z^s, z^m, z^e)``.

        The fast head predicts a *velocity* rather than a position so that the flow
        constraint ``L_flow`` has something to bind to directly.
        """

        def __init__(self, config: Optional[RepresentationConfig] = None) -> None:
            super().__init__()
            self.config = config or RepresentationConfig()
            from ..knowledge.au_anatomy import N_ROI as n_roi

            in_dim = n_roi * 4
            hidden = 512
            self.trunk = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            )
            self.slow_head = nn.Linear(hidden, self.config.slow_dim)
            self.fast_head = nn.Linear(hidden, self.config.fast_dim)
            self.belief_head = nn.Linear(hidden, self.config.belief_dim)
            # g_phi: optical flow -> latent velocity (the anchor of L_flow)
            self.flow_to_velocity = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, self.config.fast_dim),
            )

        def forward(self, measurements: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            batch = measurements.shape[0]
            flat = measurements.reshape(batch, -1)
            features = self.trunk(flat)
            return {
                "z_slow": self.slow_head(features),
                "z_fast": self.fast_head(features),
                "z_belief": self.belief_head(features),
                "flow_velocity": self.flow_to_velocity(flat),
            }

        def flow_constraint_loss(
            self, z_fast: "torch.Tensor", flow_velocity: "torch.Tensor"
        ) -> "torch.Tensor":
            """``L_flow = sum_t ||(z^m_{t+1} - z^m_t) - g_phi(F_t)||^2`` (appendix B.2)."""
            if z_fast.shape[0] < 2:
                return z_fast.new_zeros(())
            delta = z_fast[1:] - z_fast[:-1]
            return ((delta - flow_velocity[:-1]) ** 2).sum(dim=-1).mean()

        @staticmethod
        def slow_consistency_loss(
            observation: "torch.Tensor", prior_mean: "torch.Tensor",
            prior_var: "torch.Tensor",
        ) -> "torch.Tensor":
            """``L_slow`` -- the Gaussian-marginal soft constraint of appendix B.2."""
            residual = observation - prior_mean
            return (residual ** 2 / prior_var.clamp(min=1e-6)).sum(dim=-1).mean()

else:  # pragma: no cover

    class LatentEncoder:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("LatentEncoder needs PyTorch.")


# ---------------------------------------------------------------------------
# Streaming composer
# ---------------------------------------------------------------------------


class LatentComposer:
    """Per-frame recursion that assembles ``z_t`` and the slow-variable log.

    Analytic by default (no trained weights required), so the spotting path can run on a
    fresh checkout; when a :class:`LatentEncoder` is supplied its heads take over.
    """

    def __init__(
        self,
        config: Optional[RepresentationConfig] = None,
        encoder: Optional["LatentEncoder"] = None,
        emotion_labels: Optional[Sequence[str]] = None,
    ) -> None:
        self.config = config or RepresentationConfig()
        self.encoder = encoder
        self.slow = SlowStateTracker(self.config.slow_dim, self.config)
        from ..knowledge.emotion_prototypes import FINE_EMOTIONS
        self.belief = BeliefState.uniform(emotion_labels or FINE_EMOTIONS)
        self._prev_fast: Optional[np.ndarray] = None
        self._segment: List[np.ndarray] = []
        self._segment_start: int = 0

    def step(
        self,
        measurement_matrix: np.ndarray,
        t: int,
        segment_size: int = 30,
    ) -> Dict[str, np.ndarray | float | bool]:
        """Advance one frame; the slow state updates once per ``segment_size`` frames."""
        features = np.asarray(measurement_matrix, dtype=np.float64).reshape(-1)

        if self.encoder is not None and _TORCH:
            with torch.no_grad():
                tensor = torch.from_numpy(
                    np.asarray(measurement_matrix, dtype=np.float32)
                )[None]
                heads = self.encoder(tensor)
            z_fast = heads["z_fast"][0].cpu().numpy()
            slow_observation = heads["z_slow"][0].cpu().numpy()
        else:
            z_fast = _resize(features, self.config.fast_dim)
            slow_observation = _resize(features, self.config.slow_dim)

        self._segment.append(slow_observation)
        reset = False
        gain = self.slow.kalman_gain
        if len(self._segment) >= segment_size:
            mean_observation = np.mean(np.stack(self._segment), axis=0)
            _mu, gain, reset = self.slow.update(mean_observation, t, len(self._segment))
            self._segment.clear()
            self._segment_start = t

        velocity = (z_fast - self._prev_fast) if self._prev_fast is not None else np.zeros_like(z_fast)
        self._prev_fast = z_fast

        return {
            "z_slow": self.slow.mu.copy(),
            "z_fast": z_fast,
            "z_velocity": velocity,
            "z_belief": self.belief.probabilities.copy(),
            "kalman_gain": float(gain),
            "changepoint": bool(reset),
        }

    def slow_log(self) -> List[SlowDigest]:
        return list(self.slow.log)

    def low_confidence_spans(self, window: int) -> List[Tuple[int, int]]:
        """Post-change-point recovery windows -- flagged, never silently trusted."""
        return self.slow.reset_spans(window)


__all__ = [
    "SlowStateTracker", "BeliefState", "LatentEncoder", "LatentComposer",
]
