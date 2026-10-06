"""V3 latent encoder: subject-specific latent facial-dynamics baseline."""
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
except ImportError:
    torch = None
    nn = object
    _TORCH = False


class SlowStateTracker:
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
        gain = prior_var / (prior_var + obs_var)
        innovation = observation - self.mu

        reset = self._check_changepoint(innovation, prior_var + obs_var, t)
        self.mu = self.mu + gain * innovation
        self.sigma = (1.0 - gain) * prior_var
        self._last_gain = float(gain)

        self.log.append(SlowDigest(t=t, summary=_digest(self.mu),
                                   kalman_gain=round(float(gain), 5), reset=reset))
        return self.mu.copy(), float(gain), reset

    def predict(self) -> np.ndarray:
        return self.mu.copy()

    def _check_changepoint(self, innovation: np.ndarray, variance: float, t: int) -> bool:
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
        return self._last_gain

    def reset_spans(self, window: int) -> List[Tuple[int, int]]:
        return [(entry.t, entry.t + window) for entry in self.log if entry.reset]


def _digest(vector: np.ndarray, size: int = 8) -> List[float]:
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


@dataclass
class BeliefState:
    labels: List[str]
    logits: np.ndarray
    decay: float = 0.92

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
        return round(self.entropy / math.log(max(2, len(self.labels))), 5)

    @property
    def margin(self) -> float:
        ordered = np.sort(self.probabilities)[::-1]
        return float(ordered[0] - ordered[1]) if ordered.size >= 2 else 1.0

    def update(self, log_evidence: Dict[str, float], weight: float = 1.0) -> "BeliefState":
        self.logits *= self.decay
        for i, label in enumerate(self.labels):
            self.logits[i] += weight * float(log_evidence.get(label, 0.0))
        return self

    def kl_to(self, other: "BeliefState") -> float:
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


if _TORCH:

    class LatentEncoder(nn.Module):
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
            if z_fast.shape[0] < 2:
                return z_fast.new_zeros(())
            delta = z_fast[1:] - z_fast[:-1]
            return ((delta - flow_velocity[:-1]) ** 2).sum(dim=-1).mean()

        @staticmethod
        def slow_consistency_loss(
            observation: "torch.Tensor", prior_mean: "torch.Tensor",
            prior_var: "torch.Tensor",
        ) -> "torch.Tensor":
            residual = observation - prior_mean
            return (residual ** 2 / prior_var.clamp(min=1e-6)).sum(dim=-1).mean()

else:

    class LatentEncoder:
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("LatentEncoder needs PyTorch.")


class LatentComposer:
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
        return self.slow.reset_spans(window)


__all__ = [
    "SlowStateTracker", "BeliefState", "LatentEncoder", "LatentComposer",
]
