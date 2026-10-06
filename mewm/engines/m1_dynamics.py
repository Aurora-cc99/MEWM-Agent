"""M1 dynamics model: AU-centered dual-timescale state prediction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import DynamicsConfig, RepresentationConfig
from ..knowledge.au_anatomy import K_SLOTS, SLOT_AUS, SLOT_INDEX, prior_polarity
from ..knowledge.emotion_prototypes import FINE_EMOTIONS, core_aus

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    torch = None
    nn = object
    _TORCH = False

def prior_interaction_matrix() -> np.ndarray:
    matrix = np.zeros((K_SLOTS, K_SLOTS), dtype=np.float32)
    for a in SLOT_AUS:
        for b in SLOT_AUS:
            if a == b:
                continue
            polarity = prior_polarity(a, b)
            if polarity == "+":
                matrix[SLOT_INDEX[a], SLOT_INDEX[b]] = 0.6
            elif polarity == "-":
                matrix[SLOT_INDEX[a], SLOT_INDEX[b]] = -0.6
    for emotion in FINE_EMOTIONS:
        members = [au for au in core_aus(emotion) if au in SLOT_INDEX]
        for a in members:
            for b in members:
                if a == b:
                    continue
                i, j = SLOT_INDEX[a], SLOT_INDEX[b]
                if matrix[i, j] == 0.0:
                    matrix[i, j] = 0.3
    return matrix

PRIOR_INTERACTION: np.ndarray = prior_interaction_matrix()

def emotion_one_hot(emotion: str) -> np.ndarray:
    vector = np.zeros(len(FINE_EMOTIONS), dtype=np.float32)
    if emotion in FINE_EMOTIONS:
        vector[FINE_EMOTIONS.index(emotion)] = 1.0
    return vector

if _TORCH:

    class SlotGraphAttention(nn.Module):

        def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            if self.head_dim * n_heads != dim:
                raise ValueError(f"dim {dim} must be divisible by n_heads {n_heads}")
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.out = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)
            self.edge_bias = nn.Parameter(torch.from_numpy(PRIOR_INTERACTION).clone())

        def forward(self, slots: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            batch, n_slots, dim = slots.shape
            residual = slots

            def reshape(x: "torch.Tensor") -> "torch.Tensor":
                return x.view(batch, n_slots, self.n_heads, self.head_dim).transpose(1, 2)

            q, k, v = reshape(self.query(slots)), reshape(self.key(slots)), reshape(self.value(slots))
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + self.edge_bias.unsqueeze(0).unsqueeze(0)
            attention = torch.softmax(scores, dim=-1)
            context = (self.dropout(attention) @ v).transpose(1, 2).reshape(batch, n_slots, dim)
            return self.norm(residual + self.out(context)), attention

    class MixtureDensityHead(nn.Module):

        def __init__(self, in_dim: int, out_dim: int, n_components: int = 5) -> None:
            super().__init__()
            self.out_dim = out_dim
            self.n_components = n_components
            self.logits = nn.Linear(in_dim, n_components)
            self.mu = nn.Linear(in_dim, n_components * out_dim)
            self.log_sigma = nn.Linear(in_dim, n_components * out_dim)

        def forward(self, features: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            shape = features.shape[:-1]
            mu = self.mu(features).view(*shape, self.n_components, self.out_dim)
            log_sigma = self.log_sigma(features).view(*shape, self.n_components, self.out_dim)
            return {
                "logits": self.logits(features),
                "mu": mu,
                "log_sigma": log_sigma.clamp(-7.0, 3.0),
            }

        def log_prob(self, params: Dict[str, "torch.Tensor"], target: "torch.Tensor") -> "torch.Tensor":
            mu, log_sigma = params["mu"], params["log_sigma"]
            weights = torch.log_softmax(params["logits"], dim=-1)
            target = target.unsqueeze(-2)
            component = -0.5 * (
                ((target - mu) / log_sigma.exp()) ** 2
                + 2.0 * log_sigma
                + math.log(2.0 * math.pi)
            ).sum(-1)
            return torch.logsumexp(weights + component, dim=-1)

        def mode(self, params: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
            best = params["logits"].argmax(dim=-1, keepdim=True)
            index = best.unsqueeze(-1).expand(*best.shape, params["mu"].shape[-1])
            return params["mu"].gather(-2, index).squeeze(-2)

        def variance(self, params: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
            weights = torch.softmax(params["logits"], dim=-1).unsqueeze(-1)
            mu, sigma_sq = params["mu"], (2.0 * params["log_sigma"]).exp()
            mean = (weights * mu).sum(-2, keepdim=True)
            return ((weights * (sigma_sq + (mu - mean) ** 2)).sum(-2)).mean(-1)

    class SlowFastRouter(nn.Module):

        def __init__(self, confound_dim: int = 2, hidden: int = 32,
                    alpha_min: float = 0.01, alpha_max: float = 0.5) -> None:
            super().__init__()
            self.confound_dim = confound_dim
            self.alpha_min = alpha_min
            self.alpha_max = alpha_max
            self.net = nn.Sequential(
                nn.Conv1d(confound_dim, hidden, 5, padding=2), nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
                nn.Conv1d(hidden, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def alpha(self, confounds: "torch.Tensor") -> "torch.Tensor":
            x = confounds.transpose(1, 2)
            logits = self.net(x).squeeze(1)
            return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(logits)

        def split(
            self,
            activations: "torch.Tensor",
            confounds: "torch.Tensor",
            z_slow_init: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            if confounds.shape[-1] != self.confound_dim:
                raise ValueError(
                    f"SlowFastRouter configured for {self.confound_dim} confound "
                    f"channel(s), got {confounds.shape[-1]}")
            alpha = self.alpha(confounds)
            batch, length, n_slots = activations.shape
            slow = activations.new_zeros(batch, length, n_slots)
            current = (z_slow_init if z_slow_init is not None
                      else activations[:, 0, :].clone())
            for t in range(length):
                weight = alpha[:, t].unsqueeze(-1)
                current = (1.0 - weight) * current + weight * activations[:, t, :]
                slow[:, t, :] = current
            fast = activations - slow
            return {"slow": slow, "fast": fast, "alpha": alpha}

    class AUDynamicsModel(nn.Module):

        def __init__(
            self,
            config: Optional[DynamicsConfig] = None,
            representation: Optional[RepresentationConfig] = None,
            n_emotions: int = len(FINE_EMOTIONS),
        ) -> None:
            super().__init__()
            self.config = config or DynamicsConfig()
            self.representation = representation or RepresentationConfig()
            self.n_emotions = n_emotions
            slot_dim = self.representation.slot_dim
            hidden = self.config.hidden_dim

            self.belief_rnn = nn.GRUCell(slot_dim * 2 + self.representation.slow_dim, hidden)
            self.belief_head = nn.Linear(hidden, n_emotions)

            self.slot_in = nn.Linear(slot_dim, hidden)
            self.condition = nn.Linear(n_emotions + self.representation.slow_dim, hidden)
            self.layers = nn.ModuleList([
                SlotGraphAttention(hidden, self.config.n_heads)
                for _ in range(self.config.n_gat_layers)
            ])
            self.slot_head = MixtureDensityHead(hidden, slot_dim, self.config.n_mixture)
            self.activation_head = nn.Linear(hidden, 1)

            self.motion_head = MixtureDensityHead(
                hidden + self.representation.fast_dim,
                self.representation.fast_dim, self.config.n_mixture,
            )

            self.future_queries = nn.Parameter(
                torch.randn(self.config.future_queries, hidden) * 0.02
            )
            self.step_embedding = nn.Parameter(
                torch.randn(self.config.rollout_steps, hidden) * 0.02
            )
            self.imagine_attention = nn.MultiheadAttention(hidden, self.config.n_heads,
                                                           batch_first=True)

        def transition_belief(
            self, slots: "torch.Tensor", z_slow: "torch.Tensor",
            hidden: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            pooled = torch.cat([slots.mean(dim=1), slots.max(dim=1).values, z_slow], dim=-1)
            hidden = self.belief_rnn(pooled, hidden)
            return self.belief_head(hidden), hidden

        def transition_slots(
            self, slots: "torch.Tensor", emotion: "torch.Tensor", z_slow: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            features = self.slot_in(slots)
            features = features + self.condition(
                torch.cat([emotion, z_slow], dim=-1)
            ).unsqueeze(1)
            attentions = []
            for layer in self.layers:
                features, attention = layer(features)
                attentions.append(attention)
            params = self.slot_head(features)
            params["activation"] = torch.sigmoid(self.activation_head(features)).squeeze(-1)
            params["attention"] = torch.stack(attentions, dim=1)
            params["features"] = features
            return params

        def transition_motion(
            self, features: "torch.Tensor", z_fast: "torch.Tensor",
        ) -> Dict[str, "torch.Tensor"]:
            pooled = features.mean(dim=1)
            return self.motion_head(torch.cat([pooled, z_fast], dim=-1))

        def forward(
            self,
            slots: "torch.Tensor",
            z_slow: "torch.Tensor",
            z_fast: "torch.Tensor",
            emotion: Optional["torch.Tensor"] = None,
            belief_hidden: Optional["torch.Tensor"] = None,
        ) -> Dict[str, object]:
            belief_logits, hidden = self.transition_belief(slots, z_slow, belief_hidden)
            conditioning = (
                emotion if emotion is not None else torch.softmax(belief_logits, dim=-1)
            )
            slot_params = self.transition_slots(slots, conditioning, z_slow)
            motion_params = self.transition_motion(slot_params["features"], z_fast)
            return {
                "belief_logits": belief_logits,
                "belief_hidden": hidden,
                "slot_params": slot_params,
                "motion_params": motion_params,
                "next_slots": self.slot_head.mode(slot_params),
                "next_activation": slot_params["activation"],
                "predictive_variance": self.slot_head.variance(slot_params),
                "interaction": slot_params["attention"].mean(dim=(1, 2)),
            }

        def imagine(
            self,
            slots: "torch.Tensor",
            z_slow: "torch.Tensor",
            z_fast: "torch.Tensor",
            steps: Optional[int] = None,
            emotion: Optional["torch.Tensor"] = None,
        ) -> Dict[str, "torch.Tensor"]:
            steps = steps or self.config.rollout_steps
            batch = slots.shape[0]
            current, hidden = slots, None
            trajectory, activations, variances, queries = [], [], [], []

            for step in range(steps):
                out = self.forward(current, z_slow, z_fast, emotion, hidden)
                hidden = out["belief_hidden"]
                current = out["next_slots"]
                trajectory.append(current)
                activations.append(out["next_activation"])
                variances.append(out["predictive_variance"])

                query = (
                    self.future_queries.unsqueeze(0).expand(batch, -1, -1)
                    + self.step_embedding[min(step, self.step_embedding.shape[0] - 1)]
                )
                attended, _ = self.imagine_attention(
                    query, out["slot_params"]["features"], out["slot_params"]["features"]
                )
                queries.append(attended)

            return {
                "trajectory": torch.stack(trajectory, dim=1),
                "activations": torch.stack(activations, dim=1),
                "variance": torch.stack(variances, dim=1),
                "queries": torch.stack(queries, dim=1),
            }

        def imagination_loss(
            self, predicted_queries: "torch.Tensor", future_target: "torch.Tensor",
        ) -> "torch.Tensor":
            steps = predicted_queries.shape[1]
            total = predicted_queries.new_zeros(())
            for step in range(steps):
                prediction = predicted_queries[:, step]
                mse = F.mse_loss(prediction, future_target)
                cosine = 1.0 - F.cosine_similarity(prediction, future_target, dim=-1).mean()
                total = total + mse + 0.5 * cosine
            return total / max(1, steps)

        def slot_nll(self, params: Dict[str, "torch.Tensor"], target: "torch.Tensor") -> "torch.Tensor":
            return -self.slot_head.log_prob(params, target).mean()

        def version_hash(self) -> str:
            import hashlib
            digest = hashlib.blake2s(digest_size=8)
            for name, tensor in sorted(self.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(tensor.detach().cpu().numpy().tobytes())
            return f"m1-{digest.hexdigest()}"

        def save(self, path: Path | str) -> Path:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": self.state_dict(),
                "config": vars(self.config),
                "representation": vars(self.representation),
                "version": self.version_hash(),
                "slot_aus": SLOT_AUS,
                "emotions": FINE_EMOTIONS,
            }, target)
            return target

        @classmethod
        def load(cls, path: Path | str, map_location: str = "cpu") -> "AUDynamicsModel":
            payload = torch.load(Path(path), map_location=map_location, weights_only=False)
            model = cls(
                DynamicsConfig(**payload.get("config", {})),
                RepresentationConfig(**payload.get("representation", {})),
            )
            model.load_state_dict(payload["state_dict"])
            model.eval()
            return model

else:

    class AUDynamicsModel:
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("AUDynamicsModel needs PyTorch.")

    class SlowFastRouter:
        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError("SlowFastRouter needs PyTorch.")

class AnalyticDynamics:

    version = "m1-analytic"

    def __init__(self, config: Optional[DynamicsConfig] = None) -> None:
        self.config = config or DynamicsConfig()
        self.interaction = PRIOR_INTERACTION.copy()

    def step(self, activations: np.ndarray, emotion: Optional[str] = None,
             momentum: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        current = np.asarray(activations, dtype=np.float64).reshape(-1)
        if current.size != K_SLOTS:
            current = np.resize(current, K_SLOTS)
        velocity = (np.asarray(momentum, dtype=np.float64).reshape(-1)
                    if momentum is not None else np.zeros(K_SLOTS))

        influence = self.interaction.T @ current / max(1.0, K_SLOTS ** 0.5)
        prior = np.zeros(K_SLOTS)
        if emotion:
            for au in core_aus(emotion):
                if au in SLOT_INDEX:
                    from ..knowledge.emotion_prototypes import au_weight
                    prior[SLOT_INDEX[au]] = au_weight(emotion, au)

        predicted = 0.82 * current + 0.55 * velocity + 0.12 * influence + 0.08 * prior
        predicted = np.clip(predicted, 0.0, 1.0)
        variance = 0.02 + 0.10 * predicted * (1.0 - predicted)
        return predicted, variance

    def rollout(self, activations: np.ndarray, steps: int = 3,
                emotion: Optional[str] = None,
                momentum: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        current = np.asarray(activations, dtype=np.float64).reshape(-1)
        if current.size != K_SLOTS:
            current = np.resize(current, K_SLOTS)
        velocity = (np.asarray(momentum, dtype=np.float64).reshape(-1)
                    if momentum is not None else np.zeros(K_SLOTS))
        trajectory, variances = [], []
        for _ in range(steps):
            nxt, variance = self.step(current, emotion, velocity)
            velocity = 0.6 * velocity + 0.4 * (nxt - current)
            current = nxt
            trajectory.append(current.copy())
            variances.append(variance.copy())
        return {
            "trajectory": np.stack(trajectory) if trajectory else np.zeros((0, K_SLOTS)),
            "variance": np.stack(variances) if variances else np.zeros((0, K_SLOTS)),
        }

    def log_likelihood(self, observed: np.ndarray, emotion: str) -> float:
        observed = np.atleast_2d(np.asarray(observed, dtype=np.float64))
        if observed.shape[0] < 2:
            return 0.0
        total, velocity = 0.0, np.zeros(observed.shape[1])
        for i in range(observed.shape[0] - 1):
            mean, variance = self.step(observed[i], emotion, velocity)
            residual = observed[i + 1] - mean
            total += float(
                -0.5 * np.sum(residual ** 2 / variance + np.log(2.0 * np.pi * variance))
            )
            velocity = 0.6 * velocity + 0.4 * (observed[i + 1] - observed[i])
        return total / max(1, observed.shape[0] - 1)

    def interaction_weight(self, source: str, target: str) -> float:
        if source not in SLOT_INDEX or target not in SLOT_INDEX:
            return 0.0
        return float(self.interaction[SLOT_INDEX[source], SLOT_INDEX[target]])

__all__ = [
    "prior_interaction_matrix", "PRIOR_INTERACTION", "emotion_one_hot",
    "SlowFastRouter", "AUDynamicsModel", "AnalyticDynamics",
]
