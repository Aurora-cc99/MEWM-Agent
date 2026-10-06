"""MAPPO multi-agent PPO variant used during early RFT warm-up."""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, TrainingConfig, load_config
from .rewards import DEFAULT_TEAM_REWARD_WEIGHTS, TeamRewardBreakdown, team_reward_components

LOGGER = logging.getLogger(__name__)

ROLES: Tuple[str, ...] = ("perception", "structure", "reasoning", "critic")


@dataclass
class AgentSample:

    role: str
    text: str
    product: Dict[str, Any]
    logprob: float = 0.0
    ref_logprob: float = 0.0
    logprobs_measured: bool = False


@dataclass
class TeamTransition:

    candidate_id: str
    observations: Dict[str, Dict[str, Any]]
    samples: Dict[str, AgentSample]
    reward: Optional[TeamRewardBreakdown] = None
    value_estimate: float = 0.0
    advantage: float = 0.0

    def global_state_features(self) -> Dict[str, Any]:
        return dict(self.observations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reward": self.reward.to_dict() if self.reward else None,
            "value_estimate": round(self.value_estimate, 5),
            "advantage": round(self.advantage, 5),
            "actions": {role: sample.text for role, sample in self.samples.items()},
        }


class AgentPolicyBackend(ABC):

    role: str

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> AgentSample:

        ...
    @abstractmethod
    def update(self, batch: Sequence[TeamTransition], clip: float,
               kl_coefficient: float, learning_rate: float,
               entropy_coefficient: float) -> Dict[str, float]:

        ...
    def save(self, path: Path | str) -> Optional[Path]:
        return None


class DryRunAgentBackend(AgentPolicyBackend):

    def __init__(self, role: str,
                generator: Callable[[str, Dict[str, Any]], Dict[str, Any]]) -> None:
        self.role = role
        self.generator = generator
        self.steps = 0

    def act(self, observation: Dict[str, Any]) -> AgentSample:
        product = self.generator(self.role, observation)
        return AgentSample(role=self.role,
                           text=json.dumps(product, ensure_ascii=False), product=product)

    def update(self, batch: Sequence[TeamTransition], clip: float,
               kl_coefficient: float, learning_rate: float,
               entropy_coefficient: float) -> Dict[str, float]:
        self.steps += 1
        objective, kl_total, count = 0.0, 0.0, 0
        for transition in batch:
            sample = transition.samples.get(self.role)
            if sample is None:
                continue
            ratio = math.exp(sample.logprob - sample.ref_logprob)
            advantage = transition.advantage
            clipped = min(ratio * advantage,
                          float(np.clip(ratio, 1 - clip, 1 + clip)) * advantage)
            objective += clipped
            kl_total += sample.logprob - sample.ref_logprob
            count += 1
        return {
            "role": self.role,
            "objective": round(objective / max(1, count), 5),
            "kl": round(kl_total / max(1, count), 5),
            "n_samples": count,
        }


class SharedCritic(ABC):

    @abstractmethod
    def value(self, state_features: Dict[str, Any]) -> float:
        ...

    @abstractmethod
    def update(self, states: Sequence[Dict[str, Any]], returns: Sequence[float],
               learning_rate: float) -> Dict[str, float]:
        ...


class MeanBaselineCritic(SharedCritic):

    def __init__(self, momentum: float = 0.95) -> None:
        self.momentum = momentum
        self._mean = 0.0
        self._initialised = False

    def value(self, state_features: Dict[str, Any]) -> float:
        return self._mean

    def update(self, states: Sequence[Dict[str, Any]], returns: Sequence[float],
               learning_rate: float = 0.0) -> Dict[str, float]:
        for r in returns:
            if not self._initialised:
                self._mean, self._initialised = float(r), True
            else:
                self._mean = self.momentum * self._mean + (1 - self.momentum) * float(r)
        return {"critic_value": round(self._mean, 5), "n": len(returns)}


try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    torch = None
    nn = object
    _TORCH = False

if _TORCH:

    class _CriticMLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

    class TorchSharedCritic(SharedCritic):

        def __init__(self, feature_fn: Callable[[Dict[str, Any]], np.ndarray],
                    in_dim: int, hidden: int = 128, device: str = "cpu") -> None:
            self.feature_fn = feature_fn
            self.device = device
            self.model = _CriticMLP(in_dim, hidden).to(device)
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        def value(self, state_features: Dict[str, Any]) -> float:
            with torch.no_grad():
                x = torch.as_tensor(self.feature_fn(state_features),
                                    dtype=torch.float32, device=self.device).unsqueeze(0)
                return float(self.model(x).item())

        def update(self, states: Sequence[Dict[str, Any]], returns: Sequence[float],
                   learning_rate: float = 1e-3) -> Dict[str, float]:
            for group in self.optimiser.param_groups:
                group["lr"] = learning_rate
            x = torch.as_tensor(np.stack([self.feature_fn(s) for s in states]),
                               dtype=torch.float32, device=self.device)
            y = torch.as_tensor(list(returns), dtype=torch.float32, device=self.device)
            prediction = self.model(x)
            loss = torch.nn.functional.mse_loss(prediction, y)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            return {"critic_loss": round(float(loss.detach().cpu()), 5), "n": len(states)}

else:
    class TorchSharedCritic(SharedCritic):
        def __init__(self, *_a, **_kw) -> None:
            raise ImportError("TorchSharedCritic needs PyTorch; use MeanBaselineCritic "
                              "for a torch-free run.")

        def value(self, state_features: Dict[str, Any]) -> float:
            raise ImportError

        def update(self, *a, **kw) -> Dict[str, float]:
            raise ImportError


@dataclass
class MAPPOConfig:
    clip: float = 0.2
    kl_coefficient: float = 0.0
    entropy_coefficient: float = 0.01
    policy_learning_rate: float = 5e-6
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TEAM_REWARD_WEIGHTS))
    total_steps: int = 1000
    max_candidates_per_step: int = 8
    seed: int = 20260824

    @classmethod
    def from_training(cls, training: TrainingConfig) -> "MAPPOConfig":
        return cls(
            total_steps=training.rl_total_steps,
            max_candidates_per_step=training.rl_prompts_per_step,
            seed=training.fold_seed,
        )


class MAPPOTrainer:

    def __init__(
        self,
        backends: Dict[str, AgentPolicyBackend],
        critic: Optional[SharedCritic] = None,
        config: Optional[MAPPOConfig] = None,
        mewm_config: Optional[MEWMConfig] = None,
        evaluation: Optional[Any] = None,
    ) -> None:
        missing = set(ROLES) - set(backends)
        if missing:
            raise ValueError(f"MAPPOTrainer needs a backend for every role; "
                             f"missing {sorted(missing)}")
        self.backends = backends
        self.critic = critic or MeanBaselineCritic()
        self.mewm_config = mewm_config or load_config()
        self.evaluation = evaluation or self.mewm_config.evaluation
        self.config = config or MAPPOConfig.from_training(self.mewm_config.training)
        self.history: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(self.config.seed)

    def step(self, candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        transitions: List[TeamTransition] = []
        for entry in candidates[: self.config.max_candidates_per_step]:
            observations = dict(entry.get("observations", {}))
            samples = {role: self.backends[role].act(observations.get(role, {}))
                      for role in ROLES}
            critic_product = samples["critic"].product
            critic_verdict = critic_product if isinstance(critic_product, dict) else None
            reward = team_reward_components(
                samples["reasoning"].product, entry.get("truth", {}),
                critic_verdict=critic_verdict, evaluation=self.evaluation,
                weights=self.config.reward_weights,
            )
            transitions.append(TeamTransition(
                candidate_id=str(entry.get("id", "")), observations=observations,
                samples=samples, reward=reward,
            ))

        states = [t.global_state_features() for t in transitions]
        returns = [t.reward.total for t in transitions]
        baselines = [self.critic.value(s) for s in states]
        for transition, baseline in zip(transitions, baselines):
            transition.value_estimate = baseline
            transition.advantage = round(transition.reward.total - baseline, 5)

        critic_diagnostics = self.critic.update(states, returns,
                                                self.config.critic_learning_rate)

        agent_diagnostics: Dict[str, Dict[str, float]] = {}
        for role in ROLES:
            agent_diagnostics[role] = self.backends[role].update(
                transitions, self.config.clip, self.config.kl_coefficient,
                self.config.policy_learning_rate, self.config.entropy_coefficient,
            )

        record = {
            "n_candidates": len(transitions),
            "reward_mean": (round(float(np.mean(returns)), 5) if returns else None),
            "advantage_mean": (round(float(np.mean([t.advantage for t in transitions])), 5)
                              if transitions else None),
            "critic": critic_diagnostics,
            "agents": agent_diagnostics,
        }
        self.history.append(record)
        return record

    def train(self, candidate_batches: Iterable[Sequence[Dict[str, Any]]]
             ) -> List[Dict[str, Any]]:
        for step_index, batch in enumerate(candidate_batches):
            if step_index >= self.config.total_steps:
                break
            self.step(batch)
        return self.history

    def write_history(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.history, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        return target


__all__ = [
    "ROLES", "AgentSample", "TeamTransition", "AgentPolicyBackend",
    "DryRunAgentBackend", "SharedCritic", "MeanBaselineCritic", "TorchSharedCritic",
    "MAPPOConfig", "MAPPOTrainer",
]
