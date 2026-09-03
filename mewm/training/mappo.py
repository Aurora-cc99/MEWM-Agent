"""MAPPO -- shared-critic multi-agent PPO over the team reward (formwork.md 第 V 条,
``MEWM-Agent_完整执行方案.md`` 第 5.3 节).

**与 GRPO 的关系**（``mewm.training.grpo``）：两者是 stage 3 强化学习阶段的两种可切换算
法（``TrainingConfig`` 新增字段 ``rl_algorithm in {"grpo","mappo"}``），不是互相替代——
GRPO 只优化推理智能体一个策略（组内相对比较，样本效率更高）；MAPPO 面向四个智能体的联
合优化（更贴合"团队协作"的设定，但需要更多样本/更大算力）。两者的奖励都最终建立在
``mewm.eval.metrics.tp_decision`` / ``iou`` 之上（"同一把尺子"原则），只是组合方式不同。

**可在没有 GPU / 没有真实策略模型时被验证**：:class:`DryRunAgentBackend` +
:class:`MeanBaselineCritic` 让整条 advantage / PPO-clip / critic 更新的算术可以在纯
Python 环境下跑通并断言，这与 ``grpo.py`` 的 ``DryRunBackend`` 是同一测试哲学。
"""

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

#: The four agents of the control system (第 4.1 节); every MAPPO trainer needs exactly
#: one backend per role, no more, no fewer.
ROLES: Tuple[str, ...] = ("perception", "structure", "reasoning", "critic")


# ---------------------------------------------------------------------------
# Rollout containers
# ---------------------------------------------------------------------------


@dataclass
class AgentSample:
    """One agent's action for one candidate region."""

    role: str
    text: str
    product: Dict[str, Any]
    #: Total log-probability of ``text`` under the sampling policy / the frozen
    #: reference -- same convention as ``grpo.PolicySample`` (ratio measured against
    #: the *sampling* policy, KL anchored to the reference).
    logprob: float = 0.0
    ref_logprob: float = 0.0
    logprobs_measured: bool = False


@dataclass
class TeamTransition:
    """One candidate region's joint step: all four agents' actions + the shared reward.

    This is the MAPPO analogue of ``grpo.GroupRollout`` -- but the group here is across
    *agents*, not across repeated samples of one agent, since MAPPO's advantage comes
    from the shared critic rather than from within-group reward spread.
    """

    candidate_id: str
    #: role -> the observation actually handed to that agent (already visibility-
    #: filtered by the orchestrator's evidence-level matrix, 第 4.1 节) -- individual
    #: agents only ever see their own entry.
    observations: Dict[str, Dict[str, Any]]
    samples: Dict[str, AgentSample]
    reward: Optional[TeamRewardBreakdown] = None
    value_estimate: float = 0.0
    advantage: float = 0.0

    def global_state_features(self) -> Dict[str, Any]:
        """Every role's observation concatenated -- **training-time only** input to
        the shared critic (CTDE). No individual agent ever receives this dict; passing
        it to an agent's own policy would violate the evidence-visibility matrix the
        rest of the framework enforces everywhere else.
        """
        return dict(self.observations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reward": self.reward.to_dict() if self.reward else None,
            "value_estimate": round(self.value_estimate, 5),
            "advantage": round(self.advantage, 5),
            "actions": {role: sample.text for role, sample in self.samples.items()},
        }


# ---------------------------------------------------------------------------
# Per-agent policy backend
# ---------------------------------------------------------------------------


class AgentPolicyBackend(ABC):
    """Interface between the trainer and whatever holds one agent's parameters."""

    role: str

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> AgentSample:
        """Produce one action for one candidate's observation."""

    @abstractmethod
    def update(self, batch: Sequence[TeamTransition], clip: float,
               kl_coefficient: float, learning_rate: float,
               entropy_coefficient: float) -> Dict[str, float]:
        """One PPO-clip optimiser step using the *shared* ``batch[i].advantage``."""

    def save(self, path: Path | str) -> Optional[Path]:
        return None


class DryRunAgentBackend(AgentPolicyBackend):
    """No-parameter backend: actions come from a caller-supplied generator.

    Mirrors ``grpo.DryRunBackend`` -- lets the whole loop (team reward, shared
    advantage, PPO-clip arithmetic, critic update) be validated deterministically
    before any real model is wired in.
    """

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


# ---------------------------------------------------------------------------
# Shared critic
# ---------------------------------------------------------------------------


class SharedCritic(ABC):
    """``V_psi(s)`` -- one value estimate shared by all four agents' advantages."""

    @abstractmethod
    def value(self, state_features: Dict[str, Any]) -> float:
        ...

    @abstractmethod
    def update(self, states: Sequence[Dict[str, Any]], returns: Sequence[float],
               learning_rate: float) -> Dict[str, float]:
        ...


class MeanBaselineCritic(SharedCritic):
    """An exponential running mean of ``R_team`` as ``V(s)`` -- no torch required.
    """

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
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore
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
        """A learned ``V_psi(s)`` over a caller-supplied fixed-size featurisation.

        ``feature_fn`` turns ``TeamTransition.global_state_features()`` (the four
        agents' observations, concatenated -- training-time only) into a fixed-length
        vector; this module does not attempt to auto-featurise arbitrary evidence
        dicts because that mapping is a modelling choice the caller owns.
        """

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

else:  # pragma: no cover
    class TorchSharedCritic(SharedCritic):  # type: ignore[no-redef]
        def __init__(self, *_a, **_kw) -> None:
            raise ImportError("TorchSharedCritic needs PyTorch; use MeanBaselineCritic "
                              "for a torch-free run.")

        def value(self, state_features: Dict[str, Any]) -> float:  # pragma: no cover
            raise ImportError

        def update(self, *a, **kw) -> Dict[str, float]:  # pragma: no cover
            raise ImportError


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class MAPPOConfig:
    clip: float = 0.2
    kl_coefficient: float = 0.0
    entropy_coefficient: float = 0.01
    policy_learning_rate: float = 5e-6
    critic_learning_rate: float = 1e-3
    #: Kept for interface parity with single-agent PPO/GAE; each candidate region is a
    #: one-step episode (see :class:`MeanBaselineCritic`), so neither actually
    #: discounts anything today. Set >1-step credit assignment here if a future
    #: revision chains candidates within one video into a single episode.
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
    """Joint PPO-clip update of the four agents against one shared team reward."""

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
        """One optimisation step over a batch of candidate regions.
        """
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
        """Run ``config.total_steps`` steps, one per batch drawn from ``candidate_batches``."""
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
