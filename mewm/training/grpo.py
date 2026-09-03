"""The GRPO driver for the R-Agent policy (appendix F.3).
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, TrainingConfig, load_config
from .rewards import CompositeReward, RewardBreakdown, WorldModelJudge, curriculum_weights

LOGGER = logging.getLogger(__name__)

#: Ceiling on the per-token log-ratio fed to the k3 KL estimator.
#:
#: k3 is ``exp(delta) - delta - 1``. It is non-negative and low-variance, but it is
#: *exponential in delta*, so it is only safe on quantities that stay small. A per-token
#: log-ratio does: a token the reference gives e^-20 (~2e-9) more mass to than the policy
#: is already a total divergence, and k3 there is 4.8e8 -- enormous, and enough to
#: dominate the objective, which is the intended behaviour. Past that the term buys no
#: extra signal and only risks ``inf``. Clamping is therefore a saturation, not a
#: silent truncation, and every clamped token is counted into the step record.
KL_DELTA_CLAMP = 20.0


# ---------------------------------------------------------------------------
# Rollout container
# ---------------------------------------------------------------------------


@dataclass
class PolicySample:
    """One sampled output with its reward."""

    text: str
    product: Dict[str, Any]
    reward: Optional[RewardBreakdown] = None
    #: The chain of thought the policy emitted, split off :attr:`text` on the
    #: ``<think>`` delimiters. Kept out of ``text`` deliberately: the format term of
    #: the reward scores ``text`` as the answer, so a CoT left in front of it would be
    #: scored as malformed output and GRPO would train the policy to stop thinking --
    #: the exact opposite of what the thinking-enabled backend is for.
    reasoning: str = ""
    #: Total log-probability of the *completion* under the policy that drew it
    #: (pi_theta_old). The importance ratio is measured against this, not against the
    #: reference: the reference anchors the KL term and nothing else.
    logprob: float = 0.0
    #: Total log-probability of the completion under the frozen reference policy.
    ref_logprob: float = 0.0
    #: Per-token log-probabilities of the completion under the sampling policy, in
    #: generation order. The ratio and the KL are formed *token-wise* from these. The
    #: totals above are their sums, kept for reporting only: a sequence-level delta is
    #: a sum over 10^2--10^3 tokens, and ``exp`` of a difference that large is not a
    #: number float64 can hold.
    token_logprobs: List[float] = field(default_factory=list)
    #: Per-token log-probabilities under the frozen reference, aligned index-for-index
    #: with :attr:`token_logprobs`. Empty when there is no reference model.
    ref_token_logprobs: List[float] = field(default_factory=list)
    #: False when the backend could not measure the quantities above -- a dry run,
    #: or a real backend with no reference model. The update reads this instead of
    #: inferring intent from a 0.0 that is indistinguishable from an unset field.
    logprobs_measured: bool = False

    @property
    def score(self) -> float:
        return self.reward.total if self.reward else 0.0


@dataclass
class GroupRollout:
    """The ``G`` samples drawn for one prompt."""

    prompt_id: str
    prompt: str
    samples: List[PolicySample] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)

    def rewards(self) -> np.ndarray:
        return np.array([s.score for s in self.samples], dtype=np.float64)

    def compute_advantages(self, eps: float = 1e-6) -> List[float]:
        """Group-relative advantage: centre and scale within the group."""
        rewards = self.rewards()
        if rewards.size == 0:
            return []
        centre = float(rewards.mean())
        spread = float(rewards.std())
        if spread < eps:
            # A degenerate group carries no preference information; a zero advantage is
            # the honest encoding of that, and it keeps the update from amplifying noise.
            self.advantages = [0.0] * rewards.size
        else:
            self.advantages = [round(float((r - centre) / spread), 5) for r in rewards]
        return self.advantages

    def stats(self) -> Dict[str, float]:
        rewards = self.rewards()
        if rewards.size == 0:
            return {}
        return {
            "mean": round(float(rewards.mean()), 5),
            "std": round(float(rewards.std()), 5),
            "max": round(float(rewards.max()), 5),
            "min": round(float(rewards.min()), 5),
        }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class PolicyBackend(ABC):
    """Interface between the training loop and whatever holds the parameters."""

    @abstractmethod
    def sample(self, prompt: str, n: int, temperature: float) -> List[PolicySample]:
        """Draw ``n`` outputs for one prompt."""

    @abstractmethod
    def update(self, rollouts: Sequence[GroupRollout], clip: float,
               kl_coefficient: float, learning_rate: float) -> Dict[str, float]:
        """Apply one optimiser step; returns loss diagnostics."""

    def save(self, path: Path | str) -> Optional[Path]:
        return None


class DryRunBackend(PolicyBackend):
    """No-parameter backend for validating the loop, the reward and the curriculum.

    Samples come from a caller-supplied generator, so a fixture can drive the loop
    deterministically. Reports the objective it *would* have optimised without touching
    weights -- useful for checking that the advantage and clipping arithmetic is right
    before committing GPU time.
    """

    def __init__(self, generator: Callable[[str, int], List[Dict[str, Any]]]) -> None:
        self.generator = generator
        self.steps = 0

    def sample(self, prompt: str, n: int, temperature: float) -> List[PolicySample]:
        return [
            PolicySample(text=json.dumps(product, ensure_ascii=False), product=product)
            for product in self.generator(prompt, n)
        ]

    def update(self, rollouts: Sequence[GroupRollout], clip: float,
               kl_coefficient: float, learning_rate: float) -> Dict[str, float]:
        self.steps += 1
        objective, kl_total, count = 0.0, 0.0, 0
        for rollout in rollouts:
            for sample, advantage in zip(rollout.samples, rollout.advantages):
                # No parameters means no log-probabilities: both terms are 0, the ratio
                # is exactly 1, and the reported objective is the advantage itself. That
                # is the point -- it isolates the advantage and clipping arithmetic from
                # the model. :class:`LoRABackend` measures both quantities for real and
                # takes the ratio against the sampling policy, not against the reference.
                ratio = math.exp(sample.logprob - sample.ref_logprob)
                clipped = min(ratio * advantage,
                              float(np.clip(ratio, 1 - clip, 1 + clip)) * advantage)
                objective += clipped
                kl_total += sample.logprob - sample.ref_logprob
                count += 1
        return {
            "objective": round(objective / max(1, count), 5),
            "kl": round(kl_total / max(1, count), 5),
            "loss": round(-objective / max(1, count) + kl_coefficient * kl_total / max(1, count), 5),
            "n_samples": count,
        }


class LoRABackend(PolicyBackend):
    """PyTorch/PEFT backend for the real run.

    Deliberately thin: it adapts an already-constructed model and tokenizer rather than
    owning their configuration, so the training loop stays independent of any particular
    serving stack. Raises with a clear message if the optional dependencies are absent
    instead of failing deep inside a forward pass.
    """

    def __init__(self, model: Any = None, tokenizer: Any = None,
                 reference_model: Any = None, device: str = "cuda") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("LoRABackend needs PyTorch installed.") from exc
        if model is None or tokenizer is None:
            raise ValueError(
                "LoRABackend needs a constructed model and tokenizer. Build the policy "
                "(base + RL LoRA adapter) in your launcher and pass it in."
            )
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.device = device
        if reference_model is None:
            # Say it once, loudly. Without a frozen reference there is nothing to anchor
            # against, so the KL term is dropped rather than silently computed against a
            # zero that would make it a plain -log pi penalty on everything sampled.
            LOGGER.warning(
                "LoRABackend built without a reference model: the KL anchor is disabled "
                "for this run. Pass a frozen copy of the pre-RL policy to enable it.")
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=5e-6)

    def sample(self, prompt: str, n: int, temperature: float) -> List[PolicySample]:
        from ..agents.base import extract_json
        from ..llm.local_models import split_reasoning
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs: List[PolicySample] = []
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, do_sample=True, temperature=temperature,
                num_return_sequences=n, max_new_tokens=1024,
                return_dict_in_generate=True, output_scores=True,
            )
        for sequence in generated.sequences:
            raw = self.tokenizer.decode(sequence[inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            # A thinking policy emits <think>...</think> before the answer. The split
            # has to happen before extract_json: that helper takes the FIRST brace
            # block, and a chain of thought containing a brace would hand the reward
            # a fragment of the reasoning instead of the answer.
            text, reasoning = split_reasoning(raw)
            # Log-probabilities are measured over the whole completion, reasoning
            # included -- that is what the policy actually generated, and the ratio
            # and KL must be taken against the sampled sequence, not a trimmed view.
            with torch.no_grad():
                tokens = self._completion_token_logprobs(prompt, raw).detach()
                token_logprobs = [float(v) for v in tokens]
                if self.reference_model is not None:
                    ref_tokens = self._completion_token_logprobs(
                        prompt, raw, model=self.reference_model).detach()
                    ref_token_logprobs = [float(v) for v in ref_tokens]
                else:
                    ref_token_logprobs = []
            outputs.append(PolicySample(
                text=text, product=extract_json(text) or {}, reasoning=reasoning,
                logprob=float(sum(token_logprobs)),
                ref_logprob=float(sum(ref_token_logprobs)),
                token_logprobs=token_logprobs,
                ref_token_logprobs=ref_token_logprobs,
                logprobs_measured=True))
        return outputs

    def update(self, rollouts: Sequence[GroupRollout], clip: float,
               kl_coefficient: float, learning_rate: float) -> Dict[str, float]:
        """One clipped-surrogate step with a token-level k3 KL anchor.
        """
        torch = self.torch
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.zero_grad()

        unmeasured = [s for r in rollouts for s in r.samples if not s.logprobs_measured]
        if unmeasured:
            raise RuntimeError(
                f"{len(unmeasured)} sample(s) carry no measured log-probability. The "
                f"ratio and the KL anchor are both meaningless against an unset 0.0, so "
                f"the update refuses to run. Draw the samples through this backend's "
                f"own sample() rather than injecting them.")

        total_loss = torch.zeros((), device=self.device)
        kl_total, ratio_total = 0.0, 0.0
        n_sequences, n_tokens, n_clamped = 0, 0, 0
        anchored = self.reference_model is not None
        for rollout in rollouts:
            for sample, advantage in zip(rollout.samples, rollout.advantages):
                # The only differentiable term: the per-token log-probabilities of the
                # completion under the *current* parameters. Everything they are compared
                # against is a detached vector measured at sampling time.
                logprobs = self._completion_token_logprobs(rollout.prompt, sample.text)
                if logprobs.numel() == 0:
                    continue
                old = torch.tensor(sample.token_logprobs, device=self.device,
                                   dtype=logprobs.dtype)
                if old.numel() != logprobs.numel():
                    # Same strings through the same tokenizer must give the same length.
                    # If they do not, something re-tokenised between sampling and update
                    # and the alignment is a fiction -- refuse rather than pair token i
                    # of one segmentation with token i of another.
                    raise RuntimeError(
                        f"prompt {rollout.prompt_id!r}: completion re-tokenised to "
                        f"{logprobs.numel()} tokens but was sampled at {old.numel()}. "
                        f"The per-token ratio has no valid alignment.")

                # Every token of a sequence carries that sequence's advantage: GRPO
                # scores whole outputs, so there is no per-token credit assignment to be
                # had, and spreading the one scalar across the tokens is the honest
                # encoding of that.
                ratio = torch.exp(logprobs - old)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
                total_loss = total_loss - torch.min(unclipped, clipped).mean()

                if anchored:
                    ref = torch.tensor(sample.ref_token_logprobs, device=self.device,
                                       dtype=logprobs.dtype)
                    # k3 estimator, per token: non-negative, unbiased, and lower-variance
                    # than the raw difference. It is also exponential in its argument,
                    # which is exactly why it has to be fed a per-token delta and
                    # saturated -- see KL_DELTA_CLAMP.
                    raw_delta = ref - logprobs
                    n_clamped += int((raw_delta > KL_DELTA_CLAMP).sum().detach())
                    delta = torch.clamp(raw_delta, max=KL_DELTA_CLAMP)
                    kl = (torch.exp(delta) - delta - 1.0).mean()
                    total_loss = total_loss + kl_coefficient * kl
                    kl_total += float(kl.detach())

                ratio_total += float(ratio.mean().detach())
                n_tokens += int(logprobs.numel())
                n_sequences += 1

        if n_sequences:
            total_loss = total_loss / n_sequences
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.optimizer.step()
        if n_clamped:
            LOGGER.warning(
                "%d of %d token KL terms hit the delta clamp (%.1f); the policy has "
                "moved far from the reference on those tokens.",
                n_clamped, n_tokens, KL_DELTA_CLAMP)
        return {
            "loss": float(total_loss.detach()) if n_sequences else 0.0,
            "kl": round(kl_total / max(1, n_sequences), 5),
            "ratio": round(ratio_total / max(1, n_sequences), 5),
            "kl_anchored": anchored,
            "kl_clamped_tokens": n_clamped,
            "n_tokens": n_tokens,
            "n_samples": n_sequences,
        }

    def _completion_token_logprobs(self, prompt: str, completion: str,
                                   model: Any = None) -> Any:
        """Per-token log-probabilities of ``completion`` given ``prompt``, in nats.
        """
        torch = self.torch
        model = model if model is not None else self.model
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(prompt + completion,
                                  add_special_tokens=False)["input_ids"]
        n_completion = len(full_ids) - len(prompt_ids)
        if n_completion <= 0:
            # An empty completion contributes nothing; an empty tensor keeps it in the
            # graph and lets the caller skip it without a special-case sentinel.
            return torch.zeros((0,), device=self.device)
        input_ids = torch.tensor([full_ids], device=self.device)
        logits = model(input_ids=input_ids).logits[0]
        # Causal shift: position t-1 predicts token t. The completion occupies
        # ``[len(prompt_ids), len(full_ids))``, so its predictions start one earlier.
        window = logits[len(prompt_ids) - 1: len(full_ids) - 1]
        targets = input_ids[0, len(prompt_ids):]
        log_probs = torch.log_softmax(window.float(), dim=-1)
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    def _completion_logprob(self, prompt: str, completion: str,
                            model: Any = None) -> Any:
        """Total completion log-probability -- the sum of the per-token vector.

        Kept for reporting and for callers that want the scalar. The update does not use
        it: see :meth:`update` for why the sum is the wrong quantity to do algebra on.
        """
        return self._completion_token_logprobs(prompt, completion, model=model).sum()

    def save(self, path: Path | str) -> Optional[Path]:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        return target


# ---------------------------------------------------------------------------
# GRPO trainer
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Appendix F.3 defaults."""

    group_size: int = 8               # G
    clip: float = 0.2
    kl_coefficient: float = 0.01      # beta_KL
    learning_rate: float = 5e-6
    total_steps: int = 1000
    eval_every: int = 50
    temperature: float = 0.7
    max_prompts_per_step: int = 4
    #: Drop prompts whose reward group has no usable spread (see :func:`admit_prompt`).
    admit_prompts: bool = True
    admit_std_min: float = 0.05
    seed: int = 20260824

    @classmethod
    def from_training(cls, training: TrainingConfig) -> "GRPOConfig":
        """Build from the protocol-level config so one place sets the schedule.

        ``total_steps`` is a count of optimiser steps, and it also defines the curriculum
        clock (``progress = step / (total_steps - 1)``), so changing it rescales the weight
        schedule as well as the length of the run.
        """
        return cls(
            group_size=training.rl_group_size,
            total_steps=training.rl_total_steps,
            eval_every=training.rl_eval_every,
            temperature=training.rl_temperature,
            max_prompts_per_step=training.rl_prompts_per_step,
            admit_prompts=training.rl_admit_prompts,
            admit_std_min=training.reward_std_min,
            seed=training.fold_seed,
        )


def admit_prompt(rewards: Sequence[float], std_min: float = 0.05) -> Tuple[bool, str]:
    """Whether a prompt's reward group can move the policy.
    """
    values = np.asarray([float(r) for r in rewards], dtype=np.float64)
    if values.size < 2:
        return False, "fewer than two samples"
    spread = float(values.std())
    if spread < std_min:
        mean = float(values.mean())
        side = "ceiling" if mean > 0.5 else "floor"
        return False, (f"degenerate group at the {side} (std {spread:.4f} < {std_min}, "
                       f"mean {mean:.4f}): the advantage is identically zero")
    return True, ""


class GRPOTrainer:
    """Group-relative policy optimisation over proposal prompts."""

    def __init__(
        self,
        backend: PolicyBackend,
        reward: Optional[CompositeReward] = None,
        judge: Optional[WorldModelJudge] = None,
        config: Optional[GRPOConfig] = None,
        mewm_config: Optional[MEWMConfig] = None,
    ) -> None:
        self.backend = backend
        self.mewm_config = mewm_config or load_config()
        self.reward = reward or CompositeReward(
            self.mewm_config.reward, self.mewm_config.evaluation)
        self.judge = judge
        self.config = config or GRPOConfig.from_training(self.mewm_config.training)
        self.history: List[Dict[str, Any]] = []
        self.skipped: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(self.config.seed)

    def step(
        self, prompts: Sequence[Dict[str, Any]], progress: float,
    ) -> Dict[str, Any]:
        """One optimisation step over a batch of proposal prompts."""
        rollouts: List[GroupRollout] = []
        skipped: List[Dict[str, Any]] = []
        for entry in prompts[: self.config.max_prompts_per_step]:
            samples = self.backend.sample(
                entry["prompt"], self.config.group_size, self.config.temperature)
            for sample in samples:
                judge_output = self._judge(entry, sample)
                sample.reward = self.reward.score(
                    sample.product, entry.get("truth", {}), judge_output,
                    entry.get("chain_ids", ()), progress,
                )
            rollout = GroupRollout(entry.get("id", ""), entry["prompt"], samples)

            if self.config.admit_prompts:
                admitted, reason = admit_prompt(
                    [s.score for s in samples], self.config.admit_std_min)
                if not admitted:
                    skipped.append({"prompt_id": entry.get("id", ""), "reason": reason})
                    continue

            rollout.compute_advantages()
            rollouts.append(rollout)

        self.skipped.extend(skipped)
        diagnostics = self.backend.update(
            rollouts, self.config.clip, self.config.kl_coefficient,
            self.config.learning_rate,
        )
        record = {
            "progress": round(progress, 4),
            "weights": curriculum_weights(progress),
            "reward": self._reward_stats(rollouts),
            "n_prompts": len(rollouts),
            "n_skipped": len(skipped),
            **diagnostics,
        }
        if skipped:
            record["skipped"] = skipped
        self.history.append(record)
        return record

    def train(
        self, prompt_source: Iterable[Dict[str, Any]],
        validate: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full schedule; ``validate`` is called every ``eval_every`` steps.

        Prompts are reshuffled once the pass over them is exhausted. Walking a fixed order
        for the whole run would pair the same prompts into the same batches at every pass,
        which correlates the gradient with an arbitrary property of the prompt list.
        """
        prompts = list(prompt_source)
        if not prompts:
            LOGGER.warning("no prompts supplied; nothing to train on")
            return self.history

        order = list(self._rng.permutation(len(prompts)))
        cursor = 0

        for step in range(self.config.total_steps):
            progress = step / max(1, self.config.total_steps - 1)

            batch: List[Dict[str, Any]] = []
            while len(batch) < self.config.max_prompts_per_step:
                if cursor >= len(order):
                    order = list(self._rng.permutation(len(prompts)))
                    cursor = 0
                batch.append(prompts[int(order[cursor])])
                cursor += 1
                if len(batch) >= len(prompts):
                    break

            record = self.step(batch, progress)
            record["step"] = step

            if validate and self.config.eval_every and step % self.config.eval_every == 0:
                record["validation"] = validate(step)
                LOGGER.info("step %d validation: %s", step, record["validation"])
        return self.history

    def _judge(self, entry: Dict[str, Any], sample: PolicySample) -> Dict[str, Any]:
        """Frozen-engine quantities for this sample's ``R_causal``."""
        if self.judge is None or entry.get("observed") is None:
            return dict(entry.get("judge", {}))
        try:
            return self.judge.evaluate(
                observed=entry["observed"],
                fine_label=str(sample.product.get("fine_label", "")),
                k_crit=sample.product.get("k_crit") or [],
                candidates=entry.get("candidates") or [],
                predicted_graph=sample.product.get("au_graph"),
                reference_graph=entry.get("reference_graph"),
                cid=entry.get("id", ""),
            )
        except Exception as exc:  # noqa: BLE001 - the judge must not kill the run
            LOGGER.warning("judge failed for %s: %s", entry.get("id"), exc)
            return {}

    #: The five terms of eq. (11), in the order the paper lists them. Named here rather
    #: than derived from whichever breakdown happens to arrive first, so a step whose
    #: samples all failed to score still reports the full set of keys.
    REWARD_COMPONENTS = ("r_au", "r_emo", "r_fmt", "r_causal", "r_temp")

    @classmethod
    def _reward_stats(cls, rollouts: Sequence[GroupRollout]) -> Dict[str, Any]:
        """Aggregate reward statistics for one step, total *and* per component.
        """
        all_rewards = np.concatenate([r.rewards() for r in rollouts]) if rollouts else np.array([])
        if all_rewards.size == 0:
            return {}
        stats: Dict[str, Any] = {
            "mean": round(float(all_rewards.mean()), 5),
            "std": round(float(all_rewards.std()), 5),
            "group_std_mean": round(
                float(np.mean([r.rewards().std() for r in rollouts])), 5),
        }

        breakdowns = [s.reward for r in rollouts for s in r.samples if s.reward is not None]
        components: Dict[str, Any] = {}
        for name in cls.REWARD_COMPONENTS:
            values = [float(getattr(b, name)) for b in breakdowns]
            components[name] = {
                "mean": round(float(np.mean(values)), 5) if values else 0.0,
                "std": round(float(np.std(values)), 5) if values else 0.0,
                "n": len(values),
            }
        stats["components"] = components
        # Every sample in a step is scored under the same curriculum weights, so the
        # last breakdown's copy is the step's.
        if breakdowns and breakdowns[-1].weights:
            stats["component_weights"] = dict(breakdowns[-1].weights)
        stats["n_unscored"] = sum(
            1 for r in rollouts for s in r.samples if s.reward is None)
        return stats

    def save_history(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.history, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        return target


__all__ = [
    "PolicySample", "GroupRollout", "PolicyBackend", "DryRunBackend", "LoRABackend",
    "GRPOConfig", "GRPOTrainer", "admit_prompt",
]
