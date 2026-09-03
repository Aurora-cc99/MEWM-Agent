"""Stage 1 -- supervised fine-tuning of the R-Agent policy.

**The epoch count is a ceiling.** ``sft_max_epochs`` is 100, but a fold's training pool is a
few hundred samples; 100 unchecked passes over it memorises the pool. The loop therefore
tracks the loss curve and stops when it has genuinely flattened (``patience`` epochs without
a ``min_delta`` improvement), recording the epoch it selected. The configured 100 remains the
ceiling, so the setting is honoured without spending the run on the overfitting regime.

**The loss curve is a first-class output.** Sufficiency judgement 1b reads it, so it is
retained per-epoch as well as per-step, and :meth:`SFTTrainer.curve` hands back exactly the
series the diagnostic expects.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import MEWMConfig, TrainingConfig, load_config

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Localisation supervision inside the SFT loss (修改方案 §3)
# ---------------------------------------------------------------------------

#: Onset/offset/apex numbers inside a part1-style proposal payload -- the tokens that
#: carry the localisation answer. Matches both the keyed form (``"onset": 57``) and
#: the bare triple form (``[57, 71, 62]`` inside a ``proposals`` list).
_PROP_KEYED_RE = re.compile(r'"(?:onset|offset|apex)"\s*:\s*(\d+)')
_PROP_TRIPLE_RE = re.compile(
    r'"(?:proposals|part1_proposals)"\s*:\s*\[(.*?)\]\s*[,}]', re.DOTALL)
_NUMBER_RE = re.compile(r"\d+")


def proposal_number_spans(text: str) -> List[Tuple[int, int]]:
    """Character spans of the localisation numbers in an assistant target.

    These are the positions whose cross-entropy gets the extra ``lambda_prop`` weight,
    so the interval tokens and the analysis tokens share one backward pass
    (方案 §3 -- "定位网络与分析策略在一个优化步里共享梯度").
    """
    spans: List[Tuple[int, int]] = []
    for match in _PROP_KEYED_RE.finditer(text):
        spans.append(match.span(1))
    for block in _PROP_TRIPLE_RE.finditer(text):
        body = block.group(1)
        if "{" in body:
            # Keyed objects inside the list: the keyed regex above already picked the
            # localisation numbers; taking every digit here would also weight ids.
            continue
        offset = block.start(1)
        for number in _NUMBER_RE.finditer(body):
            spans.append((offset + number.start(), offset + number.end()))
    # De-duplicate (a keyed number inside a proposals block matches twice).
    return sorted(set(spans))


def soft_iou(predicted: Tuple[float, float], truth: Tuple[float, float]) -> float:
    """Plain interval IoU, exposed for the eval/reward side of ``L_prop``."""
    lo = max(min(predicted), min(truth))
    hi = min(max(predicted), max(truth))
    intersection = max(0.0, hi - lo + 1.0)
    union = ((max(predicted) - min(predicted) + 1.0)
             + (max(truth) - min(truth) + 1.0) - intersection)
    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class SFTBackend(ABC):
    """Interface between the SFT schedule and whatever holds the parameters."""

    @abstractmethod
    def step(self, batch: Sequence[Dict[str, Any]], learning_rate: float) -> float:
        """Apply one optimiser step over a batch of chat samples; return the loss."""

    def evaluate(self, batch: Sequence[Dict[str, Any]]) -> float:
        """Loss without an update. Defaults to the training loss for backends that
        cannot separate the two, which keeps a dry run honest about what it measured."""
        return float("nan")

    def save(self, path: Path | str) -> Optional[Path]:
        return None

    def snapshot(self) -> Optional[Any]:
        """A restorable copy of the trainable weights, or ``None`` if unsupported.
        """
        return None

    def restore(self, state: Any) -> None:
        """Load a snapshot back into the model."""


class DryRunSFT(SFTBackend):
    """No-parameter backend driven by a caller-supplied loss function.

    Used to validate the schedule and the stopping rule. The default loss is a decaying
    curve with noise, which is the shape the stopping rule has to handle correctly.
    """

    def __init__(self, loss_fn: Optional[Callable[[int, Sequence[Dict[str, Any]]], float]] = None,
                 seed: int = 20260824) -> None:
        self.loss_fn = loss_fn
        self.calls = 0
        self.rng = np.random.default_rng(seed)

    def step(self, batch: Sequence[Dict[str, Any]], learning_rate: float) -> float:
        self.calls += 1
        if self.loss_fn is not None:
            return float(self.loss_fn(self.calls, batch))
        return float(2.0 * math.exp(-self.calls / 40.0) + 0.05
                     + 0.01 * self.rng.standard_normal())

    def evaluate(self, batch: Sequence[Dict[str, Any]]) -> float:
        if self.loss_fn is not None:
            return float(self.loss_fn(self.calls, batch))
        return float(2.0 * math.exp(-self.calls / 40.0) + 0.05)


class LoRASFTBackend(SFTBackend):
    """PyTorch/PEFT backend.

    Thin by design: it adapts an already-constructed model and tokenizer rather than owning
    their configuration, so this module stays independent of any particular serving stack.
    Loss is computed on the assistant turn only -- supervising the prompt tokens would train
    the policy to reproduce the question, which competes with the objective for capacity.
    """

    def __init__(self, model: Any = None, tokenizer: Any = None,
                 device: str = "cuda", max_length: int = 4096,
                 weight_decay: float = 0.01, prop_weight: float = 0.0) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError("LoRASFTBackend needs PyTorch installed.") from exc
        if model is None or tokenizer is None:
            raise ValueError(
                "LoRASFTBackend needs a constructed model and tokenizer. Build the policy "
                "(open-weight base + SFT LoRA adapter) in your launcher and pass it in."
            )
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        #: 方案 §3 -- lambda_prop. Extra CE weight on the onset/offset/apex number
        #: tokens of the assistant target, so the localisation answer is supervised
        #: harder than the surrounding prose in the *same* backward pass. 0 preserves
        #: the historical plain-CE behaviour exactly.
        self.prop_weight = float(max(0.0, prop_weight))
        self._warned_no_offsets = False
        #: Identities of the samples whose prompt alone exceeded ``max_length``. A *set*,
        #: not a counter: ``_encode`` runs once per sample per batch per epoch, and again
        #: for every monitor-loss pass, so a counter would report "17 dropped" for one
        #: over-length sample seen across 17 encodings. The reported figure has to be the
        #: number of distinct training samples lost.
        self._dropped: set = set()
        #: How many times an over-length sample was re-encountered, drops included. Only
        #: interesting next to :attr:`n_dropped` -- a large ratio means the same handful
        #: of samples is being re-encoded, which is expected, not a second problem.
        self.n_drop_events = 0
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-5, weight_decay=weight_decay)

    @property
    def n_dropped(self) -> int:
        """Distinct samples dropped for being unsupervisable at this ``max_length``."""
        return len(self._dropped)

    @staticmethod
    def _sample_key(sample: Dict[str, Any]) -> str:
        """A stable identity for a sample, so re-encoding it is not a second drop."""
        for field_name in ("id", "sample_id", "pair_id"):
            value = sample.get(field_name)
            if value:
                return f"{field_name}:{value}"
        # No declared id: hash the content. Two byte-identical samples are the same
        # sample for this purpose -- the point is to count losses, not occurrences.
        digest = hashlib.blake2b(
            json.dumps(sample.get("messages", []), ensure_ascii=False,
                       sort_keys=True).encode("utf-8"),
            digest_size=8).hexdigest()
        return f"digest:{digest}"

    # -- masking ------------------------------------------------------------

    def _encode(self, sample: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
        """Tokenise one chat sample, masking everything but the assistant turn.
        """
        torch = self.torch
        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        target = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self._render_prompt(prompt_messages, target)
        else:  # pragma: no cover - tokenizers without a template
            prompt_text = "\n".join(m["content"] for m in prompt_messages) + "\n"

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            target_ids = list(target_ids) + [self.tokenizer.eos_token_id]

        if len(prompt_ids) >= self.max_length or not target_ids:
            key = self._sample_key(sample)
            self.n_drop_events += 1
            first_time = key not in self._dropped
            self._dropped.add(key)
            if first_time:
                LOGGER.warning(
                    "SFT sample %r dropped: prompt is %d token(s) against a max_length "
                    "of %d, so no supervised token survives truncation (%d distinct "
                    "sample(s) dropped so far)",
                    sample.get("id", sample.get("sample_id", "<unnamed>")),
                    len(prompt_ids), self.max_length, self.n_dropped)
            return None

        target_weights = self._target_weights(target, len(target_ids))
        input_ids = (list(prompt_ids) + list(target_ids))[: self.max_length]
        labels = ([-100] * len(prompt_ids) + list(target_ids))[: self.max_length]
        weights = ([1.0] * len(prompt_ids) + list(target_weights))[: self.max_length]
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(weights, dtype=torch.float))

    def _render_prompt(self, prompt_messages: List[Dict[str, Any]],
                       target: str) -> str:
        """Render the prompt with the template's thinking mode set from the target.

        * target with a chain of thought + ``enable_thinking=False`` -- the template
          pre-closes the block, so the target's own ``<think>`` is supervised as
          ordinary text inside the answer, and the model learns to emit a literal
          ``<think>`` tag after the block has already closed;
        * target without one + ``enable_thinking=True`` -- the model is supervised to
          jump straight to the answer where a chain of thought was expected, which is
          exactly how a thinking policy gets trained out of thinking.
        """
        from ..llm.local_models import THINK_OPEN
        wants_thinking = THINK_OPEN in (target or "")
        try:
            return self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=wants_thinking)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True)

    def _target_weights(self, target: str, n_target_tokens: int) -> List[float]:
        """Per-token CE weights over the assistant turn (方案 §3, lambda_prop).

        1.0 everywhere; ``1 + prop_weight`` on tokens overlapping an onset/offset/apex
        number span. Needs a fast tokenizer for offset mapping; without one the term
        degrades to plain CE with a single warning rather than a crash.
        """
        weights = [1.0] * n_target_tokens
        if self.prop_weight <= 0.0:
            return weights
        spans = proposal_number_spans(target)
        if not spans:
            return weights
        try:
            encoding = self.tokenizer(target, add_special_tokens=False,
                                      return_offsets_mapping=True)
            offsets = encoding["offset_mapping"]
        except Exception:  # noqa: BLE001 - slow tokenizers have no offsets
            if not self._warned_no_offsets:
                LOGGER.warning(
                    "tokenizer exposes no offset mapping; the lambda_prop interval-"
                    "token weighting is disabled and SFT falls back to plain CE")
                self._warned_no_offsets = True
            return weights
        for i, (a, b) in enumerate(offsets[:n_target_tokens]):
            if any(a < hi and b > lo for lo, hi in spans):
                weights[i] = 1.0 + self.prop_weight
        return weights

    def _collate(self, batch: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        torch = self.torch
        encoded = [trio for trio in (self._encode(s) for s in batch) if trio is not None]
        if not encoded:
            return None
        longest = max(ids.shape[0] for ids, _, _ in encoded)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0

        input_ids, labels, attention, weights = [], [], [], []
        for ids, lab, wts in encoded:
            pad = longest - ids.shape[0]
            input_ids.append(torch.cat([ids, torch.full((pad,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=torch.long)]))
            attention.append(torch.cat([torch.ones(ids.shape[0], dtype=torch.long),
                                        torch.zeros(pad, dtype=torch.long)]))
            weights.append(torch.cat([wts, torch.zeros(pad, dtype=torch.float)]))
        return {
            "input_ids": torch.stack(input_ids).to(self.device),
            "labels": torch.stack(labels).to(self.device),
            "attention_mask": torch.stack(attention).to(self.device),
            "loss_weights": torch.stack(weights).to(self.device),
        }

    def _loss(self, encoded: Dict[str, Any]) -> Any:
        """Assistant-turn CE; token-weighted when ``prop_weight`` is active (方案 §3)."""
        torch = self.torch
        weights = encoded.pop("loss_weights")
        outputs = self.model(**encoded)
        if self.prop_weight <= 0.0:
            return outputs.loss
        logits = outputs.logits[:, :-1, :]
        labels = encoded["labels"][:, 1:]
        shifted_weights = weights[:, 1:]
        mask = labels != -100
        flat_ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.clamp_min(0).reshape(-1), reduction="none",
        ).reshape(labels.shape)
        weighted = flat_ce * shifted_weights * mask
        denom = (shifted_weights * mask).sum().clamp_min(1.0)
        return weighted.sum() / denom

    # -- steps --------------------------------------------------------------

    def step(self, batch: Sequence[Dict[str, Any]], learning_rate: float) -> float:
        torch = self.torch
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.model.train()
        self.optimizer.zero_grad()
        encoded = self._collate(batch)
        if encoded is None:
            return float("nan")
        loss = self._loss(encoded)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.optimizer.step()
        return float(loss.detach())

    def evaluate(self, batch: Sequence[Dict[str, Any]]) -> float:
        torch = self.torch
        self.model.eval()
        encoded = self._collate(batch)
        if encoded is None:
            return float("nan")
        with torch.no_grad():
            loss = self._loss(encoded)
        return float(loss.detach())

    def save(self, path: Path | str) -> Optional[Path]:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(target))
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(target))
        return target

    def snapshot(self) -> Optional[Any]:
        """Detached CPU copy of the trainable parameters only.

        Trainable-only keeps this cheap: under LoRA that is the adapter, tens of MB, not
        the frozen base. CPU keeps the best snapshot from competing with the live model
        for device memory across the patience window.
        """
        return {
            name: param.detach().to("cpu").clone()
            for name, param in self.model.named_parameters() if param.requires_grad
        }

    def restore(self, state: Any) -> None:
        if not state:
            return
        torch = self.torch
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.copy_(state[name].to(param.device))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class SFTOutcome:
    """What one SFT run produced."""

    epochs_run: int = 0
    max_epochs: int = 0
    selected_epoch: int = 0
    best_loss: float = float("inf")
    final_loss: float = float("inf")
    stopped_early: bool = False
    stop_reason: str = ""
    n_samples: int = 0
    epoch_losses: List[float] = field(default_factory=list)
    step_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    evaluated_on: str = "training_pool"
    #: True when the saved weights were rolled back to ``selected_epoch``. False means
    #: the artefact is the last epoch -- read ``stop_reason`` for why.
    restored_selected_epoch: bool = False
    #: *Distinct* samples the backend refused to encode (prompt alone longer than
    #: ``max_length``), not encode attempts -- see ``LoRASFTBackend.n_dropped``.
    n_dropped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs_run": self.epochs_run, "max_epochs": self.max_epochs,
            "selected_epoch": self.selected_epoch,
            "best_loss": round(self.best_loss, 6),
            "final_loss": round(self.final_loss, 6),
            "stopped_early": self.stopped_early, "stop_reason": self.stop_reason,
            "n_samples": self.n_samples,
            "n_dropped": self.n_dropped,
            "epoch_losses": [round(v, 6) for v in self.epoch_losses],
            "eval_losses": [round(v, 6) for v in self.eval_losses],
            "evaluated_on": self.evaluated_on,
            "restored_selected_epoch": self.restored_selected_epoch,
        }


class SFTTrainer:
    """Cross-entropy fine-tuning with a plateau-aware stopping rule."""

    def __init__(
        self,
        backend: SFTBackend,
        config: Optional[TrainingConfig] = None,
        mewm_config: Optional[MEWMConfig] = None,
        seed: int = 20260824,
    ) -> None:
        self.backend = backend
        self.mewm_config = mewm_config or load_config()
        self.config = config or self.mewm_config.training
        self.rng = np.random.default_rng(seed)
        self.outcome = SFTOutcome(max_epochs=self.config.sft_max_epochs)

    # -- schedule -----------------------------------------------------------

    def _learning_rate(self, step: int, warmup: int, total: int) -> float:
        """Linear warm-up then cosine decay, matching the stage-0 schedule."""
        base = self.config.sft_learning_rate
        if step <= warmup:
            return base * step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    def _batches(self, samples: Sequence[Dict[str, Any]]) -> Iterable[List[Dict[str, Any]]]:
        order = self.rng.permutation(len(samples))
        size = max(1, self.config.sft_batch_size)
        for start in range(0, len(order), size):
            yield [samples[int(i)] for i in order[start:start + size]]

    # -- loop ---------------------------------------------------------------

    def fit(
        self,
        samples: Sequence[Dict[str, Any]],
        eval_samples: Sequence[Dict[str, Any]] = (),
        output_dir: Optional[Path | str] = None,
        in_sample_eval: bool = True,
    ) -> SFTOutcome:
        """Run the schedule.
        """
        samples = list(samples)
        self.outcome = SFTOutcome(max_epochs=self.config.sft_max_epochs,
                                  n_samples=len(samples))
        if not samples:
            LOGGER.warning("no SFT samples supplied; nothing to train on")
            self.outcome.stop_reason = "no samples"
            return self.outcome

        monitor = list(eval_samples) if eval_samples else samples
        self.outcome.evaluated_on = "training_pool" if in_sample_eval else "held_out_subjects"

        batches_per_epoch = max(1, math.ceil(len(samples) / max(1, self.config.sft_batch_size)))
        total_steps = batches_per_epoch * self.config.sft_max_epochs
        warmup = max(1, int(total_steps * self.config.sft_warmup_ratio))

        step = 0
        best = float("inf")
        best_state: Optional[Any] = None
        patience_left = self.config.sft_patience

        for epoch in range(1, self.config.sft_max_epochs + 1):
            epoch_losses: List[float] = []
            for batch in self._batches(samples):
                step += 1
                loss = self.backend.step(batch, self._learning_rate(step, warmup, total_steps))
                if math.isfinite(loss):
                    epoch_losses.append(float(loss))
                    self.outcome.step_losses.append(float(loss))

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            self.outcome.epoch_losses.append(epoch_loss)
            self.outcome.epochs_run = epoch
            self.outcome.final_loss = epoch_loss

            # Monitored loss over the *whole* monitor set, batched. Evaluating only the
            # first ``sft_batch_size`` samples -- the same four, every epoch, never
            # reshuffled -- made the stopping rule a function of whichever handful landed
            # at the front of the list: four easy samples stop the run early, four hard
            # ones never plateau. The sets here are small enough that a full pass costs
            # nothing worth saving.
            monitored = self._monitor_loss(monitor)
            if not math.isfinite(monitored):
                monitored = epoch_loss
            self.outcome.eval_losses.append(float(monitored))

            if monitored < best - self.config.sft_min_delta:
                best = float(monitored)
                self.outcome.best_loss = best
                self.outcome.selected_epoch = epoch
                patience_left = self.config.sft_patience
                # Snapshot *now*, at the epoch being selected. By the time the loop
                # knows this was the best epoch it is ``sft_patience`` epochs further on
                # and the weights have moved.
                best_state = self.backend.snapshot()
            else:
                patience_left -= 1

            if epoch % 10 == 0 or epoch == 1:
                LOGGER.info("SFT epoch %d/%d: loss %.5f (monitored %.5f, best %.5f)",
                            epoch, self.config.sft_max_epochs, epoch_loss, monitored, best)

            if self.config.sft_early_stop and patience_left <= 0:
                self.outcome.stopped_early = True
                self.outcome.stop_reason = (
                    f"no improvement greater than {self.config.sft_min_delta} for "
                    f"{self.config.sft_patience} epochs; selected epoch "
                    f"{self.outcome.selected_epoch}"
                )
                LOGGER.info("SFT stopped early at epoch %d: %s", epoch,
                            self.outcome.stop_reason)
                break

        if not self.outcome.stopped_early:
            self.outcome.stop_reason = (
                f"reached the configured ceiling of {self.config.sft_max_epochs} epochs")
        if not math.isfinite(self.outcome.best_loss):
            self.outcome.best_loss = self.outcome.final_loss
            self.outcome.selected_epoch = self.outcome.epochs_run

        # Roll back to the selected epoch before anything is written. Otherwise the
        # artefact on disk is the last epoch -- ``sft_patience`` epochs past the one the
        # outcome file names -- and stage 3 starts from weights nobody chose.
        self.outcome.restored_selected_epoch = False
        if best_state is not None and self.outcome.selected_epoch < self.outcome.epochs_run:
            self.backend.restore(best_state)
            self.outcome.restored_selected_epoch = True
            LOGGER.info("SFT restored epoch %d (best monitored loss %.5f) before saving",
                        self.outcome.selected_epoch, self.outcome.best_loss)
        elif (best_state is None
              and self.outcome.selected_epoch < self.outcome.epochs_run):
            self.outcome.stop_reason += (
                f"; backend cannot snapshot, so the saved weights are epoch "
                f"{self.outcome.epochs_run}, not the selected epoch "
                f"{self.outcome.selected_epoch}")
            LOGGER.warning(
                "SFT backend %s has no snapshot(): saving epoch %d, not the selected "
                "epoch %d", type(self.backend).__name__,
                self.outcome.epochs_run, self.outcome.selected_epoch)
        self.outcome.n_dropped = int(getattr(self.backend, "n_dropped", 0))

        if output_dir is not None:
            self.save(output_dir)
        return self.outcome

    def _monitor_loss(self, monitor: Sequence[Dict[str, Any]]) -> float:
        """Mean backend loss over the whole monitor set, batch by batch."""
        size = max(1, self.config.sft_batch_size)
        losses = []
        for start in range(0, len(monitor), size):
            value = self.backend.evaluate(monitor[start:start + size])
            if math.isfinite(value):
                losses.append(float(value))
        return float(np.mean(losses)) if losses else float("nan")

    def curve(self, per_step: bool = False) -> List[float]:
        """The loss series the plateau diagnostic reads."""
        return list(self.outcome.step_losses if per_step else self.outcome.epoch_losses)

    def save(self, output_dir: Path | str) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.backend.save(target)
        (target / "sft_outcome.json").write_text(
            json.dumps(self.outcome.to_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8")
        return target


# ---------------------------------------------------------------------------
# Sample assembly
# ---------------------------------------------------------------------------


def build_sft_samples(
    instruction_samples: Sequence[Any],
    roles: Sequence[str] = ("P", "A", "R", "C"),
    include_end_to_end: bool = True,
    augmented: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Render instruction samples to chat form, plus any augmented QA pairs.

    Augmented pairs enter as end-to-end question/answer chats -- the same shape as a
    reference triple-task item -- so a fold that has run stage 2 trains on a strictly
    larger set with no change to the loop.
    """
    from .instruction_set import to_chat_format

    chats: List[Dict[str, Any]] = []
    for sample in instruction_samples:
        for role in roles:
            try:
                chats.append(to_chat_format(sample, role))
            except KeyError:
                continue
        if include_end_to_end:
            chats.append(to_chat_format(sample))

    for record in augmented:
        chats.append({
            "id": record.get("video_id", ""),
            "messages": [
                {"role": "user", "content": record.get("question", "")},
                {"role": "assistant", "content": str(record.get("answer", ""))},
            ],
            "source": "augmented",
        })
    return chats


__all__ = [
    "SFTBackend", "DryRunSFT", "LoRASFTBackend", "SFTOutcome", "SFTTrainer",
    "build_sft_samples", "proposal_number_spans", "soft_iou",
]
