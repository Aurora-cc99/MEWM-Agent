"""Supervised fine-tuning stage: role-specific LoRA adapter training."""

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


_PROP_KEYED_RE = re.compile(r'"(?:onset|offset|apex)"\s*:\s*(\d+)')
_PROP_TRIPLE_RE = re.compile(
    r'"(?:proposals|part1_proposals)"\s*:\s*\[(.*?)\]\s*[,}]', re.DOTALL)
_NUMBER_RE = re.compile(r"\d+")


def proposal_number_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for match in _PROP_KEYED_RE.finditer(text):
        spans.append(match.span(1))
    for block in _PROP_TRIPLE_RE.finditer(text):
        body = block.group(1)
        if "{" in body:
            continue
        offset = block.start(1)
        for number in _NUMBER_RE.finditer(body):
            spans.append((offset + number.start(), offset + number.end()))
    return sorted(set(spans))


def soft_iou(predicted: Tuple[float, float], truth: Tuple[float, float]) -> float:
    lo = max(min(predicted), min(truth))
    hi = min(max(predicted), max(truth))
    intersection = max(0.0, hi - lo + 1.0)
    union = ((max(predicted) - min(predicted) + 1.0)
             + (max(truth) - min(truth) + 1.0) - intersection)
    return float(intersection / union) if union > 0 else 0.0


class SFTBackend(ABC):

    @abstractmethod
    def step(self, batch: Sequence[Dict[str, Any]], learning_rate: float) -> float:

        ...
    def evaluate(self, batch: Sequence[Dict[str, Any]]) -> float:
        return float("nan")

    def save(self, path: Path | str) -> Optional[Path]:
        return None

    def snapshot(self) -> Optional[Any]:
        return None

    def restore(self, state: Any) -> None:


        ...
class DryRunSFT(SFTBackend):

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

    def __init__(self, model: Any = None, tokenizer: Any = None,
                 device: str = "cuda", max_length: int = 4096,
                 weight_decay: float = 0.01, prop_weight: float = 0.0) -> None:
        try:
            import torch
        except ImportError as exc:
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
        self.prop_weight = float(max(0.0, prop_weight))
        self._warned_no_offsets = False
        self._dropped: set = set()
        self.n_drop_events = 0
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-5, weight_decay=weight_decay)

    @property
    def n_dropped(self) -> int:
        return len(self._dropped)

    @staticmethod
    def _sample_key(sample: Dict[str, Any]) -> str:
        for field_name in ("id", "sample_id", "pair_id"):
            value = sample.get(field_name)
            if value:
                return f"{field_name}:{value}"
        digest = hashlib.blake2b(
            json.dumps(sample.get("messages", []), ensure_ascii=False,
                       sort_keys=True).encode("utf-8"),
            digest_size=8).hexdigest()
        return f"digest:{digest}"


    def _encode(self, sample: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
        torch = self.torch
        messages = sample["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        target = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self._render_prompt(prompt_messages, target)
        else:
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
        except Exception:
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


@dataclass
class SFTOutcome:

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
    restored_selected_epoch: bool = False
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


    def _learning_rate(self, step: int, warmup: int, total: int) -> float:
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


    def fit(
        self,
        samples: Sequence[Dict[str, Any]],
        eval_samples: Sequence[Dict[str, Any]] = (),
        output_dir: Optional[Path | str] = None,
        in_sample_eval: bool = True,
    ) -> SFTOutcome:
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

            monitored = self._monitor_loss(monitor)
            if not math.isfinite(monitored):
                monitored = epoch_loss
            self.outcome.eval_losses.append(float(monitored))

            threshold = self.config.sft_min_delta
            if math.isfinite(best):
                threshold = max(threshold, best * self.config.sft_min_delta_relative)
            if monitored < best - threshold:
                best = float(monitored)
                self.outcome.best_loss = best
                self.outcome.selected_epoch = epoch
                patience_left = self.config.sft_patience
                best_state = self.backend.snapshot()
            else:
                patience_left -= 1

            if epoch % 10 == 0 or epoch == 1:
                LOGGER.info("SFT epoch %d/%d: loss %.5f (monitored %.5f, best %.5f)",
                            epoch, self.config.sft_max_epochs, epoch_loss, monitored, best)

            if self.config.sft_early_stop and patience_left <= 0:
                self.outcome.stopped_early = True
                self.outcome.stop_reason = (
                    f"no improvement greater than max({self.config.sft_min_delta}, "
                    f"{self.config.sft_min_delta_relative:.0%} of best) for "
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
        size = max(1, self.config.sft_batch_size)
        losses = []
        for start in range(0, len(monitor), size):
            value = self.backend.evaluate(monitor[start:start + size])
            if math.isfinite(value):
                losses.append(float(value))
        return float(np.mean(losses)) if losses else float("nan")

    def curve(self, per_step: bool = False) -> List[float]:
        return list(self.outcome.step_losses if per_step else self.outcome.epoch_losses)

    def save(self, output_dir: Path | str) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.backend.save(target)
        (target / "sft_outcome.json").write_text(
            json.dumps(self.outcome.to_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8")
        return target


def build_sft_samples(
    instruction_samples: Sequence[Any],
    roles: Sequence[str] = ("P", "A", "R", "C"),
    include_end_to_end: bool = True,
    augmented: Sequence[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
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
