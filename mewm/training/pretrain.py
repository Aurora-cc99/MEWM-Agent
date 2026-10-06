"""Pre-training stage: world-model prediction pre-training on video corpora."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..config import DynamicsConfig, MEWMConfig, RepresentationConfig, load_config
from ..knowledge.au_anatomy import K_SLOTS

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    _TORCH = True
except ImportError:
    torch = None
    _TORCH = False
    Dataset = object


@dataclass
class PretrainConfig:

    window: int = 64
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    kappa_low: float = 0.7
    slot_dropout: float = 0.15
    lambda_imagine: float = 1.0
    lambda_flow: float = 0.5
    lambda_slow: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    seed: int = 20260824


class LongVideoWindows(Dataset):

    def __init__(
        self,
        windows: Sequence[Dict[str, np.ndarray]],
        config: Optional[PretrainConfig] = None,
    ) -> None:
        self.windows = list(windows)
        self.config = config or PretrainConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        window = self.windows[index]
        measurements = np.asarray(window["measurements"], dtype=np.float32)
        activations = np.asarray(window["activations"], dtype=np.float32)
        n = measurements.shape[0]

        kappa = float(self.rng.uniform(self.config.kappa_low, 1.0))
        split = max(2, int(round(n * kappa)))
        split = min(split, n - 1) if n > 2 else max(1, n - 1)

        return {
            "measurements": measurements,
            "activations": activations,
            "split": split,
            "flow_velocity": np.asarray(
                window.get("flow_velocity", np.zeros_like(activations)), dtype=np.float32),
        }


def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not _TORCH:
        raise ImportError("collate needs PyTorch")
    longest = max(item["measurements"].shape[0] for item in batch)

    def _pad(array: np.ndarray) -> np.ndarray:
        pad = longest - array.shape[0]
        if pad <= 0:
            return array
        return np.concatenate([array, np.repeat(array[-1:], pad, axis=0)], axis=0)

    return {
        "measurements": torch.from_numpy(
            np.stack([_pad(item["measurements"]) for item in batch])),
        "activations": torch.from_numpy(
            np.stack([_pad(item["activations"]) for item in batch])),
        "flow_velocity": torch.from_numpy(
            np.stack([_pad(item["flow_velocity"]) for item in batch])),
        "split": torch.tensor([item["split"] for item in batch], dtype=torch.long),
    }


def harvest_windows(
    videos: Sequence[Any],
    config: Optional[MEWMConfig] = None,
    pretrain: Optional[PretrainConfig] = None,
    max_windows_per_video: int = 20,
    exclude_subjects: Sequence[str] = (),
) -> List[Dict[str, np.ndarray]]:
    from ..pipeline import run_representation

    config = config or load_config()
    pretrain = pretrain or PretrainConfig()
    excluded = set(exclude_subjects)
    windows: List[Dict[str, np.ndarray]] = []

    for video in videos:
        if video.subject in excluded:
            LOGGER.info("excluding subject %s from pre-training", video.subject)
            continue
        try:
            representation = run_representation(
                video, config, stride=1,
                max_frames=pretrain.window * max_windows_per_video,
            )
        except Exception as exc:
            LOGGER.warning("skipping %s: %s", video.video_id, exc)
            continue

        activations = representation.slot_activations
        if activations is None or activations.shape[0] < pretrain.window:
            continue

        from ..engines.v1_motion import MotionFrontEnd
        front_end = MotionFrontEnd(config.motion)
        measurement_rows = np.stack([
            front_end.measurement_matrix(
                representation.stream.frames[t].measurements
            ).reshape(-1)
            for t in representation.frames
        ])

        step = pretrain.window
        for start in range(0, activations.shape[0] - pretrain.window + 1, step):
            stop = start + pretrain.window
            windows.append({
                "measurements": measurement_rows[start:stop],
                "activations": activations[start:stop],
                "flow_velocity": np.diff(
                    activations[start:stop], axis=0, prepend=activations[start:start + 1]),
            })
            if len(windows) >= max_windows_per_video * len(videos):
                break
    LOGGER.info("harvested %d pre-training windows from %d videos",
                len(windows), len(videos))
    return windows


class WorldModelPretrainer:

    def __init__(
        self,
        config: Optional[MEWMConfig] = None,
        pretrain: Optional[PretrainConfig] = None,
    ) -> None:
        if not _TORCH:
            raise ImportError("stage-0 pre-training needs PyTorch")
        self.config = config or load_config()
        self.pretrain = pretrain or PretrainConfig()
        self.device = torch.device(
            self.pretrain.device if torch.cuda.is_available() else "cpu")

        from ..engines.m1_dynamics import AUDynamicsModel
        from ..engines.v2_slots import SlotEncoder
        from ..engines.v3_latent import LatentEncoder

        self.slot_encoder = SlotEncoder(self.config.representation).to(self.device)
        self.latent_encoder = LatentEncoder(self.config.representation).to(self.device)
        self.dynamics = AUDynamicsModel(
            self.config.dynamics, self.config.representation).to(self.device)

        parameters = (list(self.slot_encoder.parameters())
                      + list(self.latent_encoder.parameters())
                      + list(self.dynamics.parameters()))
        self.optimizer = torch.optim.AdamW(
            parameters, lr=self.pretrain.learning_rate,
            weight_decay=self.pretrain.weight_decay)
        self.history: List[Dict[str, float]] = []


    def compute_loss(self, batch: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        measurements = batch["measurements"].to(self.device)
        activations = batch["activations"].to(self.device)
        flow_velocity = batch["flow_velocity"].to(self.device)
        batch_size, steps, _ = measurements.shape
        n_roi = measurements.shape[-1] // 4

        total = torch.zeros((), device=self.device)
        parts = {"recon": 0.0, "dynamics": 0.0, "imagine": 0.0, "flow": 0.0, "slow": 0.0}

        frame_view = measurements.reshape(batch_size * steps, n_roi, 4)
        slots, predicted_activation, observed = self.slot_encoder(
            frame_view, slot_dropout=self.pretrain.slot_dropout)
        slots = slots.reshape(batch_size, steps, K_SLOTS, -1)
        predicted_activation = predicted_activation.reshape(batch_size, steps, K_SLOTS)

        mask = observed.reshape(batch_size, steps, K_SLOTS)
        recon = (F.mse_loss(predicted_activation, activations, reduction="none") * mask)
        recon = recon.sum() / mask.sum().clamp(min=1.0)
        total = total + recon
        parts["recon"] = float(recon.detach())

        latents = self.latent_encoder(measurements.reshape(batch_size * steps, n_roi, 4))
        z_slow = latents["z_slow"].reshape(batch_size, steps, -1)
        z_fast = latents["z_fast"].reshape(batch_size, steps, -1)
        predicted_velocity = latents["flow_velocity"].reshape(batch_size, steps, -1)

        dynamics_loss = torch.zeros((), device=self.device)
        for t in range(steps - 1):
            output = self.dynamics(slots[:, t], z_slow[:, t], z_fast[:, t])
            dynamics_loss = dynamics_loss + self.dynamics.slot_nll(
                output["slot_params"], slots[:, t + 1])
        dynamics_loss = dynamics_loss / max(1, steps - 1)
        total = total + dynamics_loss
        parts["dynamics"] = float(dynamics_loss.detach())

        split = int(batch["split"].min().item())
        split = max(1, min(split, steps - 1))
        imagined = self.dynamics.imagine(
            slots[:, split - 1], z_slow[:, split - 1], z_fast[:, split - 1])
        future = slots[:, split:]
        if future.shape[1] > 0:
            target = F.adaptive_avg_pool1d(
                future.mean(dim=2).transpose(1, 2),
                self.config.dynamics.future_queries,
            ).transpose(1, 2)
            queries = imagined["queries"].mean(dim=2)
            projected = queries[..., : target.shape[-1]]
            if projected.shape[-1] == target.shape[-1]:
                imagine_loss = self.dynamics.imagination_loss(
                    projected, target.mean(dim=1, keepdim=True).expand_as(projected))
                total = total + self.pretrain.lambda_imagine * imagine_loss
                parts["imagine"] = float(imagine_loss.detach())

        flow_loss = self.latent_encoder.flow_constraint_loss(
            z_fast.reshape(batch_size * steps, -1),
            predicted_velocity.reshape(batch_size * steps, -1),
        )
        total = total + self.pretrain.lambda_flow * flow_loss
        parts["flow"] = float(flow_loss.detach())

        prior_mean = z_slow[:, :-1].detach()
        prior_var = torch.full_like(prior_mean,
                                    self.config.representation.slow_process_noise
                                    + self.config.representation.slow_obs_noise)
        slow_loss = self.latent_encoder.slow_consistency_loss(
            z_slow[:, 1:], prior_mean, prior_var)
        total = total + self.pretrain.lambda_slow * slow_loss
        parts["slow"] = float(slow_loss.detach())

        parts["total"] = float(total.detach())
        return total, parts


    def fit(self, dataset: LongVideoWindows,
            output_dir: Optional[Path | str] = None) -> List[Dict[str, float]]:
        loader = DataLoader(dataset, batch_size=self.pretrain.batch_size, shuffle=True,
                            collate_fn=collate, drop_last=True)
        total_steps = max(1, len(loader) * self.pretrain.epochs)
        warmup = max(1, int(total_steps * self.pretrain.warmup_ratio))
        step = 0

        for epoch in range(self.pretrain.epochs):
            for batch in loader:
                step += 1
                learning_rate = self._schedule(step, warmup, total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = learning_rate

                self.optimizer.zero_grad()
                loss, parts = self.compute_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    self.pretrain.grad_clip,
                )
                self.optimizer.step()

                parts.update({"epoch": epoch, "step": step, "lr": learning_rate})
                self.history.append(parts)
                if step % 50 == 0:
                    LOGGER.info("step %d/%d loss %.4f", step, total_steps, parts["total"])

        if output_dir is not None:
            self.save(output_dir)
        return self.history

    def _schedule(self, step: int, warmup: int, total: int) -> float:
        if step <= warmup:
            return self.pretrain.learning_rate * step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return self.pretrain.learning_rate * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


    def save(self, output_dir: Path | str) -> Path:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.slot_encoder.state_dict(), target / "v2_slot_encoder.pt")
        torch.save(self.latent_encoder.state_dict(), target / "v3_latent_encoder.pt")
        self.dynamics.save(target / "m1_dynamics.pt")
        (target / "stage0_history.json").write_text(
            json.dumps(self.history, ensure_ascii=False, indent=1), encoding="utf-8")
        (target / "stage0_manifest.json").write_text(json.dumps({
            "version": self.dynamics.version_hash(),
            "representation": vars(self.config.representation),
            "dynamics": vars(self.config.dynamics),
            "pretrain": vars(self.pretrain),
            "n_steps": len(self.history),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        LOGGER.info("stage-0 checkpoint written to %s", target)
        return target


def calibrate_detection_thresholds(
    videos: Sequence[Any],
    config: Optional[MEWMConfig] = None,
    dynamics: Optional[Any] = None,
) -> Dict[str, float]:
    from ..engines.m2_spotting import alignment_auc, calibrate_thresholds
    from ..pipeline import run_representation, run_spotting

    config = config or load_config()
    curves, interval_sets, aucs = [], [], []

    for video in videos:
        events = video.micro_events()
        if not events:
            continue
        try:
            representation = run_representation(video, config)
            spotting = run_spotting(video, representation, config, dynamics)
        except Exception as exc:
            LOGGER.warning("calibration skipped %s: %s", video.video_id, exc)
            continue
        record = spotting.error_record
        intervals = [(e.onset - record.t_start, e.offset - record.t_start) for e in events]
        curves.append(record.s_curve)
        interval_sets.append(intervals)
        aucs.append(alignment_auc(record.s_curve, intervals))

    if not curves:
        LOGGER.warning("no labelled material for calibration; keeping the defaults")
        return {"tau_hi": config.spotting.tau_hi, "tau_lo": config.spotting.tau_lo,
                "alignment_auc": 0.0, "n_videos": 0}

    best = calibrate_thresholds(curves, interval_sets)
    best["alignment_auc"] = round(float(np.mean(aucs)), 4)
    best["n_videos"] = len(curves)
    LOGGER.info("calibrated thresholds: %s", best)
    return best


__all__ = [
    "PretrainConfig", "LongVideoWindows", "collate", "harvest_windows",
    "WorldModelPretrainer", "calibrate_detection_thresholds",
]
