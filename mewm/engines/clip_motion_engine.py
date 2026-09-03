"""CLIP-style dual-tower motion representation engine (修改方案 §1, 实施步骤 2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import ClipConfig

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:  # pragma: no cover - mirrors the other engine modules
    torch = None  # type: ignore
    nn = object  # type: ignore
    F = None  # type: ignore
    _TORCH = False


class ClipWeightsUnavailableError(RuntimeError):
    """The configured CLIP checkpoint directory is missing or unreadable."""


def _require_transformers():
    try:
        import transformers
        return transformers
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the CLIP motion engine needs transformers:\n"
            "    pip install -U transformers\n"
            f"(import failed: {exc})"
        ) from exc


def resolve_clip_weights(config: ClipConfig) -> Path:
    """The local CLIP checkpoint directory, validated before anything loads."""
    path = Path(config.weights_path)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ClipWeightsUnavailableError(
            f"CLIP weights not found at {path}. Set clip.weights_path in "
            f"configs/mewm_agent.yaml (or MEWM_CLIP__WEIGHTS_PATH) to a local "
            f"clip-vit-base-patch16 checkout containing config.json + pytorch_model.bin."
        )
    return path


if _TORCH:

    class GatedFusion(nn.Module):
        """``u = sigma(W [v; m]) ⊙ v + (1 − sigma) ⊙ m`` (方案 §1.4)."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.gate = nn.Linear(2 * dim, dim)

        def forward(self, v: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            g = torch.sigmoid(self.gate(torch.cat([v, m], dim=-1)))
            return g * v + (1.0 - g) * m

    class LocalCrossAttentionFusion(nn.Module):
        """Learnable cross-attention between visual and motion-description embeddings
        (formwork.md 第 III 条, ``MEWM-Agent_完整执行方案.md`` 第 2.4 节, 2026-09-03 新增).
        """

        def __init__(self, dim: int, n_heads: int = 4, window_radius: int = 1,
                    dropout: float = 0.1) -> None:
            super().__init__()
            self.window_radius = max(0, int(window_radius))
            self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                                   batch_first=True)
            self.norm = nn.LayerNorm(dim)

        def _band_mask(self, length: int, device: "torch.device") -> "torch.Tensor":
            """``(T, T)`` boolean mask, ``True`` where attention is blocked."""
            index = torch.arange(length, device=device)
            distance = (index.unsqueeze(0) - index.unsqueeze(1)).abs()
            return distance > self.window_radius

        def forward(self, v: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            """``(B, T, d)`` visual, ``(B, T, d)`` motion-description -> ``(B, T, d)``.

            A single-frame input (``T == 1``, e.g. inference on one frame at a time)
            degenerates to attending over itself, which is a harmless identity-like
            pass rather than a special case the caller has to know about.
            """
            length = v.shape[1]
            mask = self._band_mask(length, v.device) if length > 1 else None
            attended, _ = self.attention(v, m, m, attn_mask=mask)
            return self.norm(v + attended)

    class TransitionHead(nn.Module):
        """Maps the fused motion representation to (T, K) slot-shaped activations.

        This is the trainable replacement for ``analytic_slot_readout`` +
        ``AnalyticDynamics``: same output contract (K activations in [0, 1] per frame),
        different provenance (gradient-trained instead of rule-derived).
        """

        def __init__(self, dim: int, n_slots: int, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_slots))

        def forward(self, u: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.net(u))

    class MotionCLIP(nn.Module):
        """The dual tower: a local CLIP checkpoint with partial fine-tuning.

        * the last ``vision_unfreeze_layers`` vision encoder layers (+ post layernorm
          and the visual projection),
        * the last ``text_unfreeze_layers`` text encoder layers (+ final layernorm and
          the text projection),
        * the logit scale.
        """

        def __init__(self, config: ClipConfig) -> None:
            super().__init__()
            self.config = config
            transformers = _require_transformers()
            source = str(resolve_clip_weights(config))
            self.clip = transformers.CLIPModel.from_pretrained(source)
            self.processor = transformers.CLIPProcessor.from_pretrained(source)
            self.embed_dim = int(self.clip.config.projection_dim)
            self._freeze(config.vision_unfreeze_layers, config.text_unfreeze_layers)
            LOGGER.info(
                "MotionCLIP: loaded %s (dim %d); fine-tuning last %d vision / %d text "
                "layer(s), %d trainable parameter tensor(s)",
                source, self.embed_dim, config.vision_unfreeze_layers,
                config.text_unfreeze_layers,
                sum(1 for p in self.clip.parameters() if p.requires_grad))

        # -- freezing -----------------------------------------------------------

        def _freeze(self, vision_unfrozen: int, text_unfrozen: int) -> None:
            for parameter in self.clip.parameters():
                parameter.requires_grad_(False)

            vision_layers = self.clip.vision_model.encoder.layers
            for layer in vision_layers[len(vision_layers) - max(0, vision_unfrozen):]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            if vision_unfrozen > 0:
                for parameter in self.clip.vision_model.post_layernorm.parameters():
                    parameter.requires_grad_(True)
                for parameter in self.clip.visual_projection.parameters():
                    parameter.requires_grad_(True)

            text_layers = self.clip.text_model.encoder.layers
            for layer in text_layers[len(text_layers) - max(0, text_unfrozen):]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
            if text_unfrozen > 0:
                for parameter in self.clip.text_model.final_layer_norm.parameters():
                    parameter.requires_grad_(True)
                for parameter in self.clip.text_projection.parameters():
                    parameter.requires_grad_(True)

            self.clip.logit_scale.requires_grad_(True)

        def unfrozen_layer_counts(self) -> Dict[str, int]:
            """How many encoder layers actually carry gradient, for the audit trail."""
            def _count(layers) -> int:
                return sum(
                    1 for layer in layers
                    if any(p.requires_grad for p in layer.parameters())
                )
            return {
                "vision": _count(self.clip.vision_model.encoder.layers),
                "text": _count(self.clip.text_model.encoder.layers),
            }

        # -- encoding -----------------------------------------------------------

        def encode_images(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
            """(B, 3, H, W) -> L2-normalised (B, d)."""
            features = self.clip.get_image_features(pixel_values=pixel_values)
            return F.normalize(features, dim=-1)

        def encode_texts(self, input_ids: "torch.Tensor",
                         attention_mask: "torch.Tensor") -> "torch.Tensor":
            """Tokenised descriptions -> L2-normalised (B, d)."""
            features = self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask)
            return F.normalize(features, dim=-1)

        def preprocess_images(self, image_paths: Sequence[str | Path]) -> "torch.Tensor":
            from PIL import Image
            images = [Image.open(str(p)).convert("RGB") for p in image_paths]
            batch = self.processor(images=images, return_tensors="pt")
            return batch["pixel_values"]

        def tokenize(self, texts: Sequence[str]) -> Dict[str, "torch.Tensor"]:
            # CLIP's positional table stops at 77 tokens; truncation is the contract.
            return self.processor.tokenizer(
                list(texts), padding=True, truncation=True, max_length=77,
                return_tensors="pt")

    def info_nce(v: "torch.Tensor", m: "torch.Tensor",
                 temperature: float = 0.07) -> "torch.Tensor":
        """Bidirectional frame-level InfoNCE (方案 §1.4, eq. L_align).

        Frame ``i``'s visual embedding is pulled toward frame ``i``'s motion
        description and pushed from the other frames of the same batch (= the same
        video window), symmetrically in both directions.
        """
        logits = (v @ m.t()) / max(1e-6, temperature)
        targets = torch.arange(v.shape[0], device=v.device)
        return 0.5 * (F.cross_entropy(logits, targets)
                      + F.cross_entropy(logits.t(), targets))

    class _ResidualBlock(nn.Module):
        """Centred dilated conv block -- same shape as the supervised localiser's."""

        def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
            super().__init__()
            padding = dilation * (kernel - 1) // 2
            self.conv1 = nn.Conv1d(channels, channels, kernel,
                                   padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel,
                                   padding=padding, dilation=dilation)
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            h = self.drop(F.gelu(self.norm1(self.conv1(x))))
            h = self.drop(F.gelu(self.norm2(self.conv2(h))))
            return x + h

    class CLIPSpotterModel(nn.Module):
        """Dual tower + fusion + temporal localisation head + transition head.

        The localisation head consumes the fused representation ``u`` concatenated
        with the explicit head-motion block (方案 §2: the confound is an *input*, not
        something regressed out -- the analytic subtraction measurably failed at 0.847
        scene share).
        """

        HEAD_FEATURES = 7  # z-scored (dx, dy, rot, scale, |t|, |rot|) + speed scalar

        def __init__(self, config: ClipConfig, n_slots: int = 16) -> None:
            super().__init__()
            self.config = config
            self.towers = MotionCLIP(config)
            dim = self.towers.embed_dim
            self.fusion = GatedFusion(dim)
            # 2026-09-03: formwork.md 第 III 条 -- additive local cross-attention on
            # top of the gated fusion (方案 第 2.4 节). ``None`` when disabled so
            # ``fuse()`` reproduces the pre-2026-09-03 behaviour exactly (no extra
            # parameters, no extra forward cost).
            self.cross_attention = (
                LocalCrossAttentionFusion(
                    dim, config.cross_attention_heads,
                    config.cross_attention_window_radius, config.cross_attention_dropout,
                ) if config.use_cross_attention else None
            )
            self.transition = TransitionHead(dim, n_slots)

            channels = config.head_channels
            self.stem = nn.Conv1d(dim + self.HEAD_FEATURES, channels, 1)
            self.blocks = nn.ModuleList([
                _ResidualBlock(channels, 3, d, config.head_dropout)
                for d in (1, 2, 4, 8)
            ])
            self.head = nn.Conv1d(channels, 1, 1)

        # -- forward pieces ------------------------------------------------------

        def fuse(self, v: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            """``u' = pool(Attn(t)) + u`` (方案 第 2.4 节) when cross-attention is on,
            else the plain gated fusion ``u`` (方案 第 1.4 节).
            """
            u = self.fusion(v, m)
            if self.cross_attention is not None:
                # LocalCrossAttentionFusion is written for (B, T, d); the callers
                # (window encoder, scoring path, inference spotter) all hand over
                # unbatched (T, d) embedding sequences, so promote to a batch of
                # one for the call and drop it again on the way out.
                u = u + self.cross_attention(v[None], m[None])[0]
            return u

        def localise(self, u: "torch.Tensor",
                     head_features: "torch.Tensor") -> "torch.Tensor":
            """``u`` (B, T, d) + head block (B, T, 7) -> per-frame logits (B, T)."""
            h = torch.cat([u, head_features], dim=-1).transpose(1, 2)
            h = self.stem(h)
            for block in self.blocks:
                h = block(h)
            return self.head(h).squeeze(1)

        # -- checkpointing (trainable-only, so the file stays tens of MB) --------

        def trainable_parameters(self) -> List["torch.nn.Parameter"]:
            return [p for p in self.parameters() if p.requires_grad]

        def trainable_state_dict(self) -> Dict[str, "torch.Tensor"]:
            required = {
                name for name, p in self.named_parameters() if p.requires_grad
            }
            return {name: tensor.detach().cpu().clone()
                    for name, tensor in self.state_dict().items()
                    if name in required}

        def load_trainable_state_dict(self, state: Dict[str, Any]) -> None:
            missing, unexpected = self.load_state_dict(state, strict=False)
            unexpected = list(unexpected)
            if unexpected:
                raise KeyError(f"checkpoint carries unknown tensors: {unexpected[:5]}")

else:  # pragma: no cover - torch missing

    class GatedFusion:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError("the CLIP motion engine needs PyTorch installed.")

    class LocalCrossAttentionFusion(GatedFusion):  # type: ignore[no-redef, misc]
        pass

    class TransitionHead(GatedFusion):  # type: ignore[no-redef, misc]
        pass

    class MotionCLIP(GatedFusion):  # type: ignore[no-redef, misc]
        pass

    class CLIPSpotterModel(GatedFusion):  # type: ignore[no-redef, misc]
        pass

    def info_nce(*_a, **_kw):  # type: ignore[no-redef]
        raise ImportError("the CLIP motion engine needs PyTorch installed.")


def head_feature_block(head_motion: Optional[np.ndarray], n_frames: int) -> np.ndarray:
    """The (T, 7) explicit head-motion input block, robust-z per video.

    Column 7 is the frame-to-frame head *speed* -- the scalar the contrastive
    negatives of 方案 §2.2 are mined from, exposed here so training and mining read
    the identical quantity.
    """
    if head_motion is None or np.asarray(head_motion).ndim != 2:
        return np.zeros((n_frames, CLIPSpotterModel.HEAD_FEATURES if _TORCH else 7),
                        dtype=np.float32)
    head = np.asarray(head_motion, dtype=np.float64)
    med = np.median(head, axis=0, keepdims=True)
    mad = np.median(np.abs(head - med), axis=0, keepdims=True)
    scale = np.where(1.4826 * mad < 1e-6, 1.0, 1.4826 * mad)
    head_z = (head - med) / scale
    velocity = np.diff(head_z, axis=0, prepend=head_z[:1])
    speed = np.linalg.norm(velocity, axis=1, keepdims=True)
    block = np.concatenate([head_z, speed], axis=1)[:, :7]
    if block.shape[1] < 7:
        block = np.pad(block, ((0, 0), (0, 7 - block.shape[1])))
    return np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def head_speed(head_motion: Optional[np.ndarray], n_frames: int) -> np.ndarray:
    """Per-frame head speed |head_v| -- the negative-mining scalar of 方案 §2.2."""
    return head_feature_block(head_motion, n_frames)[:, -1]


__all__ = [
    "ClipWeightsUnavailableError", "resolve_clip_weights", "GatedFusion",
    "LocalCrossAttentionFusion", "TransitionHead", "MotionCLIP", "CLIPSpotterModel",
    "info_nce", "head_feature_block", "head_speed",
]
