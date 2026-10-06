"""Clip-level motion engine: batched feature extraction over video clips."""

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
except ImportError:
    torch = None
    nn = object
    F = None
    _TORCH = False

class ClipWeightsUnavailableError(RuntimeError):
    pass

def _require_transformers():
    try:
        import transformers
        return transformers
    except ImportError as exc:
        raise ImportError(
            "the CLIP motion engine needs transformers:\n"
            "    pip install -U transformers\n"
            f"(import failed: {exc})"
        ) from exc

def resolve_clip_weights(config: ClipConfig) -> Path:
    path = Path(config.weights_path)
    if not path.is_absolute():
        from ..config import PACKAGE_ROOT
        path = PACKAGE_ROOT / path
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ClipWeightsUnavailableError(
            f""
        )
    return path

if _TORCH:

    class GatedFusion(nn.Module):
        pass

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.gate = nn.Linear(2 * dim, dim)

        def forward(self, v: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            g = torch.sigmoid(self.gate(torch.cat([v, m], dim=-1)))
            return g * v + (1.0 - g) * m

    class LocalCrossAttentionFusion(nn.Module):
        pass

        def __init__(self, dim: int, n_heads: int = 4, window_radius: int = 1,
                    dropout: float = 0.1) -> None:
            super().__init__()
            self.window_radius = max(0, int(window_radius))
            self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout,
                                                   batch_first=True)
            self.norm = nn.LayerNorm(dim)

        def _band_mask(self, length: int, device: "torch.device") -> "torch.Tensor":
            index = torch.arange(length, device=device)
            distance = (index.unsqueeze(0) - index.unsqueeze(1)).abs()
            return distance > self.window_radius

        def forward(self, v: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            length = v.shape[1]
            mask = self._band_mask(length, v.device) if length > 1 else None
            attended, _ = self.attention(v, m, m, attn_mask=mask)
            return self.norm(v + attended)

    class TransitionHead(nn.Module):
        pass

        def __init__(self, dim: int, n_slots: int, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, n_slots))

        def forward(self, u: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.net(u))

    class MotionCLIP(nn.Module):
        pass

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
            pass
            def _count(layers) -> int:
                return sum(
                    1 for layer in layers
                    if any(p.requires_grad for p in layer.parameters())
                )
            return {
                "vision": _count(self.clip.vision_model.encoder.layers),
                "text": _count(self.clip.text_model.encoder.layers),
            }

        def encode_images(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
            features = self.clip.get_image_features(pixel_values=pixel_values)
            return F.normalize(features, dim=-1)

        def encode_texts(self, input_ids: "torch.Tensor",
                         attention_mask: "torch.Tensor") -> "torch.Tensor":
            features = self.clip.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask)
            return F.normalize(features, dim=-1)

        def preprocess_images(self, image_paths: Sequence[str | Path]) -> "torch.Tensor":
            from PIL import Image
            images = [Image.open(str(p)).convert("RGB") for p in image_paths]
            batch = self.processor(images=images, return_tensors="pt")
            return batch["pixel_values"]

        def tokenize(self, texts: Sequence[str]) -> Dict[str, "torch.Tensor"]:
            return self.processor.tokenizer(
                list(texts), padding=True, truncation=True, max_length=77,
                return_tensors="pt")

    def info_nce(v: "torch.Tensor", m: "torch.Tensor",
                 temperature: float = 0.07) -> "torch.Tensor":
        logits = (v @ m.t()) / max(1e-6, temperature)
        targets = torch.arange(v.shape[0], device=v.device)
        return 0.5 * (F.cross_entropy(logits, targets)
                      + F.cross_entropy(logits.t(), targets))

    class _ResidualBlock(nn.Module):
        pass

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

        HEAD_FEATURES = 7

        def __init__(self, config: ClipConfig, n_slots: int = 16) -> None:
            super().__init__()
            self.config = config
            self.towers = MotionCLIP(config)
            dim = self.towers.embed_dim
            self.fusion = GatedFusion(dim)
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

        def fuse(self, v: "torch.Tensor", m: "torch.Tensor") -> "torch.Tensor":
            u = self.fusion(v, m)
            if self.cross_attention is not None:
                u = u + self.cross_attention(v[None], m[None])[0]
            return u

        def localise(self, u: "torch.Tensor",
                     head_features: "torch.Tensor") -> "torch.Tensor":
            h = torch.cat([u, head_features], dim=-1).transpose(1, 2)
            h = self.stem(h)
            for block in self.blocks:
                h = block(h)
            return self.head(h).squeeze(1)

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

else:

    class GatedFusion:
        pass
        def __init__(self, *a, **kw):
            raise ImportError("the CLIP motion engine needs PyTorch installed.")

    class LocalCrossAttentionFusion(GatedFusion):
        pass

    class TransitionHead(GatedFusion):
        pass

    class MotionCLIP(GatedFusion):
        pass

    class CLIPSpotterModel(GatedFusion):
        pass

    def info_nce(*_a, **_kw):
        raise ImportError("the CLIP motion engine needs PyTorch installed.")

def head_feature_block(head_motion: Optional[np.ndarray], n_frames: int) -> np.ndarray:
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
    return head_feature_block(head_motion, n_frames)[:, -1]

__all__ = [
    "ClipWeightsUnavailableError", "resolve_clip_weights", "GatedFusion",
    "LocalCrossAttentionFusion", "TransitionHead", "MotionCLIP", "CLIPSpotterModel",
    "info_nce", "head_feature_block", "head_speed",
]
