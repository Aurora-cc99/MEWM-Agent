"""4-bit quantisation policy for open-weight models (修改方案 §4.3).

nf4 double quantisation with bfloat16 compute is what makes an 8B VLM trainable with
LoRA inside 8 GB of laptop VRAM (~6 GB resident). RTX 50-series (Blackwell) needs
bitsandbytes >= 0.45 -- older wheels miss the sm_120 kernels; the error raised here
says so instead of surfacing a CUDA kernel-image failure mid-load.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

MIN_BITSANDBYTES = (0, 45)


class QuantizationUnavailableError(RuntimeError):
    """bitsandbytes is missing or too old for this GPU generation."""


def bnb_4bit_config() -> Any:
    """The nf4 double-quantisation config of 方案 §4.3, dependency-checked."""
    try:
        import bitsandbytes
    except ImportError as exc:
        raise QuantizationUnavailableError(
            "4-bit loading needs bitsandbytes:\n"
            "    pip install -U 'bitsandbytes>=0.45'\n"
            f"(import failed: {exc})"
        ) from exc

    version = tuple(int(p) for p in bitsandbytes.__version__.split(".")[:2])
    if version < MIN_BITSANDBYTES:
        LOGGER.warning(
            "bitsandbytes %s found; RTX 50-series (Blackwell) needs >= %d.%d -- "
            "4-bit loads may fail with a kernel-image error",
            bitsandbytes.__version__, *MIN_BITSANDBYTES)

    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def quantization_for(spec: Any, requested: str = "",
                     vram_sufficient: Optional[bool] = None) -> str:
    """Which quantisation to apply for one load.

    Explicit request wins; otherwise the registry's per-model field applies whenever
    the visible VRAM cannot hold the full-precision model (方案 §4.3: automatic 4-bit
    with a warning, never a silent hosted fallback).
    """
    if requested and requested != "none":
        return requested
    if requested == "none":
        return ""
    if getattr(spec, "quantization", "") and vram_sufficient is False:
        LOGGER.warning(
            "%s does not fit the visible VRAM at full precision; loading %s as "
            "configured in the registry", spec.model_id, spec.quantization)
        return spec.quantization
    return ""


__all__ = ["QuantizationUnavailableError", "bnb_4bit_config", "quantization_for",
           "MIN_BITSANDBYTES"]
