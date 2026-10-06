"""Quantisation helpers: 4-bit / 8-bit loading wrappers for local models."""
from __future__ import annotations

import logging
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

MIN_BITSANDBYTES = (0, 45)


class QuantizationUnavailableError(RuntimeError):
    pass


def bnb_4bit_config() -> Any:
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
            "",
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
    if requested and requested != "none":
        return requested
    if requested == "none":
        return ""
    if getattr(spec, "quantization", "") and vram_sufficient is False:
        LOGGER.warning(
            "", spec.model_id, spec.quantization)
        return spec.quantization
    return ""


__all__ = ["QuantizationUnavailableError", "bnb_4bit_config", "quantization_for",
           "MIN_BITSANDBYTES"]
