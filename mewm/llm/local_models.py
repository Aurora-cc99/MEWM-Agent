"""Local open-weight model loader: Qwen, GLM and other VLLM-served models."""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .registry import ModelSpec, resolve

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE = Path(
    os.environ.get("MEWM_MODEL_CACHE")
    or os.environ.get("HF_HOME")
    or os.environ.get("TRANSFORMERS_CACHE")
    or (Path.home() / ".cache" / "huggingface")
)


class WeightsUnavailableError(RuntimeError):
    pass


class LocalBackendError(RuntimeError):
    pass


def manual_download_instructions(spec: ModelSpec, reason: str = "") -> str:
    repo = spec.hf_repo or spec.model_id
    declared = declared_local_weights(spec)
    target = declared or (DEFAULT_CACHE / "hub" / f"models--{repo.replace('/', '--')}")
    return f"""
Could not obtain weights for {spec.model_id} ({repo}).
{('Reason: ' + reason) if reason else ''}

Fetch them manually by whichever route fits your network:

1. huggingface-cli (resumable, preferred)

     pip install -U "huggingface_hub[cli]"
     huggingface-cli download {repo} --local-dir "{target / repo.replace('/', '--')}"

   Behind a mirror (common in mainland China):

     export HF_ENDPOINT=https://hf-mirror.com
     huggingface-cli download {repo} --local-dir "{target / repo.replace('/', '--')}"

2. ModelScope (Qwen weights are mirrored there)

     pip install modelscope
     modelscope download --model {repo} --local_dir ./weights/{spec.model_id}

3. git-lfs

     git lfs install
     git clone https://huggingface.co/{repo} ./weights/{spec.model_id}

Then point the framework at the directory:

     export MEWM_LOCAL_WEIGHTS_{_env_suffix(spec)}=/absolute/path/to/weights

Common causes, in the order worth checking:
  * gated repo -- accept the licence on the model page, then
    `huggingface-cli login` with a token that has read access;
  * no network / proxy -- set HF_ENDPOINT to a mirror as above;
  * out of disk -- {spec.approx_vram_gb:.0f} GB of VRAM implies roughly the same again
    on disk; free space under {target};
  * transformers too old for this architecture -- `pip install -U transformers`.
""".strip()


def _env_suffix(spec: ModelSpec) -> str:
    return spec.model_id.upper().replace("-", "_").replace(".", "_")


def declared_local_weights(spec: ModelSpec) -> Optional[Path]:

    declared = getattr(spec, "local_weights", "") or ""
    if not declared:
        return None
    path = Path(declared)
    if not path.is_absolute():
        from ..config import PACKAGE_ROOT
        path = PACKAGE_ROOT / path
    return path


def local_weights_override(spec: ModelSpec) -> Optional[Path]:

    value = os.environ.get(f"MEWM_LOCAL_WEIGHTS_{_env_suffix(spec)}")
    if value:
        path = Path(value)
        if path.is_dir():
            return path
    return declared_local_weights(spec)


def ensure_weights(spec: ModelSpec, retries: int = 3) -> Path:

    override = local_weights_override(spec)
    if override is not None and (override / "config.json").is_file():
        return override
    if not spec.hf_repo:
        raise WeightsUnavailableError(manual_download_instructions(spec, "no repo id"))

    if override is not None:
        target = override
    else:
        target = DEFAULT_CACHE / "hub" / f"models--{spec.hf_repo.replace('/', '--')}"
        if (target / "config.json").is_file():
            return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise WeightsUnavailableError(manual_download_instructions(
            spec, f"huggingface_hub not installed ({exc}); "
                  "pip install -U huggingface_hub")) from exc

    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            LOGGER.info("downloading %s -> %s (attempt %d/%d, resumable)",
                        spec.hf_repo, target, attempt + 1, retries)
            snapshot_download(repo_id=spec.hf_repo, local_dir=str(target))
            return target
        except Exception as exc:
            last = exc
            wait = 5.0 * (2 ** attempt)
            LOGGER.warning("download attempt %d/%d for %s failed (%s); retrying in %.0fs",
                           attempt + 1, retries, spec.hf_repo, exc, wait)
            time.sleep(wait)
    raise WeightsUnavailableError(
        manual_download_instructions(spec, f"{type(last).__name__}: {last}"))


@dataclass
class LoadedModel:

    spec: ModelSpec
    model: Any
    processor: Any
    is_vision: bool
    device: str
    source: str


_CACHE: Dict[Tuple[str, str, str], LoadedModel] = {}


def _require_transformers():
    try:
        import torch
        import transformers
        return transformers
    except ImportError as exc:
        raise LocalBackendError(
            "open-weight models need torch and transformers:\n"
            "    pip install -U torch transformers accelerate\n"
            f"(import failed: {exc})"
        ) from exc


def preflight(spec: ModelSpec) -> Dict[str, Any]:
    report: Dict[str, Any] = {"model": spec.model_id, "repo": spec.hf_repo}
    report["declared_weights_dir"] = str(declared_local_weights(spec) or "")
    override = local_weights_override(spec)
    report["local_override"] = str(override) if override else None

    try:
        usage = shutil.disk_usage(DEFAULT_CACHE if DEFAULT_CACHE.exists()
                                  else Path.home())
        report["disk_free_gb"] = round(usage.free / 1e9, 1)
        report["disk_sufficient"] = usage.free / 1e9 > spec.approx_vram_gb
    except OSError:
        report["disk_free_gb"] = None
        report["disk_sufficient"] = None

    try:
        import torch
        if torch.cuda.is_available():
            total = sum(torch.cuda.get_device_properties(i).total_memory
                        for i in range(torch.cuda.device_count()))
            report["vram_gb"] = round(total / 1e9, 1)
            report["vram_sufficient"] = total / 1e9 >= spec.approx_vram_gb
            report["devices"] = torch.cuda.device_count()
        else:
            report["vram_gb"] = 0.0
            report["vram_sufficient"] = False
            report["devices"] = 0
    except ImportError:
        report["vram_gb"] = None
        report["vram_sufficient"] = None
    report["required_vram_gb"] = spec.approx_vram_gb
    report["thinking"] = (getattr(spec, "thinking", "") or "")
    report["thinking_budget"] = spec.thinking_budget
    return report


def load_local_model(
    model: str | ModelSpec,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    quantization: str = "",
) -> LoadedModel:
    spec = model if isinstance(model, ModelSpec) else resolve(model)
    if not spec.open_weights:
        raise LocalBackendError(f"{spec.model_id} is not an open-weight model")

    key = (spec.model_id, dtype, device_map)
    if key in _CACHE:
        return _CACHE[key]

    transformers = _require_transformers()
    import torch

    source = str(ensure_weights(spec))

    checks = preflight(spec)
    if checks.get("disk_sufficient") is False:
        LOGGER.warning("only %.1f GB free; %s needs roughly %.0f GB",
                       checks.get("disk_free_gb") or 0.0, spec.model_id,
                       spec.approx_vram_gb)
    if checks.get("vram_sufficient") is False:
        LOGGER.warning(
            "%s wants about %.0f GB of VRAM but %.1f GB is visible; loading with "
            "device_map=%r will offload to CPU and be very slow",
            spec.model_id, spec.approx_vram_gb, checks.get("vram_gb") or 0.0, device_map)

    from .quant import bnb_4bit_config, quantization_for
    effective_quant = quantization_for(
        spec, quantization, vram_sufficient=checks.get("vram_sufficient"))
    quant_kwargs = {}
    if effective_quant == "4bit":
        quant_kwargs["quantization_config"] = bnb_4bit_config()

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)
    started = time.time()
    LOGGER.info("loading %s from %s%s", spec.model_id, source,
                " (4-bit nf4)" if effective_quant == "4bit" else "")

    try:
        if spec.vision:
            processor = transformers.AutoProcessor.from_pretrained(
                source, trust_remote_code=trust_remote_code)
            loader = (getattr(transformers, "AutoModelForImageTextToText", None)
                      or getattr(transformers, "AutoModelForVision2Seq", None)
                      or transformers.AutoModelForCausalLM)
            materialised = loader.from_pretrained(
                source, torch_dtype=torch_dtype, device_map=device_map,
                trust_remote_code=trust_remote_code, **quant_kwargs)
        else:
            processor = transformers.AutoTokenizer.from_pretrained(
                source, trust_remote_code=trust_remote_code)
            materialised = transformers.AutoModelForCausalLM.from_pretrained(
                source, torch_dtype=torch_dtype, device_map=device_map,
                trust_remote_code=trust_remote_code, **quant_kwargs)
    except Exception as exc:
        raise WeightsUnavailableError(
            manual_download_instructions(spec, f"{type(exc).__name__}: {exc}")
        ) from exc

    materialised.eval()
    loaded = LoadedModel(
        spec=spec, model=materialised, processor=processor, is_vision=spec.vision,
        device=str(getattr(materialised, "device", "cuda" if torch.cuda.is_available() else "cpu")),
        source=source,
    )
    _CACHE[key] = loaded
    LOGGER.info("loaded %s in %.1fs", spec.model_id, time.time() - started)
    return loaded


def unload(model: Optional[str] = None) -> None:
    import gc
    keys = ([k for k in _CACHE if k[0] == resolve(model).model_id] if model
            else list(_CACHE))
    for key in keys:
        _CACHE.pop(key, None)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

THINKING_DIRECTIVE = (
    "Before answering, reason step by step inside a single "
    f"{THINK_OPEN} ... {THINK_CLOSE} block: restate what the evidence actually shows, "
    "work through the alternatives it is consistent with, and say which one it "
    f"favours and why. Then close the block with {THINK_CLOSE} and give the final "
    "answer alone, in exactly the format requested. Never put the answer inside the "
    "block, and never emit more than one block."
)

_NO_THINKING_KWARG: set = set()


def thinking_mode(spec: ModelSpec) -> str:
    mode = (getattr(spec, "thinking", "") or "").strip().lower()
    return mode if mode in {"template", "prompt", "native"} else ""


_NATIVE_THINK_MARKERS = (
    ("<|begin_of_thought|>", "<think>"),
    ("<|end_of_thought|>", "</think>"),
)


def split_reasoning(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""
    if THINK_OPEN not in text and any(marker in text
                                      for marker, _ in _NATIVE_THINK_MARKERS):
        for native, shared in _NATIVE_THINK_MARKERS:
            text = text.replace(native, shared)
    open_at = text.find(THINK_OPEN)
    close_at = text.find(THINK_CLOSE)
    if close_at == -1:
        if open_at == -1:
            return text.strip(), ""
        return "", text[open_at + len(THINK_OPEN):].strip()
    if open_at == -1 or open_at > close_at:
        reasoning = text[:close_at]
    else:
        reasoning = text[open_at + len(THINK_OPEN):close_at]
    answer = text[close_at + len(THINK_CLOSE):]
    return answer.strip(), reasoning.strip()


def _apply_chat_template(processor: Any, messages: List[Dict[str, Any]],
                         spec: ModelSpec, want_thinking: bool) -> str:
    if want_thinking and spec.model_id not in _NO_THINKING_KWARG:
        try:
            return processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
        except TypeError:
            _NO_THINKING_KWARG.add(spec.model_id)
            LOGGER.info("", spec.model_id)
        except Exception as exc:
            _NO_THINKING_KWARG.add(spec.model_id)
            LOGGER.warning("",
                           spec.model_id, exc)
    return processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def call_local_model(
    spec: ModelSpec,
    system_prompt: str,
    user_prompt: str,
    image_paths: Optional[Sequence[str | Path]] = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    thinking: Optional[bool] = None,
):
    from .client import LLMResponse

    started = time.time()
    loaded = load_local_model(spec, dtype=dtype, device_map=device_map)
    import torch

    mode = thinking_mode(spec)
    want_thinking = mode != "" if thinking is None else bool(thinking) and mode != ""

    images: List[Any] = []
    if image_paths and loaded.is_vision:
        from PIL import Image
        for path in image_paths:
            try:
                images.append(Image.open(str(path)).convert("RGB"))
            except Exception as exc:
                LOGGER.warning("skipping unreadable image %s: %s", path, exc)

    effective_system = system_prompt
    if want_thinking and (mode == "prompt" or spec.model_id in _NO_THINKING_KWARG):
        effective_system = (f"{THINKING_DIRECTIVE}\n\n{system_prompt}"
                            if system_prompt else THINKING_DIRECTIVE)

    content: List[Dict[str, Any]] = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": user_prompt})
    messages = ([{"role": "system", "content": effective_system}]
                if effective_system else [])
    messages.append({"role": "user", "content": content if images else user_prompt})

    processor = loaded.processor
    try:
        text = _apply_chat_template(processor, messages, spec,
                                    want_thinking and mode == "template")
    except Exception:
        text = (f"{effective_system}\n\n{user_prompt}" if effective_system
                else user_prompt)

    if images:
        inputs = processor(text=[text], images=images, return_tensors="pt")
    elif hasattr(processor, "__call__") and loaded.is_vision:
        inputs = processor(text=[text], return_tensors="pt")
    else:
        inputs = processor(text, return_tensors="pt")
    inputs = {k: (v.to(loaded.model.device) if hasattr(v, "to") else v)
              for k, v in inputs.items()}

    budget = max_tokens + (max(0, spec.thinking_budget) if want_thinking else 0)

    with torch.no_grad():
        generated = loaded.model.generate(
            **inputs, max_new_tokens=budget,
            do_sample=temperature > 0.0, temperature=max(1e-5, temperature),
        )

    prompt_length = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    trimmed = generated[0][prompt_length:]
    decoder = getattr(processor, "decode", None) or getattr(processor, "tokenizer").decode
    output = decoder(trimmed, skip_special_tokens=not want_thinking)
    if want_thinking:
        output = _strip_special(output, processor)

    answer, reasoning = (split_reasoning(output) if want_thinking
                         else (output.strip(), ""))
    if want_thinking and not answer and reasoning:
        LOGGER.warning(
            "", spec.model_id,
            len(reasoning))

    return LLMResponse(
        text=answer, model=spec.model_id, route=f"local:{loaded.source}",
        latency_s=time.time() - started, attempts=1, reasoning_effort=mode,
        served_model=spec.model_id, reasoning=reasoning,
    )


def _strip_special(text: str, processor: Any) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    specials = [str(t) for t in (getattr(tokenizer, "all_special_tokens", None) or [])]
    for token in sorted(specials, key=len, reverse=True):
        if token and token not in (THINK_OPEN, THINK_CLOSE):
            text = text.replace(token, "")
    return text


def local_status() -> Dict[str, Any]:
    from .registry import list_models
    report: Dict[str, Any] = {"cache_dir": str(DEFAULT_CACHE),
                              "loaded": [k[0] for k in _CACHE]}
    models = []
    for spec in list_models(open_only=True):
        entry = preflight(spec)
        entry["cached"] = _is_cached(spec)
        models.append(entry)
    report["models"] = models
    return report


def _is_cached(spec: ModelSpec) -> bool:
    if local_weights_override(spec):
        return True
    hub = DEFAULT_CACHE / "hub"
    if not hub.is_dir() or not spec.hf_repo:
        return False
    marker = f"models--{spec.hf_repo.replace('/', '--')}"
    return (hub / marker).is_dir()


__all__ = [
    "DEFAULT_CACHE", "WeightsUnavailableError", "LocalBackendError",
    "manual_download_instructions", "local_weights_override", "ensure_weights",
    "LoadedModel", "preflight", "load_local_model", "unload", "call_local_model",
    "local_status", "THINK_OPEN", "THINK_CLOSE", "THINKING_DIRECTIVE",
    "thinking_mode", "split_reasoning",
]
