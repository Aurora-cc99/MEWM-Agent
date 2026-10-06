"""Model registry: maps config keys to provider credentials and generation params."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


TRANSPORT_ANTHROPIC = "anthropic"
TRANSPORT_OPENAI = "openai"
TRANSPORT_GEMINI = "gemini"
TRANSPORT_LOCAL = "local"

TRANSPORTS = (TRANSPORT_ANTHROPIC, TRANSPORT_OPENAI, TRANSPORT_GEMINI, TRANSPORT_LOCAL)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    transport: str
    base_env: Tuple[str, ...] = ()
    key_env: Tuple[str, ...] = ()
    default_base: str = ""
    user_agent: str = ""

    def base_url(self) -> str:
        for var in self.base_env:
            value = os.environ.get(var)
            if value:
                return value.rstrip("/")
        return self.default_base.rstrip("/")

    def api_key(self) -> str:
        for var in self.key_env:
            value = os.environ.get(var)
            if value:
                return value
        return ""

    def configured(self) -> bool:
        if self.transport == TRANSPORT_LOCAL:
            return True
        return bool(self.base_url() and self.api_key())


PROVIDERS: Dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        "anthropic", TRANSPORT_ANTHROPIC,
        base_env=("ANTHROPIC_BASE_URL", "ANTHROPIC_API_URL"),
        key_env=("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"),
        user_agent="......",
    ),
    "openai": ProviderSpec(
        "openai", TRANSPORT_OPENAI,
        base_env=("OPENAI_API_URL", "OPENAI_BASE_URL"),
        key_env=("OPENAI_API_KEY",),
        user_agent="......",
    ),
    "deepseek": ProviderSpec(
        "deepseek", TRANSPORT_ANTHROPIC,
        base_env=("DEEPSEEK_BASE_URL",), key_env=("DEEPSEEK_API_KEY",),
        default_base="https://api.deepseek.com/anthropic",
    ),
    "gemini": ProviderSpec(
        "gemini", TRANSPORT_GEMINI,
        base_env=("GEMINI_BASE_URL",), key_env=("GEMINI_API_KEY",),
        user_agent="......",
    ),
    "grok": ProviderSpec(
        "grok", TRANSPORT_ANTHROPIC,
        base_env=("GROK_BASE_URL",), key_env=("GROK_API_KEY",),
        user_agent=".......",
    ),
    "local": ProviderSpec("local", TRANSPORT_LOCAL),
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    provider: str
    display: str = ""
    aliases: Tuple[str, ...] = ()
    efforts: Tuple[str, ...] = ()
    default_effort: str = ""
    vision: bool = True
    open_weights: bool = False
    hf_repo: str = ""
    local_weights: str = ""
    quantization: str = ""
    approx_vram_gb: float = 0.0
    thinking: str = ""
    thinking_budget: int = 2048
    context: int = 128_000
    min_retries: int = 0
    notes: str = ""

    @property
    def transport(self) -> str:
        return PROVIDERS[self.provider].transport

    @property
    def is_local(self) -> bool:
        return self.provider == "local"

    def validate_effort(self, effort: str) -> str:
        wanted = (effort or "").strip().lower().replace(" ", "").replace("-", "")
        aliases = {"extrahigh": "xhigh", "maximum": "max", "med": "medium", "": ""}
        wanted = aliases.get(wanted, wanted)
        if not wanted:
            return self.default_effort
        if wanted == "none":
            return ""
        if self.efforts and wanted not in self.efforts:
            raise ValueError(
                f"reasoning effort {effort!r} is not selectable for {self.model_id}; "
                f"choose one of: {', '.join(self.efforts)}"
            )
        return wanted

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id, "provider": self.provider,
            "transport": self.transport, "display": self.display or self.model_id,
            "efforts": list(self.efforts), "default_effort": self.default_effort,
            "vision": self.vision, "open_weights": self.open_weights,
            "hf_repo": self.hf_repo, "quantization": self.quantization,
            "thinking": self.thinking, "thinking_budget": self.thinking_budget,
            "context": self.context, "notes": self.notes,
        }


EFFORT_THINKING_BUDGET: Dict[str, int] = {
    "low": 4096, "medium": 8192, "high": 16384, "xhigh": 32768, "max": 49152,
}


_MODEL_LIST: Tuple[ModelSpec, ...] = (
    ModelSpec(
        "", "anthropic", "Claude Sonnet 5",
        efforts=("high", "xhigh", "max"), default_effort="xhigh", context=200_000,
    ),
    ModelSpec(
        "", "anthropic", "Claude Sonnet 4",
        aliases=("", "claude-4-sonnet", "sonnet-4"),
        context=200_000,
        notes="......",
    ),
    ModelSpec(
        "[REDACTED]", "anthropic", "Claude Sonnet 4.5 (2025-09-29)",
        aliases=("[REDACTED].5", "claude-4.5-sonnet", "[REDACTED]",
                 "sonnet-4.5"),
        efforts=("high", "xhigh", "max"), default_effort="high", context=200_000,
        notes="",
    ),
    ModelSpec(
        "deepseek-v4-flash-vision-exp", "deepseek", "DeepSeek V4 Flash Vision (exp)",
        aliases=("deepseekV4-Flash-Vision-Exp", "deepseek-v4-flash-vision",
                 "deepseek-vision"),
        context=128_000, min_retries=6,
        notes="Anthropic-compatible relay at api.deepseek.com/anthropic. Text is "
              "reliable. VISION is intermittent: measured 2025-08, single image "
              "requests return an empty completion most of the time (2/6 and 0/18 in "
              "two bursts against the raw endpoint, so it is not a client bug). The "
              "min_retries=6 floor does usually recover it, at ~13 s per call versus "
              "~5 s elsewhere. Fine for R/C text roles; prefer claude / grok / gemini "
              "/ gpt for frame-reading phases. Check with: mewm test-models --vision",
    ),
    ModelSpec(
        "deepseek-v4-pro", "deepseek", "DeepSeek V4 Pro", vision=False, context=128_000,
    ),
    ModelSpec(
        "deepseek-v4-flash", "deepseek", "DeepSeek V4 Flash", vision=False,
        context=128_000,
    ),
    ModelSpec(
        "grok-4.5", "grok", "Grok 4.5", aliases=("grok-4-5", "grok4.5"),
        context=256_000,
    ),
    ModelSpec(
        "gpt-4o", "openai", "GPT-4o",
        aliases=("gpt4o",),
        efforts=("low", "medium", "high"), default_effort="medium",
        context=128_000,
    ),
    ModelSpec(
        "gpt-4o-mini", "openai", "GPT-4o mini",
        aliases=("gpt4o-mini", "gpt-4o-mini"),
        efforts=("low", "medium", "high"), default_effort="low",
        context=128_000,
    ),
    ModelSpec(
        "gpt-5.6-sol", "openai", "GPT-5.6-sol",
        efforts=("high", "xhigh"), default_effort="high", context=200_000,
    ),
    ModelSpec(
        "gpt-5.4", "openai", "GPT-5.4",
        efforts=("low", "medium", "high", "xhigh"), default_effort="high",
        context=200_000,
    ),
    ModelSpec(
        "gpt-5.4-mini", "openai", "GPT-5.4 mini",
        aliases=("gpt-5.4mini", "gpt5.4-mini"),
        efforts=("low", "medium", "high", "xhigh"), default_effort="medium",
        context=200_000,
    ),
    ModelSpec(
        "gemini-3-pro", "gemini", "Gemini 3 Pro",
        aliases=("gemini-3-pro-preview", "gemini-3pro"),
        context=1_000_000,
        notes="Native generateContent. The -preview id 500s upstream; aliased here.",
    ),
    ModelSpec("gemini-3.1-pro", "gemini", "Gemini 3.1 Pro", context=1_000_000),
    ModelSpec("gemini-3.1-pro-preview", "gemini", "Gemini 3.1 Pro (preview)",
              context=1_000_000),
    ModelSpec("gemini-2.5-pro", "gemini", "Gemini 2.5 Pro", context=1_000_000),
    ModelSpec("gemini-2.5-flash", "gemini", "Gemini 2.5 Flash", context=1_000_000),
    ModelSpec("gemini-3-flash", "gemini", "Gemini 3 Flash", context=1_000_000),
    ModelSpec(
        "Qwen3-VL-8B", "local", "Qwen3-VL 8B", open_weights=True,
        hf_repo="Qwen/Qwen3-VL-8B-Instruct", quantization="4bit",
        approx_vram_gb=18.0, context=128_000,
        aliases=("qwen3-vl-8b",),
        local_weights="Weights/Qwen3-VL-8B-Instruct",
        thinking="template", thinking_budget=2048,
    ),
    ModelSpec(
        "Qwen3-VL-30B", "local", "Qwen3-VL 30B", open_weights=True,
        hf_repo="Qwen/Qwen3-VL-30B-Instruct", quantization="4bit",
        approx_vram_gb=66.0, context=128_000,
        aliases=("qwen3-vl-30b",),
        local_weights="Weights/Qwen3-VL-30B-Instruct",
        thinking="template", thinking_budget=2048,
    ),
    ModelSpec(
        "Qwen2.5-VL-7B", "local", "Qwen2.5-VL 7B", open_weights=True,
        hf_repo="Qwen/Qwen2.5-VL-7B-Instruct", quantization="4bit",
        approx_vram_gb=18.0, context=128_000,
        aliases=("qwen2.5-vl-7b",),
        local_weights="Weights/Qwen2.5-VL-7B-Instruct",
        thinking="prompt", thinking_budget=2048,
    ),
    ModelSpec(
        "Qwen2.5-VL-32B", "local", "Qwen2.5-VL 32B", open_weights=True,
        hf_repo="Qwen/Qwen2.5-VL-32B-Instruct", quantization="4bit",
        approx_vram_gb=66.0, context=128_000,
        aliases=("qwen2.5-vl-32b",),
        local_weights="Weights/Qwen2.5-VL-32B-Instruct",
        thinking="prompt", thinking_budget=2048,
    ),
    ModelSpec(
        "Qwen2.5-Omni-7B", "local", "Qwen2.5-Omni 7B", open_weights=True,
        hf_repo="Qwen/Qwen2.5-Omni-7B", quantization="4bit",
        approx_vram_gb=18.0, context=128_000,
        aliases=("qwen2.5-omni-7b",),
        local_weights="Weights/Qwen2.5-Omni-7B",
        thinking="prompt", thinking_budget=2048,
    ),
    ModelSpec(
        "GLM-4.1V-9B-Thinking", "local", "GLM-4.1V 9B Thinking", open_weights=True,
        hf_repo="zai-org/GLM-4.1V-9B-Thinking", quantization="4bit",
        approx_vram_gb=18.0, context=200_000,
        aliases=("glm-4.1v-9b", "glm4.1v-9b-thinking"),
        local_weights="Weights/GLM-4.1V-9B-Thinking",
        thinking="native", thinking_budget=4096,
    ),
    ModelSpec(
        "Gemma-4-31B", "local", "Gemma 4 31B", open_weights=True,
        hf_repo="google/gemma-4-31b-it", quantization="4bit",
        approx_vram_gb=66.0, context=128_000,
        aliases=("gemma-4-31b", "gemma4-31b"),
        local_weights="Weights/gemma-4-31b-it",
        thinking="prompt", thinking_budget=2048,
    ),
)

MODELS: Dict[str, ModelSpec] = {spec.model_id: spec for spec in _MODEL_LIST}

_ALIASES: Dict[str, str] = {}
for _spec in _MODEL_LIST:
    _ALIASES[_spec.model_id.lower()] = _spec.model_id
    for _alias in _spec.aliases:
        _ALIASES[_alias.lower()] = _spec.model_id


class UnknownModelError(KeyError):
    pass


def resolve(name: str) -> ModelSpec:
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return MODELS[_ALIASES[key]]
    raise UnknownModelError(
        f"unknown model {name!r}. Registered: {', '.join(sorted(MODELS))}"
    )


def resolve_id(name: str) -> str:
    return resolve(name).model_id


def is_registered(name: str) -> bool:
    return (name or "").strip().lower() in _ALIASES


def provider_of(name: str) -> ProviderSpec:
    return PROVIDERS[resolve(name).provider]


def transport_of(name: str) -> str:
    return resolve(name).transport


def list_models(
    open_only: bool = False, hosted_only: bool = False, vision_only: bool = False,
) -> List[ModelSpec]:
    out = list(_MODEL_LIST)
    if open_only:
        out = [s for s in out if s.open_weights]
    if hosted_only:
        out = [s for s in out if not s.open_weights]
    if vision_only:
        out = [s for s in out if s.vision]
    return out


def describe(name: str) -> str:
    spec = resolve(name)
    provider = PROVIDERS[spec.provider]
    if spec.is_local:
        return (f"{spec.model_id} -> local transformers ({spec.hf_repo}, "
                f"~{spec.approx_vram_gb:.0f} GB VRAM)")
    base = provider.base_url() or "<base not set>"
    if spec.transport == TRANSPORT_ANTHROPIC:
        route = f"{base}/v1/messages"
    elif spec.transport == TRANSPORT_GEMINI:
        route = f"{base}/v1beta/models/{spec.model_id}:generateContent"
    else:
        route = f"{base}/responses"
    return f"{spec.model_id} -> {route} ({spec.transport})"


def credentials_available(name: str) -> bool:
    try:
        return provider_of(name).configured()
    except UnknownModelError:
        return False


def heterogeneous(model_a: str, model_b: str) -> bool:
    try:
        first, second = resolve(model_a), resolve(model_b)
    except UnknownModelError:
        return model_a != model_b
    if first.model_id == second.model_id:
        return False
    return first.provider != second.provider


def registry_report() -> Dict[str, object]:
    return {
        "hosted": [s.to_dict() for s in list_models(hosted_only=True)],
        "open_weights": [s.to_dict() for s in list_models(open_only=True)],
        "providers": {
            name: {
                "transport": p.transport,
                "base": p.base_url() or "<unset>",
                "configured": p.configured(),
            }
            for name, p in PROVIDERS.items()
        },
    }


__all__ = [
    "TRANSPORT_ANTHROPIC", "TRANSPORT_OPENAI", "TRANSPORT_GEMINI", "TRANSPORT_LOCAL",
    "TRANSPORTS", "ProviderSpec", "PROVIDERS", "ModelSpec", "EFFORT_THINKING_BUDGET",
    "MODELS", "UnknownModelError", "resolve", "resolve_id", "is_registered",
    "provider_of", "transport_of", "list_models", "describe", "credentials_available",
    "heterogeneous", "registry_report",
]
