"""Model registry: which models exist, how each is reached, and what it can do.

* **Hosted** models reached over an API. Three transport shapes are in play and they are
  *not* interchangeable: Anthropic Messages (Claude, DeepSeek, Grok), OpenAI
  responses/chat (GPT), and native Gemini ``generateContent``. Sending a model to the
  wrong shape either 404s or -- worse -- silently answers with a different model.
* **Open-weight** models run locally through transformers, downloaded on first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Transports and providers
# ---------------------------------------------------------------------------

TRANSPORT_ANTHROPIC = "anthropic"    # POST {base}/v1/messages, SSE
TRANSPORT_OPENAI = "openai"          # POST {base}/responses only -- see client.py docstring
TRANSPORT_GEMINI = "gemini"          # POST {base}/v1beta/models/{id}:generateContent
TRANSPORT_LOCAL = "local"            # in-process transformers

TRANSPORTS = (TRANSPORT_ANTHROPIC, TRANSPORT_OPENAI, TRANSPORT_GEMINI, TRANSPORT_LOCAL)


@dataclass(frozen=True)
class ProviderSpec:
    """Where a provider's endpoint and credential come from."""

    name: str
    transport: str
    base_env: Tuple[str, ...] = ()
    key_env: Tuple[str, ...] = ()
    default_base: str = ""
    #: Cloudflare in front of the newcli relays rejects unknown agents with 403/1010.
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
        user_agent="claude-cli/2.1.235",
    ),
    "openai": ProviderSpec(
        "openai", TRANSPORT_OPENAI,
        base_env=("OPENAI_API_URL", "OPENAI_BASE_URL"),
        key_env=("OPENAI_API_KEY",),
        # The codex relay sits behind the same Cloudflare rule as the others and
        # answers an unrecognised agent with 403 / error 1010.
        user_agent="claude-cli/2.1.235",
    ),
    "deepseek": ProviderSpec(
        "deepseek", TRANSPORT_ANTHROPIC,
        base_env=("DEEPSEEK_BASE_URL",), key_env=("DEEPSEEK_API_KEY",),
        default_base="https://api.deepseek.com/anthropic",
    ),
    "gemini": ProviderSpec(
        "gemini", TRANSPORT_GEMINI,
        base_env=("GEMINI_BASE_URL",), key_env=("GEMINI_API_KEY",),
        user_agent="claude-cli/2.1.235",
    ),
    "grok": ProviderSpec(
        "grok", TRANSPORT_ANTHROPIC,
        base_env=("GROK_BASE_URL",), key_env=("GROK_API_KEY",),
        user_agent="claude-cli/2.1.235",
    ),
    "local": ProviderSpec("local", TRANSPORT_LOCAL),
}


# ---------------------------------------------------------------------------
# Model specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One selectable model."""

    model_id: str                       # the id the endpoint actually serves
    provider: str
    display: str = ""
    aliases: Tuple[str, ...] = ()
    efforts: Tuple[str, ...] = ()       # selectable reasoning tiers, "" = none
    default_effort: str = ""
    vision: bool = True
    open_weights: bool = False
    hf_repo: str = ""                   # HuggingFace repo for open-weight models
    #: "4bit" = nf4 double quantisation via bitsandbytes (方案 §4.3); "" = full dtype.
    #: Applied automatically when the visible VRAM cannot hold the full-precision model.
    quantization: str = ""
    approx_vram_gb: float = 0.0
    #: Open-weight only. How this model is made to think before answering:
    #:   "template" -- its chat template takes ``enable_thinking`` and emits the
    #:                 reasoning inside <think>...</think> (the Qwen3 family);
    #:   "prompt"   -- no such switch, so a chain-of-thought directive is prepended to
    #:                 the system prompt and the model is asked for the same delimiters
    #:                 (Gemma and anything else without a thinking mode);
    #:   ""         -- do not elicit a chain of thought.
    #: :func:`mewm.llm.local_models.call_local_model` reads this; the parsed CoT comes
    #: back on ``LLMResponse.reasoning``, separated from the answer text.
    thinking: str = ""
    #: Tokens reserved for the chain of thought, added on top of the caller's answer
    #: budget so a long CoT cannot truncate the answer that follows it.
    thinking_budget: int = 2048
    context: int = 128_000
    #: Floor on retry attempts for endpoints with a measured flakiness problem; the
    #: caller's own retry count wins when it is higher.
    min_retries: int = 0
    notes: str = ""

    @property
    def transport(self) -> str:
        return PROVIDERS[self.provider].transport

    @property
    def is_local(self) -> bool:
        return self.provider == "local"

    def validate_effort(self, effort: str) -> str:
        """Normalise a requested tier, or raise listing what this model accepts."""
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


#: Thinking-token budget per tier on the Anthropic transport (the Messages API takes a
#: budget, not a tier name; the tier travels alongside it).
EFFORT_THINKING_BUDGET: Dict[str, int] = {
    "low": 4096, "medium": 8192, "high": 16384, "xhigh": 32768, "max": 49152,
}


_MODEL_LIST: Tuple[ModelSpec, ...] = (
    # -- hosted, Anthropic Messages transport --------------------------------
    ModelSpec(
        "claude-sonnet-5", "anthropic", "Claude Sonnet 5",
        efforts=("high", "xhigh", "max"), default_effort="xhigh", context=200_000,
    ),
    ModelSpec(
        "claude-sonnet-4-20250514", "anthropic", "Claude Sonnet 4 (2025-05-14)",
        aliases=("claude-sonnet-4", "claude-4-sonnet", "sonnet-4"),
        context=200_000,
        notes="Same relay and credential as claude-sonnet-5; only the served id "
              "differs. No reasoning tier is declared, so the request stays plain. "
              "NOT REACHABLE on the current account, measured 2026-08-29: the relay "
              "LISTS the id in GET /v1/models but POST /v1/messages answers HTTP 404 "
              "'模型 claude-sonnet-4-20250514 未开放'. That is a per-account "
              "entitlement, not a client bug -- claude-sonnet-4-5-20250929 over the "
              "identical request succeeds, and the grok relay rejects the id with "
              "'unknown provider'. The entry is kept so the model works the moment "
              "the account is entitled; nothing else needs changing. Re-check with: "
              "mewm test-models --models claude-sonnet-4-20250514",
    ),
    ModelSpec(
        "claude-sonnet-4-5-20250929", "anthropic", "Claude Sonnet 4.5 (2025-09-29)",
        aliases=("claude-sonnet-4.5", "claude-4.5-sonnet", "claude-sonnet-4-5",
                 "sonnet-4.5"),
        efforts=("high", "xhigh", "max"), default_effort="high", context=200_000,
        notes="Same relay and credential as claude-sonnet-5. Registered as the "
              "working stand-in for claude-sonnet-4-20250514, which 404s on this "
              "account -- see that entry above; this id answers the identical "
              "request (measured 2026-08-29, in that entry's own notes). "
              "default_effort is pinned to \"high\" rather than \"xhigh\" on user "
              "directive (2026-08-30), to cut the per-call latency the xhigh tier "
              "costs on claude-sonnet-5; pass --reasoning-effort/--critic-effort "
              "explicitly to override. Re-check with: mewm test-models --models "
              "claude-sonnet-4-5-20250929",
    ),
    ModelSpec(
        "deepseek-v4-flash-vision-exp", "deepseek", "DeepSeek V4 Flash Vision (exp)",
        # The relay rejects the mixed-case spelling outright, so it is aliased here.
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

    # -- hosted, OpenAI transport --------------------------------------------
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

    # -- hosted, native Gemini transport -------------------------------------
    ModelSpec(
        "gemini-3-pro", "gemini", "Gemini 3 Pro",
        # gemini-3-pro-preview is listed by the relay but every call to it returns
        # upstream HTTP 500, so the alias points at the id that actually serves.
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

    # -- open weights, run locally -------------------------------------------
    # Every entry declares a `thinking` mode. Without it these models answer straight
    # from the prompt, and the SFT / GRPO traces they produce carry no chain of
    # thought -- which the reward's causal term is scored on, so a non-thinking local
    # policy is not comparable with the hosted ones it is benchmarked against.
    ModelSpec(
        "Qwen3-VL-8B", "local", "Qwen3-VL 8B", open_weights=True,
        hf_repo="Qwen/Qwen3-VL-8B-Instruct", quantization="4bit",
        approx_vram_gb=18.0, context=128_000,
        aliases=("qwen3-vl-8b",),
        thinking="template", thinking_budget=2048,
        notes="Qwen3 chat template: enable_thinking=True emits <think>...</think> "
              "before the answer.",
    ),
    ModelSpec(
        "Qwen3.6-VL-27B", "local", "Qwen3.6-VL 27B", open_weights=True,
        hf_repo="Qwen/Qwen3.6-VL-27B-Instruct", quantization="4bit",
        approx_vram_gb=58.0, context=128_000,
        aliases=("qwen3.6-vl-27b",),
        thinking="template", thinking_budget=3072,
    ),
    ModelSpec(
        "Qwen3.8-27B", "local", "Qwen3.8 27B", open_weights=True, vision=False,
        hf_repo="Qwen/Qwen3.8-27B-Instruct", quantization="4bit",
        approx_vram_gb=58.0, context=128_000,
        aliases=("qwen3.8-27b",),
        thinking="template", thinking_budget=3072,
        notes="Text-only: usable for R/C, not for phases that read frames.",
    ),
    ModelSpec(
        "Gemma-4-31B", "local", "Gemma 4 31B", open_weights=True,
        hf_repo="google/gemma-4-31b-it", quantization="4bit",
        approx_vram_gb=66.0, context=128_000,
        aliases=("gemma-4-31b", "gemma4-31b"),
        thinking="prompt", thinking_budget=2048,
        notes="No enable_thinking switch in the Gemma template, so the chain of "
              "thought is elicited by a system directive and parsed out of the same "
              "<think> delimiters.",
    ),
)

MODELS: Dict[str, ModelSpec] = {spec.model_id: spec for spec in _MODEL_LIST}

_ALIASES: Dict[str, str] = {}
for _spec in _MODEL_LIST:
    _ALIASES[_spec.model_id.lower()] = _spec.model_id
    for _alias in _spec.aliases:
        _ALIASES[_alias.lower()] = _spec.model_id


class UnknownModelError(KeyError):
    """Raised for a model id that is not registered."""


def resolve(name: str) -> ModelSpec:
    """Map any accepted spelling onto its :class:`ModelSpec`."""
    key = (name or "").strip().lower()
    if key in _ALIASES:
        return MODELS[_ALIASES[key]]
    raise UnknownModelError(
        f"unknown model {name!r}. Registered: {', '.join(sorted(MODELS))}"
    )


def resolve_id(name: str) -> str:
    """The id the endpoint actually serves, for a possibly-aliased name."""
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
    """Whether two models are genuinely different bases.

    Appendix D.1 requires the critic's base to differ from the reasoner's. Two ids from
    the same provider *family* (two Gemini tiers, say) share training and therefore share
    blind spots, so provider identity -- not just id inequality -- is what is checked.
    """
    try:
        first, second = resolve(model_a), resolve(model_b)
    except UnknownModelError:
        return model_a != model_b
    if first.model_id == second.model_id:
        return False
    return first.provider != second.provider


def registry_report() -> Dict[str, object]:
    """Promptable / printable summary of the whole registry."""
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
