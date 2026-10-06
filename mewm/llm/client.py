"""Unified LLM client: dispatches to API-hosted or local open-weight backends."""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .registry import (
    EFFORT_THINKING_BUDGET, TRANSPORT_ANTHROPIC, TRANSPORT_GEMINI, TRANSPORT_LOCAL,
    TRANSPORT_OPENAI, ModelSpec, UnknownModelError, describe, is_registered,
    provider_of, resolve,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_GPT_MODEL = "gpt-5.6-sol"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_TIMEOUT = 600
MAX_IMAGE_EDGE = 1024


class LLMError(RuntimeError):
    pass


class ModelMismatchError(LLMError):
    pass


class BackendMismatch(LLMError):
    pass


BACKEND_HOSTED = "hosted"
BACKEND_LOCAL = "local"


def backend_for(model: str, backend: str) -> ModelSpec:
    spec = resolve(model)
    if backend == BACKEND_LOCAL:
        if not spec.open_weights:
            raise BackendMismatch(
                f"{spec.model_id} is a hosted model and cannot run on the local "
                f"")
        return spec
    if backend == BACKEND_HOSTED:
        if spec.open_weights:
            raise BackendMismatch(
                f"{spec.model_id} is an open-weight model and cannot run on the "
                f"hosted backend; set this role's backend to 'local'")
        return spec
    raise BackendMismatch(
        f"unknown backend {backend!r} for {spec.model_id}: expected 'hosted' or "
        f"'local'")


_BACKEND_MANIFEST: List[Dict[str, Any]] = []


def record_backend_event(role: str, model: str, backend: str, route: str,
                         latency_s: float, fallback: bool = False,
                         note: str = "") -> None:
    _BACKEND_MANIFEST.append({
        "role": role or "?", "model": model, "backend": backend, "route": route,
        "latency_s": round(latency_s, 3), "fallback": fallback,
        **({"note": note} if note else {}),
    })


def backend_manifest(reset: bool = False) -> List[Dict[str, Any]]:
    out = list(_BACKEND_MANIFEST)
    if reset:
        _BACKEND_MANIFEST.clear()
    return out


def reset_backend_manifest() -> None:
    _BACKEND_MANIFEST.clear()


@dataclass
class LLMResponse:


    text: str
    model: str
    route: str
    latency_s: float = 0.0
    attempts: int = 1
    reasoning_effort: str = ""
    served_model: str = ""
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "served_model": self.served_model or self.model,
            "route": self.route, "latency_s": round(self.latency_s, 3),
            "attempts": self.attempts, "reasoning_effort": self.reasoning_effort,
            "chars": len(self.text), "reasoning_chars": len(self.reasoning),
        }


def is_claude_model(model: str) -> bool:
    return "claude" in (model or "").lower()


def normalize_reasoning_effort(raw: str) -> str:
    value = (raw or "").strip().lower().replace(" ", "").replace("-", "")
    return {"extrahigh": "xhigh", "maximum": "max", "med": "medium"}.get(value, value)


def reasoning_effort_for(model: str) -> str:
    try:
        return resolve(model).default_effort
    except UnknownModelError:
        return ""


def validate_reasoning_effort(model: str, effort: str) -> str:
    try:
        return resolve(model).validate_effort(effort)
    except UnknownModelError:
        return normalize_reasoning_effort(effort)


def describe_route(model: str) -> str:
    try:
        return describe(model)
    except UnknownModelError:
        return f"{model} -> unregistered"


def credentials_available(model: str) -> bool:
    from .registry import credentials_available as _available
    return _available(model)


def encode_image_as_data_url(image_path: str | Path, max_edge: int = MAX_IMAGE_EDGE) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise LLMError(f"image not found: {path}")
    try:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise LLMError(f"unreadable image: {path}")
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest > max_edge:
            scale = max_edge / float(longest)
            image = cv2.resize(image, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise LLMError(f"could not encode {path}")
        payload = base64.b64encode(buffer.tobytes()).decode("ascii")
    except ImportError:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _split_data_url(data_url: str) -> tuple[str, str]:
    media_type, _, payload = data_url.partition(";base64,")
    return (media_type.replace("data:", "") or "image/jpeg"), payload


def _anthropic_headers(spec: ModelSpec) -> Dict[str, str]:
    provider = provider_of(spec.model_id)
    key = provider.api_key()
    if not key:
        raise LLMError(
            f"no API key for provider {provider.name!r}; set one of "
            f"{', '.join(provider.key_env)}"
        )
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "accept": "text/event-stream",
    }
    if provider.user_agent:
        headers["User-Agent"] = provider.user_agent
    return headers


def _anthropic_payload(spec: ModelSpec, system_prompt: str, user_prompt: str,
                       data_urls: Sequence[str], max_tokens: int,
                       effort: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for url in data_urls:
        media_type, payload = _split_data_url(url)
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": media_type,
                                   "data": payload}})
    body: Dict[str, Any] = {
        "model": spec.model_id,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system_prompt}] if system_prompt else [],
        "messages": [{"role": "user", "content": content}],
        "stream": True,
    }
    if not system_prompt:
        body.pop("system")
    if effort and spec.efforts:
        budget = EFFORT_THINKING_BUDGET.get(effort, EFFORT_THINKING_BUDGET["high"])
        body["max_tokens"] = max(max_tokens, budget + max(1024, max_tokens))
        body["thinking"] = {"type": "enabled", "budget_tokens": budget, "effort": effort}
    return body


def _post_anthropic(spec: ModelSpec, body: Dict[str, Any],
                    timeout: int) -> tuple[str, str, str]:
    provider = provider_of(spec.model_id)
    base = provider.base_url()
    if not base:
        raise LLMError(
            f"no base URL for provider {provider.name!r}; set one of "
            f"{', '.join(provider.base_env)}"
        )
    request = urllib.request.Request(
        f"{base}/v1/messages", data=json.dumps(body).encode("utf-8"),
        headers=_anthropic_headers(spec), method="POST",
    )
    chunks: List[str] = []
    thoughts: List[str] = []
    served = ""
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            kind = event.get("type", "")
            if kind == "message_start":
                served = str((event.get("message") or {}).get("model") or "")
            elif kind == "content_block_delta":
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    chunks.append(str(delta.get("text") or ""))
                elif delta.get("type") == "thinking_delta":
                    thoughts.append(str(delta.get("thinking") or ""))
            elif kind == "error":
                raise LLMError(f"stream error: {event.get('error')}")
    if served and served != spec.model_id:
        raise ModelMismatchError(
            f"requested {spec.model_id!r} but the endpoint served {served!r}; refusing "
            "a silently remapped model"
        )
    return "".join(chunks), served, "".join(thoughts)


def _openai_headers(spec: ModelSpec) -> Dict[str, str]:
    provider = provider_of(spec.model_id)
    key = provider.api_key()
    if not key:
        raise LLMError(f"no API key for provider {provider.name!r}")
    headers = {"content-type": "application/json", "authorization": f"Bearer {key}"}
    if provider.user_agent:
        headers["User-Agent"] = provider.user_agent
    return headers


def _responses_payload(spec: ModelSpec, system_prompt: str, user_prompt: str,
                       data_urls: Sequence[str], max_tokens: int,
                       effort: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    for url in data_urls:
        content.append({"type": "input_image", "image_url": url})
    payload: Dict[str, Any] = {
        "model": spec.model_id,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_tokens,
    }
    if system_prompt:
        payload["instructions"] = system_prompt
    if effort:
        payload["reasoning"] = {"effort": effort}
    return payload


def _chat_payload(spec: ModelSpec, system_prompt: str, user_prompt: str,
                  data_urls: Sequence[str], max_tokens: int,
                  effort: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for url in data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    payload: Dict[str, Any] = {"model": spec.model_id, "messages": messages,
                               "max_completion_tokens": max_tokens}
    if effort:
        payload["reasoning_effort"] = effort
    return payload


def _text_from_responses(data: Dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: List[str] = []
    for item in data.get("output", []) or []:
        for block in item.get("content", []) or []:
            if block.get("type") in {"output_text", "text"}:
                chunks.append(str(block.get("text", "")))
    return "".join(chunks)


def _text_from_chat(data: Dict[str, Any]) -> str:
    for choice in data.get("choices", []) or []:
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(b.get("text", "")) for b in content)
    return ""


def _gemini_payload(system_prompt: str, user_prompt: str, data_urls: Sequence[str],
                    max_tokens: int) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [{"text": user_prompt}]
    for url in data_urls:
        media_type, payload = _split_data_url(url)
        parts.append({"inline_data": {"mime_type": media_type, "data": payload}})
    body: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    return body


def _post_gemini(spec: ModelSpec, body: Dict[str, Any], timeout: int) -> str:
    provider = provider_of(spec.model_id)
    base = provider.base_url()
    key = provider.api_key()
    if not base or not key:
        raise LLMError("GEMINI_BASE_URL / GEMINI_API_KEY are not both set")
    headers = {"x-goog-api-key": key, "content-type": "application/json"}
    if provider.user_agent:
        headers["User-Agent"] = provider.user_agent
    request = urllib.request.Request(
        f"{base}/v1beta/models/{spec.model_id}:generateContent",
        data=json.dumps(body).encode("utf-8"), headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8", "replace"))
    candidates = data.get("candidates") or []
    if not candidates:
        blocked = (data.get("promptFeedback") or {}).get("blockReason")
        raise LLMError(f"gemini returned no candidate"
                       + (f" (blocked: {blocked})" if blocked else ""))
    return "".join(
        str(part.get("text", ""))
        for candidate in candidates
        for part in (candidate.get("content") or {}).get("parts", [])
    )


def call_model(
    system_prompt: str,
    user_prompt: str,
    image_paths: Optional[Sequence[str | Path]] = None,
    model: str = DEFAULT_CLAUDE_MODEL,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 3,
    reasoning_effort: Optional[str] = None,
    backend: str = "",
    fallback_to_api: bool = False,
    role: str = "",
    **_ignored: Any,
) -> LLMResponse:

    if not is_registered(model):
        raise LLMError(
            f"model {model!r} is not registered. Run "
            "`python -m mewm.cli.main models` to list the available ids."
        )
    spec = backend_for(model, backend) if backend else resolve(model)
    effort = spec.validate_effort(reasoning_effort or "")

    if spec.is_local:
        from .local_models import WeightsUnavailableError, call_local_model
        try:
            response = call_local_model(spec, system_prompt, user_prompt, image_paths,
                                        max_tokens=max_tokens)
            record_backend_event(role, spec.model_id, BACKEND_LOCAL, response.route,
                                 response.latency_s)
            return response
        except WeightsUnavailableError as exc:
            if not fallback_to_api:
                raise
            LOGGER.warning(
                "role %s: local weights for %s unavailable; --fallback-to-api is on, "
                "re-routing this call to %s. The output is NOT comparable with a pure "
                "local run. (%s)", role or "?", spec.model_id, DEFAULT_CLAUDE_MODEL,
                str(exc).splitlines()[0])
            response = call_model(
                system_prompt, user_prompt, image_paths=image_paths,
                model=DEFAULT_CLAUDE_MODEL, max_tokens=max_tokens, timeout=timeout,
                retries=retries, role=role)
            record_backend_event(role, DEFAULT_CLAUDE_MODEL, BACKEND_HOSTED,
                                 response.route, response.latency_s, fallback=True,
                                 note=f"local weights for {spec.model_id} unavailable")
            return response

    data_urls = [encode_image_as_data_url(p) for p in (image_paths or [])]
    if data_urls and not spec.vision:
        LOGGER.warning("%s has no vision support; dropping %d image(s)",
                       spec.model_id, len(data_urls))
        data_urls = []

    started = time.time()
    last_error: Optional[Exception] = None
    attempts_allowed = max(1, retries, spec.min_retries)

    for attempt in range(attempts_allowed):
        try:
            if spec.transport == TRANSPORT_ANTHROPIC:
                body = _anthropic_payload(spec, system_prompt, user_prompt,
                                          data_urls, max_tokens, effort)
                text, served, thinking = _post_anthropic(spec, body, timeout)
                if text.strip():
                    response = LLMResponse(text, spec.model_id,
                                           f"{provider_of(model).base_url()}/v1/messages",
                                           time.time() - started, attempt + 1, effort,
                                           served, thinking)
                    record_backend_event(role, spec.model_id, BACKEND_HOSTED,
                                         response.route, response.latency_s)
                    return response
                last_error = LLMError("empty completion")

            elif spec.transport == TRANSPORT_GEMINI:
                body = _gemini_payload(system_prompt, user_prompt, data_urls, max_tokens)
                text = _post_gemini(spec, body, timeout)
                if text.strip():
                    response = LLMResponse(
                        text, spec.model_id,
                        f"{provider_of(model).base_url()}/v1beta/models/"
                        f"{spec.model_id}:generateContent",
                        time.time() - started, attempt + 1, effort, spec.model_id)
                    record_backend_event(role, spec.model_id, BACKEND_HOSTED,
                                         response.route, response.latency_s)
                    return response
                last_error = LLMError("empty completion")

            elif spec.transport == TRANSPORT_OPENAI:
                base = provider_of(model).base_url()
                if not base:
                    raise LLMError("OPENAI_API_URL is not set")
                routes = [
                    (f"{base}/responses", _responses_payload, _text_from_responses),
                ]
                for url, build, extract in routes:
                    payload = build(spec, system_prompt, user_prompt, data_urls,
                                    max_tokens, effort)
                    try:
                        request = urllib.request.Request(
                            url, data=json.dumps(payload).encode("utf-8"),
                            headers=_openai_headers(spec), method="POST")
                        with urllib.request.urlopen(request, timeout=timeout) as response:
                            data = json.loads(response.read().decode("utf-8", "replace"))
                        text = extract(data)
                    except urllib.error.HTTPError as exc:
                        detail = exc.read().decode("utf-8", "replace")[:300]
                        last_error = LLMError(f"HTTP {exc.code} from {url}: {detail}")
                        continue
                    if text.strip():
                        response = LLMResponse(text, spec.model_id, url,
                                               time.time() - started, attempt + 1,
                                               effort, str(data.get("model", "")))
                        record_backend_event(role, spec.model_id, BACKEND_HOSTED,
                                             response.route, response.latency_s)
                        return response
                    last_error = LLMError(f"empty completion from {url}")
            else:
                raise LLMError(f"unsupported transport {spec.transport!r}")

        except ModelMismatchError:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_error = LLMError(f"HTTP {exc.code} from {spec.provider}: {detail}")
        except LLMError as exc:
            last_error = exc
        except Exception as exc:
            last_error = LLMError(f"{type(exc).__name__}: {exc}")

        LOGGER.warning("%s attempt %d/%d failed after %.0fs (%s); retrying",
                       spec.model_id, attempt + 1, attempts_allowed,
                       time.time() - started, last_error)
        time.sleep(1.5 * (attempt + 1))

    raise LLMError(
        f"{spec.model_id} transport failed after {attempts_allowed} attempts: {last_error}")


def ping(model: str, timeout: int = 120,
         image: Optional[str] = None) -> Dict[str, Any]:

    started = time.time()
    prompt = ("Answer in one word: does this image show a human face?" if image
              else "Reply with the single word: ok")
    try:
        response = call_model(
            "You are terse.", prompt, image_paths=[image] if image else None,
            model=model, max_tokens=64, timeout=timeout, retries=1,
        )
        return {
            "model": model, "resolved": resolve(model).model_id, "ok": True,
            "vision": bool(image),
            "served": response.served_model, "text": response.text.strip()[:40],
            "latency_s": round(time.time() - started, 2),
            "route": response.route,
        }
    except Exception as exc:
        return {
            "model": model,
            "resolved": resolve(model).model_id if is_registered(model) else model,
            "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300],
            "latency_s": round(time.time() - started, 2),
        }


class StubClient:

    def __init__(self, responses: Optional[Dict[str, Any]] = None) -> None:
        self.responses: Dict[str, Any] = dict(responses or {})
        self.calls: List[Dict[str, Any]] = []

    def register(self, key: str, payload: Any) -> None:
        self.responses[key] = payload

    def __call__(self, system_prompt: str, user_prompt: str,
                 image_paths: Optional[Sequence[str | Path]] = None,
                 model: str = "stub", **kwargs: Any) -> LLMResponse:
        key = kwargs.get("role") or _infer_role(system_prompt)
        self.calls.append({"role": key, "model": model,
                           "system": system_prompt[:120], "user": user_prompt[:200]})
        payload = self.responses.get(key, self.responses.get("default", {}))
        text = payload if isinstance(payload, str) else json.dumps(payload,
                                                                   ensure_ascii=False)
        return LLMResponse(text, model, "stub", 0.0, 1, "", model)


def _infer_role(system_prompt: str) -> str:
    lowered = (system_prompt or "").lower()
    for needle, role in (
        ("perception", "P"), ("感知智能体", "P"),
        ("structure", "A"), ("结构智能体", "A"),
        ("reasoning", "R"), ("推理智能体", "R"),
        ("critic", "C"), ("批评智能体", "C"),
    ):
        if needle in lowered:
            return role
    return "default"


__all__ = [
    "DEFAULT_GPT_MODEL", "DEFAULT_CLAUDE_MODEL", "LLMError", "ModelMismatchError",
    "BackendMismatch", "BACKEND_HOSTED", "BACKEND_LOCAL", "backend_for",
    "record_backend_event", "backend_manifest", "reset_backend_manifest",
    "LLMResponse", "call_model", "ping", "StubClient", "encode_image_as_data_url",
    "describe_route", "credentials_available", "is_claude_model",
    "normalize_reasoning_effort", "reasoning_effort_for", "validate_reasoning_effort",
]
