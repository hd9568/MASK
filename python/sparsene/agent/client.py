from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import ssl
import time
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4.0)))


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = None
    raw_usage: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        raw_usage: Optional[Dict[str, Any]],
        *,
        prompt_text: str,
        completion_text: str,
    ) -> "TokenUsage":
        usage = raw_usage or {}
        prompt_tokens = _as_int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
        )
        completion_tokens = _as_int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("completion_token_count")
        )
        reasoning_tokens = _as_int(
            usage.get("reasoning_tokens") or usage.get("reasoning_token_count")
        )
        if reasoning_tokens is None:
            for details_name in ("completion_tokens_details", "output_tokens_details"):
                details = usage.get(details_name)
                if isinstance(details, dict):
                    reasoning_tokens = _as_int(details.get("reasoning_tokens"))
                    if reasoning_tokens is not None:
                        break
        if prompt_tokens is None:
            prompt_tokens = _estimate_tokens(prompt_text)
        if completion_tokens is None:
            completion_tokens = _estimate_tokens(completion_text)
        total_tokens = _as_int(
            usage.get("total_tokens")
            or usage.get("total_token_count")
            or (prompt_tokens + completion_tokens)
        )
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            raw_usage=usage,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StreamingInterruptedError(RuntimeError):
    """Streaming transport failed after response headers, retain received data."""

    def __init__(self, message: str, *, partial_content: str, usage: TokenUsage):
        super().__init__(message)
        self.partial_content = partial_content
        self.usage = usage


@dataclass
class LLMTextResponse:
    provider: str
    model: str
    content: str
    raw_payload: Dict[str, Any]
    usage: TokenUsage
    elapsed_s: float
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "content": self.content,
            "usage": self.usage.to_dict(),
            "elapsed_s": self.elapsed_s,
            "finish_reason": self.finish_reason,
            "raw_payload": self.raw_payload,
        }


def _normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/").replace("；", "").replace("\r", "")


def _urlopen_with_tls_eof_retry(req: urllib_request.Request, timeout_sec: int):
    """Retry only TLS handshake EOFs, which occur before an HTTP request is sent."""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            return urllib_request.urlopen(req, timeout=timeout_sec)
        except urllib_error.URLError as exc:
            reason = exc.reason
            is_tls_eof = isinstance(reason, ssl.SSLError) and (
                "UNEXPECTED_EOF" in str(reason) or "EOF occurred in violation" in str(reason)
            )
            if not is_tls_eof or attempt == max_attempts:
                raise
            time.sleep(attempt)


def _extract_openai_text(
    raw_payload: Dict[str, Any], *, allow_reasoning_fallback: bool = True, allow_empty: bool = False
) -> str:
    choices = raw_payload.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenAI-compatible payload has no choices: {raw_payload}")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        content = "\n".join(part for part in text_parts if part)
    if not content and allow_reasoning_fallback:
        content = message.get("reasoning_content", "") or raw_payload.get("reasoning_content", "")
    if not isinstance(content, str) or not content.strip():
        if allow_empty:
            return ""
        raise RuntimeError(f"OpenAI-compatible payload has empty content: {raw_payload}")
    return content


def _extract_anthropic_text(raw_payload: Dict[str, Any]) -> str:
    content = raw_payload.get("content", [])
    text_parts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    merged = "\n".join(part for part in text_parts if part)
    if not merged.strip():
        raise RuntimeError(f"Anthropic payload has empty content: {raw_payload}")
    return merged


def _canonical_finish_reason(value: Any) -> Optional[str]:
    """Normalize relay-specific terminal names before truncation decisions."""
    if not isinstance(value, str) or not value:
        return None
    if value in {"length", "max_tokens", "max_output_tokens", "max_completion_tokens"}:
        return "length"
    if value in {"stop", "end_turn", "stop_sequence", "eos"}:
        return "stop"
    return value


def call_openai_compatible(
    *,
    model: str,
    api_key: str,
    api_base_url: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    response_format: Optional[Dict[str, str]] = None,
    reasoning_effort: Optional[str] = None,
    thinking_mode: Optional[str] = None,
    stream: bool = False,
) -> LLMTextResponse:
    api_url = _normalize_base_url(api_base_url)
    if not api_url.endswith("/chat/completions"):
        api_url = f"{api_url}/chat/completions"
    payload = {
        "model": model,
        # GPT-6 reasoning models use the current Chat Completions token field;
        # max_tokens is deprecated and some compatibility relays mishandle it.
        ("max_completion_tokens" if model.startswith("gpt-6") else "max_tokens"): max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # GPT-6.1 Sol requires an explicit reasoning budget for predictable
    # latency. OpenAI's API guidance also says not to send temperature when
    # reasoning_effort is active.
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    else:
        payload["temperature"] = temperature
    if thinking_mode is not None:
        if thinking_mode not in {"enabled", "disabled"}:
            raise ValueError(f"Unsupported thinking_mode: {thinking_mode}")
        payload["thinking"] = {"type": thinking_mode}
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    if response_format:
        payload["response_format"] = response_format
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    t0 = time.time()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(api_url, data=data, headers=headers, method="POST")
        with _urlopen_with_tls_eof_retry(req, timeout_sec) as resp:
            if stream:
                chunks = []
                message_fallbacks = []
                reasoning_chunks = []
                raw_usage: Dict[str, Any] = {}
                finish_reason = None
                response_meta: Dict[str, Any] = {}
                data_event_count = 0
                saw_done = False
                event_shape: Dict[str, Any] = {}
                try:
                    for line in resp:
                        line = line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        event_data = line[5:].strip()
                        if not event_data:
                            continue
                        if event_data == "[DONE]":
                            saw_done = True
                            continue
                        event = json.loads(event_data)
                        data_event_count += 1
                        if not response_meta:
                            response_meta = {
                                key: event[key]
                                for key in ("id", "object", "created", "model")
                                if key in event
                            }
                        if isinstance(event.get("usage"), dict):
                            raw_usage = event["usage"]
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            content_delta = delta.get("content", "")
                            if isinstance(content_delta, str) and content_delta:
                                chunks.append(content_delta)
                            # Some relays wrap a complete message in an SSE
                            # event instead of emitting the usual delta shape.
                            message_content = choice.get("message", {}).get("content", "")
                            if isinstance(message_content, str) and message_content:
                                message_fallbacks.append(message_content)
                            reasoning_delta = delta.get("reasoning_content", "")
                            if isinstance(reasoning_delta, str) and reasoning_delta:
                                reasoning_chunks.append(reasoning_delta)
                            terminal = _canonical_finish_reason(
                                choice.get("finish_reason") or choice.get("native_finish_reason")
                            )
                            if terminal:
                                finish_reason = terminal
                            if not event_shape:
                                event_shape = {
                                    "event_keys": sorted(event.keys()),
                                    "choice_keys": sorted(choice.keys()),
                                    "delta_keys": sorted(delta.keys()) if isinstance(delta, dict) else [],
                                }
                except Exception as exc:
                    partial_content = "".join(chunks)
                    if not partial_content and message_fallbacks:
                        partial_content = message_fallbacks[-1]
                    partial_reasoning = "".join(reasoning_chunks)
                    partial_usage = TokenUsage.from_payload(
                        raw_usage,
                        prompt_text=system_prompt + "\n" + user_prompt,
                        completion_text=partial_content + partial_reasoning,
                    )
                    raise StreamingInterruptedError(
                        "stream interrupted "
                        f"({type(exc).__name__}): {exc}; data_events={data_event_count}, "
                        f"saw_done={saw_done}, finish_reason={finish_reason!r}, "
                        f"content_chars={len(partial_content)}, event_shape={event_shape}",
                        partial_content=partial_content,
                        usage=partial_usage,
                    ) from exc
                content = "".join(chunks)
                # OpenLux occasionally sends a complete final message after
                # partial deltas. Prefer it only when it is strictly longer.
                if message_fallbacks and len(message_fallbacks[-1]) > len(content):
                    content = message_fallbacks[-1]
                if finish_reason is None:
                    partial_reasoning = "".join(reasoning_chunks)
                    partial_usage = TokenUsage.from_payload(
                        raw_usage,
                        prompt_text=system_prompt + "\n" + user_prompt,
                        completion_text=content + partial_reasoning,
                    )
                    raise StreamingInterruptedError(
                        "SSE ended without a terminal finish_reason; "
                        f"data_events={data_event_count}, content_chars={len(content)}, "
                        f"event_shape={event_shape}",
                        partial_content=content,
                        usage=partial_usage,
                    )
                raw = {
                    **response_meta,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            **({"reasoning_content": "".join(reasoning_chunks)} if reasoning_chunks else {}),
                        },
                        "finish_reason": finish_reason,
                    }],
                    "usage": raw_usage,
                    "stream_diagnostics": {
                        "data_event_count": data_event_count,
                        "saw_done": saw_done,
                        "finish_reason_seen": finish_reason is not None,
                        "event_shape": event_shape,
                    },
                }
            else:
                raw = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"openai_compatible failed with status={exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"openai_compatible could not reach endpoint: {exc.reason}") from exc
    terminal_reason = _canonical_finish_reason(
        raw.get("choices", [{}])[0].get("finish_reason")
        or raw.get("choices", [{}])[0].get("native_finish_reason")
    )
    content = _extract_openai_text(
        raw,
        allow_reasoning_fallback=not bool(response_format),
        # A reasoning model can spend the entire completion budget before
        # emitting any final JSON. Preserve the provider's length terminal so
        # the format loop can request a shorter complete retry.
        allow_empty=terminal_reason == "length",
    )
    usage = TokenUsage.from_payload(
        raw.get("usage"),
        prompt_text=system_prompt + "\n" + user_prompt,
        completion_text=content,
    )
    return LLMTextResponse(
        provider="openai_compatible",
        model=model,
        content=content,
        raw_payload=raw,
        usage=usage,
        elapsed_s=round(time.time() - t0, 4),
        finish_reason=terminal_reason,
    )


def call_anthropic_compatible(
    *,
    model: str,
    api_key: str,
    anthropic_base_url: str,
    anthropic_version: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> LLMTextResponse:
    api_url = _normalize_base_url(anthropic_base_url)
    if api_url.endswith("/anthropic"):
        api_url = f"{api_url}/v1/messages"
    elif not api_url.endswith("/v1/messages"):
        api_url = f"{api_url}/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
    }
    t0 = time.time()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(api_url, data=data, headers=headers, method="POST")
        with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"anthropic_compatible failed with status={exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"anthropic_compatible could not reach endpoint: {exc.reason}") from exc
    content = _extract_anthropic_text(raw)
    usage = TokenUsage.from_payload(
        raw.get("usage"),
        prompt_text=system_prompt + "\n" + user_prompt,
        completion_text=content,
    )
    return LLMTextResponse(
        provider="anthropic_compatible",
        model=model,
        content=content,
        raw_payload=raw,
        usage=usage,
        elapsed_s=round(time.time() - t0, 4),
        finish_reason=str(raw.get("stop_reason") or "") or None,
    )
