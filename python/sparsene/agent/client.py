from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
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
            usage.get("reasoning_tokens")
            or usage.get("reasoning_token_count")
            or usage.get("output_tokens_details", {}).get("reasoning_tokens")
            if isinstance(usage.get("output_tokens_details"), dict)
            else None
        )
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


@dataclass
class LLMTextResponse:
    provider: str
    model: str
    content: str
    raw_payload: Dict[str, Any]
    usage: TokenUsage
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "content": self.content,
            "usage": self.usage.to_dict(),
            "elapsed_s": self.elapsed_s,
            "raw_payload": self.raw_payload,
        }


def _normalize_base_url(url: str) -> str:
    return url.strip().rstrip("/").replace("；", "").replace("\r", "")


def _extract_openai_text(raw_payload: Dict[str, Any]) -> str:
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
    if not content:
        content = message.get("reasoning_content", "") or raw_payload.get("reasoning_content", "")
    if not isinstance(content, str) or not content.strip():
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
) -> LLMTextResponse:
    api_url = _normalize_base_url(api_base_url)
    if not api_url.endswith("/chat/completions"):
        api_url = f"{api_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    t0 = time.time()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(api_url, data=data, headers=headers, method="POST")
        with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"openai_compatible failed with status={exc.code}: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"openai_compatible could not reach endpoint: {exc.reason}") from exc
    content = _extract_openai_text(raw)
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
    )
