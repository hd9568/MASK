from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid

from .client import LLMTextResponse, TokenUsage, call_anthropic_compatible, call_openai_compatible
from .config import DEFAULT_SKILL_ROOT, ResolvedModelConfig, resolve_model_config
from .skills import load_skill_documents, render_skill_context


JsonParser = Callable[[str], Dict[str, Any]]


def make_zero_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0, raw_usage={})


def make_agent_run_dir(log_dir: str, agent_name: str, subject: str) -> Path:
    base_dir = Path(log_dir)
    if not base_dir.is_absolute():
        base_dir = Path(__file__).resolve().parents[3] / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = subject.replace(" ", "_").replace("/", "_")
    run_dir = base_dir / f"{stamp}_{agent_name}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass
class AgentModelConfig:
    provider: str = "openai_compatible"
    model_name: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    anthropic_base_url: Optional[str] = None
    anthropic_version: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout_sec: Optional[int] = None
    reasoning_effort: Optional[str] = None
    thinking_mode: Optional[str] = None
    stream: Optional[bool] = None
    model_config_path: Optional[str] = None

    def resolve(self) -> ResolvedModelConfig:
        return resolve_model_config(
            model_name=self.model_name or self.model,
            provider=self.provider,
            model=self.model,
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            anthropic_base_url=self.anthropic_base_url,
            anthropic_version=self.anthropic_version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_sec=self.timeout_sec,
            reasoning_effort=self.reasoning_effort,
            thinking_mode=self.thinking_mode,
            stream=self.stream,
            yaml_path=self.model_config_path,
        )


@dataclass
class AgentInvocationResult:
    run_dir: Path
    resolved_model: ResolvedModelConfig
    request_payload: Dict[str, Any]
    response_payload: Dict[str, Any]
    response_text: str
    usage: TokenUsage
    parsed_payload: Dict[str, Any] = field(default_factory=dict)
    finish_reason: Optional[str] = None


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compose_user_prompt(
    *,
    user_prompt: str,
    skill_names: Optional[List[str]],
    skill_root: Optional[Path],
) -> tuple[str, List[Dict[str, str]]]:
    if not skill_names:
        return user_prompt, []
    docs = load_skill_documents(skill_names, skill_root=skill_root or DEFAULT_SKILL_ROOT)
    if not docs:
        return user_prompt, []
    skill_context = render_skill_context([doc.name for doc in docs], skill_root=skill_root or DEFAULT_SKILL_ROOT)
    augmented = (
        f"{user_prompt.rstrip()}\n\n"
        "=== AVAILABLE DOMAIN SKILLS ===\n"
        "Use the following skills as authoritative project context.\n\n"
        f"{skill_context}\n"
    )
    return augmented, [doc.to_dict() for doc in docs]


def _invoke_llm(
    *,
    resolved_model: ResolvedModelConfig,
    system_prompt: str,
    user_prompt: str,
    json_output: bool = False,
) -> LLMTextResponse:
    if resolved_model.provider == "openai_compatible":
        return call_openai_compatible(
            model=resolved_model.model,
            api_key=resolved_model.api_key,
            api_base_url=resolved_model.api_base_url,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=resolved_model.temperature,
            max_tokens=resolved_model.max_tokens,
            timeout_sec=resolved_model.timeout_sec,
            reasoning_effort=resolved_model.reasoning_effort,
            thinking_mode=resolved_model.thinking_mode,
            stream=resolved_model.stream,
            response_format={"type": "json_object"} if json_output else None,
        )
    if resolved_model.provider == "anthropic_compatible":
        return call_anthropic_compatible(
            model=resolved_model.model,
            api_key=resolved_model.api_key,
            anthropic_base_url=resolved_model.anthropic_base_url,
            anthropic_version=resolved_model.anthropic_version,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=resolved_model.temperature,
            max_tokens=resolved_model.max_tokens,
            timeout_sec=resolved_model.timeout_sec,
        )
    raise ValueError(f"Unsupported runtime provider: {resolved_model.provider}")


def invoke_json_agent(
    *,
    agent_name: str,
    subject: str,
    log_dir: str,
    model_config: AgentModelConfig,
    system_prompt: str,
    user_prompt: str,
    request_payload: Dict[str, Any],
    response_parser: JsonParser,
    skill_names: Optional[List[str]] = None,
    skill_root: Optional[Path] = None,
    json_output: bool = False,
) -> AgentInvocationResult:
    run_dir = make_agent_run_dir(log_dir, agent_name, subject)
    resolved_model = model_config.resolve()
    prompt_with_skills, loaded_skills = _compose_user_prompt(
        user_prompt=user_prompt,
        skill_names=skill_names,
        skill_root=skill_root,
    )

    trace_request = {
        "agent_name": agent_name,
        "subject": subject,
        "request_id": str(uuid.uuid4()),
        "model": resolved_model.to_dict(),
        "skills": loaded_skills,
        "request_options": {
            "json_output": json_output,
            "max_tokens": resolved_model.max_tokens,
            "reasoning_effort": resolved_model.reasoning_effort,
            "thinking_mode": resolved_model.thinking_mode,
            "stream": resolved_model.stream,
        },
        "request": request_payload,
    }
    _write_json(run_dir / "request.json", trace_request)
    (run_dir / "system_prompt.txt").write_text(system_prompt)
    (run_dir / "user_prompt.txt").write_text(prompt_with_skills)
    _write_json(
        run_dir / "messages.json",
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_with_skills},
            ]
        },
    )

    started_at = datetime.now().isoformat()
    prompt_audit = {
        "system_prompt": {
            "path": "system_prompt.txt",
            "characters": len(system_prompt),
            "sha256": _sha256_text(system_prompt),
        },
        "user_prompt": {
            "path": "user_prompt.txt",
            "characters": len(prompt_with_skills),
            "sha256": _sha256_text(prompt_with_skills),
        },
    }

    try:
        llm_response = _invoke_llm(
            resolved_model=resolved_model,
            system_prompt=system_prompt,
            user_prompt=prompt_with_skills,
            json_output=json_output,
        )
    except Exception as exc:
        failed_usage = getattr(exc, "usage", None) or TokenUsage.from_payload(
            None,
            prompt_text=system_prompt + "\n" + prompt_with_skills,
            completion_text=getattr(exc, "partial_content", ""),
        )
        partial_content = getattr(exc, "partial_content", "")
        if partial_content:
            (run_dir / "partial_response_text.txt").write_text(partial_content)
        token_source = (
            "api"
            if failed_usage.raw_usage
            else ("estimated_after_failed_stream" if partial_content else "estimated_after_failed_request")
        )
        failure_payload = {
            "status": "failed",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "usage": failed_usage.to_dict(),
            "token_source": token_source,
            "input": prompt_audit,
            "output": {
                **({"path": "partial_response_text.txt"} if partial_content else {}),
                "characters": len(partial_content),
                "sha256": _sha256_text(partial_content),
            },
        }
        _write_json(run_dir / "error.json", failure_payload)
        _write_json(run_dir / "audit.json", failure_payload)
        _write_json(run_dir / "usage.json", failed_usage.to_dict())
        raise
    raw_response_payload = {
        "provider": llm_response.provider,
        "model": llm_response.model,
        "elapsed_s": llm_response.elapsed_s,
        "finish_reason": llm_response.finish_reason,
        "usage": llm_response.usage.to_dict(),
        "raw_payload": llm_response.raw_payload,
    }
    _write_json(run_dir / "raw_response.json", raw_response_payload)
    (run_dir / "response_text.txt").write_text(llm_response.content)
    parse_error: Optional[str] = None
    try:
        parsed_payload = response_parser(llm_response.content)
    except Exception as exc:
        (run_dir / "parse_error.txt").write_text(str(exc))
        parse_failure = {
            "status": "parse_failed",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "elapsed_s": llm_response.elapsed_s,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "usage": llm_response.usage.to_dict(),
            "token_source": "api" if llm_response.usage.raw_usage else "estimated",
            "input": prompt_audit,
            "output": {
                "path": "response_text.txt",
                "characters": len(llm_response.content),
                "sha256": _sha256_text(llm_response.content),
            },
        }
        _write_json(run_dir / "audit.json", parse_failure)
        _write_json(run_dir / "usage.json", llm_response.usage.to_dict())
        raise
    if not isinstance(parsed_payload, dict):
        parse_error = "JSON response parser did not return an object"
        (run_dir / "parse_error.txt").write_text(parse_error)
        parsed_payload = {}
    response_payload = {
        "provider": llm_response.provider,
        "model": llm_response.model,
        "elapsed_s": llm_response.elapsed_s,
        "finish_reason": llm_response.finish_reason,
        "usage": llm_response.usage.to_dict(),
        "parsed_payload": parsed_payload,
        "raw_payload": llm_response.raw_payload,
    }
    _write_json(run_dir / "response.json", response_payload)
    _write_json(run_dir / "usage.json", llm_response.usage.to_dict())
    _write_json(run_dir / "parsed.json", parsed_payload)
    _write_json(
        run_dir / "audit.json",
        {
            "status": (
                "truncated"
                if llm_response.finish_reason == "length"
                else ("parse_failed" if parse_error else "ok")
            ),
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "elapsed_s": llm_response.elapsed_s,
            "finish_reason": llm_response.finish_reason,
            "parse_error": parse_error,
            "usage": llm_response.usage.to_dict(),
            "token_source": "api" if llm_response.usage.raw_usage else "estimated",
            "input": prompt_audit,
            "output": {
                "path": "response_text.txt",
                "characters": len(llm_response.content),
                "sha256": _sha256_text(llm_response.content),
            },
            "parsed_output": {"path": "parsed.json"},
        },
    )

    return AgentInvocationResult(
        run_dir=run_dir,
        resolved_model=resolved_model,
        request_payload=trace_request,
        response_payload=response_payload,
        response_text=llm_response.content,
        usage=llm_response.usage,
        parsed_payload=parsed_payload,
        finish_reason=llm_response.finish_reason,
    )
