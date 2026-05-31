from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_LLM_MODELS_YAML = Path(__file__).resolve().parents[3] / "llm_models.yaml"
DEFAULT_SKILL_ROOT = Path(__file__).resolve().parents[3] / ".agent" / "skills"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


def _canonical_provider(provider: Optional[str]) -> str:
    value = (provider or "openai_compatible").strip()
    if value in {"openai_http", "openai_compatible"}:
        return "openai_compatible"
    if value in {"anthropic_http", "anthropic_compatible"}:
        return "anthropic_compatible"
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is not None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            return {}
        return payload
    payload = _read_simple_models_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _coerce_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped in {"", "null", "None"}:
        return ""
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def _read_simple_models_yaml(text: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"models": {}}
    current_model: Optional[str] = None
    in_defaults = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "models:":
            continue
        if indent == 2 and stripped.endswith(":"):
            current_model = stripped[:-1]
            payload["models"][current_model] = {}
            in_defaults = False
            continue
        if current_model is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if indent == 4 and key == "defaults":
            payload["models"][current_model]["defaults"] = {}
            in_defaults = True
            continue
        if indent == 4:
            payload["models"][current_model][key] = _coerce_scalar(value)
            in_defaults = False
            continue
        if indent == 6 and in_defaults:
            payload["models"][current_model]["defaults"][key] = _coerce_scalar(value)
    return payload


@dataclass
class ResolvedModelConfig:
    provider: str
    model_name: str
    model: str
    api_key: str
    api_base_url: str
    anthropic_base_url: str
    anthropic_version: str
    temperature: float
    max_tokens: int
    timeout_sec: int
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "model": self.model,
            "api_key_present": bool(self.api_key),
            "api_base_url": self.api_base_url,
            "anthropic_base_url": self.anthropic_base_url,
            "anthropic_version": self.anthropic_version,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_sec": self.timeout_sec,
            "source": self.source,
        }


def resolve_model_config(
    *,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    anthropic_base_url: Optional[str] = None,
    anthropic_version: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout_sec: Optional[int] = None,
    yaml_path: Optional[str] = None,
) -> ResolvedModelConfig:
    config_path = Path(yaml_path) if yaml_path else DEFAULT_LLM_MODELS_YAML
    registry = _read_yaml(config_path).get("models", {})
    entry = registry.get(model_name or "", {}) if isinstance(registry, dict) else {}
    defaults = entry.get("defaults", {}) if isinstance(entry, dict) else {}

    env_key_name = entry.get("api_key_env") if isinstance(entry, dict) else None
    resolved_api_key = api_key or ""
    if not resolved_api_key and env_key_name:
        resolved_api_key = os.environ.get(str(env_key_name), "")
    if not resolved_api_key and isinstance(entry, dict):
        resolved_api_key = str(entry.get("api_key", "") or "")

    resolved_provider = _canonical_provider(provider or entry.get("provider"))
    resolved_model = str(model or entry.get("model") or model_name or "")
    resolved_model_name = str(model_name or resolved_model or "custom")
    resolved_api_base = str(api_base_url or entry.get("api_base_url") or "")
    resolved_anthropic_base = str(
        anthropic_base_url
        or entry.get("anthropic_base_url")
        or entry.get("api_base_url")
        or resolved_api_base
        or ""
    )

    if not resolved_model:
        raise ValueError("Could not resolve LLM model name.")
    if not resolved_api_base and resolved_provider == "openai_compatible":
        raise ValueError(f"Model '{resolved_model_name}' is missing api_base_url.")
    if not resolved_anthropic_base and resolved_provider == "anthropic_compatible":
        raise ValueError(f"Model '{resolved_model_name}' is missing anthropic_base_url.")

    return ResolvedModelConfig(
        provider=resolved_provider,
        model_name=resolved_model_name,
        model=resolved_model,
        api_key=resolved_api_key,
        api_base_url=resolved_api_base,
        anthropic_base_url=resolved_anthropic_base or resolved_api_base,
        anthropic_version=str(
            anthropic_version
            or entry.get("anthropic_version")
            or defaults.get("anthropic_version")
            or DEFAULT_ANTHROPIC_VERSION
        ),
        temperature=float(temperature if temperature is not None else defaults.get("temperature", 0.1)),
        max_tokens=int(max_tokens if max_tokens is not None else defaults.get("max_tokens", 4096)),
        timeout_sec=int(timeout_sec if timeout_sec is not None else defaults.get("timeout_sec", 180)),
        source="llm_models.yaml" if entry else "explicit",
    )
