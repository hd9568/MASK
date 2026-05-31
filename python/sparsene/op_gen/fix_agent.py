"""
Fix Agent — repairs compile/runtime failures in generated CUDA kernels.

Invoked after codegen+compile or after runtime correctness check when a
failure is detected.  The agent sees the error + source code and
iteratively proposes patches.

Configurable:
  --fix-agent-iterations N   (default 3)
  --fix-agent-model NAME      (default from llm_models.yaml)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sparsene.agent.runtime import AgentModelConfig, invoke_json_agent, make_zero_usage
from sparsene.logging import get_logger

logger = get_logger("fix_agent")


@dataclass
class FixAgentConfig:
    model_name: str = "deepseek-v4-pro"
    model: str = "deepseek-v4-pro"
    api_key: str = ""
    api_base_url: str = "https://api.deepseek.com/v1"
    log_dir: str = "docs/agent_runs"
    max_iterations: int = 3
    timeout_sec: int = 120
    temperature: float = 0.1
    max_tokens: int = 4096
    model_config_path: Optional[str] = None
    skill_names: List[str] = field(default_factory=lambda: ["sparsene-framework", "cuda-fix"])


@dataclass
class FixResult:
    success: bool
    iterations: int
    original_error: str
    final_error: str
    patch_applied: bool
    token_usage: Dict[str, Any] = field(default_factory=dict)
    agent_metadata: Dict[str, Any] = field(default_factory=dict)


FIX_AGENT_SYSTEM_PROMPT = """You are a CUDA kernel debugging expert. You will be given:
1. A snippet of generated CUDA code (CUTLASS-style inline PTX wrappers)
2. Compilation or runtime error messages

Your job: output a precise, minimal patch to fix the errors.

Rules:
- Only fix the specific errors shown. Do NOT refactor or restructure.
- If the error is "identifier X is undefined", add that identifier as
  a function parameter or constructor parameter (whichever is appropriate).
- If it's a template parameter redeclaration, remove the duplicate.
- Output ONLY a JSON object with this exact schema:
{
  "action": "patch | abort",
  "patches": [
    {
      "old_text": "exact lines to replace (must be unique in the file)",
      "new_text": "replacement lines"
    }
  ],
  "rationale": "brief explanation"
}
If you cannot fix the error, set "action": "abort".
"""


def _call_llm(config: FixAgentConfig, user_content: str, log_dir: Path) -> tuple[Optional[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    try:
        result = invoke_json_agent(
            agent_name="fix_agent",
            subject="generated_kernel",
            log_dir=str(log_dir),
            model_config=AgentModelConfig(
                provider="openai_compatible",
                model_name=config.model_name or config.model,
                model=config.model,
                api_key=config.api_key,
                api_base_url=config.api_base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout_sec=config.timeout_sec,
                model_config_path=config.model_config_path,
            ),
            system_prompt=FIX_AGENT_SYSTEM_PROMPT,
            user_prompt=user_content,
            request_payload={"kernel_prompt": user_content},
            response_parser=lambda text: json.loads(
                re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", text.strip()))
            ),
            skill_names=config.skill_names,
        )
        return (
            result.parsed_payload,
            result.usage.to_dict(),
            {"run_dir": str(result.run_dir), "resolved_model": result.resolved_model.to_dict()},
        )
    except Exception as e:
        logger.error("Fix agent LLM call failed: %s", e)
        (log_dir / "fix_agent_error.txt").write_text(str(e))
        return None, make_zero_usage().to_dict(), {"error": str(e)}


def _apply_patch(kernel_path: Path, patches: List[Dict[str, str]]) -> bool:
    source = kernel_path.read_text()
    for p in patches:
        old = p["old_text"]
        new = p["new_text"]
        if old not in source:
            # Try with stripped leading whitespace
            old_stripped = "\n".join(line.lstrip(" ") for line in old.splitlines())
            source_stripped = "\n".join(line.lstrip(" ") for line in source.splitlines())
            if old_stripped in source_stripped:
                old = old_stripped
                source = source_stripped
            else:
                # Find best match by truncating lines
                first_line = old.splitlines()[0].strip() if old.splitlines() else ""
                found = False
                for line in source.splitlines():
                    if first_line and first_line == line.strip():
                        # Approximate match by line content
                        logger.info("  Found approximate match for: %s", first_line[:80])
                        found = True
                        break
                if not found:
                    logger.error("  Patch old_text not found. First 100 chars: %s", old[:100])
                    return False
        source = source.replace(old, new, 1)
    kernel_path.write_text(source)
    return True


def run_fix_agent(
    kernel_path: Path,
    error_log: str,
    config: FixAgentConfig,
    log_dir: Path,
    compile_fn,  # callable: () -> Tuple[bool, str] — returns (success, new_error)
) -> FixResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "original_error.txt").write_text(error_log)

    for iteration in range(1, config.max_iterations + 1):
        logger.info("[Fix Agent] Iteration %d/%d", iteration, config.max_iterations)

        source_snippet = kernel_path.read_text()
        max_source = 15000
        if len(source_snippet) > max_source:
            half = max_source // 2
            source_snippet = source_snippet[:half] + "\n// ... [truncated] ...\n" + source_snippet[-half:]

        user_prompt = f"""=== COMPILATION / RUNTIME ERROR ===
{error_log[-3000:]}

=== KERNEL SOURCE ({kernel_path}) ===
{source_snippet}
"""
        (log_dir / f"iter{iteration:02d}_prompt.txt").write_text(user_prompt)

        response, token_usage, agent_metadata = _call_llm(config, user_prompt, log_dir)
        if response is None:
            return FixResult(False, iteration, error_log, "LLM call failed", False, token_usage, agent_metadata)

        (log_dir / f"iter{iteration:02d}_response.json").write_text(
            json.dumps(response, indent=2, ensure_ascii=False)
        )

        if response.get("action") != "patch":
            logger.info("  Agent chose 'abort' — cannot fix")
            return FixResult(False, iteration, error_log,
                             response.get("rationale", "agent aborted"), False, token_usage, agent_metadata)

        patches = response.get("patches", [])
        if not patches:
            logger.info("  No patches provided")
            return FixResult(False, iteration, error_log, "no patches", False, token_usage, agent_metadata)

        if not _apply_patch(kernel_path, patches):
            return FixResult(False, iteration, error_log, "patch apply failed", False, token_usage, agent_metadata)

        success, new_error = compile_fn()
        if success:
            logger.info("  Fix SUCCESS at iteration %d", iteration)
            return FixResult(True, iteration, error_log, "", True, token_usage, agent_metadata)

        error_log = new_error
        logger.info("  Fix iteration %d: still failing. %s", iteration, new_error[:120])

    return FixResult(
        False,
        config.max_iterations,
        error_log,
        f"still failing after {config.max_iterations} iterations",
        True,
        make_zero_usage().to_dict(),
        {"source": "max_iterations_exhausted"},
    )
