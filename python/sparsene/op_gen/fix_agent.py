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
from datetime import datetime
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

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


@dataclass
class FixResult:
    success: bool
    iterations: int
    original_error: str
    final_error: str
    patch_applied: bool


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


def _call_llm(config: FixAgentConfig, user_content: str, log_dir: Path) -> Optional[Dict[str, Any]]:
    api_url = f"{config.api_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": FIX_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(api_url, data=data, headers=headers, method="POST")
        with urllib_request.urlopen(req, timeout=config.timeout_sec) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        (log_dir / f"{datetime.now().strftime('%H%M%S')}_response.json").write_text(
            json.dumps(raw, indent=2, ensure_ascii=False)
        )
        msg = raw["choices"][0]["message"]
        content = msg.get("content", "") or msg.get("reasoning_content", "") or ""
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception as e:
        logger.error("Fix agent LLM call failed: %s", e)
        (log_dir / f"{datetime.now().strftime('%H%M%S')}_error.txt").write_text(str(e))
        return None


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

        response = _call_llm(config, user_prompt, log_dir)
        if response is None:
            return FixResult(False, iteration, error_log, "LLM call failed", False)

        (log_dir / f"iter{iteration:02d}_response.json").write_text(
            json.dumps(response, indent=2, ensure_ascii=False)
        )

        if response.get("action") != "patch":
            logger.info("  Agent chose 'abort' — cannot fix")
            return FixResult(False, iteration, error_log,
                             response.get("rationale", "agent aborted"), False)

        patches = response.get("patches", [])
        if not patches:
            logger.info("  No patches provided")
            return FixResult(False, iteration, error_log, "no patches", False)

        if not _apply_patch(kernel_path, patches):
            return FixResult(False, iteration, error_log, "patch apply failed", False)

        success, new_error = compile_fn()
        if success:
            logger.info("  Fix SUCCESS at iteration %d", iteration)
            return FixResult(True, iteration, error_log, "", True)

        error_log = new_error
        logger.info("  Fix iteration %d: still failing. %s", iteration, new_error[:120])

    return FixResult(False, config.max_iterations, error_log,
                     f"still failing after {config.max_iterations} iterations", True)
