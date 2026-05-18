from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

from sparsene.op_gen.strategy_agent import (
    DEFAULT_TOKEN_PLAN_API_KEY,
    DEFAULT_TOKEN_PLAN_BASE_URL,
    DEFAULT_TOKEN_PLAN_MODEL,
)

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class ModelEndpoint:
    label: str
    base_url: str
    model: str
    api_key: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check connectivity for configured strategy models with a tiny prompt."
    )
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--token-plan-api-key", type=str, default=DEFAULT_TOKEN_PLAN_API_KEY)
    parser.add_argument("--token-plan-base-url", type=str, default=DEFAULT_TOKEN_PLAN_BASE_URL)
    parser.add_argument("--token-plan-model", type=str, default=DEFAULT_TOKEN_PLAN_MODEL)
    parser.add_argument("--deepseek-api-key", type=str, default=None)
    parser.add_argument("--deepseek-base-url", type=str, default=DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument(
        "--models",
        type=str,
        default="mimo-v2.5-pro,deepseek-v4-flash,deepseek-v4-pro",
        help="Comma-separated subset of model labels to test.",
    )
    return parser.parse_args()


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _selected_labels(spec: str) -> List[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def _build_endpoints(args: argparse.Namespace) -> List[ModelEndpoint]:
    token_plan_key = args.token_plan_api_key
    deepseek_key = args.deepseek_api_key
    presets = [
        ModelEndpoint(
            label="mimo-v2.5-pro",
            base_url=args.token_plan_base_url,
            model=args.token_plan_model,
            api_key=token_plan_key,
        ),
        ModelEndpoint(
            label="deepseek-v4-flash",
            base_url=args.deepseek_base_url,
            model="deepseek-v4-flash",
            api_key=deepseek_key,
        ),
        ModelEndpoint(
            label="deepseek-v4-pro",
            base_url=args.deepseek_base_url,
            model="deepseek-v4-pro",
            api_key=deepseek_key,
        ),
    ]
    wanted = set(_selected_labels(args.models))
    return [item for item in presets if item.label in wanted]


def _request_payload(model: str) -> Dict[str, Any]:
    return {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 128,
        "messages": [
            {"role": "system", "content": "You are a connectivity check endpoint."},
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
    }


def _extract_text(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                pieces.append(item.get("text", ""))
        return "".join(pieces).strip()
    return str(content).strip()


def _check_endpoint(endpoint: ModelEndpoint, timeout_sec: int) -> Dict[str, Any]:
    if not endpoint.api_key:
        return {
            "label": endpoint.label,
            "model": endpoint.model,
            "base_url": endpoint.base_url,
            "ok": False,
            "reason": "missing_api_key",
            "reply": "",
        }
    request_body = json.dumps(_request_payload(endpoint.model)).encode("utf-8")
    url = _normalize_base_url(endpoint.base_url) + "/chat/completions"
    http_request = urllib_request.Request(
        url,
        data=request_body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {endpoint.api_key}",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(http_request, timeout=timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {
            "label": endpoint.label,
            "model": endpoint.model,
            "base_url": endpoint.base_url,
            "ok": False,
            "reason": f"http_{exc.code}",
            "reply": detail[:400],
        }
    except Exception as exc:
        return {
            "label": endpoint.label,
            "model": endpoint.model,
            "base_url": endpoint.base_url,
            "ok": False,
            "reason": type(exc).__name__,
            "reply": str(exc)[:400],
        }
    reply = _extract_text(payload)
    return {
        "label": endpoint.label,
        "model": endpoint.model,
        "base_url": endpoint.base_url,
        "ok": bool(reply),
        "reason": "ok" if reply else "empty_reply",
        "reply": reply,
    }


def _default_output_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "docs" / "agent_runs" / f"model_connectivity_{stamp}.json"


def main() -> int:
    args = parse_args()
    endpoints = _build_endpoints(args)
    results = [_check_endpoint(endpoint, args.timeout_sec) for endpoint in endpoints]
    output_path = Path(args.output) if args.output else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"wrote: {output_path}")
    for row in results:
        status = "OK" if row["ok"] else "FAIL"
        print(
            f"{status:4s} | {row['label']:18s} | model={row['model']} | "
            f"base={row['base_url']} | reason={row['reason']} | reply={row['reply'][:80]}"
        )
    return 0 if all(row["ok"] for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
