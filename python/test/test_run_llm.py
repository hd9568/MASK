#!/usr/bin/env python3
"""
Test script: validate all models in llm_models.yaml by sending a simple "hello".

Usage:
  python test/test_run_llm.py [--config llm_models.yaml]
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import yaml


def load_models(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("models", {})


def resolve_api_key(entry: Dict[str, Any]) -> str:
    env_name = entry.get("api_key_env")
    if env_name:
        val = os.environ.get(env_name, "")
        if val:
            return val
    return entry.get("api_key", "")


def test_model(name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    api_key = resolve_api_key(entry)
    base_url = entry.get("api_base_url", "").rstrip("/")
    model = entry.get("model", name)
    defaults = entry.get("defaults", {})
    timeout = defaults.get("timeout_sec", 60)

    result: Dict[str, Any] = {
        "model_name": name,
        "api_model": model,
        "provider": entry["provider"],
        "status": "unknown",
        "elapsed_s": 0.0,
        "error": "",
        "response_text": "",
        "request_payload": {},
    }

    api_url = f"{base_url}/chat/completions"
    if model.startswith("gpt-6"):
        # Match the current GPT-6 Chat Completions contract used by runtime:
        # max_completion_tokens + explicit reasoning effort; max_tokens is
        # deprecated, and temperature must not accompany active reasoning.
        payload = {
            "model": model,
            "reasoning_effort": defaults.get("reasoning_effort", "low"),
            "max_completion_tokens": 128,
            "stream": bool(defaults.get("stream", True)),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "user", "content": 'Return only JSON: {"ok": true}.'},
            ],
        }
        if payload["stream"]:
            payload["stream_options"] = {"include_usage": True}
    else:
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
        }
    result["request_payload"] = payload

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.time()
    try:
        data = json.dumps(payload).encode("utf-8")
        http_req = urllib_request.Request(api_url, data=data, headers=headers, method="POST")
        with urllib_request.urlopen(http_req, timeout=timeout) as resp:
            if payload.get("stream"):
                content_parts = []
                raw_usage = {}
                finish_reason = None
                for line in resp:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    event_data = line[5:].strip()
                    if not event_data or event_data == "[DONE]":
                        continue
                    event = json.loads(event_data)
                    if isinstance(event.get("usage"), dict):
                        raw_usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {}).get("content", "")
                        if isinstance(delta, str):
                            content_parts.append(delta)
                        finish_reason = choice.get("finish_reason") or finish_reason
                response_text = "".join(content_parts)
                raw = {
                    "choices": [{
                        "message": {"content": response_text},
                        "finish_reason": finish_reason,
                    }],
                    "usage": raw_usage,
                }
            else:
                raw = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        if not payload.get("stream"):
            msg = raw["choices"][0]["message"]
            response_text = msg.get("content", "") or msg.get("reasoning_content", "") or ""
        result["status"] = "ok"
        result["elapsed_s"] = round(elapsed, 2)
        result["response_text"] = response_text
        result["usage"] = raw.get("usage", {})
    except urllib_error.HTTPError as e:
        result["status"] = "http_error"
        result["elapsed_s"] = round(time.time() - t0, 2)
        try:
            body = e.read().decode("utf-8")
            result["error"] = f"HTTP {e.code}: {body[:300]}"
        except Exception:
            result["error"] = f"HTTP {e.code}: {e.reason}"
    except urllib_error.URLError as e:
        result["status"] = "url_error"
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["error"] = str(e.reason)[:300]
    except Exception as e:
        result["status"] = "error"
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["error"] = str(e)[:300]

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="llm_models.yaml")
    ap.add_argument(
        "--model",
        action="append",
        dest="model_names",
        help="Only test this configured model name; repeat to test several models.",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Persist complete request, response, usage, and summary under this directory.",
    )
    args = ap.parse_args()

    config_path = REPO_ROOT / args.config
    models = load_models(config_path)

    if args.model_names:
        missing = sorted(set(args.model_names) - set(models))
        if missing:
            print(f"ERROR: unknown configured model(s): {', '.join(missing)}")
            sys.exit(1)
        models = {name: models[name] for name in args.model_names}

    if not models:
        print("ERROR: no models found in config")
        sys.exit(1)

    print(f"Testing {len(models)} model(s) from {config_path}")
    print("=" * 80)

    audit_dir: Optional[Path] = None
    if args.log_dir:
        audit_dir = Path(args.log_dir)
        if not audit_dir.is_absolute():
            audit_dir = REPO_ROOT / audit_dir
        audit_dir = audit_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_model_connectivity"
        audit_dir.mkdir(parents=True, exist_ok=False)

    results = []
    all_ok = True
    for name, entry in models.items():
        print(f"\n[{name}] Testing ...", end=" ", flush=True)
        r = test_model(name, entry)
        results.append(r)
        if audit_dir is not None:
            model_dir = audit_dir / name
            model_dir.mkdir(parents=True, exist_ok=False)
            (model_dir / "user_prompt.txt").write_text("hello")
            (model_dir / "response_text.txt").write_text(r["response_text"])
            (model_dir / "request.json").write_text(
                json.dumps(r["request_payload"], indent=2, ensure_ascii=False)
            )
            (model_dir / "response.json").write_text(
                json.dumps(
                    {
                        "status": r["status"],
                        "elapsed_s": r["elapsed_s"],
                        "response_text": r["response_text"],
                        "error": r["error"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            (model_dir / "usage.json").write_text(
                json.dumps(r.get("usage", {}), indent=2, ensure_ascii=False)
            )

        icon = "OK" if r["status"] == "ok" else "FAIL"
        print(f"{icon} ({r['elapsed_s']:.1f}s)")

        if r["status"] == "ok":
            resp = r["response_text"].replace("\n", " ")
            print(f"  Response:   {resp[:120]}")
            usage = r.get("usage", {})
            if usage:
                print(f"  Usage:      prompt={usage.get('prompt_tokens','?')} completion={usage.get('completion_tokens','?')}")
        else:
            print(f"  Error:      {r['error'][:200]}")
            all_ok = False

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<24s} {'API Model':<24s} {'Status':<12s} {'Time':>6s} {'Response'}")
    print("-" * 100)
    for r in results:
        resp = r["response_text"].replace("\n", " ")[:60] if r["response_text"] else r["error"][:60]
        print(f"{r['model_name']:<24s} {r['api_model']:<24s} {r['status']:<12s} {r['elapsed_s']:>5.1f}s {resp}")

    print("-" * 100)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"  {ok_count}/{len(results)} models OK")
    if audit_dir is not None:
        summary = {
            "created_at": datetime.now().isoformat(),
            "config_path": str(config_path),
            "models": [
                {
                    key: value
                    for key, value in result.items()
                    if key != "request_payload"
                }
                for result in results
            ],
        }
        (audit_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        print(f"  Audit log: {audit_dir}")

    if not all_ok:
        print("\nWARNING: some models failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\nAll models are ready.")


if __name__ == "__main__":
    main()
