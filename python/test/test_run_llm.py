#!/usr/bin/env python3
"""
Test script: validate all models in llm_models.yaml by sending a simple "hello".

Usage:
  python test/test_run_llm.py [--config llm_models.yaml]
"""
from __future__ import annotations

import argparse
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
    }

    api_url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "hello"},
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.time()
    try:
        data = json.dumps(payload).encode("utf-8")
        http_req = urllib_request.Request(api_url, data=data, headers=headers, method="POST")
        with urllib_request.urlopen(http_req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        msg = raw["choices"][0]["message"]
        response_text = msg.get("content", "") or msg.get("reasoning_content", "") or ""
        result["status"] = "ok"
        result["elapsed_s"] = round(elapsed, 2)
        result["response_text"] = response_text[:200]
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
    args = ap.parse_args()

    config_path = REPO_ROOT / args.config
    models = load_models(config_path)

    if not models:
        print("ERROR: no models found in config")
        sys.exit(1)

    print(f"Testing {len(models)} model(s) from {config_path}")
    print("=" * 80)

    results = []
    all_ok = True
    for name, entry in models.items():
        print(f"\n[{name}] Testing ...", end=" ", flush=True)
        r = test_model(name, entry)
        results.append(r)

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

    if not all_ok:
        print("\nWARNING: some models failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\nAll models are ready.")


if __name__ == "__main__":
    main()
