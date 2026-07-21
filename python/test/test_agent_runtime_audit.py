from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sparsene.agent.client import LLMTextResponse, TokenUsage, call_openai_compatible
from sparsene.agent.config import ResolvedModelConfig
from sparsene.agent.runtime import AgentModelConfig, invoke_json_agent


def _resolved() -> ResolvedModelConfig:
    return ResolvedModelConfig(
        provider="openai_compatible",
        model_name="mock-model",
        model="mock-model",
        api_key="not-logged",
        api_base_url="https://example.invalid/v1",
        anthropic_base_url="",
        anthropic_version="",
        temperature=0.1,
        max_tokens=128,
        timeout_sec=5,
        source="test",
    )


class AgentRuntimeAuditTest(unittest.TestCase):
    def test_openai_json_output_option_reaches_http_payload(self) -> None:
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {"content": '{"answer":"ok"}'},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                    }
                ).encode()

        def fake_urlopen(request, timeout):
            captured.update(json.loads(request.data.decode()))
            return FakeResponse()

        with patch("sparsene.agent.client.urllib_request.urlopen", side_effect=fake_urlopen):
            response = call_openai_compatible(
                model="mock-model",
                api_key="not-logged",
                api_base_url="https://example.invalid/v1",
                system_prompt="return json",
                user_prompt="return json",
                temperature=0.1,
                max_tokens=128,
                timeout_sec=5,
                response_format={"type": "json_object"},
            )
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertEqual(response.finish_reason, "stop")

    def test_success_records_complete_messages_response_and_api_usage(self) -> None:
        response = LLMTextResponse(
            provider="openai_compatible",
            model="mock-model",
            content='{"answer": "ok"}',
            raw_payload={"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 7}},
            usage=TokenUsage(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                raw_usage={"prompt_tokens": 11, "completion_tokens": 7},
            ),
            elapsed_s=0.25,
            finish_reason="stop",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            AgentModelConfig, "resolve", return_value=_resolved()
        ), patch("sparsene.agent.runtime._invoke_llm", return_value=response) as invoke_mock:
            result = invoke_json_agent(
                agent_name="audit",
                subject="matrix",
                log_dir=tmp,
                model_config=AgentModelConfig(),
                system_prompt="system content",
                user_prompt="user content",
                request_payload={"matrix": "matrix"},
                response_parser=json.loads,
                json_output=True,
            )
            run_dir = Path(result.run_dir)
            self.assertEqual((run_dir / "system_prompt.txt").read_text(), "system content")
            self.assertEqual((run_dir / "user_prompt.txt").read_text(), "user content")
            self.assertEqual((run_dir / "response_text.txt").read_text(), response.content)
            self.assertEqual(json.loads((run_dir / "messages.json").read_text())["messages"][1]["content"], "user content")
            audit = json.loads((run_dir / "audit.json").read_text())
            self.assertEqual(audit["status"], "ok")
            self.assertEqual(audit["usage"]["prompt_tokens"], 11)
            self.assertEqual(audit["usage"]["completion_tokens"], 7)
            self.assertEqual(audit["token_source"], "api")
            self.assertEqual(audit["finish_reason"], "stop")
            self.assertTrue(json.loads((run_dir / "request.json").read_text())["request_options"]["json_output"])
            self.assertTrue(invoke_mock.call_args.kwargs["json_output"])
            self.assertNotIn("not-logged", (run_dir / "request.json").read_text())

    def test_reasoning_tokens_support_openai_completion_details(self) -> None:
        usage = TokenUsage.from_payload(
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
            prompt_text="ignored",
            completion_text="ignored",
        )
        self.assertEqual(usage.reasoning_tokens, 7)

    def test_length_finish_reason_is_audited_as_truncated(self) -> None:
        response = LLMTextResponse(
            provider="openai_compatible",
            model="mock-model",
            content='{"partial": true}',
            raw_payload={"choices": [{"finish_reason": "length"}]},
            usage=TokenUsage(prompt_tokens=5, completion_tokens=128, total_tokens=133),
            elapsed_s=0.5,
            finish_reason="length",
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            AgentModelConfig, "resolve", return_value=_resolved()
        ), patch("sparsene.agent.runtime._invoke_llm", return_value=response):
            result = invoke_json_agent(
                agent_name="audit",
                subject="matrix",
                log_dir=tmp,
                model_config=AgentModelConfig(),
                system_prompt="return json",
                user_prompt="return json",
                request_payload={},
                response_parser=json.loads,
                json_output=True,
            )
            audit = json.loads((Path(result.run_dir) / "audit.json").read_text())
            self.assertEqual(audit["status"], "truncated")
            self.assertEqual(result.finish_reason, "length")

    def test_failed_request_still_records_input_and_estimated_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            AgentModelConfig, "resolve", return_value=_resolved()
        ), patch("sparsene.agent.runtime._invoke_llm", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                invoke_json_agent(
                    agent_name="audit",
                    subject="matrix",
                    log_dir=tmp,
                    model_config=AgentModelConfig(),
                    system_prompt="system content",
                    user_prompt="user content",
                    request_payload={"matrix": "matrix"},
                    response_parser=json.loads,
                )
            run_dir = next(Path(tmp).iterdir())
            audit = json.loads((run_dir / "audit.json").read_text())
            self.assertEqual(audit["status"], "failed")
            self.assertGreater(audit["usage"]["prompt_tokens"], 0)
            self.assertEqual(audit["usage"]["completion_tokens"], 0)
            self.assertTrue((run_dir / "messages.json").exists())


if __name__ == "__main__":
    unittest.main()
