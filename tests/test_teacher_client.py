import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import TeacherConfig
from distillation.teacher_client import (
    TeacherAPIError,
    TeacherClient,
    TeacherThinkingError,
    api_key_source,
    resolve_api_key,
)

API_KEY_ENV = "QUIN_TEST_TEACHER_KEY"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def chat_response(content="ok", finish_reason="stop", **message_extra):
    message = {"role": "assistant", "content": content}
    message.update(message_extra)
    return FakeResponse(
        payload={
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        }
    )


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self.responses.pop(0)


def make_client(session, **overrides):
    cfg_values = {
        "base_url": "https://teacher.example/v1",
        "api_key_env": API_KEY_ENV,
        "model": "qwen-test",
        "max_retries": 2,
        "retry_backoff": 1.0,
        "requests_per_minute": 0,
        "timeout": 9.0,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 128,
        "seed": 7,
        "response_format": "json_object",
        "extra_body": {"enable_thinking": False},
    }
    cfg_values.update(overrides)
    cfg = TeacherConfig(**cfg_values)
    sleeps = []
    client = TeacherClient(
        cfg,
        session_factory=lambda: session,
        sleep_func=sleeps.append,
    )
    return client, sleeps


class TeacherClientTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {API_KEY_ENV: "secret-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_payload_and_headers(self):
        session = FakeSession([chat_response()])
        client, _ = make_client(session)
        messages = [{"role": "user", "content": "hello"}]

        client.chat(messages)

        call = session.calls[0]
        self.assertEqual(call["url"], "https://teacher.example/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(call["timeout"], 9.0)
        payload = call["json"]
        self.assertEqual(payload["model"], "qwen-test")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["seed"], 7)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIs(payload["enable_thinking"], False)

    def test_base_url_with_endpoint_suffix_is_not_duplicated(self):
        session = FakeSession([chat_response()])
        client, _ = make_client(
            session, base_url="https://teacher.example/v1/chat/completions"
        )
        client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(
            session.calls[0]["url"],
            "https://teacher.example/v1/chat/completions",
        )

    def test_overrides_only_affect_that_request(self):
        session = FakeSession([chat_response(), chat_response()])
        client, _ = make_client(session)
        messages = [{"role": "user", "content": "hi"}]
        client.chat(messages, temperature=0.0, max_tokens=8)
        client.chat(messages)

        self.assertEqual(session.calls[0]["json"]["temperature"], 0.0)
        self.assertEqual(session.calls[0]["json"]["max_tokens"], 8)
        self.assertEqual(session.calls[1]["json"]["temperature"], 0.7)
        self.assertEqual(session.calls[1]["json"]["max_tokens"], 128)

    def test_request_id_is_deterministic_and_sensitive_to_identity(self):
        session = FakeSession([])
        client, _ = make_client(session)

        first = client.request_id("task-1", "v1")
        self.assertEqual(first, client.request_id("task-1", "v1"))
        self.assertNotEqual(first, client.request_id("task-2", "v1"))
        self.assertNotEqual(first, client.request_id("task-1", "v2"))
        self.assertNotEqual(
            first, client.request_id("task-1", "v1", {"temperature": 0.1})
        )
        self.assertNotIn("secret-key", first)

    def test_retries_retryable_status_and_honours_retry_after(self):
        session = FakeSession(
            [
                FakeResponse(status_code=429, text="slow down", headers={"Retry-After": "0"}),
                chat_response("recovered"),
            ]
        )
        client, sleeps = make_client(session)

        data = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(len(session.calls), 2)
        self.assertEqual(sleeps, [0.0])
        self.assertEqual(client.extract_content(data), "recovered")

    def test_retries_connection_errors(self):
        session = FakeSession([RuntimeError("boom"), chat_response("ok")])

        def post(*args, **kwargs):
            item = session.responses.pop(0)
            session.calls.append({"url": args[0]})
            if isinstance(item, Exception):
                import requests

                raise requests.ConnectionError(str(item))
            return item

        session.post = post
        client, sleeps = make_client(session)
        data = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(client.extract_content(data), "ok")
        self.assertEqual(len(sleeps), 1)

    def test_non_retryable_status_fails_fast(self):
        session = FakeSession([FakeResponse(status_code=400, text="bad request")])
        client, sleeps = make_client(session)
        with self.assertRaises(TeacherAPIError):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(sleeps, [])

    def test_retry_exhaustion_raises(self):
        session = FakeSession(
            [
                FakeResponse(status_code=503, text="unavailable"),
                FakeResponse(status_code=503, text="unavailable"),
                FakeResponse(status_code=503, text="unavailable"),
            ]
        )
        client, sleeps = make_client(session, max_retries=2)
        with self.assertRaises(TeacherAPIError):
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(sleeps), 2)

    def test_thinking_mode_is_reported(self):
        session = FakeSession([chat_response("", reasoning_content="let me think...")])
        client, _ = make_client(session)
        with self.assertRaises(TeacherThinkingError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_empty_content_without_reasoning_is_an_error(self):
        session = FakeSession([chat_response("   ")])
        client, _ = make_client(session)
        with self.assertRaises(TeacherAPIError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertNotIsInstance(ctx.exception, TeacherThinkingError)

    def test_structured_content_parts_are_joined(self):
        session = FakeSession(
            [
                chat_response(
                    [
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ]
                )
            ]
        )
        client, _ = make_client(session)
        data = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(client.extract_content(data), "hello world")

    def test_finish_reason_and_usage(self):
        data = chat_response("done", finish_reason="length").json()
        client, _ = make_client(FakeSession([]))
        self.assertEqual(client.finish_reason(data), "length")
        self.assertEqual(
            client.usage(data),
            {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        )

    def test_missing_api_key_fails_loudly(self):
        with mock.patch.dict(os.environ, {API_KEY_ENV: ""}):
            with self.assertRaises(ValueError):
                make_client(FakeSession([]))

    def test_api_key_from_config_is_used(self):
        with mock.patch.dict(os.environ, {API_KEY_ENV: ""}):
            session = FakeSession([chat_response()])
            client, _ = make_client(session, api_key="config-key")
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"], "Bearer config-key"
        )
        self.assertEqual(api_key_source(client.cfg), "config:api_key")

    def test_api_key_file_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "teacher_key.txt"
            key_path.write_text("file-key\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {API_KEY_ENV: ""}):
                session = FakeSession([chat_response()])
                client, _ = make_client(
                    session, api_key_file=str(key_path)
                )
                client.chat([{"role": "user", "content": "hi"}])
            self.assertEqual(
                session.calls[0]["headers"]["Authorization"], "Bearer file-key"
            )
            self.assertEqual(
                api_key_source(client.cfg), f"file:{key_path}"
            )

    def test_literal_api_key_takes_precedence_over_file_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "teacher_key.txt"
            key_path.write_text("file-key\n", encoding="utf-8")
            cfg = TeacherConfig(
                base_url="https://teacher.example/v1",
                model="qwen-test",
                api_key="config-key",
                api_key_file=str(key_path),
                api_key_env=API_KEY_ENV,
            )
            self.assertEqual(resolve_api_key(cfg), "config-key")
            self.assertEqual(api_key_source(cfg), "config:api_key")

    def test_missing_api_key_file_fails_loudly(self):
        with mock.patch.dict(os.environ, {API_KEY_ENV: ""}):
            with self.assertRaises(ValueError) as ctx:
                make_client(
                    FakeSession([]),
                    api_key_file=str(Path(tempfile.gettempdir()) / "no-such-key.txt"),
                )
            self.assertIn("api_key_file", str(ctx.exception))

    def test_unresolved_base_url_reference_fails_loudly(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUIN_UNSET_BASE_URL", None)
            with self.assertRaises(ValueError) as ctx:
                make_client(FakeSession([]), base_url="${QUIN_UNSET_BASE_URL}")
            self.assertIn("base_url", str(ctx.exception))

    def test_unresolved_model_reference_fails_loudly(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QUIN_UNSET_MODEL", None)
            with self.assertRaises(ValueError) as ctx:
                make_client(FakeSession([]), model="${QUIN_UNSET_MODEL}")
            self.assertIn("model", str(ctx.exception))

    def test_describe_exposes_no_secret(self):
        client, _ = make_client(FakeSession([]))
        described = client.describe()
        self.assertEqual(described["model"], "qwen-test")
        self.assertNotIn("secret-key", json.dumps(described))


if __name__ == "__main__":
    unittest.main()
