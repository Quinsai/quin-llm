"""OpenAI-compatible teacher API client.

Only the Chat Completions endpoint is used.  The client is deliberately
synchronous; ``data_generator`` runs it behind a thread pool.  Per-thread
``requests.Session`` objects avoid sharing a connection pool across threads.

The API key is resolved from ``TeacherConfig``: a literal ``api_key`` in the
YAML, a file named by ``api_key_file``, or the environment variable named by
``api_key_env`` (in that order).  It is never written to logs or artifacts.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

import requests

from config import TeacherConfig

from .text_utils import canonical_json, hash_text


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_ENV_REF_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _expand_env(value: str | None) -> str:
    """Expand ``${VAR}``; unresolved references become empty."""
    expanded = os.path.expandvars(value or "")
    return "" if _ENV_REF_RE.search(expanded) else expanded


def resolve_api_key(cfg: TeacherConfig) -> str:
    """Resolve the teacher API key from config, key file, or environment.

    Precedence is ``api_key`` > ``api_key_file`` > ``api_key_env``.  A
    configured file that is missing or empty is an error rather than a silent
    fallback, so a typo cannot turn into an unauthenticated run.
    """
    direct = _expand_env(cfg.api_key).strip()
    if direct:
        return direct
    if cfg.api_key_file:
        path = Path(_expand_env(cfg.api_key_file)).expanduser()
        if not path.exists():
            raise ValueError(f"teacher api_key_file not found: {path}")
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"teacher api_key_file is empty: {path}")
        return key
    env_name = (cfg.api_key_env or "").strip()
    return os.environ.get(env_name, "").strip() if env_name else ""


def api_key_source(cfg: TeacherConfig) -> str:
    """Human-readable key source label, without exposing the key itself."""
    if _expand_env(cfg.api_key).strip():
        return "config:api_key"
    if cfg.api_key_file:
        return f"file:{_expand_env(cfg.api_key_file)}"
    if cfg.api_key_env:
        return f"env:{cfg.api_key_env}"
    return "none"


class TeacherAPIError(RuntimeError):
    """A non-retryable teacher API failure."""


class TeacherThinkingError(TeacherAPIError):
    """The endpoint returned reasoning content but no final answer."""


class _RateLimiter:
    def __init__(self, requests_per_minute: int | None) -> None:
        rpm = float(requests_per_minute or 0)
        self.min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next_time - now
            self._next_time = max(now, self._next_time) + self.min_interval
        if delay > 0:
            time.sleep(delay)


class TeacherClient:
    def __init__(
        self,
        cfg: TeacherConfig,
        session_factory: Callable[[], Any] | None = None,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.base_url = _expand_env(cfg.base_url).rstrip("/")
        self.model = _expand_env(cfg.model)
        self.api_key = resolve_api_key(cfg)
        if not self.base_url:
            raise ValueError(
                "teacher base_url is empty (set base_url or TEACHER_BASE_URL)"
            )
        if not self.model:
            raise ValueError("teacher model is empty (set model or TEACHER_MODEL)")
        if not self.api_key and (cfg.api_key or cfg.api_key_file or cfg.api_key_env):
            raise ValueError(
                "teacher API key is empty; set api_key/api_key_file in the "
                "teacher YAML or the configured environment variable"
            )

        self.url = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        self.session_factory = session_factory or requests.Session
        self._sleep = sleep_func
        self._rate_limiter = _RateLimiter(cfg.requests_per_minute)
        self._local = threading.local()

    # ------------------------------------------------------------------
    # request construction
    # ------------------------------------------------------------------
    def _session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = self.session_factory()
            self._local.session = session
        return session

    def generation_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.response_format:
            params["response_format"] = {"type": self.cfg.response_format}
        if self.cfg.seed is not None:
            params["seed"] = self.cfg.seed
        if self.cfg.extra_body:
            params.update(self.cfg.extra_body)
        return params

    def build_payload(
        self, messages: list[dict[str, str]], **overrides: Any
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        payload.update(self.generation_params())
        payload.update(overrides)
        return payload

    def request_id(
        self, task_id: str, prompt_version: str, params: dict[str, Any] | None = None
    ) -> str:
        identity = {
            "task_id": task_id,
            "prompt_version": prompt_version,
            "model": self.model,
            "params": params if params is not None else self.generation_params(),
        }
        return hash_text(canonical_json(identity))

    def describe(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "params": self.generation_params(),
        }

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = None
        if response is not None:
            retry_after = getattr(response, "headers", {}).get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
        base = float(self.cfg.retry_backoff) ** attempt
        return base + random.uniform(0.0, 0.5)

    def chat(
        self, messages: list[dict[str, str]], **overrides: Any
    ) -> dict[str, Any]:
        payload = self.build_payload(messages, **overrides)
        attempts = max(0, int(self.cfg.max_retries))
        last_error: Exception | None = None

        for attempt in range(attempts + 1):
            self._rate_limiter.wait()
            try:
                response = self._session().post(
                    self.url,
                    headers=self._headers(),
                    json=payload,
                    timeout=float(self.cfg.timeout),
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= attempts:
                    raise TeacherAPIError(
                        f"teacher request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                self._sleep(self._retry_delay(None, attempt))
                continue

            status = int(getattr(response, "status_code", 0))
            if status == 200:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise TeacherAPIError(
                        f"teacher returned non-JSON response: {response.text[:300]!r}"
                    ) from exc
                if isinstance(data, dict) and data.get("error"):
                    raise TeacherAPIError(f"teacher API error: {data['error']}")
                self._validate_response(data)
                return data

            if status in RETRYABLE_STATUS and attempt < attempts:
                last_error = TeacherAPIError(
                    f"HTTP {status}: {getattr(response, 'text', '')[:300]}"
                )
                self._sleep(self._retry_delay(response, attempt))
                continue

            raise TeacherAPIError(
                f"teacher HTTP {status}: {getattr(response, 'text', '')[:500]}"
            )

        raise TeacherAPIError(f"teacher request failed: {last_error}")

    @staticmethod
    def _message_content(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            content = "".join(parts)
        return content if isinstance(content, str) else ""

    def _validate_response(self, data: dict[str, Any]) -> None:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise TeacherAPIError(f"teacher response has no choices: {str(data)[:300]}")
        message = choices[0].get("message") or {}
        content = self._message_content(message)
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if not content.strip():
            if reasoning:
                raise TeacherThinkingError(
                    "teacher returned reasoning_content but empty content; "
                    "disable thinking mode (e.g. extra_body.enable_thinking: false)"
                )
            raise TeacherAPIError("teacher returned an empty message content")

    def extract_content(self, data: dict[str, Any]) -> str:
        self._validate_response(data)
        return self._message_content(data["choices"][0]["message"]).strip()

    def finish_reason(self, data: dict[str, Any]) -> str | None:
        choices = data.get("choices") or []
        return choices[0].get("finish_reason") if choices else None

    def usage(self, data: dict[str, Any]) -> dict[str, int]:
        raw = data.get("usage") or {}
        return {
            "prompt_tokens": int(raw.get("prompt_tokens") or 0),
            "completion_tokens": int(raw.get("completion_tokens") or 0),
            "total_tokens": int(raw.get("total_tokens") or 0),
        }

    def ping(self) -> str:
        data = self.chat(
            [{"role": "user", "content": "Reply with the single word OK."}],
            temperature=0.0,
            max_tokens=8,
        )
        return self.extract_content(data)
