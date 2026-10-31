"""Tests for server startup configuration."""

import json
import os
import unittest
from unittest.mock import patch

from server import app
from server.api import _rate_limited_response, _stream, chat_completions
from server.schemas import ChatCompletionRequest


class ServerStartupTests(unittest.TestCase):
    @patch("server.api._rate_limiter.try_acquire", return_value=(True, 0))
    @patch("server.api.client.chat", side_effect=ConnectionError("stream interrupted"))
    def test_interrupted_chat_returns_502(self, _chat, _acquire):
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])
        response = chat_completions(request)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(json.loads(response.body)["error"]["type"], "upstream_error")

    @patch("server.api.client.stream")
    def test_interrupted_stream_does_not_emit_success(self, stream):
        def interrupted():
            yield "partial"
            raise ConnectionError("stream interrupted")

        stream.return_value = interrupted()
        events = list(_stream("hello", "copilot"))
        chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
        finishes = [chunk["choices"][0]["finish_reason"] for chunk in chunks]
        self.assertNotIn("stop", finishes)
        self.assertEqual(finishes[-1], "error")
        self.assertEqual(events[-1], "data: [DONE]\n\n")

    @patch("server.api._rate_limiter.try_acquire", return_value=(False, 1.2))
    def test_retry_after_rounds_up(self, _acquire):
        response = _rate_limited_response()
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "2")
        self.assertIn("Retry in 2s", json.loads(response.body)["error"]["message"])

    @patch("uvicorn.run")
    @patch("copilot.auth.load_auth")
    def test_uses_default_address(self, _load_auth, run):
        with patch.dict(os.environ, {}, clear=True):
            app()

        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(run.call_args.kwargs["port"], 8000)

    @patch("uvicorn.run")
    @patch("copilot.auth.load_auth")
    def test_uses_address_from_environment(self, _load_auth, run):
        with patch.dict(os.environ, {"HOST": "0.0.0.0", "PORT": "8080"}, clear=True):
            app()

        self.assertEqual(run.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(run.call_args.kwargs["port"], 8080)

    @patch("uvicorn.run")
    @patch("copilot.auth.load_auth")
    def test_explicit_address_takes_precedence(self, _load_auth, run):
        with patch.dict(os.environ, {"HOST": "0.0.0.0", "PORT": "8080"}, clear=True):
            app(host="localhost", port=0)

        self.assertEqual(run.call_args.kwargs["host"], "localhost")
        self.assertEqual(run.call_args.kwargs["port"], 0)


if __name__ == "__main__":
    unittest.main()
