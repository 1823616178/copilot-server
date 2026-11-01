"""Thinking-mode selection: request knobs -> Copilot chat frame ``tone``."""

import json
import os
import unittest
from unittest.mock import patch

from copilot.protocol import chat_frame
from copilot.models import Conversation  # noqa: F401  (import sanity)
from server.api import _stream, chat_completions, list_models
from server.reasoning import resolve_mode, resolve_tone
from server.schemas import ChatCompletionRequest
from copilot.client import ChatReply


def _tone(**kw):
    return chat_frame("hi", "r", "s", True, **kw)["arguments"][0]["tone"]


class FrameToneTests(unittest.TestCase):
    def test_default_is_auto_magic(self):
        self.assertEqual(_tone(), "Magic")

    def test_explicit_tones(self):
        self.assertEqual(_tone(tone="Chat"), "Chat")
        self.assertEqual(_tone(tone="Reasoning"), "Reasoning")

    def test_unknown_tone_falls_back(self):
        self.assertEqual(_tone(tone="Bogus"), "Magic")


class ResolveModeTests(unittest.TestCase):
    def setUp(self):
        p = patch.dict(os.environ, {}, clear=False)
        p.start(); self.addCleanup(p.stop)
        os.environ.pop("COPILOT_DEFAULT_MODE", None)

    def test_effort_levels(self):
        for effort, mode in [("none", "quick"), ("minimal", "quick"), ("low", "auto"),
                             ("medium", "auto"), ("high", "deep"), ("HIGH", "deep")]:
            self.assertEqual(resolve_mode(reasoning_effort=effort), mode, effort)

    def test_responses_style_reasoning(self):
        self.assertEqual(resolve_mode(reasoning={"effort": "high"}), "deep")

    def test_thinking(self):
        self.assertEqual(resolve_mode(thinking={"type": "enabled", "budget_tokens": 9}), "deep")
        self.assertEqual(resolve_mode(thinking={"type": "disabled"}), "quick")
        self.assertEqual(resolve_mode(thinking={"type": "auto"}), "auto")
        self.assertEqual(resolve_mode(thinking=False), "quick")

    def test_model_aliases(self):
        self.assertEqual(resolve_mode(model="copilot-think"), "deep")
        self.assertEqual(resolve_mode(model="copilot-quick"), "quick")
        self.assertEqual(resolve_mode(model="copilot"), "auto")
        self.assertEqual(resolve_mode(model="gpt-4o"), "auto")  # unknown -> default

    def test_precedence(self):
        self.assertEqual(resolve_mode(model="copilot-think", reasoning_effort="none"), "quick")
        self.assertEqual(resolve_mode(reasoning_effort="none", thinking={"type": "enabled"}), "deep")
        self.assertEqual(resolve_mode(thinking={"type": "enabled"}, copilot_mode="quick"), "quick")

    def test_env_default(self):
        with patch.dict(os.environ, {"COPILOT_DEFAULT_MODE": "deep"}):
            self.assertEqual(resolve_mode(model="copilot"), "deep")
            self.assertEqual(resolve_mode(model="copilot-quick"), "quick")

    def test_invalid(self):
        with self.assertRaises(ValueError):
            resolve_mode(reasoning_effort="extreme")
        with self.assertRaises(ValueError):
            resolve_mode(copilot_mode="slow")

    def test_tone(self):
        self.assertEqual(resolve_tone(reasoning_effort="high"), "Reasoning")


class ApiToneTests(unittest.TestCase):
    @patch("server.api._rate_limiter.try_acquire", return_value=(True, 0))
    @patch("server.api.client.chat", return_value=ChatReply("ok", "c1", []))
    def test_chat_passes_tone(self, chat, _acq):
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "q"}],
                                    reasoning_effort="high")
        body = chat_completions(req)
        self.assertEqual(chat.call_args.kwargs["tone"], "Reasoning")
        self.assertEqual(body["copilot_mode"], "deep")

    @patch("server.api._rate_limiter.try_acquire", return_value=(True, 0))
    @patch("server.api.client.chat", return_value=ChatReply("ok", "c1", []))
    def test_backward_compatible_default(self, chat, _acq):
        with patch.dict(os.environ, {"COPILOT_DEFAULT_MODE": "auto"}):
            chat_completions(ChatCompletionRequest(messages=[{"role": "user", "content": "q"}]))
        self.assertEqual(chat.call_args.kwargs["tone"], "Magic")

    @patch("server.api._rate_limiter.try_acquire", return_value=(True, 0))
    def test_bad_effort_400(self, _acq):
        r = chat_completions(ChatCompletionRequest(messages=[{"role": "user", "content": "q"}],
                                                   reasoning_effort="ultra"))
        self.assertEqual(r.status_code, 400)

    @patch("server.api.client.stream")
    def test_stream_passes_tone(self, stream):
        class S(list):
            conversation_id = "c2"
        stream.return_value = S(["a"])
        events = list(_stream("q", "copilot-quick", mode="quick"))
        self.assertEqual(stream.call_args.kwargs["tone"], "Chat")
        final = json.loads(events[-2].removeprefix("data: "))
        self.assertEqual(final["copilot_mode"], "quick")

    def test_models_listed(self):
        ids = [m["id"] for m in list_models()["data"]]
        self.assertEqual(ids[0], "copilot")
        self.assertIn("copilot-think", ids)
        self.assertIn("copilot-quick", ids)


class ReasoningContentTests(unittest.TestCase):
    def _cot(self, mid, text):
        return {"type": 1, "target": "update", "arguments": [{"messages": [{
            "author": "bot", "messageType": "Progress", "contentOrigin": "ChainOfThoughtSummary",
            "messageId": mid, "text": text}]}]}

    def test_parser_collects_cot_summary(self):
        from copilot.protocol import TurnParser
        from copilot.models import ReasoningChunk
        p = TurnParser()
        out = p.feed(self._cot("a", "**Step** one")) + p.feed(self._cot("a", "**Step** one two"))
        out += p.feed(self._cot("b", "next"))
        early = {"type": 1, "target": "update", "arguments": [{"messages": [{
            "author": "bot", "messageType": "Progress", "contentOrigin": "EarlyProgress", "text": "Hang on"}]}]}
        out += p.feed(early)
        self.assertTrue(all(isinstance(c, ReasoningChunk) for c in out))
        self.assertEqual("".join(c.text for c in out), "**Step** one two\n\nnext")
        self.assertEqual(p.text, "")

    @patch("server.api._rate_limiter.try_acquire", return_value=(True, 0))
    @patch("server.api.client.chat", return_value=ChatReply("ok", "c1", [], "thinking..."))
    def test_nonstream_reasoning_content(self, _chat, _acq):
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "q"}], model="copilot-think")
        body = chat_completions(req)
        self.assertEqual(body["choices"][0]["message"]["reasoning_content"], "thinking...")
        req.include_reasoning = False
        self.assertNotIn("reasoning_content", chat_completions(req)["choices"][0]["message"])

    @patch("server.api.client.stream")
    def test_stream_reasoning_delta(self, stream):
        from copilot.models import ReasoningChunk
        class S(list):
            conversation_id = "c"
        stream.return_value = S([ReasoningChunk("hmm"), "answer"])
        ev = [json.loads(e.removeprefix("data: ")) for e in list(_stream("q", "m", mode="deep"))[:-1]]
        deltas = [c["choices"][0]["delta"] for c in ev]
        self.assertIn({"reasoning_content": "hmm"}, deltas)
        self.assertIn({"content": "answer"}, deltas)
        stream.return_value = S([ReasoningChunk("hmm"), "answer"])
        ev = [e for e in _stream("q", "m", mode="deep", include_reasoning=False)]
        self.assertFalse(any("reasoning_content" in e for e in ev))


if __name__ == "__main__":
    unittest.main()
