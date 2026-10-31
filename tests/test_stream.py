"""Offline regressions for transport completion, decoding, and deadlines."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from curl_cffi.requests import CurlWsFlag

from copilot.driver import Copilot
from copilot.protocol import encode


def reply_frame(text):
    record = {"type": 1, "target": "update", "arguments": [{"writeAtCursor": text}]}
    return (json.dumps(record, ensure_ascii=False) + "\x1e").encode()


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.driver = Copilot()

    def test_partial_answer_does_not_hide_connection_failure(self):
        with patch.object(self.driver, "_recv_frame", side_effect=[
            reply_frame("partial"), ConnectionError("closed"),
        ]):
            stream = self.driver._read_stream(Mock(), 30)
            self.assertEqual(next(stream), "partial")
            with self.assertRaisesRegex(ConnectionError, "before completion"):
                next(stream)

    def test_empty_frame_is_not_success_or_busy_loop(self):
        with patch.object(self.driver, "_recv_frame", return_value=b"") as receive:
            with self.assertRaises(ConnectionError):
                list(self.driver._read_stream(Mock(), 30))
            receive.assert_called_once()

    def test_utf8_split_between_frames_is_preserved(self):
        frame = reply_frame("你好")
        split = frame.index("你".encode()) + 1
        with patch.object(self.driver, "_recv_frame", side_effect=[
            frame[:split], frame[split:], encode({"type": 3}),
        ]):
            self.assertEqual(list(self.driver._read_stream(Mock(), 30)), ["你好"])

    def test_records_after_completion_are_ignored(self):
        frame = reply_frame("done") + encode({"type": 3}) + reply_frame("unexpected")
        with patch.object(self.driver, "_recv_frame", return_value=frame):
            self.assertEqual(list(self.driver._read_stream(Mock(), 30)), ["done"])

    def test_continuous_frames_still_obey_overall_timeout(self):
        with (
            patch("copilot.driver.time.monotonic", side_effect=[0, 0, 0, 11]),
            patch.object(self.driver, "_recv_frame", return_value=encode({"type": 6})),
        ):
            with self.assertRaisesRegex(TimeoutError, "exceeded 10s"):
                list(self.driver._read_stream(Mock(), 10))

    def test_idle_timeout_is_reported_without_chat_content(self):
        with (
            patch("copilot.driver.time.monotonic", return_value=0),
            patch.object(self.driver, "_recv_frame", return_value=None),
        ):
            with self.assertRaisesRegex(TimeoutError, "silent for 5s"):
                list(self.driver._read_stream(Mock(), 30, idle_timeout=5))

    def test_continuous_fragments_obey_deadline(self):
        socket = Mock()
        socket.curl.getinfo.return_value = 1
        socket.recv_fragment.return_value = (
            b"fragment", SimpleNamespace(bytesleft=1, flags=CurlWsFlag.TEXT),
        )
        with patch("copilot.driver.time.monotonic", side_effect=[0, 11]):
            self.assertIsNone(self.driver._recv_frame(socket, deadline=10))
        socket.recv_fragment.assert_called_once()

    def test_expired_deadline_does_not_read_socket(self):
        socket = Mock()
        socket.curl.getinfo.return_value = 1
        with patch("copilot.driver.time.monotonic", return_value=11):
            self.assertIsNone(self.driver._recv_frame(socket, deadline=10))
        socket.recv_fragment.assert_not_called()

    def test_unexpected_programming_error_is_not_swallowed(self):
        with patch.object(self.driver, "_recv_frame", side_effect=ValueError("invalid")):
            with self.assertRaises(ValueError):
                list(self.driver._read_stream(Mock(), 30))
