"""Disconnects must close upstream sockets and release the single-account lock."""

import asyncio
import socket
import unittest
from unittest.mock import Mock, patch

from curl_cffi.const import CurlECode
from curl_cffi.curl import CurlError

import server.api as api
from copilot.client import ChatStream
from server.schemas import ChatCompletionRequest


class StreamCancellationTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        with patch.object(api._rate_limiter, "try_acquire", return_value=(True, 0)):
            return api.chat_completions(ChatCompletionRequest(
                messages=[{"role": "user", "content": "q"}], stream=True))

    async def test_disconnect_interrupts_idle_socket_and_closes_it(self):
        for version in ("2.3", "2.4"):
            with self.subTest(asgi=version):
                left, right = socket.socketpair()
                self.addCleanup(left.close)
                self.addCleanup(right.close)
                reading = asyncio.Event()
                loop = asyncio.get_running_loop()
                wss = Mock()
                wss.curl.getinfo.return_value = left.fileno()

                def fragment():
                    loop.call_soon_threadsafe(reading.set)
                    raise CurlError("try again", code=CurlECode.AGAIN)

                wss.recv_fragment.side_effect = fragment
                sent = []

                async def send(message):
                    sent.append(message)

                async def receive():
                    await reading.wait()
                    return {"type": "http.disconnect"}

                with patch("copilot.driver.Session") as session, \
                        patch.object(api.client, "_fresh_auth", return_value={"access_token": "test", "oid": "test"}):
                    session.return_value.__enter__.return_value.ws_connect.return_value = wss
                    response = self.request()
                    await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": version}},
                                                    receive, send), timeout=2)
                self.assertTrue(response.cancel_event.is_set())
                wss.close.assert_called_once()
                wss.terminate.assert_called_once()
                self.assertFalse(api._upstream_lock.locked())
                self.assertFalse(any(b"[error:" in message.get("body", b"") for message in sent))

    async def test_disconnected_queued_request_never_calls_upstream(self):
        async def receive():
            await asyncio.sleep(0.02)
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        api._upstream_lock.acquire()
        try:
            with patch.object(api.client, "stream") as upstream:
                response = self.request()
                await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}},
                                                receive, send), timeout=1)
                upstream.assert_not_called()
        finally:
            api._upstream_lock.release()

    async def test_failed_send_releases_lock_before_return(self):
        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        with patch.object(api.client, "stream") as upstream:
            response = self.request()
            await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}},
                                            receive, send), timeout=1)
            upstream.assert_not_called()
        self.assertFalse(api._upstream_lock.locked())

    async def test_completed_stream_still_emits_final_chunk_and_done(self):
        async def receive():
            await asyncio.Event().wait()

        sent = []

        async def send(message):
            sent.append(message)

        def chunks():
            yield "answer"

        with patch.object(api.client, "stream", return_value=ChatStream(chunks(), "c1")):
            response = self.request()
            await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}},
                                            receive, send), timeout=1)
        body = b"".join(message.get("body", b"") for message in sent).decode()
        self.assertIn('"content": "answer"', body)
        self.assertIn('"conversation_id": "c1"', body)
        self.assertIn('"finish_reason": "stop"', body)
        self.assertIn("data: [DONE]", body)
        self.assertFalse(api._upstream_lock.locked())
