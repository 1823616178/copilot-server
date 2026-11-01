"""SSE transport that cancels the synchronous upstream on disconnect."""

import threading

import anyio
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool


class CancellableStreamingResponse(StreamingResponse):
    def __init__(self, factory, *args, **kwargs):
        self.cancel_event = threading.Event()
        self._iterator = factory(self.cancel_event)
        super().__init__(self._iterator, *args, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            # Listen even on ASGI 2.4: waiting for a failed send cannot detect
            # disconnects while the upstream is silent in a worker thread.
            async with anyio.create_task_group() as group:
                async def stream():
                    try:
                        await self.stream_response(send)
                    except OSError:
                        pass  # Client disappeared during send.
                    finally:
                        self.cancel_event.set()
                        group.cancel_scope.cancel()

                group.start_soon(stream)
                await self.listen_for_disconnect(receive)
                self.cancel_event.set()
                group.cancel_scope.cancel()
        finally:
            self.cancel_event.set()
            # The task group has joined the worker; closing a generator while
            # next() is still running would raise "generator already executing".
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
                await run_in_threadpool(self._iterator.close)
        if self.background is not None:
            await self.background()
