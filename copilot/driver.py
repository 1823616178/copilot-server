"""Pure-HTTP Copilot driver.

Speaks copilot.com's chat protocol (a SignalR WebSocket on substrate.office.com,
see :mod:`copilot.protocol`) directly over ``curl_cffi`` — no browser required.
This is the low-level engine; most callers should use
:class:`copilot.client.CopilotClient`. The browser (:mod:`copilot.browser`) is
only used to sign in and, as a fallback, to mint a fresh token.
"""

import time
import uuid
from http.cookiejar import CookieJar
from select import select
from typing import Optional
from urllib.parse import quote, urlencode

from curl_cffi.const import CurlECode, CurlInfo
from curl_cffi.curl import CurlError
from curl_cffi.requests import CurlWsFlag, Session

from .models import AbstractProvider, Conversation, ImageType
from .protocol import (
    CHATHUB_URL, HANDSHAKE, HUB_PARAMS, MSA_TENANT, PING, VARIANTS,
    TurnParser, chat_frame, encode, split_records,
    UPLOAD_MIME_EXT, UPLOAD_URL, UPLOAD_VARIANTS, image_annotation, upload_form,
)


def _as_image(item):
    """Normalise ``bytes`` / ``(bytes, mime)`` / ``data:`` URL into ``(bytes, mime)``."""
    import base64
    if isinstance(item, dict):
        item = (item["data"], item.get("mime_type") or item.get("mimeType"))
    if isinstance(item, str):
        if not item.startswith("data:"):
            raise ValueError("image strings must be data: URLs")
        head, b64 = item.split(",", 1)
        return base64.b64decode(b64), head[5:].split(";", 1)[0]
    if isinstance(item, (bytes, bytearray)):
        return bytes(item), _sniff_mime(item)
    data, mime = item
    return bytes(data), (mime or _sniff_mime(data)).lower()


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"GIF8":
        return "image/gif"
    return "image/png"

# curl_cffi's WebSocket.recv() loops on CURLE_AGAIN forever (select() then retry)
# and never returns on an idle socket, so we drive the fragment loop ourselves to
# honour a deadline. CURL_SOCKET_BAD is libcurl's "no active socket" sentinel.
_CURL_SOCKET_BAD = -1


class ClearanceRequired(RuntimeError):
    """copilot.com refused the turn pending a human-verification check.

    The chat backend accepts signed-in turns without one today; this is raised
    if it starts rejecting them with a challenge/captcha result, so callers get
    a clear error instead of a generic failure.
    """


class AuthExpired(RuntimeError):
    """The chat socket rejected the access token (expired or revoked)."""


class Copilot(AbstractProvider):
    label = "Microsoft Copilot"
    url = "https://copilot.com"
    working = True
    supports_stream = True
    default_model = "Copilot"
    needs_auth = True  # copilot.com only chats when signed in

    def create_completion(
            self,
            prompt: str,
            stream: bool = False,
            proxy: str = None,
            timeout: int = 900,
            image: ImageType = None,
            conversation: Optional[Conversation] = None,
            conversation_id: str = None,
            return_conversation: bool = False,
            access_token: str = None,
            oid: str = None,
            tenant: str = MSA_TENANT,
            **kwargs
        ):
        """Stream a Copilot reply to ``prompt``.

        Opens the chat hub socket with the signed-in ``access_token`` (scope
        ``substrate.office.com/sydney``) for the account ``oid``, sends one
        ``chat`` invocation and yields reply text as it streams. ``prompt`` is
        sent as one user message (the protocol has no separate system/role
        channel).

        Conversation targeting (first match wins):
          * ``conversation`` — reuse an existing :class:`Conversation` object;
          * ``conversation_id`` — continue a conversation by its id string;
          * neither — start a fresh one (the client picks its id). With
            ``return_conversation`` the new :class:`Conversation` is yielded first.
        """
        images = list(kwargs.pop("images", None) or [])
        if image is not None:
            images.insert(0, image)
        anchor = kwargs.pop("anchor", None)
        if conversation is not None:
            conversation_id = conversation.conversation_id
            access_token = access_token or conversation.access_token
        if not access_token or not oid:
            raise ValueError("A signed-in access_token and account oid are required "
                             "(see copilot.auth.load_auth).")

        start_of_session = conversation_id is None
        if start_of_session:
            conversation_id = str(uuid.uuid4())
            if return_conversation:
                yield Conversation(conversation_id, CookieJar(), access_token)

        annotations = []
        for i, img in enumerate(images):
            annotations.append(self.upload_image(
                img, access_token=access_token, oid=oid, tenant=tenant, anchor=anchor,
                conversation_id=conversation_id, proxy=proxy, name=f"image{i + 1}"))

        request_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        query = {
            "chatsessionid": request_id,
            "XRoutingParameterSessionKey": oid,
            "clientrequestid": request_id,
            "X-SessionId": session_id,
            "ConversationId": conversation_id,
            "access_token": access_token,
            "variants": VARIANTS,
            **HUB_PARAMS,
        }
        websocket_url = (CHATHUB_URL.format(oid=oid, tenant=tenant) + "?"
                         + urlencode(query, quote_via=quote, safe=","))

        with Session(timeout=timeout, proxy=proxy, impersonate="chrome",
                     headers={"Origin": "https://copilot.com"}) as session:
            try:
                wss = session.ws_connect(websocket_url)
            except CurlError as exc:
                if "401" in str(exc) or "403" in str(exc):
                    raise AuthExpired(f"Copilot rejected the access token: {exc}") from exc
                raise
            wss.send(HANDSHAKE, CurlWsFlag.TEXT)
            wss.send(PING, CurlWsFlag.TEXT)
            wss.send(encode(chat_frame(prompt, request_id, session_id, start_of_session,
                                        annotations or None)),
                     CurlWsFlag.TEXT)
            yield from self._read_stream(wss, timeout)

    @staticmethod
    def upload_image(image, access_token: str, oid: str, tenant: str = MSA_TENANT,
                     anchor: Optional[str] = None, conversation_id: Optional[str] = None,
                     proxy: Optional[str] = None, name: str = "image", timeout: int = 120) -> dict:
        """Upload one reference image; return its ``messageAnnotations`` entry.

        ``image`` is ``bytes``, ``(bytes, mime)`` or a ``data:`` URL.
        """
        import base64
        data, mime = _as_image(image)
        ext = UPLOAD_MIME_EXT.get(mime)
        if not ext:
            raise ValueError(f"unsupported image type {mime!r} (use png/jpeg/webp/gif)")
        data_url = f"data:{mime};base64," + base64.b64encode(data).decode()
        body, ctype = upload_form(data_url, conversation_id or str(uuid.uuid4()))
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": ctype,
            "Origin": "https://copilot.com",
            "Referer": "https://copilot.com/",
            "X-AnchorMailbox": anchor or f"Oid:{oid}@{tenant}",
            "X-Scenario": HUB_PARAMS["scenario"],
            "X-Variants": UPLOAD_VARIANTS,
        }
        with Session(timeout=timeout, proxy=proxy, impersonate="chrome") as s:
            r = s.post(UPLOAD_URL, data=body, headers=headers)
        if r.status_code in (401, 403):
            raise AuthExpired(f"Copilot rejected the access token on upload (HTTP {r.status_code})")
        try:
            js = r.json()
        except Exception:
            js = {}
        doc_id = js.get("docId")
        if r.status_code != 200 or not doc_id or (js.get("result") or {}).get("value") not in (None, "Success"):
            msg = (js.get("result") or {}).get("message") or r.text[:300]
            raise RuntimeError(f"Copilot image upload failed (HTTP {r.status_code}): {msg}")
        return image_annotation(doc_id, f"{name}.{ext}", ext)

    def _read_stream(self, wss, timeout: int, idle_timeout: int = 90):
        """Consume hub records until the turn completes, yielding reply text.

        ``idle_timeout`` bounds how long we wait for the *next* frame, so a
        stalled socket raises instead of blocking for the full ``timeout``.
        """
        parser = TurnParser()
        buffer = b""
        overall_deadline = time.monotonic() + timeout
        while not parser.done:
            if time.monotonic() >= overall_deadline:
                raise TimeoutError(f"Copilot stream exceeded {timeout}s")
            idle_deadline = time.monotonic() + idle_timeout
            try:
                chunk = self._recv_frame(wss, min(overall_deadline, idle_deadline))
            except (CurlError, ConnectionError, OSError) as exc:
                raise ConnectionError("Copilot stream interrupted before completion") from exc
            if chunk is None:  # deadline passed with no frame
                if time.monotonic() >= overall_deadline:
                    raise TimeoutError(f"Copilot stream exceeded {timeout}s")
                raise TimeoutError(
                    f"Copilot chat socket went silent for {idle_timeout}s"
                )
            if not chunk:
                raise ConnectionError("Copilot stream closed before completion")
            # Decode only complete records: UTF-8 characters can span frames.
            buffer += chunk
            *complete, buffer = buffer.split(b"\x1e")
            for record in split_records(b"\x1e".join(complete)):
                for text in parser.feed(record):
                    yield text
                if parser.error:
                    self._raise_error(parser.error)
                if parser.done:
                    break

    @staticmethod
    def _raise_error(error) -> None:
        message = str(error)
        lowered = message.lower()
        if any(word in lowered for word in ("challenge", "captcha", "turnstile")):
            raise ClearanceRequired(f"Copilot asked for human verification: {message}")
        if any(word in lowered for word in ("unauthorized", "token", "authentication")):
            raise AuthExpired(f"Copilot rejected the access token: {message}")
        raise RuntimeError(f"Copilot error: {message}")

    @staticmethod
    def _recv_frame(wss, deadline: float):
        """Block for one complete WS frame, or return ``None`` past ``deadline``.

        Reassembles libcurl's fragments like ``curl_cffi``'s own ``recv()`` but
        breaks out of the ``CURLE_AGAIN`` wait once ``deadline`` (monotonic seconds)
        is reached, so an idle socket can't hang us indefinitely. Non-AGAIN curl
        errors (e.g. a closed connection) propagate to the caller.
        """
        sock_fd = wss.curl.getinfo(CurlInfo.ACTIVESOCKET)
        if sock_fd == _CURL_SOCKET_BAD:
            raise ConnectionError("WebSocket has no active socket")
        chunks = []
        while True:
            if time.monotonic() >= deadline:
                return None
            try:
                chunk, frame = wss.recv_fragment()
                chunks.append(chunk)
                if frame.bytesleft == 0 and frame.flags & CurlWsFlag.CONT == 0:
                    return b"".join(chunks)
            except CurlError as e:
                if e.code != CurlECode.AGAIN:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                select([sock_fd], [], [], min(0.5, remaining))
