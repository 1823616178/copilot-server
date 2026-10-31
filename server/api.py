"""FastAPI app wiring Copilot onto the OpenAI Chat Completions API."""

import base64
import math
import os
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from copilot import CopilotClient
from copilot.driver import ClearanceRequired

from .config import (MAX_INPUT_IMAGE_BYTES, MAX_INPUT_IMAGES, MODEL_NAME,
                     PROXY, RATE_LIMIT_BURST, RATE_LIMIT_RPM)
from .openai_format import (
    completion_response,
    new_id,
    sse_event,
    stream_chunk,
)
from .prompt import image_urls, messages_to_prompt
from .ratelimit import TokenBucket
from .schemas import ChatCompletionRequest, ImageGenerationRequest
from copilot.models import ImageResponse

import logging

from copilot.auth import NotSignedIn
from copilot.driver import AuthExpired

logger = logging.getLogger("copilot.server")

app = FastAPI(title="Copilot OpenAI-compatible API", version="1.0.0")
# Server must never pop a visible browser mid-request: chat and token refresh
# are plain HTTP, and if the session is gone entirely the request fails with a
# message to run `python -m copilot login` out of band.
client = CopilotClient(interactive_clear=False, proxy=PROXY)

_CLEARANCE_HELP = (
    "Copilot asked for human verification before answering. Open copilot.com in "
    "your browser, pass the check, then retry (or run `python -m copilot login`)."
)

# Self-imposed rate limit on top of the concurrency lock below: this caps
# requests-per-minute, the lock caps requests-in-flight. See server/ratelimit.py.
_rate_limiter = TokenBucket(RATE_LIMIT_RPM, RATE_LIMIT_BURST)


def _rate_limited_response():
    """Spend a token; return an OpenAI-shaped 429 if none left, else ``None``."""
    allowed, wait = _rate_limiter.try_acquire()
    if allowed:
        return None
    secs = max(1, math.ceil(wait))
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(secs)},
        content={"error": {
            "message": (
                f"Rate limit exceeded (>{RATE_LIMIT_RPM:g} req/min). "
                f"Retry in {secs}s."
            ),
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }},
    )

# Copilot's per-account chat socket doesn't tolerate concurrent conversations
# from one process (parallel requests error out or hang). This server bridges a
# single signed-in account, so we serialize upstream calls: concurrent HTTP
# requests queue here and run one at a time. Predictable, at the cost of
# parallelism — fine for a personal bridge.
_upstream_lock = threading.Lock()


def _stream(prompt: str, model: str, conversation_id=None, images=None):
    """Yield OpenAI ``chat.completion.chunk`` SSE events for ``prompt``.

    ``conversation_id`` continues an existing Copilot thread; ``None`` starts a
    fresh one (its id is emitted on the final chunk).
    """
    cid = new_id()
    created = int(time.time())
    try:
        with _upstream_lock:  # one upstream chat at a time (released on disconnect)
            yield sse_event(stream_chunk(cid, created, model, {"role": "assistant"}))
            stream = client.stream(prompt, conversation_id=conversation_id, images=images or None)
            for piece in stream:
                if isinstance(piece, str) and piece:
                    yield sse_event(stream_chunk(cid, created, model, {"content": piece}))
                elif isinstance(piece, ImageResponse):
                    md = _image_markdown(piece)
                    if md:
                        yield sse_event(stream_chunk(cid, created, model, {"content": md}))
            # Copilot's conversation id is known once the stream has run; emit it
            # on the final chunk so callers can track the upstream thread.
            yield sse_event(
                stream_chunk(
                    cid, created, model, {}, finish="stop",
                    conversation_id=stream.conversation_id,
                )
            )
    except ClearanceRequired:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_CLEARANCE_HELP}]"}, finish="error")
        )
    except Exception as exc:  # surface errors to the client instead of hanging
        logger.warning("stream failed: %s: %s", type(exc).__name__, exc)
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {exc}]"}, finish="error")
        )
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "microsoft"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    prompt = messages_to_prompt(req.messages)
    if not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no text content in messages", "type": "invalid_request_error"}},
        )
    model = req.model or MODEL_NAME
    try:
        images = _load_images(image_urls(req.messages))
    except ValueError as exc:
        return _bad_request(str(exc))

    # Enforce the per-minute ceiling before touching the upstream lock, so excess
    # callers get a fast 429 instead of piling up behind the serialized queue.
    limited = _rate_limited_response()
    if limited is not None:
        return limited

    if req.stream:
        return StreamingResponse(
            _stream(prompt, model, req.conversation_id, images), media_type="text/event-stream"
        )

    try:
        with _upstream_lock:  # serialize: one upstream chat at a time
            reply = client.chat(prompt, conversation_id=req.conversation_id, images=images or None)
    except ClearanceRequired:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": _CLEARANCE_HELP, "type": "clearance_required"}},
        )
    except (NotSignedIn, AuthExpired) as exc:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": str(exc), "type": "authentication_error"}},
        )
    except TimeoutError as exc:
        return JSONResponse(
            status_code=504,
            content={"error": {"message": str(exc), "type": "upstream_timeout"}},
        )
    except Exception as exc:
        logger.warning("chat failed: %s: %s", type(exc).__name__, exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )
    text = reply.text + "".join(_image_markdown(i) for i in reply.images)
    return completion_response(text.lstrip("\n") if not reply.text else text,
                               model, reply.conversation_id)


# data: URIs are megabytes; only inline them when explicitly allowed.
_ALLOW_DATA_URI = os.environ.get("COPILOT_IMAGE_DATA_URI", "").lower() in ("1", "true", "yes")


def _image_markdown(img: ImageResponse) -> str:
    url = img.url or ""
    if url.startswith("data:") and not _ALLOW_DATA_URI:
        return "\n\n[image generated, but only available as an inline data URI; set COPILOT_IMAGE_DATA_URI=1 to include it]\n"
    return f"\n\n![image]({url})\n" if url else ""


def _fetch_b64(url: str) -> str:
    if url.startswith("data:"):
        return url.split(",", 1)[1]
    from curl_cffi.requests import Session

    with Session(impersonate="chrome", proxy=PROXY, timeout=60) as s:
        r = s.get(url)
        r.raise_for_status()
        return base64.b64encode(r.content).decode()


def _bad_request(msg: str):
    return JSONResponse(status_code=400, content={"error": {"message": msg, "type": "invalid_request_error"}})


def _load_images(sources) -> list:
    """Turn data:/http(s) URLs, raw bytes or (bytes, mime) into ``(bytes, mime)``."""
    if len(sources) > MAX_INPUT_IMAGES:
        raise ValueError(f"too many images ({len(sources)}); at most {MAX_INPUT_IMAGES} per request")
    out = []
    for src in sources:
        if isinstance(src, tuple):
            data, mime = src
        elif isinstance(src, (bytes, bytearray)):
            data, mime = bytes(src), None
        elif isinstance(src, str) and src.startswith("data:"):
            head, _, b64 = src.partition(",")
            try:
                data = base64.b64decode(b64)
            except Exception:
                raise ValueError("invalid base64 in data: URL")
            mime = head[5:].split(";", 1)[0]
        elif isinstance(src, str) and src.startswith(("http://", "https://")):
            from curl_cffi.requests import Session
            try:
                with Session(impersonate="chrome", proxy=PROXY, timeout=60) as s:
                    r = s.get(src)
                    r.raise_for_status()
            except Exception as exc:
                raise ValueError(f"could not download image {src[:100]}: {exc}")
            data, mime = r.content, (r.headers.get("content-type") or "").split(";")[0] or None
        elif isinstance(src, str) and src:
            try:  # bare base64
                data, mime = base64.b64decode(src, validate=True), None
            except Exception:
                raise ValueError("image must be a data: URL, http(s) URL or base64 string")
        else:
            raise ValueError("empty image")
        if not data:
            raise ValueError("empty image")
        if len(data) > MAX_INPUT_IMAGE_BYTES:
            raise ValueError(f"image too large ({len(data)} bytes; limit {MAX_INPUT_IMAGE_BYTES})")
        from copilot.driver import _sniff_mime
        if not mime or not mime.startswith("image/") or mime == "application/octet-stream":
            mime = _sniff_mime(data)
        out.append((data, mime.lower()))
    return out


def _image_api(prompt: str, n, response_format, images=None):
    limited = _rate_limited_response()
    if limited is not None:
        return limited
    try:
        with _upstream_lock:
            reply = client.chat(prompt, images=images or None)
    except (NotSignedIn, AuthExpired) as exc:
        return JSONResponse(status_code=401, content={"error": {"message": str(exc), "type": "authentication_error"}})
    except Exception as exc:
        logger.warning("image generation failed: %s: %s", type(exc).__name__, exc)
        return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "upstream_error"}})
    if not reply.images:
        return JSONResponse(status_code=502, content={"error": {
            "message": f"Copilot returned no image: {reply.text[:300]}", "type": "upstream_error"}})
    data = []
    for img in reply.images[: max(1, n or 1)]:
        if response_format == "b64_json":
            try:
                data.append({"b64_json": _fetch_b64(img.url), "revised_prompt": img.prompt or None})
            except Exception as exc:
                return JSONResponse(status_code=502, content={"error": {"message": f"image download failed: {exc}", "type": "upstream_error"}})
        else:
            data.append({"url": img.url, "revised_prompt": img.prompt or None})
    return {"created": int(time.time()), "data": data}


@app.post("/v1/images/generations")
def images_generations(req: ImageGenerationRequest):
    """OpenAI Images API shim: asks Copilot to draw ``prompt``."""
    return _image_api(f"请生成一张图片：{req.prompt}", req.n, req.response_format)


EDIT_PROMPT = "请参考我上传的图片，按以下要求生成一张新图片：{prompt}"


@app.post("/v1/images/edits")
async def images_edits(request: Request):
    """OpenAI Images edit shim: reference image(s) + prompt -> new image.

    Accepts OpenAI's multipart form (``image`` / ``image[]`` files, ``prompt``,
    ``n``, ``response_format``) or JSON ``{"prompt", "image" | "images"}`` with
    data:/http(s) URLs or bare base64 strings. ``mask`` is ignored.
    """
    from starlette.concurrency import run_in_threadpool

    ctype = request.headers.get("content-type", "")
    sources = []
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        prompt = form.get("prompt") or ""
        n, fmt = form.get("n") or 1, form.get("response_format") or "url"
        for key in ("image", "image[]", "images", "images[]"):
            for f in form.getlist(key):
                if hasattr(f, "read"):
                    sources.append((await f.read(), getattr(f, "content_type", None)))
                elif isinstance(f, str) and f:
                    sources.append(f)
    else:
        try:
            body = await request.json()
        except Exception:
            return _bad_request("expected multipart/form-data or JSON body")
        prompt = body.get("prompt") or ""
        n, fmt = body.get("n") or 1, body.get("response_format") or "url"
        imgs = body.get("images") if body.get("images") is not None else body.get("image")
        for item in (imgs if isinstance(imgs, list) else [imgs] if imgs else []):
            if isinstance(item, dict):
                item = item.get("image_url") or item.get("url") or item.get("b64_json") or ""
                item = item.get("url", "") if isinstance(item, dict) else item
            sources.append(item)
    if not str(prompt).strip():
        return _bad_request("prompt is required")
    if not sources:
        return _bad_request("at least one image is required")
    try:
        n = int(n)
        images = await run_in_threadpool(_load_images, sources)
    except ValueError as exc:
        return _bad_request(str(exc))
    return await run_in_threadpool(_image_api, EDIT_PROMPT.format(prompt=prompt), n, fmt, images)


@app.get("/healthz")
def healthz():
    """Liveness + whether a cached token exists (no upstream call)."""
    from copilot.auth import DEFAULT_AUTH_FILE
    import os.path
    return {"status": "ok", "token_file": os.path.exists(DEFAULT_AUTH_FILE)}


@app.get("/")
def root():
    return {"service": "Copilot OpenAI-compatible API", "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/images/generations", "/v1/images/edits"]}
