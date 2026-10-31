"""Offline: reference-image upload, chat-frame annotations and the edits API."""

import base64
import json
import unittest
from pathlib import Path
from unittest import mock

from copilot import driver as drv
from copilot.protocol import MULTI_IMAGE_OPTION, chat_frame, upload_form

FIXTURE = Path(__file__).parent / "fixtures" / "upload_response.json"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._p


class _Session:
    calls = []

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def post(self, url, data=None, headers=None):
        _Session.calls.append((url, data, headers))
        return _Resp(200, json.loads(FIXTURE.read_text(encoding="utf-8")))


class UploadTests(unittest.TestCase):
    def test_upload_form_fields(self):
        body, ctype = upload_form("data:image/png;base64,AAAA", "conv-1")
        self.assertTrue(ctype.startswith("multipart/form-data; boundary="))
        text = body.decode()
        for needle in ('name="scenario"\r\n\r\nUploadImage', 'name="conversationId"\r\n\r\nconv-1',
                       'name="FileBase64"\r\n\r\ndata:image/png;base64,AAAA', MULTI_IMAGE_OPTION):
            self.assertIn(needle, text)

    def test_upload_image_returns_annotation(self):
        _Session.calls = []
        with mock.patch.object(drv, "Session", _Session):
            ann = drv.Copilot.upload_image(PNG, access_token="tok", oid="oid-1", conversation_id="c")
        self.assertEqual(ann["id"], "0-ea-d11-1639bd9805fc32cebbdd999c4414027b")
        self.assertEqual(ann["messageAnnotationType"], "ImageFile")
        self.assertEqual(ann["messageAnnotationMetadata"]["fileType"], "png")
        url, body, headers = _Session.calls[0]
        self.assertTrue(url.endswith("/m365Copilot/UploadFile"))
        self.assertEqual(headers["Authorization"], "Bearer tok")
        self.assertIn(b"data:image/png;base64,", body)

    def test_upload_failure_raises(self):
        class Bad(_Session):
            def post(self, *a, **k):
                return _Resp(200, {"result": {"value": "Failure", "message": "nope"}})
        with mock.patch.object(drv, "Session", Bad):
            with self.assertRaises(RuntimeError):
                drv.Copilot.upload_image(PNG, access_token="t", oid="o")

    def test_unsupported_type_rejected(self):
        with self.assertRaises(ValueError):
            drv.Copilot.upload_image((b"%PDF", "application/pdf"), access_token="t", oid="o")

    def test_chat_frame_annotations(self):
        ann = [{"id": "d1", "messageAnnotationType": "ImageFile"}]
        arg = chat_frame("hi", "r", "s", True, ann)["arguments"][0]
        self.assertEqual(arg["message"]["messageAnnotations"], ann)
        self.assertIn(MULTI_IMAGE_OPTION, arg["optionsSets"])
        plain = chat_frame("hi", "r", "s", True)["arguments"][0]
        self.assertNotIn("messageAnnotations", plain["message"])
        self.assertNotIn(MULTI_IMAGE_OPTION, plain["optionsSets"])


class ServerImageInputTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from copilot.client import ChatReply
        from copilot.models import ImageResponse
        import server.api as api
        self.api = api
        self.http = TestClient(api.app)
        reply = ChatReply("", "c1", [ImageResponse(url="https://example.com/x.png", prompt="", metadata={})])
        self.chat = mock.patch.object(api.client, "chat", return_value=reply).start()
        self.addCleanup(mock.patch.stopall)
        api._rate_limiter = api.TokenBucket(0, 1)  # disabled

    def test_chat_completions_image_url_part(self):
        data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        r = self.http.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": [
            {"type": "text", "text": "改成水彩"}, {"type": "image_url", "image_url": {"url": data_url}}]}]})
        self.assertEqual(r.status_code, 200, r.text)
        imgs = self.chat.call_args.kwargs["images"]
        self.assertEqual(imgs, [(PNG, "image/png")])

    def test_edits_multipart(self):
        r = self.http.post("/v1/images/edits", data={"prompt": "水彩风格"},
                           files=[("image[]", ("a.png", PNG, "image/png")), ("image[]", ("b.png", PNG, "image/png"))])
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["data"][0]["url"], "https://example.com/x.png")
        self.assertEqual(len(self.chat.call_args.kwargs["images"]), 2)
        self.assertIn("水彩风格", self.chat.call_args.args[0])

    def test_edits_json_base64(self):
        r = self.http.post("/v1/images/edits", json={"prompt": "p", "image": base64.b64encode(PNG).decode()})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.chat.call_args.kwargs["images"][0][1], "image/png")

    def test_edits_requires_image_and_limit(self):
        self.assertEqual(self.http.post("/v1/images/edits", json={"prompt": "p"}).status_code, 400)
        many = [base64.b64encode(PNG).decode()] * (self.api.MAX_INPUT_IMAGES + 1)
        self.assertEqual(self.http.post("/v1/images/edits", json={"prompt": "p", "images": many}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
