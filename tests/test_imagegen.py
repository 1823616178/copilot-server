"""Offline: image-generation turns surface images, not progress text."""

import json
import unittest
from pathlib import Path

from copilot.models import ImageResponse
from copilot.protocol import TurnParser

FIXTURE = Path(__file__).parent / "fixtures" / "imagegen_turn.json"


class ImageGenTests(unittest.TestCase):
    def _run(self, records):
        p = TurnParser()
        out = []
        for r in records:
            out.extend(p.feed(r))
        return p, out

    def test_capture_yields_one_https_image_and_no_progress_text(self):
        records = json.loads(FIXTURE.read_text(encoding="utf-8"))
        p, out = self._run(records)
        self.assertTrue(p.done)
        imgs = [o for o in out if isinstance(o, ImageResponse)]
        self.assertEqual(len(imgs), 1)
        self.assertTrue(imgs[0].url.startswith("https://designerapp.officeapps.live.com/"))
        text = "".join(o for o in out if isinstance(o, str))
        self.assertNotIn("Loading image", text)
        self.assertEqual(p.images, imgs)

    def test_data_uri_fallback_when_no_https_url(self):
        card = {"author": "bot", "messageType": "InternalSearchQuery", "adaptiveCards": [
            {"type": "AdaptiveCard", "body": [{"type": "ImageSet", "images": [
                {"type": "Image", "url": "data:image/png;base64,AAAA"}]}]}]}
        records = [{"type": 1, "target": "update", "arguments": [{"messages": [card]}]},
                   {"type": 3, "invocationId": "0"}]
        p, out = self._run(records)
        self.assertEqual([i.url for i in p.images], ["data:image/png;base64,AAAA"])

    def test_text_turn_unchanged(self):
        records = [{"type": 1, "target": "update", "arguments": [{"writeAtCursor": "hi"}]},
                   {"type": 3, "invocationId": "0"}]
        p, out = self._run(records)
        self.assertEqual(out, ["hi"])
        self.assertEqual(p.images, [])


if __name__ == "__main__":
    unittest.main()
