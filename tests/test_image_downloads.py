"""Image fetches must not reach private hosts or buffer unbounded bodies."""

import ipaddress
import gzip
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from curl_cffi.const import CurlOpt

from server.images import _resolve_url, download_image, load_images


def dns(address):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (address, 443))


class DownloadPolicyTests(unittest.TestCase):
    def test_nonpublic_literals_are_rejected_before_connecting(self):
        with patch("server.images.Session") as session:
            for host in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "[::1]",
                         "[::ffff:127.0.0.1]", "224.0.0.1", "0.0.0.0", "[fec0::1]",
                         "[64:ff9b::7f00:1]", "[2002:7f00:1::]"):
                with self.subTest(host=host), self.assertRaises(ValueError):
                    download_image(f"http://{host}/image", 1024)
            session.assert_not_called()

    def test_mixed_public_and_private_dns_is_rejected(self):
        with patch("server.images.socket.getaddrinfo", return_value=[dns("8.8.8.8"), dns("10.0.0.1")]), \
                patch("server.images.Session") as session:
            with self.assertRaisesRegex(ValueError, "public"):
                download_image("https://images.example/p.png", 1024)
            session.assert_not_called()

    def test_credentials_schemes_and_ambiguous_urls_are_rejected(self):
        for url in ("https://user:pass@example.com/x", "file:///tmp/x", "https://example.com\\@127.0.0.1/x",
                    "https://[fe80::1%25eth0]/x", "https://example.com/\nx", "https://example.com:99999/x"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                _resolve_url(url)

    def test_dns_is_pinned_and_environment_proxies_are_disabled(self):
        response = SimpleNamespace(status_code=200, headers={"content-type": "image/png"},
                                   raise_for_status=lambda: None)
        with patch("server.images.socket.getaddrinfo", return_value=[dns("8.8.8.8")]) as resolve, \
                patch("server.images.Session") as session:
            def get(url, **kw):
                self.assertFalse(kw["allow_redirects"])
                kw["content_callback"](b"png")
                return response
            session.return_value.__enter__.return_value.get.side_effect = get
            self.assertEqual(download_image("https://images.example/p.png", 10), (b"png", "image/png"))
            resolve.assert_called_once()
            options = session.call_args.kwargs
            self.assertFalse(options["trust_env"])
            self.assertEqual(options["curl_options"][CurlOpt.PROXY], b"")
            self.assertEqual(options["curl_options"][CurlOpt.RESOLVE], [b"images.example:443:8.8.8.8"])

    def test_redirect_to_private_address_is_blocked(self):
        response = SimpleNamespace(status_code=302, headers={"location": "http://127.0.0.1/private"},
                                   raise_for_status=lambda: None)
        with patch("server.images.socket.getaddrinfo", return_value=[dns("8.8.8.8")]), \
                patch("server.images.Session") as session:
            session.return_value.__enter__.return_value.get.return_value = response
            with self.assertRaisesRegex(ValueError, "public"):
                download_image("https://images.example/p.png", 1024)
            self.assertEqual(session.call_count, 1)

    def test_base64_limit_is_checked_before_decoding(self):
        with patch("server.images.base64.b64decode") as decode:
            with self.assertRaisesRegex(ValueError, "too large"):
                load_images(["data:image/png;base64," + "A" * 100], 1, 8)
            decode.assert_not_called()

    def test_invalid_base64_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            load_images(["data:image/png;base64,!!!!"], 1, 1024)


class DownloadTransportTests(unittest.TestCase):
    """Exercise real libcurl against a test-only local target, never the internet."""

    @classmethod
    def setUpClass(cls):
        cls.paths = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.paths.append(self.path)
                self.send_response(302 if self.path == "/redirect" else 200)
                if self.path == "/redirect":
                    self.send_header("Location", "/small")
                if self.path == "/gzip":
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                try:
                    body = b"x" * 131072 if self.path in ("/large", "/gzip") else b"small"
                    self.wfile.write(gzip.compress(body) if self.path == "/gzip" else body)
                except (ConnectionError, OSError):
                    pass

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        # Allow only the test server through the address policy. Resolution and
        # the actual pinned libcurl connection still use the production code.
        allow = patch("server.images._public_address", side_effect=ipaddress.ip_address)
        allow.start()
        self.addCleanup(allow.stop)
        self.url = f"http://localhost:{self.server.server_port}"

    def test_download_and_relative_redirect(self):
        with patch.dict("os.environ", {"HTTP_PROXY": "http://127.0.0.1:1", "ALL_PROXY": "http://127.0.0.1:1"}):
            self.assertEqual(download_image(self.url + "/redirect", 32), (b"small", "image/png"))

    def test_large_response_aborts_with_size_error(self):
        for path in ("/large", "/gzip"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "too large"):
                download_image(self.url + path, 1024)
