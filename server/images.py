"""Bounded image decoding and downloads restricted to public destinations."""

import base64
import binascii
import ipaddress
import socket
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

from curl_cffi.const import CurlOpt
from curl_cffi.curl import CURL_WRITEFUNC_ERROR
from curl_cffi.requests import Session

from copilot.driver import _sniff_mime


def _public_address(value):
    address = ipaddress.ip_address(value)
    mapped = getattr(address, "ipv4_mapped", None)
    transitions = ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32")
    if (not address.is_global or address.is_multicast or getattr(address, "is_site_local", False)
            or (mapped and not mapped.is_global)
            or (address.version == 6 and any(address in ipaddress.ip_network(n) for n in transitions))):
        raise ValueError("image URLs must resolve only to public IP addresses")
    return address


def _resolve_url(url):
    if "\\" in url or any(ord(c) <= 32 or ord(c) == 127 for c in url):
        raise ValueError("invalid image URL")
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if (parsed.scheme not in ("http", "https") or not host or "%" in host
                or not 1 <= port <= 65535
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("image URLs must be http(s) URLs without credentials")
        host = host.encode("idna").decode("ascii")
        if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789.-_:" for c in host):
            raise ValueError("invalid image hostname")
        try:
            addresses = [str(_public_address(host))]
        except ValueError:
            # A numeric address must not be retried as a hostname.
            try:
                ipaddress.ip_address(host)
            except ValueError:
                records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
                addresses = list(dict.fromkeys(str(_public_address(r[4][0])) for r in records))
            else:
                raise
        if not addresses:
            raise ValueError("image hostname has no addresses")
    except (OSError, UnicodeError) as exc:
        raise ValueError("could not resolve image hostname") from exc
    netloc = f"[{host}]" if ":" in host else host
    canonical = urlunsplit((parsed.scheme, f"{netloc}:{port}", parsed.path or "/", parsed.query, ""))
    # Pin the validated result so libcurl cannot re-resolve a rebinding hostname.
    ips = ",".join(f"[{a}]" if ":" in a else a for a in addresses)
    return canonical, f"{netloc}:{port}:{ips}".encode()


def download_image(url, max_bytes):
    """Download at most max_bytes, validating and pinning every redirect hop.

    Downloads are direct: a proxy can resolve the hostname independently and
    bypass the validated DNS result. Copilot's authenticated traffic still uses
    the configured outbound proxy.
    """
    deadline = time.monotonic() + 60
    for _ in range(6):
        url, resolve = _resolve_url(url)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError("image download timed out")
        data = bytearray()
        too_large = False

        def receive(chunk):
            nonlocal too_large
            if len(data) + len(chunk) > max_bytes:
                too_large = True
                return CURL_WRITEFUNC_ERROR  # Abort without buffering the rest.
            data.extend(chunk)
            return len(chunk)

        try:
            with Session(impersonate="chrome", trust_env=False,
                         curl_options={CurlOpt.PROXY: b"", CurlOpt.RESOLVE: [resolve]}) as session:
                response = session.get(url, allow_redirects=False, content_callback=receive,
                                       timeout=(min(10, remaining), remaining))
                response.raise_for_status()
        except Exception as exc:
            if too_large:
                raise ValueError(f"image too large (limit {max_bytes} bytes)") from exc
            raise ValueError("could not download image") from exc
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("location")
            if not location:
                raise ValueError("image redirect has no Location header")
            url = urljoin(url, location)
            continue
        mime = (response.headers.get("content-type") or "").split(";", 1)[0] or None
        return bytes(data), mime
    raise ValueError("too many image redirects")


def _decode_base64(value, max_bytes):
    if len(value) > 4 * ((max_bytes + 2) // 3):
        raise ValueError(f"image too large (limit {max_bytes} bytes)")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 image") from exc


def load_images(sources, max_images, max_bytes):
    """Normalize bytes, (bytes, mime), data URLs and public http(s) URLs."""
    if len(sources) > max_images:
        raise ValueError(f"too many images ({len(sources)}); at most {max_images} per request")
    out = []
    for src in sources:
        if isinstance(src, tuple):
            data, mime = src
        elif isinstance(src, (bytes, bytearray)):
            data, mime = bytes(src), None
        elif isinstance(src, str) and src.startswith("data:"):
            head, sep, encoded = src.partition(",")
            if not sep or not head.lower().endswith(";base64"):
                raise ValueError("image data URL must use base64 encoding")
            data, mime = _decode_base64(encoded, max_bytes), head[5:].split(";", 1)[0]
        elif isinstance(src, str) and src.startswith(("http://", "https://")):
            data, mime = download_image(src, max_bytes)
        elif isinstance(src, str) and src:
            data, mime = _decode_base64(src, max_bytes), None
        else:
            raise ValueError("image must be a data: URL, public http(s) URL or base64 string")
        if not data:
            raise ValueError("empty image")
        if len(data) > max_bytes:
            raise ValueError(f"image too large (limit {max_bytes} bytes)")
        if not mime or not mime.lower().startswith("image/"):
            mime = _sniff_mime(data)
        out.append((data, mime.lower()))
    return out
