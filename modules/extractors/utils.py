"""
Utility functions for hashing, ID generation, normalization, and SSRF security.
"""
from __future__ import annotations

import hashlib
import io
import ipaddress
import os
import re
import socket
from typing import Optional, Union
from urllib.parse import urlparse

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    pytesseract = None
    HAS_PYTESSERACT = False

def generate_document_id(source_type: str, canonical_source: str, content_bytes: Optional[bytes] = None) -> str:
    """
    Deterministic document ID. Re-ingesting an unchanged file or URL yields
    the identical hash, preventing duplicate index bloat.
    """
    h = hashlib.sha256()
    h.update(source_type.encode("utf-8"))
    h.update(canonical_source.strip().lower().encode("utf-8"))
    if content_bytes:
        h.update(content_bytes)
    return f"doc_{h.hexdigest()[:16]}"


def generate_block_id(
    document_id: str, element_type: str, page_number: Optional[int], reading_order: int, text: str
) -> str:
    """Deterministic ID for a single content block."""
    h = hashlib.sha256()
    h.update(f"{document_id}:{element_type}:{page_number or 0}:{reading_order}:{text.strip()[:64]}".encode("utf-8"))
    return f"blk_{h.hexdigest()[:16]}"


def normalize_text(text: str) -> str:
    """
    Clean text without losing scientific notation, math symbols, or punctuation.
    """
    if not text:
        return ""
    # Strip zero-width & non-breaking spaces
    t = text.replace("\u00a0", " ").replace("\u200b", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.splitlines()]
    return "\n".join(lines).strip()


def validate_and_guard_url(url: str, block_private: bool = True) -> str:
    """
    Protects against Server-Side Request Forgery (SSRF).
    Blocks requests to localhost, internal LANs, and cloud metadata endpoints (169.254.169.254).
    """
    p = urlparse(url.strip())
    if p.scheme.lower() not in ("http", "https"):
        raise ValueError(f"Disallowed scheme '{p.scheme}'. Only HTTP and HTTPS are permitted.")

    host = p.hostname
    if not host:
        raise ValueError("URL does not contain a valid hostname.")

    if block_private:
        try:
            for item in socket.getaddrinfo(host, None):
                ip = ipaddress.ip_address(item[4][0])
                if (
                    ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast
                ):
                    raise PermissionError(
                        f"SSRF Security Violation: Host '{host}' resolves to restricted IP '{ip}'."
                    )
        except socket.gaierror as e:
            raise ValueError(f"Could not resolve host '{host}': {str(e)}")

    return p.geturl()


def detect_source_type(source: Union[str, bytes, io.BytesIO], filename: Optional[str] = None) -> str:
    """Infers source format from URL pattern, file extension, or magic bytes."""
    if isinstance(source, str):
        s = source.strip()
        if re.search(r"(?:youtube\.com\/(?:watch\?|shorts\/|embed\/)|youtu\.be\/)", s, re.I):
            return "youtube"
        if s.startswith(("http://", "https://")):
            return "webpage"
        if os.path.isfile(s):
            ext = os.path.splitext(s)[1].lower()
            if ext == ".pdf":
                return "pdf"
            if ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"):
                return "image"

    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return "pdf"
        if ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"):
            return "image"

    # Magic bytes check
    header = b""
    if isinstance(source, bytes):
        header = source[:16]
    elif hasattr(source, "read") and hasattr(source, "seek"):
        pos = source.tell()
        header = source.read(16)
        source.seek(pos)

    if header.startswith(b"%PDF"):
        return "pdf"
    if header.startswith((b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"II*\x00", b"MM\x00*")):
        return "image"

    return "unknown"


def _autodetect_tesseract():
    """Autodetects common Windows Tesseract installation directories."""
    if not HAS_PYTESSERACT:
        return
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            break


_autodetect_tesseract()

