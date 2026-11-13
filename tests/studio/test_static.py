"""The studio's shipped static files: nothing fetched from anywhere (CD24), and the
placeholder page names every API route the server answers (Task 17, server half)."""

from __future__ import annotations

import html
import re
from pathlib import Path

import fukikae.studio

STATIC = Path(fukikae.studio.__file__).parent / "static"
EXTERNAL = re.compile(r"https?://|//cdn|@import\s+url\(|\bsrc=[\"']//", re.IGNORECASE)


def _files() -> list[Path]:
    return [p for p in STATIC.rglob("*") if p.is_file()]


def test_the_page_ships_in_the_package():
    assert (STATIC / "index.html").is_file()


def test_no_external_urls_in_static_files():
    for p in _files():
        text = p.read_text(encoding="utf-8", errors="replace")
        assert not EXTERNAL.search(text), f"{p.name}: {EXTERNAL.search(text)}"


def test_placeholder_page_lists_every_route():
    from fukikae.studio.server import ROUTES

    page = html.unescape((STATIC / "index.html").read_text(encoding="utf-8"))
    assert "pending" in page.lower()
    for method, path, _ in ROUTES:
        assert f"{method} {path}" in page, (method, path)
