"""The studio page in headless Chromium (marker ``browser``, deselected by default): the
fixed-height layout leaves no document overflow, the token leaves the address bar and goes
on every request as the header, Apply posts exactly the previewed decision, and a blind pair
names no version before Reveal.

Needs the ``browser`` dependency group (``uv sync --group browser``) and a Chromium headless
shell (``uv run playwright install chromium-headless-shell``); an older installed shell under
``~/.cache/ms-playwright`` is used when this playwright's own isn't there. Skipped when
playwright isn't installed."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.studio import test_audition, test_season, test_server
from tests.studio.test_audition import _draft
from tests.studio.test_server import TOKEN, Client

pytestmark = pytest.mark.browser
sync_api = pytest.importorskip("playwright.sync_api")

# the fixtures this module shares (the season workspace, the fake clone backends, the server)
ws = test_season.ws
aud_ws = test_audition.aud_ws
studio = test_server.studio
api = test_audition.api

GUARDED = re.compile(r"^/(api|audio|video|audition)(/|$)")
OVERFLOW = """() => ({
  h: document.scrollingElement.scrollHeight, w: document.scrollingElement.scrollWidth,
  ih: innerHeight, iw: innerWidth,
})"""


def _shell() -> str | None:
    """The newest Chromium headless shell playwright has installed, any revision."""
    root = Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or Path.home() / ".cache/ms-playwright")
    found = root.glob(
        "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"
    )
    shells = sorted(found, key=lambda p: int(p.parts[-3].rsplit("-", 1)[-1]))
    return str(shells[-1]) if shells else None


@pytest.fixture(scope="module")
def browser() -> Iterator[Any]:
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except sync_api.Error as e:
            if "Executable doesn't exist" not in str(e):
                raise
            shell = _shell()
            if shell is None:
                pytest.skip("no Chromium: uv run playwright install chromium-headless-shell")
            b = p.chromium.launch(executable_path=shell)
        yield b
        b.close()


class Page:
    """A page on the studio with every request and page error recorded."""

    def __init__(self, browser: Any, client: Client, width: int = 1366, height: int = 768):
        self.base = f"http://127.0.0.1:{client.port}/"
        self.ctx = browser.new_context(viewport={"width": width, "height": height})
        self.pg = self.ctx.new_page()
        self.requests: list[Any] = []
        self.errors: list[str] = []
        self.pg.on("request", lambda r: self.requests.append(r))
        self.pg.on("pageerror", lambda e: self.errors.append(str(e)))
        self.pg.on("console", lambda m: m.type == "error" and self.errors.append(m.text))

    def open(self) -> None:
        self.pg.goto(f"{self.base}#t={TOKEN}")
        self.pg.wait_for_selector(".brow.c", timeout=30_000)

    def route(self, route: str, ready: str) -> None:
        self.pg.evaluate("r => { location.hash = r; }", route)
        self.pg.wait_for_selector(ready, timeout=30_000)
        self.pg.wait_for_timeout(300)  # the screen's own fetches land

    def overflow(self) -> dict[str, int]:
        return self.pg.evaluate(OVERFLOW)

    def close(self) -> None:
        self.ctx.close()


@pytest.fixture
def page(browser, api) -> Iterator[Page]:
    p = Page(browser, api)
    yield p
    p.close()


SCREENS = [
    ("#/board", ".brow.c"),
    ("#/board/1", ".drawer"),
    ("#/voice", ".picker"),
    ("#/voice/1/en", ".lgrid"),
    ("#/episodes", ".planned"),
]


@pytest.mark.parametrize("size", [(1366, 768), (1440, 900), (1920, 1080)])
def test_no_screen_overflows_the_document(browser, api, size):
    p = Page(browser, api, *size)
    try:
        p.open()
        for route, ready in SCREENS:
            p.route(route, ready)
            o = p.overflow()
            assert o["h"] <= o["ih"] and o["w"] <= o["iw"], (size, route, o)
        assert not p.errors, p.errors
    finally:
        p.close()


def test_the_fragment_is_cleared_and_every_request_carries_the_token(page):
    page.open()
    assert page.pg.evaluate("location.hash") == ""
    assert TOKEN not in page.pg.url
    assert TOKEN not in page.pg.evaluate("JSON.stringify(Object.entries(localStorage))")
    page.route("#/board/1", ".drawer")
    page.pg.locator(".samples .clip").first.click()  # an /audio or /video fetch
    page.pg.wait_for_timeout(500)
    page.route("#/voice/1/en", ".lgrid")
    paths = [re.sub(r"^http://[^/]+", "", r.url) for r in page.requests]
    guarded = [r for r, path in zip(page.requests, paths, strict=True) if GUARDED.match(path)]
    kinds = {path.split("/")[1] for path in paths if GUARDED.match(path)}
    assert {"api", "audio"} <= kinds, kinds
    for r in guarded:
        assert r.headers.get("x-fukikae-token") == TOKEN, r.url
    assert all(TOKEN not in r.url for r in page.requests)
    assert not page.errors, page.errors


def test_apply_posts_exactly_the_previewed_decision(page):
    page.open()
    page.route("#/board/1", ".drawer")
    with page.pg.expect_response("**/api/decisions/preview") as got:
        page.pg.locator(".opt", has_text="Looks right").click()
    preview = got.value.json()
    page.pg.wait_for_selector(".pending .pv")
    with page.pg.expect_request(lambda r: r.url.endswith("/api/decisions")) as sent:
        page.pg.locator(".pending .btn.pri", has_text="Apply").click()
    assert sent.value.post_data_json == preview["decision"]
    page.pg.wait_for_selector(".pending .note.ok")
    assert not page.errors, page.errors


def test_a_merge_from_the_page_carries_both_rows_digests(page, api):
    _, doc = api.json("GET", "/api/season")
    digest = {c["id"]: c["digest"] for c in doc["characters"]}
    other = next(c["id"] for c in doc["characters"] if c["id"] != 1)
    page.open()
    page.route("#/board/1", ".drawer")
    page.pg.select_option("#merge-other", str(other))
    with page.pg.expect_request("**/api/decisions/preview") as sent:
        page.pg.locator(".ask .btn", has_text="Merge").click()
    body = sent.value.post_data_json
    assert body["op"] == "merge" and body["keep"] == 1 and body["drop"] == other
    assert body["digests"] == {"keep": digest[1], "drop": digest[other]}
    page.pg.wait_for_selector(".pending .pv")
    assert "carried no digest" not in page.pg.locator(".pending").inner_text()
    assert not page.errors, page.errors


def test_blind_compare_names_no_version_before_reveal(browser, api, aud_ws):
    _draft(aud_ws, "clonefake2")  # versions 1 and 2
    p = Page(browser, api)
    try:
        p.open()
        p.route("#/voice/1/en", ".lgrid")
        boxes = p.pg.locator(".lgrid .lh input[type=checkbox]")
        assert boxes.count() == 2
        boxes.nth(0).check()
        boxes.nth(1).check()
        p.pg.locator("input[name=cmpline]").first.check()
        p.pg.locator(".cmpbar .btn", has_text="Blind A/B").click()
        p.pg.wait_for_selector(".cmpbar .btn:has-text('Reveal')", timeout=30_000)
        sides = p.pg.locator(".cmpbar .side").all_inner_texts()
        assert [s for s in sides if s.strip().startswith(("A", "B"))], sides
        for s in sides:
            assert not re.search(r"\bv\d", s), sides
        cells = p.pg.evaluate("""() => {
          const on = document.querySelector('.lgrid .lc.on'); const out = [];
          for (let e = on.nextElementSibling; e && e.classList.contains('lv');
               e = e.nextElementSibling)
            out.push({text: e.textContent.trim(), clips: e.querySelectorAll('.clip').length});
          return out; }""")
        assert len(cells) == 2 and all(c["clips"] == 0 and c["text"] == "blind" for c in cells), (
            cells
        )
        p.pg.locator(".cmpbar .btn", has_text="Reveal").click()
        p.pg.wait_for_selector(".cmpbar .side:has-text('= v')")
        assert sorted(
            re.findall(r"= v(\d)", " ".join(p.pg.locator(".cmpbar .side").all_inner_texts()))
        ) == ["1", "2"]
        assert not p.errors, p.errors
    finally:
        p.close()
