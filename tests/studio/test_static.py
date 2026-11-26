"""The studio page's shipped static files (Task 17 to 20, page half): nothing fetched from
anywhere (CD24), no inline script or style (the CSP), the vendored Preact and htm exactly as
recorded, a control for every identity decision, only the routes the server answers, and no
voice that isn't a clone (D41)."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

import fukikae.studio

STATIC = Path(fukikae.studio.__file__).parent / "static"
VENDOR = STATIC / "vendor"
EXTERNAL = re.compile(r"https?://|//cdn|@import\s+url\(|\bsrc=[\"']//|\burl\(\s*[\"']?//", re.I)
# XML namespace names inside Preact (createElementNS arguments): identifiers, never fetched
NAMESPACES = (
    "http://www.w3.org/2000/svg",
    "http://www.w3.org/1998/Math/MathML",
    "http://www.w3.org/1999/xhtml",
)
PAGE_JS = ("app.js", "lib.js", "board.js", "voice.js", "token.js")


def _files() -> list[Path]:
    return [p for p in STATIC.rglob("*") if p.is_file()]


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _page_js() -> str:
    return "\n".join(_text(STATIC / n) for n in PAGE_JS)


def test_the_page_ships_in_the_package():
    for name in ("index.html", "app.css", *PAGE_JS):
        assert (STATIC / name).is_file(), name


def test_no_external_urls_in_static_files():
    for p in _files():
        text = _text(p)
        if p.parent == VENDOR:
            for ns in NAMESPACES:
                text = text.replace(ns, "")
        assert not EXTERNAL.search(text), f"{p.name}: {EXTERNAL.search(text)}"


def test_no_inline_script_or_style():
    page = _text(STATIC / "index.html")
    scripts = re.findall(r"<script\b[^>]*>", page)
    assert scripts and all(re.search(r'\bsrc="/static/[\w./-]+\.js"', s) for s in scripts)
    assert re.search(r'<script type="module" src="/static/token\.js"></script>', page)
    assert re.search(r'<script type="module" src="/static/app\.js"></script>', page)
    assert "<style" not in page.lower() and not re.search(r"\sstyle=", page)
    assert not re.search(r"\son[a-z]+=", page)  # no inline event handlers
    assert '<link rel="stylesheet" href="/static/app.css">' in page
    for name in PAGE_JS:
        js = _text(STATIC / name)
        # innerHTML in any case (dangerouslySetInnerHTML too), and the other HTML sinks
        for bad in ("innerhtml", "outerhtml", "insertadjacenthtml", "<style", "eval(",
                    "new function", "document.write", "createcontextualfragment"):  # fmt: skip
            assert bad not in js.lower(), (name, bad)


def test_the_token_stays_out_of_cookies_and_queries():
    js = _text(STATIC / "token.js")
    for needed in ("location.hash", "sessionStorage", "history.replaceState",
                   "X-Fukikae-Token", "export async function api", "export function audioUrl",
                   "export async function mediaUrl", "URL.createObjectURL"):  # fmt: skip
        assert needed in js, needed
    assert "?t=" not in js and "cookie" not in js.lower()
    # the page modules (the vendored files are minified code; token.js reads #t= and &t= in
    # the fragment only)
    ours = "".join(_text(STATIC / n) for n in PAGE_JS if n != "token.js")
    ours += _text(STATIC / "index.html")
    assert "document.cookie" not in ours and 'credentials: "include"' not in ours
    assert "?t=" not in ours and "&t=" not in ours
    # media is played from blob URLs fetched with the header, never a bare src to the API
    assert not re.search(r"src=\$\{\s*[`\"']/(audio|video|audition)/", _page_js())


def _ours() -> dict[str, str]:
    """The page modules other than token.js, by name."""
    return {n: _text(STATIC / n) for n in PAGE_JS if n != "token.js"}


def test_only_token_js_fetches_or_stores_the_token():
    """token.js is the one module that calls fetch() and touches sessionStorage (where the
    spec keeps the token, this tab only); nothing writes it to localStorage."""
    token_js = _text(STATIC / "token.js")
    assert "localStorage" not in token_js
    assert "fetch(" in token_js
    for name, js in _ours().items():
        for bad in ("fetch(", "XMLHttpRequest", "sessionStorage", "EventSource", "WebSocket",
                    "sendBeacon", "fukikae-studio-token"):  # fmt: skip
            assert bad not in js, (name, bad)
        # the localStorage convenience store (lib.js `store`) never sees the token
        for m in re.finditer(r"(?:localStorage|store)\.set\w*\(([^;]*)", js):
            assert not re.search(r"\btoken\b", m.group(1)), (name, m.group(0))
    uses = [(n, m) for n, js in _ours().items() for m in re.findall(r"localStorage\.\w+", js)]
    assert uses == [("lib.js", "localStorage.getItem"), ("lib.js", "localStorage.setItem")]


def test_media_reaches_an_element_only_as_a_blob_url_from_token_js():
    """Every <audio>/<video> src is the blob URL a player acquired (lib.js `acquire`, which
    fetches through token.js `mediaUrl`); no module but token.js turns an /audio, /video or
    /audition path into a fetch or an element's src."""
    token_js = _text(STATIC / "token.js")
    assert "export async function mediaUrl" in token_js
    assert re.search(r"MEDIA = /\^\\/\(audio\|video\|audition\)", token_js)
    srcs: set[str] = set()
    for name, js in _ours().items():
        srcs |= {m.strip() for m in re.findall(r"\bsrc=\$\{([^}]*)\}", js)}
        assert not re.search(r"\.src\s*=", js), name
        assert "new Audio" not in js and "<source" not in js, name
        assert "createObjectURL" not in js, name  # blob URLs are made in token.js only
        assert "api(" not in js.replace("call(", "") or name == "lib.js", name
    assert srcs == {"url"}, srcs
    lib = _text(STATIC / "lib.js")
    # the player's url state is set only from an acquired blob URL
    assert sorted(re.findall(r"setUrl\(([^)]*)\)", lib)) == ["null", "u"]
    assert "mediaUrl(path)" in lib and "api(path)" not in lib
    assert "return mediaUrl(`/audio/" in token_js and "return mediaUrl(`/video/" in token_js


def test_the_page_never_revokes_a_url_a_mounted_player_uses():
    lib = _text(STATIC / "lib.js")
    assert lib.count("URL.revokeObjectURL") == 1
    evict = lib[lib.index("function evict()") : lib.index("export function acquire")]
    assert "e.users > 0" in evict and "URL.revokeObjectURL" in evict
    for name, js in _ours().items():
        if name != "lib.js":
            assert "revokeObjectURL" not in js, name


def test_vendored_files_match_their_recorded_hashes():
    versions = tomllib.loads(_text(VENDOR / "VERSIONS"))
    files = versions["files"]
    shipped = {p.name for p in VENDOR.iterdir() if p.is_file() and p.name != "VERSIONS"}
    assert shipped == set(files), "every vendored file is recorded, and only those"
    for name, rec in files.items():
        got = hashlib.sha256((VENDOR / name).read_bytes()).hexdigest()
        assert got == rec["sha256"], name
        assert rec["package"] in versions["packages"]
    for pkg in versions["packages"].values():
        assert re.fullmatch(r"\d+\.\d+\.\d+", pkg["version"]), "an exact pinned version"
        assert re.fullmatch(r"[0-9a-f]{64}", pkg["tarball_sha256"])
    # the page imports the vendored modules by relative path only (no bare specifier: no
    # import map, which would be an inline script)
    for name in ("preact.module.js", "hooks.module.js", "htm.module.js"):
        for spec in re.findall(r"from\s*[\"']([^\"']+)[\"']", _text(VENDOR / name)):
            assert spec.startswith("./"), (name, spec)


def test_page_imports_resolve_to_shipped_files():
    for name in PAGE_JS:
        for spec in re.findall(r"from\s*\"([^\"]+)\"", _text(STATIC / name)):
            assert spec.startswith("./"), (name, spec)
            assert (STATIC / spec).resolve().is_file(), (name, spec)


def test_page_has_a_control_for_every_identity_op():
    from fukikae.studio.model import IDENTITY_OPS

    js = _text(STATIC / "board.js")
    for op in sorted(IDENTITY_OPS):
        assert re.search(rf'op:\s*"{op}"', js), f"no control builds a {op} decision"
    assert 'by: "pitch"' in js and 'by: "cluster"' in js  # both kinds of split
    assert re.search(r'gender:\s*"f"', js) and re.search(r'gender:\s*"m"', js)
    assert re.search(r"gender:\s*null", js)  # clear
    for kind in ("character", "narrator", "ignored"):
        assert re.search(rf'kind:\s*"{kind}"', js), kind
    # each is previewed, then the previewed decision is what gets posted, exactly: the one
    # post to /api/decisions sends the preview's returned decision, untouched
    assert '"/api/decisions/preview"' in js
    posts = re.findall(r'post\(\s*"/api/decisions"\s*,\s*([^)]*)\)', js)
    assert posts == ["p.preview.decision"], posts
    assert len(re.findall(r'"/api/decisions"', _page_js())) == 1
    assert not re.search(
        r"preview\.decision\s*(?:\.\w+\s*)?=[^=]|\.\.\.\s*p\.preview\.decision", js
    )
    # a merge carries both characters' row digests from the index
    assert re.search(r"digests:\s*\{\s*keep:\s*keep\.digest,\s*drop:\s*drop\.digest\s*\}", js)
    # an accept is never posted as a raw decision
    assert 'op: "accept"' not in _page_js()


def _route_patterns() -> list[re.Pattern[str]]:
    from fukikae.studio.server import ROUTES

    out = []
    for _, path, _ in ROUTES:
        if path.startswith("/api"):
            out.append(re.compile("^" + re.sub(r"<[^>]+>", "[^/]+", path) + "$"))
    return out


def test_the_page_calls_only_routes_the_server_answers():
    js = _page_js()
    called = re.findall(r"[`\"](/api/[^`\"?]*)", js)
    assert called
    patterns = _route_patterns()
    for path in called:
        concrete = re.sub(r"\$\{[^}]*\}", "x", path)
        assert any(p.match(concrete) for p in patterns), path


def test_the_page_offers_no_voice_but_a_clone():
    """D41: no stock or designed voice anywhere in the page; the backends listed are the
    server's clone backends (`/api/backends`), and nothing assigns a card but accept."""
    js = _page_js().lower()
    # ("stock" itself appears only as a flag: a retired stock card kept and flagged)
    for bad in (
        '"stock:',
        "`stock:",
        'source: "stock"',
        "designed",
        "voices assign",
        "restock",
        "kokoro",
    ):
        assert bad not in js, bad
    assert '"/api/backends"' in _text(STATIC / "voice.js")


def test_the_pages_bands_and_floor_match_the_pipeline():
    from fukikae.cards.select import FLOOR_CLIPS, FLOOR_S, THIN_MAX_S
    from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW

    js = _text(STATIC / "lib.js")

    def const(name: str) -> float:
        m = re.search(rf"export const {name} = ([0-9.]+);", js)
        assert m, name
        return float(m.group(1))

    assert const("MALE_BELOW") == MALE_BELOW and const("FEMALE_ABOVE") == FEMALE_ABOVE
    assert const("FLOOR_CLIPS") == FLOOR_CLIPS and const("FLOOR_S") == FLOOR_S
    assert const("PROMPT_S") == THIN_MAX_S


@pytest.mark.parametrize("screen", ["episodes", "script"])
def test_planned_screens_are_placeholders_in_the_navigation(screen):
    js = _text(STATIC / "app.js")
    assert re.search(rf'id: "{screen}",[^}}]*planned:', js), screen
