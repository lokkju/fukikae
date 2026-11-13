"""The studio's HTTP server: a loopback `ThreadingHTTPServer` with a token-guarded JSON API.

`StudioServer` binds 127.0.0.1 only unless the caller passes ``allow_non_loopback`` (the
command's ``--allow-non-loopback``), and then only an IPv4 literal. Every request needs the
server's per-run random token: the page load ``GET /?t=<token>`` trades it for an
``HttpOnly``, ``SameSite=Strict`` cookie scoped by port and redirects to ``/`` so the token
leaves the address bar; reads (the page, static files, ``/api`` GETs, ``/audio``) take the
cookie or the ``X-Fukikae-Token`` header; writes (every POST) take only the header, so a
form or a link on another site can't make one even with the cookie in the jar. ``Host``
must be ``127.0.0.1:<port>`` or ``localhost:<port>`` (DNS rebinding), an ``Origin``, when
sent, must be this server's, POST bodies must be JSON and at most `MAX_BODY` bytes, and the
request log never prints a query string (the token rides in one once).

The API reads the season index (`fukikae.studio.season.load_or_build`, rebuilt when its
inputs change; a decision changes them) and writes only through
`fukikae.studio.apply.apply_decision`, the studio's one writer. Audio is served only for
the clips the current index refers to (`referenced_clips`), as bare ``<sha256>.wav`` names
under ``<ws>/studio/audio``, opened without following symlinks. Static files come only from
the package's ``static/`` directory.

The page itself waits on the design pick (D47); ``static/index.html`` is a placeholder that
lists `ROUTES`.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import sqlite3
import stat
import threading
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from fukikae.registry.store import Registry
from fukikae.studio.apply import (
    DecisionError,
    DecisionRefused,
    IncompleteDecision,
    LogUnreadable,
    RegistryUnreadable,
    StaleDecision,
    StudioBusy,
    UncheckedDecision,
    UnfinishedRecords,
    apply_decision,
    check_digest,
    plan_decision,
)
from fukikae.studio.model import (
    IDENTITY_OPS,
    CharacterEntry,
    Decision,
    MergeDecision,
    SeasonDoc,
    ValidationError,
    parse_decision,
)
from fukikae.studio.season import CLIP_NAME, load_or_build, referenced_clips
from fukikae.workspace import Workspace

if TYPE_CHECKING:
    from fukikae.config import Config

DEFAULT_PORT = 8765
MAX_BODY = 1024 * 1024  # bytes; a larger POST body is 413
_DRAIN_MAX = 8 * MAX_BODY  # a too-large body up to this is read and dropped before the 413
TOKEN_HEADER = "X-Fukikae-Token"  # noqa: S105 (a header name)
STATIC_DIR = Path(__file__).parent / "static"

# (method, path, what it does): the API as the placeholder page lists it
ROUTES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/", "the page; ?t=<token> once sets the session cookie and redirects here"),
    ("GET", "/static/<file>", "the page's files, from the package only"),
    ("GET", "/api/season", "the season index (SeasonDoc), rebuilt when its inputs changed"),
    ("GET", "/api/characters/<id>", "one character's entry (CharacterEntry) from the index"),
    ("GET", "/audio/<sha256>.wav", "a clip the index refers to (a sample's clip field)"),
    (
        "POST",
        "/api/decisions/preview",
        "an identity decision's preview: what reruns, and whether its digest is stale",
    ),
    (
        "POST",
        "/api/decisions",
        "apply an identity decision; returns the rebuilt character, 409 when stale",
    ),
)

_SEGMENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_CHARACTER = re.compile(r"/api/characters/([0-9]{1,9})")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)


def check_bind(host: str, *, allow_non_loopback: bool) -> str:
    """The IPv4 address to bind for ``host``: ``localhost`` or a loopback address always;
    any other IPv4 literal only with ``allow_non_loopback``. ValueError otherwise."""
    if host == "localhost":
        return "127.0.0.1"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and ip.version == 4 and ip.is_loopback:
        return str(ip)
    if not allow_non_loopback:
        raise ValueError(
            f"the studio binds a loopback address only (127.0.0.1), not {host!r}: reach it "
            "over `ssh -L`; binding another address needs --allow-non-loopback"
        )
    if ip is None or ip.version != 4:
        raise ValueError(f"the studio binds an IPv4 address; {host!r} is not one")
    return str(ip)


def _loopback(host: str) -> bool:
    return ipaddress.ip_address(host).is_loopback


class StudioServer(ThreadingHTTPServer):
    """One workspace's studio. ``port`` 0 takes a free one (tests); ``token`` defaults to a
    fresh random one per server."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        ws: Workspace | Path,
        cfg: Config,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        token: str | None = None,
        clips: int = 6,
        allow_non_loopback: bool = False,
    ) -> None:
        bind = check_bind(host, allow_non_loopback=allow_non_loopback)
        self.ws = ws if isinstance(ws, Workspace) else Workspace(Path(ws))
        self.cfg = cfg
        self.clips = clips
        self.token = token or secrets.token_urlsafe(32)
        self._lock = threading.RLock()  # the index and every decision, one at a time
        self._doc: SeasonDoc | None = None
        super().__init__((bind, port), _Handler)
        p = self.server_address[1]
        names = ["127.0.0.1", "localhost"] + ([] if _loopback(bind) else [bind])
        self.allowed_hosts = frozenset(f"{n}:{p}" for n in names)
        self.allowed_origins = frozenset(f"http://{h}" for h in self.allowed_hosts)
        self.cookie_name = f"fukikae_studio_{p}"  # cookies ignore ports; the name doesn't

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def loopback(self) -> bool:
        return _loopback(str(self.server_address[0]))

    def url(self) -> str:
        return f"http://{self.server_address[0]}:{self.port}/?t={self.token}"

    def tunnel(self) -> str:
        return f"ssh -L {self.port}:127.0.0.1:{self.port} {socket.gethostname()}"

    def season(self) -> SeasonDoc:
        """The index as it stands now: ``season.json``, or a rebuild when its inputs changed."""
        with self._lock:
            self._doc = load_or_build(self.ws, self.cfg, clips=self.clips)
            return self._doc

    def audio_names(self) -> set[str]:
        """The clip names ``/audio`` may serve: those the last index served refers to."""
        with self._lock:
            doc = self._doc or self.season()
            return referenced_clips(doc)

    def character(self, character_id: int) -> CharacterEntry | None:
        return next((c for c in self.season().characters if c.id == character_id), None)

    def resume_command(self) -> str:
        return f"fukikae studio resume {self.ws.root}"


class _Reply(Exception):  # noqa: N818
    """Raised by a handler step to answer with a JSON error."""

    def __init__(self, status: int, error: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.body = {"error": error, "message": message, **extra}


class _Handler(BaseHTTPRequestHandler):
    server: StudioServer
    server_version = "fukikae-studio"
    sys_version = ""

    # -- logging: never the query string (the token rides in one on the page load) ----------

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if isinstance(code, HTTPStatus):
            code = code.value
        path = (self.path or "").split("?", 1)[0]
        self.log_message('"%s %s" %s %s', self.command, path, str(code), str(size))

    # -- methods ----------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("HEAD")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:  # noqa: N802
        self._dispatch("PATCH")

    def _dispatch(self, method: str) -> None:
        try:
            self._route(method)
        except _Reply as r:
            self._json(r.status, r.body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- guards -----------------------------------------------------------------------------

    def _guard(self) -> None:
        host = self.headers.get("Host", "")
        if host not in self.server.allowed_hosts:
            raise _Reply(403, "forbidden", "this request's Host is not the studio's")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            raise _Reply(403, "forbidden", "this request's Origin is not the studio's")

    def _token_ok(self, value: str | None) -> bool:
        return value is not None and hmac.compare_digest(
            value.encode("utf-8", "replace"), self.server.token.encode()
        )

    def _cookie(self) -> str | None:
        try:
            jar = SimpleCookie(self.headers.get("Cookie", ""))
        except CookieError:
            return None
        morsel = jar.get(self.server.cookie_name)
        return morsel.value if morsel is not None else None

    def _authorized(self, method: str) -> bool:
        if self._token_ok(self.headers.get(TOKEN_HEADER)):
            return True
        return method in ("GET", "HEAD") and self._token_ok(self._cookie())

    # -- routing ----------------------------------------------------------------------------

    def _route(self, method: str) -> None:
        self._guard()
        split = urlsplit(self.path)
        path = split.path
        if method == "GET" and path == "/" and "t" in parse_qs(split.query):
            given = parse_qs(split.query)["t"][0]
            if not self._token_ok(given):
                raise _Reply(403, "forbidden", "wrong token")
            self._login()
            return
        if not self._authorized(method):
            raise _Reply(403, "forbidden", f"no valid studio token (send {TOKEN_HEADER})")
        get = method in ("GET", "HEAD")
        if path == "/":
            self._need(method, get, "GET")
            self._static("index.html", page=True)
        elif path.startswith("/static/"):
            self._need(method, get, "GET")
            self._static(path[len("/static/") :])
        elif path == "/api/season":
            self._need(method, get, "GET")
            self._json(200, self.server.season().model_dump(mode="json"))
        elif m := _CHARACTER.fullmatch(path):
            self._need(method, get, "GET")
            entry = self.server.character(int(m.group(1)))
            if entry is None:
                raise _Reply(404, "not-found", f"no character {m.group(1)} in the index")
            self._json(200, entry.model_dump(mode="json"))
        elif path.startswith("/audio/"):
            self._need(method, get, "GET")
            self._audio(path[len("/audio/") :])
        elif path == "/api/decisions/preview":
            self._need(method, method == "POST", "POST")
            self._preview(self._decision())
        elif path == "/api/decisions":
            self._need(method, method == "POST", "POST")
            self._apply(self._decision())
        else:
            raise _Reply(404, "not-found", "no such route")

    def _need(self, method: str, ok: bool, allow: str) -> None:
        if not ok:
            raise _Reply(405, "method", f"{method} is not allowed here; {allow} is", allow=allow)

    def _login(self) -> None:
        s = self.server
        self.send_response(303)
        self.send_header("Location", "/")
        self.send_header(
            "Set-Cookie", f"{s.cookie_name}={s.token}; HttpOnly; SameSite=Strict; Path=/"
        )
        self.send_header("Content-Length", "0")
        self._common_headers()
        self.end_headers()

    # -- bodies -----------------------------------------------------------------------------

    def _body(self) -> bytes:
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise _Reply(411, "length", "a POST needs a Content-Length")
        try:
            n = int(raw)
        except ValueError:
            raise _Reply(400, "length", "Content-Length is not a number") from None
        if n < 0:
            raise _Reply(400, "length", "Content-Length is negative")
        if n > MAX_BODY:
            if n <= _DRAIN_MAX:  # read it, so the client sees the 413 and not a reset
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(left, 65536))
                    if not chunk:
                        break
                    left -= len(chunk)
            self.close_connection = True
            raise _Reply(413, "too-large", f"request bodies are at most {MAX_BODY} bytes")
        body = self.rfile.read(n)
        ctype = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            raise _Reply(415, "media-type", "POST bodies are application/json")
        return body

    def _decision(self) -> Decision:
        body = self._body()
        try:
            obj = json.loads(body)
        except (ValueError, UnicodeDecodeError) as e:
            raise _Reply(400, "bad-json", f"the body is not JSON: {e}") from None
        try:
            d = parse_decision(obj)
        except ValidationError as e:
            raise _Reply(
                400, "invalid", "not a decision", detail=json.loads(e.json(include_url=False))
            ) from None
        if d.op not in IDENTITY_OPS:
            raise _Reply(
                422, "not-identity",
                f"{d.op} is not an identity decision; an accept is made from a voice draft",
            )  # fmt: skip
        return d

    # -- decisions --------------------------------------------------------------------------

    def _preview(self, d: Decision) -> None:
        s = self.server
        if not s.ws.registry_path.is_file():
            raise _Reply(409, "no-registry", f"{s.ws.root} has no registry yet")
        stale: str | None = None
        unchecked = False
        try:
            with s._lock, Registry(s.ws.registry_path) as reg:
                lines, invs = plan_decision(reg, s.ws, d)
                try:
                    check_digest(reg, d)
                except StaleDecision as e:
                    stale = str(e)
                except UncheckedDecision:
                    unchecked = True
        except DecisionRefused as e:
            raise _Reply(422, "refused", str(e)) from None
        except sqlite3.OperationalError as e:
            raise _Reply(423, "busy", f"the registry is busy ({e}); try again") from None
        self._json(200, {
            "lines": lines,
            "invalidations": [{"episode": i.episode, "stage": i.stage} for i in invs],
            "stale": stale,
            "unchecked": unchecked,
        })  # fmt: skip

    def _apply(self, d: Decision) -> None:
        s = self.server
        with s._lock:
            try:
                result = apply_decision(s.ws, d)
            except DecisionError as e:
                raise self._refusal(e) from None
            doc = s.season()  # rebuilt: the registry changed
        cid = d.keep if isinstance(d, MergeDecision) else d.character
        entry = next((c for c in doc.characters if c.id == cid), None)
        self._json(200, {
            "lines": result.lines,
            "invalidations": [{"episode": i.episode, "stage": i.stage}
                              for i in result.invalidations],
            "character": entry.model_dump(mode="json") if entry is not None else None,
            "queue": doc.queue,
        })  # fmt: skip

    def _refusal(self, e: DecisionError) -> _Reply:
        s = self.server
        msg = str(e)
        if isinstance(e, StaleDecision):
            return _Reply(409, "stale", msg)
        if isinstance(e, UncheckedDecision):
            # the page sends the digest it was shown; a decision without one is a client bug
            return _Reply(422, "unchecked", msg)
        if isinstance(e, IncompleteDecision):
            return _Reply(409, "incomplete", msg, landed=True, resume=s.resume_command())
        if isinstance(e, UnfinishedRecords):
            return _Reply(409, "unfinished", msg, resume=s.resume_command())
        if isinstance(e, StudioBusy):
            return _Reply(423, "busy", msg)
        if isinstance(e, LogUnreadable):
            return _Reply(409, "log-unreadable", msg)
        if isinstance(e, RegistryUnreadable):
            return _Reply(409, "registry-unreadable", msg)
        return _Reply(422, "refused", msg)

    # -- files ------------------------------------------------------------------------------

    def _static(self, rel: str, *, page: bool = False) -> None:
        parts = rel.split("/")
        if not rel or not all(_SEGMENT.fullmatch(p) for p in parts):
            raise _Reply(404, "not-found", "no such file")
        path = STATIC_DIR
        for p in parts:
            path = path / p
            try:
                st = os.lstat(path)
            except OSError:
                raise _Reply(404, "not-found", "no such file") from None
            if stat.S_ISLNK(st.st_mode):
                raise _Reply(404, "not-found", "no such file")
        if not stat.S_ISREG(st.st_mode):
            raise _Reply(404, "not-found", "no such file")
        ctype = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._bytes(200, ctype, path.read_bytes(), csp=page or ctype.startswith("text/html"))

    def _audio(self, name: str) -> None:
        if not CLIP_NAME.fullmatch(name) or name not in self.server.audio_names():
            raise _Reply(404, "not-found", "no such clip in the index")
        studio = self.server.ws.root / "studio"
        try:
            if stat.S_ISLNK(os.lstat(studio).st_mode):
                raise _Reply(404, "not-found", "no such clip")
            dfd = os.open(studio / "audio", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError:
            raise _Reply(404, "not-found", "no such clip") from None
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
        except OSError:
            raise _Reply(404, "not-found", "no such clip") from None
        finally:
            os.close(dfd)
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise _Reply(404, "not-found", "no such clip")
            data = f.read()
        self._bytes(200, "audio/wav", data)

    # -- responses --------------------------------------------------------------------------

    def _common_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def _bytes(self, status: int, ctype: str, data: bytes, *, csp: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if csp:
            self.send_header("Content-Security-Policy", _CSP)
        self._common_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status: int, obj: Any) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._bytes(status, "application/json; charset=utf-8", data)


__all__ = ["DEFAULT_PORT", "MAX_BODY", "ROUTES", "StudioServer", "check_bind"]
