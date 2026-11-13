"""The studio's HTTP server: a loopback `ThreadingHTTPServer` with a token-guarded JSON API.

`StudioServer` binds 127.0.0.1 only unless the caller passes ``allow_non_loopback`` (the
command's ``--allow-non-loopback``), and then only one IPv4 interface address (never
0.0.0.0). The per-run random token reaches the page in the start-up URL's fragment
(``/#t=<token>``), which a browser never sends to a server; the page (``static/token.js``)
moves it into ``sessionStorage``, clears the fragment, and sends it as the
``X-Fukikae-Token`` header on every request, audio included (fetched into a blob URL). There
is no cookie and no ``?t=``: every ``/api`` and ``/audio`` request needs the header
(compared in constant time), so a link, form or media element on another site can't make
one. ``GET /`` and ``/static/*`` serve only the shipped shell, which holds no data, without
the token. ``Host`` must be the bound address (or ``localhost``) with the port (DNS
rebinding), an ``Origin``, when sent, must be this server's, POST bodies must be JSON and at
most `MAX_BODY` bytes, and the request log never prints a query string.

Each connection gets `StudioServer.request_timeout` seconds per read (30), so a client that
stops sending (a body shorter than its Content-Length, headers that never end) is dropped,
and at most `MAX_HANDLERS` connections are handled at once; past that a connection gets a
prompt 503. A handler that fails unexpectedly answers 500 ``{"error": "internal"}`` and
prints the traceback to stderr only.

The API reads the season index (`fukikae.studio.season.load_or_build`, rebuilt when its
inputs change; a decision changes them) and writes only through
`fukikae.studio.apply.apply_decision`, the studio's one writer; a preview is its dry run.
Audio is served only for the clips the current index refers to (`referenced_clips`), as bare
``<sha256>.wav`` names under ``<ws>/studio/audio``, each directory and the file opened
without following symlinks, the file non-blocking and only when it is a regular file. Static
files come only from the package's ``static/`` directory, opened the same way.

Voice drafts (`fukikae.studio.drafts`): a character's reference candidates, its draft per
target language and the draft's versions, and the clone backends with their voice keys. These
routes never take the decisions lock, so a long candidate read never holds up a decision; they
take their own lock, one draft request at a time. A candidate list is cached per character
until the index's inputs change, and is served a page at a time (at most `MAX_CANDIDATES`).
Candidate and draft clips are cut only when first fetched, by the server itself
(`fukikae.studio.season.cut_sample`), and only a clip the server has cut joins the audio it
serves; listing cuts nothing.

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
import sys
import threading
import traceback
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from fukikae.registry.store import RegistrySchemaError
from fukikae.studio.apply import (
    DecisionError,
    IncompleteDecision,
    LogUnreadable,
    RegistryUnreadable,
    StaleDecision,
    StudioBusy,
    UncheckedDecision,
    UnfinishedRecords,
    apply_decision,
)
from fukikae.studio.drafts import (
    MAX_REFERENCES,
    DraftError,
    NoRegistry,
    UnknownCharacter,
    add_version,
    backends,
    candidates,
    floor_meter,
    language_of,
    load_draft,
    open_draft,
    save_draft,
)
from fukikae.studio.model import (
    IDENTITY_OPS,
    Candidate,
    CharacterEntry,
    Decision,
    Draft,
    MergeDecision,
    RefSpan,
    SeasonDoc,
    ValidationError,
    parse_decision,
)
from fukikae.studio.season import (
    CLIP_NAME,
    cut_sample,
    inputs_digest,
    load_or_build,
    referenced_clips,
    sample_name,
)
from fukikae.workspace import Workspace

if TYPE_CHECKING:
    from fukikae.config import Config

DEFAULT_PORT = 8765
MAX_BODY = 1024 * 1024  # bytes; a larger POST body is 413
_DRAIN_MAX = 8 * MAX_BODY  # a too-large body up to this is read and dropped before the 413
TOKEN_HEADER = "X-Fukikae-Token"  # noqa: S105 (a header name)
REQUEST_TIMEOUT = 30.0  # seconds a connection may wait between reads
MAX_HANDLERS = 16  # connections handled at once; past that, a prompt 503
STATIC_DIR = Path(__file__).parent / "static"
MAX_CANDIDATES = 200  # candidates per page at most
DEFAULT_CANDIDATES = 50
MAX_OFFERED = 20000  # clip names offered for lazy cutting, oldest forgotten first

# (method, path, what it does): the API as the placeholder page lists it
ROUTES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/", "the page shell (no data); needs no token"),
    ("GET", "/static/<file>", "the page's files, from the package only; need no token"),
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
    (
        "GET",
        "/api/characters/<id>/candidates",
        "the character's reference candidates with their verdicts, a page at a time "
        "(?offset=, ?limit= up to 200, ?gated=1 to include refused turns)",
    ),
    (
        "GET",
        "/api/drafts/<id>/<lang>",
        "the character's voice draft for a target language, opened (version 1) when missing",
    ),
    (
        "POST",
        "/api/drafts/<id>/<lang>/versions",
        "a new draft version from {references, backend, params}: its card id, overrides and meter",
    ),
    ("GET", "/api/backends", "the clone backends and each one's voice keys (type and range)"),
)

_SEGMENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_CHARACTER = re.compile(r"/api/characters/([0-9]{1,9})")
_CANDIDATES = re.compile(r"/api/characters/([0-9]{1,9})/candidates")
_LANG = r"([A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,3})"
_DRAFT = re.compile(r"/api/drafts/([0-9]{1,9})/" + _LANG)
_VERSIONS = re.compile(r"/api/drafts/([0-9]{1,9})/" + _LANG + "/versions")
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
    """The IPv4 address to bind for ``host``. Loopback is 127.0.0.1 (or ``localhost``)
    only; with ``allow_non_loopback``, one specific IPv4 interface address (not 0.0.0.0, a
    broadcast or multicast address, or another loopback one). ValueError otherwise."""
    if host in ("localhost", "127.0.0.1"):
        return "127.0.0.1"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and ip.is_loopback:
        raise ValueError(
            f"the studio's loopback address is 127.0.0.1, not {host!r}: it binds 127.0.0.1 "
            "and nothing else on loopback"
        )
    if not allow_non_loopback:
        raise ValueError(
            f"the studio binds a loopback address only (127.0.0.1), not {host!r}: reach it "
            "over `ssh -L`; binding another address needs --allow-non-loopback"
        )
    if not isinstance(ip, ipaddress.IPv4Address):
        raise ValueError(f"the studio binds an IPv4 address; {host!r} is not one")
    if ip.is_unspecified or ip.is_multicast or ip == ipaddress.IPv4Address("255.255.255.255"):
        raise ValueError(
            f"--allow-non-loopback needs one interface address of this machine, not {host}"
        )
    return str(ip)


def _loopback(host: str) -> bool:
    return ipaddress.ip_address(host).is_loopback


_BUSY_REPLY = json.dumps(
    {"error": "busy", "message": f"the studio is handling {MAX_HANDLERS} requests; try again"}
).encode()


class StudioServer(ThreadingHTTPServer):
    """One workspace's studio. ``port`` 0 takes a free one (tests); ``token`` defaults to a
    fresh random one per server."""

    daemon_threads = True
    allow_reuse_address = True
    request_timeout = REQUEST_TIMEOUT

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
        request_timeout: float | None = None,
        max_handlers: int = MAX_HANDLERS,
    ) -> None:
        bind = check_bind(host, allow_non_loopback=allow_non_loopback)
        self.ws = ws if isinstance(ws, Workspace) else Workspace(Path(ws))
        self.cfg = cfg
        self.clips = clips
        self.token = token or secrets.token_urlsafe(32)
        self._lock = threading.RLock()  # the index and every decision, one at a time
        self._doc: SeasonDoc | None = None
        if request_timeout is not None:
            self.request_timeout = request_timeout
        self._slots = threading.BoundedSemaphore(max_handlers)
        # drafts: their own lock (never the decisions lock), the candidate lists cached per
        # character with the index inputs digest they were read under, the clip names offered
        # for cutting (name -> episode, start, end) and the ones the server has cut
        self._drafts_lock = threading.RLock()
        self._candidates: dict[int, tuple[str, list[Candidate]]] = {}
        self._offered: OrderedDict[str, tuple[str, float, float]] = OrderedDict()
        self._cut: set[str] = set()
        self._clip_lock = threading.Lock()
        super().__init__((bind, port), _Handler)
        p = self.server_address[1]
        names = ["127.0.0.1", "localhost"] if _loopback(bind) else [bind]
        self.allowed_hosts = frozenset(f"{n}:{p}" for n in names)
        self.allowed_origins = frozenset(f"http://{h}" for h in self.allowed_hosts)

    # -- the handler cap ----------------------------------------------------------------------

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._slots.acquire(blocking=False):
            self._refuse_busy(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def _refuse_busy(self, request: Any) -> None:
        try:
            request.settimeout(1.0)
            request.sendall(
                b"HTTP/1.1 503 Service Unavailable\r\n"
                b"Content-Type: application/json; charset=utf-8\r\n"
                b"Content-Length: " + str(len(_BUSY_REPLY)).encode() + b"\r\n"
                b"Retry-After: 1\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n"
                + _BUSY_REPLY
            )
        except OSError:
            pass
        self.shutdown_request(request)

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def loopback(self) -> bool:
        return _loopback(str(self.server_address[0]))

    def url(self) -> str:
        """The start-up URL: the token rides in the fragment, which no browser sends."""
        return f"http://{self.server_address[0]}:{self.port}/#t={self.token}"

    def tunnel(self) -> str:
        return f"ssh -L {self.port}:127.0.0.1:{self.port} {socket.gethostname()}"

    def season(self) -> SeasonDoc:
        """The index as it stands now: ``season.json``, or a rebuild when its inputs changed."""
        with self._lock:
            self._doc = load_or_build(self.ws, self.cfg, clips=self.clips)
            return self._doc

    def audio_names(self) -> set[str]:
        """The clip names ``/audio`` may serve: those the last index served refers to, and the
        candidate and draft clips the server itself has cut."""
        with self._lock:
            doc = self._doc or self.season()
            names = referenced_clips(doc)
        with self._clip_lock:
            return names | self._cut

    # -- drafts -------------------------------------------------------------------------------

    def candidates(self, character_id: int) -> list[Candidate]:
        """The character's candidates (`drafts.candidates`), read again only when the index's
        inputs changed since; under the drafts lock, never the decisions lock."""
        with self._drafts_lock:
            digest = inputs_digest(self.ws, self.cfg, clips=self.clips)
            got = self._candidates.get(character_id)
            if got is None or got[0] != digest:
                got = (digest, candidates(self.ws, self.cfg, character_id))
                self._candidates[character_id] = got
            return got[1]

    def offer(self, episode: str, start: float, end: float) -> str | None:
        """The clip name of a span the page may fetch, remembered so ``/audio`` cuts it on
        first request; None when the episode has no vocals to cut it from."""
        try:
            name = sample_name(self.ws, episode, start, end)
        except OSError:
            return None
        with self._clip_lock:
            self._offered[name] = (episode, start, end)
            self._offered.move_to_end(name)
            while len(self._offered) > MAX_OFFERED:
                self._offered.popitem(last=False)
        return name

    def ensure_clip(self, name: str) -> None:
        """Cut an offered clip that isn't on disk (first fetch, or pruned by a rebuild), and
        let ``/audio`` serve it."""
        with self._clip_lock:
            span = self._offered.get(name)
        if span is None:
            return
        path = self.ws.root / "studio" / "audio" / name
        if not path.is_file():
            try:
                cut = cut_sample(self.ws, *span)
            except (OSError, ValueError):  # the vocals went away: nothing to serve
                return
            if cut.name != name:  # the vocals changed since the span was offered
                return
        with self._clip_lock:
            self._cut.add(name)

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


_PATHS = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][^\s'\"\\/:]+)+[\\/]?")


def _why(e: BaseException) -> str:
    """An exception for a client: its type and message, any filesystem path replaced."""
    return f"{type(e).__name__}: {_PATHS.sub('<path>', str(e))}"


class _Handler(BaseHTTPRequestHandler):
    server: StudioServer
    server_version = "fukikae-studio"
    sys_version = ""
    timeout = REQUEST_TIMEOUT  # the server's request_timeout replaces it per connection

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.request_timeout)

    # -- logging: never the query string ------------------------------------------------------

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
            try:
                self._route(method)
            except _Reply as r:
                self._json(r.status, r.body)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                raise
            except Exception:  # noqa: BLE001 (the client gets a generic 500, stderr the rest)
                traceback.print_exc(file=sys.stderr)
                self.close_connection = True
                self._json(500, {"error": "internal"})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True  # the client went away; nothing to tell it

    # -- guards -----------------------------------------------------------------------------

    def _guard(self) -> None:
        host = self.headers.get("Host", "")
        if host not in self.server.allowed_hosts:
            raise _Reply(403, "forbidden", "this request's Host is not the studio's")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in self.server.allowed_origins:
            raise _Reply(403, "forbidden", "this request's Origin is not the studio's")

    def _authorized(self) -> bool:
        value = self.headers.get(TOKEN_HEADER)
        return value is not None and hmac.compare_digest(
            value.encode("utf-8", "replace"), self.server.token.encode()
        )

    # -- routing ----------------------------------------------------------------------------

    def _route(self, method: str) -> None:
        self._guard()
        path = urlsplit(self.path).path
        guarded = path == "/api" or path.startswith(("/api/", "/audio/"))
        if guarded and not self._authorized():
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
            self._json(200, self._season().model_dump(mode="json"))
        elif m := _CANDIDATES.fullmatch(path):
            self._need(method, get, "GET")
            self._candidates(int(m.group(1)), urlsplit(self.path).query)
        elif m := _VERSIONS.fullmatch(path):
            self._need(method, method == "POST", "POST")
            self._version(int(m.group(1)), m.group(2))
        elif m := _DRAFT.fullmatch(path):
            self._need(method, get, "GET")
            self._draft(int(m.group(1)), m.group(2))
        elif path == "/api/backends":
            self._need(method, get, "GET")
            self._json(200, {"backends": backends(self.server.cfg)})
        elif m := _CHARACTER.fullmatch(path):
            self._need(method, get, "GET")
            cid = int(m.group(1))
            entry = next((c for c in self._season().characters if c.id == cid), None)
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

    def _season(self) -> SeasonDoc:
        """The index, its known failures as clear errors: a registry from a newer fukikae or
        one that isn't a database is 409, a locked one 503."""
        try:
            return self.server.season()
        except RegistrySchemaError as e:
            raise _Reply(409, "registry-schema", str(e)) from None
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                raise _Reply(
                    503, "busy", f"the registry is locked ({e}): a run or another writer has "
                    "it; try again",
                ) from None  # fmt: skip
            raise _Reply(409, "registry-unreadable", f"the registry can't be read: {e}") from None
        except sqlite3.DatabaseError as e:
            raise _Reply(409, "registry-unreadable", f"the registry can't be read: {e}") from None

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
        try:
            return self._read_body(n)
        except TimeoutError:
            self.close_connection = True
            raise _Reply(408, "timeout", "the request body stopped short of its length") from None

    def _read_body(self, n: int) -> bytes:
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
        if len(body) < n:
            self.close_connection = True
            raise _Reply(400, "length", "the request body is shorter than its Content-Length")
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
        """`apply_decision`'s dry run (never creates or migrates a registry; write-time
        refusals included). A stale or digest-less decision is previewed as it would apply
        with the check skipped, and says so; ``decision`` is what to post to apply it."""
        s = self.server
        if not s.ws.registry_path.is_file():
            raise _Reply(409, "no-registry", f"{s.ws.root} has no registry yet")
        stale: str | None = None
        unchecked = False
        with s._lock:
            try:
                try:
                    result = apply_decision(s.ws, d, dry_run=True)
                except StaleDecision as e:
                    stale = str(e)
                    result = apply_decision(s.ws, d, dry_run=True, force=True)
                except UncheckedDecision:
                    unchecked = True
                    result = apply_decision(s.ws, d, dry_run=True, force=True)
            except DecisionError as e:
                raise self._refusal(e) from None
        stamped = result.decision if result.decision is not None else d
        self._json(200, {
            "lines": result.lines,
            "invalidations": [{"episode": i.episode, "stage": i.stage}
                              for i in result.invalidations],
            "stale": stale,
            "unchecked": unchecked,
            "decision": stamped.model_dump(mode="json"),
        })  # fmt: skip

    def _apply(self, d: Decision) -> None:
        s = self.server
        warning: str | None = None
        with s._lock:
            try:
                result = apply_decision(s.ws, d)
            except DecisionError as e:
                raise self._refusal(e) from None
            try:
                doc: SeasonDoc | None = s.season()  # rebuilt: the registry changed
            except Exception as e:  # noqa: BLE001 (the decision landed; say so either way)
                traceback.print_exc(file=sys.stderr)
                doc = None
                warning = (
                    "the decision landed, but the season index couldn't be rebuilt "
                    f"({_why(e)}); reload the page to try again"
                )
        cid = d.keep if isinstance(d, MergeDecision) else d.character
        entry = None if doc is None else next((c for c in doc.characters if c.id == cid), None)
        body: dict[str, Any] = {
            "lines": result.lines,
            "invalidations": [{"episode": i.episode, "stage": i.stage}
                              for i in result.invalidations],
            "character": entry.model_dump(mode="json") if entry is not None else None,
            "queue": doc.queue if doc is not None else None,
        }  # fmt: skip
        if warning is not None:
            body["warning"] = warning
        self._json(200, body)

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

    # -- drafts -----------------------------------------------------------------------------

    def _draft_refusal(self, e: DraftError) -> _Reply:
        if isinstance(e, UnknownCharacter):
            return _Reply(404, "not-found", str(e))
        if isinstance(e, NoRegistry):
            return _Reply(409, "no-registry", "the workspace has no registry yet")
        return _Reply(422, "refused", _PATHS.sub("<path>", str(e)))

    def _candidates(self, cid: int, query: str) -> None:
        q = parse_qs(query, keep_blank_values=True)

        def number(key: str, default: int, lo: int, hi: int) -> int:
            raw = q.get(key, [str(default)])[-1]
            if not raw.isdigit() or not lo <= int(raw) <= hi:
                raise _Reply(400, "bad-query", f"{key} must be a whole number {lo} to {hi}")
            return int(raw)

        offset = number("offset", 0, 0, 10**9)
        limit = number("limit", DEFAULT_CANDIDATES, 1, MAX_CANDIDATES)
        gated = q.get("gated", ["0"])[-1]
        if gated not in ("0", "1"):
            raise _Reply(400, "bad-query", "gated is 0 or 1")
        try:
            got = self.server.candidates(cid)
        except DraftError as e:
            raise self._draft_refusal(e) from None
        shown = got if gated == "1" else [c for c in got if c.rank is not None]
        page = shown[offset : offset + limit]
        for c in page:
            self.server.offer(c.episode, c.start, c.end)
        self._json(200, {
            "character": cid, "total": len(shown),
            "eligible": sum(c.rank is not None for c in got), "offset": offset, "limit": limit,
            "candidates": [c.model_dump(mode="json") for c in page],
        })  # fmt: skip

    def _draft_body(self, draft: Draft) -> dict[str, Any]:
        clips: dict[str, str] = {}
        for v in draft.versions:
            for r in v.references:
                name = self.server.offer(r.episode, r.start, r.end)
                if name is not None:
                    clips[r.label()] = name
        latest = draft.versions[-1].references if draft.versions else []
        return {"draft": draft.model_dump(mode="json"), "meter": floor_meter(latest),
                "clips": clips}  # fmt: skip

    def _load_or_open(self, cid: int, lang: str) -> Draft:
        s = self.server
        try:
            language = language_of(lang)
            draft = load_draft(s.ws, cid, language)
            if draft is None:
                draft = open_draft(s.ws, s.cfg, cid, language, offered=s.candidates(cid))
                save_draft(s.ws, draft)
        except DraftError as e:
            raise self._draft_refusal(e) from None
        except ValueError as e:  # a draft file that isn't one
            raise _Reply(409, "draft-unreadable", f"the saved draft can't be read: {_why(e)}"
                         ) from None  # fmt: skip
        return draft

    def _draft(self, cid: int, lang: str) -> None:
        with self.server._drafts_lock:
            draft = self._load_or_open(cid, lang)
        self._json(200, self._draft_body(draft))

    def _version(self, cid: int, lang: str) -> None:
        body = self._body()
        try:
            req = _VersionRequest.model_validate_json(body)
        except ValidationError as e:
            raise _Reply(
                400, "invalid", "not a draft version", detail=json.loads(e.json(include_url=False))
            ) from None
        s = self.server
        with s._drafts_lock:
            draft = self._load_or_open(cid, lang)
            try:
                offered = s.candidates(cid)
                draft, version = add_version(
                    s.ws, s.cfg, draft, req.references, req.backend, req.params, offered=offered
                )
            except DraftError as e:
                raise self._draft_refusal(e) from None
        clips = {r.label(): n for r in version.references
                 if (n := s.offer(r.episode, r.start, r.end)) is not None}  # fmt: skip
        self._json(200, {"version": version.model_dump(mode="json"),
                         "meter": floor_meter(version.references), "clips": clips})  # fmt: skip

    # -- files ------------------------------------------------------------------------------

    def _static(self, rel: str, *, page: bool = False) -> None:
        parts = rel.split("/")
        if not rel or not all(_SEGMENT.fullmatch(p) for p in parts):
            raise _Reply(404, "not-found", "no such file")
        try:
            dfd = os.open(STATIC_DIR, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            raise _Reply(404, "not-found", "no such file") from None
        try:
            data = _read_regular(dfd, parts)
        finally:
            os.close(dfd)
        if data is None:
            raise _Reply(404, "not-found", "no such file")
        ctype = _CONTENT_TYPES.get(Path(parts[-1]).suffix, "application/octet-stream")
        self._bytes(200, ctype, data, csp=page or ctype.startswith("text/html"))

    def _audio(self, name: str) -> None:
        if CLIP_NAME.fullmatch(name):
            self.server.ensure_clip(name)
        if not CLIP_NAME.fullmatch(name) or name not in self.server.audio_names():
            raise _Reply(404, "not-found", "no such clip in the index")
        try:
            dfd = os.open(self.server.ws.root, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            raise _Reply(404, "not-found", "no such clip") from None
        try:
            data = _read_regular(dfd, ["studio", "audio", name])
        finally:
            os.close(dfd)
        if data is None:
            raise _Reply(404, "not-found", "no such clip")
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


class _VersionRequest(BaseModel):
    """``POST /api/drafts/<id>/<lang>/versions``: the references in order, the backend and
    its voice keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    references: list[RefSpan] = Field(min_length=1, max_length=MAX_REFERENCES)
    backend: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = {}


def _read_regular(dfd: int, parts: list[str]) -> bytes | None:
    """The bytes of ``parts`` under the directory ``dfd``, each directory opened relative to
    the last with ``O_NOFOLLOW | O_DIRECTORY`` and the file with ``O_NOFOLLOW | O_NONBLOCK``
    (a FIFO doesn't block the open), read only when ``fstat`` says it is a regular file. None
    for a symlink anywhere, a missing part, or anything but a regular file."""
    fds: list[int] = []
    try:
        cur = dfd
        for p in parts[:-1]:
            cur = os.open(p, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=cur)
            fds.append(cur)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=cur)
        fds.append(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        with os.fdopen(os.dup(fd), "rb") as f:
            return f.read()
    except OSError:
        return None
    finally:
        for fd in fds:
            os.close(fd)


__all__ = [
    "DEFAULT_PORT",
    "MAX_BODY",
    "MAX_HANDLERS",
    "REQUEST_TIMEOUT",
    "ROUTES",
    "StudioServer",
    "check_bind",
]
