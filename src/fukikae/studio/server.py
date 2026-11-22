"""The studio's HTTP server: a loopback `ThreadingHTTPServer` with a token-guarded JSON API.

`StudioServer` binds 127.0.0.1 only unless the caller passes ``allow_non_loopback`` (the
command's ``--allow-non-loopback``), and then only one IPv4 interface address (never
0.0.0.0). The per-run random token reaches the page in the start-up URL's fragment
(``/#t=<token>``), which a browser never sends to a server; the page (``static/token.js``)
moves it into ``sessionStorage``, clears the fragment, and sends it as the
``X-Fukikae-Token`` header on every request, audio and video included (fetched into a blob
URL). There is no cookie and no ``?t=``: every ``/api``, ``/audio`` and ``/video`` request
needs the header
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
without following symlinks, the file non-blocking and only when it is a regular file. Video
is served the same way, only for the ``<sha256>.mp4`` names the current index refers to
(`referenced_videos`), from ``<ws>/studio/video``; a named clip not yet on disk is cut on its
first fetch (`fukikae.studio.season.cut_video`), at most `MAX_CUTS` at once and each name
once, and only when the source still gives that name (a source replaced since the index was
built gives a 404 until the page reloads the index). A request waits at most `CUT_WAIT_S` for
its name's cut or a cut slot and then gets 503 ``busy``; a failed cut answers 404 without
running ffmpeg again for `VIDEO_FAILURE_TTL_S` or until the index is rebuilt; HEAD never cuts.
The ``source`` tree is scanned off the request path (`fukikae.studio.season.SourceScanner`),
never under the index lock, so a stalled media share can't freeze the API. ``/audio`` and
``/video`` answer a single ``Range: bytes=`` range with 206 (416 when it starts past the
end; any other Range form gets the whole file), after the same token check and allow-list as
a whole fetch; the names are content hashes, so a clip never changes under a range. Static
files come only from the package's ``static/`` directory, opened the same way.

Voice drafts (`fukikae.studio.drafts`): a character's reference candidates, its draft per
target language and the draft's versions, and the clone backends with their voice keys. These
routes never take the decisions lock, so a long candidate read never holds up a decision; they
take their own lock, one draft request at a time. A candidate list is cached per character
until the files it reads change (`fukikae.studio.drafts.candidates_digest`, a stat digest
computed outside that lock), and is served a page at a time (at most `MAX_CANDIDATES`).
HEAD on a draft only reads: it never opens or saves one.
Candidate and draft clips are cut only when first fetched, by the server itself
(`fukikae.studio.season.cut_sample`), and only a clip the server has cut joins the audio it
serves; listing cuts nothing.

Auditions (`fukikae.studio.audition`): ``POST /api/auditions`` checks the draft version and
the line and queues a job for the server's one `AuditionWorker`, answering 202 at once; the
page polls ``/api/auditions/<job>``, never waits on a render, and can cancel a job.
A backend the worker finds can't run here fails the job with ``refused`` set (the check can
take 30 s, so it never runs on a request thread), and a submit for it is then a 422 at once.
``/api/gpu`` reports the worker and who holds the card (read from the lock file, never
waiting on it; a file naming a process that exited while the lock is still held is reported
as ``leftover of pid N``); ``/api/gpu/release`` cancels what hasn't started rendering,
unloads the model and lets go of the GPU. An audition's
audio is served at ``/audition/<key>.wav`` (16 hex, the render's cache key) from
``<ws>/studio/auditions``, opened the way ``/audio`` opens a clip, with the same token check
and Range handling, only for the names this server's worker rendered or a draft it served
lists. `server_close` stops the worker, which unloads its model and releases the lock.

Compare and accept (Task 20). ``GET /api/drafts/<id>/<lang>/compare`` reports two versions
of the saved draft on one line, pairing auditions by their line (never by ``cue_key``, which
differs by card): a version's latest audition of the line in the draft, else its latest job's,
else a new job queued through the worker (never on HEAD; a failed or cancelled side again only
with ``retry=1``, even after the worker's job ring forgot the job), so the page polls the
route and never waits on a render. The lookup and the submit hold one lock per draft, version
and line (`StudioServer.compare_lock`), so two tabs never queue a side twice. With ``blind=1``
the sides are ``A`` and ``B`` (`fukikae.abpair.blind_pair` under a seed drawn per server and
kept in memory, the slot the draft and line, so a pair keeps its labels either way round),
queued in label order, never query order. A blind answer has one state for the pair
(``rendering`` until both sides are done, then ``ready``; `_pair_state`) and generic messages,
and each side only a ``wav_url`` once the pair is ready: an opaque alias
(`StudioServer.blind_name`, an HMAC of the seed, draft, pair, line and label) that
``/audition`` resolves to the render, never the render key the draft JSON lists. So neither
the order sides finish in nor a URL tells the versions apart; ``POST .../reveal`` gives the
key. ``POST .../accept`` takes ``{version, card_id}`` (409 ``version-changed`` when the
version is another card now: the draft was opened again since the page loaded it), rechecks
the version's `fukikae.studio.drafts.accept_problems` (422 naming them), builds the accept
decision from the saved version (`fukikae.studio.drafts.accept_decision`: its card,
references, overrides, audition count, and the digest of the language's card assignment taken
when the version was made, so a tts run that re-picked the card since makes it stale, 409)
and applies it through `apply_decision` under the decisions lock, after a dry run that fills
its key and external id; ``/api/decisions`` still refuses a raw accept. ``.../accept/preview``
is the same checks and the dry run alone. Both add ``cached``, per episode whose tts goes
stale the character's lines already in the tts cache with that card
(`fukikae.studio.audition.cached_lines` under the workspace's config re-read now, as the next
run reads it; worked out outside both locks).

The page itself waits on the design pick (D47); ``static/index.html`` is a placeholder that
lists `ROUTES`.
"""

from __future__ import annotations

import contextlib
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
import time
import traceback
from collections import OrderedDict
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from fukikae.abpair import blind_pair
from fukikae.artifacts import ArtifactVersionError
from fukikae.media import MediaError
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
from fukikae.studio.audition import (
    FINISHED,
    AuditionJob,
    AuditionRefused,
    AuditionWorker,
    NoSuchDraft,
    QueueFull,
    cached_lines,
)
from fukikae.studio.drafts import (
    MAX_REFERENCES,
    DraftError,
    NoRegistry,
    UnknownCharacter,
    UnknownVersion,
    accept_decision,
    accept_problems,
    add_version,
    backends,
    candidates,
    candidates_digest,
    character_gone,
    floor_meter,
    language_of,
    load_draft,
    open_draft,
    save_draft,
    version_of,
)
from fukikae.studio.model import (
    IDENTITY_OPS,
    AcceptDecision,
    Audition,
    AuditionLine,
    Candidate,
    CharacterEntry,
    Decision,
    Draft,
    DraftVersion,
    MergeDecision,
    RealLine,
    RefSpan,
    SeasonDoc,
    TypedLine,
    ValidationError,
    parse_decision,
)
from fukikae.studio.season import (
    CLIP_NAME,
    VIDEO_NAME,
    NoVideo,
    SourceScanner,
    cut_sample,
    cut_video,
    load_or_build,
    referenced_clips,
    referenced_videos,
    sample_name,
    video_name,
    video_source,
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
MAX_LINES_PAGE = 500  # translated lines per page at most (`/api/characters/<id>/lines`)
MAX_OFFERED = 20000  # clip names offered for lazy cutting, oldest forgotten first
MAX_CUTS = 2  # video clips cut at once (each ffmpeg uses every core)
CUT_WAIT_S = 20.0  # how long a /video request waits for its name's cut or a cut slot: 503 after
VIDEO_FAILURE_TTL_S = 300.0  # a failed cut answers 404 without ffmpeg for this long
MAX_CUT = 2000  # clip names the server has cut and serves, oldest forgotten first

# (method, path, what it does): the API as the placeholder page lists it
ROUTES: tuple[tuple[str, str, str], ...] = (
    ("GET", "/", "the page shell (no data); needs no token"),
    ("GET", "/static/<file>", "the page's files, from the package only; need no token"),
    ("GET", "/api/season", "the season index (SeasonDoc), rebuilt when its inputs changed"),
    ("GET", "/api/characters/<id>", "one character's entry (CharacterEntry) from the index"),
    (
        "GET",
        "/audio/<sha256>.wav",
        "a clip the index refers to (a sample's clip field); one Range: bytes= range, 206/416",
    ),
    (
        "GET",
        "/video/<sha256>.mp4",
        "a sample's video clip the index refers to (its video field), cut on first fetch; "
        "one Range: bytes= range, 206/416",
    ),
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
        "/api/characters/<id>/lines",
        "the character's spoken translated cues (real audition lines) for a target language "
        "(?language=, default the workspace's), each with its source clip, a page at a time "
        "(?offset=, ?limit= up to 500)",
    ),
    (
        "GET",
        "/api/drafts/<id>/<lang>",
        "the character's voice draft for a target language, opened (version 1) when missing; "
        "409 character-merged (naming merged_into) for a character merged away",
    ),
    (
        "POST",
        "/api/drafts/<id>/<lang>/versions",
        "a new draft version from {references, backend, params}: its card id, overrides and meter",
    ),
    (
        "GET",
        "/api/drafts/<id>/<lang>/compare",
        "two versions' auditions of one line (?a=<n>&b=<n>&line=<JSON line>), the missing ones "
        "submitted; &blind=1 labels them A and B with one state for the pair and the versions "
        "hidden, &retry=1 submits a failed side again",
    ),
    (
        "POST",
        "/api/drafts/<id>/<lang>/reveal",
        "the key of a blind compare ({a, b, line}): which version A and B are",
    ),
    (
        "POST",
        "/api/drafts/<id>/<lang>/accept/preview",
        "a dry run of accepting a draft version ({version, card_id}): what reruns, whether it "
        "is stale, the accept decision, and the lines already cached",
    ),
    (
        "POST",
        "/api/drafts/<id>/<lang>/accept",
        "accept a draft version ({version, card_id}) as the character's card for the "
        "language: studio, locked; 409 when the card assignment changed since the version was "
        "made, or the version is another card now",
    ),
    (
        "GET",
        "/api/backends",
        "the clone backends and each one's voice keys (type and range), and whether it can run "
        "here (checked false until a background check answers)",
    ),
    (
        "POST",
        "/api/auditions",
        "queue an audition of a draft version on a line ({character, language, version, line}): "
        "202 with the job; 422 when its backend can't render here, 503 when the queue is full",
    ),
    (
        "GET",
        "/api/auditions/<job>",
        "an audition job: queued, waiting (for the GPU), loading, running, done (its wav_url and "
        "a real line's source_clip), failed (its message) or cancelled",
    ),
    ("POST", "/api/auditions/<job>/cancel", "cancel an audition job"),
    ("GET", "/api/gpu", "the audition worker and the GPU: backend loaded, who holds the card"),
    ("POST", "/api/gpu/release", "unload the audition model and let go of the GPU"),
    (
        "GET",
        "/audition/<key>.wav",
        "an audition's audio (a job's or a draft's wav); one Range: bytes= range, 206/416",
    ),
)

_SEGMENT = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*")
_CHARACTER = re.compile(r"/api/characters/([0-9]{1,9})")
_CANDIDATES = re.compile(r"/api/characters/([0-9]{1,9})/candidates")
_LINES = re.compile(r"/api/characters/([0-9]{1,9})/lines")
_NUMBER = re.compile(r"[0-9]{1,10}")  # ASCII digits only (str.isdigit takes any script's)
_LANG = r"([A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,3})"
_RANGE = re.compile(r"bytes=([0-9]{0,18})-([0-9]{0,18})")
_DRAFT = re.compile(r"/api/drafts/([0-9]{1,9})/" + _LANG)
_VERSIONS = re.compile(r"/api/drafts/([0-9]{1,9})/" + _LANG + "/versions")
_COMPARE = re.compile(r"/api/drafts/([0-9]{1,9})/" + _LANG + "/compare")
_REVEAL = re.compile(r"/api/drafts/([0-9]{1,9})/" + _LANG + "/reveal")
_ACCEPT = re.compile(r"/api/drafts/([0-9]{1,9})/" + _LANG + "/accept(/preview)?")
_LINE: TypeAdapter[TypedLine | RealLine] = TypeAdapter(AuditionLine)
_JOB = re.compile(r"/api/auditions/([0-9a-f]{16})")
_JOB_CANCEL = re.compile(r"/api/auditions/([0-9a-f]{16})/cancel")
AUDITION_NAME = re.compile(r"[0-9a-f]{16}\.wav")
BLIND_NAME = re.compile(r"b[0-9a-f]{32}\.wav")  # a blind compare side's alias
_USER_CONFIG = object()  # StudioServer's default: ~/.config/fukikae/config.toml, as load_config
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}
# media-src takes blob: because a clip, video or audition is fetched with the token header and
# played from a blob URL (an <audio> or <video> element can't send a header); nothing else does
_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
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
        cut_wait: float = CUT_WAIT_S,
        user_config: Path | None | object = _USER_CONFIG,
    ) -> None:
        bind = check_bind(host, allow_non_loopback=allow_non_loopback)
        self.ws = ws if isinstance(ws, Workspace) else Workspace(Path(ws))
        self.cfg = cfg
        self._user_config = user_config  # the user config `config` reads beneath fukikae.toml
        self.clips = clips
        self.token = token or secrets.token_urlsafe(32)
        self._lock = threading.RLock()  # the index and every decision, one at a time
        self._doc: SeasonDoc | None = None
        # the source tree, scanned off the request path and never under self._lock
        self._sources = SourceScanner()
        if request_timeout is not None:
            self.request_timeout = request_timeout
        self._slots = threading.BoundedSemaphore(max_handlers)
        # drafts: their own lock (never the decisions lock), the candidate lists cached per
        # character with the index inputs digest they were read under, the clip names offered
        # for cutting (name -> episode, start, end) and the ones the server has cut
        self._drafts_lock = threading.RLock()
        self._candidates: dict[int, tuple[str, list[Candidate]]] = {}
        self._offered: OrderedDict[str, tuple[str, float, float]] = OrderedDict()
        self._cut: OrderedDict[str, None] = OrderedDict()
        self._clip_lock = threading.Lock()
        # video: one cut per name at a time (name -> [its lock, requests holding or waiting
        # on it], dropped when none is), at most MAX_CUTS cuts at once, a request waiting at
        # most cut_wait seconds for either, and failed cuts by name (time.monotonic()), all
        # under _clip_lock; the failures are forgotten when the index is rebuilt
        self._video_locks: dict[str, list[Any]] = {}
        self._cut_slots = threading.BoundedSemaphore(MAX_CUTS)
        self.cut_wait = cut_wait
        self._video_failed: dict[str, float] = {}
        # auditions: the worker (its own thread; takes the drafts lock to read and record a
        # draft, never the decisions lock) and the audition wavs listed to the page in drafts
        self.auditions = AuditionWorker(self.ws, cfg, drafts_lock=self._drafts_lock)
        self._audition_names: OrderedDict[str, None] = OrderedDict()
        # blind compares: the labels' seed, drawn per server and kept only in memory; the
        # opaque names blind sides' audio is served under (alias -> render wav name, under
        # _clip_lock); and one lock per (draft, version, line) a compare may submit
        self.blind_seed = secrets.randbits(64)
        self._blind_names: OrderedDict[str, str] = OrderedDict()
        self._compare_locks: dict[tuple[str, ...], list[Any]] = {}
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

    def server_close(self) -> None:
        """Stop the audition worker (unloading its model, letting go of the GPU), then close."""
        self.auditions.close()
        super().server_close()

    def list_audition(self, name: str) -> None:
        """Let ``/audition`` serve ``name``, a wav a saved draft lists."""
        if AUDITION_NAME.fullmatch(name):
            with self._clip_lock:
                self._audition_names[name] = None
                self._audition_names.move_to_end(name)
                while len(self._audition_names) > MAX_CUT:
                    self._audition_names.popitem(last=False)

    def blind_name(self, draft: str, pair: tuple[int, int], line: str, label: str,
                   wav: str) -> str:  # fmt: skip
        """The opaque name a blind side's audio is served under: an HMAC of the server's seed,
        the draft, the pair, the line and the label (so it says nothing of the version or its
        render key), remembered (bounded, `MAX_CUT`) as an alias of ``wav`` for ``/audition``
        to resolve."""
        lo, hi = sorted(pair)
        msg = "\0".join((draft, str(lo), str(hi), line, label)).encode()
        name = "b" + hmac.new(self.blind_seed.to_bytes(8, "big"), msg, "sha256").hexdigest()[:32]
        name += ".wav"
        with self._clip_lock:
            self._blind_names[name] = wav
            self._blind_names.move_to_end(name)
            while len(self._blind_names) > MAX_CUT:
                self._blind_names.popitem(last=False)
        return name

    def blind_wav(self, name: str) -> str | None:
        """The render wav a blind alias stands for, or None."""
        with self._clip_lock:
            return self._blind_names.get(name)

    @contextlib.contextmanager
    def compare_lock(self, key: tuple[str, ...]) -> Iterator[None]:
        """One compare at a time looking up and submitting the same (draft, version, line), so
        two tabs polling one pair never queue a side twice. The lock is dropped when no request
        holds or waits on it."""
        with self._clip_lock:
            entry = self._compare_locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                yield
        finally:
            with self._clip_lock:
                entry[1] -= 1
                if entry[1] == 0 and self._compare_locks.get(key) is entry:
                    del self._compare_locks[key]

    def config(self) -> Config:
        """The workspace's config as it is now (re-read, as the next `fukikae run` reads it),
        for what must match a run; ConfigError when it no longer reads."""
        from fukikae.config import load_config

        if self._user_config is _USER_CONFIG:
            return load_config(self.ws.root)
        return load_config(self.ws.root, cast(Path | None, self._user_config))

    def audition_names(self) -> set[str]:
        """The audition wavs ``/audition`` may serve: those this server's worker rendered and
        those the drafts it served list."""
        with self._clip_lock:
            listed = set(self._audition_names)
        return listed | self.auditions.made()

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
        """The index as it stands now: ``season.json``, or a rebuild when its inputs changed.
        The source tree is scanned first (`SourceScanner`, bounded by its timeout), outside
        the index lock; a rebuild forgets the failed video cuts."""
        sources = self._sources.get(self.cfg.source)
        with self._lock:
            self._doc = load_or_build(self.ws, self.cfg, clips=self.clips, sources=sources,
                                      on_build=self._rebuilt)  # fmt: skip
            return self._doc

    def _rebuilt(self, _doc: SeasonDoc) -> None:
        with self._clip_lock:
            self._video_failed.clear()

    def _current(self) -> SeasonDoc:
        """The last index served, or the index now when none has been."""
        doc = self._doc
        return doc if doc is not None else self.season()

    def audio_names(self) -> set[str]:
        """The clip names ``/audio`` may serve: those the last index served refers to, and the
        candidate and draft clips the server itself has cut."""
        names = referenced_clips(self._current())
        with self._clip_lock:
            return names | set(self._cut)

    def video_names(self) -> dict[str, tuple[str, float, float]]:
        """The video names ``/video`` may serve, each with its sample's episode and span:
        those the last index served refers to."""
        return referenced_videos(self._current())

    def ensure_video(self, name: str, span: tuple[str, float, float]) -> None:
        """Cut a referenced video clip that isn't on disk; anything already at its path (a
        file, a symlink, a FIFO) is left for the read to judge. Cuts nothing when the source
        is gone or now gives another name. One cut per name at a time: a request for a name
        being cut waits for that cut and doesn't cut again. `_Reply` 503 ``busy`` when the
        name's cut or a cut slot isn't free within ``cut_wait`` seconds, 404 ``cut-failed``
        when the name's cut failed in the last `VIDEO_FAILURE_TTL_S` seconds (ffmpeg isn't
        run again until then, or until the index is rebuilt)."""
        path = self.ws.root / "studio" / "video" / name
        if os.path.lexists(path):
            return
        with self._clip_lock:
            entry = self._video_locks.setdefault(name, [threading.Lock(), 0])
            entry[1] += 1
        try:
            if not entry[0].acquire(timeout=self.cut_wait):
                raise _busy("this clip is still being cut")
            try:
                if os.path.lexists(path):
                    return
                self._refuse_failed(name)
                if not self._cut_slots.acquire(timeout=self.cut_wait):
                    raise _busy(f"{MAX_CUTS} clips are being cut")
                try:
                    src = video_source(self.ws, self.cfg, span[0],
                                       self._sources.get(self.cfg.source).files)  # fmt: skip
                    if video_name(src, span[1], span[2]) == name:
                        cut_video(self.ws, src, span[1], span[2])
                except (NoVideo, OSError, MediaError):
                    traceback.print_exc(file=sys.stderr)
                    with self._clip_lock:
                        self._video_failed[name] = time.monotonic()
                    self._refuse_failed(name)
                finally:
                    self._cut_slots.release()
            finally:
                entry[0].release()
        finally:
            with self._clip_lock:
                entry[1] -= 1
                if entry[1] == 0 and self._video_locks.get(name) is entry:
                    del self._video_locks[name]

    def _refuse_failed(self, name: str) -> None:
        with self._clip_lock:
            failed = self._video_failed.get(name)
            if failed is not None and time.monotonic() - failed >= VIDEO_FAILURE_TTL_S:
                del self._video_failed[name]
                failed = None
        if failed is not None:
            raise _Reply(404, "cut-failed", "this clip couldn't be cut (the studio's log says "
                         f"why); it is tried again after {VIDEO_FAILURE_TTL_S:g} s or when "
                         "the index is rebuilt")  # fmt: skip

    # -- drafts -------------------------------------------------------------------------------

    def candidates(self, character_id: int) -> list[Candidate]:
        """The character's candidates (`drafts.candidates`), read again only when the files
        they read changed since (`drafts.candidates_digest`, taken before the lock); under the
        drafts lock, never the decisions lock."""
        digest = candidates_digest(self.ws)
        with self._drafts_lock:
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
            self._cut[name] = None
            self._cut.move_to_end(name)
            while len(self._cut) > MAX_CUT:
                self._cut.popitem(last=False)

    def character(self, character_id: int) -> CharacterEntry | None:
        return next((c for c in self.season().characters if c.id == character_id), None)

    def resume_command(self) -> str:
        return f"fukikae studio resume {self.ws.root}"


class _Reply(Exception):  # noqa: N818
    """Raised by a handler step to answer with a JSON error."""

    def __init__(
        self, status: int, error: str, message: str, *, headers: dict[str, str] | None = None,
        **extra: Any,
    ) -> None:  # fmt: skip
        super().__init__(message)
        self.status = status
        self.body = {"error": error, "message": message, **extra}
        self.headers = headers or {}


def _busy(why: str) -> _Reply:
    return _Reply(503, "busy", f"{why}; try again", headers={"Retry-After": "2"})


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
                self._json(r.status, r.body, headers=r.headers)
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
        guarded = path == "/api" or path.startswith(("/api/", "/audio/", "/video/", "/audition/"))
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
        elif m := _LINES.fullmatch(path):
            self._need(method, get, "GET")
            self._lines(int(m.group(1)), urlsplit(self.path).query)
        elif m := _COMPARE.fullmatch(path):
            self._need(method, get, "GET")
            self._compare(int(m.group(1)), m.group(2), urlsplit(self.path).query)
        elif m := _REVEAL.fullmatch(path):
            self._need(method, method == "POST", "POST")
            self._reveal(int(m.group(1)), m.group(2))
        elif m := _ACCEPT.fullmatch(path):
            self._need(method, method == "POST", "POST")
            self._accept(int(m.group(1)), m.group(2), preview=m.group(3) is not None)
        elif m := _VERSIONS.fullmatch(path):
            self._need(method, method == "POST", "POST")
            self._version(int(m.group(1)), m.group(2))
        elif m := _DRAFT.fullmatch(path):
            self._need(method, get, "GET")
            self._draft(int(m.group(1)), m.group(2))
        elif path == "/api/backends":
            self._need(method, get, "GET")
            cfg = self.server.cfg
            status = lambda cls: self.server.auditions.clone_status(cls, cfg)  # noqa: E731
            self._json(200, {"backends": backends(cfg, status)})
        elif path == "/api/auditions":
            self._need(method, method == "POST", "POST")
            self._submit()
        elif m := _JOB_CANCEL.fullmatch(path):
            self._need(method, method == "POST", "POST")
            self._drain()
            self._job(self.server.auditions.cancel(m.group(1)))
        elif m := _JOB.fullmatch(path):
            self._need(method, get, "GET")
            self._job(self.server.auditions.job(m.group(1)))
        elif path == "/api/gpu":
            self._need(method, get, "GET")
            self._json(200, self.server.auditions.status().model_dump(mode="json"))
        elif path == "/api/gpu/release":
            self._need(method, method == "POST", "POST")
            self._drain()
            self.server.auditions.release()
            self._json(200, self.server.auditions.status().model_dump(mode="json"))
        elif path.startswith("/audition/"):
            self._need(method, get, "GET")
            self._audition(path[len("/audition/") :])
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
        elif path.startswith("/video/"):
            self._need(method, get, "GET")
            self._video(path[len("/video/") :])
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
        self._json(200, self._dry_run(d))

    def _dry_run(self, d: Decision) -> dict[str, Any]:
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
        return {
            "lines": result.lines,
            "invalidations": [{"episode": i.episode, "stage": i.stage}
                              for i in result.invalidations],
            "stale": stale,
            "unchecked": unchecked,
            "decision": stamped.model_dump(mode="json"),
        }  # fmt: skip

    def _apply(self, d: Decision) -> None:
        self._json(200, self._applied(d))

    def _applied(self, d: Decision, *, previewed: bool = False) -> dict[str, Any]:
        """Apply ``d`` under the decisions lock and answer with the rebuilt character.
        ``previewed``: fill its key and external id from a dry run first (an accept, built
        server-side; its digest is its own and is kept)."""
        s = self.server
        warning: str | None = None
        with s._lock:
            try:
                if previewed:
                    filled = apply_decision(s.ws, d, dry_run=True).decision
                    d = filled if filled is not None else d
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
        return body

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
            return _Reply(404, "unknown-character", str(e))
        if isinstance(e, UnknownVersion):
            return _Reply(404, "not-found", str(e))
        if isinstance(e, NoRegistry):
            return _Reply(409, "no-registry", "the workspace has no registry yet")
        return _Reply(422, "refused", _PATHS.sub("<path>", str(e)))

    def _candidates(self, cid: int, query: str) -> None:
        self._live_character(cid)
        q = parse_qs(query, keep_blank_values=True)

        def number(key: str, default: int, lo: int, hi: int) -> int:
            raw = q.get(key, [str(default)])[-1]
            if not _NUMBER.fullmatch(raw) or not lo <= int(raw) <= hi:
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

    def _lines(self, cid: int, query: str) -> None:
        """The character's spoken cues in each ready episode's ``06-translate/cues.<lang>.json``
        (what ``POST /api/auditions`` takes as a real line), in episode and time order, with
        the clip of each cue's source span (cut on first fetch). Read-only; an episode with no
        cues for the language, or cues that can't be read, is left out. A merged-away or
        unknown character is refused as the draft routes refuse it (`_live_character`)."""
        from fukikae.artifacts import read_json
        from fukikae.models import DubCues
        from fukikae.stages.tts import MIN_REFERENCE_S, TtsStage

        self._live_character(cid)
        q = parse_qs(query, keep_blank_values=True)

        def number(key: str, default: int, lo: int, hi: int) -> int:
            raw = q.get(key, [str(default)])[-1]
            if not _NUMBER.fullmatch(raw) or not lo <= int(raw) <= hi:
                raise _Reply(400, "bad-query", f"{key} must be a whole number {lo} to {hi}")
            return int(raw)

        offset = number("offset", 0, 0, 10**9)
        limit = number("limit", 100, 1, MAX_LINES_PAGE)
        try:
            language = language_of(q.get("language", [self.server.cfg.target_language])[-1])
        except DraftError as e:
            raise _Reply(400, "bad-query", str(e)) from None
        found: list[tuple[str, Any]] = []
        for ep in self._season().episodes:
            if not ep.ready:
                continue
            path = self.server.ws.work_dir(ep.id) / "06-translate" / f"cues.{language}.json"
            if not path.is_file():
                continue
            try:
                cues = read_json(path, DubCues).cues
            except (OSError, ValueError, ArtifactVersionError):
                continue
            found += [(ep.id, c) for c in cues
                      if c.character_id == cid and TtsStage._skipped(c) is None]  # fmt: skip
        page = found[offset : offset + limit]
        out = []
        for ep_id, c in page:
            a, b = c.delivery.reference
            clip = self.server.offer(ep_id, a, b) if b - a >= MIN_REFERENCE_S else None
            out.append({
                "episode": ep_id, "cue_id": c.id, "start": c.start, "end": c.end,
                "text": c.text, "source_text": c.source_text, "source_clip": clip,
            })  # fmt: skip
        self._json(200, {"character": cid, "language": language, "total": len(found),
                         "offset": offset, "limit": limit, "lines": out})  # fmt: skip

    def _draft_body(self, draft: Draft) -> dict[str, Any]:
        clips: dict[str, str] = {}
        for v in draft.versions:
            for a in v.auditions:
                self.server.list_audition(a.wav)
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
        except (ValueError, OSError, ArtifactVersionError) as e:  # not a draft, or unreadable
            raise _Reply(409, "draft-unreadable", f"the saved draft can't be read: {_why(e)}"
                         ) from None  # fmt: skip
        return draft

    def _live_character(self, cid: int) -> None:
        """A draft route's character must be in the registry: one merged away is 409
        ``character-merged``, naming the character it was merged into (``merged_into``) as
        the decisions log records it; any other missing one is 404 ``unknown-character``.
        Nothing is opened, saved or queued for either. A registry that can't be read is left
        to the route's own refusals."""
        gone, into = character_gone(self.server.ws, cid)
        if not gone:
            return
        if into is not None:
            raise _Reply(409, "character-merged", f"character {cid} was merged into character "
                         f"{into}, and its drafts went with it; open character {into}'s draft",
                         merged_into=into)  # fmt: skip
        raise _Reply(404, "unknown-character", f"no character {cid} in the registry")

    def _draft(self, cid: int, lang: str) -> None:
        self._live_character(cid)
        if self.command == "HEAD":  # read-only: never opens or saves a draft
            try:
                draft = load_draft(self.server.ws, cid, language_of(lang))
            except DraftError as e:
                raise self._draft_refusal(e) from None
            except (ValueError, OSError, ArtifactVersionError) as e:
                raise _Reply(409, "draft-unreadable", _why(e)) from None
            if draft is None:
                raise _Reply(404, "not-found", "no draft yet (GET opens one)")
            self._json(200, {"draft": draft.model_dump(mode="json")})
            return
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
        self._live_character(cid)
        s = self.server
        with s._drafts_lock:
            draft = self._load_or_open(cid, lang)
            try:
                offered = s.candidates(cid)
                draft, version = add_version(
                    s.ws,
                    s.cfg,
                    draft,
                    req.references,
                    req.backend,
                    req.params,
                    offered=offered,
                    keep_unlisted=req.keep_unlisted,
                )
            except DraftError as e:
                raise self._draft_refusal(e) from None
        clips = {r.label(): n for r in version.references
                 if (n := s.offer(r.episode, r.start, r.end)) is not None}  # fmt: skip
        self._json(200, {"version": version.model_dump(mode="json"),
                         "meter": floor_meter(version.references), "clips": clips})  # fmt: skip

    # -- compare and accept -------------------------------------------------------------------

    def _saved_draft(self, cid: int, lang: str) -> Draft:
        """The saved draft (never opened here: compare and accept need the versions the page
        has seen); 404 without one, 409 when it can't be read."""
        try:
            with self.server._drafts_lock:
                draft = load_draft(self.server.ws, cid, language_of(lang))
        except DraftError as e:
            raise self._draft_refusal(e) from None
        except (ValueError, OSError, ArtifactVersionError) as e:
            raise _Reply(409, "draft-unreadable", f"the saved draft can't be read: {_why(e)}"
                         ) from None  # fmt: skip
        if draft is None:
            raise _Reply(404, "not-found", f"character {cid} has no {lang} draft; GET "
                         f"/api/drafts/{cid}/{lang} opens one")  # fmt: skip
        return draft

    def _version_of(self, draft: Draft, n: int) -> DraftVersion:
        try:
            return version_of(draft, n)
        except DraftError as e:
            raise self._draft_refusal(e) from None

    def _blind_key(
        self, draft: Draft, a: int, b: int, line: TypedLine | RealLine
    ) -> dict[str, int]:
        """Label -> version number for this pair and line (`abpair.blind_pair` under the
        server's seed): the same two versions on the same line keep their labels, either way
        round, until the server stops."""
        slot = f"{draft.name}:{line.model_dump_json()}"
        _, key = blind_pair(str(a), str(b), self.server.blind_seed, slot)
        return {label: int(n) for label, n in key.items()}

    def _compare(self, cid: int, lang: str, query: str) -> None:
        """Two versions' auditions of one line. A side with no audition of the line gets one
        submitted (never on HEAD, and a failed or cancelled one only with ``retry=1``); the
        page polls this route until both are done. Pairs by line, never by cue key (which
        differs by card)."""
        self._live_character(cid)
        q = parse_qs(query, keep_blank_values=True)

        def value(key: str) -> str:
            got = q.get(key)
            if not got:
                raise _Reply(400, "bad-query", f"{key} is required")
            return got[-1]

        def number(key: str) -> int:
            raw = value(key)
            if not _NUMBER.fullmatch(raw) or int(raw) < 1:
                raise _Reply(400, "bad-query", f"{key} must be a version number")
            return int(raw)

        def flag(key: str) -> bool:
            raw = q.get(key, ["0"])[-1]
            if raw not in ("0", "1"):
                raise _Reply(400, "bad-query", f"{key} is 0 or 1")
            return raw == "1"

        a, b = number("a"), number("b")
        if a == b:
            raise _Reply(400, "bad-query", "a and b must be two different versions")
        blind, retry = flag("blind"), flag("retry")
        line = _line(value("line"))
        draft = self._saved_draft(cid, lang)
        versions = {n: self._version_of(draft, n) for n in (a, b)}
        submit = self.command != "HEAD"
        key = self._blind_key(draft, a, b, line) if blind else None
        # blind: the sides are looked up and queued in label order, so the queue (and the
        # order they finish in) follows the labels, never the query
        order = [key["A"], key["B"]] if key is not None else [a, b]
        sides = {n: self._side(draft, versions[n], line, submit=submit, retry=retry,
                               blind=blind) for n in order}  # fmt: skip
        source = next((x.source for x, _ in sides.values()
                       if x is not None and x.source is not None), None)  # fmt: skip
        body: dict[str, Any] = {
            "character": cid, "language": draft.language, "line": line.model_dump(mode="json"),
            "blind": blind,
            "source_clip": (self.server.offer(source.episode, source.start, source.end)
                            if source is not None else None),
        }  # fmt: skip
        if key is None:
            body["sides"] = {"a": sides[a][1], "b": sides[b][1]}
            self._json(200, body)
            return
        state = _pair_state([sides[n][1]["state"] for n in order])
        body["state"] = state
        body["message"] = _BLIND_MESSAGES.get(state)
        line_json = line.model_dump_json()

        def url(label: str, n: int) -> str | None:
            got = sides[n][0]
            if state != "ready" or got is None:
                return None
            return "/audition/" + self.server.blind_name(draft.name, (a, b), line_json, label,
                                                         got.wav)  # fmt: skip

        body["sides"] = {label: {"wav_url": url(label, n)} for label, n in sorted(key.items())}
        self._json(200, body)

    def _side(
        self, draft: Draft, v: DraftVersion, line: TypedLine | RealLine, *, submit: bool,
        retry: bool, blind: bool = False,
    ) -> tuple[Audition | None, dict[str, Any]]:  # fmt: skip
        """One version's audition of ``line``: the draft's latest (by line), else its latest
        job's, else a new job (``state`` ``missing`` on HEAD). The job lookup and the submit
        hold the server's lock for this draft, version and line, so two compares never queue
        it twice; a failed or cancelled job (remembered by the worker past its job ring) is
        submitted again only with ``retry``. ``blind``: a refused submit names no backend or
        version."""
        s = self.server
        got = next((x for x in reversed(v.auditions) if x.line == line), None)
        side: dict[str, Any] = {"version": v.n, "state": "missing", "message": None,
                                "job": None, "audition": None, "wav_url": None}  # fmt: skip
        if got is None:
            lock = (draft.name, str(v.n), v.card_id, line.model_dump_json())
            with s.compare_lock(lock):
                job = s.auditions.latest(draft.character_id, draft.language, v.n, line)
                if job is not None and job.state == "done" and job.audition is not None:
                    got = job.audition if job.audition.card_id == v.card_id else None
                if got is None:
                    # a pending job is polled, a failed or cancelled one reported unless
                    # retry, and a done one of a replaced card submitted again
                    if (
                        job is not None
                        and job.state != "done"
                        and (job.state not in FINISHED or not retry)
                    ):
                        side.update(
                            state=job.state,
                            job=job.id,
                            message=_PATHS.sub("<path>", job.message or "") or None,
                        )
                        return None, side  # fmt: skip
                    if not submit:
                        return None, side
                    jid = self._submit_side(draft, v, line, blind=blind)
                    snap = s.auditions.job(jid)
                    side.update(state=snap.state if snap is not None else "queued", job=jid)
                    return None, side
        s.list_audition(got.wav)
        side.update(state="done", audition=got.model_dump(mode="json"),
                    wav_url=f"/audition/{got.wav}")  # fmt: skip
        return got, side

    def _submit_side(
        self, draft: Draft, v: DraftVersion, line: TypedLine | RealLine, *, blind: bool
    ) -> str:
        """Queue a compare side's audition; the refusals as `_submit` gives them, except that
        in a blind compare they name no backend or version (the status is kept)."""
        try:
            return self.server.auditions.submit(draft.character_id, draft.language, v.n, line)
        except NoSuchDraft as e:
            why = _BLIND_REFUSAL if blind else _PATHS.sub("<path>", str(e))
            raise _Reply(404, "not-found", why) from None
        except AuditionRefused as e:
            why = _BLIND_REFUSAL if blind else _PATHS.sub("<path>", str(e))
            raise _Reply(422, "refused", why) from None
        except QueueFull as e:
            raise _busy("the audition queue is full" if blind else str(e)) from None

    def _reveal(self, cid: int, lang: str) -> None:
        body = self._body()
        try:
            req = _RevealRequest.model_validate_json(body)
        except ValidationError as e:
            raise _Reply(
                400, "invalid", "not a reveal", detail=json.loads(e.json(include_url=False))
            ) from None
        self._live_character(cid)
        if req.a == req.b:
            raise _Reply(400, "bad-query", "a and b must be two different versions")
        draft = self._saved_draft(cid, lang)
        for n in (req.a, req.b):
            self._version_of(draft, n)
        self._json(200, {"key": self._blind_key(draft, req.a, req.b, req.line)})

    def _accept(self, cid: int, lang: str, *, preview: bool) -> None:
        """Accept a draft version: the decision is built here from the saved version
        (`drafts.accept_decision`, its digest the one taken when the version was made) and
        applied through `apply_decision` under the decisions lock, a dry run first. The
        response adds, per episode whose tts goes stale, the character's lines already in the
        tts cache with that card (``cached``; None with ``cached_error`` when that can't be
        worked out)."""
        body = self._body()
        try:
            req = _AcceptRequest.model_validate_json(body)
        except ValidationError as e:
            raise _Reply(
                400, "invalid", "not an accept", detail=json.loads(e.json(include_url=False))
            ) from None
        self._live_character(cid)
        s = self.server
        draft = self._saved_draft(cid, lang)
        try:
            v = version_of(draft, req.version)
            d = accept_decision(draft, req.version)
        except DraftError as e:
            raise self._draft_refusal(e) from None
        if v.card_id != req.card_id:
            raise _Reply(409, "version-changed", f"draft {draft.name} version {v.n} is card "
                         f"{v.card_id} now, not {req.card_id}: the draft was opened again since "
                         "the page loaded it; reload it")  # fmt: skip
        if not s.ws.registry_path.is_file():
            raise _Reply(409, "no-registry", f"{s.ws.root} has no registry yet")
        problems = accept_problems(s.ws, cid, v.card_id, v.references)
        if problems:
            raise _Reply(422, "refused", _PATHS.sub(
                "<path>", f"draft {draft.name} version {v.n} can't be accepted: "
                + "; ".join(problems)))  # fmt: skip
        if preview:
            out = self._dry_run(d)
        else:
            if d.digest is None:
                raise _Reply(422, "unchecked", f"draft {draft.name} version {d.version} records "
                             "no digest of the card assignment it was made against; make a "
                             "new version to accept")  # fmt: skip
            out = self._applied(d, previewed=True)
        out.update(self._cached(d, out["invalidations"]))
        self._json(200, out)

    def _cached(self, d: AcceptDecision, invalidations: list[dict[str, str]]) -> dict[str, Any]:
        episodes = sorted({i["episode"] for i in invalidations if i["stage"] == "tts"})
        s = self.server
        try:
            got = cached_lines(s.ws, s.config(), d.character, d.card_id, d.language, episodes)
        except Exception as e:  # noqa: BLE001 (a read-only report; the accept stands either way)
            traceback.print_exc(file=sys.stderr)
            return {"cached": None, "cached_error": _why(e)}
        return {"cached": got}

    # -- auditions --------------------------------------------------------------------------

    def _drain(self) -> None:
        """Read a POST body that carries nothing the route needs (none, or JSON)."""
        if self.headers.get("Content-Length") not in (None, "0"):
            self._body()

    def _submit(self) -> None:
        body = self._body()
        try:
            req = _AuditionRequest.model_validate_json(body)
        except ValidationError as e:
            raise _Reply(
                400, "invalid", "not an audition", detail=json.loads(e.json(include_url=False))
            ) from None
        self._live_character(req.character)
        try:
            job = self.server.auditions.submit(req.character, req.language, req.version, req.line)
        except NoSuchDraft as e:
            raise _Reply(404, "not-found", _PATHS.sub("<path>", str(e))) from None
        except AuditionRefused as e:
            raise _Reply(422, "refused", _PATHS.sub("<path>", str(e))) from None
        except QueueFull as e:
            raise _busy(str(e)) from None
        snap = self.server.auditions.job(job)
        assert snap is not None
        self._json(202, {"job": self._job_body(snap)})

    def _job_body(self, job: AuditionJob) -> dict[str, Any]:
        """The job as the page reads it: with the URL of its wav and, for a line with a source
        span, the clip of that span (served by ``/audio``, cut on first fetch)."""
        out = job.model_dump(mode="json")
        a = job.audition
        out["wav_url"] = f"/audition/{a.wav}" if a is not None else None
        src = a.source if a is not None else None
        out["source_clip"] = (
            self.server.offer(src.episode, src.start, src.end) if src is not None else None
        )
        if out["message"]:
            out["message"] = _PATHS.sub("<path>", out["message"])
        return out

    def _job(self, job: AuditionJob | None) -> None:
        if job is None:
            raise _Reply(404, "not-found", "no such audition job")
        self._json(200, self._job_body(job))

    def _audition(self, name: str) -> None:
        if BLIND_NAME.fullmatch(name):  # a blind side: its alias stands for one render
            real = self.server.blind_wav(name)
            if real is None:
                raise _Reply(404, "not-found", "no such audition")
            name = real
        elif not AUDITION_NAME.fullmatch(name) or name not in self.server.audition_names():
            raise _Reply(404, "not-found", "no such audition")
        try:
            dfd = os.open(self.server.ws.root, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            raise _Reply(404, "not-found", "no such audition") from None
        try:
            data = _read_regular(dfd, ["studio", "auditions", name])
        finally:
            os.close(dfd)
        if data is None:
            raise _Reply(404, "not-found", "no such audition")
        self._media("audio/wav", data)

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
        self._media("audio/wav", data)

    def _video(self, name: str) -> None:
        span = self.server.video_names().get(name) if VIDEO_NAME.fullmatch(name) else None
        if span is None:
            raise _Reply(404, "not-found", "no such video in the index")
        if self.command != "HEAD":  # HEAD answers for a clip already cut and never cuts one
            self.server.ensure_video(name, span)
        try:
            dfd = os.open(self.server.ws.root, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            raise _Reply(404, "not-found", "no such video") from None
        try:
            data = _read_regular(dfd, ["studio", "video", name])
        finally:
            os.close(dfd)
        if data is None:
            raise _Reply(404, "not-found", "no such video")
        self._media("video/mp4", data)

    def _media(self, ctype: str, data: bytes) -> None:
        """A clip, whole or the one byte range the request asks for (`_byte_range`)."""
        n = len(data)
        try:
            got = _byte_range(self.headers.get("Range"), n)
        except _Unsatisfiable:
            body = json.dumps({"error": "range", "message": f"the clip is {n} bytes"}).encode()
            self._bytes(416, "application/json; charset=utf-8", body, headers={
                "Accept-Ranges": "bytes", "Content-Range": f"bytes */{n}",
            })  # fmt: skip
            return
        if got is None:
            self._bytes(200, ctype, data, headers={"Accept-Ranges": "bytes"})
        else:
            a, b = got
            self._bytes(206, ctype, data[a : b + 1], headers={
                "Accept-Ranges": "bytes", "Content-Range": f"bytes {a}-{b}/{n}",
            })  # fmt: skip

    # -- responses --------------------------------------------------------------------------

    def _common_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")

    def _bytes(
        self, status: int, ctype: str, data: bytes, *, csp: bool = False,
        headers: dict[str, str] | None = None,
    ) -> None:  # fmt: skip
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if csp:
            self.send_header("Content-Security-Policy", _CSP)
        self._common_headers()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, status: int, obj: Any, *, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._bytes(status, "application/json; charset=utf-8", data, headers=headers)


class _VersionRequest(BaseModel):
    """``POST /api/drafts/<id>/<lang>/versions``: the references in order, the backend and
    its voice keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    references: list[RefSpan] = Field(min_length=1, max_length=MAX_REFERENCES)
    backend: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = {}
    keep_unlisted: bool = True  # carry the previous version's origin-less references


class _AcceptRequest(BaseModel):
    """``POST /api/drafts/<id>/<lang>/accept`` (and ``/accept/preview``): the version and the
    card the page showed for it (409 when the version is another card now)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int = Field(ge=1, le=10**6)
    card_id: str = Field(pattern=r"^[0-9a-f]{16}$")


class _RevealRequest(BaseModel):
    """``POST /api/drafts/<id>/<lang>/reveal``: the blind compare's two versions and line."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    a: int = Field(ge=1, le=10**6)
    b: int = Field(ge=1, le=10**6)
    line: AuditionLine


def _line(raw: str) -> TypedLine | RealLine:
    """A compare's ``line`` query value: an audition line as JSON."""
    try:
        return _LINE.validate_json(raw)
    except ValidationError as e:
        raise _Reply(
            400,
            "bad-query",
            "line must be an audition line as JSON ({kind: real, episode, "
            "cue_id} or {kind: typed, text, emotion_from})",
            detail=json.loads(e.json(include_url=False)),
        ) from None


_PENDING = frozenset({"queued", "waiting", "loading", "running"})
_BLIND_MESSAGES = {
    "failed": "an audition of this pair did not finish; compare unblinded to see why, or "
    "retry with retry=1",
    "missing": "this pair has auditions still to queue; GET the compare to queue them",
}
_BLIND_REFUSAL = "a side of this blind pair can't be auditioned; compare unblinded to see why"


def _pair_state(states: list[str]) -> str:
    """A blind pair's one state from its sides': ``rendering`` while either is queued or
    rendering, ``missing`` while either is unqueued (HEAD), ``failed`` when either failed or
    was cancelled, else ``ready``. Never per side, which would tell which renders first."""
    if any(x in _PENDING for x in states):
        return "rendering"
    if "missing" in states:
        return "missing"
    if any(x in ("failed", "cancelled") for x in states):
        return "failed"
    return "ready"


class _AuditionRequest(BaseModel):
    """``POST /api/auditions``: the draft version (character, target language, version
    number) and the line."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    character: int = Field(ge=1, le=10**9)
    language: str = Field(min_length=2, max_length=40)
    version: int = Field(ge=1, le=10**6)
    line: AuditionLine


class _Unsatisfiable(Exception):  # noqa: N818
    """A byte range that starts at or past the end of the file (416)."""


def _byte_range(header: str | None, n: int) -> tuple[int, int] | None:
    """The inclusive ``(first, last)`` byte range of one ``bytes=`` range in a Range header
    over ``n`` bytes, clamped to the end; None for no header or any form not taken (several
    ranges, another unit, last before first), which gets the whole file. `_Unsatisfiable`
    when it starts at or past the end, or asks for an empty suffix."""
    m = _RANGE.fullmatch(header.strip()) if header else None
    if m is None or not (m.group(1) or m.group(2)):
        return None
    if not m.group(1):  # the last k bytes
        k = int(m.group(2))
        if k == 0 or n == 0:
            raise _Unsatisfiable
        return max(0, n - k), n - 1
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else n - 1
    if m.group(2) and b < a:
        return None
    if a >= n:
        raise _Unsatisfiable
    return a, min(b, n - 1)


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
