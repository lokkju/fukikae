"""Voice packs: a series' cards shared between registries, keyed by character, never by id.

A pack is a directory or a ``.tar.zst`` of one (the spec's two forms). The archive form is a
plain ustar stream compressed with zstd: read front to back with a bounded decompressor, so a
pack is checked member by member as it streams, nothing in it is ever extracted by
``tarfile`` itself, and the same bytes come out of every export of the same cards (sorted
members, zeroed owners and times). A zip would add random access, which import doesn't need,
and a second container format to defend.

Layout (nothing else is accepted)::

    pack.toml                   format, series_title, tmdb, tvdb, anidb, title, author, notes
    cast.toml                   [[cast]] name, external_id, target_language, card
    cards/<16 hex>.toml         the cards, in the store's own serialization
    clips/<sha256>.wav          reference clips
    latents/<sha256>.npy        speaker latents
    vectors/<sha256>.npy        fingerprint vectors

Registry ids never appear: a character is its ``CharacterKey`` (name and optional external
character id, such as an AniDB character id), and import matches by external id, then by
name, case- and width-insensitively (``normalize_name``).

A pack is untrusted input. ``read_pack`` copies it into a private staging directory and checks
everything there before anything else sees it: member names (no absolute paths, no ``..``,
nothing outside the layout, nothing twice), member types (regular files only, never links,
devices or sparse files; in a directory pack, never a symlink), sizes (``PackLimits``: per
file, in total and the number of files, enforced on the bytes actually read, and the
decompressed stream is capped too), every card through the strict card loader and its id
recomputed from its content, every sidecar hashed against its name and referenced by a card
under that kind, clips readable as wav and at most ``CLIP_MAX_S`` long, ``.npy`` files loaded
without pickle. Nothing from a pack is executed. ``apply_import`` writes only what was checked,
removes what it wrote if anything fails, and makes every assignment in one registry
transaction, so a failed import changes nothing. Names from a pack are printed only through
``printable``, which drops control characters.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import stat
import tarfile
import tempfile
import tomllib
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import IO, Any, NamedTuple, cast

import numpy as np
import tomli_w

from fukikae.cards.card import Card, CardError, card_id, dump_card, parse_card, sidecars
from fukikae.cards.store import SIDECAR_DIRS, CardStore, SidecarKind, _atomic_write
from fukikae.config import SERIES_ID_KEYS, SeriesInfo
from fukikae.registry.store import Registry, replace_refusal

PACK_FORMAT = 1
CLIP_MAX_S = 30.0
# The extension a sidecar of each kind has in a pack (and in the store, for exports).
SIDECAR_EXT = {"clip": ".wav", "latent": ".npy", "vector": ".npy"}
_KIND_OF_DIR = {d: k for k, d in SIDECAR_DIRS.items()}
_PACK_KEYS = frozenset({"format", "series_title", "title", "author", "notes", *SERIES_ID_KEYS})
_CAST_KEYS = frozenset({"name", "external_id", "target_language", "card"})
_LANG = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*")
_CARD_FILE = re.compile(r"cards/([0-9a-f]{16})\.toml")
_SIDECAR_FILE = re.compile(r"(clips|latents|vectors)/([0-9a-f]{64})(\.[a-z0-9]+)")
TITLE_CAP = 100
NAME_CAP = 200
TEXT_CAPS = {"series_title": 300, "author": 200, "notes": 2000}
EXTERNAL_ID_CAP = 100
VECTOR_MAX_DIM = 8192


class PackError(Exception):
    """A pack that can't be written, read or trusted."""


@dataclass(frozen=True)
class PackLimits:
    """What a pack may hold: bytes per file, bytes in total, files, and bytes per TOML file."""

    max_member: int = 64 << 20
    max_total: int = 2 << 30
    max_files: int = 10_000
    max_toml: int = 1 << 20


DEFAULT_LIMITS = PackLimits()


@dataclass(frozen=True)
class CharacterKey:
    name: str
    external_id: str | None = None


@dataclass(frozen=True)
class CastEntry:
    key: CharacterKey
    target_language: str
    card: str


@dataclass(frozen=True)
class PackMeta:
    format: int
    title: str
    series_title: str = ""
    ids: dict[str, str] = field(default_factory=dict)
    author: str = ""
    notes: str = ""


@dataclass
class Pack:
    """A checked pack, staged in ``root`` (removed by ``close``): ``cards`` by id, ``sidecars``
    by ``sha256:`` ref to their staged file."""

    meta: PackMeta
    cast: list[CastEntry]
    cards: dict[str, Card]
    sidecars: dict[str, Path]
    root: Path

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> Pack:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def entry(self, name: str, target_language: str) -> CastEntry:
        """The cast entry for ``name`` (a name or an external id) in ``target_language``."""
        folded = normalize_name(name)
        hits = [
            e
            for e in self.cast
            if e.target_language == target_language
            and (normalize_name(e.key.name) == folded or e.key.external_id == name.strip())
        ]
        if len(hits) != 1:
            names = ", ".join(sorted({printable(e.key.name) for e in self.cast})) or "none"
            why = "several characters" if hits else "no character"
            raise PackError(
                f"pack {printable(self.meta.title)!r} has {why} {printable(name)!r} for "
                f"{target_language!r}; its characters: {names}"
            )
        return hits[0]


@dataclass(frozen=True)
class ExportReport:
    written: int
    left_out: list[tuple[str, str]]  # (what, why)


class Match(NamedTuple):
    key: CharacterKey
    character_id: int
    target_language: str
    card_id: str
    matched_by: str  # "external id" or "name"


class Skip(NamedTuple):
    key: CharacterKey
    character_id: int
    reason: str


class Unmatched(NamedTuple):
    key: CharacterKey
    target_language: str
    reason: str


@dataclass(frozen=True)
class ImportPlan:
    assign: list[Match]
    skipped: list[Skip]
    unmatched: list[Unmatched]
    unchanged: list[Match]
    replace: bool = False


# -- text ---------------------------------------------------------------------------------


def printable(text: str) -> str:
    """``text`` without control and format characters, safe to print on a terminal."""
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def normalize_name(s: str) -> str:
    """The form names are matched in: NFKC (width), casefold (case), no control or format
    characters, whitespace collapsed."""
    return " ".join(printable(unicodedata.normalize("NFKC", s)).casefold().split())


def _plain_text(value: Any, what: str, cap: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise PackError(f"{what} must be a string, got {type(value).__name__}")
    text = unicodedata.normalize("NFC", value)
    if len(text) > cap:
        raise PackError(f"{what} is {len(text)} characters; the most is {cap}")
    if required and not normalize_name(text):
        raise PackError(f"{what} must not be empty")
    return text


def _external_id(value: Any, what: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise PackError(f"{what} must be a string or an integer, got {type(value).__name__}")
    text = str(value).strip()
    if not text or len(text) > EXTERNAL_ID_CAP or printable(text) != text:
        raise PackError(f"{what} {printable(text)!r} is not a usable id")
    return text


# -- export ---------------------------------------------------------------------------------


def _needs_clips(card: Card, cards: dict[str, Card]) -> bool:
    if card.voice.source == "clone" or card.voice.references:
        return True
    blend = card.voice.blend
    if blend and blend["with"].startswith("card:"):
        other = cards.get(blend["with"].removeprefix("card:"))
        return other is not None and _needs_clips(other, cards)
    return False


def _with_blends(store: CardStore, cid: str, out: dict[str, Card]) -> None:
    """``cid`` and every card it blends with, into ``out``."""
    if cid in out:
        return
    card = store.get(cid)
    out[cid] = card
    blend = card.voice.blend
    if blend and blend["with"].startswith("card:"):
        _with_blends(store, blend["with"].removeprefix("card:"), out)


def _clip_seconds(path: Path) -> float:
    from scipy.io import wavfile

    rate, data = wavfile.read(path, mmap=True)
    if rate <= 0:
        raise ValueError(f"sample rate {rate}")
    return len(data) / rate


def _pack_toml(series: SeriesInfo, title: str, author: str, notes: str) -> dict[str, Any]:
    d: dict[str, Any] = {"format": PACK_FORMAT, "title": title}
    if series.title:
        d["series_title"] = series.title
    for key in SERIES_ID_KEYS:
        if key in series.ids:
            d[key] = series.ids[key]
    if author:
        d["author"] = author
    if notes:
        d["notes"] = notes
    return d


def export_selection(
    reg: Registry,
    store: CardStore,
    *,
    characters: list[int] | None,
    languages: list[str] | None,
    no_clips: bool,
) -> tuple[list[CastEntry], dict[str, Card], list[tuple[str, str]]]:
    """The rows ``export_pack`` writes: (cast entries, every card they need by id, blends
    included, and what was left out with why). Unnamed characters, names that normalise to
    nothing, keys two characters share, unreadable cards, cards missing sidecars and, with
    ``no_clips``, cards that need audio are left out."""
    wanted = set(characters) if characters is not None else None
    left_out: list[tuple[str, str]] = []
    rows: list[tuple[CharacterKey, str, str]] = []
    for char in reg.characters():
        if wanted is not None and char.id not in wanted:
            continue
        langs = [x for x in reg.card_languages(char.id) if languages is None or x in languages]
        label = char.name or f"character {char.id}"
        if not char.name or not normalize_name(char.name):
            if langs:
                left_out.append(
                    (label, "unnamed: packs match by name; name it with `fukikae registry rename`")
                )
            continue
        for lang in langs:
            a = reg.card_for(char.id, lang)
            if a is None or a.card_id is None:
                continue
            key = CharacterKey(char.name, char.external_id or None)
            rows.append((key, lang, a.card_id))

    # a key two characters share would be ambiguous on import
    seen: dict[tuple[str, str | None, str], int] = {}
    for key, lang, _ in rows:
        k = (normalize_name(key.name), key.external_id, lang)
        seen[k] = seen.get(k, 0) + 1
    cast: list[CastEntry] = []
    cards: dict[str, Card] = {}
    for key, lang, cid in rows:
        what = f"{key.name} {lang} card {cid}"
        if seen[(normalize_name(key.name), key.external_id, lang)] > 1:
            left_out.append((what, "another character has the same name; rename one or give "
                                   "them external ids"))  # fmt: skip
            continue
        mine: dict[str, Card] = {}
        try:
            _with_blends(store, cid, mine)
        except CardError as e:
            left_out.append((what, f"not readable from the card store: {e}"))
            continue
        if no_clips and _needs_clips(mine[cid], mine):
            left_out.append((what, "needs audio clips (a clone); --no-clips leaves it out"))
            continue
        missing = [r for c in mine.values() for r in store.missing_sidecars(c)]
        if missing:
            left_out.append((what, f"sidecar(s) missing from the store: {', '.join(missing)}"))
            continue
        cards.update(mine)
        cast.append(CastEntry(key, lang, cid))

    return cast, cards, left_out


def export_pack(
    reg: Registry,
    store: CardStore,
    series: SeriesInfo,
    out: Path,
    *,
    characters: list[int] | None,
    languages: list[str] | None,
    no_clips: bool,
    title: str,
    author: str,
    notes: str = "",
) -> ExportReport:
    """Write the current cards of ``characters`` (default all) in ``languages`` (default all)
    to ``out``, a new directory or, when ``out`` ends in ``.tar.zst``, an archive. Unnamed
    characters, cards missing from the store or their sidecars, characters whose key another
    character shares, and with ``no_clips`` cards that need audio, are left out and listed.
    A clip over ``CLIP_MAX_S`` refuses the export. Nothing is written unless it all succeeds."""
    title = _plain_text(title, "pack title", TITLE_CAP, required=True)
    if printable(title) != title:
        raise PackError("pack title must not hold control characters")
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise PackError(f"{out} already exists; export to a new path")
    cast, cards, left_out = export_selection(
        reg, store, characters=characters, languages=languages, no_clips=no_clips
    )
    files: dict[str, Path | bytes] = {}
    for cid, card in sorted(cards.items()):
        files[f"cards/{cid}.toml"] = dump_card(card).encode("utf-8")
        kinds = _sidecar_kinds(card)
        for ref in sidecars(card):
            path = store.sidecar_path(ref)  # re-checks the hash
            kind = kinds[ref]
            if path.suffix.lower() != SIDECAR_EXT[kind]:
                raise PackError(
                    f"sidecar {ref} is {path.name}; a {kind} must be {SIDECAR_EXT[kind]}"
                )
            if kind == "clip":
                seconds = _clip_seconds(path)
                if seconds > CLIP_MAX_S:
                    raise PackError(
                        f"card {cid}'s clip {ref} is {seconds:.1f} s; a pack carries clips of at "
                        f"most {CLIP_MAX_S:g} s"
                    )
            files[f"{SIDECAR_DIRS[kind]}/{ref.removeprefix('sha256:')}{SIDECAR_EXT[kind]}"] = path
    files["pack.toml"] = tomli_w.dumps(_pack_toml(series, title, author, notes)).encode()
    files["cast.toml"] = tomli_w.dumps(
        {
            "cast": [
                {
                    "name": e.key.name,
                    **({"external_id": e.key.external_id} if e.key.external_id else {}),
                    "target_language": e.target_language,
                    "card": e.card,
                }
                for e in cast
            ]
        }
    ).encode()

    if out.name.endswith(".tar.zst"):
        _write_tar_zst(out, files)
    else:
        _write_dir(out, files)
    return ExportReport(written=len(cast), left_out=left_out)


def _sidecar_kinds(card: Card) -> dict[str, str]:
    v = card.voice
    kinds = {r.clip: "clip" for r in v.references}
    if v.latent:
        kinds[v.latent] = "latent"
    if v.fingerprint is not None:
        kinds[v.fingerprint.vector] = "vector"
    return kinds


def _bytes_of(data: Path | bytes) -> bytes:
    return data if isinstance(data, bytes) else data.read_bytes()


def _write_dir(out: Path, files: dict[str, Path | bytes]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}.", suffix=".tmp"))
    try:
        for name, data in sorted(files.items()):
            dest = tmp / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(dest, _bytes_of(data))
        tmp.chmod(0o755)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _write_tar_zst(out: Path, files: dict[str, Path | bytes]) -> None:
    import zstandard

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in sorted(files.items()):
            raw = _bytes_of(data)
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(raw), 0o644, 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tf.addfile(info, io.BytesIO(raw))
    _atomic_write(out, zstandard.ZstdCompressor(level=10).compress(buf.getvalue()))


# -- read -----------------------------------------------------------------------------------


class _Budget:
    def __init__(self, limits: PackLimits) -> None:
        self.limits, self.files, self.total = limits, 0, 0
        self.seen: set[str] = set()

    def directory(self, name: str) -> None:
        """A directory entry: no bytes, but it counts toward the file limit."""
        self.files += 1
        if self.files > self.limits.max_files:
            raise PackError(f"pack holds more than {self.limits.max_files} files")
        if name in self.seen:
            raise PackError(f"pack holds {name} twice")
        self.seen.add(name)

    def member(self, name: str, size: int) -> None:
        self.files += 1
        if self.files > self.limits.max_files:
            raise PackError(f"pack holds more than {self.limits.max_files} files")
        if name in self.seen:
            raise PackError(f"pack holds {name} twice")
        self.seen.add(name)
        cap = self.limits.max_toml if name.endswith(".toml") else self.limits.max_member
        if size > cap:
            raise PackError(f"{name} is too large ({size} bytes; the most is {cap})")
        self.total += size
        if self.total > self.limits.max_total:
            raise PackError(
                f"pack is larger than the total allowed ({self.limits.max_total} bytes)"
            )


def _member_name(raw: str) -> str | None:
    """The pack-relative name of an archive member, or None for the root itself; refuses
    absolute names, ``..``, backslashes and anything outside the layout."""
    if raw.startswith("/") or re.match(r"[A-Za-z]:", raw):
        raise PackError(f"pack member {printable(raw)!r} is an absolute path")
    if "\\" in raw or "\0" in raw:
        raise PackError(f"pack member {printable(raw)!r} has a backslash or NUL in its name")
    parts = PurePosixPath(raw).parts
    if ".." in parts:
        raise PackError(f"pack member {printable(raw)!r} has '..' in its path")
    parts = tuple(p for p in parts if p != ".")
    if not parts:
        return None
    return "/".join(parts)


def _allowed(name: str, is_dir: bool) -> bool:
    if is_dir:
        return name in ("cards", *SIDECAR_DIRS.values())
    return name in ("pack.toml", "cast.toml") or bool(
        _CARD_FILE.fullmatch(name) or _SIDECAR_FILE.fullmatch(name)
    )


def _check_layout(name: str, is_dir: bool) -> None:
    if not _allowed(name, is_dir):
        raise PackError(
            f"unexpected pack member {printable(name)!r}; a pack holds only pack.toml, "
            "cast.toml, cards/<id>.toml, clips/<sha256>.wav, latents/<sha256>.npy and "
            "vectors/<sha256>.npy"
        )


def _copy_bounded(src: IO[bytes], dest: Path, size: int, name: str) -> None:
    """Copy exactly ``size`` bytes (no more) into a new file ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        left = size
        while left > 0:
            chunk = src.read(min(left, 1 << 20))
            if not chunk:
                raise PackError(f"{name} is shorter than its header says")
            f.write(chunk)
            left -= len(chunk)
        if src.read(1):
            raise PackError(f"{name} is longer than its size")


class _CappedStream(io.RawIOBase):
    """A decompressed stream that refuses to yield more than ``cap`` bytes."""

    def __init__(self, inner: IO[bytes], cap: int) -> None:
        self.inner, self.cap, self.n = inner, cap, 0

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int:
        data = self.inner.read(len(b))
        self.n += len(data)
        if self.n > self.cap:
            raise PackError("pack decompresses to more than its size limits allow")
        b[: len(data)] = data
        return len(data)


def _scan_tar_zst(path: Path, limits: PackLimits, staging: Path | None) -> None:
    """One streaming pass over the archive with every member check; ``staging`` None reads
    and discards the data (the size pass), else copies each member into it."""
    import zstandard

    budget = _Budget(limits)
    # headers and padding: at most 2 x 512 bytes per member, plus the end blocks
    cap = limits.max_total + (limits.max_files + 4) * 1536 + 20 * 512
    with path.open("rb") as fh:
        dctx = zstandard.ZstdDecompressor(max_window_size=1 << 27)
        try:
            with dctx.stream_reader(fh, read_across_frames=True) as zr:
                stream = io.BufferedReader(_CappedStream(zr, cap), buffer_size=1 << 16)
                with tarfile.open(fileobj=stream, mode="r|") as tf:
                    for m in tf:
                        name = _member_name(m.name)
                        if not (m.isreg() or m.isdir()) or m.issparse():
                            raise PackError(
                                f"pack member {printable(m.name)!r} is not a regular file "
                                "(links, devices and sparse files are refused)"
                            )
                        if name is None:
                            continue
                        _check_layout(name, m.isdir())
                        if m.isdir():
                            budget.directory(name)
                            continue
                        budget.member(name, m.size)
                        f = tf.extractfile(m)
                        if f is None:  # pragma: no cover - isreg members always have data
                            raise PackError(f"{name} has no data")
                        if staging is None:
                            _discard_bounded(f, m.size, name)
                        else:
                            _copy_bounded(f, staging / name, m.size, name)
        except (tarfile.TarError, zstandard.ZstdError, EOFError) as e:
            raise PackError(f"{path} is not a readable .tar.zst pack: {e}") from e


def _discard_bounded(src: IO[bytes], size: int, name: str) -> None:
    left = size
    while left > 0:
        chunk = src.read(min(left, 1 << 20))
        if not chunk:
            raise PackError(f"{name} is shorter than its header says")
        left -= len(chunk)


def _stage_tar_zst(path: Path, staging: Path, limits: PackLimits) -> None:
    """Check every member's name, type and size over the whole stream first, writing
    nothing, so a bomb or an oversized pack is refused before anything is staged; then stage
    it in a second pass, with the same checks again."""
    _scan_tar_zst(path, limits, None)
    _scan_tar_zst(path, limits, staging)


def _stage_dir(path: Path, staging: Path, limits: PackLimits) -> None:
    budget = _Budget(limits)

    def walk(d: Path, rel: str) -> Iterator[tuple[os.DirEntry[str], str]]:
        with os.scandir(d) as it:
            for entry in sorted(it, key=lambda e: e.name):
                name = f"{rel}/{entry.name}" if rel else entry.name
                yield entry, name
                if entry.is_dir(follow_symlinks=False):
                    yield from walk(Path(entry.path), name)

    for entry, name in walk(path, ""):
        if entry.is_symlink():
            raise PackError(f"pack member {printable(name)!r} is a symlink; packs hold no links")
        is_dir = entry.is_dir(follow_symlinks=False)
        if not is_dir and not entry.is_file(follow_symlinks=False):
            raise PackError(f"pack member {printable(name)!r} is not a regular file")
        _check_layout(name, is_dir)
        if is_dir:
            budget.directory(name)
            continue
        # no symlink followed, and a FIFO swapped in after the scan can't block the open
        fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as src:
            st = os.fstat(src.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise PackError(f"pack member {printable(name)!r} is not a regular file")
            budget.member(name, st.st_size)
            _copy_bounded(src, staging / name, st.st_size, name)


def _toml(staging: Path, name: str) -> dict[str, Any]:
    path = staging / name
    if not path.is_file():
        raise PackError(f"pack has no {name}")
    try:
        return tomllib.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, RecursionError) as e:
        raise PackError(f"{name} is not valid UTF-8 TOML: {type(e).__name__}") from e


def _read_meta(raw: dict[str, Any]) -> PackMeta:
    unknown = sorted(set(raw) - _PACK_KEYS)
    if unknown:
        raise PackError(f"pack.toml has unknown key(s): {', '.join(map(printable, unknown))}")
    fmt = raw.get("format")
    if isinstance(fmt, bool) or not isinstance(fmt, int) or fmt < 1:
        raise PackError(f"pack.toml format must be a positive integer, got {fmt!r}")
    if fmt > PACK_FORMAT:
        raise PackError(
            f"pack format {fmt} is newer than this fukikae reads ({PACK_FORMAT}); upgrade fukikae"
        )
    title = _plain_text(raw.get("title"), "pack.toml title", TITLE_CAP, required=True)
    if printable(title) != title:
        raise PackError("pack.toml title must not hold control characters")
    texts = {
        key: _plain_text(raw[key], f"pack.toml {key}", cap)
        for key, cap in TEXT_CAPS.items()
        if key in raw
    }
    ids = {key: _external_id(raw[key], f"pack.toml {key}") for key in SERIES_ID_KEYS if key in raw}
    return PackMeta(format=fmt, title=title, ids=ids, **texts)


def _read_cast(raw: dict[str, Any]) -> list[CastEntry]:
    unknown = sorted(set(raw) - {"cast"})
    if unknown:
        raise PackError(f"cast.toml has unknown key(s): {', '.join(map(printable, unknown))}")
    rows = raw.get("cast")
    if not isinstance(rows, list) or not rows:
        raise PackError("cast.toml must hold a non-empty [[cast]] array")
    cast: list[CastEntry] = []
    seen: set[tuple[str, str | None, str]] = set()
    for i, row in enumerate(rows, start=1):
        where = f"cast.toml entry {i}"
        if not isinstance(row, dict):
            raise PackError(f"{where} must be a table")
        bad = sorted(set(row) - _CAST_KEYS)
        if bad:
            raise PackError(f"{where} has unknown key(s): {', '.join(map(printable, bad))}")
        for key in ("name", "target_language", "card"):
            if key not in row:
                raise PackError(f"{where} has no {key}")
        name = _plain_text(row["name"], f"{where} name", NAME_CAP, required=True)
        ext = (
            _external_id(row["external_id"], f"{where} external_id")
            if "external_id" in row
            else None
        )
        lang = row["target_language"]
        if not isinstance(lang, str) or not _LANG.fullmatch(lang):
            raise PackError(f"{where} target_language {printable(str(lang))!r} is not a language")
        cid = row["card"]
        if not isinstance(cid, str) or not re.fullmatch(r"[0-9a-f]{16}", cid):
            raise PackError(f"{where} card {printable(str(cid))!r} is not a card id")
        k = (normalize_name(name), ext, lang)
        if k in seen:
            raise PackError(f"cast.toml lists {printable(name)!r} for {lang!r} twice")
        seen.add(k)
        cast.append(CastEntry(CharacterKey(name, ext), lang, cid))
    return cast


def _check_npy(path: Path, kind: str) -> None:
    try:
        # mapped, not read: a header claiming more data than the file holds fails here instead
        # of allocating it; no pickle, so no code runs
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError, EOFError) as e:
        raise PackError(f"{kind} {path.name} is not a plain .npy array: {e}") from e
    if not isinstance(arr, np.ndarray) or arr.dtype.kind not in "fiu" or arr.size == 0:
        raise PackError(f"{kind} {path.name} must be a non-empty numeric array")
    if arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise PackError(f"{kind} {path.name} holds non-finite values")
    if kind == "vector" and (arr.ndim != 1 or arr.size > VECTOR_MAX_DIM):
        raise PackError(f"vector {path.name} must be one-dimensional, at most {VECTOR_MAX_DIM}")


def _check_clip(path: Path) -> None:
    try:
        seconds = _clip_seconds(path)
    except Exception as e:  # scipy raises ValueError, struct.error, ... on bad files
        raise PackError(f"clip {path.name} is not a readable wav: {e}") from e
    if seconds > CLIP_MAX_S:
        raise PackError(f"clip {path.name} is {seconds:.1f} s; the most is {CLIP_MAX_S:g} s")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_staged(staging: Path) -> Pack:
    meta = _read_meta(_toml(staging, "pack.toml"))
    cast = _read_cast(_toml(staging, "cast.toml"))
    cards: dict[str, Card] = {}
    cards_dir = staging / "cards"
    for path in sorted(cards_dir.iterdir()) if cards_dir.is_dir() else []:
        cid = path.stem
        try:
            card = parse_card(path.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, RecursionError) as e:
            raise PackError(f"cards/{path.name} is not a readable card: {type(e).__name__}") from e
        except CardError as e:
            raise PackError(f"cards/{path.name}: {e}") from e
        actual = card_id(card)
        if actual != cid:
            raise PackError(
                f"cards/{path.name} holds card {actual}, not {cid} (tampered or edited)"
            )
        cards[cid] = card
    used: set[str] = set()
    for e in cast:
        card = cards.get(e.card)
        if card is None:
            raise PackError(f"cast.toml names card {e.card}, which the pack doesn't hold")
        if card.target_language != e.target_language:
            raise PackError(
                f"card {e.card} is for {card.target_language!r}, and cast.toml lists it for "
                f"{e.target_language!r}"
            )
        _closure(e.card, cards, used)
    extra = sorted(set(cards) - used)
    if extra:
        raise PackError(f"pack holds card(s) no cast entry uses: {', '.join(extra)}")

    kinds: dict[str, str] = {}
    for card in cards.values():
        for ref, kind in _sidecar_kinds(card).items():
            if kinds.setdefault(ref, kind) != kind:
                raise PackError(f"sidecar {ref} is used as both a {kinds[ref]} and a {kind}")
    found: dict[str, Path] = {}
    for sub, kind in _KIND_OF_DIR.items():
        d = staging / sub
        for path in sorted(d.iterdir()) if d.is_dir() else []:
            m = _SIDECAR_FILE.fullmatch(f"{sub}/{path.name}")
            assert m is not None  # the layout check passed
            if m.group(3) != SIDECAR_EXT[kind]:
                raise PackError(f"{sub}/{path.name}: a {kind} must be {SIDECAR_EXT[kind]}")
            ref = f"sha256:{m.group(2)}"
            actual = _sha256(path)
            if actual != m.group(2):
                raise PackError(
                    f"{sub}/{path.name} hash mismatch: its bytes hash to sha256:{actual}"
                )
            if kinds.get(ref) != kind:
                raise PackError(f"{sub}/{path.name} is not a {kind} any card in the pack uses")
            if kind == "clip":
                _check_clip(path)
            else:
                _check_npy(path, kind)
            found[ref] = path
    missing = sorted(set(kinds) - set(found))
    if missing:
        raise PackError(f"pack is missing sidecar(s) its cards need: {', '.join(missing)}")
    return Pack(meta=meta, cast=cast, cards=cards, sidecars=found, root=staging)


def _closure(cid: str, cards: dict[str, Card], used: set[str]) -> None:
    if cid in used:
        return
    used.add(cid)
    blend = cards[cid].voice.blend
    if blend and blend["with"].startswith("card:"):
        other = blend["with"].removeprefix("card:")
        if other not in cards:
            raise PackError(f"card {cid} blends with card {other}, which the pack doesn't hold")
        _closure(other, cards, used)


def read_pack(path: Path, *, limits: PackLimits | None = None) -> Pack:
    """Stage and check the pack at ``path`` (a directory or a ``.tar.zst``); raises
    ``PackError`` naming the first problem, leaving nothing behind."""
    limits = limits or DEFAULT_LIMITS
    path = Path(path)
    staging = Path(tempfile.mkdtemp(prefix="fukikae-pack-"))
    try:
        if path.is_dir():
            _stage_dir(path, staging, limits)
        elif path.is_file() and path.name.endswith(".tar.zst"):
            _stage_tar_zst(path, staging, limits)
        elif path.exists():
            raise PackError(f"{path} is neither a pack directory nor a .tar.zst")
        else:
            raise PackError(f"no pack at {path}")
        return _check_staged(staging)
    except OSError as e:  # a member that vanished or changed type under us, a full disk
        shutil.rmtree(staging, ignore_errors=True)
        raise PackError(f"could not read the pack at {path}: {e}") from e
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# -- series check ---------------------------------------------------------------------------


def series_check(pack: Pack, series: SeriesInfo) -> tuple[bool, str]:
    """(passes, why). Refuses when an id both sides have differs; with no id on both sides,
    matches normalized titles (passing with a warning that it was by title)."""
    ours = {k: str(v).strip() for k, v in series.ids.items() if k in SERIES_ID_KEYS}
    theirs = pack.meta.ids
    shared = sorted(set(ours) & set(theirs))
    conflicts = [k for k in shared if ours[k] != theirs[k]]
    if conflicts:
        detail = "; ".join(f"{k} {theirs[k]} in the pack, {ours[k]} here" for k in conflicts)
        return False, f"the pack is for another series ({detail})"
    if shared:
        return True, f"series matched by {', '.join(shared)}"
    a, b = normalize_name(pack.meta.series_title), normalize_name(series.title)
    if a and a == b:
        return True, (
            f"warning: series matched by title only ({printable(pack.meta.series_title)!r}); "
            "set tmdb, tvdb or anidb in [series] to match by id"
        )
    return False, (
        f"the pack's series {printable(pack.meta.series_title)!r} is not this series "
        f"{printable(series.title)!r}, and they share no id to match by"
    )


# -- import ---------------------------------------------------------------------------------


def plan_import(pack: Pack, reg: Registry, *, replace: bool) -> ImportPlan:
    """Match each cast entry to a character by external id, then by normalized name (never to
    a character whose own external id differs), and sort the matches into assignments,
    skipped protected characters (unless ``replace``) and already-current cards."""
    chars = reg.characters()
    by = f"import:{pack.meta.title}"
    assign: list[Match] = []
    skipped: list[Skip] = []
    unmatched: list[Unmatched] = []
    unchanged: list[Match] = []
    taken: dict[tuple[int, str], CharacterKey] = {}
    for e in pack.cast:
        key, lang = e.key, e.target_language
        hits, how = [], "name"
        if key.external_id:
            hits = [c for c in chars if (c.external_id or "").strip() == key.external_id]
            how = "external id"
        if not hits:
            folded = normalize_name(key.name)
            hits = [
                c
                for c in chars
                if c.name
                and normalize_name(c.name) == folded
                and not (
                    key.external_id and c.external_id and c.external_id.strip() != key.external_id
                )
            ]
            how = "name"
        if not hits:
            unmatched.append(Unmatched(key, lang, "no character with this name or external id"))
            continue
        if len(hits) > 1:
            ids = ", ".join(str(c.id) for c in hits)
            unmatched.append(Unmatched(key, lang, f"several characters match by {how} ({ids})"))
            continue
        char = hits[0]
        if (char.id, lang) in taken:
            other = taken[(char.id, lang)]
            skipped.append(Skip(key, char.id, f"{lang}: character also matched by pack entry "
                                              f"{printable(other.name)!r}"))  # fmt: skip
            continue
        taken[(char.id, lang)] = key
        m = Match(key, char.id, lang, e.card, how)
        cur = reg.card_for(char.id, lang)
        if cur is not None and cur.card_id == e.card:
            unchanged.append(m)
        elif not replace and (why := replace_refusal(cur, by)) is not None:
            skipped.append(Skip(key, char.id, f"{lang}: {why}; pass --replace to replace it"))
        else:
            assign.append(m)
    return ImportPlan(assign, skipped, unmatched, unchanged, replace)


@dataclass
class CopyReport:
    """What ``copy_cards`` did: files it created, files it rewrote (with their old bytes, for
    ``rollback``), cards whose fingerprint it merged in, and fingerprint conflicts it left."""

    created: list[Path] = field(default_factory=list)
    rewritten: dict[Path, bytes] = field(default_factory=dict)
    merged: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def rollback(self) -> None:
        """Put the store back as it was: remove what was created, restore what was rewritten."""
        for p in self.created:
            p.unlink(missing_ok=True)
        for p, old in self.rewritten.items():
            _atomic_write(p, old)


def copy_cards(pack: Pack, store: CardStore, ids: list[str]) -> CopyReport:
    """Copy ``ids`` (with the cards they blend with and every sidecar) from the checked pack
    into ``store``. A card the store already holds keeps its file, except (ruling CD22) when
    it has no fingerprint and the pack's copy has one: the fingerprint is merged in (the id
    doesn't change; the fingerprint is outside it) and the old bytes kept for rollback. When
    both have fingerprints and they differ, the local one stays and the conflict is reported.
    On any failure everything it did is rolled back before the error propagates."""
    report = CopyReport()
    try:
        wanted: set[str] = set()
        for cid in ids:
            _closure(cid, pack.cards, wanted)
        for cid in sorted(wanted):
            card = pack.cards[cid]
            theirs = card.voice.fingerprint
            local = store.get(cid) if store.has(cid) else None
            merge = local is not None and local.voice.fingerprint is None and theirs is not None
            if local is not None and theirs is not None and local.voice.fingerprint is not None:
                mine = local.voice.fingerprint
                if mine != theirs:
                    report.conflicts.append(
                        f"card {cid}: kept the local fingerprint ({printable(mine.embedder)}, "
                        f"{mine.vector}, threshold {mine.threshold:g}); the pack's differs "
                        f"({printable(theirs.embedder)}, {theirs.vector}, threshold "
                        f"{theirs.threshold:g})"
                    )
            for ref, kind in _sidecar_kinds(card).items():
                if kind == "vector" and local is not None and not merge:
                    continue
                new = store.adopt_sidecar(pack.sidecars[ref], ref, cast(SidecarKind, kind))
                if new is not None:
                    report.created.append(new)
            path = store.card_path(cid)
            if local is None:
                report.created.append(path)
                store.put(card)
            elif merge:
                report.rewritten[path] = path.read_bytes()
                store.put(replace(local, voice=replace(local.voice, fingerprint=theirs)))
                report.merged.append(cid)
        return report
    except BaseException:
        report.rollback()
        raise


def apply_import(plan: ImportPlan, pack: Pack, reg: Registry, store: CardStore) -> CopyReport:
    """Copy the planned cards into the store (merging fingerprints into cards it already
    holds, assigned or unchanged, CD22) and assign them, ``by = import:<pack title>`` and
    unlocked, in one registry transaction; on any failure the store is rolled back."""
    ids = [m.card_id for m in plan.assign] + [m.card_id for m in plan.unchanged]
    if not ids:
        return CopyReport()
    report = copy_cards(pack, store, ids)
    by = f"import:{pack.meta.title}"
    try:
        with reg.transaction():
            for m in plan.assign:
                note = f"pack key {printable(m.key.name)}" + (
                    f", external id {m.key.external_id}" if m.key.external_id else ""
                )
                reg.assign_card(
                    m.character_id, m.target_language, m.card_id, by=by, locked=False,
                    override=plan.replace, note=note,
                )  # fmt: skip
    except BaseException:
        report.rollback()
        raise
    return report


__all__ = [
    "CLIP_MAX_S",
    "DEFAULT_LIMITS",
    "PACK_FORMAT",
    "CastEntry",
    "CharacterKey",
    "CopyReport",
    "ExportReport",
    "ImportPlan",
    "Match",
    "Pack",
    "PackError",
    "PackLimits",
    "PackMeta",
    "Skip",
    "Unmatched",
    "apply_import",
    "copy_cards",
    "export_pack",
    "normalize_name",
    "plan_import",
    "printable",
    "read_pack",
    "series_check",
]
