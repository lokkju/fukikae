"""Cast priors: the operator's hand-written ``<ws>/cast.toml`` (Task 22, decision 10).

```toml
[[cast]]
name = "Natsu Dragneel"         # required; keyed as the names stage keys a subtitle Name
gender = "m"                    # optional: "f" or "m" (voice register)
external_id = "tmdb:12345"      # optional: a string or an integer
aliases = ["Natsu", "Salamander"]  # optional: other spellings, for matching only
```

`read_cast` reads it strictly: only ``[[cast]]`` at the top, only those four keys in an
entry (an unknown key fails, naming the file and the entry's line), every name and alias
keying someone (`fukikae.registry.store.alias_key`) and none of them, nor any external id,
listed twice. An absent file is an empty cast. A key naming a voice (``voice``, ``card``,
``stock`` and the like) is refused with D41's reason: every character is cloned from its own
speech, so a cast list never names one.

The cast is a suggestion below the operator, never applied by itself. The season index
(`fukikae.studio.season.build_season`) carries it (`SeasonDoc.cast`) and matches each
character to an entry (`match_cast`, `CharacterEntry.cast`); the page offers the cast names in
the name field, shows `queue_header`, and offers a matched entry's gender as one click, which
posts `cast_gender_decision` and so is stored as an operator gender (``gender_by =
"operator"``), never ``cast``. An operator gender is never offered over. Cast aliases only
match an entry to a character: they never become registry aliases, the names stage's
evidence. Only the studio reads this file, so cast names never reach the names stage,
identity or eval, and held-out episodes stay blind to them.
"""

from __future__ import annotations

import os
import re
import stat
import tomllib
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fukikae.studio.model import CastEntry, CastMatch, GenderDecision, SeasonDoc

if TYPE_CHECKING:
    from fukikae.registry.store import Character
    from fukikae.studio.model import CharacterEntry
    from fukikae.workspace import Workspace

CAST_FILE = "cast.toml"
CAST_KEYS = frozenset({"name", "gender", "external_id", "aliases"})
# keys that would name a voice: refused with D41's reason rather than as merely unknown
VOICE_KEYS = frozenset({
    "voice", "voices", "voice_id", "card", "cards", "stock", "clone", "backend", "speaker",
    "reference", "references",
})  # fmt: skip
GENDERS = ("f", "m")
NAME_CAP = 200  # characters in a name or alias
ALIASES_MAX = 50
EXTERNAL_ID_CAP = 200
FILE_CAP = 1 << 20  # bytes
_HEADER = re.compile(r"""^\s*\[\[\s*(?:cast|"cast"|'cast')\s*\]\]""")


class CastError(ValueError):
    """``cast.toml`` can't be read: the message names the file and, where it can, the line."""


def cast_path(ws: Workspace | Path) -> Path:
    root = ws if isinstance(ws, Path) else ws.root
    return Path(root) / CAST_FILE


def _printable(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def _header_lines(text: str) -> list[int]:
    return [i for i, line in enumerate(text.splitlines(), start=1) if _HEADER.match(line)]


def _name(value: Any, where: str, what: str) -> tuple[str, str]:
    """``value`` as a name (NFC) and its key; CastError when it isn't one or keys nobody."""
    from fukikae.registry.store import alias_key

    if not isinstance(value, str):
        raise CastError(f"{where}: {what} must be a string, got {type(value).__name__}")
    text = unicodedata.normalize("NFC", value).strip()
    if len(text) > NAME_CAP:
        raise CastError(f"{where}: {what} is {len(text)} characters; the most is {NAME_CAP}")
    if _printable(text) != text:
        raise CastError(f"{where}: {what} {_printable(text)!r} holds control characters")
    key = alias_key(text) if text else None
    if key is None:
        raise CastError(
            f"{where}: {what} {text!r} names nobody (two speakers, a group word, a sign or a "
            "song, or blank)"
        )
    return text, key


def _external_id(value: Any, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise CastError(
            f"{where}: external_id must be a string or an integer, got {type(value).__name__}"
        )
    text = str(value).strip()
    if not text or len(text) > EXTERNAL_ID_CAP or _printable(text) != text:
        raise CastError(f"{where}: external_id {_printable(text)!r} is not a usable id")
    return text


def _entry(row: Any, where: str, line: int | None) -> tuple[CastEntry, list[str]]:
    """One ``[[cast]]`` table as an entry, with the keys its name and aliases answer to."""
    if not isinstance(row, dict):
        raise CastError(f"{where}: a [[cast]] entry must be a table")
    voice = sorted(set(row) & VOICE_KEYS)
    if voice:
        raise CastError(
            f"{where}: {', '.join(voice)} would name a voice, and cast.toml names none (D41: "
            "every character is cloned from its own speech; tune voices in the studio)"
        )
    unknown = sorted(set(row) - CAST_KEYS)
    if unknown:
        raise CastError(
            f"{where}: unknown key(s): {', '.join(map(_printable, unknown))}; an entry takes "
            f"{', '.join(sorted(CAST_KEYS))}"
        )
    if "name" not in row:
        raise CastError(f"{where}: no name")
    name, key = _name(row["name"], where, "name")
    gender = row.get("gender")
    if gender is not None and gender not in GENDERS:
        raise CastError(f'{where}: gender must be "f" or "m" (voice register), not {gender!r}')
    ext = _external_id(row["external_id"], where) if "external_id" in row else None
    raw = row.get("aliases", [])
    if not isinstance(raw, list):
        raise CastError(f"{where}: aliases must be a list of names, got {type(raw).__name__}")
    if len(raw) > ALIASES_MAX:
        raise CastError(f"{where}: {len(raw)} aliases; the most is {ALIASES_MAX}")
    aliases: list[str] = []
    keys = [key]
    for a in raw:
        text, k = _name(a, where, "alias")
        aliases.append(text)
        keys.append(k)
    entry = CastEntry(name=name, gender=gender, external_id=ext, aliases=aliases, line=line)
    return entry, keys


def _read_capped(path: Path) -> bytes | None:
    """At most ``FILE_CAP + 1`` bytes of ``path``, None when it doesn't exist. Opened without
    blocking and refused unless it is a regular file (after symlinks), so a FIFO, a device
    such as ``/dev/zero`` or a huge file never makes a reader wait or read without limit."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    except FileNotFoundError:
        return None
    except OSError as e:
        raise CastError(f"{path}: {e}") from e
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CastError(f"{path}: not a regular file")
        chunks: list[bytes] = []
        left = FILE_CAP + 1
        while left > 0:
            chunk = os.read(fd, min(left, 1 << 16))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)
    except OSError as e:
        raise CastError(f"{path}: {e}") from e
    finally:
        os.close(fd)


def cast_digest(ws: Workspace | Path) -> str | None:
    """sha256 of what `read_cast` would read (through the same cap), None with no file; a
    file it can't read hashes its error, so the index still notices a change."""
    import hashlib

    try:
        data = _read_capped(cast_path(ws))
    except CastError as e:
        return hashlib.sha256(f"error: {e}".encode()).hexdigest()
    return None if data is None else hashlib.sha256(data).hexdigest()


def read_cast(ws: Workspace | Path) -> list[CastEntry]:
    """``<ws>/cast.toml``'s entries in file order (the module docstring); ``[]`` when there
    is no such file. CastError, naming the file and the line where it can, otherwise."""
    path = cast_path(ws)
    data = _read_capped(path)
    if data is None:
        return []
    if len(data) > FILE_CAP:
        raise CastError(f"{path}: more than {FILE_CAP} bytes; the most is {FILE_CAP}")
    try:
        text = data.decode("utf-8-sig")  # a leading BOM (an editor's) is dropped
        raw = tomllib.loads(text)
    except UnicodeDecodeError as e:
        raise CastError(f"{path}: not UTF-8 ({e})") from e
    except tomllib.TOMLDecodeError as e:
        raise CastError(f"{path}: {e}") from e
    unknown = sorted(set(raw) - {"cast"})
    if unknown:
        raise CastError(
            f"{path}: unknown key(s): {', '.join(map(_printable, unknown))}; the file holds "
            "only [[cast]] entries"
        )
    rows = raw.get("cast", [])
    if not isinstance(rows, list):
        raise CastError(f"{path}: cast must be an array of [[cast]] tables")
    headers = _header_lines(text)
    lines: list[int | None] = list(headers) if len(headers) == len(rows) else [None] * len(rows)
    out: list[CastEntry] = []
    seen: dict[str, tuple[str, str]] = {}  # key -> (name as written, where)
    ids: dict[str, str] = {}
    for i, (row, line) in enumerate(zip(rows, lines, strict=True), start=1):
        at = f"line {line}" if line is not None else f"entry {i}"
        where = f"{path} {at}"
        entry, keys = _entry(row, where, line)
        written = [entry.name, *entry.aliases]
        for k, name in zip(keys, written, strict=True):
            if k in seen and seen[k][1] != at:
                first, first_at = seen[k]
                raise CastError(
                    f"{where}: {name!r} answers to the same name as {first!r} at {first_at}; "
                    "each name and alias belongs to one entry"
                )
            seen.setdefault(k, (name, at))
        if entry.external_id is not None:
            if entry.external_id in ids:
                raise CastError(
                    f"{where}: external_id {entry.external_id!r} is already the entry's at "
                    f"{ids[entry.external_id]}"
                )
            ids[entry.external_id] = at
        out.append(entry)
    return out


def match_cast(cast: list[CastEntry], c: Character) -> CastMatch | None:
    """The entry ``c`` matches: by external id first, then by its name's key against each
    entry's name and aliases (skipping an entry whose external id differs from ``c``'s, when
    both carry one). Its ``gender`` is offered only when the entry has one, the
    operator stored none and ``c`` doesn't already store it."""
    from fukikae.registry.store import alias_key

    found: tuple[int, Literal["external_id", "name", "alias"]] | None = None
    if c.external_id is not None:
        found = next(((i, "external_id") for i, e in enumerate(cast)
                      if e.external_id == c.external_id), None)  # fmt: skip
    key = alias_key(c.name) if found is None and c.name else None
    if key is not None:
        for i, e in enumerate(cast):
            if c.external_id is not None and e.external_id not in (None, c.external_id):
                continue  # the same name, but another person's id
            if alias_key(e.name) == key:
                found = (i, "name")
                break
            if any(alias_key(a) == key for a in e.aliases):
                found = (i, "alias")
                break
    if found is None:
        return None
    i, by = found
    e = cast[i]
    offer = e.gender
    if c.gender_by == "operator" or c.gender == offer:
        offer = None
    where = f"line {e.line}" if e.line is not None else f"entry {i + 1}"
    return CastMatch(entry=i, by=by, gender=offer, source=f"{CAST_FILE} {where}")


def mark_ambiguous(matches: dict[int, CastMatch | None]) -> dict[int, CastMatch | None]:
    """``matches`` (character id to match) with each match whose entry another character
    also matches naming those characters (`CastMatch.ambiguous_with`), for the page to say
    so. Nothing changes otherwise: an ambiguous offer is still only an offer."""
    by_entry: dict[int, list[int]] = {}
    for cid, m in matches.items():
        if m is not None:
            by_entry.setdefault(m.entry, []).append(cid)
    out: dict[int, CastMatch | None] = {}
    for cid, m in matches.items():
        others = [] if m is None else sorted(x for x in by_entry[m.entry] if x != cid)
        out[cid] = m.model_copy(update={"ambiguous_with": others}) if m and others else m
    return out


def cast_gender_decision(entry: CharacterEntry) -> GenderDecision:
    """The gender decision the page's one click posts for a character's cast match: made
    against the rows the index showed (its digest), applied as an operator gender.
    ValueError when the match offers no gender."""
    m = entry.cast
    if m is None or m.gender is None:
        raise ValueError(f"character {entry.id}: the cast has nothing to offer for its gender")
    return GenderDecision(
        op="gender", character=entry.id, gender=m.gender, digest=entry.digest, key=entry.key,
        external_id=entry.external_id,
    )  # fmt: skip


def queue_header(doc: SeasonDoc) -> str | None:
    """The review queue's cast line, None without a cast: how many speaking roles the cast
    lists against the season's characters (ignored ones left out)."""
    if not doc.cast:
        return None
    n = len(doc.cast)
    m = sum(1 for c in doc.characters if c.kind != "ignored")
    roles = "speaking role" if n == 1 else "speaking roles"
    chars = "character" if m == 1 else "characters"
    return f"cast lists {n} {roles}; this season has {m} {chars}"


__all__ = [
    "CAST_FILE",
    "CAST_KEYS",
    "CastEntry",
    "CastError",
    "CastMatch",
    "VOICE_KEYS",
    "cast_digest",
    "cast_gender_decision",
    "cast_path",
    "mark_ambiguous",
    "match_cast",
    "queue_header",
    "read_cast",
]
