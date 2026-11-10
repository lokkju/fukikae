"""SQLite registry of actors, characters and per-episode voice exemplars.

Actors own the voice embeddings; characters belong to an actor and get one assigned character
card per target language (schema 2), with every assignment kept in ``card_history``.
Embeddings are stored as float32 bytes.
"""

from __future__ import annotations

import io
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2

# `by` values of an assignment; `import:<pack title>` is the one with a suffix. `migrated`
# (the schema-1 voice note) and `lock`/`unlock` rows are written by the registry itself.
ASSIGNED_BY = frozenset({"auto", "auto-fallback", "assign", "revert", "edit", "derive", "restock"})
# Operator actions (ruling CD2): a card they chose is never replaced by `auto` or an import
# unless the caller passes override (an import the operator confirmed with --replace). An
# imported card is never replaced by `auto` without override either (CD3).
OPERATOR_BY = frozenset({"assign", "revert", "edit", "derive"})
# Automatic sources (ruling CD9): `restock` re-pins an automatic stock card to a new model
# revision. It is no operator choice, so a restocked card stays uncurated, and it is refused
# over a locked, operator-chosen or imported card exactly as `auto` is.
# `auto-fallback` is the stock voice a character got while the clone backend could not run
# here (Task 12 fix round 2); automatic assignment replaces it with a clone once it can.
AUTOMATIC_BY = frozenset({"auto", "auto-fallback", "restock"})
_LOCK_BY = frozenset({"lock", "unlock"})
_CARD_ID = re.compile(r"[0-9a-f]{16}")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS actors (
    id INTEGER PRIMARY KEY,
    name TEXT,
    external_id TEXT,
    centroid BLOB,
    locked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS characters (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER NOT NULL REFERENCES actors(id),
    name TEXT,
    voice TEXT, -- schema 1's free-text voice, always NULL from schema 2 (card_assignments)
    locked INTEGER NOT NULL DEFAULT 0,
    external_id TEXT
);
CREATE TABLE IF NOT EXISTS exemplars (
    id INTEGER PRIMARY KEY,
    actor_id INTEGER NOT NULL REFERENCES actors(id),
    episode TEXT NOT NULL,
    cluster TEXT,
    embedding BLOB NOT NULL,
    seconds REAL
);
CREATE TABLE IF NOT EXISTS id_high_water (tbl TEXT PRIMARY KEY, last INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS projections (
    series TEXT NOT NULL,
    kind TEXT NOT NULL,
    matrix BLOB NOT NULL,
    mean BLOB NOT NULL,
    created TEXT NOT NULL,
    PRIMARY KEY (series, kind)
);
CREATE TABLE IF NOT EXISTS card_history (
    id INTEGER PRIMARY KEY,
    character_id INTEGER NOT NULL REFERENCES characters(id),
    target_language TEXT NOT NULL,
    card_id TEXT,
    assigned TEXT NOT NULL,
    by TEXT NOT NULL,
    locked INTEGER NOT NULL DEFAULT 0,
    note TEXT
);
CREATE TABLE IF NOT EXISTS card_assignments (
    character_id INTEGER NOT NULL REFERENCES characters(id),
    target_language TEXT NOT NULL,
    card_id TEXT NOT NULL,
    history_id INTEGER NOT NULL REFERENCES card_history(id),
    PRIMARY KEY (character_id, target_language)
);
CREATE INDEX IF NOT EXISTS card_history_character ON card_history(character_id, target_language);
CREATE INDEX IF NOT EXISTS exemplars_actor ON exemplars(actor_id);
CREATE INDEX IF NOT EXISTS exemplars_episode ON exemplars(episode);
"""


@dataclass(frozen=True)
class Actor:
    id: int
    name: str | None
    external_id: str | None
    centroid: np.ndarray | None
    locked: bool


@dataclass(frozen=True)
class Character:
    id: int
    actor_id: int
    name: str | None
    external_id: str | None
    locked: bool


@dataclass(frozen=True)
class CardAssignment:
    """One card_history row; ``n`` numbers a (character, target language)'s rows from 1, in
    ``(assigned, id)`` order. ``chosen_by`` is how its card was chosen: ``by`` itself, except
    on a ``lock``/``unlock`` row, which keeps the ``by`` of the row before it."""

    n: int
    character_id: int
    target_language: str
    card_id: str | None
    assigned: str  # ISO 8601, UTC
    by: str
    locked: bool
    note: str | None
    chosen_by: str


class ProtectedAssignmentError(ValueError):
    """An automatic assignment or an import would replace a locked or operator-chosen card."""


@dataclass(frozen=True)
class Exemplar:
    id: int
    actor_id: int
    episode: str
    cluster: str | None
    embedding: np.ndarray


@dataclass(frozen=True)
class ReembedResult:
    centroids: list[int]  # actors whose centroid is now the mean of their new exemplars
    kept: list[int]  # locked or exemplar-less actors whose centroid stayed as stored
    mapped: list[int]  # locked or exemplar-less actors given a mapped centroid
    projections: list[tuple[str, str, int]]  # (series, kind, dim) dropped


@dataclass(frozen=True)
class StoredProjection:
    series: str
    kind: str
    matrix: np.ndarray  # (dim, k)
    mean: np.ndarray  # (dim,)
    created: str  # ISO 8601, UTC


def _array_blob(a: np.ndarray) -> bytes:
    """An array with its shape and dtype (``np.save``), never pickled."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(a), allow_pickle=False)
    return buf.getvalue()


def _blob_array(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False)


def _to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _from_blob(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32).copy()


def _fold(text: str) -> str:
    """Case- and width-insensitive form of a name (NFKC, then casefold)."""
    return unicodedata.normalize("NFKC", text).casefold().strip()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _normalized(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


def _curated_card(a: CardAssignment | None) -> bool:
    """A current assignment someone has curated: locked, or its card not chosen automatically
    (``auto`` or ``restock``, ruling CD9)."""
    return a is not None and (a.locked or a.chosen_by not in AUTOMATIC_BY)


def replace_refusal(cur: CardAssignment | None, by: str) -> str | None:
    """Why an assignment ``by`` may not replace ``cur`` without override, or None when it may:
    the one rule (rulings CD2, CD3, CD9) that ``assign_card`` enforces and pack import plans
    with. Operator actions replace anything; ``auto``, ``restock`` and imports never replace
    a locked or operator-chosen card; ``auto`` and ``restock`` never replace an imported one."""
    if cur is None or by in OPERATOR_BY:
        return None
    imported = cur.chosen_by.startswith("import:")
    if cur.locked:
        why = "locked"
    elif cur.chosen_by in OPERATOR_BY:
        why = "operator-chosen"
    elif by in AUTOMATIC_BY and imported:
        why = "imported"
    else:
        return None
    return f"card {cur.card_id} is {why} ({cur.chosen_by})"


class Registry:
    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path, isolation_level=None)
        self._con.execute("PRAGMA foreign_keys=ON")
        self._con.execute("PRAGMA journal_mode=WAL")
        try:
            self._init_schema(path)
        except BaseException:
            self._con.close()
            raise

    def _init_schema(self, path: Path) -> None:
        """Create a new registry, or migrate an older one, in one transaction: a migration
        that fails leaves the file exactly as it was."""
        with self._tx() as con:
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    con.execute(stmt)
            row = con.execute("SELECT MAX(version) FROM schema_version").fetchone()
            if row[0] is None:
                con.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
            elif row[0] > SCHEMA_VERSION:
                raise RuntimeError(
                    f"registry {path} has schema version {row[0]}, "
                    f"newer than this fukikae supports ({SCHEMA_VERSION}); upgrade fukikae"
                )
            elif row[0] == 1:
                self._migrate_1_to_2(con)

    def _migrate_1_to_2(self, con: sqlite3.Connection) -> None:
        """Schema 1 to 2: ``characters.external_id``, the card tables (already created by
        ``_SCHEMA``), and each non-empty ``characters.voice`` moved into history as a note.

        Schema 1's voice was an English voice (the project dubs into English and nothing
        assigned it per language), so the note goes under target language ``en``.
        """
        cols = {r[1] for r in con.execute("PRAGMA table_info(characters)")}
        if "external_id" not in cols:
            con.execute("ALTER TABLE characters ADD COLUMN external_id TEXT")
        now = _now()
        for cid, voice in con.execute(
            "SELECT id, voice FROM characters WHERE voice IS NOT NULL AND TRIM(voice) != ''"
            " ORDER BY id"
        ).fetchall():
            con.execute(
                "INSERT INTO card_history(character_id, target_language, card_id, assigned, by,"
                " locked, note) VALUES (?, 'en', NULL, ?, 'migrated', 0, ?)",
                (cid, now, voice),
            )
        con.execute("UPDATE characters SET voice = NULL")
        con.execute("UPDATE schema_version SET version = 2")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        if self._con.in_transaction:  # nested: the outermost transaction commits or rolls back
            yield self._con
            return
        self._con.execute("BEGIN IMMEDIATE")
        try:
            yield self._con
        except BaseException:
            self._con.execute("ROLLBACK")
            raise
        else:
            self._con.execute("COMMIT")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group several writes into one transaction: all of them land, or none do."""
        with self._tx():
            yield

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # reads

    def actors(self) -> list[Actor]:
        rows = self._con.execute(
            "SELECT id, name, external_id, centroid, locked FROM actors ORDER BY id"
        ).fetchall()
        return [Actor(r[0], r[1], r[2], _from_blob(r[3]), bool(r[4])) for r in rows]

    def characters(self) -> list[Character]:
        rows = self._con.execute(
            "SELECT id, actor_id, name, external_id, locked FROM characters ORDER BY id"
        ).fetchall()
        return [Character(r[0], r[1], r[2], r[3], bool(r[4])) for r in rows]

    def character_by_ref(self, ref: str) -> Character:
        """A character by id, or by name case- and width-insensitively ("ＮＡＴＳＵ" finds
        "Natsu"). ValueError when nothing matches, or when several names do (listing them)."""
        chars = self.characters()
        folded = _fold(ref)
        if folded.isdecimal():
            hit = next((c for c in chars if c.id == int(folded)), None)
            if hit is not None:
                return hit
        matches = [c for c in chars if c.name is not None and _fold(c.name) == folded]
        if len(matches) == 1:
            return matches[0]
        if matches:
            listed = ", ".join(f"{c.id} ({c.name})" for c in matches)
            raise ValueError(f"{ref!r} names several characters: {listed}; use the id")
        raise ValueError(f"no character {ref!r} (by id or name); see `fukikae registry list`")

    def _history_rows(
        self, con: sqlite3.Connection, character_id: int, target_language: str
    ) -> list[tuple[int, CardAssignment]]:
        rows = con.execute(
            "SELECT id, card_id, assigned, by, locked, note FROM card_history"
            " WHERE character_id = ? AND target_language = ? ORDER BY assigned, id",
            (character_id, target_language),
        ).fetchall()
        out: list[tuple[int, CardAssignment]] = []
        chosen = "auto"
        for n, r in enumerate(rows, start=1):
            if r[3] not in _LOCK_BY:
                chosen = r[3]
            out.append(
                (
                    r[0],
                    CardAssignment(
                        n, character_id, target_language, r[1], r[2], r[3], bool(r[4]), r[5], chosen
                    ),
                )
            )
        return out

    def card_history(self, character_id: int, target_language: str) -> list[CardAssignment]:
        """Every assignment of this character in this target language, oldest first."""
        return [h for _, h in self._history_rows(self._con, character_id, target_language)]

    def _current(
        self, con: sqlite3.Connection, character_id: int, target_language: str
    ) -> CardAssignment | None:
        row = con.execute(
            "SELECT history_id FROM card_assignments WHERE character_id = ?"
            " AND target_language = ?",
            (character_id, target_language),
        ).fetchone()
        if row is None:
            return None
        return next(
            h for hid, h in self._history_rows(con, character_id, target_language) if hid == row[0]
        )

    def card_for(self, character_id: int, target_language: str) -> CardAssignment | None:
        """The character's current card assignment for this target language, if any."""
        return self._current(self._con, character_id, target_language)

    def card_languages(self, character_id: int) -> list[str]:
        """Target languages this character has a current card for."""
        return [
            r[0]
            for r in self._con.execute(
                "SELECT target_language FROM card_assignments WHERE character_id = ?"
                " ORDER BY target_language",
                (character_id,),
            )
        ]

    def exemplar_count(self, actor_id: int) -> int:
        return self._con.execute(
            "SELECT COUNT(*) FROM exemplars WHERE actor_id = ?", (actor_id,)
        ).fetchone()[0]

    def episode_counts(self, actor_id: int) -> dict[str, int]:
        """Exemplar count per episode for one actor."""
        rows = self._con.execute(
            "SELECT episode, COUNT(*) FROM exemplars WHERE actor_id = ? GROUP BY episode"
            " ORDER BY episode",
            (actor_id,),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def cluster_actors(self, episode: str, cluster: str) -> list[int]:
        """Ids of the actors holding exemplars of one episode cluster."""
        return [
            r[0]
            for r in self._con.execute(
                "SELECT DISTINCT actor_id FROM exemplars WHERE episode = ? AND cluster = ?"
                " ORDER BY actor_id",
                (episode, cluster),
            )
        ]

    def exemplar_rows(self) -> dict[int, np.ndarray]:
        """Every actor's exemplar rows, (n, dim) float32; an actor with none is absent.

        Raises ValueError for an actor whose exemplars have different dimensions (the
        embedder changed part-way): `fukikae registry reembed` recovers.
        """
        rows: dict[int, list[np.ndarray]] = {}
        for aid, blob in self._con.execute(
            "SELECT actor_id, embedding FROM exemplars ORDER BY actor_id, id"
        ):
            rows.setdefault(aid, []).append(np.frombuffer(blob, dtype=np.float32))
        for aid, r in rows.items():
            dims = sorted({len(x) for x in r})
            if len(dims) > 1:
                raise ValueError(
                    f"actor {aid} holds exemplars of different dimensions "
                    f"({', '.join(map(str, dims))}): the embedder changed part-way. Run "
                    "`fukikae registry reembed <workspace>` to re-embed every exemplar with "
                    "the configured embedder, then re-run the pipeline"
                )
        return {aid: np.vstack(r) for aid, r in rows.items()}

    def exemplars(self) -> list[Exemplar]:
        """Every exemplar, in id order (within an episode cluster, the order of its
        ``clusters.npz`` rows)."""
        return [
            Exemplar(r[0], r[1], r[2], r[3], np.frombuffer(r[4], dtype=np.float32).copy())
            for r in self._con.execute(
                "SELECT id, actor_id, episode, cluster, embedding FROM exemplars ORDER BY id"
            )
        ]

    def projections(self) -> list[StoredProjection]:
        rows = self._con.execute(
            "SELECT series, kind, matrix, mean, created FROM projections ORDER BY series, kind"
        ).fetchall()
        return [
            StoredProjection(r[0], r[1], _blob_array(r[2]), _blob_array(r[3]), r[4]) for r in rows
        ]

    def _is_curated(self, con: sqlite3.Connection, character_id: int) -> bool:
        """Named, with an external id, with a schema-1 voice migrated into history, or with a
        current card (any target language) that is locked or was not chosen by ``auto``
        (ruling CD2). Lock and unlock rows by themselves curate nothing."""
        row = con.execute(
            "SELECT name IS NOT NULL OR external_id IS NOT NULL OR EXISTS (SELECT 1 FROM"
            " card_history WHERE character_id = characters.id AND by = 'migrated')"
            " FROM characters WHERE id = ?",
            (character_id,),
        ).fetchone()
        if row is None:
            return False
        if row[0]:
            return True
        return any(
            _curated_card(self._current(con, character_id, lang))
            for (lang,) in con.execute(
                "SELECT target_language FROM card_assignments WHERE character_id = ?",
                (character_id,),
            ).fetchall()
        )

    def curated_characters(self) -> set[int]:
        """Characters someone has curated (`_is_curated`)."""
        ids = [r[0] for r in self._con.execute("SELECT id FROM characters").fetchall()]
        return {cid for cid in ids if self._is_curated(self._con, cid)}

    def protected_actors(self) -> set[int]:
        """Actors someone has curated, which pruning never deletes: locked or named, or with
        a curated character (`curated_characters`)."""
        sql = "SELECT id FROM actors WHERE locked OR name IS NOT NULL"
        actors = {r[0] for r in self._con.execute(sql)}
        curated = self.curated_characters()
        return actors | {c.actor_id for c in self.characters() if c.id in curated}

    def load_projection(self, series: str, kind: str) -> StoredProjection | None:
        row = self._con.execute(
            "SELECT matrix, mean, created FROM projections WHERE series = ? AND kind = ?",
            (series, kind),
        ).fetchone()
        if row is None:
            return None
        return StoredProjection(series, kind, _blob_array(row[0]), _blob_array(row[1]), row[2])

    # writes

    def save_projection(self, series: str, kind: str, matrix: np.ndarray, mean: np.ndarray) -> str:
        """Store the series' projection of this kind, replacing any earlier one; returns its
        creation time."""
        created = datetime.now(UTC).isoformat()
        with self._tx() as con:
            con.execute(
                "INSERT INTO projections(series, kind, matrix, mean, created)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(series, kind) DO UPDATE SET matrix = excluded.matrix,"
                " mean = excluded.mean, created = excluded.created",
                (series, kind, _array_blob(matrix), _array_blob(mean), created),
            )
        return created

    def replace_embeddings(
        self,
        embeddings: dict[int, np.ndarray],
        keep: set[int] | None = None,
        fixed: dict[int, np.ndarray] | None = None,
    ) -> ReembedResult:
        """Give every exemplar a new embedding (an embedder change), in one transaction.

        `embeddings` must cover every exemplar id, all of one dimension. Each unlocked actor
        with exemplars gets the normalized mean of its new rows as its centroid. A locked or
        exemplar-less actor with a centroid is never recomputed from exemplars (Ruling R20):
        it must be in `keep` (centroid left exactly as stored) or `fixed` (given that
        centroid, e.g. one mapped into the new space); one in neither refuses. Stored
        projections of another input dimension are deleted. Names, locks, voices, external
        ids, card assignments and the id high-water mark are untouched.
        """
        keep, fixed = set(keep or ()), dict(fixed or {})
        dims = {len(np.ravel(v)) for v in embeddings.values()}
        if len(dims) > 1:
            raise ValueError(f"new embeddings have several dimensions: {sorted(dims)}")
        with self._tx() as con:
            ids = {r[0] for r in con.execute("SELECT id FROM exemplars")}
            if ids != set(embeddings):
                raise ValueError(
                    "the exemplars changed while they were being re-embedded; run reembed again"
                )
            dim = next(iter(dims), None)
            for eid, v in embeddings.items():
                con.execute(
                    "UPDATE exemplars SET embedding = ? WHERE id = ?", (_to_blob(np.ravel(v)), eid)
                )
            centroids, kept, mapped = [], [], []
            for aid, blob, locked in con.execute(
                "SELECT id, centroid, locked FROM actors ORDER BY id"
            ).fetchall():
                rows = [
                    np.frombuffer(r[0], dtype=np.float32)
                    for r in con.execute(
                        "SELECT embedding FROM exemplars WHERE actor_id = ?", (aid,)
                    )
                ]
                if aid in fixed:
                    con.execute(
                        "UPDATE actors SET centroid = ? WHERE id = ?",
                        (_to_blob(_normalized(fixed[aid])), aid),
                    )
                    mapped.append(aid)
                elif aid in keep:
                    kept.append(aid)
                elif rows and not locked:
                    mean = np.stack(rows).mean(axis=0)
                    con.execute(
                        "UPDATE actors SET centroid = ? WHERE id = ?",
                        (_to_blob(_normalized(mean)), aid),
                    )
                    centroids.append(aid)
                elif blob is not None:
                    raise ValueError(
                        f"actor {aid} is locked or has no exemplars: its centroid must be kept "
                        "or mapped, never recomputed; run reembed again"
                    )
            dropped = []
            for p in self.projections():
                if dim is not None and p.matrix.shape[0] != dim:
                    con.execute(
                        "DELETE FROM projections WHERE series = ? AND kind = ?", (p.series, p.kind)
                    )
                    dropped.append((p.series, p.kind, int(p.matrix.shape[0])))
            return ReembedResult(centroids, kept, mapped, dropped)

    def _next_id(self, con: sqlite3.Connection, table: str) -> int:
        """An id above any this table has ever held, so a deleted id is never handed out
        again (assignments files and reports may still name it)."""
        if table not in ("actors", "characters"):
            raise ValueError(table)
        row = con.execute("SELECT last FROM id_high_water WHERE tbl = ?", (table,)).fetchone()
        top = con.execute(f"SELECT MAX(id) FROM {table}").fetchone()[0]  # noqa: S608
        nid = max(row[0] if row else 0, top or 0) + 1
        con.execute(
            "INSERT INTO id_high_water(tbl, last) VALUES (?, ?)"
            " ON CONFLICT(tbl) DO UPDATE SET last = excluded.last",
            (table, nid),
        )
        return nid

    def _insert_actor_with_character(
        self, con: sqlite3.Connection, centroid: np.ndarray
    ) -> tuple[int, int]:
        aid = self._next_id(con, "actors")
        con.execute(
            "INSERT INTO actors(id, centroid) VALUES (?, ?)",
            (aid, _to_blob(_normalized(centroid))),
        )
        cid = self._next_id(con, "characters")
        con.execute("INSERT INTO characters(id, actor_id) VALUES (?, ?)", (cid, aid))
        return aid, cid

    def create_actor_with_character(self, centroid: np.ndarray) -> tuple[int, int]:
        with self._tx() as con:
            return self._insert_actor_with_character(con, centroid)

    def add_exemplars(
        self, actor_id: int, episode: str, cluster: str, embeddings: np.ndarray, seconds: float
    ) -> None:
        rows = np.atleast_2d(np.asarray(embeddings, dtype=np.float32))
        with self._tx() as con:
            for row in rows:
                con.execute(
                    "INSERT INTO exemplars(actor_id, episode, cluster, embedding, seconds)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (actor_id, episode, cluster, _to_blob(row), seconds),
                )

    def recompute_centroid(self, actor_id: int) -> None:
        with self._tx() as con:
            self._recompute(con, actor_id)

    def _recompute(self, con: sqlite3.Connection, actor_id: int) -> None:
        """Set the centroid to the normalized mean of the actor's exemplars.

        No-op when the actor is locked, unknown, or has no exemplars.
        """
        row = con.execute("SELECT locked FROM actors WHERE id = ?", (actor_id,)).fetchone()
        if row is None or row[0]:
            return
        blobs = [
            r[0]
            for r in con.execute("SELECT embedding FROM exemplars WHERE actor_id = ?", (actor_id,))
        ]
        if not blobs:
            return
        mean = np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs]).mean(axis=0)
        con.execute(
            "UPDATE actors SET centroid = ? WHERE id = ?", (_to_blob(_normalized(mean)), actor_id)
        )

    def _append(
        self,
        con: sqlite3.Connection,
        character_id: int,
        target_language: str,
        card_id: str,
        by: str,
        locked: bool,
        note: str | None = None,
    ) -> CardAssignment:
        """Write a history row and make it the current assignment."""
        cur = con.execute(
            "INSERT INTO card_history(character_id, target_language, card_id, assigned, by,"
            " locked, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (character_id, target_language, card_id, _now(), by, int(locked), note),
        )
        con.execute(
            "INSERT INTO card_assignments(character_id, target_language, card_id, history_id)"
            " VALUES (?, ?, ?, ?) ON CONFLICT(character_id, target_language) DO UPDATE SET"
            " card_id = excluded.card_id, history_id = excluded.history_id",
            (character_id, target_language, card_id, cur.lastrowid),
        )
        current = self._current(con, character_id, target_language)
        assert current is not None
        return current

    def assign_card(
        self,
        character_id: int,
        target_language: str,
        card_id: str,
        by: str,
        locked: bool,
        override: bool = False,
        note: str | None = None,
    ) -> CardAssignment:
        """Make ``card_id`` the character's card for ``target_language``, recorded in history.

        ``by`` is one of auto, assign, revert, edit, derive, restock, or
        ``import:<pack title>``. Other target languages are never touched. Raises KeyError for
        an unknown character. ``auto``, ``restock`` and imports refuse
        (ProtectedAssignmentError) to replace a card that is locked or was chosen by an
        operator action, unless ``override`` (ruling CD2); ``auto`` and ``restock`` (CD9) also
        refuse to replace an imported card, which a later import may replace (CD3).
        """
        if not _CARD_ID.fullmatch(card_id):
            raise ValueError(f"not a card id (16 lowercase hex digits): {card_id!r}")
        if by not in ASSIGNED_BY and not (by.startswith("import:") and len(by) > 7):
            raise ValueError(
                f"unknown assignment by {by!r}; use one of {', '.join(sorted(ASSIGNED_BY))} "
                "or import:<pack title>"
            )
        if not target_language:
            raise ValueError("target language must not be empty")
        with self._tx() as con:
            self._require_character(con, character_id)
            if not override:
                why = replace_refusal(self._current(con, character_id, target_language), by)
                if why is not None:
                    raise ProtectedAssignmentError(
                        f"character {character_id}'s {target_language!r} {why}; {by} does not "
                        "replace it without override"
                    )
            return self._append(con, character_id, target_language, card_id, by, locked, note)

    def revert_card(
        self, character_id: int, target_language: str, to: int | None
    ) -> CardAssignment:
        """Assign again the card of history row ``to``, or by default the most recent earlier
        card that differs from the current one (so two reverts in a row swap back). The new
        row is ``by = "revert"`` and locked: the operator picked it."""
        with self._tx() as con:
            self._require_character(con, character_id)
            rows = [h for _, h in self._history_rows(con, character_id, target_language)]
            current = self._current(con, character_id, target_language)
            if to is not None:
                if not 1 <= to <= len(rows):
                    raise ValueError(
                        f"character {character_id} has no history row {to} for "
                        f"{target_language!r} (rows 1 to {len(rows)})"
                    )
                card = rows[to - 1].card_id
                if card is None:
                    raise ValueError(
                        f"history row {to} has no card (it is a {rows[to - 1].by!r} note); "
                        "pick a row with a card"
                    )
            else:
                if current is None:
                    raise ValueError(
                        f"character {character_id} has no card for {target_language!r} to revert"
                    )
                card = next(
                    (
                        h.card_id
                        for h in reversed(rows[: current.n - 1])
                        if h.card_id is not None and h.card_id != current.card_id
                    ),
                    None,
                )
                if card is None:
                    raise ValueError(
                        f"character {character_id} has no earlier card for {target_language!r}"
                    )
            return self._append(con, character_id, target_language, card, "revert", True)

    def lock_card(self, character_id: int, target_language: str, locked: bool) -> CardAssignment:
        """Lock (or unlock) the current assignment, recorded as a ``lock``/``unlock`` history
        row with the same card; a no-op (no row) when it is already in that state."""
        with self._tx() as con:
            self._require_character(con, character_id)
            current = self._current(con, character_id, target_language)
            if current is None or current.card_id is None:
                raise ValueError(
                    f"character {character_id} has no card for {target_language!r} to "
                    f"{'lock' if locked else 'unlock'}"
                )
            if current.locked == locked:
                return current
            return self._append(
                con,
                character_id,
                target_language,
                current.card_id,
                "lock" if locked else "unlock",
                locked,
            )

    def set_character_external_id(self, character_id: int, external_id: str | None) -> None:
        with self._tx() as con:
            cur = con.execute(
                "UPDATE characters SET external_id = ? WHERE id = ?", (external_id, character_id)
            )
            if cur.rowcount == 0:
                raise KeyError(character_id)

    def _require_character(self, con: sqlite3.Connection, character_id: int) -> None:
        if con.execute("SELECT 1 FROM characters WHERE id = ?", (character_id,)).fetchone() is None:
            raise KeyError(character_id)

    def _delete_character_cards(self, con: sqlite3.Connection, character_id: int) -> None:
        con.execute("DELETE FROM card_assignments WHERE character_id = ?", (character_id,))
        con.execute("DELETE FROM card_history WHERE character_id = ?", (character_id,))

    def _fold_character(
        self, con: sqlite3.Connection, drop: int, keep: int, report: list[str]
    ) -> None:
        """Fold character ``drop`` into ``keep``: its card history (`_fold_cards`) and its
        external id when keep has none. Keep's own external id wins a conflict."""
        ids = dict(
            con.execute("SELECT id, external_id FROM characters WHERE id IN (?, ?)", (drop, keep))
        )
        theirs, mine = ids.get(drop), ids.get(keep)
        if theirs is not None and mine is None:
            con.execute("UPDATE characters SET external_id = ? WHERE id = ?", (theirs, keep))
        elif theirs is not None and theirs != mine:
            report.append(
                f"character {keep} keeps external id {mine}; character {drop}'s {theirs} "
                "was dropped"
            )
        self._fold_cards(con, drop, keep, report)

    def _fold_cards(self, con: sqlite3.Connection, drop: int, keep: int, report: list[str]) -> None:
        """Move character ``drop``'s card history to ``keep``. Per target language, keep's
        current card stays unless it is uncurated and drop's is curated (or keep has none):
        a merge never loses a hand-picked voice for an automatic one. When both are curated,
        keep's wins and ``report`` says so."""
        langs = [
            r[0]
            for r in con.execute(
                "SELECT target_language FROM card_assignments WHERE character_id = ?", (drop,)
            )
        ]
        winners: dict[str, int] = {}
        for lang in langs:
            mine, theirs = self._current(con, keep, lang), self._current(con, drop, lang)
            if mine is not None and _curated_card(mine) and _curated_card(theirs):
                assert theirs is not None
                report.append(
                    f"character {keep} {lang}: kept its card {mine.card_id} ({mine.chosen_by}); "
                    f"character {drop}'s card {theirs.card_id} ({theirs.chosen_by}) stays in "
                    "history"
                )
            if mine is None or (_curated_card(theirs) and not _curated_card(mine)):
                winners[lang] = con.execute(
                    "SELECT history_id FROM card_assignments WHERE character_id = ?"
                    " AND target_language = ?",
                    (drop, lang),
                ).fetchone()[0]
        con.execute("DELETE FROM card_assignments WHERE character_id = ?", (drop,))
        con.execute("UPDATE card_history SET character_id = ? WHERE character_id = ?", (keep, drop))
        for lang, hid in winners.items():
            card = con.execute("SELECT card_id FROM card_history WHERE id = ?", (hid,)).fetchone()
            con.execute(
                "INSERT INTO card_assignments(character_id, target_language, card_id, history_id)"
                " VALUES (?, ?, ?, ?) ON CONFLICT(character_id, target_language) DO UPDATE SET"
                " card_id = excluded.card_id, history_id = excluded.history_id",
                (keep, lang, card[0], hid),
            )

    def rename_character(self, character_id: int, name: str) -> None:
        with self._tx() as con:
            cur = con.execute("UPDATE characters SET name = ? WHERE id = ?", (name, character_id))
            if cur.rowcount == 0:
                raise KeyError(character_id)

    def set_actor(
        self, actor_id: int, name: str | None = None, external_id: str | None = None
    ) -> None:
        with self._tx() as con:
            if con.execute("SELECT 1 FROM actors WHERE id = ?", (actor_id,)).fetchone() is None:
                raise KeyError(actor_id)
            if name is not None:
                con.execute("UPDATE actors SET name = ? WHERE id = ?", (name, actor_id))
            if external_id is not None:
                con.execute(
                    "UPDATE actors SET external_id = ? WHERE id = ?", (external_id, actor_id)
                )

    def lock_actor(self, actor_id: int, locked: bool) -> None:
        with self._tx() as con:
            cur = con.execute("UPDATE actors SET locked = ? WHERE id = ?", (int(locked), actor_id))
            if cur.rowcount == 0:
                raise KeyError(actor_id)

    def merge_actors(self, keep: int, drop: int, report: list[str] | None = None) -> dict[int, int]:
        """Fold actor drop into keep, delete drop, recompute keep.

        drop's exemplars move to keep. v1 has one character per actor, so drop's
        characters are deleted and folded into keep's lowest-id character (card history and
        external id, `_fold_character`); returns {drop character id: keep character id} for
        the caller to remap references. Lines worth telling the operator (a curated card or
        an external id that lost) are appended to ``report``.
        """
        notes: list[str] = report if report is not None else []
        if keep == drop:
            raise ValueError("cannot merge an actor into itself")
        with self._tx() as con:
            for aid in (keep, drop):
                if con.execute("SELECT 1 FROM actors WHERE id = ?", (aid,)).fetchone() is None:
                    raise KeyError(aid)
            dropped: list[int] = [
                r[0]
                for r in con.execute(
                    "SELECT id FROM characters WHERE actor_id = ? ORDER BY id", (drop,)
                )
            ]
            row = con.execute(
                "SELECT MIN(id) FROM characters WHERE actor_id = ?", (keep,)
            ).fetchone()
            target: int | None = row[0]
            if target is None and dropped:  # keep has no character: adopt drop's first
                target = dropped.pop(0)
                con.execute("UPDATE characters SET actor_id = ? WHERE id = ?", (keep, target))
            con.execute("UPDATE exemplars SET actor_id = ? WHERE actor_id = ?", (keep, drop))
            for cid in dropped:
                if target is not None:
                    self._fold_character(con, cid, target, notes)
                else:
                    self._delete_character_cards(con, cid)
            con.execute("DELETE FROM characters WHERE actor_id = ?", (drop,))
            con.execute("DELETE FROM actors WHERE id = ?", (drop,))
            self._recompute(con, keep)
        if target is None:
            return {}
        return dict.fromkeys(dropped, target)

    def delete_episode(self, episode: str) -> list[int]:
        """Remove an episode's exemplars and recompute each affected unlocked actor.

        Returns the ids of the actors left with no exemplars: every exemplar they had came
        from this episode. They keep their last centroid until `prune_actors` deletes them.
        """
        with self._tx() as con:
            affected = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT actor_id FROM exemplars WHERE episode = ? ORDER BY actor_id",
                    (episode,),
                )
            ]
            con.execute("DELETE FROM exemplars WHERE episode = ?", (episode,))
            for aid in affected:
                self._recompute(con, aid)
            return [
                aid
                for aid in affected
                if con.execute("SELECT 1 FROM exemplars WHERE actor_id = ?", (aid,)).fetchone()
                is None
            ]

    def prune_actors(self, actor_ids: list[int]) -> list[int]:
        """Delete those actors (and their characters) that have no exemplars, and return them.

        An actor someone has curated is kept: one that is locked or named, or that has a
        curated character (`curated_characters`). A pruned character's card rows (all ``auto``
        and unlocked, or it would be curated) go with it.
        """
        deleted: list[int] = []
        with self._tx() as con:
            for aid in sorted(set(actor_ids)):
                row = con.execute("SELECT locked, name FROM actors WHERE id = ?", (aid,)).fetchone()
                if row is None or row[0] or row[1] is not None:
                    continue
                if con.execute("SELECT 1 FROM exemplars WHERE actor_id = ?", (aid,)).fetchone():
                    continue
                cids = [
                    r[0]
                    for r in con.execute(
                        "SELECT id FROM characters WHERE actor_id = ?", (aid,)
                    ).fetchall()
                ]
                if any(self._is_curated(con, cid) for cid in cids):
                    continue
                for cid in cids:
                    self._delete_character_cards(con, cid)
                con.execute("DELETE FROM characters WHERE actor_id = ?", (aid,))
                con.execute("DELETE FROM actors WHERE id = ?", (aid,))
                deleted.append(aid)
        return deleted

    def split_cluster(self, actor_id: int, episode: str, cluster: str) -> tuple[int, int]:
        """Move one episode cluster's exemplars to a new actor with a new character.

        Returns (new actor id, new character id). Both actors' centroids are recomputed
        (a locked actor keeps its own).
        """
        with self._tx() as con:
            rows = con.execute(
                "SELECT id, embedding FROM exemplars WHERE actor_id = ? AND episode = ?"
                " AND cluster = ?",
                (actor_id, episode, cluster),
            ).fetchall()
            if not rows:
                raise KeyError((actor_id, episode, cluster))
            mean = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows]).mean(axis=0)
            new_aid, new_cid = self._insert_actor_with_character(con, mean)
            con.execute(
                "UPDATE exemplars SET actor_id = ? WHERE actor_id = ? AND episode = ?"
                " AND cluster = ?",
                (new_aid, actor_id, episode, cluster),
            )
            self._recompute(con, actor_id)
        return new_aid, new_cid
