"""SQLite registry of actors, characters and per-episode voice exemplars.

Actors own the voice embeddings; characters (which get an English voice) belong
to an actor. Embeddings are stored as float32 bytes.
"""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

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
    voice TEXT,
    locked INTEGER NOT NULL DEFAULT 0
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
    voice: dict | None
    locked: bool


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
    cleared: list[int]  # actors with no exemplars whose centroid had another dimension
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


def _normalized(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


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
        with self._tx():
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    self._con.execute(stmt)
            row = self._con.execute("SELECT MAX(version) FROM schema_version").fetchone()
            if row[0] is None:
                self._con.execute(
                    "INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row[0] > SCHEMA_VERSION:
                raise RuntimeError(
                    f"registry {path} has schema version {row[0]}, "
                    f"newer than this fukikae supports ({SCHEMA_VERSION}); upgrade fukikae"
                )

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
            "SELECT id, actor_id, name, voice, locked FROM characters ORDER BY id"
        ).fetchall()
        return [
            Character(r[0], r[1], r[2], None if r[3] is None else json.loads(r[3]), bool(r[4]))
            for r in rows
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

    def protected_actors(self) -> set[int]:
        """Actors someone has curated, which pruning never deletes: locked or named, or with
        a character that has a name or a voice."""
        return {
            r[0]
            for r in self._con.execute(
                "SELECT id FROM actors WHERE locked OR name IS NOT NULL"
                " UNION SELECT actor_id FROM characters"
                " WHERE name IS NOT NULL OR voice IS NOT NULL"
            )
        }

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

    def replace_embeddings(self, embeddings: dict[int, np.ndarray]) -> ReembedResult:
        """Give every exemplar a new embedding (an embedder change), in one transaction.

        `embeddings` must cover every exemplar id, all of one dimension. Then every actor
        with exemplars, locked or not, gets the normalized mean of its new rows as its
        centroid (a centroid in the old space matches nothing); an actor with none keeps its
        centroid only when it already has the new dimension, else it is cleared. Stored
        projections of another input dimension are deleted. Names, locks, voices, external
        ids and the id high-water mark are untouched.
        """
        dims = {len(np.ravel(v)) for v in embeddings.values()}
        if len(dims) > 1:
            raise ValueError(f"new embeddings have several dimensions: {sorted(dims)}")
        with self._tx() as con:
            ids = {r[0] for r in con.execute("SELECT id FROM exemplars")}
            if ids != set(embeddings):
                raise ValueError(
                    "the exemplars changed while they were being re-embedded; run reembed again"
                )
            if not ids:
                return ReembedResult([], [], [])
            (dim,) = dims
            for eid, v in embeddings.items():
                con.execute(
                    "UPDATE exemplars SET embedding = ? WHERE id = ?", (_to_blob(np.ravel(v)), eid)
                )
            centroids, cleared = [], []
            for aid, blob in con.execute("SELECT id, centroid FROM actors ORDER BY id").fetchall():
                rows = [
                    np.frombuffer(r[0], dtype=np.float32)
                    for r in con.execute(
                        "SELECT embedding FROM exemplars WHERE actor_id = ?", (aid,)
                    )
                ]
                if rows:
                    mean = np.stack(rows).mean(axis=0)
                    con.execute(
                        "UPDATE actors SET centroid = ? WHERE id = ?",
                        (_to_blob(_normalized(mean)), aid),
                    )
                    centroids.append(aid)
                elif blob is not None and len(blob) // 4 != dim:
                    con.execute("UPDATE actors SET centroid = NULL WHERE id = ?", (aid,))
                    cleared.append(aid)
            dropped = []
            for p in self.projections():
                if p.matrix.shape[0] != dim:
                    con.execute(
                        "DELETE FROM projections WHERE series = ? AND kind = ?", (p.series, p.kind)
                    )
                    dropped.append((p.series, p.kind, int(p.matrix.shape[0])))
            return ReembedResult(centroids, cleared, dropped)

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

    def merge_actors(self, keep: int, drop: int) -> dict[int, int]:
        """Fold actor drop into keep, delete drop, recompute keep.

        drop's exemplars move to keep. v1 has one character per actor, so drop's
        characters are deleted and folded into keep's lowest-id character; returns
        {drop character id: keep character id} for the caller to remap references.
        """
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
        character with a name or a voice.
        """
        deleted: list[int] = []
        with self._tx() as con:
            for aid in sorted(set(actor_ids)):
                row = con.execute("SELECT locked, name FROM actors WHERE id = ?", (aid,)).fetchone()
                if row is None or row[0] or row[1] is not None:
                    continue
                if con.execute("SELECT 1 FROM exemplars WHERE actor_id = ?", (aid,)).fetchone():
                    continue
                if con.execute(
                    "SELECT 1 FROM characters WHERE actor_id = ?"
                    " AND (name IS NOT NULL OR voice IS NOT NULL)",
                    (aid,),
                ).fetchone():
                    continue
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
