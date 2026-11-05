"""SQLite registry of actors, characters and per-episode voice exemplars.

Actors own the voice embeddings; characters (which get an English voice) belong
to an actor. Embeddings are stored as float32 bytes.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
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
                    f"newer than this doppio supports ({SCHEMA_VERSION}); upgrade doppio"
                )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._con.execute("BEGIN IMMEDIATE")
        try:
            yield self._con
        except BaseException:
            self._con.execute("ROLLBACK")
            raise
        else:
            self._con.execute("COMMIT")

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

    # writes

    def create_actor_with_character(self, centroid: np.ndarray) -> tuple[int, int]:
        with self._tx() as con:
            aid = con.execute(
                "INSERT INTO actors(centroid) VALUES (?)", (_to_blob(_normalized(centroid)),)
            ).lastrowid
            cid = con.execute("INSERT INTO characters(actor_id) VALUES (?)", (aid,)).lastrowid
        assert aid is not None and cid is not None
        return aid, cid

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

    def merge_actors(self, keep: int, drop: int) -> None:
        """Move drop's exemplars and characters to keep, delete drop, recompute keep."""
        if keep == drop:
            raise ValueError("cannot merge an actor into itself")
        with self._tx() as con:
            for aid in (keep, drop):
                if con.execute("SELECT 1 FROM actors WHERE id = ?", (aid,)).fetchone() is None:
                    raise KeyError(aid)
            con.execute("UPDATE exemplars SET actor_id = ? WHERE actor_id = ?", (keep, drop))
            con.execute("UPDATE characters SET actor_id = ? WHERE actor_id = ?", (keep, drop))
            con.execute("DELETE FROM actors WHERE id = ?", (drop,))
            self._recompute(con, keep)

    def delete_exemplars(self, episode: str, cluster: str) -> None:
        """Remove one cluster's exemplars in an episode and recompute the affected actors.

        Like delete_episode, an actor left with none keeps its last centroid.
        """
        with self._tx() as con:
            affected = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT actor_id FROM exemplars WHERE episode = ? AND cluster = ?",
                    (episode, cluster),
                )
            ]
            con.execute(
                "DELETE FROM exemplars WHERE episode = ? AND cluster = ?", (episode, cluster)
            )
            for aid in affected:
                self._recompute(con, aid)

    def relabel_exemplars(self, episode: str, mapping: dict[str, str]) -> None:
        """Rename clusters of an episode's exemplars at once (old -> new; swaps are fine)."""
        moves = {o: n for o, n in mapping.items() if o != n}
        if not moves:
            return
        tmp = "\x00"
        with self._tx() as con:
            for old, new in moves.items():
                con.execute(
                    "UPDATE exemplars SET cluster = ? WHERE episode = ? AND cluster = ?",
                    (tmp + new, episode, old),
                )
            for new in moves.values():
                con.execute(
                    "UPDATE exemplars SET cluster = ? WHERE episode = ? AND cluster = ?",
                    (new, episode, tmp + new),
                )

    def delete_episode(self, episode: str) -> None:
        """Remove an episode's exemplars and recompute each affected unlocked actor.

        An actor left with no exemplars keeps its last centroid (it is not nulled),
        because a later episode may still match it.
        """
        with self._tx() as con:
            affected = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT actor_id FROM exemplars WHERE episode = ?", (episode,)
                )
            ]
            con.execute("DELETE FROM exemplars WHERE episode = ?", (episode,))
            for aid in affected:
                self._recompute(con, aid)

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
            new_aid = con.execute(
                "INSERT INTO actors(centroid) VALUES (?)", (_to_blob(_normalized(mean)),)
            ).lastrowid
            new_cid = con.execute(
                "INSERT INTO characters(actor_id) VALUES (?)", (new_aid,)
            ).lastrowid
            assert new_aid is not None and new_cid is not None
            con.execute(
                "UPDATE exemplars SET actor_id = ? WHERE actor_id = ? AND episode = ?"
                " AND cluster = ?",
                (new_aid, actor_id, episode, cluster),
            )
            self._recompute(con, actor_id)
        return new_aid, new_cid
