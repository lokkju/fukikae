"""A names-less characters scenario whose assignments and registry the pre-names code
(debd501) wrote to `characters_golden.json` (regenerate only with that commit's code): the
stage must reproduce them byte for byte without names (Review Focus 4). Each step records the
assignments (``<step>``) and a dump of the registry after it (``<step>/registry``); the last
steps run with a locked actor and an operator-named character. Uses only API (and registry
columns) both versions share."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from fukikae.artifacts import write_json
from fukikae.config import Config, RegistrySettings
from fukikae.models import Diarization, Turn
from fukikae.registry.store import Registry
from fukikae.stages.characters import CharactersStage
from fukikae.workspace import Episode, Workspace

DIM = 4
VOICES = np.eye(DIM, dtype=np.float32)


def _rows(rng: np.random.Generator, voice: int, n: int, spread: float) -> np.ndarray:
    r = VOICES[voice] + spread * rng.standard_normal((n, DIM)).astype(np.float32)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def _episode(root: Path, eid: str, rng: np.random.Generator, spec: list[tuple]) -> Episode:
    """spec: [(label, voice, rows, clean_seconds, [(start, end)])]."""
    ep = Episode(source=root / f"{eid}.mkv", id=eid, work=root / "eps" / eid, workspace_root=root)
    d = ep.work / "04-diarize"
    d.mkdir(parents=True, exist_ok=True)
    turns = sorted(
        (Turn(start=s, end=e, speaker=lb) for lb, _v, _n, _c, spans in spec for s, e in spans),
        key=lambda t: (t.start, t.speaker),
    )
    write_json(d / "turns.json", Diarization(backend="x", turns=turns))
    arrays: dict[str, Any] = {"labels": np.array([s[0] for s in spec], dtype=str)}
    for lb, voice, n, clean, _spans in spec:
        r = _rows(rng, voice, n, 0.15) if n else np.zeros((0, DIM), dtype=np.float32)
        arrays[f"{lb}/embeddings"] = r
        arrays[f"{lb}/spans"] = np.zeros((len(r), 2))
        arrays[f"{lb}/clean_seconds"] = np.array(clean, dtype=np.float64)
    np.savez(d / "clusters.npz", **arrays)
    return ep


def registry_dump(root: Path) -> str:
    """The registry's actors, characters, aliases and exemplars as JSON text (centroids and
    rows rounded to 6 places), read with sqlite so either version's store reads the same."""
    con = sqlite3.connect(Workspace(root).registry_path)
    try:

        def vec(blob: bytes | None) -> list[float] | None:
            if blob is None:
                return None
            return [round(float(x), 6) for x in np.frombuffer(blob, dtype=np.float32)]

        dump = {
            "actors": [
                [r[0], r[1], r[2], vec(r[3]), r[4]]
                for r in con.execute(
                    "SELECT id, name, external_id, centroid, locked FROM actors ORDER BY id"
                )
            ],
            "characters": [
                list(r)
                for r in con.execute(
                    "SELECT id, actor_id, name, external_id, locked, name_by FROM characters"
                    " ORDER BY id"
                )
            ],
            "aliases": [
                list(r)
                for r in con.execute(
                    "SELECT alias, character_id, by, language FROM character_aliases ORDER BY alias"
                )
            ],
            "exemplars": [
                list(r)
                for r in con.execute(
                    "SELECT actor_id, episode, cluster, COUNT(*) FROM exemplars"
                    " GROUP BY actor_id, episode, cluster ORDER BY actor_id, episode, cluster"
                )
            ],
        }
    finally:
        con.close()
    return json.dumps(dump, indent=1)


def scenario(root: Path) -> dict[str, str]:
    """Run the stage through first runs, pinned reruns, a rematch and the lda matcher; return
    each step's assignments.json text."""
    rng = np.random.default_rng(7)
    cfg = Config(registry=RegistrySettings(min_new_seconds=20.0, match_threshold=0.8))
    lda = Config(registry=cfg.registry, raw={"characters": {"matcher": "lda"}})
    e1 = _episode(
        root,
        "S01E01",
        rng,
        [
            ("SPEAKER_00", 0, 6, 60.0, [(0, 30), (60, 90)]),
            ("SPEAKER_01", 1, 4, 35.0, [(30, 60)]),
            ("SPEAKER_02", 2, 1, 5.0, [(90, 95)]),
            ("SPEAKER_03", 3, 0, 0.0, [(95, 96)]),
        ],
    )
    e2 = _episode(
        root,
        "S01E02",
        rng,
        [
            ("SPEAKER_00", 1, 5, 40.0, [(0, 40)]),
            ("SPEAKER_01", 0, 5, 50.0, [(40, 90)]),
            ("SPEAKER_02", 0, 2, 12.0, [(90, 102)]),
            ("SPEAKER_03", 3, 4, 30.0, [(102, 132)]),
        ],
    )
    out: dict[str, str] = {}

    def step(name: str, ep: Episode, c: Config, rematch: bool = False) -> None:
        CharactersStage(rematch=rematch).run(ep, c)
        out[name] = (ep.work / "05-characters" / "assignments.json").read_text()
        out[f"{name}/registry"] = registry_dump(root)

    step("e1-first", e1, cfg)
    step("e2-first", e2, cfg)
    step("e1-rerun", e1, cfg)
    step("e2-rematch", e2, cfg, rematch=True)
    with Registry(Workspace(root).registry_path) as reg:
        reg.save_projection("default", "lda", np.eye(DIM)[:, :2], np.zeros(DIM))
    step("e2-lda-rematch", e2, lda, rematch=True)
    step("e2-lda-rerun", e2, lda)
    # operator state without names: a locked actor and an operator-named character
    with Registry(Workspace(root).registry_path) as reg:
        first = {a.cluster: a for a in _assignments(e1).clusters}
        locked, named = first["SPEAKER_00"].actor_id, first["SPEAKER_01"].character_id
        assert locked is not None and named is not None
        reg.lock_actor(locked, True)
        reg.rename_character(named, "Natsu")
    step("e1-locked-rematch", e1, cfg, rematch=True)
    step("e2-locked-rematch", e2, cfg, rematch=True)
    step("e2-locked-rerun", e2, cfg)
    step("e2-locked-lda-rematch", e2, lda, rematch=True)
    return out


def _assignments(ep: Episode) -> Any:
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments

    return read_json(ep.work / "05-characters" / "assignments.json", Assignments)
