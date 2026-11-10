"""A names-less characters scenario whose assignments the pre-names code (debd501) wrote to
`characters_golden.json` (regenerate only with that commit's code): the stage must reproduce
them byte for byte without names (Review Focus 4). Uses only API both versions share."""

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

    step("e1-first", e1, cfg)
    step("e2-first", e2, cfg)
    step("e1-rerun", e1, cfg)
    step("e2-rematch", e2, cfg, rematch=True)
    with Registry(Workspace(root).registry_path) as reg:
        reg.save_projection("default", "lda", np.eye(DIM)[:, :2], np.zeros(DIM))
    step("e2-lda-rematch", e2, lda, rematch=True)
    step("e2-lda-rerun", e2, lda)
    return out
