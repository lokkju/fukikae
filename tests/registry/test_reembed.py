"""`fukikae registry reembed` (Ruling R19): re-embed every exemplar with the configured embedder
after an embedder change, from the episodes' existing diarize spans, in one transaction."""

import sqlite3
from datetime import UTC, datetime
from typing import Any, ClassVar

import numpy as np
from typer.testing import CliRunner

from fukikae.artifacts import write_json
from fukikae.cli import app
from fukikae.job import Manifest, StageRecord, save_manifest
from fukikae.models import Diarization, Turn
from fukikae.registry import reembed as rr
from fukikae.registry.store import Registry
from fukikae.stages.characters import CharactersStage
from fukikae.workspace import Episode, Workspace

runner = CliRunner()
OLD, NEW = 192, 256
SPEAKERS = ("ana", "ben")
rng = np.random.default_rng(7)
OLD_VOICE = {s: rng.normal(size=OLD) for s in SPEAKERS}
NEW_VOICE = {s: rng.normal(size=NEW) for s in SPEAKERS}


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _row(voice, start):
    noise = np.random.default_rng(round(start * 1000)).normal(0, 0.05, len(voice))
    return _unit(voice + noise * np.linalg.norm(voice) / np.sqrt(len(voice)))


# episode -> cluster label -> (speaker, span starts); every span is 3 s long
EPISODES = {
    "S01E01": {"S0": ("ana", [0.0, 10.0, 20.0]), "S1": ("ben", [30.0, 40.0, 50.0])},
    "S01E02": {"S0": ("ben", [100.0, 110.0, 120.0]), "S1": ("ana", [130.0, 140.0, 150.0])},
    "S01E03": {"S0": ("ana", [200.0, 210.0, 220.0])},
}
SPEAKER_AT = {
    st: spk for clusters in EPISODES.values() for spk, starts in clusters.values() for st in starts
}


class FakeEmbedder:
    dim: ClassVar[int] = NEW
    vram_mb: ClassVar[int] = 1
    calls: ClassVar[list[str]] = []

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.options, self.device = options, device

    def load(self) -> None:
        FakeEmbedder.calls.append("load")

    def unload(self) -> None:
        FakeEmbedder.calls.append("unload")

    def embed(self, _audio, spans):
        return np.stack([_row(NEW_VOICE[SPEAKER_AT[s]], s) for s, _ in spans])


def _ep(ws: Workspace, eid: str) -> Episode:
    return Episode(source=ws.root / eid, id=eid, work=ws.work_dir(eid), workspace_root=ws.root)


def write_episode(ws: Workspace, eid: str, voices: dict, dim_shift: float = 0.0) -> None:
    """Diarize outputs (turns, clusters.npz with spans and rows from `voices`), the stem and a
    done manifest for the stages before characters."""
    work = ws.work_dir(eid)
    (work / "04-diarize").mkdir(parents=True, exist_ok=True)
    (work / "01-separate").mkdir(parents=True, exist_ok=True)
    (work / "01-separate" / "vocals_16k.wav").write_bytes(b"x")
    turns = [
        Turn(start=st - 0.1, end=st + 3.1 + dim_shift, speaker=label)
        for label, (_, starts) in EPISODES[eid].items()
        for st in starts
    ]
    write_json(work / "04-diarize" / "turns.json", Diarization(backend="x", turns=turns))
    arrays: dict[str, Any] = {"labels": np.array(list(EPISODES[eid]), dtype=str)}
    for label, (spk, starts) in EPISODES[eid].items():
        arrays[f"{label}/embeddings"] = np.stack([_row(voices[spk], st) for st in starts])
        arrays[f"{label}/spans"] = np.array([(st, st + 3.0) for st in starts], dtype=np.float64)
        arrays[f"{label}/clean_seconds"] = np.array(30.0)
    np.savez(work / "04-diarize" / "clusters.npz", **arrays)
    now = datetime.now(UTC)
    names = ["extract", "separate", "language", "asr", "diarize"]
    save_manifest(
        _ep(ws, eid),
        Manifest(stages={n: StageRecord(key="k", status="done", started=now) for n in names}),
    )


def make_registry(tmp_path, monkeypatch) -> tuple[Workspace, int]:
    """Two episodes resolved with a 192-d embedder; ana's actor named, locked, voiced."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "fukikae.toml").write_text("")  # default profile: WeSpeaker
    ws = Workspace(root)
    for eid in ("S01E01", "S01E02"):
        write_episode(ws, eid, OLD_VOICE)
    r = runner.invoke(
        app, ["registry", "resolve", str(root), "--episodes", "S01E01,S01E02", "--threshold", "0.5"]
    )
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        ana = reg.cluster_actors("S01E01", "S0")[0]
        reg.set_actor(ana, name="Ana Actor", external_id="tmdb:1")
        reg.lock_actor(ana, True)
        cid = next(c.id for c in reg.characters() if c.actor_id == ana)
        reg.rename_character(cid, "ANA")
        reg.save_projection("default", "lda", np.ones((OLD, 1)), np.zeros(OLD))
    con = sqlite3.connect(ws.registry_path)
    con.execute("UPDATE characters SET voice = ? WHERE id = ?", ('{"voice": "v1"}', cid))
    con.commit()
    con.close()
    FakeEmbedder.calls.clear()
    monkeypatch.setattr(rr, "create", lambda _k, bc, dev: FakeEmbedder(bc.options, dev))
    monkeypatch.setattr(rr, "device_string", lambda _cfg: "cpu")
    return ws, ana


def curation(ws: Workspace):
    with Registry(ws.registry_path) as reg:
        actors = [(a.id, a.name, a.external_id, a.locked) for a in reg.actors()]
        chars = reg.characters()
    con = sqlite3.connect(ws.registry_path)
    hw = con.execute("SELECT tbl, last FROM id_high_water ORDER BY tbl").fetchall()
    con.close()
    return actors, chars, hw


def snapshot(ws: Workspace):
    with Registry(ws.registry_path) as reg:
        ex = [
            (e.id, e.actor_id, e.episode, e.cluster, e.embedding.tolist()) for e in reg.exemplars()
        ]
        cents = [np.asarray(a.centroid).tolist() for a in reg.actors()]
        proj = [(p.series, p.kind, p.matrix.shape) for p in reg.projections()]
    return curation(ws), ex, cents, proj


def test_reembed_moves_every_exemplar_to_the_new_embedder_keeping_curation(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch)
    before = curation(ws)
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert FakeEmbedder.calls == ["load", "unload"]
    with Registry(ws.registry_path) as reg:
        exemplars = reg.exemplars()
        assert exemplars and {len(e.embedding) for e in exemplars} == {NEW}
        for e in exemplars:
            spk, starts = EPISODES[e.episode][str(e.cluster)]
            assert any(np.allclose(e.embedding, _row(NEW_VOICE[spk], st)) for st in starts)
        for a in reg.actors():
            mine = np.stack([e.embedding for e in exemplars if e.actor_id == a.id])
            want = mine.mean(axis=0)
            assert a.centroid is not None and len(a.centroid) == NEW
            assert np.allclose(a.centroid, want / np.linalg.norm(want), atol=1e-6)
        assert reg.projections() == []  # the 192-d lda projection is dropped
        assert next(a for a in reg.actors() if a.id == ana).locked
    assert curation(ws) == before


def test_dry_run_reports_counts_per_dimension_and_changes_nothing(tmp_path, monkeypatch):
    ws, _ = make_registry(tmp_path, monkeypatch)
    before = snapshot(ws)
    r = runner.invoke(app, ["registry", "reembed", str(ws.root), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "exemplars: 12 x 192-d" in r.output and "(256-d)" in r.output
    assert "12 spans to embed from 2 episodes" in r.output
    assert FakeEmbedder.calls == []
    assert snapshot(ws) == before


def test_missing_diarize_output_refuses_with_no_changes(tmp_path, monkeypatch):
    ws, _ = make_registry(tmp_path, monkeypatch)
    (ws.work_dir("S01E02") / "04-diarize" / "clusters.npz").unlink()
    before = snapshot(ws)
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 1
    assert "S01E02" in r.output and "clusters.npz" in r.output and "S01E01" not in r.output
    assert FakeEmbedder.calls == []
    assert snapshot(ws) == before


def test_a_changed_diarization_refuses(tmp_path, monkeypatch):
    """Diarize re-ran with the new embedder and different turns: the labels may name other
    speakers now, so the episode is refused, never guessed."""
    ws, _ = make_registry(tmp_path, monkeypatch)
    write_episode(ws, "S01E01", NEW_VOICE, dim_shift=0.5)
    before = snapshot(ws)
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 1 and "S01E01" in r.output and "diarize re-ran" in r.output
    assert snapshot(ws) == before


def test_a_rediarization_with_the_same_turns_is_accepted(tmp_path, monkeypatch):
    ws, _ = make_registry(tmp_path, monkeypatch)
    write_episode(ws, "S01E01", NEW_VOICE)  # same turns, rows already from the new embedder
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 0, r.output


def test_rematch_and_resolve_work_after_reembed(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch)
    # Before: a new episode diarized with the new embedder cannot be matched.
    write_episode(ws, "S01E03", NEW_VOICE)
    r = runner.invoke(app, ["registry", "resolve", str(ws.root), "--episodes", "S01E03"])
    assert r.exit_code == 1 and "registry reembed" in r.output
    assert runner.invoke(app, ["registry", "reembed", str(ws.root)]).exit_code == 0
    for eid in ("S01E01", "S01E02"):  # the pipeline re-runs diarize with the new embedder
        write_episode(ws, eid, NEW_VOICE)
    CharactersStage(rematch=True).run(_ep(ws, "S01E03"), ws.config)
    from tests.registry.test_resolve import by_cluster

    e3, _ = by_cluster(ws, "S01E03")
    assert e3["S0"].status == "matched" and e3["S0"].actor_id == ana
    r = runner.invoke(
        app, ["registry", "resolve", str(ws.root), "--episodes", "S01E01,S01E02,S01E03"]
    )
    assert r.exit_code == 0, r.output
    for eid, label in (("S01E01", "S0"), ("S01E02", "S1"), ("S01E03", "S0")):
        got, _ = by_cluster(ws, eid)
        assert got[label].actor_id == ana, (eid, got)
