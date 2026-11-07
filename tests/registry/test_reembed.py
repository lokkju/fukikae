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
VOICE = {s: rng.normal(size=OLD) for s in SPEAKERS}  # voices in the old embedder's space
# The new embedder is a synthetic linear image of the old one: new = Q @ old, Q with
# orthonormal columns (so unit rows stay unit). SAME is a 192-d embedder (a rotation).
Q = np.linalg.qr(rng.normal(size=(NEW, OLD)))[0]
SAME = np.linalg.qr(rng.normal(size=(OLD, OLD)))[0]


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


# episode -> cluster label -> (speaker, first span start); a cluster's spans start every 5 s
# from there and are 3 s long. The thousands of a start name its cluster's speaker.
BASES = {
    "S01E01": {"S0": ("ana", 0.0), "S1": ("ben", 1000.0)},
    "S01E02": {"S0": ("ben", 2000.0), "S1": ("ana", 3000.0)},
    "S01E03": {"S0": ("ana", 4000.0)},
}
SPEAKER_OF = {int(b // 1000): spk for cl in BASES.values() for spk, b in cl.values()}
N = {"spans": 100}  # spans per cluster: 2 episodes x 2 clusters x 100 = 400 >= 2 x 192


def starts(eid: str, label: str) -> list[float]:
    return [BASES[eid][label][1] + 5.0 * k for k in range(N["spans"])]


def old_row(start: float) -> np.ndarray:
    voice = VOICE[SPEAKER_OF[int(start // 1000)]]
    noise = np.random.default_rng(round(start * 1000)).normal(0, 0.05, OLD)
    return _unit(voice + noise * np.linalg.norm(voice) / np.sqrt(OLD))


def new_row(start: float, m: np.ndarray = Q) -> np.ndarray:
    return _unit(m @ old_row(start))


class FakeEmbedder:
    map: ClassVar[np.ndarray] = Q
    vram_mb: ClassVar[int] = 1
    calls: ClassVar[list[str]] = []

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.options, self.device = options, device
        self.dim = FakeEmbedder.map.shape[0]

    def load(self) -> None:
        FakeEmbedder.calls.append("load")

    def unload(self) -> None:
        FakeEmbedder.calls.append("unload")

    def embed(self, _audio, spans):
        return np.stack([new_row(s, FakeEmbedder.map) for s, _ in spans])


def _ep(ws: Workspace, eid: str) -> Episode:
    return Episode(source=ws.root / eid, id=eid, work=ws.work_dir(eid), workspace_root=ws.root)


def write_episode(ws: Workspace, eid: str, new: bool = False, shift: float = 0.0) -> None:
    """Diarize outputs (turns, clusters.npz with spans and old or new-embedder rows), the stem
    and a done manifest for the stages before characters."""
    work = ws.work_dir(eid)
    (work / "04-diarize").mkdir(parents=True, exist_ok=True)
    (work / "01-separate").mkdir(parents=True, exist_ok=True)
    (work / "01-separate" / "vocals_16k.wav").write_bytes(b"x")
    turns = [
        Turn(start=st - 0.1, end=st + 3.1 + shift, speaker=label)
        for label in BASES[eid]
        for st in starts(eid, label)
    ]
    write_json(work / "04-diarize" / "turns.json", Diarization(backend="x", turns=turns))
    arrays: dict[str, Any] = {"labels": np.array(list(BASES[eid]), dtype=str)}
    for label in BASES[eid]:
        sts = starts(eid, label)
        rows = [new_row(st, FakeEmbedder.map) if new else old_row(st) for st in sts]
        arrays[f"{label}/embeddings"] = np.stack(rows)
        arrays[f"{label}/spans"] = np.array([(st, st + 3.0) for st in sts], dtype=np.float64)
        arrays[f"{label}/clean_seconds"] = np.array(300.0)
    np.savez(work / "04-diarize" / "clusters.npz", **arrays)
    now = datetime.now(UTC)
    names = ["extract", "separate", "language", "asr", "diarize"]
    save_manifest(
        _ep(ws, eid),
        Manifest(stages={n: StageRecord(key="k", status="done", started=now) for n in names}),
    )


def _resolve(ws: Workspace, episodes: str) -> None:
    r = runner.invoke(
        app, ["registry", "resolve", str(ws.root), "--episodes", episodes, "--threshold", "0.5"]
    )
    assert r.exit_code == 0, r.output


def make_registry(tmp_path, monkeypatch, spans=100, embedder=Q) -> tuple[Workspace, int]:
    """Two episodes resolved with the old 192-d embedder; ana's actor named, voiced, and
    locked after E01 (so E02's exemplars arrived after the lock). `embedder` is the
    configured embedder's map from old rows."""
    N["spans"] = spans
    FakeEmbedder.map = embedder
    monkeypatch.setattr(rr, "create", lambda _k, bc, dev: FakeEmbedder(bc.options, dev))
    monkeypatch.setattr(rr, "device_string", lambda _cfg: "cpu")
    root = tmp_path / "ws"
    root.mkdir()
    (root / "fukikae.toml").write_text("")  # default profile
    ws = Workspace(root)
    for eid in ("S01E01", "S01E02"):
        write_episode(ws, eid)
    _resolve(ws, "S01E01")
    with Registry(ws.registry_path) as reg:
        ana = reg.cluster_actors("S01E01", "S0")[0]
        reg.set_actor(ana, name="Ana Actor", external_id="tmdb:1")
        reg.lock_actor(ana, True)
        cid = next(c.id for c in reg.characters() if c.actor_id == ana)
        reg.rename_character(cid, "ANA")
        reg.save_projection("default", "lda", np.ones((OLD, 1)), np.zeros(OLD))
    _resolve(ws, "S01E01,S01E02")
    with Registry(ws.registry_path) as reg:
        assert set(reg.episode_counts(ana)) == {"S01E01", "S01E02"}  # added after the lock
    con = sqlite3.connect(ws.registry_path)
    con.execute("UPDATE characters SET voice = ? WHERE id = ?", ('{"voice": "v1"}', cid))
    con.commit()
    con.close()
    FakeEmbedder.calls.clear()
    return ws, ana


def centroid(ws: Workspace, aid: int) -> np.ndarray:
    with Registry(ws.registry_path) as reg:
        c = next(a for a in reg.actors() if a.id == aid).centroid
    assert c is not None
    return c


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


def assert_new_rows_and_unlocked_means(ws: Workspace, ana: int) -> None:
    """Every exemplar is the new embedder's row of its span; every unlocked actor's centroid
    is the normalized mean of its new rows."""
    with Registry(ws.registry_path) as reg:
        exemplars = reg.exemplars()
        assert exemplars and {len(e.embedding) for e in exemplars} == {FakeEmbedder.map.shape[0]}
        for e in exemplars:
            sts = starts(e.episode, str(e.cluster))
            assert any(np.allclose(e.embedding, new_row(st, FakeEmbedder.map)) for st in sts)
        for a in reg.actors():
            if a.id == ana:
                continue
            want = np.stack([e.embedding for e in exemplars if e.actor_id == a.id]).mean(axis=0)
            assert a.centroid is not None
            assert np.allclose(a.centroid, want / np.linalg.norm(want), atol=1e-6)


def test_reembed_moves_every_exemplar_and_maps_the_locked_centroid(tmp_path, monkeypatch):
    """192 -> 256 with new = Q @ old: the locked centroid (fixed at lock time, from E01 only)
    is carried by the fitted map, close to its true image, not recomputed from exemplars."""
    ws, ana = make_registry(tmp_path, monkeypatch)
    before = curation(ws)
    locked_old = centroid(ws, ana)
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert f"actor {ana} (locked, named Ana Actor): centroid mapped 192-d -> 256-d" in r.output
    assert FakeEmbedder.calls == ["load", "unload"]
    assert_new_rows_and_unlocked_means(ws, ana)
    got, truth = centroid(ws, ana), _unit(Q @ locked_old)
    assert len(got) == NEW and float(np.dot(got, truth)) > 0.99
    with Registry(ws.registry_path) as reg:
        assert reg.projections() == []  # the 192-d lda projection is dropped
        mean_all = np.stack([e.embedding for e in reg.exemplars() if e.actor_id == ana]).mean(0)
    # not the mean of its exemplars (E02's arrived after the lock)
    assert not np.allclose(got, mean_all / np.linalg.norm(mean_all), atol=1e-4)
    assert curation(ws) == before


def test_same_dimension_reembed_keeps_a_locked_centroid_exactly(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch, embedder=SAME)
    locked = centroid(ws, ana).copy()
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert f"actor {ana} (locked, named Ana Actor): centroid kept as stored (192-d)" in r.output
    assert np.array_equal(centroid(ws, ana), locked)
    assert_new_rows_and_unlocked_means(ws, ana)


def test_same_dimension_reembed_keeps_an_exemplarless_named_actor(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch, embedder=SAME)
    with Registry(ws.registry_path) as reg:
        ghost, _ = reg.create_actor_with_character(_unit(rng.normal(size=OLD)))
        reg.set_actor(ghost, name="Ghost")
    stored = centroid(ws, ghost).copy()
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert f"actor {ghost} (no exemplars, named Ghost): centroid kept" in r.output
    assert np.array_equal(centroid(ws, ghost), stored)


def test_too_few_pairs_to_map_refuses_with_no_changes(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch, spans=3)  # 12 pairs < 2 x 192
    before = snapshot(ws)
    for extra in ([], ["--dry-run"]):
        r = runner.invoke(app, ["registry", "reembed", str(ws.root), *extra])
        assert r.exit_code == 1
        assert f"actor {ana} (locked, named Ana Actor, 192-d: 12 pairs, need 384)" in r.output
        assert "--unlock" in r.output
    assert FakeEmbedder.calls == []
    assert snapshot(ws) == before


def test_dry_run_reports_counts_per_dimension_and_changes_nothing(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch)
    before = snapshot(ws)
    r = runner.invoke(app, ["registry", "reembed", str(ws.root), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "exemplars: 400 x 192-d" in r.output and "(256-d)" in r.output
    assert "400 spans to embed from 2 episodes" in r.output
    assert f"actor {ana} (locked, named Ana Actor): centroid mapped" in r.output
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
    write_episode(ws, "S01E01", new=True, shift=0.5)
    before = snapshot(ws)
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 1 and "S01E01" in r.output and "diarize re-ran" in r.output
    assert snapshot(ws) == before


def test_a_rediarization_with_the_same_turns_is_accepted(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch)
    write_episode(ws, "S01E01", new=True)  # same turns, rows already from the new embedder
    r = runner.invoke(app, ["registry", "reembed", str(ws.root)])
    assert r.exit_code == 0, r.output
    assert_new_rows_and_unlocked_means(ws, ana)
    assert float(np.dot(centroid(ws, ana), _unit(Q @ old_row(0.0)))) > 0.95  # ana's voice


def test_rematch_and_resolve_work_after_reembed(tmp_path, monkeypatch):
    ws, ana = make_registry(tmp_path, monkeypatch)
    # Before: a new episode diarized with the new embedder cannot be matched.
    write_episode(ws, "S01E03", new=True)
    r = runner.invoke(app, ["registry", "resolve", str(ws.root), "--episodes", "S01E03"])
    assert r.exit_code == 1 and "registry reembed" in r.output
    assert runner.invoke(app, ["registry", "reembed", str(ws.root)]).exit_code == 0
    # resolve anchors the locked actor by its mapped centroid
    _resolve(ws, "S01E03")
    from tests.registry.test_resolve import by_cluster

    e3, _ = by_cluster(ws, "S01E03")
    assert e3["S0"].status == "matched" and e3["S0"].actor_id == ana
    for eid in ("S01E01", "S01E02"):  # the pipeline re-runs diarize with the new embedder
        write_episode(ws, eid, new=True)
    CharactersStage(rematch=True).run(_ep(ws, "S01E03"), ws.config)
    e3, _ = by_cluster(ws, "S01E03")
    assert e3["S0"].status == "matched" and e3["S0"].actor_id == ana
    _resolve(ws, "S01E01,S01E02,S01E03")
    for eid, label in (("S01E01", "S0"), ("S01E02", "S1"), ("S01E03", "S0")):
        got, _ = by_cluster(ws, eid)
        assert got[label].actor_id == ana, (eid, got)
