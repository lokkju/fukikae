from datetime import UTC, datetime

import numpy as np
from typer.testing import CliRunner

from doppio.artifacts import read_json
from doppio.cli import app
from doppio.job import Manifest, StageRecord, load_manifest, save_manifest
from doppio.models import Assignments
from doppio.registry.cluster import seed_clusters
from doppio.registry.matching import ClusterEvidence, mean_embedding
from doppio.registry.store import Registry
from doppio.workspace import Workspace
from tests.stages.test_characters import unit, write_diarize

runner = CliRunner()
CENTERS = [unit([1, 0, 0]), unit([0, 1, 0])]
OUTLIER = unit([0, 0, 1])


def noisy(rng, center, n=6):
    return [unit(center + rng.normal(0, 0.05, 3)) for _ in range(n)]


def synthetic():
    rng = np.random.default_rng(7)
    data = {}
    for i, ep in enumerate(["S01E01", "S01E02", "S01E03"]):
        clusters = {
            "S0": (noisy(rng, CENTERS[0]), 40.0),
            "S1": (noisy(rng, CENTERS[1]), 35.0),
        }
        if i == 0:
            clusters["S2"] = ([OUTLIER], 5.0)
        data[ep] = clusters
    return data


def evidence(data):
    return {
        ep: [ClusterEvidence(lb, mean_embedding(np.array(r)), s) for lb, (r, s) in cl.items()]
        for ep, cl in data.items()
    }


def test_seed_two_groups_outlier_dropped():
    groups = seed_clusters(evidence(synthetic()), distance_threshold=0.3, min_new_seconds=20.0)
    assert sorted(groups) == [
        [("S01E01", "S0"), ("S01E02", "S0"), ("S01E03", "S0")],
        [("S01E01", "S1"), ("S01E02", "S1"), ("S01E03", "S1")],
    ]


def test_seed_empty_and_no_embedding():
    assert seed_clusters({}, 0.3, 20.0) == []
    ev = {"S01E01": [ClusterEvidence("S0", None, 99.0)]}
    assert seed_clusters(ev, 0.3, 20.0) == []


def make_ws(tmp_path, done=True):
    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace(root)
    now = datetime.now(UTC)
    for ep_id, clusters in synthetic().items():
        src = tmp_path / f"{ep_id}.mkv"
        src.write_bytes(b"x")
        ep = ws.find_episode(str(src))
        write_diarize(ep, clusters, [(0, 20, "S0"), (20, 40, "S1")])
        if done:
            names = ["extract", "separate", "language", "asr", "diarize"]
            save_manifest(
                ep,
                Manifest(
                    stages={n: StageRecord(key="k", status="done", started=now) for n in names}
                    | {"characters": StageRecord(key="k", status="done", started=now)}
                ),
            )
    return ws


def test_cli_cluster_seeds_registry(tmp_path):
    ws = make_ws(tmp_path)
    r = runner.invoke(
        app, ["registry", "cluster", str(ws.root), "--episodes", "S01E01,S01E02,S01E03"]
    )
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        assert len(reg.actors()) == 2 and len(reg.characters()) == 2
        assert reg.exemplar_count(1) == 18
    ep = ws.work_dir("S01E01")
    a = read_json(ep / "05-characters" / "assignments.json", Assignments)
    by = {c.cluster: c for c in a.clusters}
    assert by["S0"].status == "matched" and by["S1"].status == "matched"
    assert by["S2"].status == "extra" and by["S2"].actor_id is None
    assert by["S0"].character_id != by["S1"].character_id
    assert a.turns is not None and len(a.turns) == 2


def test_cli_cluster_refuses_nonempty_registry(tmp_path):
    ws = make_ws(tmp_path)
    with Registry(ws.registry_path) as reg:
        reg.create_actor_with_character(unit([1, 0, 0]))
    r = runner.invoke(
        app, ["registry", "cluster", str(ws.root), "--episodes", "S01E01,S01E02,S01E03"]
    )
    assert r.exit_code == 1
    assert "per-episode" in r.output


def test_cli_cluster_needs_diarize(tmp_path):
    ws = make_ws(tmp_path, done=False)
    r = runner.invoke(app, ["registry", "cluster", str(ws.root), "--episodes", "S01E01"])
    assert r.exit_code == 1
    assert "diarize" in r.output


def seeded(tmp_path):
    ws = make_ws(tmp_path)
    r = runner.invoke(
        app, ["registry", "cluster", str(ws.root), "--episodes", "S01E01,S01E02,S01E03"]
    )
    assert r.exit_code == 0, r.output
    return ws


def test_cli_ops(tmp_path):
    ws = seeded(tmp_path)
    w = str(ws.root)
    r = runner.invoke(app, ["registry", "list", w])
    assert r.exit_code == 0 and "18" in r.output
    r = runner.invoke(app, ["registry", "show", w, "1"])
    assert r.exit_code == 0 and "S01E02" in r.output
    r = runner.invoke(app, ["registry", "actor", w, "1", "--name", "Ann", "--external-id", "x1"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["registry", "lock", w, "1"])
    assert r.exit_code == 0
    r = runner.invoke(app, ["registry", "lock", w, "1", "--unlock"])
    assert r.exit_code == 0
    r = runner.invoke(app, ["registry", "rename", w, "1", "Hero"])
    assert r.exit_code == 0 and "S01E01" in r.output
    # rename marks the characters stage stale
    assert "characters" not in load_manifest(_ep(ws, "S01E02")).stages
    with Registry(ws.registry_path) as reg:
        assert [c.name for c in reg.characters()][0] == "Hero"
        assert reg.actors()[0].name == "Ann" and not reg.actors()[0].locked


def _ep(ws, eid):
    from doppio.workspace import Episode

    return Episode(source=ws.root / "x", id=eid, work=ws.work_dir(eid), workspace_root=ws.root)


def test_cli_merge_and_split(tmp_path):
    ws = seeded(tmp_path)
    w = str(ws.root)
    r = runner.invoke(app, ["registry", "split", w, "--episode", "S01E02", "--cluster", "S0"])
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        assert len(reg.actors()) == 3
        assert reg.exemplar_count(1) == 12 and reg.exemplar_count(3) == 6
    r = runner.invoke(app, ["registry", "merge", w, "1", "3"])
    assert r.exit_code == 0 and "S01E02" in r.output
    with Registry(ws.registry_path) as reg:
        assert len(reg.actors()) == 2 and reg.exemplar_count(1) == 18


def test_split_cluster_store(tmp_path):
    with Registry(tmp_path / "r.sqlite") as reg:
        aid, _ = reg.create_actor_with_character(unit([1, 0, 0]))
        reg.add_exemplars(aid, "E1", "S0", np.array([unit([1, 0, 0])] * 2), 10.0)
        reg.add_exemplars(aid, "E1", "S1", np.array([unit([0, 1, 0])] * 2), 10.0)
        new_aid, new_cid = reg.split_cluster(aid, "E1", "S1")
        assert new_aid != aid and reg.exemplar_count(aid) == 2 and reg.exemplar_count(new_aid) == 2
        assert [c.actor_id for c in reg.characters() if c.id == new_cid] == [new_aid]
        old = next(a for a in reg.actors() if a.id == aid)
        assert old.centroid is not None and abs(float(old.centroid[0]) - 1.0) < 1e-6


def test_cli_cluster_two_clusters_of_one_episode_then_run(tmp_path):
    """A seeded group holding two clusters of one episode: the one with more clean speech
    is matched, the other goes to review; the next characters run pins the seeded clusters
    and keeps the second one in review against its claimed actor."""
    from doppio.config import Config
    from doppio.job import run_stages
    from doppio.stages.characters import CharactersStage

    ws = make_ws(tmp_path)
    ep = _ep(ws, "S01E01")
    rng = np.random.default_rng(3)
    data = synthetic()["S01E01"]
    data["S2"] = (noisy(rng, CENTERS[0]), 25.0)
    write_diarize(ep, data, [(0, 20, "S0"), (20, 40, "S1"), (40, 60, "S2")])
    r = runner.invoke(
        app, ["registry", "cluster", str(ws.root), "--episodes", "S01E01,S01E02,S01E03"]
    )
    assert r.exit_code == 0, r.output
    path = ep.work / "05-characters" / "assignments.json"
    by = {c.cluster: c for c in read_json(path, Assignments).clusters}
    assert by["S0"].status == "matched"
    assert by["S2"].status == "review" and by["S2"].actor_id == by["S0"].actor_id
    assert by["S2"].character_id is None
    with Registry(ws.registry_path) as reg:
        assert reg.cluster_actors("S01E01", "S2") == []
        assert reg.exemplar_count(1) == 18

    # what the next `doppio run` does for this episode
    run_stages(ep, Config(), [CharactersStage()])
    by = {c.cluster: c for c in read_json(path, Assignments).clusters}
    assert by["S0"].status == "pinned" and by["S1"].status == "pinned"
    assert by["S2"].status == "review" and by["S2"].actor_id == by["S0"].actor_id
    with Registry(ws.registry_path) as reg:
        assert len(reg.actors()) == 2
        assert reg.exemplar_count(1) == 18 and reg.exemplar_count(2) == 18


def test_cli_cluster_interrupted_leaves_registry_empty(tmp_path, monkeypatch):
    ws = make_ws(tmp_path)
    calls = []
    real = Registry.add_exemplars

    def flaky(self, *a, **k):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(self, *a, **k)

    monkeypatch.setattr(Registry, "add_exemplars", flaky)
    eps = "S01E01,S01E02,S01E03"
    runner.invoke(app, ["registry", "cluster", str(ws.root), "--episodes", eps])
    monkeypatch.undo()
    with Registry(ws.registry_path) as reg:
        assert reg.actors() == [] and reg.characters() == []
    r = runner.invoke(app, ["registry", "cluster", str(ws.root), "--episodes", eps])
    assert r.exit_code == 0, r.output
