"""`fukikae registry resolve`: the joint matcher over a season's episodes at once (Ruling R12),
rewriting the registry and the episodes' assignments in one transaction."""

from datetime import UTC, datetime

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.artifacts import read_json
from fukikae.cli import app
from fukikae.job import Manifest, StageRecord, cache_key, load_manifest, save_manifest
from fukikae.models import Assignments
from fukikae.registry.store import Registry
from fukikae.stages.characters import CharactersStage
from fukikae.workspace import Episode, Workspace
from tests.stages.test_characters import unit, write_diarize

runner = CliRunner()
A, B, C = unit([1, 0, 0]), unit([0, 1, 0]), unit([0, 0, 1])


def noisy(center, n=6, seed=0):
    rng = np.random.default_rng(seed)
    return [unit(center + rng.normal(0, 0.05, 3)) for _ in range(n)]


SEASON = {
    "S01E01": {"S0": (noisy(A, seed=1), 40.0), "S1": (noisy(B, seed=2), 35.0), "S2": ([C], 5.0)},
    "S01E02": {"S0": (noisy(B, seed=3), 30.0), "S1": (noisy(A, seed=4), 30.0)},
}


def episode(ws: Workspace, eid: str) -> Episode:
    return Episode(source=ws.root / eid, id=eid, work=ws.work_dir(eid), workspace_root=ws.root)


def add_episode(ws: Workspace, eid: str, clusters, done: bool = True) -> Episode:
    ep = episode(ws, eid)
    write_diarize(ep, clusters, [(10.0 * i, 10.0 * i + 8, lb) for i, lb in enumerate(clusters)])
    if done:
        now = datetime.now(UTC)
        names = ["extract", "separate", "language", "asr", "diarize"]
        save_manifest(
            ep,
            Manifest(stages={n: StageRecord(key="k", status="done", started=now) for n in names}),
        )
    return ep


def make_ws(tmp_path, season=SEASON) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace(root)
    for eid, clusters in season.items():
        add_episode(ws, eid, clusters)
    return ws


def resolve(ws: Workspace, episodes: str, *extra: str):
    return runner.invoke(
        app,
        ["registry", "resolve", str(ws.root), "--episodes", episodes, "--threshold", "0.6"]
        + list(extra),
    )


def by_cluster(ws: Workspace, eid: str):
    a = read_json(ws.work_dir(eid) / "05-characters" / "assignments.json", Assignments)
    return {c.cluster: c for c in a.clusters}, a


def test_resolve_links_a_season_and_writes_assignments(tmp_path):
    ws = make_ws(tmp_path)
    r = resolve(ws, "S01E01,S01E02")
    assert r.exit_code == 0, r.output
    e1, a1 = by_cluster(ws, "S01E01")
    e2, a2 = by_cluster(ws, "S01E02")
    assert e1["S0"].status == "new" and e1["S1"].status == "new"
    assert e2["S1"].character_id == e1["S0"].character_id  # A
    assert e2["S0"].character_id == e1["S1"].character_id  # B
    assert e1["S0"].character_id != e1["S1"].character_id
    assert e1["S2"].status == "extra" and e1["S2"].character_id is None
    assert a1.turns is not None and [t.speaker for t in a1.turns] == ["S0", "S1", "S2"]
    assert a2.threshold == 0.6
    with Registry(ws.registry_path) as reg:
        assert len(reg.actors()) == 2
        aid = e1["S0"].actor_id
        assert aid is not None
        assert reg.episode_counts(aid) == {"S01E01": 6, "S01E02": 6}
    # the characters stage is recorded done and current: `fukikae run` won't redo it
    for eid in ("S01E01", "S01E02"):
        ep = episode(ws, eid)
        rec = load_manifest(ep).stages["characters"]
        assert rec.status == "done"
        assert rec.key == cache_key(CharactersStage(), ep, ws.config)


def test_resolve_refuses_episodes_without_a_done_diarize_stage(tmp_path):
    ws = make_ws(tmp_path)
    add_episode(ws, "S01E03", {"S0": (noisy(A), 30.0)}, done=False)
    r = resolve(ws, "S01E01,S01E03")
    assert r.exit_code == 1
    assert "S01E03" in r.output and "diarize" in r.output
    assert not (ws.work_dir("S01E01") / "05-characters" / "assignments.json").exists()
    with Registry(ws.registry_path) as reg:
        assert reg.actors() == []


def test_re_resolve_prunes_its_own_actors_and_never_reuses_ids(tmp_path):
    ws = make_ws(tmp_path)
    assert resolve(ws, "S01E01,S01E02").exit_code == 0
    r = resolve(ws, "S01E01,S01E02")
    assert r.exit_code == 0, r.output
    e1, _ = by_cluster(ws, "S01E01")
    with Registry(ws.registry_path) as reg:
        assert sorted(a.id for a in reg.actors()) == [3, 4]  # 1 and 2 pruned, not reused
        assert sorted(reg.exemplar_count(a.id) for a in reg.actors()) == [12, 12]  # not doubled
    assert {e1["S0"].actor_id, e1["S1"].actor_id} == {3, 4}


def test_named_actor_of_listed_episodes_is_kept_and_matched(tmp_path):
    ws = make_ws(tmp_path)
    assert resolve(ws, "S01E01,S01E02").exit_code == 0
    e1, _ = by_cluster(ws, "S01E01")
    cid = e1["S0"].character_id
    assert cid is not None
    assert runner.invoke(app, ["registry", "rename", str(ws.root), str(cid), "Lucy"]).exit_code == 0
    r = resolve(ws, "S01E01,S01E02")
    assert r.exit_code == 0, r.output
    again, _ = by_cluster(ws, "S01E01")
    e2, _ = by_cluster(ws, "S01E02")
    assert again["S0"].status == "matched" and again["S0"].character_id == cid
    assert e2["S1"].character_id == cid
    with Registry(ws.registry_path) as reg:
        names = {c.id: c.name for c in reg.characters()}
        assert names[cid] == "Lucy"
        assert len(reg.actors()) == 2  # Lucy's actor kept; the unnamed one replaced


def test_actors_of_episodes_outside_the_list_are_matched_not_replaced(tmp_path):
    ws = make_ws(tmp_path)
    assert resolve(ws, "S01E01,S01E02").exit_code == 0
    e1, a1_before = by_cluster(ws, "S01E01")
    a_actor, b_actor = e1["S0"].actor_id, e1["S1"].actor_id
    assert a_actor is not None and b_actor is not None
    assert runner.invoke(app, ["registry", "lock", str(ws.root), str(a_actor)]).exit_code == 0
    with Registry(ws.registry_path) as reg:
        locked_centroid = next(a.centroid for a in reg.actors() if a.id == a_actor)
    add_episode(ws, "S01E03", {"S0": (noisy(A, seed=9), 30.0), "S1": (noisy(C, seed=8), 25.0)})
    r = resolve(ws, "S01E03")
    assert r.exit_code == 0, r.output
    e3, _ = by_cluster(ws, "S01E03")
    assert e3["S0"].status == "matched" and e3["S0"].actor_id == a_actor
    assert e3["S0"].character_id == e1["S0"].character_id
    assert e3["S1"].status == "new" and e3["S1"].actor_id not in (a_actor, b_actor)
    with Registry(ws.registry_path) as reg:
        actors = {a.id: a for a in reg.actors()}
        assert set(actors) == {a_actor, b_actor, e3["S1"].actor_id}
        got = actors[a_actor].centroid
        assert got is not None and locked_centroid is not None
        assert np.array_equal(got, locked_centroid)  # locked
        assert reg.episode_counts(b_actor) == {"S01E01": 6, "S01E02": 6}
    _, a1_after = by_cluster(ws, "S01E01")
    assert a1_after == a1_before  # an episode outside the list is untouched


def test_existing_actors_never_merge_with_each_other(tmp_path):
    """Two actors of other episodes with close voices (cosine ~0.9) stay two; the listed
    episode's cluster joins the closer one."""
    ws = make_ws(tmp_path, {"S01E05": {"S0": (noisy(unit([1, 0.45, 0]), seed=5), 30.0)}})
    with Registry(ws.registry_path) as reg:
        for center in (unit([1, 0.5, 0]), unit([1, -0.0, 0])):
            aid, _ = reg.create_actor_with_character(center)
            reg.add_exemplars(aid, "S01E09", "X", np.array(noisy(center, seed=aid)), 30.0)
            reg.recompute_centroid(aid)
    r = resolve(ws, "S01E05", "--threshold", "0.5")
    assert r.exit_code == 0, r.output
    e5, _ = by_cluster(ws, "S01E05")
    assert e5["S0"].status == "matched" and e5["S0"].actor_id == 1
    with Registry(ws.registry_path) as reg:
        assert [a.id for a in reg.actors()] == [1, 2]


def test_interrupted_resolve_changes_nothing(tmp_path, monkeypatch):
    ws = make_ws(tmp_path)
    assert resolve(ws, "S01E01,S01E02").exit_code == 0
    with Registry(ws.registry_path) as reg:
        before = [(a.id, reg.exemplar_count(a.id)) for a in reg.actors()]
    _, a1 = by_cluster(ws, "S01E01")
    calls = {"n": 0}
    real = Registry.add_exemplars

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Registry, "add_exemplars", flaky)
    with pytest.raises(KeyboardInterrupt):
        from fukikae.registry.resolve import resolve_episodes

        resolve_episodes(ws, [episode(ws, "S01E01"), episode(ws, "S01E02")], threshold=0.6)
    monkeypatch.undo()
    with Registry(ws.registry_path) as reg:
        assert [(a.id, reg.exemplar_count(a.id)) for a in reg.actors()] == before
    _, again = by_cluster(ws, "S01E01")
    assert again == a1


def test_resolve_options_are_checked(tmp_path):
    ws = make_ws(tmp_path)
    r = resolve(ws, "S01E01,S01E02", "--matcher", "centroid")
    assert r.exit_code == 1 and "joint" in r.output
    r = resolve(ws, "S01E01,S01E02", "--matcher-option", "seed=segments")
    assert r.exit_code == 1 and "seed" in r.output
    r = resolve(ws, "S01E01,S01E02", "--matcher-option", "bogus=1")
    assert r.exit_code == 1 and "bogus" in r.output


def test_episode_cannot_link_off_lets_one_character_take_two_clusters(tmp_path):
    """A speaker split in two by the diarizer within an episode: with episode_cannot_link
    off both clusters get the one character."""
    season = {
        "S01E01": {"S0": (noisy(A, seed=1), 30.0), "S1": (noisy(A, seed=2), 30.0)},
        "S01E02": {"S0": (noisy(A, seed=3), 30.0)},
    }
    ws = make_ws(tmp_path, season)
    assert resolve(ws, "S01E01,S01E02").exit_code == 0
    e1, _ = by_cluster(ws, "S01E01")
    assert e1["S0"].character_id != e1["S1"].character_id
    r = resolve(ws, "S01E01,S01E02", "--matcher-option", "episode_cannot_link=false")
    assert r.exit_code == 0, r.output
    e1, _ = by_cluster(ws, "S01E01")
    e2, _ = by_cluster(ws, "S01E02")
    assert e1["S0"].character_id == e1["S1"].character_id == e2["S0"].character_id
