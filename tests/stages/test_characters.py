from typing import Any

import numpy as np
import pytest

from doppio.artifacts import read_json, write_json
from doppio.config import Config, RegistrySettings
from doppio.models import Assignments, Diarization, Turn
from doppio.registry.store import Registry
from doppio.stages.characters import CharactersStage
from doppio.workspace import Episode, Workspace


def unit(v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def make_ep(tmp_path, eid="e1"):
    root = tmp_path / "ws"
    return Episode(source=tmp_path / "s.mkv", id=eid, work=root / "eps" / eid, workspace_root=root)


def write_diarize(ep, clusters, turns):
    """clusters: {label: (embedding rows, clean_seconds)}; turns: [(start, end, label)]."""
    d = ep.work / "04-diarize"
    d.mkdir(parents=True, exist_ok=True)
    write_json(
        d / "turns.json",
        Diarization(backend="x", turns=[Turn(start=s, end=e, speaker=lb) for s, e, lb in turns]),
    )
    arrays: dict[str, Any] = {"labels": np.array(list(clusters), dtype=str)}
    for lb, (rows, secs) in clusters.items():
        rows = np.array(rows, dtype=np.float32).reshape(-1, 3)
        arrays[f"{lb}/embeddings"] = rows
        arrays[f"{lb}/spans"] = np.zeros((len(rows), 2))
        arrays[f"{lb}/clean_seconds"] = np.array(secs, dtype=np.float64)
    np.savez(d / "clusters.npz", **arrays)


CFG = Config(registry=RegistrySettings(min_new_seconds=20.0, match_threshold=0.8))


def registry(ep):
    return Registry(Workspace(ep.workspace_root).registry_path)


def assignments(ep):
    return read_json(ep.work / "05-characters" / "assignments.json", Assignments)


def test_first_run_creates_actors_for_new_only(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(
        ep,
        {
            "S0": ([unit([1, 0, 0]), unit([1, 0.1, 0])], 40.0),
            "S1": ([unit([0, 1, 0])], 5.0),
        },
        [(0, 20, "S0"), (20, 25, "S1")],
    )
    CharactersStage().run(ep, CFG)
    a = {c.cluster: c for c in assignments(ep).clusters}
    assert a["S0"].status == "new" and a["S0"].actor_id and a["S0"].character_id
    assert a["S1"].status == "extra" and a["S1"].actor_id is None
    with registry(ep) as reg:
        assert len(reg.actors()) == 1
        assert reg.exemplar_count(1) == 2 and a["S0"].actor_id == 1
    assert (ep.work / "05-characters" / "turns.prev.json").exists()


def test_second_episode_matches_same_character(tmp_path):
    ep1, ep2 = make_ep(tmp_path, "e1"), make_ep(tmp_path, "e2")
    write_diarize(ep1, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep1, CFG)
    write_diarize(ep2, {"S3": ([unit([1, 0.05, 0])], 30.0)}, [(0, 20, "S3")])
    CharactersStage().run(ep2, CFG)
    c1, c2 = assignments(ep1).clusters[0], assignments(ep2).clusters[0]
    assert c2.status == "matched" and c2.character_id == c1.character_id
    with registry(ep1) as reg:
        assert len(reg.actors()) == 1
        assert reg.exemplar_count(1) == 2 and c1.actor_id == 1


@pytest.mark.parametrize(
    "clusters, turns",
    [
        ({}, []),
        ({"S0": ([unit([1, 0, 0])], 3.0), "S1": ([], 0.0)}, [(0, 4, "S0"), (4, 5, "S1")]),
    ],
)
def test_no_qualifying_clusters(tmp_path, clusters, turns):
    ep = make_ep(tmp_path)
    write_diarize(ep, clusters, turns)
    CharactersStage().run(ep, CFG)
    got = assignments(ep).clusters
    assert [c.cluster for c in got] == list(clusters)
    assert all(c.status == "extra" for c in got)
    with registry(ep) as reg:
        assert reg.actors() == [] and reg.characters() == []


def test_rerun_after_registry_drift_keeps_assignment(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    first = assignments(ep).clusters[0]
    aid = first.actor_id
    assert aid is not None
    with registry(ep) as reg:
        reg.add_exemplars(aid, "other", "X", np.array([unit([0, 0, 1])] * 5), 10.0)
        reg.recompute_centroid(aid)
    # the diarization is rerun (labels renamed); the same speech
    write_diarize(ep, {"S7": ([unit([1, 0, 0])], 40.0)}, [(0.5, 19.5, "S7")])
    CharactersStage().run(ep, CFG)
    again = assignments(ep).clusters[0]
    assert again.cluster == "S7"
    assert again.status == "pinned"
    assert again.character_id == first.character_id
    with registry(ep) as reg:
        assert reg.exemplar_count(aid) == 1 + 5  # pinned adds none


def test_review_cluster_adds_nothing(tmp_path):
    ep1, ep2 = make_ep(tmp_path, "e1"), make_ep(tmp_path, "e2")
    write_diarize(ep1, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep1, CFG)
    write_diarize(
        ep2,
        {"A": ([unit([1, 0.05, 0])], 30.0), "B": ([unit([1, 0.2, 0])], 30.0)},
        [(0, 20, "A"), (20, 40, "B")],
    )
    CharactersStage().run(ep2, CFG)
    st = {c.cluster: c.status for c in assignments(ep2).clusters}
    assert st == {"A": "matched", "B": "review"}
    with registry(ep1) as reg:
        assert len(reg.actors()) == 1 and reg.exemplar_count(1) == 2


def test_stage_declares_files(tmp_path):
    ep = make_ep(tmp_path)
    s = CharactersStage()
    assert s.name == "characters"
    assert [p.name for p in s.inputs(ep)] == ["turns.json", "clusters.npz"]
    assert s.config_key(CFG, ep) == {"threshold": 0.8, "min_new_seconds": 20.0}


def test_rerun_without_assignments_does_not_duplicate_exemplars(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0]), unit([1, 0.1, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    first = assignments(ep).clusters[0]
    for p in CharactersStage().outputs(ep):
        p.unlink()
    CharactersStage().run(ep, CFG)
    again = assignments(ep).clusters[0]
    with registry(ep) as reg:
        assert reg.exemplar_count(1) == 2
        assert len(reg.actors()) == 1
    assert again.character_id == first.character_id


def test_previous_extra_is_rematched_not_pinned(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 5.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    assert assignments(ep).clusters[0].status == "extra"
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    assert assignments(ep).clusters[0].status == "new"
