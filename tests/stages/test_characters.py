from typing import Any

import numpy as np
import pytest

from fukikae.artifacts import read_json, write_json
from fukikae.config import Config, RegistrySettings
from fukikae.models import Assignments, Diarization, Turn
from fukikae.registry.store import Registry
from fukikae.stages.characters import CharactersStage
from fukikae.workspace import Episode, Workspace


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
    stored = assignments(ep).turns
    assert stored is not None and [t.speaker for t in stored] == ["S0", "S1"]
    assert not (ep.work / "05-characters" / "turns.prev.json").exists()


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
    assert [p.name for p in CharactersStage().outputs(ep)] == ["assignments.json"]
    CharactersStage().run(ep, CFG)
    again = assignments(ep).clusters[0]
    # matched afresh: the actor only this episode made is replaced, never self-matched
    assert again.status == "new" and again.character_id != first.character_id
    assert again.actor_id is not None
    with registry(ep) as reg:
        assert [a.id for a in reg.actors()] == [again.actor_id]
        assert reg.exemplar_count(again.actor_id) == 2


def test_previous_extra_is_rematched_not_pinned(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 5.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    assert assignments(ep).clusters[0].status == "extra"
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    assert assignments(ep).clusters[0].status == "new"


def test_previous_assignments_without_turns_cannot_pin(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    path = ep.work / "05-characters" / "assignments.json"
    prev = read_json(path, Assignments)
    write_json(path, prev.model_copy(update={"turns": None}))
    CharactersStage().run(ep, CFG)
    got = assignments(ep).clusters[0]
    # re-matched, not pinned; the actor only this episode made was deleted first
    assert got.status == "new" and got.actor_id is not None
    with registry(ep) as reg:
        assert [a.id for a in reg.actors()] == [got.actor_id]
        assert reg.exemplar_count(got.actor_id) == 1


def test_pin_uses_turns_stored_with_assignments(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(
        ep,
        {"A": ([unit([1, 0, 0])], 40.0), "B": ([unit([0, 1, 0])], 40.0)},
        [(0, 20, "A"), (20, 40, "B")],
    )
    CharactersStage().run(ep, CFG)
    first = {c.cluster: c.character_id for c in assignments(ep).clusters}
    # labels swapped by the new diarization; only the stored turns say who is who
    write_diarize(
        ep,
        {"A": ([unit([0, 1, 0])], 40.0), "B": ([unit([1, 0, 0])], 40.0)},
        [(0, 20, "B"), (20, 40, "A")],
    )
    CharactersStage().run(ep, CFG)
    now = {c.cluster: c.character_id for c in assignments(ep).clusters}
    assert all(c.status == "pinned" for c in assignments(ep).clusters)
    assert now == {"A": first["B"], "B": first["A"]}


def test_orphaned_cluster_exemplars_are_dropped(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    first = assignments(ep).clusters[0]
    # a re-diarization that only renames the label: pinned, exemplars follow the new label
    write_diarize(ep, {"S7": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S7")])
    CharactersStage().run(ep, CFG)
    assert assignments(ep).clusters[0].status == "pinned"
    # S7's speech now sits in a bigger cluster with only 1/3 coverage: nothing inherits it
    write_diarize(ep, {"X": ([unit([1, 0, 0])], 60.0)}, [(0, 60, "X")])
    CharactersStage().run(ep, CFG)
    got = assignments(ep).clusters[0]
    assert got.status == "matched" and got.character_id == first.character_id
    with registry(ep) as reg:
        assert len(reg.actors()) == 1
        assert reg.exemplar_count(1) == 1


LOW = Config(registry=RegistrySettings(min_new_seconds=20.0, match_threshold=0.7))


def two_episodes(tmp_path):
    """e1 creates actors 1 (A) and 2 (B); e2's X and Y sit at 0.74 to them, so at 0.8 they
    are new (actors 3, 4) and at 0.7 they would join e1's actors."""
    ep1, ep2 = make_ep(tmp_path, "e1"), make_ep(tmp_path, "e2")
    write_diarize(
        ep1,
        {"A": ([unit([1, 0, 0])], 40.0), "B": ([unit([0, 1, 0])], 40.0)},
        [(0, 20, "A"), (20, 40, "B")],
    )
    CharactersStage().run(ep1, CFG)
    write_diarize(
        ep2,
        {"X": ([unit([1, 0, 0.9])] * 2, 30.0), "Y": ([unit([0, 1, 0.9])] * 2, 30.0)},
        [(0, 20, "X"), (20, 40, "Y")],
    )
    CharactersStage().run(ep2, CFG)
    assert {c.cluster: c.actor_id for c in assignments(ep2).clusters} == {"X": 3, "Y": 4}
    return ep1, ep2


def check_joined(ep1, ep2):
    first = {c.cluster: c for c in assignments(ep1).clusters}
    got = {c.cluster: c for c in assignments(ep2).clusters}
    assert got["X"].status == "matched" and got["X"].actor_id == first["A"].actor_id
    assert got["Y"].status == "matched" and got["Y"].actor_id == first["B"].actor_id
    assert got["X"].character_id == first["A"].character_id
    with registry(ep1) as reg:
        assert [a.id for a in reg.actors()] == [1, 2]  # e2's own actors were deleted
        assert sorted(c.actor_id for c in reg.characters()) == [1, 2]
        assert reg.exemplar_count(1) == 3 and reg.exemplar_count(2) == 3


def test_rerun_without_assignments_never_matches_own_deleted_actors(tmp_path):
    ep1, ep2 = two_episodes(tmp_path)
    (ep2.work / "05-characters" / "assignments.json").unlink()
    CharactersStage().run(ep2, LOW)
    check_joined(ep1, ep2)


def test_rematch_ignores_previous_assignments(tmp_path):
    ep1, ep2 = two_episodes(tmp_path)
    CharactersStage(rematch=True).run(ep2, LOW)
    check_joined(ep1, ep2)


def test_rematch_spares_protected_actors(tmp_path):
    ep1, ep2 = two_episodes(tmp_path)
    with registry(ep1) as reg:
        reg.set_actor(3, name="Kept")
    CharactersStage(rematch=True).run(ep2, LOW)
    got = {c.cluster: c.actor_id for c in assignments(ep2).clusters}
    assert got == {"X": 3, "Y": 2}  # the named actor stays, and stays matchable
    with registry(ep1) as reg:
        assert [a.id for a in reg.actors()] == [1, 2, 3]


def test_pinned_actor_from_this_episode_survives_rerun(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])] * 2, 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    first = assignments(ep).clusters[0]
    write_diarize(ep, {"S7": ([unit([1, 0, 0])] * 2, 40.0)}, [(0, 20, "S7")])
    CharactersStage().run(ep, CFG)
    again = assignments(ep).clusters[0]
    assert again.status == "pinned" and again.actor_id == first.actor_id
    with registry(ep) as reg:
        assert [a.id for a in reg.actors()] == [first.actor_id]
        assert reg.cluster_actors(ep.id, "S7") == [first.actor_id]
        assert reg.exemplar_count(1) == 2


def test_unclaimed_actor_of_this_episode_is_pruned_on_rerun(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep, CFG)
    # a new diarization: different speech at other times; nothing inherits or matches S0
    write_diarize(ep, {"S1": ([unit([0, 1, 0])], 40.0)}, [(30, 50, "S1")])
    CharactersStage().run(ep, CFG)
    got = assignments(ep).clusters[0]
    assert got.status == "new" and got.actor_id is not None
    with registry(ep) as reg:
        assert [a.id for a in reg.actors()] == [got.actor_id]
        assert len(reg.characters()) == 1


def test_failed_run_leaves_registry_untouched(tmp_path, monkeypatch):
    ep = make_ep(tmp_path)
    write_diarize(
        ep,
        {"A": ([unit([1, 0, 0])], 40.0), "B": ([unit([0, 1, 0])], 40.0)},
        [(0, 20, "A"), (20, 40, "B")],
    )
    calls = []
    real = Registry.add_exemplars

    def flaky(self, *a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(self, *a, **k)

    monkeypatch.setattr(Registry, "add_exemplars", flaky)
    with pytest.raises(KeyboardInterrupt):
        CharactersStage().run(ep, CFG)
    monkeypatch.undo()
    with registry(ep) as reg:
        assert reg.actors() == [] and reg.characters() == []
    assert not (ep.work / "05-characters" / "assignments.json").exists()
    CharactersStage().run(ep, CFG)
    with registry(ep) as reg:
        assert len(reg.actors()) == 2


def test_split_half_of_a_pinned_speaker_goes_to_review(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"S0": ([unit([1, 0, 0])], 40.0)}, [(0, 40, "S0")])
    CharactersStage().run(ep, CFG)
    first = assignments(ep).clusters[0]
    # a diarize rerun splits the speaker: S0 keeps most turns (pinned); S1 holds the rest
    # plus new speech, too little overlap to pin, but it best-matches S0's actor
    write_diarize(
        ep,
        {"S0": ([unit([1, 0, 0])], 30.0), "S1": ([unit([1, 0.05, 0])], 25.0)},
        [(0, 30, "S0"), (30, 40, "S1"), (50, 70, "S1")],
    )
    CharactersStage().run(ep, CFG)
    got = {c.cluster: c for c in assignments(ep).clusters}
    assert got["S0"].status == "pinned" and got["S0"].actor_id == first.actor_id
    assert got["S1"].status == "review" and got["S1"].actor_id == first.actor_id
    assert got["S1"].character_id is None
    with registry(ep) as reg:
        assert reg.cluster_actors(ep.id, "S1") == []  # review adds no exemplars


# ---- matcher option: centroid (default) or lda with the series' stored projection ----

LDA = {"characters": {"matcher": "lda"}}
ONTO_E0 = np.array([[1.0], [0.0], [0.0]])  # keeps only the first axis


def lda_setup(tmp_path, project=True):
    """E1 makes an actor at [1, 1, 0]; E2's cluster at [1, 0, 1] has cosine 0.5 with it
    (new under centroid) and cosine 1 once projected onto the first axis."""
    ep1, ep2 = make_ep(tmp_path, "e1"), make_ep(tmp_path, "e2")
    write_diarize(ep1, {"S0": ([unit([1, 1, 0])], 40.0)}, [(0, 20, "S0")])
    CharactersStage().run(ep1, CFG)
    if project:
        with registry(ep1) as reg:
            reg.save_projection("default", "lda", ONTO_E0, np.zeros(3))
    write_diarize(ep2, {"S3": ([unit([1, 0, 1])], 30.0)}, [(0, 20, "S3")])
    return ep1, ep2


def with_raw(raw, profile_raw=None, source_language=None):
    return Config(
        registry=CFG.registry,
        raw=raw,
        profile_raw=profile_raw or {},
        source_language=source_language,
    )


def test_centroid_is_the_default_matcher_and_its_key_is_unchanged(tmp_path):
    _, ep2 = lda_setup(tmp_path)
    assert CharactersStage().config_key(CFG, ep2) == {"threshold": 0.8, "min_new_seconds": 20.0}
    CharactersStage().run(ep2, CFG)
    assert assignments(ep2).clusters[0].status == "new"


def test_lda_matcher_applies_the_stored_projection(tmp_path):
    ep1, ep2 = lda_setup(tmp_path)
    cfg = with_raw(LDA)
    CharactersStage().run(ep2, cfg)
    c = assignments(ep2).clusters[0]
    assert c.status == "matched" and c.actor_id == assignments(ep1).clusters[0].actor_id
    assert c.score == pytest.approx(1.0)
    key = CharactersStage().config_key(cfg, ep2)
    assert key["matcher"]["name"] == "lda" and key["matcher"]["projection"]


def test_lda_without_a_projection_falls_back_to_centroid_with_a_warning(tmp_path, caplog):
    import logging

    _, ep2 = lda_setup(tmp_path, project=False)
    with caplog.at_level(logging.WARNING):
        CharactersStage().run(ep2, with_raw(LDA))
    assert assignments(ep2).clusters[0].status == "new"
    assert "no lda projection" in caplog.text and "registry adapt" in caplog.text


def test_lda_projection_of_another_dimension_is_an_error(tmp_path):
    from fukikae.config import ConfigError

    ep1, ep2 = lda_setup(tmp_path, project=False)
    with registry(ep1) as reg:
        reg.save_projection("default", "lda", np.ones((5, 1)), np.zeros(5))
    with pytest.raises(ConfigError, match="registry adapt"):
        CharactersStage().run(ep2, with_raw(LDA))


def test_matcher_resolves_through_profile_and_language_layers():
    from fukikae.stages.characters import matcher_config

    prof = {"characters": {"matcher": "lda", "dims": 4}}
    assert matcher_config(with_raw({}, prof), None).backend == "lda"
    assert matcher_config(with_raw({}, prof), None).options == {"dims": 4}
    # the workspace's own table wins over the profile (and gets only its own options)
    got = matcher_config(with_raw({"characters": {"matcher": "centroid"}}, prof), None)
    assert got.backend == "centroid" and got.options == {}
    lang = {"characters": {"lang": {"ja": {"matcher": "lda"}}}}
    assert matcher_config(with_raw(lang), "ja").backend == "lda"
    assert matcher_config(with_raw(lang), "ko").backend == "centroid"
    assert matcher_config(with_raw({}), None).backend == "centroid"


def test_unknown_or_batch_matcher_is_refused(tmp_path):
    from fukikae.config import ConfigError

    _, ep2 = lda_setup(tmp_path)
    with pytest.raises(ConfigError, match="registry resolve"):
        CharactersStage().run(ep2, with_raw({"characters": {"matcher": "joint"}}))
    with pytest.raises(ConfigError, match="centroid"):
        CharactersStage().run(ep2, with_raw({"characters": {"matcher": "knn"}}))
    with pytest.raises(ConfigError, match="dims"):
        CharactersStage().run(ep2, with_raw({"characters": {"matcher": "lda", "dims": 0}}))


def test_centroid_matching_refuses_actors_of_another_dimension(tmp_path):
    from fukikae.config import ConfigError

    ep1, ep2 = lda_setup(tmp_path, project=False)
    with registry(ep1) as reg:
        aid, _ = reg.create_actor_with_character(np.ones(5, dtype=np.float32))
        reg.add_exemplars(aid, "other", "X", np.ones((1, 5), dtype=np.float32), 30.0)
    with pytest.raises(ConfigError, match="embedder changed"):
        CharactersStage().run(ep2, CFG)


def test_lda_refuses_actor_rows_of_another_dimension(tmp_path):
    from fukikae.config import ConfigError

    ep1, ep2 = lda_setup(tmp_path)
    with registry(ep1) as reg:
        aid, _ = reg.create_actor_with_character(np.ones(5, dtype=np.float32))
        reg.add_exemplars(aid, "other", "X", np.ones((1, 5), dtype=np.float32), 30.0)
    with pytest.raises(ConfigError, match="registry adapt"):
        CharactersStage().run(ep2, with_raw(LDA))


def test_adapt_and_stage_docs_say_to_leave_out_held_out_episodes():
    from typer.testing import CliRunner

    from fukikae.cli import app
    from fukikae.stages import characters

    assert "held-out" in (characters.__doc__ or "")
    r = CliRunner().invoke(app, ["registry", "adapt", "--help"], env={"COLUMNS": "200"})
    assert "held-out" in r.output
