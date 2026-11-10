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
    assert [p.name for p in s.inputs(ep)] == ["names.json", "turns.json", "clusters.npz"]
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


TITLED = {**LDA, "series": {"title": "Grid"}}


def test_lda_falls_back_to_the_default_series_with_a_warning(tmp_path, caplog):
    import logging

    ep1, ep2 = lda_setup(tmp_path)  # stored under "default" only
    cfg = with_raw(TITLED)
    with registry(ep1) as reg:
        created = reg.load_projection("default", "lda").created  # pyright: ignore[reportOptionalMemberAccess]
    with caplog.at_level(logging.WARNING):
        CharactersStage().run(ep2, cfg)
    c = assignments(ep2).clusters[0]
    assert c.status == "matched" and c.score == pytest.approx(1.0)
    assert "'Grid'" in caplog.text and "'default'" in caplog.text
    # The manifest key names the projection that was applied.
    assert CharactersStage().config_key(cfg, ep2)["matcher"]["projection"] == created


def test_lda_uses_the_titled_series_without_fallback(tmp_path, caplog):
    import logging

    ep1, ep2 = lda_setup(tmp_path, project=False)
    with registry(ep1) as reg:
        reg.save_projection("Grid", "lda", ONTO_E0, np.zeros(3))
        reg.save_projection("default", "lda", np.array([[0.0], [0.0], [1.0]]), np.zeros(3))
        created = reg.load_projection("Grid", "lda").created  # pyright: ignore[reportOptionalMemberAccess]
    cfg = with_raw(TITLED)
    with caplog.at_level(logging.WARNING):
        CharactersStage().run(ep2, cfg)
    assert assignments(ep2).clusters[0].score == pytest.approx(1.0)  # Grid's, not default's
    assert "default" not in caplog.text
    assert CharactersStage().config_key(cfg, ep2)["matcher"]["projection"] == created


def test_lda_missing_under_both_keys_names_them(tmp_path, caplog):
    import logging

    _, ep2 = lda_setup(tmp_path, project=False)
    with caplog.at_level(logging.WARNING):
        CharactersStage().run(ep2, with_raw(TITLED))
    assert assignments(ep2).clusters[0].status == "new"
    assert "'Grid' or 'default'" in caplog.text


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


# --- names (Task 6): fake names.json, refined turns and clusters.npz ---------------------

NAMES_CFG = Config(registry=RegistrySettings(min_new_seconds=20.0, match_threshold=0.8))
WENDY, HAPPY, CHARLE, NATSU = (
    unit([1, 0, 0]),
    unit([0.9, 0.3, 0]),
    unit([0.9, 0, 0.3]),
    unit([0, 1, 0]),
)


def write_names(ep, labels, used=True, language="en"):
    """labels: {label: (parent, key or None, cls, rows, [(start, end)])}; writes names.json and
    the refined turns.json and clusters.npz under 04c-names/ as the names stage would."""
    from fukikae.names.models import LabelInfo, NamesResult

    d = ep.work / "04c-names"
    d.mkdir(parents=True, exist_ok=True)
    infos = {
        lb: LabelInfo(
            parent=parent,
            key=key,
            display=None if key is None else key.title(),
            cls="remainder" if key is None else cls,
        )
        for lb, (parent, key, cls, _rows, _spans) in labels.items()
    }
    turns = sorted(
        ((s, e, lb) for lb, (*_x, spans) in labels.items() for s, e in spans), key=lambda t: t[0]
    )
    write_json(
        d / "names.json",
        NamesResult(
            episode=ep.id,
            used=used,
            verdict="full" if used else "none",
            reason=None if used else "no named track",
            source={"how": "target", "language": language} if used else None,
            offset_s=0.0,
            coverage={},
            lines=[],
            hidden=[],
            pieces=[],
            labels=infos if used else {},
            unvoiced=[],
            excluded=[],
            dropped_row_seconds=0.0,
            suggestions=[],
        ),
    )
    if not used:
        return
    write_json(
        d / "turns.json",
        Diarization(backend="x", turns=[Turn(start=s, end=e, speaker=lb) for s, e, lb in turns]),
    )
    arrays: dict[str, Any] = {"labels": np.array(list(labels), dtype=str)}
    for lb, (*_x, rows, spans) in labels.items():
        r = np.array(rows, dtype=np.float32).reshape(-1, 3)
        arrays[f"{lb}/embeddings"] = r
        arrays[f"{lb}/spans"] = np.zeros((len(r), 2))
        arrays[f"{lb}/clean_seconds"] = np.array(sum(e - s for s, e in spans) * 0.8)
    np.savez(d / "clusters.npz", **arrays)


def merged_cluster():
    """E14's SPEAKER_08 in miniature: wendy 130 s of 208 s named, plus happy and charle."""
    return {
        "SPEAKER_08~wendy": ("SPEAKER_08", "wendy", "named", [WENDY] * 3, [(0, 130)]),
        "SPEAKER_08~happy": ("SPEAKER_08", "happy", "named", [HAPPY] * 2, [(130, 170)]),
        "SPEAKER_08~charle": ("SPEAKER_08", "charle", "named", [CHARLE] * 2, [(170, 208)]),
        "SPEAKER_08": ("SPEAKER_08", None, "remainder", [WENDY], [(208, 220)]),
    }


def pinned_merged(tmp_path, eid="e1"):
    """A names-less first run: SPEAKER_08 over the same 220 s becomes one unnamed character."""
    ep = make_ep(tmp_path, eid)
    rows = [WENDY] * 3 + [HAPPY] * 2 + [CHARLE] * 2
    write_diarize(ep, {"SPEAKER_08": (rows, 170.0)}, [(0, 220, "SPEAKER_08")])
    CharactersStage().run(ep, NAMES_CFG)
    (a,) = assignments(ep).clusters
    assert a.status == "new" and a.character_id is not None
    return ep, a.character_id


def by_cluster(ep):
    return {a.cluster: a for a in assignments(ep).clusters}


def test_stage_reads_names_json_and_the_identity_inputs(tmp_path):
    ep = make_ep(tmp_path)
    s = CharactersStage()
    assert [p.relative_to(ep.work).as_posix() for p in s.inputs(ep)] == [
        "04c-names/names.json",
        "04-diarize/turns.json",
        "04-diarize/clusters.npz",
    ]
    write_names(ep, merged_cluster())
    assert [p.relative_to(ep.work).as_posix() for p in s.inputs(ep)] == [
        "04c-names/names.json",
        "04c-names/turns.json",
        "04c-names/clusters.npz",
    ]
    assert s.config_key(NAMES_CFG, ep) == {
        "threshold": 0.8,
        "min_new_seconds": 20.0,
        "names_min_new_seconds": 5.0,
    }
    write_names(ep, {}, used=False)
    assert s.config_key(NAMES_CFG, ep) == {"threshold": 0.8, "min_new_seconds": 20.0}


def test_unused_names_leave_assignments_byte_identical(tmp_path):
    ep, _ = pinned_merged(tmp_path)
    CharactersStage().run(ep, NAMES_CFG)  # the steady state: pinned to its own first run
    before = (ep.work / "05-characters" / "assignments.json").read_bytes()
    write_names(ep, {}, used=False)
    CharactersStage().run(ep, NAMES_CFG)
    assert (ep.work / "05-characters" / "assignments.json").read_bytes() == before


def test_two_names_in_one_cluster_become_two_characters(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"SPEAKER_12": ([WENDY, NATSU], 40.0)}, [(0, 40, "SPEAKER_12")])
    write_names(
        ep,
        {
            "SPEAKER_12~wendy": ("SPEAKER_12", "wendy", "named", [WENDY], [(0, 25)]),
            "SPEAKER_12~natsu": ("SPEAKER_12", "natsu", "named", [NATSU], [(25, 40)]),
        },
    )
    CharactersStage().run(ep, NAMES_CFG)
    by = by_cluster(ep)
    w, n = by["SPEAKER_12~wendy"], by["SPEAKER_12~natsu"]
    assert w.character_id is not None and n.character_id is not None and n.actor_id is not None
    assert w.character_id != n.character_id
    assert (w.status, w.source, w.name) == ("new", "name", "wendy")
    assert (n.status, n.source, n.name) == ("new", "name", "natsu")
    with registry(ep) as reg:
        names = {c.id: (c.name, c.name_by) for c in reg.characters()}
        assert names[w.character_id] == ("Wendy", "subs")
        assert names[n.character_id] == ("Natsu", "subs")
        assert reg.exemplar_count(n.actor_id) == 1


def test_known_key_takes_several_clusters_of_one_episode(tmp_path):
    ep = make_ep(tmp_path)
    with registry(ep) as reg:
        aid, natsu = reg.create_actor_with_character(NATSU)
        reg.rename_character(natsu, "Natsu")
    write_diarize(
        ep,
        {"SPEAKER_04": ([NATSU], 20.0), "SPEAKER_09": ([unit([0, 1, 0.2])], 20.0)},
        [(0, 20, "SPEAKER_04"), (20, 40, "SPEAKER_09")],
    )
    write_names(
        ep,
        {
            "SPEAKER_04~natsu": ("SPEAKER_04", "natsu", "named", [NATSU], [(0, 20)]),
            "SPEAKER_09~natsu": ("SPEAKER_09", "natsu", "named", [unit([0, 1, 0.2])], [(20, 40)]),
        },
    )
    CharactersStage().run(ep, NAMES_CFG)
    got = list(by_cluster(ep).values())
    assert {a.character_id for a in got} == {natsu}
    assert {(a.status, a.source) for a in got} == {("matched", "name")}
    assert got[0].score is not None and got[0].score > 0.99
    with registry(ep) as reg:
        assert reg.exemplar_count(aid) == 2 and len(reg.characters()) == 1


def test_unnamed_character_is_renamed_when_one_name_dominates(tmp_path):
    ep, cid = pinned_merged(tmp_path)
    write_names(ep, merged_cluster())
    CharactersStage().run(ep, NAMES_CFG)
    by = by_cluster(ep)
    assert by["SPEAKER_08~wendy"].character_id == cid
    assert (by["SPEAKER_08~wendy"].status, by["SPEAKER_08~wendy"].source) == ("matched", "name")
    assert by["SPEAKER_08"].character_id == cid and by["SPEAKER_08"].status == "pinned"
    assert by["SPEAKER_08"].name is None and by["SPEAKER_08"].source == "voice"
    with registry(ep) as reg:
        chars = {c.name: c for c in reg.characters()}
        assert chars["Wendy"].id == cid and chars["Wendy"].name_by == "subs"
        assert {"Happy", "Charle"} <= set(chars)
        assert by["SPEAKER_08~happy"].character_id == chars["Happy"].id
        hit = reg.character_for_key("wendy")
        assert hit is not None and hit.id == cid
        # the episode's evidence moved: Wendy keeps only Wendy's and the remainder's rows
        assert reg.exemplar_count(chars["Wendy"].actor_id) == 3 + 1


def test_rerun_with_names_is_stable(tmp_path):
    ep, cid = pinned_merged(tmp_path)
    write_names(ep, merged_cluster())
    CharactersStage().run(ep, NAMES_CFG)
    first = (ep.work / "05-characters" / "assignments.json").read_bytes()
    with registry(ep) as reg:
        before = [(c.id, c.name, c.name_by) for c in reg.characters()]
    CharactersStage().run(ep, NAMES_CFG)
    by = by_cluster(ep)
    assert by["SPEAKER_08~wendy"].character_id == cid
    with registry(ep) as reg:
        assert [(c.id, c.name, c.name_by) for c in reg.characters()] == before
    assert json_clusters(first) == json_clusters(
        (ep.work / "05-characters" / "assignments.json").read_bytes()
    )


def json_clusters(raw: bytes):
    import json

    return {c["cluster"]: (c["character_id"], c.get("name")) for c in json.loads(raw)["clusters"]}


def test_no_rename_below_sixty_percent(tmp_path):
    ep, cid = pinned_merged(tmp_path)
    labels = merged_cluster()
    labels["SPEAKER_08~wendy"] = ("SPEAKER_08", "wendy", "named", [WENDY] * 3, [(0, 100)])
    labels["SPEAKER_08~happy"] = ("SPEAKER_08", "happy", "named", [HAPPY] * 2, [(100, 170)])
    write_names(ep, labels)  # wendy 100 of 208 s: under 60% of the character's named speech
    CharactersStage().run(ep, NAMES_CFG)
    with registry(ep) as reg:
        old = next(c for c in reg.characters() if c.id == cid)
        assert old.name is None
        assert {"Wendy", "Happy", "Charle"} <= {c.name for c in reg.characters()}


def test_curated_character_is_not_renamed(tmp_path):
    ep, cid = pinned_merged(tmp_path)
    with registry(ep) as reg:
        reg.assign_card(cid, "en", "0123456789abcdef", by="auto", locked=True)
    write_names(ep, merged_cluster())
    CharactersStage().run(ep, NAMES_CFG)
    with registry(ep) as reg:
        assert next(c for c in reg.characters() if c.id == cid).name is None
        wendy = reg.character_for_key("wendy")
        assert wendy is not None and wendy.id != cid


def test_operator_named_character_is_never_renamed_by_subs(tmp_path):
    ep, cid = pinned_merged(tmp_path)
    with registry(ep) as reg:
        reg.rename_character(cid, "Sky Maiden")
    write_names(ep, merged_cluster())
    CharactersStage().run(ep, NAMES_CFG)
    with registry(ep) as reg:
        c = next(c for c in reg.characters() if c.id == cid)
        assert (c.name, c.name_by) == ("Sky Maiden", "operator")
        wendy = reg.character_for_key("wendy")
        assert wendy is not None and wendy.id != cid


def test_operator_alias_beats_a_subtitle_name(tmp_path):
    ep, cid = pinned_merged(tmp_path)
    with registry(ep) as reg:
        _, other = reg.create_actor_with_character(NATSU)
        reg.add_alias(other, "WENDY", by="operator")  # the operator says wendy is `other`
    write_names(ep, merged_cluster())
    CharactersStage().run(ep, NAMES_CFG)
    assert by_cluster(ep)["SPEAKER_08~wendy"].character_id == other
    with registry(ep) as reg:
        assert next(c for c in reg.characters() if c.id == other).name is None
        assert next(c for c in reg.characters() if c.id == cid).name is None


def test_locked_actor_keeps_its_cluster_and_reports(tmp_path, caplog):
    import logging

    ep, cid = pinned_merged(tmp_path)
    with registry(ep) as reg:
        aid = next(c.actor_id for c in reg.characters() if c.id == cid)
        reg.lock_actor(aid, True)
    write_names(ep, merged_cluster())
    with caplog.at_level(logging.INFO, logger="fukikae.stages.characters"):
        CharactersStage().run(ep, NAMES_CFG)
    subs = [a for a in assignments(ep).clusters if a.cluster.startswith("SPEAKER_08~")]
    assert {a.character_id for a in subs} == {cid} and {a.source for a in subs} == {"lock"}
    assert {a.name for a in subs} == {"wendy", "happy", "charle"}
    assert "kept by lock" in caplog.text
    with registry(ep) as reg:
        assert [c.id for c in reg.characters()] == [cid]


def test_role_label_and_short_names_are_extras_with_name(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(
        ep,
        {"SPEAKER_00": ([WENDY], 20.0), "SPEAKER_03": ([NATSU], 4.0)},
        [(0, 20, "SPEAKER_00"), (20, 24, "SPEAKER_03")],
    )
    write_names(
        ep,
        {
            "SPEAKER_00~townspeople-b": ("SPEAKER_00", "townspeople b", "role", [WENDY], [(0, 20)]),
            "SPEAKER_03~loke": ("SPEAKER_03", "loke", "named", [NATSU], [(20, 24)]),
        },
    )
    CharactersStage().run(ep, NAMES_CFG)
    got = by_cluster(ep)
    assert {(a.status, a.source) for a in got.values()} == {("extra", "name")}
    assert got["SPEAKER_03~loke"].name == "loke" and got["SPEAKER_03~loke"].actor_id is None
    with registry(ep) as reg:
        assert reg.characters() == [] and reg.actors() == []


def test_known_key_with_short_speech_still_maps(tmp_path):
    ep = make_ep(tmp_path)
    with registry(ep) as reg:
        _, loke = reg.create_actor_with_character(NATSU)
        reg.rename_character(loke, "Loke")
    write_diarize(ep, {"SPEAKER_03": ([NATSU], 3.0)}, [(20, 23, "SPEAKER_03")])
    write_names(ep, {"SPEAKER_03~loke": ("SPEAKER_03", "loke", "named", [NATSU], [(20, 23)])})
    CharactersStage().run(ep, NAMES_CFG)
    a = by_cluster(ep)["SPEAKER_03~loke"]
    assert (a.status, a.character_id) == ("matched", loke)


def test_remainder_still_matches_acoustically_and_names_never_block_it(tmp_path):
    ep = make_ep(tmp_path)
    with registry(ep) as reg:
        _, natsu = reg.create_actor_with_character(NATSU)
        reg.rename_character(natsu, "Natsu")
    write_diarize(
        ep,
        {"SPEAKER_04": ([NATSU], 20.0), "SPEAKER_05": ([unit([0, 1, 0.1])], 30.0)},
        [(0, 20, "SPEAKER_04"), (20, 50, "SPEAKER_05")],
    )
    write_names(
        ep,
        {
            "SPEAKER_04~natsu": ("SPEAKER_04", "natsu", "named", [NATSU], [(0, 20)]),
            "SPEAKER_05": ("SPEAKER_05", None, "remainder", [unit([0, 1, 0.1])], [(20, 50)]),
        },
    )
    CharactersStage().run(ep, NAMES_CFG)
    rem = by_cluster(ep)["SPEAKER_05"]
    assert (rem.status, rem.character_id, rem.source, rem.name) == ("matched", natsu, "voice", None)


def test_an_ambiguous_key_is_reported_and_takes_the_acoustic_path(tmp_path, caplog):
    import logging

    ep = make_ep(tmp_path)
    with registry(ep) as reg:
        a = reg.create_actor_with_character(WENDY)[1]
        b = reg.create_actor_with_character(NATSU)[1]
        with reg._tx() as con:
            con.execute("UPDATE characters SET name = 'Lucy', name_by = 'operator'")
    write_diarize(ep, {"SPEAKER_01": ([HAPPY], 30.0)}, [(0, 30, "SPEAKER_01")])
    write_names(ep, {"SPEAKER_01~lucy": ("SPEAKER_01", "lucy", "named", [HAPPY], [(0, 30)])})
    with caplog.at_level(logging.INFO, logger="fukikae.stages.characters"):
        CharactersStage().run(ep, NAMES_CFG)
    got = by_cluster(ep)["SPEAKER_01~lucy"]
    # HAPPY is 0.95 from WENDY: matched by voice, as a remainder would be, and named nobody
    assert (got.status, got.character_id, got.source, got.name) == ("matched", a, "voice", None)
    assert "left to acoustics" in caplog.text and "registry rename" in caplog.text
    with registry(ep) as reg:
        assert sorted(c.id for c in reg.characters()) == sorted([a, b])


def test_disagreement_with_acoustics_is_reported_not_acted_on(tmp_path, caplog):
    import logging

    ep = make_ep(tmp_path)
    with registry(ep) as reg:
        _, natsu = reg.create_actor_with_character(NATSU)
        reg.rename_character(natsu, "Natsu")
        _, gray = reg.create_actor_with_character(WENDY)
        reg.rename_character(gray, "Gray")
    write_diarize(ep, {"SPEAKER_02": ([WENDY], 20.0)}, [(0, 20, "SPEAKER_02")])
    write_names(ep, {"SPEAKER_02~natsu": ("SPEAKER_02", "natsu", "named", [WENDY], [(0, 20)])})
    with caplog.at_level(logging.INFO, logger="fukikae.stages.characters"):
        CharactersStage().run(ep, NAMES_CFG)
    assert by_cluster(ep)["SPEAKER_02~natsu"].character_id == natsu
    assert "SPEAKER_02~natsu" in caplog.text and f"character {gray}" in caplog.text


def test_named_label_without_rows_creates_an_empty_centroid_character(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"SPEAKER_01": ([HAPPY], 30.0)}, [(0, 30, "SPEAKER_01")])
    write_names(ep, {"SPEAKER_01~happy": ("SPEAKER_01", "happy", "named", [], [(0, 30)])})
    CharactersStage().run(ep, NAMES_CFG)
    a = by_cluster(ep)["SPEAKER_01~happy"]
    assert a.status == "new" and a.character_id is not None
    with registry(ep) as reg:
        (actor,) = reg.actors()
        assert actor.centroid is None and reg.characters()[0].name == "Happy"


# --- Task 6 review: precedence of operator state, acoustic evidence of named labels ---------


def _below_rename(tmp_path):
    """pinned_merged with wendy under 60%, so the parent's character stays unnamed and wendy,
    happy and charle are created by subtitles."""
    ep, cid = pinned_merged(tmp_path)
    labels = merged_cluster()
    labels["SPEAKER_08~wendy"] = ("SPEAKER_08", "wendy", "named", [WENDY] * 3, [(0, 100)])
    labels["SPEAKER_08~happy"] = ("SPEAKER_08", "happy", "named", [HAPPY] * 2, [(100, 170)])
    write_names(ep, labels)
    return ep, cid


def test_lock_after_a_names_run_takes_effect_on_rerun(tmp_path):
    ep, cid = _below_rename(tmp_path)
    CharactersStage().run(ep, NAMES_CFG)
    assert by_cluster(ep)["SPEAKER_08~happy"].source == "name"
    with registry(ep) as reg:
        reg.lock_actor(next(c.actor_id for c in reg.characters() if c.id == cid), True)
    CharactersStage().run(ep, NAMES_CFG)
    by = by_cluster(ep)
    assert by["SPEAKER_08"].character_id == cid
    for k in ("wendy", "happy", "charle"):
        assert (by[f"SPEAKER_08~{k}"].character_id, by[f"SPEAKER_08~{k}"].source) == (cid, "lock")


def test_locked_character_keeps_its_cluster(tmp_path, caplog):
    import logging

    ep, cid = pinned_merged(tmp_path)
    with registry(ep) as reg:
        reg.lock_character(cid, True)
    write_names(ep, merged_cluster())
    with caplog.at_level(logging.INFO, logger="fukikae.stages.characters"):
        CharactersStage().run(ep, NAMES_CFG)
    subs = [a for a in assignments(ep).clusters if a.cluster.startswith("SPEAKER_08~")]
    assert {(a.character_id, a.source, a.status) for a in subs} == {(cid, "lock", "pinned")}
    assert "kept by lock" in caplog.text
    with registry(ep) as reg:
        assert [(c.id, c.name) for c in reg.characters()] == [(cid, None)]


def _disagreements(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "acoustics disagree" in r.getMessage()]


def test_disagreement_report_reproduces_on_rerun(tmp_path, caplog):
    import logging

    ep, _ = _below_rename(tmp_path)
    with caplog.at_level(logging.INFO, logger="fukikae.stages.characters"):
        CharactersStage().run(ep, NAMES_CFG)
        first = _disagreements(caplog)
        caplog.clear()
        CharactersStage().run(ep, NAMES_CFG)
        second = _disagreements(caplog)
    assert len(first) == 3 and second == first


def test_operator_rename_takes_a_key_from_a_subtitle_character(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"SPEAKER_12": ([WENDY, NATSU], 40.0)}, [(0, 40, "SPEAKER_12")])
    write_names(
        ep,
        {
            "SPEAKER_12~wendy": ("SPEAKER_12", "wendy", "named", [WENDY], [(0, 25)]),
            "SPEAKER_12~natsu": ("SPEAKER_12", "natsu", "named", [NATSU], [(25, 40)]),
        },
    )
    CharactersStage().run(ep, NAMES_CFG)
    y = by_cluster(ep)["SPEAKER_12~wendy"].character_id
    with registry(ep) as reg:
        x_actor, x = reg.create_actor_with_character(HAPPY)
        reg.add_exemplars(x_actor, "e0", "S0", np.stack([HAPPY]), 5.0)
        assert reg.rename_character(x, "Wendy") is None
    CharactersStage().run(ep, NAMES_CFG)
    a = by_cluster(ep)["SPEAKER_12~wendy"]
    assert (a.character_id, a.source) == (x, "name")
    with registry(ep) as reg:
        wendy = {(al.alias, al.character_id, al.by) for al in reg.aliases() if al.alias == "wendy"}
        assert wendy == {("wendy", x, "operator")}
        assert reg.aliases(y) == []
        assert reg.exemplar_count(x_actor) == 2


def test_rematch_with_names_keeps_a_subtitle_character(tmp_path):
    ep = make_ep(tmp_path)
    write_diarize(ep, {"SPEAKER_12": ([WENDY, NATSU], 40.0)}, [(0, 40, "SPEAKER_12")])
    write_names(
        ep,
        {
            "SPEAKER_12~wendy": ("SPEAKER_12", "wendy", "named", [WENDY], [(0, 25)]),
            "SPEAKER_12~natsu": ("SPEAKER_12", "natsu", "named", [NATSU], [(25, 40)]),
        },
    )
    CharactersStage().run(ep, NAMES_CFG)
    y = by_cluster(ep)["SPEAKER_12~wendy"].character_id
    assert y is not None
    with registry(ep) as reg:
        reg.assign_card(y, "en", "0123456789abcdef", by="auto", locked=False)
    CharactersStage(rematch=True).run(ep, NAMES_CFG)
    a = by_cluster(ep)["SPEAKER_12~wendy"]
    assert (a.character_id, a.status, a.source) == (y, "matched", "name")
    with registry(ep) as reg:
        card = reg.card_for(y, "en")
        assert card is not None and card.card_id == "0123456789abcdef"


def test_remainder_matching_never_sees_this_episodes_named_rows(tmp_path):
    """Natsu (operator-named) holds one WENDY row from e0. e1's named natsu speech is NATSU;
    the remainder SPEAKER_05 sounds like NATSU. Both matchers judge it against the registry
    before e1's named rows: under lda (identity projection) as under centroid, it is new."""
    got = {}
    for matcher in ("centroid", "lda"):
        ep0 = make_ep(tmp_path / matcher, "e0")
        write_diarize(ep0, {"S0": ([WENDY], 40.0)}, [(0, 40, "S0")])
        CharactersStage().run(ep0, NAMES_CFG)
        with registry(ep0) as reg:
            reg.rename_character(reg.characters()[0].id, "Natsu")
            reg.save_projection("default", "lda", np.eye(3), np.zeros(3))
        ep = make_ep(tmp_path / matcher, "e1")
        near = unit([0, 1, 0.1])
        write_diarize(
            ep,
            {"SPEAKER_04": ([NATSU] * 9, 30.0), "SPEAKER_05": ([near], 30.0)},
            [(0, 30, "SPEAKER_04"), (30, 60, "SPEAKER_05")],
        )
        write_names(
            ep,
            {
                "SPEAKER_04~natsu": ("SPEAKER_04", "natsu", "named", [NATSU] * 9, [(0, 30)]),
                "SPEAKER_05": ("SPEAKER_05", None, "remainder", [near], [(30, 60)]),
            },
        )
        cfg = Config(registry=NAMES_CFG.registry, raw={"characters": {"matcher": matcher}})
        CharactersStage().run(ep, cfg)
        got[matcher] = by_cluster(ep)["SPEAKER_05"].status
    assert got == {"centroid": "new", "lda": "new"}


def test_remainder_may_match_a_character_names_created_this_episode(tmp_path):
    ep = make_ep(tmp_path)
    near = unit([0.95, 0.3, 0])
    write_diarize(
        ep,
        {"SPEAKER_08": ([HAPPY], 30.0), "SPEAKER_09": ([near], 30.0)},
        [(0, 30, "SPEAKER_08"), (30, 60, "SPEAKER_09")],
    )
    write_names(
        ep,
        {
            "SPEAKER_08~happy": ("SPEAKER_08", "happy", "named", [HAPPY], [(0, 30)]),
            "SPEAKER_09": ("SPEAKER_09", None, "remainder", [near], [(30, 60)]),
        },
    )
    CharactersStage().run(ep, NAMES_CFG)
    by = by_cluster(ep)
    assert by["SPEAKER_09"].character_id == by["SPEAKER_08~happy"].character_id
    assert (by["SPEAKER_09"].status, by["SPEAKER_09"].source) == ("matched", "voice")


def test_without_names_assignments_match_the_pre_names_code_byte_for_byte(tmp_path):
    """tests/fixtures/characters_golden.json is what debd501 (before names) wrote for the
    scenario in characters_golden.py beside it."""
    import importlib.util
    import json
    from pathlib import Path

    fixtures = Path(__file__).parent.parent / "fixtures"
    spec = importlib.util.spec_from_file_location(
        "characters_golden", fixtures / "characters_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    golden = json.loads((fixtures / "characters_golden.json").read_text())
    got = module.scenario(tmp_path)
    assert set(got) == set(golden)
    for step, text in golden.items():
        assert got[step] == text, step
