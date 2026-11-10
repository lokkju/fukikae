"""`fukikae registry resolve`: the joint matcher over a season's episodes at once (Ruling R12),
rewriting the registry and the episodes' assignments in one transaction."""

from datetime import UTC, datetime
from typing import Any

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


def test_existing_actors_never_merge_even_with_episode_cannot_link_off(tmp_path):
    """The live-action configuration: episode_cannot_link off. Anchors still stay apart
    (always_cannot_link), though their voices are close enough to merge at 0.5."""
    ws = make_ws(tmp_path, {"S01E05": {"S0": (noisy(unit([1, 0.45, 0]), seed=5), 30.0)}})
    with Registry(ws.registry_path) as reg:
        for center in (unit([1, 0.5, 0]), unit([1, -0.0, 0])):
            aid, _ = reg.create_actor_with_character(center)
            reg.add_exemplars(aid, "S01E09", "X", np.array(noisy(center, seed=aid)), 30.0)
            reg.recompute_centroid(aid)
    r = resolve(ws, "S01E05", "--threshold", "0.5", "--matcher-option", "episode_cannot_link=false")
    assert r.exit_code == 0, r.output
    e5, _ = by_cluster(ws, "S01E05")
    assert e5["S0"].status == "matched" and e5["S0"].actor_id == 1
    with Registry(ws.registry_path) as reg:
        assert [a.id for a in reg.actors()] == [1, 2]
        assert reg.episode_counts(2) == {"S01E09": 6}


def test_resolve_refuses_an_anchor_of_another_dimension(tmp_path):
    ws = make_ws(tmp_path, {"S01E05": {"S0": (noisy(A, seed=5), 30.0)}})
    with Registry(ws.registry_path) as reg:
        aid, _ = reg.create_actor_with_character(np.ones(5, dtype=np.float32))
        reg.add_exemplars(aid, "S01E09", "X", np.ones((2, 5), dtype=np.float32), 30.0)
    r = resolve(ws, "S01E05")
    assert r.exit_code == 1
    assert f"actor {aid}" in r.output and "embedder changed" in r.output


def test_resolve_refuses_episodes_of_different_dimensions(tmp_path):
    ws = make_ws(tmp_path, {"S01E01": SEASON["S01E01"]})
    ep = episode(ws, "S01E02")
    add_episode(ws, "S01E02", {"S0": (noisy(A, seed=3), 30.0)})
    # rewrite E02's rows as 4-dimensional (another embedder)
    arrays: dict[str, Any] = {"labels": np.array(["S0"], dtype=str)}
    arrays["S0/embeddings"] = np.ones((3, 4), dtype=np.float32) / 2
    arrays["S0/spans"] = np.zeros((3, 2))
    arrays["S0/clean_seconds"] = np.array(30.0)
    np.savez(ep.work / "04-diarize" / "clusters.npz", **arrays)
    r = resolve(ws, "S01E01,S01E02")
    assert r.exit_code == 1
    assert "S01E02" in r.output and "embedder changed" in r.output
    with Registry(ws.registry_path) as reg:
        assert reg.actors() == []


def _capture_resolve(monkeypatch):
    """Replace only resolve_episodes; record what the CLI passes it."""
    from fukikae.registry import resolve as resolve_mod

    seen: dict[str, Any] = {}

    def fake(_ws, eps, threshold, min_new_seconds=None, options=None):  # noqa: ARG001
        seen.update(episodes=[e.id for e in eps], threshold=threshold, options=options)
        return resolve_mod.ResolveSummary([e.id for e in eps], 0, [], 0, [])

    monkeypatch.setattr(resolve_mod, "resolve_episodes", fake)
    return seen


def _set_language(ws: Workspace, eid: str, lang: str) -> None:
    from fukikae.artifacts import write_json
    from fukikae.stages.language import LanguageResult

    path = ws.work_dir(eid) / "02-language" / "language.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, LanguageResult(language=lang, confidence=0.9, source="detected"))


def test_resolve_without_flags_uses_the_profile_and_flags_override_it(tmp_path, monkeypatch):
    """No typer default may beat the profile's [resolve] values (Ruling R17)."""
    ws = make_ws(tmp_path)
    (ws.root / "fukikae.toml").write_text('profile = "anime"\n')
    seen = _capture_resolve(monkeypatch)
    args = ["registry", "resolve", str(ws.root), "--episodes", "S01E01,S01E02"]
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    assert seen == {
        "episodes": ["S01E01", "S01E02"],
        "threshold": 0.6,
        "options": {"episode_cannot_link": True},
    }
    r = runner.invoke(
        app, [*args, "--threshold", "0.75", "--matcher-option", "episode_cannot_link=false"]
    )
    assert r.exit_code == 0, r.output
    assert seen["threshold"] == 0.75 and seen["options"] == {"episode_cannot_link": False}


def test_resolve_reads_the_language_override_of_the_detected_language(tmp_path, monkeypatch):
    ws = make_ws(tmp_path)
    (ws.root / "fukikae.toml").write_text(
        'profile = "anime"\nsource_language = "en"\n[resolve.lang.ja]\nthreshold = 0.65\n'
    )
    for eid in ("S01E01", "S01E02"):
        _set_language(ws, eid, "ja")
    seen = _capture_resolve(monkeypatch)
    r = runner.invoke(app, ["registry", "resolve", str(ws.root), "--episodes", "S01E01,S01E02"])
    assert r.exit_code == 0, r.output
    assert seen["threshold"] == 0.65


def test_resolve_refuses_episodes_detected_in_different_languages(tmp_path, monkeypatch):
    ws = make_ws(tmp_path)
    _set_language(ws, "S01E01", "ja")
    _set_language(ws, "S01E02", "en")
    seen = _capture_resolve(monkeypatch)
    r = runner.invoke(app, ["registry", "resolve", str(ws.root), "--episodes", "S01E01,S01E02"])
    assert r.exit_code == 1
    assert "S01E01: ja" in r.output and "S01E02: en" in r.output
    assert seen == {}


def _snapshot(ws: Workspace, eids):
    with Registry(ws.registry_path) as reg:
        actors = [(a.id, a.name, a.locked, np.asarray(a.centroid).tolist()) for a in reg.actors()]
        rows = {aid: r.tolist() for aid, r in reg.exemplar_rows().items()}
        chars = reg.characters()
    eps = {
        e: (
            load_manifest(episode(ws, e)).stages.get("characters"),
            (ws.work_dir(e) / "05-characters" / "assignments.json").read_text(),
        )
        for e in eids
    }
    return actors, rows, chars, eps


def test_refused_resolve_leaves_characters_records_and_registry_intact(tmp_path):
    ws = make_ws(tmp_path)
    assert resolve(ws, "S01E01,S01E02").exit_code == 0
    with Registry(ws.registry_path) as reg:  # an actor from another embedder
        aid, _ = reg.create_actor_with_character(np.ones(5, dtype=np.float32))
        reg.add_exemplars(aid, "S01E09", "X", np.ones((2, 5), dtype=np.float32), 30.0)
    before = _snapshot(ws, ["S01E01", "S01E02"])
    assert all(rec is not None and rec.status == "done" for rec, _ in before[3].values())
    r = resolve(ws, "S01E01,S01E02")
    assert r.exit_code == 1 and "embedder changed" in r.output
    assert _snapshot(ws, ["S01E01", "S01E02"]) == before


def test_min_new_seconds_option_says_where_it_is_set(tmp_path):
    ws = make_ws(tmp_path)
    r = resolve(ws, "S01E01,S01E02", "--matcher-option", "min_new_seconds=5")
    assert r.exit_code == 1
    assert "[registry] min_new_seconds" in r.output and "--min-new-seconds" not in r.output


# --- names (Task 7): name decisions survive resolve -------------------------------------


def _named_season(tmp_path):
    """Two names-used episodes run through the characters stage: wendy (near A) and natsu
    (near B) are keys in both; S01E02's SPEAKER_04 is unnamed speech near A."""
    from tests.stages.test_characters import NAMES_CFG, write_names

    ws = make_ws(tmp_path, season={})
    e1 = add_episode(
        ws, "S01E01", {"SPEAKER_00": (noisy(A, seed=1) + noisy(B, seed=2), 60.0),
                       "SPEAKER_01": (noisy(C, seed=3), 30.0)},
    )  # fmt: skip
    write_names(
        e1,
        {
            "SPEAKER_00~wendy": ("SPEAKER_00", "wendy", "named", noisy(A, seed=1), [(0, 30)]),
            "SPEAKER_00~natsu": ("SPEAKER_00", "natsu", "named", noisy(B, seed=2), [(30, 60)]),
            "SPEAKER_01": ("SPEAKER_01", None, "remainder", noisy(C, seed=3), [(60, 90)]),
        },
    )
    e2 = add_episode(
        ws, "S01E02", {"SPEAKER_03": (noisy(A, seed=4), 30.0),
                       "SPEAKER_04": (noisy(A, seed=5), 30.0),
                       "SPEAKER_05": (noisy(B, seed=6), 30.0)},
    )  # fmt: skip
    write_names(
        e2,
        {
            "SPEAKER_03~wendy": ("SPEAKER_03", "wendy", "named", noisy(A, seed=4), [(0, 30)]),
            "SPEAKER_04": ("SPEAKER_04", None, "remainder", noisy(A, seed=5), [(30, 60)]),
            "SPEAKER_05~natsu": ("SPEAKER_05", "natsu", "named", noisy(B, seed=6), [(60, 90)]),
        },
    )
    for ep in (e1, e2):
        CharactersStage().run(ep, NAMES_CFG)
    return ws, e1, e2


def _named(ws, eid):
    got, _ = by_cluster(ws, eid)
    return {lb: (a.character_id, a.name, a.source) for lb, a in got.items() if a.name}


@pytest.mark.parametrize("threshold", [0.6, 0.0])
def test_named_labels_keep_their_character_through_resolve(tmp_path, threshold):
    from fukikae.registry.resolve import resolve_episodes

    ws, e1, e2 = _named_season(tmp_path)
    before = {e: _named(ws, e) for e in ("S01E01", "S01E02")}
    with Registry(ws.registry_path) as reg:
        wendy, natsu = reg.character_for_key("wendy"), reg.character_for_key("natsu")
    assert wendy is not None and natsu is not None and wendy.id != natsu.id
    resolve_episodes(ws, [e1, e2], threshold=threshold)
    assert {e: _named(ws, e) for e in ("S01E01", "S01E02")} == before
    assert before["S01E02"]["SPEAKER_03~wendy"][0] == wendy.id
    assert before["S01E02"]["SPEAKER_05~natsu"][0] == natsu.id
    with Registry(ws.registry_path) as reg:  # both keys still answer, to different characters
        w2, n2 = reg.character_for_key("wendy"), reg.character_for_key("natsu")
        assert w2 is not None and n2 is not None and (w2.id, n2.id) == (wendy.id, natsu.id)
        assert reg.exemplar_count(w2.actor_id) >= 12 and reg.exemplar_count(n2.actor_id) >= 12


def test_unnamed_speech_joins_the_anchor_its_names_seeded(tmp_path):
    from fukikae.registry.resolve import resolve_episodes

    ws, e1, e2 = _named_season(tmp_path)
    resolve_episodes(ws, [e1, e2], threshold=0.6)
    got, _ = by_cluster(ws, "S01E02")
    with Registry(ws.registry_path) as reg:
        wendy = reg.character_for_key("wendy")
    assert wendy is not None
    rem = got["SPEAKER_04"]
    assert (rem.status, rem.character_id, rem.source, rem.name) == (
        "matched", wendy.id, "voice", None
    )  # fmt: skip
