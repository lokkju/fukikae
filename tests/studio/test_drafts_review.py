"""Task 18 review fixes: the strict own-speech rule at draft time and at accept, the pipeline
thin card's provenance, duplicate and over-long references, spoken cues only for a character
with no turns, stale episodes, names-used episodes, parameter and query parsing, carried
origin-less references and the candidate cache."""

# ruff: noqa: F811 (pytest fixtures imported from the modules that define them)

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from fukikae.artifacts import read_json, write_json
from fukikae.cards.store import CardStore
from fukikae.models import Assignments, ClusterAssignment, Delivery, DubCue, DubCues, Turn
from fukikae.registry.store import Registry
from fukikae.studio.model import RefSpan
from fukikae.workspace import Workspace
from tests.studio.test_drafts import _cfg, api, clone_ws  # noqa: F401 (fixtures)
from tests.studio.test_season import E1, E2, _clone, ws  # noqa: F401 (fixture)
from tests.studio.test_server import Client, studio  # noqa: F401 (fixture)


def _asg_path(w: Workspace, ep: str):
    return w.work_dir(ep) / "05-characters" / "assignments.json"


def _add_turn(w: Workspace, ep: str, cluster: str, a: float, b: float) -> None:
    path = _asg_path(w, ep)
    asg = read_json(path, Assignments)
    turns: list[Turn] = sorted([*(asg.turns or []), Turn(start=a, end=b, speaker=cluster)],
                   key=lambda t: float(t.start))  # fmt: skip
    write_json(path, asg.model_copy(update={"turns": turns}))


def _span(ep: str, a: float, b: float) -> RefSpan:
    return RefSpan(episode=ep, start=a, end=b)


def _accept_check(w: Workspace, cid: int, card_id: str):
    """accept's own-speech rule (`apply._own_speech`) on a stored card."""
    from fukikae.studio.apply import _own_speech

    with Registry(w.registry_path) as reg:
        return _own_speech(w, reg, cid, card_id, CardStore(w.root).get(card_id))


def _assign(w: Workspace, cid: int, card: str, by: str = "auto") -> None:
    with Registry(w.registry_path) as reg:
        reg.assign_card(cid, "en", card, by=by, locked=by != "auto")


def _open(w: Workspace, cid: int = 1):
    from fukikae.studio.drafts import open_draft, save_draft

    d = open_draft(w, _cfg(w), cid, "en")
    save_draft(w, d)
    return d


def _add(w: Workspace, d, refs, params=None, **kw):
    from fukikae.studio.drafts import add_version

    return add_version(w, _cfg(w), d, refs, "clonefake", params or {}, **kw)


# --- 1. the strict rule at draft time, and the pipeline thin card's provenance -----------


@pytest.fixture
def overlap_ws(clone_ws: Workspace) -> Workspace:
    """Natsu (S1, character 2) speaks 12-13 s inside Lucy's 10-14 s turn in S01E01."""
    _add_turn(clone_ws, E1, "S1", 12.0, 13.0)
    return clone_ws


def test_a_short_overlapping_selection_is_refused_at_draft_naming_the_other_speaker(overlap_ws):
    from fukikae.studio.drafts import DraftError

    d = _open(overlap_ws)
    with pytest.raises(DraftError, match=r"S01E01 11\.00-13\.00.*character 2"):
        _add(overlap_ws, d, [_span(E1, 11.0, 13.0)])  # 2.0 s: thin by its seconds


def test_a_short_overlapping_selection_is_refused_at_accept(overlap_ws):
    from fukikae.studio.apply import DecisionRefused

    card = _clone(overlap_ws, CardStore(overlap_ws.root), E1, [(11.0, 13.0)], tier="thin")
    with pytest.raises(DecisionRefused, match="overlaps another speaker"):
        _accept_check(overlap_ws, 1, card)
    _assign(overlap_ws, 1, card, by="studio")  # an operator's thin card is no pipeline card
    with pytest.raises(DecisionRefused, match="overlaps another speaker"):
        _accept_check(overlap_ws, 1, card)


def test_a_clean_and_overlapping_mix_over_three_seconds_is_refused_at_draft(overlap_ws):
    from fukikae.studio.drafts import DraftError

    d = _open(overlap_ws)
    with pytest.raises(DraftError, match=r"S01E01 11\.00-13\.00"):
        _add(overlap_ws, d, [_span(E1, 2.0, 5.0), _span(E1, 11.0, 13.0)])


def test_the_pipeline_thin_cards_exact_set_is_accepted_with_the_warning(overlap_ws):
    store = CardStore(overlap_ws.root)
    thin = _clone(overlap_ws, store, E1, [(10.0, 14.0)], tier="thin")
    _assign(overlap_ws, 1, thin)
    d = _open(overlap_ws)
    (v1,) = d.versions
    assert d.base == "card" and v1.thin_origin == thin and v1.tier == "thin"
    line, overlapping = _accept_check(overlap_ws, 1, v1.card_id)
    assert overlapping == ["S01E01 10.00-14.00"] and "overlaps another speaker" in line
    d, v2 = _add(overlap_ws, d, [_span(E1, 10.0, 14.0)], {"seed": 7})
    assert v2.thin_origin == thin and v2.tier == "thin"
    card = store.get(v2.card_id)
    assert card.meta["thin_origin"] == thin and card.voice.seed == 7
    line, overlapping = _accept_check(overlap_ws, 1, v2.card_id)
    assert overlapping == ["S01E01 10.00-14.00"]


def test_dropping_a_reference_from_the_pipeline_thin_card_makes_the_rest_strict(overlap_ws):
    from fukikae.studio.apply import DecisionRefused
    from fukikae.studio.drafts import DraftError

    store = CardStore(overlap_ws.root)
    thin = _clone(overlap_ws, store, E1, [(2.0, 5.0), (10.0, 14.0)], tier="thin")
    _assign(overlap_ws, 1, thin)
    d = _open(overlap_ws)
    assert d.versions[0].thin_origin == thin
    with pytest.raises(DraftError, match=r"S01E01 10\.00-14\.00.*character 2"):
        _add(overlap_ws, d, [_span(E1, 10.0, 14.0)])
    d, v = _add(overlap_ws, d, [_span(E1, 2.0, 5.0)])  # clean: allowed, no provenance
    assert v.thin_origin is None
    _accept_check(overlap_ws, 1, v.card_id)
    # a card claiming the provenance with another set is held to the clean rule at accept
    fake = store.put(replace(store.get(_clone(overlap_ws, store, E1, [(10.0, 14.0)],
                                              tier="thin")),
                             meta={"reference_tier": "thin", "thin_origin": thin}))  # fmt: skip
    with pytest.raises(DecisionRefused, match="overlaps another speaker"):
        _accept_check(overlap_ws, 1, fake)


def test_a_thin_tier_version_of_the_operators_own_is_not_exempt(overlap_ws):
    """A version made thin by its seconds gives no later version the exemption."""
    from fukikae.studio.drafts import DraftError

    d = _open(overlap_ws)
    d, v = _add(overlap_ws, d, [_span(E2, 70.0, 72.0)])  # 2 s, clean
    assert v.tier == "thin" and v.thin_origin is None
    with pytest.raises(DraftError):
        _add(overlap_ws, d, [_span(E1, 11.0, 13.0)])


# --- 2. duplicates, nesting and overlaps within a version; the longest reference ---------


@pytest.mark.parametrize(
    "refs",
    [
        [(E1, 10.0, 14.0), (E1, 10.0, 14.0)],
        [(E1, 10.0, 14.0), (E1, 11.0, 12.0)],
        [(E1, 60.0, 62.0), (E1, 61.0, 63.0)],
    ],
)
def test_duplicate_nested_or_overlapping_references_are_refused(clone_ws, refs):
    from fukikae.studio.drafts import DraftError

    d = _open(clone_ws)
    with pytest.raises(DraftError, match="more than once"):
        _add(clone_ws, d, [_span(*r) for r in refs])


def test_the_same_times_in_two_episodes_are_not_duplicates(clone_ws):
    d = _open(clone_ws)
    _, v = _add(clone_ws, d, [_span(E1, 2.0, 5.0), _span(E2, 2.0, 5.0)])
    assert len(v.references) == 2


def test_a_reference_over_twelve_seconds_is_refused(clone_ws):
    from fukikae.studio.drafts import DraftError

    d = _open(clone_ws)
    with pytest.raises(DraftError, match="12 s"):
        _add(clone_ws, d, [_span(E1, 0.0, 12.5)])


def test_duplicates_are_refused_by_the_api(api):
    body = {"references": [{"episode": E1, "start": 10.0, "end": 14.0}] * 2,
            "backend": "clonefake", "params": {}}  # fmt: skip
    status, got = api.json("POST", "/api/drafts/1/en/versions", body)
    assert status == 422 and "more than once" in got["message"]


# --- 3. spoken cues count only for a character with no turns in the episode --------------


def _add_cue(w: Workspace, ep: str, cid: int, a: float, b: float) -> None:
    path = w.work_dir(ep) / "06-translate" / "cues.en.json"
    doc = read_json(path, DubCues)
    cue = DubCue(
        id=f"x{cid}-{a:g}",
        start=a,
        end=b,
        character_id=cid,
        extra=None,
        speaker="x",
        source_text="src",
        text="line",
        budget=40,
        delivery=Delivery(reference=(a, b)),
    )
    write_json(path, doc.model_copy(update={"cues": [*doc.cues, cue]}))


def test_a_cue_span_is_not_own_speech_for_a_character_with_turns(clone_ws):
    from fukikae.studio.apply import DecisionRefused
    from fukikae.studio.drafts import DraftError, check_own_speech

    _add_cue(clone_ws, E1, 1, 90.0, 93.0)  # Lucy holds turns in S01E01
    with pytest.raises(DraftError, match="own speech"):
        check_own_speech(clone_ws, 1, [_span(E1, 90.0, 93.0)])
    card = _clone(clone_ws, CardStore(clone_ws.root), E1, [(90.0, 93.0)])
    with pytest.raises(DecisionRefused, match="not inside"):
        _accept_check(clone_ws, 1, card)


def test_a_cue_span_is_own_speech_for_a_character_without_turns_there(clone_ws):
    from fukikae.studio.drafts import check_own_speech

    _add_cue(clone_ws, E1, 5, 90.0, 93.0)  # character 5 speaks only in S01E02
    check_own_speech(clone_ws, 5, [_span(E1, 90.0, 93.0)])
    card = _clone(clone_ws, CardStore(clone_ws.root), E1, [(90.0, 93.0)])
    line, _ = _accept_check(clone_ws, 5, card)
    assert "1 of 1" in line


def test_cue_candidates_leave_out_tagged_singing(clone_ws):
    from fukikae.models import Tags, TagSpan
    from fukikae.studio.drafts import candidates

    path = _asg_path(clone_ws, E1)
    asg = read_json(path, Assignments)
    write_json(
        path,
        asg.model_copy(
            update={
                "clusters": [
                    c.model_copy(update={"status": "extra"}) if c.character_id == 3 else c
                    for c in asg.clusters
                ]
            }
        ),
    )  # fmt: skip  (character 3 now holds no turn anywhere)
    tags = clone_ws.work_dir(E1) / "04b-tag" / "tags.json"
    tags.parent.mkdir(parents=True, exist_ok=True)
    write_json(tags, Tags(backend="fake", events=[TagSpan(start=80.0, end=84.0, label="singing")],
                          emotions=[]))  # fmt: skip
    got = candidates(clone_ws, _cfg(clone_ws), 3)
    assert got and all(c.source == "cue" for c in got)
    sung = next(c for c in got if c.start == 80.0)
    assert sung.rank is None and sung.reasons == ["excluded"]
    assert all(c.rank is not None for c in got if c.start != 80.0)


# --- 4. stale episodes -------------------------------------------------------------------


def test_an_episode_whose_characters_stage_is_stale_is_not_ready(clone_ws):
    from fukikae import job
    from fukikae.registry.invalidate import episode
    from fukikae.studio.apply import DecisionRefused
    from fukikae.studio.drafts import DraftError, candidates

    d = _open(clone_ws)
    job.mark_stale(episode(clone_ws, E2), "characters")
    assert not [c for c in candidates(clone_ws, _cfg(clone_ws), 1) if c.episode == E2]
    with pytest.raises(DraftError, match=f"fukikae run <ws> {E2} --to characters"):
        _add(clone_ws, d, [_span(E2, 2.0, 6.0)])
    card = _clone(clone_ws, CardStore(clone_ws.root), E2, [(2.0, 6.0)])
    with pytest.raises(DecisionRefused, match=f"fukikae run <ws> {E2} --to characters"):
        _accept_check(clone_ws, 1, card)


def test_an_episode_older_than_an_applied_decision_on_the_character_is_not_ready(clone_ws):
    from fukikae.studio.apply import decisions_log
    from fukikae.studio.drafts import DraftError, candidates

    rid = uuid.uuid4().hex
    rec = {
        "id": rid, "time": (datetime.now(UTC) + timedelta(hours=1)).isoformat(timespec="seconds"),
        "workspace": str(clone_ws.root.resolve()),
        "decision": {"op": "kind", "character": 1, "kind": "character"},
        "invalidations": [[E2, "characters"]],
    }  # fmt: skip
    log = decisions_log(clone_ws)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with Registry(clone_ws.registry_path) as reg:
        reg.mark_decision(rid, "kind")
    assert not [c for c in candidates(clone_ws, _cfg(clone_ws), 1) if c.episode == E2]
    assert [c for c in candidates(clone_ws, _cfg(clone_ws), 2) if c.episode == E2]
    d = _open(clone_ws)
    with pytest.raises(DraftError, match="--to characters"):
        _add(clone_ws, d, [_span(E2, 2.0, 6.0)])


def _names_used(w: Workspace, ep: str, *, refined_turns: bool = True) -> None:
    from fukikae.models import Diarization
    from fukikae.names.inputs import names_path
    from fukikae.names.models import NamesResult

    work = w.work_dir(ep)
    names_path(work).parent.mkdir(parents=True, exist_ok=True)
    write_json(names_path(work), NamesResult(
        episode=ep, used=True, verdict="full", reason=None, source=None, offset_s=0.0,
        coverage={}, lines=[], hidden=[], pieces=[], labels={}, unvoiced=[], excluded=[],
        dropped_row_seconds=0.0, suggestions=[],
    ))  # fmt: skip
    if refined_turns:
        asg = read_json(_asg_path(w, ep), Assignments)
        write_json(names_path(work).parent / "turns.json",
                   Diarization(backend="fake", turns=list(asg.turns or [])))  # fmt: skip


def test_a_names_used_episode_older_than_its_names_is_refused_like_candidates(clone_ws):
    from fukikae.studio.drafts import DraftError, candidates, check_own_speech

    _names_used(clone_ws, E2, refined_turns=False)  # its assignments predate its names
    assert not [c for c in candidates(clone_ws, _cfg(clone_ws), 1) if c.episode == E2]
    with pytest.raises(DraftError, match="--to characters"):
        check_own_speech(clone_ws, 1, [_span(E2, 2.0, 6.0)])


def test_a_names_used_episode_keeps_every_unconfirmed_turn(clone_ws):
    """An acoustic candidate joining a confirmed turn and an unnamed one is split, not
    dropped: the unnamed turn is still offered."""
    from fukikae.studio.drafts import candidates
    from tests.fixtures.speechlike import speech

    w = clone_ws
    path = _asg_path(w, E1)
    asg = read_json(path, Assignments)
    clusters = [c.model_copy(update={"source": "name"}) if c.cluster == "S0" else c
                for c in asg.clusters]  # fmt: skip
    clusters.append(ClusterAssignment(cluster="S6", actor_id=1, character_id=1, score=0.9,
                                      clean_seconds=1.7, status="matched"))  # fmt: skip
    turns: list[Turn] = sorted([*(asg.turns or []), Turn(start=14.3, end=16.0, speaker="S6")],
                   key=lambda t: float(t.start))  # fmt: skip
    write_json(path, asg.model_copy(update={"clusters": clusters, "turns": turns}))
    import wave

    vpath = w.work_dir(E1) / "01-separate" / "vocals.wav"
    with wave.open(str(vpath), "rb") as f:
        rate, data = f.getframerate(), np.frombuffer(f.readframes(f.getnframes()), "<i2")
    x = data.astype(np.float32) / 32767
    seg = speech(230, 1.7, seed=143, voiced_share=0.7)
    x[int(14.3 * rate) : int(14.3 * rate) + len(seg)] = seg
    with wave.open(str(vpath), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    _names_used(w, E1)
    got = [c for c in candidates(w, _cfg(w), 1) if c.episode == E1]
    assert any(c.start - 1e-6 <= 14.3 and c.end + 1e-6 >= 16.0 for c in got)
    assert [c for c in got if c.confirmed]
    for c in got:  # no candidate offered twice over the same audio across the two tiers
        assert not any(o is not c and o.start < c.end and c.start < o.end and
                       o.confirmed != c.confirmed for o in got), (c.start, c.end)  # fmt: skip


# --- 5. parameters, carried references, sidecars -----------------------------------------


def test_check_params_refuses_a_huge_int_without_overflow():
    from fukikae.studio.drafts import DraftError, check_params
    from tests.cards.test_select import CloneSynth

    with pytest.raises(DraftError, match="emotion_strength"):
        check_params(CloneSynth, {"emotion_strength": 10**400})


def _files(w: Workspace) -> set[str]:
    return {str(p) for p in (w.root / ".fukikae" / "cards").rglob("*") if p.is_file()}


@pytest.mark.parametrize("params", [{"emotion_strength": 2.0}, {"seed": 2**40}])
def test_a_refused_parameter_cuts_no_sidecar(clone_ws, params):
    from fukikae.studio.drafts import DraftError

    d = _open(clone_ws)
    before = _files(clone_ws)
    with pytest.raises(DraftError):
        _add(clone_ws, d, [_span(E1, 60.0, 63.0)], params)
    assert _files(clone_ws) == before


def test_origin_less_references_are_carried_forward_unless_removed(clone_ws):
    store = CardStore(clone_ws.root)
    cid = _clone(clone_ws, store, E1, [(10.0, 14.0), (60.0, 63.0)], tier="floor")
    card = store.get(cid)
    bare = replace(card.voice.references[1], origin=None)
    cid = store.put(
        replace(card, voice=replace(card.voice, references=(card.voice.references[0], bare)))
    )
    _assign(clone_ws, 1, cid)
    d = _open(clone_ws)
    assert d.versions[0].unlisted == 1
    d, v = _add(clone_ws, d, [_span(E1, 2.0, 5.0)])
    assert v.unlisted == 1
    refs = store.get(v.card_id).voice.references
    assert len(refs) == 2 and refs[-1].clip == bare.clip and refs[-1].origin is None
    d, v = _add(clone_ws, d, [_span(E1, 2.0, 5.0)], keep_unlisted=False)
    assert v.unlisted == 0 and len(store.get(v.card_id).voice.references) == 1


# --- the API: queries, unreadable drafts, HEAD, the clip set, the cache, the locks -------


@pytest.mark.parametrize("q", ["limit=%C2%B2", "offset=%D9%A3", "limit=" + "1" * 30, "limit=-1",
                               "offset=1e3", "limit="])  # fmt: skip
def test_candidate_queries_take_only_ascii_digits(api, q):
    assert api.json("GET", f"/api/characters/1/candidates?{q}")[0] == 400


@pytest.mark.parametrize("exc", [OSError("disk gone"), "version"])
def test_an_unreadable_draft_is_a_409(api, monkeypatch, exc):
    from fukikae.artifacts import ArtifactVersionError
    from fukikae.studio import server as server_mod

    err = ArtifactVersionError("written by a newer schema") if exc == "version" else exc

    def boom(*_a, **_k):
        raise err

    monkeypatch.setattr(server_mod, "load_draft", boom)
    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 409 and body["error"] == "draft-unreadable"


def test_head_on_a_draft_saves_nothing(api, clone_ws):
    status, _, _ = api.request("HEAD", "/api/drafts/1/en")
    assert status == 404
    assert not (clone_ws.root / "studio" / "drafts" / "1.en.json").exists()
    assert api.json("GET", "/api/drafts/1/en")[0] == 200
    assert api.request("HEAD", "/api/drafts/1/en")[0] == 200


def test_the_cut_clip_set_is_bounded(api, monkeypatch):
    from fukikae.studio import server as server_mod

    monkeypatch.setattr(server_mod, "MAX_CUT", 3)
    s = api.server
    names = []
    for a in (2.0, 2.5, 3.0, 3.5, 4.0):
        n = s.offer(E1, a, a + 0.5)
        assert n is not None
        s.ensure_clip(n)
        names.append(n)
    assert len(s._cut) == 3 and list(s._cut) == names[-3:]


def test_candidates_are_cached_until_a_gender_decision(api, monkeypatch, clone_ws):
    from fukikae.studio import server as server_mod

    calls: list[int] = []
    real = server_mod.candidates

    def counting(ws, cfg, cid):
        calls.append(cid)
        return real(ws, cfg, cid)

    monkeypatch.setattr(server_mod, "candidates", counting)
    assert api.json("GET", "/api/characters/1/candidates")[0] == 200
    assert api.json("GET", "/api/characters/1/candidates")[0] == 200
    assert calls == [1]
    with Registry(clone_ws.registry_path) as reg:
        reg.set_character_gender(1, "m")
    assert api.json("GET", "/api/characters/1/candidates")[0] == 200
    assert calls == [1, 1]


def test_a_decision_proceeds_while_a_drafts_request_holds_the_drafts_lock(api):
    from tests.studio.test_server import _name, _season

    doc = _season(api)
    s = api.server
    got: dict[str, int] = {}
    with s._drafts_lock:
        t = threading.Thread(
            target=lambda: got.update(
                status=api.json("POST", "/api/decisions", _name(doc, 3, "Erza"))[0]
            )
        )
        t.start()
        t.join(timeout=20)
        assert not t.is_alive() and got["status"] == 200
