"""Reference candidates and voice drafts (Task 18): every turn of a character with its gate
verdicts, drafts opened from the current card, an accepted card of another language or the
automatic proposal, draft cards built as the automatic clone builds them, and the API."""

from __future__ import annotations

import stat

import pytest

from fukikae.backends import loader
from fukikae.cards.store import CardStore
from fukikae.config import load_config
from fukikae.registry.store import Registry
from fukikae.workspace import Workspace
from tests.cards.test_select import CloneSynth
from tests.studio.test_season import (
    E1,
    E2,
    _clone,
    ws,  # noqa: F401 (the season fixture workspace)
)
from tests.studio.test_server import Client, studio  # noqa: F401 (fixture)


@pytest.fixture
def clone_ws(ws: Workspace, monkeypatch) -> Workspace:  # noqa: F811
    """The season workspace with ``[tts] clone_backend`` set to the fake clone backend."""
    (ws.root / "fukikae.toml").write_text(
        'source_language = "ja"\ntarget_language = "en"\n\n[tts]\nclone_backend = "clonefake"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        loader, "backend_class", lambda _kind, name: {"clonefake": CloneSynth}[name]
    )
    monkeypatch.setattr(loader, "available", lambda _kind: {"clonefake": CloneSynth})
    monkeypatch.setattr(loader, "engine_problem", lambda *_a: None)
    return ws


def _cfg(w: Workspace):
    return load_config(w.root, user_config=None)


# --- candidates ----------------------------------------------------------------------------


def _covers(c, start: float, end: float) -> bool:
    return c.start - 1e-6 <= start and end <= c.end + 1e-6


def test_candidates_cover_every_turn_with_reasons_for_refused_ones(clone_ws):
    from fukikae.studio.drafts import candidates

    got = candidates(clone_ws, _cfg(clone_ws), 1)
    turns = {
        E1: [(2, 5), (10, 14), (60, 63), (70, 76)],
        E2: [(2, 6), (8, 11), (60, 67), (70, 72)],
    }
    for ep, spans in turns.items():
        for a, b in spans:
            assert any(c.episode == ep and _covers(c, a, b) for c in got), (ep, a, b)
    short = next(c for c in got if c.episode == E2 and c.start == 70)
    assert short.rank is None and "length" in short.reasons
    assert short.verdicts["length"] is False
    assert short.median is not None and short.gender == "f"
    assert short.clip.endswith(".wav") and len(short.clip) == 68
    eligible = [c for c in got if c.rank is not None]
    assert eligible and all(not c.reasons for c in eligible)
    assert all(all(c.verdicts[k] for k in c.verdicts if k != "scene") for c in eligible)


def test_eligible_candidates_keep_cd5_rank(clone_ws):
    from fukikae.cards.auto import character_candidates
    from fukikae.studio.drafts import candidates

    got = candidates(clone_ws, _cfg(clone_ws), 1)
    ranked = [c for c in got if c.rank is not None]
    assert [c.rank for c in ranked] == list(range(len(ranked)))
    assert got[: len(ranked)] == ranked  # eligible first, in rank order
    auto = character_candidates(clone_ws, [E1, E2], 1)
    for ep in (E1, E2):
        mine = [(c.start, c.end) for c in ranked if c.episode == ep]
        assert mine == [(p.start, p.end) for p in auto[ep]]


# --- opening a draft -----------------------------------------------------------------------


def _row_digest(w, cid, lang):
    from fukikae.studio.apply import row_digest

    with Registry(w.registry_path) as reg:
        return row_digest(reg, cid, lang)


def test_draft_opens_from_the_current_card(clone_ws):
    from fukikae.studio.drafts import open_draft

    store = CardStore(clone_ws.root)
    card = _clone(clone_ws, store, E1, [(10.0, 14.0), (70.0, 76.0)], tier="floor")
    with Registry(clone_ws.registry_path) as reg:
        reg.assign_card(1, "en", card, by="auto", locked=False)
    d = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    assert (d.character_id, d.language, d.base, d.derived_from) == (1, "en", "card", None)
    (v,) = d.versions
    assert v.n == 1 and v.card_id == card and v.backend == "clonefake"
    assert [(r.episode, r.start, r.end) for r in v.references] == [
        (E1, 10.0, 14.0), (E1, 70.0, 76.0),
    ]  # fmt: skip
    assert v.params["seed"] == 1234 and v.overrides == [] and v.auditions == []
    assert v.digest == _row_digest(clone_ws, 1, "en") and v.tier == "floor"


def test_draft_for_a_new_language_derives_from_an_accepted_card(clone_ws):
    from fukikae.studio.drafts import open_draft

    store = CardStore(clone_ws.root)
    card = _clone(clone_ws, store, E1, [(10.0, 14.0), (70.0, 76.0)], tier="floor")
    with Registry(clone_ws.registry_path) as reg:
        reg.assign_card(1, "en", card, by="studio", locked=True)
    d = open_draft(clone_ws, _cfg(clone_ws), 1, "de")
    assert (d.base, d.derived_from) == ("derived", "en")
    (v,) = d.versions
    derived = store.get(v.card_id)
    assert derived.target_language == "de"
    assert derived.voice.references == store.get(card).voice.references
    assert derived.meta["derived_from"] == card
    with Registry(clone_ws.registry_path) as reg:
        assert reg.card_for(1, "de") is None  # nothing assigned until accept
    assert v.digest == _row_digest(clone_ws, 1, "de")


def test_draft_without_cards_opens_from_the_proposal_and_assigns_nothing(clone_ws):
    from fukikae.studio.drafts import open_draft
    from fukikae.studio.season import registry_digest

    with Registry(clone_ws.registry_path) as reg:
        before = registry_digest(reg)
    d = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    assert d.base == "proposed"
    (v,) = d.versions
    assert v.references and v.backend == "clonefake" and v.tier == "floor"
    assert CardStore(clone_ws.root).get(v.card_id).meta["character"] == 1
    with Registry(clone_ws.registry_path) as reg:
        assert reg.card_for(1, "en") is None
        assert registry_digest(reg) == before


def test_a_stock_card_is_never_a_starting_point(clone_ws):
    """D41: a retired stock card opens no draft; the draft starts from the proposal."""
    from fukikae.cards.card import Card, Voice
    from fukikae.studio.drafts import open_draft

    stock = CardStore(clone_ws.root).put(Card(
        target_language="en",
        voice=Voice(backend="kokoro", model_revision="rev-k", source="stock", voice="am_echo"),
    ))  # fmt: skip
    with Registry(clone_ws.registry_path) as reg:
        reg.assign_card(2, "en", stock, by="auto", locked=False)
    d = open_draft(clone_ws, _cfg(clone_ws), 2, "en")
    assert d.base == "proposed"
    assert CardStore(clone_ws.root).get(d.versions[0].card_id).voice.source == "clone"


# --- draft cards ---------------------------------------------------------------------------


def test_draft_card_matches_the_automatic_clone_for_the_same_references(clone_ws):
    from fukikae.cards.auto import Cloning, _clone, find_references
    from fukikae.studio.drafts import draft_card, open_draft
    from fukikae.studio.model import RefSpan

    found = find_references(clone_ws, [E1, E2], 1, "ja")
    assert found.picks and found.episode
    store = CardStore(clone_ws.root)
    auto_id, _ = _clone(Cloning(CloneSynth, "ja", lambda _c: found), found, store, "en")
    draft = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    refs = [RefSpan(episode=found.episode, start=p.start, end=p.end) for p in found.picks]
    assert draft_card(clone_ws, _cfg(clone_ws), draft, refs, "clonefake", {}) == auto_id
    # the same with the defaults spelled out
    params = {"seed": 1234, "emotion_strength": 0.8}
    assert draft_card(clone_ws, _cfg(clone_ws), draft, refs, "clonefake", params) == auto_id
    other = draft_card(clone_ws, _cfg(clone_ws), draft, refs, "clonefake", {"seed": 7})
    assert other != auto_id and store.get(other).voice.seed == 7


def test_draft_card_takes_references_across_episodes(clone_ws):
    from fukikae.studio.drafts import draft_card, open_draft
    from fukikae.studio.model import RefSpan

    draft = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    refs = [RefSpan(episode=E1, start=10.0, end=14.0), RefSpan(episode=E2, start=60.0, end=67.0)]
    card = CardStore(clone_ws.root).get(
        draft_card(clone_ws, _cfg(clone_ws), draft, refs, "clonefake", {})
    )
    assert [r.origin for r in card.voice.references] == [
        {"episode": E1, "start": 10.0, "end": 14.0},
        {"episode": E2, "start": 60.0, "end": 67.0},
    ]
    assert [r.transcript for r in card.voice.references] == [
        f"{E1} line at 10", f"{E2} line at 60",
    ]  # fmt: skip
    assert card.target_language == "en" and card.meta["reference_tier"] == "floor"


@pytest.mark.parametrize(
    "params",
    [{"pitch": 1.0}, {"temperature": 0.7}, {"emotion_strength": 2.0}, {"seed": -1},
     {"seed": "x"}],
)  # fmt: skip
def test_unknown_voice_key_is_refused(clone_ws, params):
    from fukikae.studio.drafts import DraftError, draft_card, open_draft
    from fukikae.studio.model import RefSpan

    draft = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    refs = [RefSpan(episode=E1, start=10.0, end=14.0)]
    with pytest.raises(DraftError):
        draft_card(clone_ws, _cfg(clone_ws), draft, refs, "clonefake", params)


def test_a_reference_outside_the_characters_speech_is_refused(clone_ws):
    """D41: every draft is a clone of the character's own speech; Natsu's turn is not Lucy's."""
    from fukikae.studio.drafts import DraftError, draft_card, open_draft
    from fukikae.studio.model import RefSpan

    draft = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    with pytest.raises(DraftError, match="own speech"):
        draft_card(clone_ws, _cfg(clone_ws), draft,
                   [RefSpan(episode=E1, start=20.0, end=24.0)], "clonefake", {})  # fmt: skip


def test_gated_pick_is_recorded_as_an_override(clone_ws):
    from fukikae.studio.drafts import add_version, load_draft, open_draft, save_draft
    from fukikae.studio.model import RefSpan

    cfg = _cfg(clone_ws)
    draft = open_draft(clone_ws, cfg, 1, "en")
    save_draft(clone_ws, draft)
    short = RefSpan(episode=E2, start=70.0, end=72.0)  # 2 s: refused for its length
    good = RefSpan(episode=E2, start=2.0, end=6.0)
    draft, v = add_version(clone_ws, cfg, draft, [good, short], "clonefake", {"seed": 5})
    assert v.n == 2 and v.overrides == [short]
    assert list(v.references) == [good, short]
    assert v.params["seed"] == 5 and v.digest == _row_digest(clone_ws, 1, "en")
    again = load_draft(clone_ws, 1, "en")
    assert again is not None and [x.n for x in again.versions] == [1, 2]


def test_floor_meter_reports_the_tier():
    from fukikae.studio.drafts import floor_meter
    from fukikae.studio.model import RefSpan

    def m(*spans):
        return floor_meter([RefSpan(episode=E1, start=a, end=b) for a, b in spans])

    assert m((0, 6), (10, 14)) == {
        "clips": 2, "seconds": 10.0, "floor_clips": 2, "floor_s": 10.0, "prompt_s": 15.0,
        "over_prompt": False, "tier": "floor",
    }  # fmt: skip
    assert m((0, 4))["tier"] == "short"
    assert m((0, 2.5))["tier"] == "thin"
    assert m((0, 9), (10, 18))["over_prompt"] is True


def test_save_draft_is_owner_only_and_atomic(clone_ws):
    from fukikae.studio.drafts import draft_path, open_draft, save_draft

    draft = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    save_draft(clone_ws, draft)
    path = draft_path(clone_ws, 1, "en")
    assert path == clone_ws.root / "studio" / "drafts" / "1.en.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert [p.name for p in path.parent.iterdir()] == ["1.en.json"]


def test_drafts_never_touch_the_registry(clone_ws):
    from fukikae.studio.drafts import add_version, open_draft
    from fukikae.studio.model import RefSpan
    from fukikae.studio.season import registry_digest

    with Registry(clone_ws.registry_path) as reg:
        before = registry_digest(reg)
    draft = open_draft(clone_ws, _cfg(clone_ws), 1, "en")
    add_version(clone_ws, _cfg(clone_ws), draft, [RefSpan(episode=E1, start=10.0, end=14.0)],
                "clonefake", {})  # fmt: skip
    with Registry(clone_ws.registry_path) as reg:
        assert registry_digest(reg) == before


# --- the API -------------------------------------------------------------------------------


@pytest.fixture
def api(clone_ws, studio) -> Client:  # noqa: ARG001, F811 (clone_ws first: the config)
    return studio


def test_draft_routes_need_the_token(api):
    for method, path in (
        ("GET", "/api/characters/1/candidates"), ("GET", "/api/drafts/1/en"),
        ("POST", "/api/drafts/1/en/versions"), ("GET", "/api/backends"),
    ):  # fmt: skip
        assert api.request(method, path, {} if method == "POST" else None, token=None)[0] == 403


def test_candidates_endpoint_pages_and_hides_gated_by_default(api):
    status, body = api.json("GET", "/api/characters/1/candidates")
    assert status == 200, body
    assert body["character"] == 1 and all(c["rank"] is not None for c in body["candidates"])
    status, everything = api.json("GET", "/api/characters/1/candidates?gated=1")
    assert status == 200
    assert everything["total"] > body["total"] == everything["eligible"]
    status, page = api.json("GET", "/api/characters/1/candidates?gated=1&limit=2&offset=1")
    assert [c["start"] for c in page["candidates"]] == [
        c["start"] for c in everything["candidates"][1:3]
    ]
    assert api.json("GET", "/api/characters/1/candidates?limit=100000")[0] == 400
    assert api.json("GET", "/api/characters/99/candidates")[0] == 404


def test_candidate_clips_are_cut_lazily_and_served_with_the_token(api, clone_ws):
    _, body = api.json("GET", "/api/characters/1/candidates?gated=1")
    clip = next(c["clip"] for c in body["candidates"] if c["rank"] is None)
    audio = clone_ws.root / "studio" / "audio" / clip
    assert not audio.exists()  # listing cuts nothing
    status, headers, data = api.request("GET", f"/audio/{clip}")
    assert status == 200 and headers["content-type"] == "audio/wav" and data[:4] == b"RIFF"
    assert audio.is_file()
    assert api.request("GET", f"/audio/{clip}", token=None)[0] == 403
    assert api.request("GET", f"/audio/{'0' * 64}.wav")[0] == 404  # never listed


def test_draft_endpoint_opens_once_and_versions_are_posted(api, clone_ws):
    status, d = api.json("GET", "/api/drafts/1/en")
    assert status == 200, d
    assert d["draft"]["base"] == "proposed" and len(d["draft"]["versions"]) == 1
    assert d["meter"]["tier"] in ("floor", "short", "thin")
    assert (clone_ws.root / "studio" / "drafts" / "1.en.json").is_file()
    status, again = api.json("GET", "/api/drafts/1/en")
    assert again["draft"] == d["draft"]  # opened once, then read
    body = {
        "references": [{"episode": E2, "start": 2.0, "end": 6.0},
                       {"episode": E2, "start": 70.0, "end": 72.0}],
        "backend": "clonefake", "params": {"seed": 9},
    }  # fmt: skip
    status, v = api.json("POST", "/api/drafts/1/en/versions", body)
    assert status == 200, v
    assert v["version"]["n"] == 2 and v["version"]["overrides"] == [body["references"][1]]
    assert v["meter"]["clips"] == 2 and CardStore(clone_ws.root).has(v["version"]["card_id"])


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"references": [{"episode": E1, "start": 10.0, "end": 14.0}], "backend": "clonefake",
          "params": {"pitch": 2}}, 422),
        ({"references": [{"episode": E1, "start": 20.0, "end": 24.0}], "backend": "clonefake",
          "params": {}}, 422),
        ({"references": [], "backend": "clonefake", "params": {}}, 400),
        ({"references": [{"episode": E1, "start": 10.0, "end": 14.0}], "backend": "nope",
          "params": {}}, 422),
        ({"references": "x"}, 400),
    ],
)  # fmt: skip
def test_bad_versions_are_refused(api, body, status):
    assert api.json("POST", "/api/drafts/1/en/versions", body)[0] == status


def test_bad_draft_paths_are_404(api):
    assert api.json("GET", "/api/drafts/99/en")[0] == 404
    assert api.json("GET", "/api/drafts/1/EN!")[0] == 404


def test_backends_endpoint_lists_voice_keys(api):
    status, body = api.json("GET", "/api/backends")
    assert status == 200, body
    (b,) = body["backends"]
    assert b["name"] == "clonefake" and b["default"] is True
    keys = {k["key"]: k for k in b["voice_keys"]}
    assert set(keys) == {"seed", "emotion_strength"}
    assert keys["seed"]["type"] == "int" and keys["seed"]["min"] == 0
    assert keys["emotion_strength"] == {
        "key": "emotion_strength", "type": "float", "min": 0.0, "max": 1.0, "default": 0.8,
        "choices": None,
    }  # fmt: skip


def test_routes_list_the_new_endpoints():
    from fukikae.studio.server import ROUTES

    paths = {p for _, p, _ in ROUTES}
    assert {"/api/characters/<id>/candidates", "/api/drafts/<id>/<lang>",
            "/api/drafts/<id>/<lang>/versions", "/api/backends"} <= paths  # fmt: skip
