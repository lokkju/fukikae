"""Compare, accept and other languages (Task 20): two draft versions on one line side by side
or blind, an accept built server-side from a draft version and applied through
`fukikae.studio.apply.apply_decision`, and a second language's draft opened from the accepted
card."""

from __future__ import annotations

import json
import time
from urllib.parse import quote

import pytest

from fukikae.artifacts import read_json
from fukikae.cards.store import CardStore
from fukikae.models import TtsPlan
from fukikae.registry.invalidate import episode
from fukikae.registry.store import Registry
from fukikae.stages import tts
from fukikae.workspace import Workspace
from tests.studio import test_audition, test_season, test_server
from tests.studio.test_audition import FakeClone, _cfg, _draft, _poll, _real, _typed
from tests.studio.test_season import E1, E2, _clone
from tests.studio.test_server import Client

# the fixtures this module shares (the season workspace, the fake clone backends, the server)
ws = test_season.ws
aud_ws = test_audition.aud_ws
worker = test_audition.worker
studio = test_server.studio
api = test_audition.api


def _assignment(w: Workspace, cid: int = 1, lang: str = "en"):
    with Registry(w.registry_path) as reg:
        return reg.card_for(cid, lang)


def _renders() -> int:
    return sum(line.startswith("render") for line in FakeClone.LOG)


def _audition(api: Client, version: int, line: dict) -> dict:
    status, body = api.json(
        "POST", "/api/auditions", {"character": 1, "language": "en", "version": version,
                                   "line": line},
    )  # fmt: skip
    assert status == 202, body
    done = _poll(api, body["job"]["id"])
    assert done["state"] == "done", done
    return done


# --- accept ----------------------------------------------------------------------------------


def test_accept_writes_a_curated_locked_card_through_apply(aud_ws, api):
    from fukikae.studio.apply import decisions_log

    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 200, body
    v = body["draft"]["versions"][0]
    assert v["acceptable"], v["accept_problems"]
    _audition(api, 1, _real(E1, "c10"))
    before = _assignment(aud_ws)

    status, prev = api.json("POST", "/api/drafts/1/en/accept/preview", {"version": 1})
    assert status == 200, prev
    d = prev["decision"]
    assert (d["op"], d["character"], d["language"], d["card_id"]) == ("accept", 1, "en",
                                                                        v["card_id"])  # fmt: skip
    assert (d["draft"], d["version"], d["auditions"], d["digest"]) == ("1.en", 1, 1, v["digest"])
    assert d["key"] == "lucy" and prev["stale"] is None
    assert {"episode": E1, "stage": "tts"} in prev["invalidations"]
    assert "c10" in prev["cached"][E1]  # the audition's render is the stage's
    assert _assignment(aud_ws) == before  # a preview writes nothing

    status, got = api.json("POST", "/api/drafts/1/en/accept", {"version": 1})
    assert status == 200, got
    a = _assignment(aud_ws)
    assert a is not None and (a.card_id, a.chosen_by, a.locked) == (v["card_id"], "studio", True)
    assert a.note is not None and a.note.startswith(
        "studio: draft 1.en version 1, 1 auditions; references "
    )
    assert any("accepted (studio, locked)" in line for line in got["lines"])
    assert {"episode": E1, "stage": "tts"} in got["invalidations"]
    assert "c10" in got["cached"][E1]
    card = next(c for c in got["character"]["cards"] if c["language"] == "en")
    assert (card["card_id"], card["chosen_by"], card["locked"]) == (v["card_id"], "studio", True)
    records = [json.loads(x) for x in decisions_log(aud_ws).read_text().splitlines()]
    assert any(r.get("decision", {}).get("op") == "accept" for r in records)


def test_accept_after_a_run_repicked_the_card_is_409(aud_ws, api):
    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 200, body
    other = _clone(aud_ws, CardStore(aud_ws.root), E1, [(10.0, 14.0)])
    with Registry(aud_ws.registry_path) as reg:  # a tts run re-picked the automatic card
        reg.assign_card(1, "en", other, by="auto", locked=False)
    status, got = api.json("POST", "/api/drafts/1/en/accept/preview", {"version": 1})
    assert status == 200 and got["stale"], got  # previewed, and says why it would be refused
    status, got = api.json("POST", "/api/drafts/1/en/accept", {"version": 1})
    assert status == 409 and got["error"] == "stale", got
    a = _assignment(aud_ws)
    assert a is not None and (a.card_id, a.chosen_by) == (other, "auto")


def test_accepting_twice_is_stale_the_second_time(aud_ws, api):  # noqa: ARG001
    api.json("GET", "/api/drafts/1/en")
    assert api.json("POST", "/api/drafts/1/en/accept", {"version": 1})[0] == 200
    status, got = api.json("POST", "/api/drafts/1/en/accept", {"version": 1})
    assert status == 409 and got["error"] == "stale"


def test_accept_refusals(aud_ws, api):  # noqa: ARG001
    assert api.json("POST", "/api/drafts/1/en/accept", {"version": 1})[0] == 404  # no draft
    api.json("GET", "/api/drafts/1/en")
    assert api.json("POST", "/api/drafts/1/en/accept", {"version": 7})[0] == 404
    assert api.json("POST", "/api/drafts/1/en/accept", {"version": 0})[0] == 400
    assert api.json("POST", "/api/drafts/1/en/accept", {"version": 1, "x": 1})[0] == 400
    assert api.json("GET", "/api/drafts/1/en/accept")[0] == 405
    for path in ("/api/drafts/1/en/accept", "/api/drafts/1/en/accept/preview",
                 "/api/drafts/1/en/reveal"):  # fmt: skip
        status, _ = api.json("POST", path, {"version": 1}, token=None)
        assert status == 403
    assert api.json("GET", "/api/drafts/1/en/compare?a=1&b=2", token=None)[0] == 403


def test_a_version_without_a_digest_is_refused_unchecked(aud_ws, api):
    from fukikae.studio.drafts import load_draft, save_draft

    api.json("GET", "/api/drafts/1/en")
    d = load_draft(aud_ws, 1, "en")
    assert d is not None
    v = d.versions[0].model_copy(update={"digest": None})
    save_draft(aud_ws, d.model_copy(update={"versions": [v]}))
    status, got = api.json("POST", "/api/drafts/1/en/accept", {"version": 1})
    assert status == 422 and got["error"] == "unchecked", got
    assert _assignment(aud_ws) is None


def test_an_unacceptable_version_is_refused_by_accept(aud_ws, api):
    """Accept re-verifies own speech (D41) whatever the draft says: a version 1 opened from a
    card made of another speaker's turn is refused with accept's own reason."""
    natsu = _clone(aud_ws, CardStore(aud_ws.root), E1, [(20.0, 24.0)])  # Natsu's turn
    with Registry(aud_ws.registry_path) as reg:
        reg.assign_card(1, "en", natsu, by="auto", locked=False)
    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 200 and not body["draft"]["versions"][0]["acceptable"]
    status, got = api.json("POST", "/api/drafts/1/en/accept", {"version": 1})
    assert status == 422 and got["error"] == "refused", got
    assert _assignment(aud_ws).card_id == natsu  # type: ignore[union-attr]


def test_accepted_card_survives_the_next_tts_run(aud_ws, api):
    api.json("GET", "/api/drafts/1/en")
    status, got = api.json("POST", "/api/drafts/1/en/accept", {"version": 1})
    assert status == 200, got
    accepted = _assignment(aud_ws)
    cfg = _cfg(aud_ws)
    tts.TtsStage(cfg).run(episode(aud_ws, E1), cfg)
    assert _assignment(aud_ws) == accepted  # CD2: no automatic rule replaces it
    plan = read_json(aud_ws.work_dir(E1) / "07-tts" / "plan.json", TtsPlan)
    mine = [c for c in plan.cues if c.character_id == 1]
    assert mine and all(c.card_id == accepted.card_id for c in mine)  # type: ignore[union-attr]


def test_later_episode_uses_the_accepted_card(aud_ws, api):
    api.json("GET", "/api/drafts/1/en")
    assert api.json("POST", "/api/drafts/1/en/accept", {"version": 1})[0] == 200
    accepted = _assignment(aud_ws)
    assert accepted is not None
    cfg = _cfg(aud_ws)
    tts.TtsStage(cfg).run(episode(aud_ws, E2), cfg)  # Lucy speaks in E2 too
    plan = read_json(aud_ws.work_dir(E2) / "07-tts" / "plan.json", TtsPlan)
    mine = [c for c in plan.cues if c.character_id == 1]
    assert mine and all(c.card_id == accepted.card_id for c in mine)


# --- compare ---------------------------------------------------------------------------------


def _compare(api: Client, a: int, b: int, line: dict, *, blind: bool = False,
             extra: str = "") -> tuple[int, dict]:  # fmt: skip
    q = f"a={a}&b={b}&line={quote(json.dumps(line))}" + ("&blind=1" if blind else "") + extra
    return api.json("GET", f"/api/drafts/1/en/compare?{q}")


def _settled(api: Client, a: int, b: int, line: dict, *, blind: bool = False) -> dict:
    deadline = time.monotonic() + 20
    while True:
        status, body = _compare(api, a, b, line, blind=blind)
        assert status == 200, body
        if all(s["state"] in ("done", "failed", "cancelled") for s in body["sides"].values()):
            return body
        assert time.monotonic() < deadline, body
        time.sleep(0.02)


def test_compare_submits_the_missing_side_and_pairs_by_line(aud_ws, api):
    _draft(aud_ws, "clonefake2")  # version 2 on another backend: another card, another key
    first = _audition(api, 1, _real(E1, "c10"))
    line = _real(E1, "c10")
    status, body = _compare(api, 1, 2, line)
    assert status == 200, body
    assert body["blind"] is False and set(body["sides"]) == {"a", "b"}
    assert body["sides"]["a"]["state"] == "done" and body["sides"]["a"]["version"] == 1
    assert body["sides"]["a"]["wav_url"] == first["wav_url"]  # the existing audition, reused
    assert body["sides"]["b"]["version"] == 2 and body["sides"]["b"]["state"] != "done"
    done = _settled(api, 1, 2, line)
    a, b = done["sides"]["a"], done["sides"]["b"]
    assert b["state"] == "done" and a["audition"]["cue_key"] != b["audition"]["cue_key"]
    assert a["audition"]["line"] == b["audition"]["line"]
    assert done["source_clip"] is not None
    n = _renders()
    _settled(api, 1, 2, line)  # polling again submits nothing
    assert _renders() == n
    status, _, raw = api.request("GET", b["wav_url"])
    assert status == 200 and raw[:4] == b"RIFF"


def test_compare_takes_typed_lines(aud_ws, api):
    _draft(aud_ws, "clonefake2")
    body = _settled(api, 1, 2, _typed("Shall we go?"))
    assert all(s["state"] == "done" for s in body["sides"].values())
    assert body["source_clip"] is None


def test_blind_compare_hides_the_versions_until_reveal(aud_ws, api):
    _draft(aud_ws, "clonefake2")
    line = _real(E1, "c10")
    body = _settled(api, 1, 2, line, blind=True)
    assert body["blind"] is True and set(body["sides"]) == {"A", "B"}
    for side in body["sides"].values():
        assert set(side) <= {"state", "wav_url", "message"}, side
    text = json.dumps(body)
    assert "card_id" not in text and "clonefake" not in text and "version" not in text
    again = _settled(api, 2, 1, line, blind=True)  # the same pair either way round
    assert again["sides"] == body["sides"]
    status, revealed = api.json("POST", "/api/drafts/1/en/reveal", {"a": 1, "b": 2,
                                                                    "line": line})  # fmt: skip
    assert status == 200, revealed
    key = revealed["key"]
    assert sorted(key.values()) == [1, 2]
    _, plain = _compare(api, 1, 2, line)
    by_version = {s["version"]: s["wav_url"] for s in plain["sides"].values()}
    assert {lab: by_version[n] for lab, n in key.items()} == {
        lab: s["wav_url"] for lab, s in body["sides"].items()
    }


def test_compare_query_refusals(aud_ws, api):
    line = quote(json.dumps(_real(E1, "c10")))
    assert api.json("GET", f"/api/drafts/1/en/compare?a=1&b=2&line={line}")[0] == 404  # no draft
    _draft(aud_ws, "clonefake2")
    for q in (f"a=1&b=1&line={line}", f"a=1&line={line}", "a=1&b=2", "a=1&b=2&line=nope",
              f"a=x&b=2&line={line}", f"a=1&b=2&line={line}&blind=2"):  # fmt: skip
        status, got = api.json("GET", f"/api/drafts/1/en/compare?{q}")
        assert status == 400, (q, got)
    assert api.json("GET", f"/api/drafts/1/en/compare?a=1&b=9&line={line}")[0] == 404
    other = quote(json.dumps(_real(E1, "c20")))  # Natsu's line
    assert api.json("GET", f"/api/drafts/1/en/compare?a=1&b=2&line={other}")[0] == 422
    assert api.json("POST", "/api/drafts/1/en/compare", {})[0] == 405
    status, _ = api.json("POST", "/api/drafts/1/en/reveal", {"a": 1, "b": 1,
                                                             "line": _real(E1, "c10")})  # fmt: skip
    assert status == 400


def test_head_on_compare_submits_nothing(aud_ws, api):
    _draft(aud_ws, "clonefake2")
    line = quote(json.dumps(_real(E1, "c10")))
    status, _, _ = api.request("HEAD", f"/api/drafts/1/en/compare?a=1&b=2&line={line}")
    assert status == 200
    time.sleep(0.1)
    assert _renders() == 0 and api.server.auditions.status().queued == 0


# --- other languages -------------------------------------------------------------------------


def test_second_language_draft_starts_from_the_accepted_card(aud_ws, api):
    _, v2 = _draft(aud_ws, "clonefake2")
    assert api.json("POST", "/api/drafts/1/en/accept", {"version": 2})[0] == 200
    status, body = api.json("GET", "/api/drafts/1/de")
    assert status == 200, body
    d = body["draft"]
    assert (d["base"], d["derived_from"], d["language"]) == ("derived", "en", "de")
    v = d["versions"][0]
    assert v["backend"] == "clonefake2" and v["references"] == [
        r.model_dump(mode="json") for r in v2.references
    ]
    card = CardStore(aud_ws.root).get(v["card_id"])
    assert card.voice.source == "clone" and card.target_language == "de"
    assert card.meta["derived_from"] == v2.card_id
    assert _assignment(aud_ws, 1, "de") is None  # opening a draft assigns nothing


@pytest.mark.parametrize("by", ["assign", "derive"])
def test_second_language_falls_back_to_another_operator_card(aud_ws, api, by):
    _, v2 = _draft(aud_ws, "clonefake2")
    with Registry(aud_ws.registry_path) as reg:
        reg.assign_card(1, "en", v2.card_id, by=by, locked=True)
    status, body = api.json("GET", "/api/drafts/1/de")
    assert status == 200 and body["draft"]["base"] == "derived"
    assert body["draft"]["derived_from"] == "en"
