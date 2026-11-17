"""Compare, accept and other languages (Task 20): two draft versions on one line side by side
or blind, an accept built server-side from a draft version and applied through
`fukikae.studio.apply.apply_decision`, and a second language's draft opened from the accepted
card."""

from __future__ import annotations

import json
import re
import threading
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
from tests.studio.test_audition import CLASSES, FakeClone, _cfg, _draft, _poll, _real, _typed
from tests.studio.test_season import E1, E2, _clone
from tests.studio.test_server import Client

# the fixtures this module shares (the season workspace, the fake clone backends, the server)
ws = test_season.ws
aud_ws = test_audition.aud_ws
worker = test_audition.worker
studio = test_server.studio
api = test_audition.api


def _at_revision(w: Workspace, card_id: str) -> str:
    """The card again at the audition backend's model revision (the season's `_clone` makes
    ``rev-1``), so the revision check passes it and a refusal is about its references."""
    from dataclasses import replace

    store = CardStore(w.root)
    card = store.get(card_id)
    rev = CLASSES["clonefake"].model_revision
    return store.put(replace(card, voice=replace(card.voice, model_revision=rev)))


def _assignment(w: Workspace, cid: int = 1, lang: str = "en"):
    with Registry(w.registry_path) as reg:
        return reg.card_for(cid, lang)


def _renders() -> int:
    return sum(line.startswith("render") for line in FakeClone.LOG)


def _card(w: Workspace, version: int, lang: str = "en") -> str:
    """The card id of the saved draft's version, as the page sends it with an accept."""
    from fukikae.studio.drafts import load_draft

    d = load_draft(w, 1, lang)
    assert d is not None
    return next(v.card_id for v in d.versions if v.n == version)


def _accept(api: Client, w: Workspace, version: int = 1, *, preview: bool = False,
            card_id: str | None = None) -> tuple[int, dict]:  # fmt: skip
    path = "/api/drafts/1/en/accept" + ("/preview" if preview else "")
    return api.json("POST", path, {"version": version,
                                   "card_id": card_id or _card(w, version)})  # fmt: skip


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

    status, prev = _accept(api, aud_ws, preview=True)
    assert status == 200, prev
    d = prev["decision"]
    assert (d["op"], d["character"], d["language"], d["card_id"]) == ("accept", 1, "en",
                                                                        v["card_id"])  # fmt: skip
    assert (d["draft"], d["version"], d["auditions"], d["digest"]) == ("1.en", 1, 1, v["digest"])
    assert d["key"] == "lucy" and prev["stale"] is None
    assert {"episode": E1, "stage": "tts"} in prev["invalidations"]
    assert prev["cached"][E1] == ["c10"]  # the audition's render is the stage's, and only it
    assert _assignment(aud_ws) == before  # a preview writes nothing

    status, got = _accept(api, aud_ws)
    assert status == 200, got
    a = _assignment(aud_ws)
    assert a is not None and (a.card_id, a.chosen_by, a.locked) == (v["card_id"], "studio", True)
    assert a.note is not None and a.note.startswith(
        "studio: draft 1.en version 1, 1 auditions; references "
    )
    assert any("accepted (studio, locked)" in line for line in got["lines"])
    assert {"episode": E1, "stage": "tts"} in got["invalidations"]
    assert got["cached"][E1] == ["c10"]
    card = next(c for c in got["character"]["cards"] if c["language"] == "en")
    assert (card["card_id"], card["chosen_by"], card["locked"]) == (v["card_id"], "studio", True)
    records = [json.loads(x) for x in decisions_log(aud_ws).read_text().splitlines()]
    assert any(r.get("decision", {}).get("op") == "accept" for r in records)


def test_cached_lines_measure_the_gate_on_every_spoken_cue_as_the_stage_does(aud_ws, monkeypatch):
    """The line gate's pitch tracks are cached by the spans they are taken over, so
    `cached_lines` hands the gate every spoken cue of the episode, as the tts stage does,
    and reads the stage's pitch cache entry instead of measuring a new one."""
    from fukikae.models import DubCues
    from fukikae.studio.audition import cached_lines

    _, v = _draft(aud_ws)
    seen: list[list[str]] = []
    real = tts.gender_gate

    def spy(work, cues, *a, **kw):
        seen.append([c.id for c in cues])
        return real(work, cues, *a, **kw)

    monkeypatch.setattr(tts, "gender_gate", spy)
    cached_lines(aud_ws, _cfg(aud_ws), 1, v.card_id, "en", [E1])
    cues = read_json(aud_ws.work_dir(E1) / "06-translate" / "cues.en.json", DubCues).cues
    spoken = [c.id for c in cues if tts.TtsStage._skipped(c) is None]
    assert any(c.character_id != 1 for c in cues if c.id in spoken)  # others speak in E1
    assert seen == [spoken]


def test_accept_reports_cached_lines_under_the_current_config(aud_ws, api):
    """A ``[tts.<backend>]`` option changed since the server started changes every cue key:
    the accept's ``cached`` reads the workspace's config as it is now, as the next run will."""
    api.json("GET", "/api/drafts/1/en")
    _audition(api, 1, _real(E1, "c10"))
    assert _accept(api, aud_ws, preview=True)[1]["cached"][E1] == ["c10"]
    toml = aud_ws.root / "fukikae.toml"
    toml.write_text(toml.read_text() + "\n[tts.clonefake]\nspeed_hint = 2\n", encoding="utf-8")
    status, got = _accept(api, aud_ws, preview=True)
    assert status == 200 and got["cached"][E1] == [], got


def test_accept_after_a_run_repicked_the_card_is_409(aud_ws, api):
    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 200, body
    other = _clone(aud_ws, CardStore(aud_ws.root), E1, [(10.0, 14.0)])
    with Registry(aud_ws.registry_path) as reg:  # a tts run re-picked the automatic card
        reg.assign_card(1, "en", other, by="auto", locked=False)
    status, got = _accept(api, aud_ws, preview=True)
    assert status == 200 and got["stale"], got  # previewed, and says why it would be refused
    status, got = _accept(api, aud_ws)
    assert status == 409 and got["error"] == "stale", got
    a = _assignment(aud_ws)
    assert a is not None and (a.card_id, a.chosen_by) == (other, "auto")


def test_accepting_twice_is_stale_the_second_time(aud_ws, api):  # noqa: ARG001
    api.json("GET", "/api/drafts/1/en")
    assert _accept(api, aud_ws)[0] == 200
    status, got = _accept(api, aud_ws)
    assert status == 409 and got["error"] == "stale"


def test_accept_refusals(aud_ws, api):
    nocard = "0" * 16
    assert _accept(api, aud_ws, card_id=nocard)[0] == 404  # no draft
    api.json("GET", "/api/drafts/1/en")
    card = _card(aud_ws, 1)
    url = "/api/drafts/1/en/accept"
    assert api.json("POST", url, {"version": 7, "card_id": card})[0] == 404
    assert api.json("POST", url, {"version": 0, "card_id": card})[0] == 400
    assert api.json("POST", url, {"version": 1})[0] == 400  # the card id is required
    assert api.json("POST", url, {"version": 1, "card_id": "nope"})[0] == 400
    assert api.json("POST", url, {"version": 1, "card_id": card, "x": 1})[0] == 400
    assert api.json("GET", url)[0] == 405
    for path in (url, url + "/preview", "/api/drafts/1/en/reveal"):
        status, _ = api.json("POST", path, {"version": 1, "card_id": card}, token=None)
        assert status == 403
    assert api.json("GET", "/api/drafts/1/en/compare?a=1&b=2", token=None)[0] == 403


def test_accept_of_a_version_whose_card_changed_is_409(aud_ws, api):
    """The page sends the card it showed; a draft deleted and opened again since (version 1
    now another card) is refused, never accepted under the old number."""
    from fukikae.studio.drafts import draft_path

    api.json("GET", "/api/drafts/1/en")
    shown = _card(aud_ws, 1)
    draft_path(aud_ws, 1, "en").unlink()
    other = _clone(aud_ws, CardStore(aud_ws.root), E1, [(10.0, 14.0)])
    with Registry(aud_ws.registry_path) as reg:
        reg.assign_card(1, "en", other, by="auto", locked=False)
    api.json("GET", "/api/drafts/1/en")
    assert _card(aud_ws, 1) != shown
    for preview in (True, False):
        status, got = _accept(api, aud_ws, preview=preview, card_id=shown)
        assert status == 409 and got["error"] == "version-changed", got
    assert _assignment(aud_ws).card_id == other  # type: ignore[union-attr]


def test_a_draft_file_for_another_character_or_language_is_refused(aud_ws, api):
    from fukikae.studio.drafts import draft_path, load_draft

    api.json("GET", "/api/drafts/1/en")
    for cid, lang in ((2, "en"), (1, "de")):
        draft_path(aud_ws, cid, lang).write_bytes(draft_path(aud_ws, 1, "en").read_bytes())
        with pytest.raises(ValueError, match="character 1"):
            load_draft(aud_ws, cid, lang)
        status, got = api.json("POST", f"/api/drafts/{cid}/{lang}/accept",
                               {"version": 1, "card_id": _card(aud_ws, 1)})  # fmt: skip
        assert status == 409 and got["error"] == "draft-unreadable", got


def test_accept_while_a_run_holds_the_lock_is_423(aud_ws, api):
    from fukikae.studio.apply import run_lock

    api.json("GET", "/api/drafts/1/en")
    with run_lock(aud_ws.root):
        status, got = _accept(api, aud_ws)
    assert status == 423 and got["error"] == "busy", got
    assert _assignment(aud_ws) is None


def test_accept_refuses_a_version_with_duplicate_references(aud_ws, api):
    """Accept and its preview re-run `accept_problems` for the version, the duplicate check
    included: a version 1 opened from a card that names one span twice (at the backend's own
    revision, so nothing else refuses it) is 422, naming the duplicate."""
    dup = _at_revision(aud_ws, _clone(aud_ws, CardStore(aud_ws.root), E1,
                                      [(10.0, 14.0), (10.0, 14.0)]))  # fmt: skip
    with Registry(aud_ws.registry_path) as reg:
        reg.assign_card(1, "en", dup, by="auto", locked=False)
    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 200 and not body["draft"]["versions"][0]["acceptable"]
    for preview in (True, False):
        status, got = _accept(api, aud_ws, preview=preview)
        assert status == 422 and got["error"] == "refused", got
        assert "more than once" in got["message"], got
    assert _assignment(aud_ws).card_id == dup  # type: ignore[union-attr]


def test_apply_refuses_a_hand_built_accept_of_duplicate_references(aud_ws):
    from fukikae.studio.apply import DecisionRefused, apply_decision, stamp
    from fukikae.studio.model import parse_decision

    dup = _at_revision(aud_ws, _clone(aud_ws, CardStore(aud_ws.root), E1,
                                      [(10.0, 14.0), (10.0, 14.0)]))  # fmt: skip
    refs = [{"episode": E1, "start": 10.0, "end": 14.0}]
    with Registry(aud_ws.registry_path) as reg:
        d = stamp(reg, parse_decision({
            "op": "accept", "character": 1, "language": "en", "card_id": dup, "draft": "1.en",
            "version": 1, "references": refs, "overrides": [], "auditions": 0,
        }))  # fmt: skip
    with pytest.raises(DecisionRefused, match="more than once"):
        apply_decision(aud_ws, d)
    assert _assignment(aud_ws) is None


def test_a_version_without_a_digest_is_refused_unchecked(aud_ws, api):
    from fukikae.studio.drafts import load_draft, save_draft

    api.json("GET", "/api/drafts/1/en")
    d = load_draft(aud_ws, 1, "en")
    assert d is not None
    v = d.versions[0].model_copy(update={"digest": None})
    save_draft(aud_ws, d.model_copy(update={"versions": [v]}))
    status, got = _accept(api, aud_ws)
    assert status == 422 and got["error"] == "unchecked", got
    assert _assignment(aud_ws) is None


def test_an_unacceptable_version_is_refused_by_accept(aud_ws, api):
    """Accept re-verifies own speech (D41) whatever the draft says: a version 1 opened from a
    card made of another speaker's turn (at the backend's own revision, so the revision check
    passes it) is refused with the own-speech reason."""
    natsu = _at_revision(aud_ws, _clone(aud_ws, CardStore(aud_ws.root), E1, [(20.0, 24.0)]))
    with Registry(aud_ws.registry_path) as reg:
        reg.assign_card(1, "en", natsu, by="auto", locked=False)
    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 200 and not body["draft"]["versions"][0]["acceptable"]
    status, got = _accept(api, aud_ws)
    assert status == 422 and got["error"] == "refused", got
    assert "not verifiably the character's own speech" in got["message"], got
    assert "model revision" not in got["message"], got
    assert _assignment(aud_ws).card_id == natsu  # type: ignore[union-attr]


def test_accepted_card_survives_the_next_tts_run(aud_ws, api):
    api.json("GET", "/api/drafts/1/en")
    status, got = _accept(api, aud_ws)
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
    assert _accept(api, aud_ws)[0] == 200
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
        if blind and body["state"] in ("ready", "failed"):
            return body
        if not blind and all(
            s["state"] in ("done", "failed", "cancelled") for s in body["sides"].values()
        ):
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


_WAV = re.compile(r"(?<![0-9a-f])[0-9a-f]{16}\.wav")  # an audition's render key


def test_blind_compare_hides_the_versions_until_reveal(aud_ws, api):
    _draft(aud_ws, "clonefake2")
    line = _real(E1, "c10")
    body = _settled(api, 1, 2, line, blind=True)
    assert body["blind"] is True and body["state"] == "ready" and body["message"] is None
    assert set(body["sides"]) == {"A", "B"}
    for side in body["sides"].values():
        assert set(side) == {"wav_url"} and side["wav_url"], side
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
    by_version = {s["version"]: api.request("GET", s["wav_url"])[2]
                  for s in plain["sides"].values()}  # fmt: skip
    for label, n in key.items():
        status, _, raw = api.request("GET", body["sides"][label]["wav_url"])
        assert status == 200 and raw == by_version[n]


def test_blind_responses_name_no_wav_the_draft_lists(aud_ws, api):
    """A blind side's audio is served under an opaque alias, never the render key the draft
    JSON lists (which would tell the versions apart); the alias needs the token and takes a
    Range like any audition."""
    _draft(aud_ws, "clonefake2")
    line = _real(E1, "c10")
    blind = [_settled(api, 1, 2, line, blind=True), _settled(api, 2, 1, line, blind=True)]
    _, draft = api.json("GET", "/api/drafts/1/en")
    listed = set(_WAV.findall(json.dumps(draft)))
    rendered = {a["wav"] for v in draft["draft"]["versions"] for a in v["auditions"]}
    assert len(rendered) == 2 and rendered <= listed
    for body in blind:
        text = json.dumps(body)
        assert not _WAV.search(text) and not any(name in text for name in rendered), text
    url = blind[0]["sides"]["A"]["wav_url"]
    assert api.request("GET", url, token=None)[0] == 403
    status, headers, raw = api.request("GET", url, headers={"Range": "bytes=0-3"})
    assert status == 206 and raw == b"RIFF" and headers["content-range"].startswith("bytes 0-3/")
    assert api.request("GET", "/audition/b" + "0" * 32 + ".wav")[0] == 404


class _Turnstile:
    """`FakeClone.GATE` letting one render through per `let`."""

    def __init__(self) -> None:
        self.permits = threading.Semaphore(0)

    def wait(self, timeout: float | None = None) -> bool:
        return self.permits.acquire(timeout=timeout)

    def let(self, n: int = 1) -> None:
        for _ in range(n):
            self.permits.release()


@pytest.mark.parametrize("order", [(1, 2), (2, 1)])
def test_blind_polls_never_reveal_which_label_renders_first(aud_ws, api, order):
    """The FIFO worker renders one side before the other; a blind poll answers one state for
    the pair ("rendering" until both are done) and no audio until then, so the poll that sees
    the first side finish looks like every other. The sides are queued in label order, never
    query order."""
    _draft(aud_ws, "clonefake2")
    line = _real(E1, "c10")
    a, b = order
    gate = _Turnstile()
    FakeClone.GATE = gate  # type: ignore[assignment]
    try:
        bodies = [_compare(api, a, b, line, blind=True)[1]]
        jobs = api.server.auditions
        deadline = time.monotonic() + 20
        while sum(j.state == "done" for j in jobs._jobs.values()) < 1:
            gate.let()
            time.sleep(0.05)
            assert time.monotonic() < deadline
        first = next(j for j in jobs._jobs.values() if j.state == "done")
        bodies.append(_compare(api, a, b, line, blind=True)[1])
        assert sum(j.state == "done" for j in jobs._jobs.values()) == 1
        gate.let(50)
        bodies.append(_settled(api, a, b, line, blind=True))
    finally:
        FakeClone.GATE = None
        gate.let(50)
    for body in bodies[:-1]:
        assert body["state"] == "rendering" and body["message"] is None, body
        assert body["sides"] == {"A": {"wav_url": None}, "B": {"wav_url": None}}, body
    assert bodies[-1]["state"] == "ready"
    _, revealed = api.json("POST", "/api/drafts/1/en/reveal", {"a": 1, "b": 2, "line": line})
    assert first.version == revealed["key"]["A"]  # A was queued first


def test_two_tabs_comparing_at_once_queue_each_side_once(aud_ws, api, monkeypatch):
    _draft(aud_ws, "clonefake2")
    line = _real(E1, "c10")
    jobs = api.server.auditions
    real = jobs.submit

    def slow(*a, **kw):
        time.sleep(0.2)  # the window two compares both used to find no job in
        return real(*a, **kw)

    monkeypatch.setattr(jobs, "submit", slow)
    got: list[int] = []
    tabs = [threading.Thread(target=lambda: got.append(_compare(api, 1, 2, line)[0]))
            for _ in range(2)]  # fmt: skip
    for t in tabs:
        t.start()
    for t in tabs:
        t.join(20)
    assert got == [200, 200]
    _settled(api, 1, 2, line)
    submitted = [j.version for j in jobs._jobs.values()]
    assert sorted(submitted) == [1, 2], submitted


def test_a_failed_side_out_of_the_job_ring_waits_for_retry(aud_ws, api, monkeypatch):
    from fukikae.studio import audition

    class Broken:
        def wait(self, timeout: float | None = None) -> bool:  # noqa: ARG002
            raise RuntimeError("the fake backend broke")

    _draft(aud_ws, "clonefake2")
    line = _real(E1, "c10")
    _audition(api, 1, line)
    FakeClone.GATE = Broken()  # type: ignore[assignment]
    try:
        assert _settled(api, 1, 2, line)["sides"]["b"]["state"] == "failed"
    finally:
        FakeClone.GATE = None
    monkeypatch.setattr(audition, "MAX_JOBS", 1)
    _audition(api, 1, _typed("Push the failed job out."))
    jobs = api.server.auditions
    assert all(j.version == 1 for j in jobs._jobs.values())  # version 2's job is forgotten
    n = len(FakeClone.LOG)
    status, body = _compare(api, 1, 2, line)
    assert status == 200 and body["sides"]["b"]["state"] == "failed", body
    status, blind = _compare(api, 1, 2, line, blind=True)
    assert status == 200 and blind["state"] == "failed" and blind["message"], blind
    assert "clonefake" not in json.dumps(blind)
    time.sleep(0.1)
    assert len(FakeClone.LOG) == n and jobs.status().queued == 0
    status, body = _compare(api, 1, 2, line, extra="&retry=1")
    assert status == 200 and body["sides"]["b"]["state"] != "failed", body
    assert _settled(api, 1, 2, line)["sides"]["b"]["state"] == "done"


def test_a_blind_submit_error_names_no_backend_or_version(aud_ws, api, monkeypatch):
    from fukikae.studio.audition import AuditionRefused, QueueFull

    _draft(aud_ws, "clonefake2")
    line = _real(E1, "c10")
    jobs = api.server.auditions
    for err, status in ((AuditionRefused("clonefake2 for version 2 can't render here"), 422),
                        (QueueFull("version 2 on clonefake2: 16 queued"), 503)):  # fmt: skip

        def boom(*_a, err=err, **_k):
            raise err

        monkeypatch.setattr(jobs, "submit", boom)
        got, body = _compare(api, 1, 2, line, blind=True)
        assert got == status, body
        text = json.dumps(body)
        assert "clonefake" not in text and "version" not in text, text
        got, body = _compare(api, 1, 2, line)  # unblinded, the reason is shown
        assert got == status and "clonefake2" in body["message"], body


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
    assert _accept(api, aud_ws, 2)[0] == 200
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


# --- drafts of a character merged away ---------------------------------------------------------


def test_a_draft_of_a_merged_character_is_refused_naming_the_survivor(aud_ws, api):
    from fukikae.studio.apply import apply_decision, stamp
    from fukikae.studio.model import parse_decision

    status, body = api.json("GET", "/api/drafts/1/en")
    assert status == 200, body
    v = body["draft"]["versions"][-1]
    with Registry(aud_ws.registry_path) as reg:
        d = stamp(reg, parse_decision({"op": "merge", "keep": 2, "drop": 1}))
    apply_decision(aud_ws, d)
    with Registry(aud_ws.registry_path) as reg:
        assert 1 not in {c.id for c in reg.characters()}

    calls = [
        ("GET", "/api/drafts/1/en", None),
        ("HEAD", "/api/drafts/1/en", None),
        ("POST", "/api/drafts/1/en/accept/preview", {"version": v["n"], "card_id": v["card_id"]}),
        ("POST", "/api/drafts/1/en/accept", {"version": v["n"], "card_id": v["card_id"]}),
        ("POST", "/api/drafts/1/en/versions",
         {"references": v["references"], "backend": v["backend"], "params": {}}),
        ("GET", f"/api/drafts/1/en/compare?a=1&b=1&line={quote(json.dumps(_real(E1, 'c10')))}",
         None),
        ("POST", "/api/drafts/1/en/reveal", {"a": 1, "b": 2, "line": _real(E1, "c10")}),
        ("POST", "/api/auditions",
         {"character": 1, "language": "en", "version": v["n"], "line": _real(E1, "c10")}),
    ]  # fmt: skip
    for method, path, payload in calls:
        if method == "HEAD":
            assert api.request("HEAD", path)[0] == 409, path
            continue
        status, got = api.json(method, path, payload)
        assert status == 409, (path, status, got)
        assert got["error"] == "character-merged" and got["merged_into"] == 2, got
        assert "character 2" in got["message"], got
    assert not api.server.auditions._jobs  # nothing was queued for the merged character
    assert api.json("GET", "/api/drafts/2/en")[1].get("error") != "character-merged"


def test_a_draft_of_a_character_that_never_existed_is_unknown(aud_ws, api):  # noqa: ARG001
    status, got = api.json("GET", "/api/drafts/99/en")
    assert status == 404 and got["error"] == "unknown-character", got
    status, got = api.json("POST", "/api/auditions",
                           {"character": 99, "language": "en", "version": 1,
                            "line": _real(E1, "c10")})  # fmt: skip
    assert status == 404 and got["error"] == "unknown-character", got
