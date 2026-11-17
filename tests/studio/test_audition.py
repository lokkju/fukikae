"""The audition worker (Task 19): draft versions rendered through the tts stage's own render
path and cache, one job at a time under the GPU lock, the line gate on every emotion prompt,
and the API that queues jobs, polls them and serves their audio."""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from fukikae import gpulock
from fukikae.artifacts import read_json, write_json
from fukikae.backends import loader
from fukikae.cards.card import Voice
from fukikae.cards.store import CardStore
from fukikae.config import load_config
from fukikae.models import DubCues, TtsRun
from fukikae.registry.invalidate import episode
from fukikae.registry.store import Registry
from fukikae.stages import tts
from fukikae.workspace import Workspace
from tests.cards.test_select import CloneSynth
from tests.studio.test_season import E1, E2, ws  # noqa: F401 (the season fixture workspace)
from tests.studio.test_server import Client, studio  # noqa: F401 (fixture)


def _tone(path: Path, seconds: float, rate: int) -> float:
    n = max(1, round(seconds * rate))
    x = 0.25 * np.sin(2 * np.pi * 220 * np.arange(n) / rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return n / rate


class FakeClone(CloneSynth):
    """The season fixture's clone backend, able to load and render: a tone of 0.06 s per
    character of the text, each call logged."""

    LOG: ClassVar[list[str]] = []
    KW: ClassVar[list[dict[str, Any]]] = []
    ACTIVE: ClassVar[list[int]] = [0, 0]  # renders now, most at once
    GATE: ClassVar[threading.Event | None] = None  # set: renders wait for it

    def __init__(self, options: dict[str, Any], device: str) -> None:  # noqa: ARG002
        self.device = device

    def load(self) -> None:
        self.LOG.append(f"load {self.name}")

    def unload(self) -> None:
        self.LOG.append(f"unload {self.name}")

    def synthesize(self, text: str, voice: Voice, language: str, out: Path, **kw: Any) -> float:  # noqa: ARG002
        a = type(self).ACTIVE
        a[0] += 1
        a[1] = max(a[1], a[0])
        try:
            if FakeClone.GATE is not None:
                FakeClone.GATE.wait(20)
            emo = kw.get("emotion_audio")
            given = None if emo is None else Path(emo).is_file()
            self.KW.append({"text": text, "backend": self.name, "emotion_audio": given})
            self.LOG.append(f"render {self.name} {text}")
            return _tone(out, len(text) * 0.06, self.sample_rate)
        finally:
            a[0] -= 1


class FakeClone2(FakeClone):
    name: ClassVar[str] = "clonefake2"


class GpuClone(FakeClone):
    name: ClassVar[str] = "gpuclone"
    devices: ClassVar = frozenset({"cuda"})
    default_device: ClassVar[str] = "cuda"


CLASSES: dict[str, type] = {"clonefake": FakeClone, "clonefake2": FakeClone2, "gpuclone": GpuClone}


@pytest.fixture
def aud_ws(ws: Workspace, monkeypatch, tmp_path) -> Workspace:  # noqa: F811
    (ws.root / "fukikae.toml").write_text(
        'source_language = "ja"\ntarget_language = "en"\n\n[tts]\nclone_backend = "clonefake"\n',
        encoding="utf-8",
    )
    FakeClone.LOG = []
    FakeClone.KW = []
    FakeClone.ACTIVE = [0, 0]
    FakeClone.GATE = None
    monkeypatch.setattr(loader, "backend_class", lambda _kind, name: CLASSES[name])
    monkeypatch.setattr(loader, "available", lambda _kind: dict(CLASSES))
    monkeypatch.setattr(loader, "engine_problem", lambda *_a: None)
    monkeypatch.setattr(loader, "create", lambda _k, bc, dev: CLASSES[bc.backend](bc.options, dev))
    monkeypatch.setattr(tts, "identity_key", lambda _k, bc: {"model": bc.backend, "rev": "r"})
    monkeypatch.setattr(tts, "device_string", lambda _cfg: "cuda:0")
    monkeypatch.setattr(tts, "wait_for_vram", lambda *_a, **_k: None)
    monkeypatch.setattr(gpulock, "leftover_jobs", lambda _i: [])
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    monkeypatch.setenv(gpulock.LOCK_DIR_ENV, str(lock_dir))
    return ws


def _cfg(w: Workspace):
    return load_config(w.root, user_config=None)


def _draft(w: Workspace, backend: str | None = None):
    """Character 1's English draft, saved; with ``backend``, a version 2 on that backend."""
    from fukikae.studio.drafts import add_version, open_draft, save_draft

    d = open_draft(w, _cfg(w), 1, "en")
    save_draft(w, d)
    if backend is None:
        return d, d.versions[0]
    d, v = add_version(w, _cfg(w), d, d.versions[0].references, backend, {})
    return d, v


@pytest.fixture
def worker(aud_ws):
    from fukikae.studio.audition import AuditionWorker

    w = AuditionWorker(aud_ws, _cfg(aud_ws), idle_s=30.0, poll_s=0.02)
    yield w
    w.close()


def _real(ep: str, cue: str) -> dict:
    return {"kind": "real", "episode": ep, "cue_id": cue}


def _typed(text: str, emotion_from: dict | None = None) -> dict:
    return {"kind": "typed", "text": text, "emotion_from": emotion_from}


def _done(worker, job: str, timeout: float = 20.0):
    got = worker.wait(job, timeout)
    assert got.state == "done", (got.state, got.message)
    return got


def _until(pred, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _stage_job(w: Workspace, ep: str, cue_id: str, card_id: str):
    """The RenderJob the tts stage would make for the cue with the card (its key, passed)."""
    cfg = _cfg(w)
    work = w.work_dir(ep)
    cues = read_json(work / "06-translate" / "cues.en.json", DubCues).cues
    spoken = [c for c in cues if tts.TtsStage._skipped(c) is None]
    gate = tts.gender_gate(work, spoken, w.root, operator=tts.TtsStage.operator_state(w.root)[0])
    cue = next(c for c in cues if c.id == cue_id)
    return tts.prepare_cue(
        cue, card_id, cfg, CardStore(w.root), who="x", vocals=work / "01-separate" / "vocals.wav",
        events=tts.tag_events(work), line=gate.line(cue_id), voice=gate.voice(cue_id),
    )  # fmt: skip


def _only_character(w: Workspace, ep: str, cid: int, extra: list | None = None) -> None:
    """Keep only character ``cid``'s cues in the episode's English cues (plus ``extra``)."""
    path = w.work_dir(ep) / "06-translate" / "cues.en.json"
    doc = read_json(path, DubCues)
    keep = [c for c in doc.cues if c.character_id == cid] + list(extra or [])
    write_json(path, doc.model_copy(update={"cues": keep}))


def _add_cue(w: Workspace, ep: str, cue_id: str, cid: int, ref: tuple[float, float],
             text: str = "borrowed line") -> None:  # fmt: skip
    path = w.work_dir(ep) / "06-translate" / "cues.en.json"
    doc = read_json(path, DubCues)
    base = doc.cues[0]
    cue = base.model_copy(update={
        "id": cue_id, "character_id": cid, "text": text, "start": ref[0], "end": ref[1],
        "delivery": base.delivery.model_copy(update={"reference": ref}),
    })  # fmt: skip
    write_json(path, doc.model_copy(update={"cues": [*doc.cues, cue]}))


# --- the render path -------------------------------------------------------------------------


def test_real_line_audition_uses_the_stage_cue_key(aud_ws, worker):
    _, v = _draft(aud_ws)
    job = worker.submit(1, "en", v.n, _real(E1, "c10"))
    a = _done(worker, job).audition
    stage = _stage_job(aud_ws, E1, "c10", v.card_id)
    assert a.cue_key == stage.key and a.passed == stage.passed
    assert "emotion_audio" in a.passed and a.gender_gate is False
    assert a.card_id == v.card_id and a.renders == 1 and a.seconds > 0
    assert (aud_ws.root / ".fukikae" / "tts-cache" / f"{a.key}.wav").is_file()
    assert (aud_ws.root / "studio" / "auditions" / a.wav).is_file()
    assert a.source is not None and (a.source.episode, a.source.start) == (E1, 10.0)
    assert FakeClone.KW[-1]["emotion_audio"] is True  # the cue's own span was passed


def test_the_audition_is_recorded_on_the_draft_version(aud_ws, worker):
    from fukikae.studio.drafts import load_draft

    _, v = _draft(aud_ws)
    job = worker.submit(1, "en", v.n, _real(E1, "c10"))
    a = _done(worker, job).audition
    draft = load_draft(aud_ws, 1, "en")
    assert draft is not None
    (saved,) = draft.versions[0].auditions
    assert saved == a and saved.job == job


def test_a_second_audition_of_the_same_line_is_a_cache_hit(aud_ws, worker):
    _, v = _draft(aud_ws)
    first = _done(worker, worker.submit(1, "en", v.n, _real(E1, "c10"))).audition
    again = _done(worker, worker.submit(1, "en", v.n, _real(E1, "c10"))).audition
    assert again.renders == 0 and again.key == first.key
    assert sum(line.startswith("render") for line in FakeClone.LOG) == 1


def test_audition_render_is_reused_by_the_next_tts_run(aud_ws, worker):
    _only_character(aud_ws, E1, 1)
    _, v = _draft(aud_ws)
    with Registry(aud_ws.registry_path) as reg:  # what an accept leaves: the version's card
        reg.assign_card(1, "en", v.card_id, by="studio", locked=True)
    for cue in ("c10", "c60"):
        _done(worker, worker.submit(1, "en", v.n, _real(E1, cue)))
    worker.release()
    rendered = [line for line in FakeClone.LOG if line.startswith("render")]
    cfg = _cfg(aud_ws)
    tts.TtsStage(cfg).run(episode(aud_ws, E1), cfg)
    run = read_json(aud_ws.work_dir(E1) / "07-tts" / "run.json", TtsRun)
    assert {"c10", "c60"} <= set(run.cached) and not {"c10", "c60"} & set(run.rendered)
    later = [line for line in FakeClone.LOG if line.startswith("render")]
    assert len(later) - len(rendered) == len(run.rendered)  # only the other lines rendered


def test_wrong_gender_source_is_never_an_emotion_prompt_in_auditions(aud_ws, worker):
    _add_cue(aud_ws, E1, "cx", 1, (20.0, 24.0))  # Lucy's line over Natsu's (male) speech
    _, v = _draft(aud_ws)
    a = _done(worker, worker.submit(1, "en", v.n, _real(E1, "cx"))).audition
    assert a.gender_gate is True and a.source_gender == "m"
    assert "emotion_audio" not in a.passed and a.passed["emotion_dropped"] == "gender"
    assert FakeClone.KW[-1]["emotion_audio"] is None
    assert a.cue_key == _stage_job(aud_ws, E1, "cx", v.card_id).key


def test_the_operators_gender_beats_the_estimate_in_the_gate(aud_ws, worker):
    with Registry(aud_ws.registry_path) as reg:
        reg.set_character_gender(1, "m")
    _, v = _draft(aud_ws)
    a = _done(worker, worker.submit(1, "en", v.n, _real(E1, "c10"))).audition
    assert a.gender_gate is True and "emotion_audio" not in a.passed  # her f line, an m voice
    assert FakeClone.KW[-1]["emotion_audio"] is None


def test_typed_line_passes_no_emotion_audio_by_default(aud_ws, worker):
    _, v = _draft(aud_ws)
    a = _done(worker, worker.submit(1, "en", v.n, _typed("Hello there."))).audition
    assert "emotion_audio" not in a.passed and a.source is None and a.gender_gate is False
    assert FakeClone.KW[-1] == {"text": "Hello there.", "backend": "clonefake",
                                "emotion_audio": None}  # fmt: skip


def test_typed_line_takes_an_emotion_prompt_under_the_same_gate(aud_ws, worker):
    _, v = _draft(aud_ws)
    same = _typed("Hello there.", {"episode": E1, "cue_id": "c10"})  # her own line: f on f
    a = _done(worker, worker.submit(1, "en", v.n, same)).audition
    assert a.passed["emotion_audio"] == [10.0, 14.0] and FakeClone.KW[-1]["emotion_audio"]
    other = _typed("Hello there.", {"episode": E1, "cue_id": "c20"})  # Natsu's line: m
    b = _done(worker, worker.submit(1, "en", v.n, other)).audition
    assert b.gender_gate is True and "emotion_audio" not in b.passed
    assert FakeClone.KW[-1]["emotion_audio"] is None


@pytest.mark.parametrize(
    ("line", "why"),
    [
        (_real(E1, "nope"), "no cue"),
        (_real(E1, "c20"), "character 2"),  # another character's line
        (_real(E2, "c2"), "translate"),  # E2 never ran translate
        (_typed("Hi", {"episode": E1, "cue_id": "nope"}), "no cue"),
    ],
)
def test_bad_lines_are_refused_at_submit(aud_ws, worker, line, why):
    from fukikae.studio.audition import AuditionRefused

    if why == "translate":
        (aud_ws.work_dir(E2) / "06-translate" / "cues.en.json").unlink()
    _, v = _draft(aud_ws)
    with pytest.raises(AuditionRefused, match=why):
        worker.submit(1, "en", v.n, line)


def test_an_uninstalled_backend_is_refused_and_nothing_falls_back(aud_ws, worker, monkeypatch):
    from fukikae.studio.audition import AuditionRefused

    _, v = _draft(aud_ws)
    monkeypatch.setattr("fukikae.cards.auto.clone_problem", lambda *_: "no worker environment")
    with pytest.raises(AuditionRefused, match=r"clonefake.*no worker environment"):
        worker.submit(1, "en", v.n, _real(E1, "c10"))
    assert FakeClone.LOG == []


def test_a_version_naming_an_unknown_backend_is_refused(aud_ws, worker, monkeypatch):
    from fukikae.studio.audition import AuditionRefused

    _, v = _draft(aud_ws)
    monkeypatch.setattr(loader, "backend_class", lambda _k, name: {}[name])
    with pytest.raises(AuditionRefused, match="clonefake"):
        worker.submit(1, "en", v.n, _real(E1, "c10"))
    assert FakeClone.LOG == []


# --- the queue and the GPU lock --------------------------------------------------------------


def test_jobs_run_one_at_a_time_in_order(aud_ws, worker):
    _, v = _draft(aud_ws)
    texts = ["one line", "two lines here", "three"]
    jobs = [worker.submit(1, "en", v.n, _typed(t)) for t in texts]
    for j in jobs:
        _done(worker, j)
    assert [k["text"] for k in FakeClone.KW] == texts
    assert FakeClone.ACTIVE[1] == 1


def test_switching_backend_unloads_the_previous_one(aud_ws, worker):
    _, v1 = _draft(aud_ws)
    _, v2 = _draft(aud_ws, "clonefake2")
    _done(worker, worker.submit(1, "en", v1.n, _typed("first")))
    _done(worker, worker.submit(1, "en", v2.n, _typed("second")))
    loads = [x for x in FakeClone.LOG if not x.startswith("render")]
    assert loads == ["load clonefake", "unload clonefake", "load clonefake2"]
    assert worker.status().loaded_backend == "clonefake2"


def _lock_free(index: int = 0) -> bool:
    try:
        with gpulock.gpu_lock(index, "probe", timeout_s=0):
            return True
    except gpulock.GpuBusy:
        return False


def _probe(index: int = 0) -> bool:
    """Whether another thread could take the card now (the worker's thread holds it per
    thread, so the probe runs in one of its own)."""
    out: list[bool] = []
    t = threading.Thread(target=lambda: out.append(_lock_free(index)))
    t.start()
    t.join(10)
    return out[0]


def test_worker_takes_the_gpu_lock_while_loaded_and_releases_on_idle(aud_ws):
    from fukikae.studio.audition import AuditionWorker

    _, v = _draft(aud_ws, "gpuclone")
    w = AuditionWorker(aud_ws, _cfg(aud_ws), idle_s=0.5, poll_s=0.02)
    try:
        _done(w, w.submit(1, "en", v.n, _typed("on the card")))
        s = w.status()
        assert s.loaded_backend == "gpuclone" and s.device == "cuda:0" and s.ours is True
        assert not _probe()  # held while the model is loaded (CD15)
        _until(lambda: w.status().loaded_backend is None)
        assert _probe() and "unload gpuclone" in FakeClone.LOG
    finally:
        w.close()


def test_release_unloads_and_frees_the_lock(aud_ws, worker):
    _, v = _draft(aud_ws, "gpuclone")
    _done(worker, worker.submit(1, "en", v.n, _typed("on the card")))
    assert not _probe()
    worker.release()
    _until(lambda: worker.status().loaded_backend is None)
    assert _probe() and FakeClone.LOG[-1] == "unload gpuclone"


def test_busy_gpu_is_reported_with_its_holder(aud_ws, worker):
    _, v = _draft(aud_ws, "gpuclone")
    held, done = threading.Event(), threading.Event()

    def run() -> None:
        with gpulock.gpu_lock(0, "tts indextts2", timeout_s=0):
            held.set()
            done.wait(20)

    t = threading.Thread(target=run)
    t.start()
    try:
        held.wait(5)
        job = worker.submit(1, "en", v.n, _typed("waits its turn"))
        _until(lambda: worker.job(job).state == "waiting")
        s = worker.status()
        assert s.holder_job == "tts indextts2" and s.ours is False and s.queued == 0
        assert "tts indextts2" in (s.holder or "") and s.running == job
        assert FakeClone.LOG == []  # nothing loads while another job holds the card
    finally:
        done.set()
        t.join(5)
    got = _done(worker, job)
    assert got.waited_s is not None and got.waited_s > 0


def test_a_waiting_job_can_be_cancelled(aud_ws, worker):
    _, v = _draft(aud_ws, "gpuclone")
    held, done = threading.Event(), threading.Event()

    def run() -> None:
        with gpulock.gpu_lock(0, "tts indextts2", timeout_s=0):
            held.set()
            done.wait(20)

    t = threading.Thread(target=run)
    t.start()
    try:
        held.wait(5)
        first = worker.submit(1, "en", v.n, _typed("first"))
        second = worker.submit(1, "en", v.n, _typed("second"))
        _until(lambda: worker.job(first).state == "waiting")
        assert worker.cancel(second).state == "cancelled"  # queued: dropped
        worker.cancel(first)
        assert worker.wait(first, 10).state == "cancelled"
    finally:
        done.set()
        t.join(5)
    assert FakeClone.LOG == [] and worker.status().queued == 0


def test_release_cancels_the_waiting_and_queued_jobs(aud_ws, worker):
    _, v = _draft(aud_ws, "gpuclone")
    held, done = threading.Event(), threading.Event()

    def run() -> None:
        with gpulock.gpu_lock(0, "tts indextts2", timeout_s=0):
            held.set()
            done.wait(20)

    t = threading.Thread(target=run)
    t.start()
    try:
        held.wait(5)
        jobs = [worker.submit(1, "en", v.n, _typed(x)) for x in ("a", "b")]
        _until(lambda: worker.job(jobs[0]).state == "waiting")
        worker.release()
        assert [worker.wait(j, 10).state for j in jobs] == ["cancelled", "cancelled"]
    finally:
        done.set()
        t.join(5)


def test_a_failed_render_fails_the_job_and_the_worker_goes_on(aud_ws, worker, monkeypatch):
    _, v = _draft(aud_ws)
    real = FakeClone.synthesize

    def boom(self, text, *a, **kw):
        if text == "bad":
            raise RuntimeError("worker died")
        return real(self, text, *a, **kw)

    monkeypatch.setattr(FakeClone, "synthesize", boom)
    bad = worker.wait(worker.submit(1, "en", v.n, _typed("bad")), 10)
    assert bad.state == "failed" and "worker died" in (bad.message or "")
    _done(worker, worker.submit(1, "en", v.n, _typed("good")))


def test_the_queue_is_bounded(aud_ws, worker):
    from fukikae.studio.audition import MAX_QUEUE, QueueFull

    _, v = _draft(aud_ws)
    FakeClone.GATE = threading.Event()
    try:
        with pytest.raises(QueueFull):
            for i in range(MAX_QUEUE + 2):
                worker.submit(1, "en", v.n, _typed(f"line {i}"))
    finally:
        FakeClone.GATE.set()


# --- the API ---------------------------------------------------------------------------------


@pytest.fixture
def api(aud_ws, studio) -> Client:  # noqa: ARG001, F811 (aud_ws first: the config)
    return studio


def _poll(api: Client, job: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        status, body = api.json("GET", f"/api/auditions/{job}")
        assert status == 200, body
        if body["state"] in ("done", "failed", "cancelled"):
            return body
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_audition_routes_need_the_token(api):
    for method, path in (
        ("POST", "/api/auditions"), ("GET", "/api/auditions/0123456789abcdef"),
        ("POST", "/api/auditions/0123456789abcdef/cancel"), ("GET", "/api/gpu"),
        ("POST", "/api/gpu/release"), ("GET", "/audition/0123456789abcdef.wav"),
    ):  # fmt: skip
        assert api.request(method, path, {} if method == "POST" else None, token=None)[0] == 403


def test_an_audition_is_queued_polled_and_served(api, aud_ws):
    _, v = _draft(aud_ws)
    status, body = api.json("POST", "/api/auditions", {
        "character": 1, "language": "en", "version": v.n, "line": _real(E1, "c10"),
    })  # fmt: skip
    assert status == 202, body
    done = _poll(api, body["job"]["id"])
    assert done["state"] == "done", done
    url = done["wav_url"]
    assert url.startswith("/audition/") and done["audition"]["cue_key"]
    status, headers, data = api.request("GET", url)
    assert status == 200 and headers["content-type"] == "audio/wav" and data[:4] == b"RIFF"
    status, headers, part = api.request("GET", url, headers={"Range": "bytes=0-3"})
    assert status == 206 and part == b"RIFF"
    status, headers, src = api.request("GET", f"/audio/{done['source_clip']}")
    assert status == 200 and src[:4] == b"RIFF"  # the source line, played beside it
    assert api.request("GET", "/audition/fedcba9876543210.wav")[0] == 404  # never made


def test_audition_audio_is_never_served_through_a_symlink(api, aud_ws):
    _, v = _draft(aud_ws)
    _, body = api.json("POST", "/api/auditions", {
        "character": 1, "language": "en", "version": v.n, "line": _typed("hello"),
    })  # fmt: skip
    done = _poll(api, body["job"]["id"])
    path = aud_ws.root / "studio" / "auditions" / done["audition"]["wav"]
    path.unlink()
    path.symlink_to(aud_ws.root / "fukikae.toml")
    assert api.request("GET", done["wav_url"])[0] == 404


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"character": 1, "language": "en", "version": 1, "line": {"kind": "sung"}}, 400),
        ({"character": 1, "language": "en", "version": 1,
          "line": {"kind": "typed", "text": ""}}, 400),
        ({"character": 1, "language": "en", "version": 1,
          "line": {"kind": "real", "episode": "../x", "cue_id": "c1"}}, 400),
        ({"character": 1, "language": "en", "version": 9, "line": _typed("x")}, 404),
        ({"character": 99, "language": "en", "version": 1, "line": _typed("x")}, 404),
        ({"character": 1, "language": "en", "version": 1, "line": _real(E1, "c20")}, 422),
    ],
)  # fmt: skip
def test_bad_auditions_are_refused(api, aud_ws, body, status):
    _draft(aud_ws)
    assert api.json("POST", "/api/auditions", body)[0] == status


def test_gpu_status_and_release(api, aud_ws):
    _, v = _draft(aud_ws, "gpuclone")
    _, body = api.json("POST", "/api/auditions", {
        "character": 1, "language": "en", "version": v.n, "line": _typed("on the card"),
    })  # fmt: skip
    _poll(api, body["job"]["id"])
    status, gpu = api.json("GET", "/api/gpu")
    assert status == 200 and gpu["loaded_backend"] == "gpuclone" and gpu["ours"] is True
    assert api.json("POST", "/api/gpu/release")[0] == 200
    _until(lambda: api.json("GET", "/api/gpu")[1]["loaded_backend"] is None)
    assert _probe()


def test_cancel_route(api, aud_ws):
    _, v = _draft(aud_ws)
    FakeClone.GATE = threading.Event()
    try:
        _, a = api.json("POST", "/api/auditions", {
            "character": 1, "language": "en", "version": v.n, "line": _typed("first"),
        })  # fmt: skip
        _, b = api.json("POST", "/api/auditions", {
            "character": 1, "language": "en", "version": v.n, "line": _typed("second"),
        })  # fmt: skip
        status, got = api.json("POST", f"/api/auditions/{b['job']['id']}/cancel")
        assert status == 200 and got["state"] == "cancelled"
    finally:
        FakeClone.GATE.set()
    assert _poll(api, a["job"]["id"])["state"] == "done"
    assert api.json("GET", "/api/auditions/0000000000000000")[0] == 404


def test_the_server_answers_while_a_render_hangs(api, aud_ws):
    _, v = _draft(aud_ws)
    FakeClone.GATE = threading.Event()
    try:
        _, body = api.json("POST", "/api/auditions", {
            "character": 1, "language": "en", "version": v.n, "line": _typed("stuck"),
        })  # fmt: skip
        _until(lambda: FakeClone.ACTIVE[0] == 1)
        started = time.monotonic()
        assert api.json("GET", "/api/gpu")[0] == 200
        assert api.json("GET", f"/api/auditions/{body['job']['id']}")[1]["state"] == "running"
        assert api.json("GET", "/api/drafts/1/en")[0] == 200
        assert time.monotonic() - started < 5
    finally:
        FakeClone.GATE.set()
    assert _poll(api, body["job"]["id"])["state"] == "done"


def test_old_auditions_on_a_draft_are_served_after_a_restart(api, aud_ws):
    from fukikae.studio.audition import AuditionWorker

    _, v = _draft(aud_ws)
    w = AuditionWorker(aud_ws, _cfg(aud_ws), idle_s=30.0)  # an earlier server's worker
    try:
        a = _done(w, w.submit(1, "en", v.n, _typed("from before"))).audition
        assert a is not None
    finally:
        w.close()
    assert api.request("GET", f"/audition/{a.wav}")[0] == 404  # not yet listed to the page
    assert api.json("GET", "/api/drafts/1/en")[0] == 200
    assert api.request("GET", f"/audition/{a.wav}")[0] == 200


def test_routes_list_the_audition_endpoints():
    from fukikae.studio.server import ROUTES

    paths = {(m, p) for m, p, _ in ROUTES}
    assert {("POST", "/api/auditions"), ("GET", "/api/auditions/<job>"),
            ("POST", "/api/auditions/<job>/cancel"), ("GET", "/api/gpu"),
            ("POST", "/api/gpu/release"), ("GET", "/audition/<key>.wav")} <= paths  # fmt: skip
