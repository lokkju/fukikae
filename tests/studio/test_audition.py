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
    LOAD_GATE: ClassVar[threading.Event | None] = None  # set: loads wait for it

    def __init__(self, options: dict[str, Any], device: str) -> None:  # noqa: ARG002
        self.device = device

    def load(self) -> None:
        self.LOG.append(f"load {self.name}")
        if FakeClone.LOAD_GATE is not None:
            FakeClone.LOAD_GATE.wait(20)

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


class InstructClone(FakeClone):
    name: ClassVar[str] = "instructclone"
    capabilities: ClassVar = frozenset({"emotion_audio", "instruction"})


class StockOk(FakeClone):
    """A backend that would take a stock card (the audition still refuses one, D41)."""

    name: ClassVar[str] = "stockok"
    sources: ClassVar = frozenset({"clone", "stock"})


CLASSES: dict[str, type] = {
    "clonefake": FakeClone, "clonefake2": FakeClone2, "gpuclone": GpuClone,
    "instructclone": InstructClone, "stockok": StockOk,
}  # fmt: skip


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
    FakeClone.LOAD_GATE = None
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
    """The clone check runs on the worker (docker inspect can take 30 s): the first job fails
    as refused, and a submit while the answer is fresh is refused at once."""
    from fukikae.studio.audition import AuditionRefused

    _, v = _draft(aud_ws)
    monkeypatch.setattr("fukikae.cards.auto.clone_problem", lambda *_: "no worker environment")
    got = worker.wait(worker.submit(1, "en", v.n, _real(E1, "c10")), 10)
    assert got.state == "failed" and got.refused is True
    assert "clonefake" in (got.message or "") and "no worker environment" in (got.message or "")
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


# --- review fixes: the GPU lock is never held idle or forever -------------------------------


def _state(w, job: str) -> str | None:
    snap = w.job(job)
    return snap.state if snap is not None else None


def _held_by_us(w) -> bool:
    s = w.status()
    return s.ours and not _probe()


@pytest.fixture
def no_vram(monkeypatch):
    """A card whose free memory never reaches what the model needs, used by pid 999."""
    from fukikae.studio import audition

    monkeypatch.setattr(GpuClone, "vram_mb", 4000)
    monkeypatch.setattr(audition, "free_mb", lambda _i: 100)
    monkeypatch.setattr(audition, "gpu_processes", lambda _i: [(999, "llama-server")])


def test_vram_wait_is_finite_and_lets_go_of_the_lock(aud_ws, no_vram):  # noqa: ARG001
    from fukikae.studio.audition import AuditionWorker

    _, v = _draft(aud_ws, "gpuclone")
    w = AuditionWorker(aud_ws, _cfg(aud_ws), idle_s=30.0, poll_s=0.02, vram_s=0.3)
    try:
        got = w.wait(w.submit(1, "en", v.n, _typed("no room")), 10)
        assert got.state == "failed", got
        assert "4000 MiB" in (got.message or "") and "pid 999" in (got.message or "")
        assert _probe() and w.status().loaded_backend is None
        assert FakeClone.LOG == []
    finally:
        w.close()


@pytest.mark.parametrize("how", ["cancel", "release", "close"])
def test_a_vram_wait_stops_promptly_and_lets_go_of_the_lock(aud_ws, no_vram, how):  # noqa: ARG001
    from fukikae.studio.audition import AuditionWorker

    _, v = _draft(aud_ws, "gpuclone")
    w = AuditionWorker(aud_ws, _cfg(aud_ws), idle_s=30.0, poll_s=0.02, vram_s=600.0)
    try:
        job = w.submit(1, "en", v.n, _typed("no room"))
        _until(lambda: _state(w, job) == "waiting" and _held_by_us(w))
        snap = w.job(job)
        assert snap is not None and "MiB" in (snap.message or "")
        started = time.monotonic()
        {"cancel": lambda: w.cancel(job), "release": w.release, "close": lambda: w.close(5)}[how]()
        assert w.wait(job, 5).state == "cancelled"
        _until(_probe, 5)
        assert time.monotonic() - started < 5 and FakeClone.LOG == []
    finally:
        w.close()


def test_api_release_and_server_close_stop_a_vram_wait(aud_ws, no_vram):  # noqa: ARG001
    from fukikae.studio.server import StudioServer
    from tests.studio.test_server import TOKEN

    _, v = _draft(aud_ws, "gpuclone")
    server = StudioServer(aud_ws, _cfg(aud_ws), port=0, token=TOKEN)
    server.auditions.poll_s = 0.02
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    api = Client(server)
    try:
        body = {"character": 1, "language": "en", "version": v.n, "line": _typed("no room")}
        job = api.json("POST", "/api/auditions", body)[1]["job"]["id"]
        _until(lambda: api.json("GET", f"/api/auditions/{job}")[1]["state"] == "waiting")
        assert api.json("POST", "/api/gpu/release")[0] == 200
        assert _poll(api, job, 5)["state"] == "cancelled"
        _until(_probe, 5)
        job = api.json("POST", "/api/auditions", body)[1]["job"]["id"]
        _until(lambda: api.json("GET", f"/api/auditions/{job}")[1]["state"] == "waiting")
        assert not _probe()
    finally:
        server.shutdown()
        started = time.monotonic()
        server.server_close()
        t.join(5)
    assert _probe() and time.monotonic() - started < 5


def test_server_close_lets_go_of_a_loaded_model(aud_ws):
    from fukikae.studio.server import StudioServer
    from tests.studio.test_server import TOKEN

    _, v = _draft(aud_ws, "gpuclone")
    server = StudioServer(aud_ws, _cfg(aud_ws), port=0, token=TOKEN)
    try:
        _done(server.auditions, server.auditions.submit(1, "en", v.n, _typed("on the card")))
        assert not _probe()
    finally:
        server.server_close()
    assert _probe() and FakeClone.LOG[-1] == "unload gpuclone"


def test_a_cancel_right_after_the_lock_is_taken_lets_go_of_it(aud_ws, worker, monkeypatch):
    """Cancelled after the lock is taken and before the model loads: the lock goes at once,
    and /api/gpu never shows a lock held with nothing loaded."""
    _, v = _draft(aud_ws, "gpuclone")
    real = worker._take_gpu

    def take_then_cancel(job, device):
        real(job, device)
        worker.cancel(job.id)

    monkeypatch.setattr(worker, "_take_gpu", take_then_cancel)
    job = worker.submit(1, "en", v.n, _typed("never loads"))
    assert worker.wait(job, 10).state == "cancelled"
    s = worker.status()
    assert s.loaded_backend is None and s.holder is None and s.ours is False
    assert _probe() and FakeClone.LOG == []


def test_a_failed_load_lets_go_of_the_lock(aud_ws, worker, monkeypatch):
    _, v = _draft(aud_ws, "gpuclone")

    def boom(_self):
        raise RuntimeError("no model files")

    monkeypatch.setattr(GpuClone, "load", boom)
    got = worker.wait(worker.submit(1, "en", v.n, _typed("never loads")), 10)
    assert got.state == "failed" and "no model files" in (got.message or "")
    assert _probe() and worker.status().loaded_backend is None


def test_a_busy_lock_with_no_named_holder_is_tried_again(aud_ws, worker, monkeypatch):
    """flock fails and the file names nobody yet (or the holder is just letting go): wait."""
    from fukikae.studio import audition

    _, v = _draft(aud_ws, "gpuclone")
    real, busy = gpulock.gpu_lock, []

    def flaky(*a, **kw):
        if len(busy) < 2:
            busy.append(1)
            raise gpulock.GpuBusy("GPU 0 is held by another fukikae job")
        return real(*a, **kw)

    monkeypatch.setattr(audition.gpulock, "gpu_lock", flaky)
    monkeypatch.setattr(audition.gpulock, "current_holder", lambda _i: None)
    _done(worker, worker.submit(1, "en", v.n, _typed("third time")))
    assert len(busy) == 2


def test_a_leftover_holder_fails_the_job(aud_ws, worker, monkeypatch):
    from fukikae.studio import audition

    _, v = _draft(aud_ws, "gpuclone")

    def leftover(*_a, **_k):
        raise gpulock.GpuLeftover("GPU 0 is locked by what is left of pid 77 (tts x)")

    monkeypatch.setattr(audition.gpulock, "gpu_lock", leftover)
    got = worker.wait(worker.submit(1, "en", v.n, _typed("blocked")), 10)
    assert got.state == "failed" and "left of pid 77" in (got.message or "")


def _dead_pid() -> int:
    import subprocess

    p = subprocess.Popen(["true"])  # noqa: S607
    p.wait()
    return p.pid


@pytest.fixture
def leftover_holder(aud_ws):  # noqa: ARG001 (the lock dir)
    """A process holding card 0's lock under a line naming a pid that has exited: what a
    killed run's surviving worker child leaves."""
    import subprocess
    import sys

    dead = _dead_pid()
    code = (
        "import fcntl, os, sys, time\n"
        "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
        "fcntl.flock(fd, fcntl.LOCK_EX)\n"
        "os.ftruncate(fd, 0)\n"
        "os.pwrite(fd, (sys.argv[2] + ' tts indextts2\\n').encode(), 0)\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    p = subprocess.Popen([sys.executable, "-c", code, str(gpulock.lock_path(0)), str(dead)],
                         stdout=subprocess.PIPE, text=True)  # fmt: skip
    try:
        assert p.stdout is not None and p.stdout.readline().strip() == "held"
        yield dead, p.pid
    finally:
        p.kill()
        p.wait()


def test_gpulock_names_a_leftover_holder_as_such(leftover_holder):
    dead, _ = leftover_holder
    with pytest.raises(gpulock.GpuLeftover, match=f"left of pid {dead}"):  # noqa: SIM117
        with gpulock.gpu_lock(0, "probe", timeout_s=0):
            pass


def test_gpu_status_reports_a_leftover_holder(aud_ws, worker, leftover_holder):  # noqa: ARG001
    dead, _ = leftover_holder
    s = worker.status()
    assert s.leftover is True and s.holder_pid == dead and s.ours is False
    assert s.holder is not None and s.holder.startswith(f"leftover of pid {dead}")
    assert "kill it or wait" in s.holder


def test_a_stale_line_with_no_lock_is_nobody(aud_ws, worker):  # noqa: ARG001
    """A holder killed outright with no child left: its line stays, the lock is free."""
    gpulock.lock_path(0).write_text(f"{_dead_pid()} tts indextts2\n", encoding="utf-8")
    s = worker.status()
    assert s.holder is None and s.leftover is False


def test_release_stops_a_job_still_preparing(aud_ws, worker, monkeypatch):
    _, v = _draft(aud_ws, "gpuclone")
    real, inside, go = worker._prepare, threading.Event(), threading.Event()

    def slow(job, version):
        inside.set()
        go.wait(10)
        return real(job, version)

    monkeypatch.setattr(worker, "_prepare", slow)
    job = worker.submit(1, "en", v.n, _typed("never renders"))
    assert inside.wait(5)
    worker.release()
    go.set()
    assert worker.wait(job, 10).state == "cancelled"
    _until(lambda: worker.status().loaded_backend is None)
    assert _probe() and not any(x.startswith("render") for x in FakeClone.LOG)


@pytest.mark.parametrize("how", ["cancel", "release"])
def test_a_job_stopped_while_loading_never_renders(aud_ws, worker, how):
    _, v = _draft(aud_ws, "gpuclone")
    FakeClone.LOAD_GATE = threading.Event()
    job = worker.submit(1, "en", v.n, _typed("never renders"))
    _until(lambda: worker.job(job).state == "loading")
    worker.cancel(job) if how == "cancel" else worker.release()
    FakeClone.LOAD_GATE.set()
    assert worker.wait(job, 10).state == "cancelled"
    assert not any(x.startswith("render") for x in FakeClone.LOG)
    if how == "release":
        _until(lambda: worker.status().loaded_backend is None)
        assert _probe()


def test_a_cached_line_on_another_backend_unloads_nothing(aud_ws, worker):
    _, v1 = _draft(aud_ws)
    _, v2 = _draft(aud_ws, "clonefake2")
    _done(worker, worker.submit(1, "en", v2.n, _typed("second")))
    _done(worker, worker.submit(1, "en", v1.n, _typed("first")))
    before = list(FakeClone.LOG)
    again = _done(worker, worker.submit(1, "en", v2.n, _typed("second"))).audition
    assert again is not None and again.renders == 0
    assert before == FakeClone.LOG  # clonefake stays loaded; clonefake2 never loads
    assert worker.status().loaded_backend == "clonefake"


def test_an_audition_is_recorded_only_on_the_version_it_rendered(aud_ws, worker):
    from fukikae.studio.drafts import load_draft, save_draft

    d, v = _draft(aud_ws)
    FakeClone.GATE = threading.Event()
    try:
        job = worker.submit(1, "en", v.n, _typed("which version"))
        _until(lambda: FakeClone.ACTIVE[0] == 1)
        # the draft was replaced meanwhile: its version 1 is now another card
        other = v.model_copy(update={"card_id": "0123456789abcdef"})
        save_draft(aud_ws, d.model_copy(update={"versions": [other]}))
    finally:
        FakeClone.GATE.set()
    _done(worker, job)
    draft = load_draft(aud_ws, 1, "en")
    assert draft is not None and draft.versions[0].auditions == []


def test_a_recording_racing_a_new_version_keeps_both(api, aud_ws):
    from fukikae.studio.drafts import load_draft

    _, v = _draft(aud_ws)
    FakeClone.GATE = threading.Event()
    try:
        _, body = api.json("POST", "/api/auditions", {
            "character": 1, "language": "en", "version": v.n, "line": _typed("racing"),
        })  # fmt: skip
        _until(lambda: FakeClone.ACTIVE[0] == 1)
        refs = [r.model_dump(mode="json") for r in v.references]
        new = {"references": refs, "backend": "clonefake2", "params": {}}
        status, made = api.json("POST", "/api/drafts/1/en/versions", new)
        assert status == 200, made
    finally:
        FakeClone.GATE.set()
    assert _poll(api, body["job"]["id"])["state"] == "done"
    draft = load_draft(aud_ws, 1, "en")
    assert draft is not None and [x.n for x in draft.versions] == [1, 2]
    assert len(draft.versions[0].auditions) == 1 and draft.versions[1].auditions == []


def test_the_cache_file_mode_is_left_alone(aud_ws, worker):
    """The audition wav shares the cache's inode: never chmod it (the auditions dir is 0700),
    and the dir is made owner-only again on every use."""
    _, v = _draft(aud_ws)
    a = _done(worker, worker.submit(1, "en", v.n, _typed("shared inode"))).audition
    cached = aud_ws.root / ".fukikae" / "tts-cache" / a.wav
    cached.chmod(0o644)
    out = aud_ws.root / "studio" / "auditions"
    out.chmod(0o755)
    _done(worker, worker.submit(1, "en", v.n, _typed("shared inode")))
    assert cached.stat().st_mode & 0o777 == 0o644
    assert out.stat().st_mode & 0o777 == 0o700


def test_unreferenced_emotion_clips_are_pruned_when_a_draft_is_saved(aud_ws, worker):
    _, v = _draft(aud_ws)
    ref = aud_ws.root / "studio" / "auditions" / "ref"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "c999.wav").write_bytes(b"stale")
    _done(worker, worker.submit(1, "en", v.n, _real(E1, "c10")))
    assert sorted(p.name for p in ref.iterdir()) == ["c10.wav"]


def test_the_clone_check_is_not_run_per_request(aud_ws, worker, monkeypatch):
    calls: list[float] = []

    def slow(*_a):
        calls.append(time.monotonic())
        time.sleep(0.3)

    monkeypatch.setattr("fukikae.cards.auto.clone_problem", slow)
    _, v = _draft(aud_ws)
    started = time.monotonic()
    jobs = [worker.submit(1, "en", v.n, _typed(f"line {i}")) for i in range(3)]
    assert time.monotonic() - started < 0.3  # the HTTP thread never ran it
    for j in jobs:
        _done(worker, j)
    assert len(calls) == 1  # once per backend while the answer is fresh


def test_a_typed_line_takes_its_prompt_cues_direction(aud_ws, worker):
    path = aud_ws.work_dir(E1) / "06-translate" / "cues.en.json"
    doc = read_json(path, DubCues)
    whisper = {"direction": "whispering"}
    cues = [c.model_copy(update={"delivery": c.delivery.model_copy(update=whisper)})
            if c.id == "c10" else c for c in doc.cues]  # fmt: skip
    write_json(path, doc.model_copy(update={"cues": cues}))
    _, v = _draft(aud_ws, "instructclone")
    line = _typed("Hello there.", {"episode": E1, "cue_id": "c10"})
    a = _done(worker, worker.submit(1, "en", v.n, line)).audition
    assert a is not None and a.passed.get("instruction") == "whispering"
    real = _done(worker, worker.submit(1, "en", v.n, _real(E1, "c10"))).audition
    assert real is not None and real.passed.get("instruction") == "whispering"


def test_a_stock_card_is_never_auditioned(aud_ws, worker, monkeypatch):
    from fukikae.cards.card import Card
    from fukikae.studio.audition import AuditionRefused
    from fukikae.studio.drafts import save_draft

    monkeypatch.setattr(tts, "check_stock_voice", lambda *_a: None)
    stock = CardStore(aud_ws.root).put(Card(
        target_language="en",
        voice=Voice(backend="stockok", model_revision="rev-9", source="stock", voice="am_echo"),
    ))  # fmt: skip
    d, v = _draft(aud_ws)
    edited = v.model_copy(update={"card_id": stock, "backend": "stockok"})
    save_draft(aud_ws, d.model_copy(update={"versions": [edited]}))
    with pytest.raises(AuditionRefused, match=r"stock.*D41"):
        worker.submit(1, "en", v.n, _typed("never"))
    assert FakeClone.LOG == []


def test_the_operators_gender_gates_a_typed_lines_prompt(aud_ws, worker):
    with Registry(aud_ws.registry_path) as reg:
        reg.set_character_gender(1, "m")
    _, v = _draft(aud_ws)
    line = _typed("Hello there.", {"episode": E1, "cue_id": "c10"})  # her f line, an m voice
    a = _done(worker, worker.submit(1, "en", v.n, line)).audition
    assert a is not None and a.gender_gate is True and "emotion_audio" not in a.passed


def test_a_typed_prompt_from_an_episode_it_never_speaks_in_uses_the_season(aud_ws, worker):
    """Character 5 speaks only in E2 (f): a prompt from Natsu's E1 line (m) is dropped, one
    from Lucy's (f) is passed."""
    from fukikae.studio.drafts import open_draft, save_draft

    d = open_draft(aud_ws, _cfg(aud_ws), 5, "en")
    save_draft(aud_ws, d)
    n = d.versions[0].n
    m = _done(worker, worker.submit(5, "en", n, _typed("Hi.", {"episode": E1, "cue_id": "c20"})))
    assert m.audition is not None and m.audition.gender_gate is True
    f = _done(worker, worker.submit(5, "en", n, _typed("Hi.", {"episode": E1, "cue_id": "c10"})))
    assert f.audition is not None and "emotion_audio" in f.audition.passed


def test_the_queue_holds_exactly_sixteen(aud_ws, worker):
    from fukikae.studio.audition import MAX_QUEUE, QueueFull

    assert MAX_QUEUE == 16
    _, v = _draft(aud_ws)
    FakeClone.GATE = threading.Event()
    try:
        worker.submit(1, "en", v.n, _typed("running"))
        _until(lambda: FakeClone.ACTIVE[0] == 1)
        for i in range(16):
            worker.submit(1, "en", v.n, _typed(f"line {i}"))
        assert worker.status().queued == 16
        with pytest.raises(QueueFull):
            worker.submit(1, "en", v.n, _typed("one too many"))
    finally:
        FakeClone.GATE.set()


def test_a_full_queue_is_503_over_http(api, aud_ws):
    _, v = _draft(aud_ws)
    FakeClone.GATE = threading.Event()
    body = {"character": 1, "language": "en", "version": v.n, "line": _typed("x")}
    try:
        assert api.json("POST", "/api/auditions", body)[0] == 202
        _until(lambda: FakeClone.ACTIVE[0] == 1)
        for _ in range(16):
            assert api.json("POST", "/api/auditions", body)[0] == 202
        status, headers, _ = api.request("POST", "/api/auditions", body)
        assert status == 503 and headers.get("retry-after") == "2"
    finally:
        FakeClone.GATE.set()


def test_a_range_past_the_end_of_an_audition_is_416(api, aud_ws):
    _, v = _draft(aud_ws)
    _, body = api.json("POST", "/api/auditions", {
        "character": 1, "language": "en", "version": v.n, "line": _typed("hello"),
    })  # fmt: skip
    done = _poll(api, body["job"]["id"])
    size = (aud_ws.root / "studio" / "auditions" / done["audition"]["wav"]).stat().st_size
    status, headers, _ = api.request("GET", done["wav_url"], headers={"Range": f"bytes={size}-"})
    assert status == 416 and headers.get("content-range") == f"bytes */{size}"
