"""The out-of-process TTS worker: protocol, lifecycle and validation, with a fake impl."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fukikae.backends import loader
from fukikae.backends.tts import remote, worker
from fukikae.backends.tts.remote import BackendError, WorkerSynthesizer
from fukikae.cards.card import Reference, Voice

ROOT = Path(__file__).resolve().parents[2]


class FakeWorker(WorkerSynthesizer):
    name: ClassVar[str] = "faketts"
    extra: ClassVar[str] = "faketts"
    languages: ClassVar = frozenset({"en"})
    vram_mb: ClassVar[int] = 0
    sources: ClassVar = frozenset({"clone"})
    voice_keys: ClassVar = frozenset({"seed", "emotion_strength"})
    capabilities: ClassVar = frozenset({"emotion_audio"})
    sample_rate: ClassVar[int] = 16000
    speech_cps: ClassVar = {"en": 15.0}
    model: ClassVar[str] = "fake/model"
    revision: ClassVar[str] = "r1"
    model_revision: str = "r1"
    load_timeout_s: ClassVar[float] = 30.0
    synth_timeout_s: ClassVar[float] = 30.0

    def interpreter(self) -> Path:
        return Path(sys.executable)


@pytest.fixture(autouse=True)
def fake_impl(monkeypatch):
    monkeypatch.setenv("FUKIKAE_TTS_FAKE_IMPL", "tests.fixtures.fake_tts_impl")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setattr(remote, "leftover_workers", lambda _gpu: [])


def _ref(tmp_path: Path) -> Path:
    p = tmp_path / "ref.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return p


def _voice() -> Voice:
    ref = Reference(clip="sha256:" + "a" * 64, transcript="なんだよ", language="ja")
    return Voice(backend="faketts", model_revision="r1", source="clone", seed=7, references=(ref,))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); a reaped child does not exist at all
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def test_round_trip_synthesize(tmp_path):
    ref = _ref(tmp_path)
    w = FakeWorker({}, "cpu")
    w.load()
    try:
        out = tmp_path / "out.wav"
        seconds = w.synthesize(
            "echo", _voice(), "en", out, resolve=lambda _r: ref, emotion_audio=ref,
            emotion_strength=0.5,
        )  # fmt: skip
        assert out.is_file() and seconds == pytest.approx(0.05 * 4, abs=1e-3)
        seen = json.loads((tmp_path / "out.wav.json").read_text())
        assert seen["voice"]["references"][0]["clip"] == str(ref)
        assert seen["voice"]["seed"] == 7 and seen["language"] == "en"
        assert seen["emotion_audio"] == str(ref) and seen["emotion_strength"] == 0.5
        assert seen["out"] == str(out)
        # a second line on the same worker
        assert w.synthesize("hello", _voice(), "en", tmp_path / "b.wav", resolve=lambda _r: ref)
    finally:
        w.unload()


def test_stdout_noise_from_library_does_not_break_protocol(tmp_path):
    """The fake impl prints to stdout on load and on every line; the protocol survives."""
    w = FakeWorker({}, "cpu")
    w.load()
    try:
        for i in range(3):
            w.synthesize(
                "hi", _voice(), "en", tmp_path / f"{i}.wav", resolve=lambda _r: _ref(tmp_path)
            )
    finally:
        w.unload()


def test_worker_error_raises_with_stderr_tail(tmp_path):
    w = FakeWorker({}, "cpu")
    w.load()
    try:
        with pytest.raises(BackendError, match="boom from impl") as e:
            w.synthesize(
                "boom", _voice(), "en", tmp_path / "x.wav", resolve=lambda _r: _ref(tmp_path)
            )
        assert "model exploded" in str(e.value)
        # an impl error leaves the worker usable
        w.synthesize("ok", _voice(), "en", tmp_path / "y.wav", resolve=lambda _r: _ref(tmp_path))
    finally:
        w.unload()


def test_worker_death_raises_and_is_reaped(tmp_path):
    w = FakeWorker({}, "cpu")
    w.load()
    pid = w.pid
    assert pid is not None
    with pytest.raises(BackendError, match="dying on purpose"):
        w.synthesize("die", _voice(), "en", tmp_path / "x.wav", resolve=lambda _r: _ref(tmp_path))
    assert not _alive(pid)
    w.unload()  # safe after a death


def test_unload_reaps_the_worker():
    w = FakeWorker({}, "cpu")
    w.load()
    pid = w.pid
    assert pid is not None and _alive(pid)
    w.unload()
    assert not _alive(pid) and w.pid is None
    w.unload()  # idempotent


def test_a_hung_worker_is_killed_after_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeWorker, "synth_timeout_s", 1.0)
    w = FakeWorker({}, "cpu")
    w.load()
    pid = w.pid
    assert pid is not None
    with pytest.raises(BackendError, match="did not answer"):
        w.synthesize("hang", _voice(), "en", tmp_path / "x.wav", resolve=lambda _r: _ref(tmp_path))
    assert not _alive(pid)
    w.unload()


def test_output_is_checked_not_trusted(tmp_path):
    w = FakeWorker({}, "cpu")
    w.load()
    try:
        with pytest.raises(BackendError, match="sample rate"):
            w.synthesize(
                "wrong-rate", _voice(), "en", tmp_path / "a.wav", resolve=lambda _r: _ref(tmp_path)
            )
        with pytest.raises(BackendError, match="seconds"):
            w.synthesize(
                "lie", _voice(), "en", tmp_path / "b.wav", resolve=lambda _r: _ref(tmp_path)
            )
    finally:
        w.unload()


class ScriptWorker(FakeWorker):
    SCRIPT: ClassVar[str] = ""

    def command(self) -> list[str]:
        return [sys.executable, "-c", self.SCRIPT]


_READ = "import sys; sys.stdin.readline(); "
_HOLD = "; sys.stdout.flush(); sys.stdin.read()"


@pytest.mark.parametrize(
    "script,match",
    [
        (_READ + "print('not json')" + _HOLD, "protocol"),
        (_READ + "print('[1, 2]')" + _HOLD, "protocol"),
        (_READ + "sys.stdout.write('x' * 3000000)" + _HOLD, "longer than"),
        (_READ + 'print(\'{"ok": true, "protocol": 99}\')' + _HOLD, "protocol"),
    ],
)
def test_malformed_worker_output_is_refused_and_reaped(monkeypatch, script, match):
    monkeypatch.setattr(ScriptWorker, "SCRIPT", script)
    w = ScriptWorker({}, "cpu")
    with pytest.raises(BackendError, match=match):
        w.load()
    assert w.pid is None


def test_missing_worker_env_names_the_setup_command(monkeypatch, tmp_path):
    monkeypatch.setattr(FakeWorker, "interpreter", WorkerSynthesizer.interpreter)
    monkeypatch.setenv("FUKIKAE_WORKER_CHECKOUT", str(tmp_path))
    w = FakeWorker({}, "cpu")
    with pytest.raises(loader.BackendUnavailable, match="uv sync") as e:
        w.load()
    assert ".venvs/faketts" in str(e.value)


def test_a_leftover_worker_on_the_gpu_refuses_to_start(monkeypatch):
    monkeypatch.setattr(remote, "nvml_index", lambda _dev: 0)
    monkeypatch.setattr(
        remote, "leftover_workers", lambda _gpu: [(4242, "python -m fukikae.backends.tts.worker x")]
    )
    started: list[Any] = []
    monkeypatch.setattr(remote.subprocess, "Popen", lambda *a, **_k: started.append(a))
    w = FakeWorker({}, "cuda:0")
    with pytest.raises(BackendError, match="pid 4242"):
        w.load()
    assert started == []


def test_gpu_worker_sees_only_its_card(monkeypatch, tmp_path):
    monkeypatch.setattr(remote, "nvml_index", lambda _dev: 0)
    w = FakeWorker({}, "cuda:0")
    w.load()
    try:
        w.synthesize("echo", _voice(), "en", tmp_path / "o.wav", resolve=lambda _r: _ref(tmp_path))
        seen = json.loads((tmp_path / "o.wav.json").read_text())
        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert seen["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert seen["device"] == "cuda:0"
    finally:
        w.unload()


def test_leftover_workers_reads_the_command_line(monkeypatch):
    monkeypatch.undo()  # the real leftover_workers
    me = os.getpid()
    monkeypatch.setattr(remote, "gpu_processes", lambda _gpu: [(me, "python")])
    monkeypatch.setattr(
        remote, "_cmdline", lambda _pid: "python -m fukikae.backends.tts.worker indextts2"
    )
    assert remote.leftover_workers(0) == [(me, "python -m fukikae.backends.tts.worker indextts2")]
    monkeypatch.setattr(remote, "_cmdline", lambda _pid: "python -m vllm")
    assert remote.leftover_workers(0) == []


# ---- the worker side --------------------------------------------------------------------


def _synth_req(tmp_path: Path, **kw: Any) -> dict[str, Any]:
    ref = _ref(tmp_path)
    req = {
        "op": "synthesize",
        "text": "Hello.",
        "voice": {"source": "clone", "references": [{"clip": str(ref), "transcript": "x"}]},
        "language": "en",
        "out": str(tmp_path / "o.wav"),
        "emotion_audio": None,
        "emotion_vector": None,
        "instruction": None,
        "emotion_strength": 1.0,
        "target_seconds": None,
    }
    req.update(kw)
    return req


def test_worker_validates_requests(tmp_path):
    assert worker.validate_request(_synth_req(tmp_path))["text"] == "Hello."
    bad = [
        {"op": "format_disk"},
        _synth_req(tmp_path, out="relative.wav"),
        _synth_req(tmp_path, out=str(tmp_path / "missing" / "o.wav")),
        _synth_req(tmp_path, text="x" * (worker.MAX_TEXT + 1)),
        _synth_req(tmp_path, emotion_strength=1.5),
        _synth_req(tmp_path, emotion_strength=float("nan")),
        _synth_req(tmp_path, emotion_audio="/nonexistent/emo.wav"),
        _synth_req(tmp_path, target_seconds=-1),
        _synth_req(tmp_path, voice={"source": "clone", "references": [{"clip": "rel.wav"}]}),
        _synth_req(tmp_path, extra_key=1),
    ]
    for req in bad:
        with pytest.raises(worker.RequestError):
            worker.validate_request(req)


def test_worker_answers_bad_lines_with_errors_and_keeps_going():
    env = {**os.environ, "FUKIKAE_TTS_FAKE_IMPL": "tests.fixtures.fake_tts_impl"}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    lines = ["not json", json.dumps({"op": "nope"}), json.dumps({"op": "hello"}), ""]
    r = subprocess.run(
        [sys.executable, "-m", "fukikae.backends.tts.worker", "faketts"],
        input="\n".join(lines), capture_output=True, text=True, env=env, timeout=60, check=False,
    )  # fmt: skip
    answers = [json.loads(x) for x in r.stdout.splitlines()]
    assert [a["ok"] for a in answers] == [False, False, True]
    assert answers[2]["protocol"] == worker.PROTOCOL
    assert r.returncode == 0


def test_worker_impl_name_must_be_an_identifier():
    env = {k: v for k, v in os.environ.items() if k != "FUKIKAE_TTS_FAKE_IMPL"}
    env["PYTHONPATH"] = str(ROOT / "src")
    r = subprocess.run(
        [sys.executable, "-m", "fukikae.backends.tts.worker", "../../etc"],
        input="", capture_output=True, text=True, env=env, timeout=60, check=False,
    )  # fmt: skip
    assert r.returncode != 0 and r.stdout == ""


@pytest.mark.gpu
def test_real_indextts2_renders_a_clone_line(tmp_path, monkeypatch):
    """Needs `.venvs/indextts2` (README, "Cloning backends") and GPU 0."""
    from fukikae.backends.tts.indextts2 import IndexTts2Synthesizer

    monkeypatch.delenv("FUKIKAE_TTS_FAKE_IMPL")

    ref = tmp_path / "ref.wav"
    import numpy as np
    from scipy.io import wavfile

    t = np.arange(16000 * 4) / 16000
    wavfile.write(ref, 16000, (0.2 * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2"))
    synth = IndexTts2Synthesizer({}, "cuda:0")
    synth.load()
    try:
        v = Voice(
            backend="indextts2",
            model_revision=synth.model_revision,
            source="clone",
            seed=1234,
            references=(Reference(clip="sha256:" + "b" * 64, transcript="", language="ja"),),
        )
        out = tmp_path / "o.wav"
        seconds = synth.synthesize("Hello there, Lucy.", v, "en", out, resolve=lambda _r: ref)
        assert 0.5 < seconds < 6
    finally:
        synth.unload()
