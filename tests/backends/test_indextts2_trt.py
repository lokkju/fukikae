"""IndexTTS2 on TensorRT (D28 lever 2): declarations, the container command and the staging of
files into it, with a fake ``docker`` that runs the worker locally with the fake impl."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from fukikae import gpulock
from fukikae.backends import loader
from fukikae.backends.loader import BackendUnavailable
from fukikae.backends.tts import impl_indextts2 as base_impl
from fukikae.backends.tts import impl_indextts2_trt as impl
from fukikae.backends.tts import indextts2_trt as trt
from fukikae.backends.tts import remote
from fukikae.backends.tts.indextts2 import IndexTts2Synthesizer
from fukikae.backends.tts.indextts2_trt import IndexTts2TrtSynthesizer
from fukikae.cards.card import Card, Reference, Voice, validate_for_backend
from fukikae.config import ConfigError

ROOT = Path(__file__).resolve().parents[2]
FAKE_DOCKER = ROOT / "tests" / "fixtures" / "fake_docker.py"


def _card(**voice: object) -> Card:
    ref = Reference(clip="sha256:" + "c" * 64, transcript="なんだよ", language="ja")
    v = Voice(
        backend="indextts2",
        model_revision=IndexTts2Synthesizer.model_revision,
        source="clone",
        seed=1234,
        references=(ref,),
    )
    return Card(target_language="en", voice=replace(v, **voice))  # type: ignore[arg-type]


# ---- declarations --------------------------------------------------------------------------


def test_renders_indextts2_cards_with_the_same_capabilities():
    c, p = IndexTts2TrtSynthesizer, IndexTts2Synthesizer
    assert c.name == "indextts2_trt" and c.card_backend == "indextts2"
    assert c.sources == p.sources and c.voice_keys == p.voice_keys
    assert c.capabilities == p.capabilities == frozenset({"emotion_audio"})
    assert c.languages == p.languages and c.sample_rate == p.sample_rate
    assert c.speech_cps == p.speech_cps
    # the cards' pin is the model's, so every IndexTTS2 card stays valid
    assert c.model_revision == p.model_revision
    # its own identity, so its renders never share a cache entry with PyTorch's
    assert c.revision != p.revision and "trt" in c.revision
    assert c.devices == frozenset({"cuda"})


def test_existing_indextts2_cards_validate_against_it():
    validate_for_backend(_card(), IndexTts2TrtSynthesizer)
    validate_for_backend(_card(emotion_strength=0.5, temperature=0.7), IndexTts2TrtSynthesizer)


def test_its_only_stage_option_is_fp32():
    IndexTts2TrtSynthesizer({}, "cuda:0")
    IndexTts2TrtSynthesizer({"fp32": ["dit", "campplus"]}, "cuda:0")
    IndexTts2TrtSynthesizer({"fp32": []}, "cuda:0")
    with pytest.raises(ConfigError, match="fp16"):
        IndexTts2TrtSynthesizer({"fp16": True}, "cuda:0")


@pytest.mark.parametrize(
    ("value", "match"),
    [(["gpt"], "gpt"), (["dit", "ditt"], "ditt"), ("dit", "list"), ([1], "list")],
)
def test_fp32_names_only_the_plain_tensorrt_engines(value, match):
    with pytest.raises(ConfigError, match=match) as e:
        IndexTts2TrtSynthesizer({"fp32": value}, "cuda:0")
    assert "speech_semantic_encoder" in str(e.value)  # it lists what it takes


def test_the_conditioners_are_the_engines_before_the_gpt_and_the_decoder():
    assert set(impl.CONDITIONERS) == set(impl.EXPORTS) - {"dit", "bigvgan"}
    assert len(impl.CONDITIONERS) == 7


def test_registered_as_a_tts_backend():
    assert loader.backend_class("tts", "indextts2_trt") is IndexTts2TrtSynthesizer


def test_impl_shares_the_pytorch_pins_and_is_stdlib_only_at_module_level():
    assert impl.REVISION == base_impl.REVISION and impl.CODE_REVISION == base_impl.CODE_REVISION
    code = (
        "import sys; import fukikae.backends.tts.impl_indextts2_trt; "
        "print(sorted(m for m in ('numpy', 'torch', 'tensorrt', 'tensorrt_llm') "
        "if m in sys.modules))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )  # fmt: skip
    assert r.stdout.strip() == "[]"


def test_dockerfile_clones_upstream_at_the_code_revision():
    text = (ROOT / "workers" / "indextts2_trt" / "Dockerfile").read_text()
    assert f"index-tts.git#{base_impl.CODE_REVISION}" in text
    # every base image pinned by digest
    froms = [ln for ln in text.splitlines() if ln.startswith("FROM ")]
    assert froms and all("@sha256:" in ln for ln in froms)
    assert (ROOT / "workers" / "indextts2_trt" / "uv.lock").is_file()
    # every apt package pinned to a version
    joined = text.replace("\\\n", " ")
    install = next(ln for ln in joined.splitlines() if "apt-get install" in ln)
    packages = install.split("--no-install-recommends", 1)[1].split("&&", 1)[0].split()
    assert packages and all("=" in p for p in packages), packages


# ---- the hub shim --------------------------------------------------------------------------


def test_hub_shim_resolves_main_to_the_pins(tmp_path):
    real = tmp_path / "real"
    pins = [(base_impl.REPO, base_impl.REVISION)]
    pins += [(r, rev) for r, rev, _ in base_impl.AUX.values()]
    for repo, rev in pins:
        snap = real / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(repo)
    shim = tmp_path / "shim"
    impl.make_hub_shim(real, shim)
    for repo, rev in pins:
        folder = shim / ("models--" + repo.replace("/", "--"))
        assert (folder / "refs" / "main").read_text() == rev
        assert (folder / "snapshots" / rev / "config.json").read_text() == repo
    impl.make_hub_shim(real, shim)  # again, over the first


def test_hub_shim_names_the_setup_when_a_pin_is_missing(tmp_path):
    with pytest.raises(RuntimeError, match="fukikae backends setup indextts2_trt"):
        impl.make_hub_shim(tmp_path / "empty", tmp_path / "shim")


# ---- the container -------------------------------------------------------------------------


@pytest.fixture
def setup_done(monkeypatch, tmp_path):
    """A checkout with the worker project, an image that exists and built engines."""
    checkout = tmp_path / "checkout"
    proj = checkout / "workers" / "indextts2_trt"
    proj.mkdir(parents=True)
    (proj / "uv.lock").write_text("version = 1\n")
    (proj / "Dockerfile").write_text("FROM scratch@sha256:00\n")
    monkeypatch.setenv("FUKIKAE_WORKER_CHECKOUT", str(checkout))
    cache = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    synth = IndexTts2TrtSynthesizer({}, "cuda:0")
    engines = synth.engine_dir()
    engines.mkdir(parents=True)
    (engines / impl.MANIFEST).write_text(
        json.dumps({"build_id": "b1d0000000ab", "compute_capability": "8.6"})
    )
    monkeypatch.setattr(trt, "DOCKER", str(FAKE_DOCKER))
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_LOG", str(tmp_path / "docker.log"))
    # the fake docker client's own settings, as a real client's DOCKER_HOST would reach it
    monkeypatch.setattr(trt, "DOCKER_ENV", trt.DOCKER_ENV | FAKE_CLIENT)
    monkeypatch.setattr(IndexTts2TrtSynthesizer, "gpu_compute_capability", lambda _self, _i: "8.6")
    monkeypatch.setattr(gpulock, "leftover_jobs", lambda _gpu: [])
    return checkout


FAKE_CLIENT = frozenset({
    "FUKIKAE_FAKE_DOCKER_PYTHON", "FUKIKAE_FAKE_DOCKER_PYTHONPATH", "FUKIKAE_FAKE_DOCKER_LOG",
    "FUKIKAE_FAKE_DOCKER_PS", "FUKIKAE_FAKE_DOCKER_RM_RC", "FUKIKAE_FAKE_DOCKER_STUCK",
    "FUKIKAE_FAKE_DOCKER_DAEMON", "FUKIKAE_FAKE_DOCKER_IMAGE", "FUKIKAE_FAKE_DOCKER_INFO",
})  # fmt: skip


def _docker_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "docker.log"
    if not log.is_file():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def _flag(argv: list[str], flag: str) -> list[str]:
    """Every value ``flag`` takes in ``argv``."""
    return [b for a, b in zip(argv, argv[1:], strict=False) if a == flag]


def _wav(path: Path, seconds: float = 1.0, rate: int = 22050) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(seconds * rate))
    return path


def test_image_tag_and_engine_dir_follow_the_project(setup_done):
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    lock = s.project_digest()
    assert lock is not None and s.image() == f"fukikae-indextts2-trt:{lock}"
    assert s.engine_dir().name == lock
    # a changed Dockerfile is a new image, new engines and a new identity
    (setup_done / "workers" / "indextts2_trt" / "Dockerfile").write_text("FROM other@sha256:1\n")
    assert IndexTts2TrtSynthesizer({}, "cuda:0").project_digest() != lock


@pytest.mark.usefixtures("setup_done")
def test_the_engine_build_is_in_the_identity():
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    assert s.lock_revision().endswith("-b1d0000000ab")  # type: ignore[union-attr]


def _variant_manifest(s: IndexTts2TrtSynthesizer, fp32: list[str], build_id: str) -> None:
    s.engine_dir().mkdir(parents=True, exist_ok=True)
    (s.engine_dir() / impl.MANIFEST).write_text(
        json.dumps({"build_id": build_id, "compute_capability": "8.6", "fp32": fp32})
    )


@pytest.mark.usefixtures("setup_done")
def test_fp32_engines_live_apart_and_never_share_an_identity_with_fp16():
    fp16 = IndexTts2TrtSynthesizer({}, "cuda:0")
    fp32 = IndexTts2TrtSynthesizer({"fp32": ["dit", "campplus"]}, "cuda:0")
    digest = fp16.project_digest()
    assert fp16.engine_dir().name == digest  # the default set stays where it was
    assert IndexTts2TrtSynthesizer({"fp32": []}, "cuda:0").engine_dir() == fp16.engine_dir()
    assert fp32.engine_dir() != fp16.engine_dir()
    assert fp32.engine_dir().parent == fp16.engine_dir().parent
    assert fp32.engine_dir().name.startswith(f"{digest}-fp32-")
    # order and repeats don't make another set
    same = IndexTts2TrtSynthesizer({"fp32": ["campplus", "dit", "dit"]}, "cuda:0")
    assert same.engine_dir() == fp32.engine_dir()
    assert IndexTts2TrtSynthesizer({"fp32": ["dit"]}, "cuda:0").engine_dir() != fp32.engine_dir()
    # even the same build id (it can't happen, but the key mustn't rest on it) keeps them apart
    _variant_manifest(fp32, ["campplus", "dit"], "b1d0000000ab")
    assert fp32.lock_revision() != fp16.lock_revision()
    assert "fp32" in fp32.lock_revision()  # type: ignore[operator]
    assert "fp32" not in fp16.lock_revision()  # type: ignore[operator]
    assert fp32.environment_problem() is None


@pytest.mark.usefixtures("setup_done")
def test_engines_built_with_other_fp32_engines_are_refused():
    s = IndexTts2TrtSynthesizer({"fp32": ["dit"]}, "cuda:0")
    _variant_manifest(s, ["bigvgan"], "c0ffee000000")
    assert "fp32" in (s.environment_problem() or "")
    # a default set whose manifest says it has fp32 engines is refused as well
    d = IndexTts2TrtSynthesizer({}, "cuda:0")
    _variant_manifest(d, ["dit"], "c0ffee000000")
    assert "fp32" in (d.environment_problem() or "")


@pytest.mark.usefixtures("setup_done")
def test_command_runs_the_worker_in_an_isolated_container(tmp_path):
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    s.gpu_index = 0
    s._cwd = tmp_path / "stage"
    cmd = s.command()
    s.gpu_index = 3
    assert f"-e {gpulock.GPU_INDEX_ENV}=3" in " ".join(s.command())
    assert cmd[:4] == [str(FAKE_DOCKER), "run", "-i", "--rm"]
    joined = " ".join(cmd)
    assert "--gpus device=0" in joined and "--network none" in joined
    assert f"--user {os.getuid()}:{os.getgid()}" in joined
    assert f"-v {tmp_path / 'stage'}:{tmp_path / 'stage'}" in joined
    assert f"-v {s.engine_dir()}:{s.engine_dir()}:ro" in joined
    assert f"-v {s.checkout() / 'src'}:/fukikae/src:ro" in joined
    for k in ("HF_HUB_OFFLINE=1", "HF_HUB_DISABLE_TELEMETRY=1", "DO_NOT_TRACK=1"):
        assert f"-e {k}" in joined
    # the host's card, for other cards' leftover scans (the worker itself sees device 0)
    assert f"-e {gpulock.GPU_INDEX_ENV}=0" in joined
    # python as the entrypoint: the CUDA image's own prints a banner on stdout
    assert cmd[-6:] == ["--entrypoint", "python", s.image(), "-m", remote.WORKER_MODULE,
                        "indextts2_trt"]  # fmt: skip


@pytest.mark.usefixtures("setup_done")
def test_command_never_passes_a_token(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_do_not_leak")
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    s.gpu_index = 0
    s._cwd = tmp_path / "stage"
    assert "hf_secret_do_not_leak" not in " ".join(s.command())
    assert "HF_TOKEN" not in s.worker_env()


@pytest.mark.usefixtures("setup_done")
def test_environment_problem_names_the_setup(monkeypatch):
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    assert s.environment_problem() is None
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_IMAGE", "missing")
    assert "image" in (s.environment_problem() or "")
    with pytest.raises(BackendUnavailable, match="fukikae backends setup indextts2_trt"):
        s.load()


@pytest.mark.usefixtures("setup_done")
def test_missing_engines_are_reported():
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    (s.engine_dir() / impl.MANIFEST).unlink()
    assert "engines" in (s.environment_problem() or "")


@pytest.mark.usefixtures("setup_done")
def test_engines_for_another_gpu_are_refused(monkeypatch):
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    monkeypatch.setattr(IndexTts2TrtSynthesizer, "gpu_compute_capability", lambda _self, _i: "8.9")
    s.gpu_index = 0
    with pytest.raises(BackendUnavailable, match="8.6"):
        s.load()


@pytest.fixture
def fake_worker(setup_done, monkeypatch):
    monkeypatch.setenv("FUKIKAE_TTS_FAKE_IMPL", "tests.fixtures.fake_tts_impl")
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_PYTHON", sys.executable)
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_PYTHONPATH", f"{ROOT}{os.pathsep}{ROOT / 'src'}")
    monkeypatch.setattr(IndexTts2TrtSynthesizer, "sample_rate", 16000)
    monkeypatch.setattr(trt, "CONTAINER_PASS", frozenset({"FUKIKAE_TTS_FAKE_IMPL"}))
    return setup_done


@pytest.mark.usefixtures("fake_worker")
def test_files_are_staged_into_the_container_and_the_render_comes_back(tmp_path):
    ref = _wav(tmp_path / "clip.wav")
    emo = _wav(tmp_path / "emo.wav", 0.5)
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    s.load()
    try:
        stage = s._cwd
        assert stage is not None
        voice = _card().voice
        out = tmp_path / "out" / "line.wav"
        out.parent.mkdir()
        seconds = s.synthesize(
            "echo", voice, "en", out, resolve=lambda _r: ref, emotion_audio=emo,
            emotion_strength=0.5,
        )  # fmt: skip
        assert out.is_file() and seconds == pytest.approx(0.2, abs=1e-3)
        seen = json.loads(next(stage.rglob("*.wav.json")).read_text())
        # the container saw only staged copies, never the workspace's own files
        clip = Path(seen["voice"]["references"][0]["clip"])
        assert clip.is_relative_to(stage) and clip.read_bytes() == ref.read_bytes()
        assert Path(seen["emotion_audio"]).is_relative_to(stage)
        assert Path(seen["out"]).is_relative_to(stage)
        assert seen["env"]["HF_HUB_OFFLINE"] == "1"
        # the emotion copy and the staged output are gone; the reference stays for the next line
        assert not Path(seen["emotion_audio"]).exists() and not Path(seen["out"]).exists()
        assert clip.exists()
        # a second line with the same clip reuses the staged copy (the prompt cache's key)
        out2 = tmp_path / "out" / "two.wav"
        s.synthesize("echo two", voice, "en", out2, resolve=lambda _r: ref)
        seen2 = json.loads(sorted(stage.rglob("*.wav.json"))[-1].read_text())
        assert seen2["voice"]["references"][0]["clip"] == str(clip)
    finally:
        s.unload()
    assert not stage.exists()


@pytest.mark.usefixtures("fake_worker")
def test_the_gpu_lock_is_held_while_the_container_runs():
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    s.load()
    try:
        assert gpulock.held_fds()
    finally:
        s.unload()
    assert not gpulock.held_fds()


# ---- the engine build ------------------------------------------------------------------------


def _old_engines(engines: Path) -> None:
    (engines / "trt_engines_fp16").mkdir(parents=True)
    (engines / "trt_engines_fp16" / "dit.engine").write_text("old")
    (engines / impl.MANIFEST).write_text(json.dumps({"build_id": "0ld000000000"}))


def test_a_build_replaces_the_engines_whole(tmp_path, monkeypatch):
    engines = tmp_path / "trt" / "abc123"
    _old_engines(engines)

    def into(dest: Path, **_kw: object) -> dict:
        assert dest.parent == engines.parent and dest.name.startswith("abc123.tmp-")
        (dest / "trt_engines_fp16").mkdir()
        (dest / "trt_engines_fp16" / "bigvgan.engine").write_text("new")
        return {"build_id": "new000000000"}

    monkeypatch.setattr(impl, "_build_into", into)
    assert impl.build(engines)["build_id"] == "new000000000"
    assert json.loads((engines / impl.MANIFEST).read_text())["build_id"] == "new000000000"
    assert (engines / "trt_engines_fp16" / "bigvgan.engine").read_text() == "new"
    assert not (engines / "trt_engines_fp16" / "dit.engine").exists()  # nothing of the old set
    assert sorted(p.name for p in engines.parent.iterdir()) == ["abc123"]


def test_a_failed_build_leaves_the_old_engines_untouched(tmp_path, monkeypatch):
    engines = tmp_path / "trt" / "abc123"
    _old_engines(engines)
    (engines.parent / "abc123.tmp-deadbeef").mkdir()  # a killed build's leftovers go too

    def into(dest: Path, **_kw: object) -> dict:
        (dest / "trt_engines_fp16").mkdir()
        (dest / "trt_engines_fp16" / "bigvgan.engine").write_text("partial")
        raise subprocess.CalledProcessError(1, ["build_bigvgan_trt.py"])

    monkeypatch.setattr(impl, "_build_into", into)
    with pytest.raises(subprocess.CalledProcessError):
        impl.build(engines)
    assert json.loads((engines / impl.MANIFEST).read_text())["build_id"] == "0ld000000000"
    assert sorted(p.name for p in (engines / "trt_engines_fp16").iterdir()) == ["dit.engine"]
    assert sorted(p.name for p in engines.parent.iterdir()) == ["abc123"]


def test_a_first_build_needs_no_old_engines(tmp_path, monkeypatch):
    engines = tmp_path / "trt" / "abc123"
    monkeypatch.setattr(impl, "_build_into", lambda _dest, **_kw: {"build_id": "first0000000"})
    impl.build(engines)
    assert json.loads((engines / impl.MANIFEST).read_text())["build_id"] == "first0000000"


def test_a_build_hands_its_fp32_engines_to_the_builder(tmp_path, monkeypatch):
    seen: list[tuple[str, ...]] = []

    def into(_dest: Path, *, fp32: tuple[str, ...] = ()) -> dict:
        seen.append(fp32)
        return {"build_id": "first0000000", "fp32": list(fp32)}

    monkeypatch.setattr(impl, "_build_into", into)
    impl.build(tmp_path / "a", fp32=("dit", "campplus"))
    impl.build(tmp_path / "b")
    assert seen == [("campplus", "dit"), ()]  # sorted: the manifest's order


def test_only_the_named_engines_are_built_without_fp16():
    onnx, dest = Path("/e/onnx/dit.onnx"), Path("/e/trt/dit.engine")
    argv = impl.engine_build_argv("py", "dit", onnx, dest, fp32=("dit",))
    assert argv[-1] == "--no-fp16"
    at = argv.index("--max-batch")
    assert argv[at : at + 2] == ["--max-batch", "2"]  # the DiT's conditional and unconditional
    argv = impl.engine_build_argv("py", "bigvgan", onnx, dest, fp32=("dit",))
    at = argv.index("--max-batch")
    assert "--no-fp16" not in argv and argv[at : at + 2] == ["--max-batch", "1"]
    assert argv[:2] == ["py", f"{impl.INDEX_TTS}/backends/trt/build_engines/build_bigvgan_trt.py"]


def test_the_build_command_takes_the_fp32_engines(tmp_path, monkeypatch, capsys):
    got: list[tuple[Path, tuple[str, ...]]] = []

    def build(engines: Path, *, fp32: tuple[str, ...] = ()) -> dict:
        got.append((engines, fp32))
        return {"build_id": "x", "gpu": "g", "times_s": {}, "sizes": {}}

    monkeypatch.setattr(impl, "build", build)
    assert impl.main(["build", str(tmp_path), "--fp32", "dit,campplus"]) == 0
    assert impl.main(["build", str(tmp_path)]) == 0
    assert got == [(tmp_path, ("dit", "campplus")), (tmp_path, ())]
    assert impl.main(["build", str(tmp_path), "--fp32", "gpt"]) == 2
    assert "gpt" in capsys.readouterr().err


def test_the_worker_refuses_engines_built_with_other_fp32_engines(tmp_path, monkeypatch):
    (tmp_path / impl.MANIFEST).write_text(json.dumps({
        "build_id": "x", "code": impl.CODE_REVISION, "model": f"{impl.REPO}@{impl.REVISION}",
        "fp32": ["dit"],
    }))  # fmt: skip
    monkeypatch.setenv("FUKIKAE_TRT_ENGINES", str(tmp_path))
    with pytest.raises(RuntimeError, match="fp32"):
        impl.Impl().load({}, "cuda")
    with pytest.raises(RuntimeError, match="fp32"):
        impl.Impl().load({"fp32": ["bigvgan"]}, "cuda")
    with pytest.raises(ValueError, match="fp16"):
        impl.Impl().load({"fp16": True}, "cuda")


# ---- the container's lifecycle -------------------------------------------------------------


@pytest.mark.usefixtures("fake_worker")
def test_the_container_runs_isolated_and_is_removed_before_the_lock_goes(tmp_path, monkeypatch):
    held_at_rm: list[tuple[int, ...]] = []
    original = IndexTts2TrtSynthesizer._remove_container

    def remove(self, name):
        held_at_rm.append(gpulock.held_fds())
        return original(self, name)

    monkeypatch.setattr(IndexTts2TrtSynthesizer, "_remove_container", remove)
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    s.load()
    stage = s._cwd
    s.unload()
    calls = _docker_calls(tmp_path)
    runs = [c for c in calls if c[0] == "run"]
    assert len(runs) == 1
    run = runs[0]
    assert "--rm" in run and _flag(run, "--network") == ["none"]
    vols = _flag(run, "-v")
    engines, hub = s.engine_dir(), tmp_path / "hub"
    assert f"{engines}:{engines}:ro" in vols and f"{hub}:{hub}:ro" in vols
    assert f"{s.checkout() / 'src'}:/fukikae/src:ro" in vols
    assert f"{stage}:{stage}" in vols  # the one read-write mount
    assert [v for v in vols if not v.endswith(":ro")] == [f"{stage}:{stage}"]
    assert _flag(run, "--cap-drop") == ["ALL"]
    assert "no-new-privileges" in _flag(run, "--security-opt")
    assert _flag(run, "--pids-limit") and "--read-only" in run and _flag(run, "--tmpfs")
    name = _flag(run, "--name")[0]
    assert ["rm", "-f", name] in calls
    assert held_at_rm and all(held_at_rm)  # removed while the GPU lock was still held
    assert not gpulock.held_fds()


@pytest.mark.usefixtures("setup_done")
def test_a_container_that_stays_is_logged(monkeypatch, caplog):
    import logging

    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_RM_RC", "1")
    with caplog.at_level(logging.ERROR, logger="fukikae.backends.tts.indextts2_trt"):
        s._remove_container("fukikae-indextts2_trt-1-gone")  # rm failed, but it is gone
    assert not caplog.records
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_STUCK", "1")
    with caplog.at_level(logging.ERROR, logger="fukikae.backends.tts.indextts2_trt"):
        s._remove_container("fukikae-indextts2_trt-1-stuck")
    assert any("fukikae-indextts2_trt-1-stuck" in r.getMessage() for r in caplog.records)


@pytest.mark.usefixtures("setup_done")
def test_removing_a_container_without_docker_is_logged_not_raised(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(trt, "DOCKER", "/nonexistent/docker")
    with caplog.at_level(logging.ERROR, logger="fukikae.backends.tts.indextts2_trt"):
        IndexTts2TrtSynthesizer({}, "cuda:0")._remove_container("fukikae-indextts2_trt-1-x")
    assert any("fukikae-indextts2_trt-1-x" in r.getMessage() for r in caplog.records)


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


@pytest.mark.usefixtures("fake_worker")
def test_load_reaps_containers_whose_fukikae_is_gone(tmp_path, monkeypatch):
    dead = _dead_pid()
    stale = f"fukikae-indextts2_trt-{dead}-0a0a0a0a"
    live = f"fukikae-indextts2_trt-{os.getpid()}-0b0b0b0b"
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_PS", f"{stale},{live},someone-else,fukikae-x-1-y")
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    s.load()
    s.unload()
    removed = [c[2] for c in _docker_calls(tmp_path) if c[:2] == ["rm", "-f"]]
    assert stale in removed
    assert live not in removed and "someone-else" not in removed


# ---- docker's state ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "match"),
    [("down", "daemon"), ("denied", "permission denied")],
)
@pytest.mark.usefixtures("setup_done")
def test_an_unreachable_docker_is_not_a_missing_image(monkeypatch, state, match):
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_DAEMON", state)
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    with pytest.raises(BackendUnavailable, match=match):
        s.image_exists()
    problem = s.environment_problem() or ""
    assert match in problem and "no image" not in problem


@pytest.mark.usefixtures("setup_done")
def test_a_missing_image_is_false(monkeypatch):
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_IMAGE", "missing")
    assert IndexTts2TrtSynthesizer({}, "cuda:0").image_exists() is False


@pytest.mark.usefixtures("setup_done")
def test_docker_missing_is_reported_and_refuses_to_load(monkeypatch):
    monkeypatch.setattr(trt, "DOCKER", "/nonexistent/docker")
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    assert "docker" in (s.environment_problem() or "")
    with pytest.raises(BackendUnavailable, match="docker"):
        s.load()


# ---- setup ---------------------------------------------------------------------------------


class _Recorder:
    """A ``run`` for ``setup``: records (argv, env, the GPU locks held) and succeeds, or raises
    for the argv ``fail_on`` matches."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, str], tuple[int, ...]]] = []
        self.fail_on = fail_on

    def __call__(self, argv, *, env, check):
        assert check
        argv = [str(a) for a in argv]
        self.calls.append((argv, dict(env), gpulock.held_fds()))
        if self.fail_on and self.fail_on in argv:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0)

    def runs(self, marker: str) -> list[list[str]]:
        return [a for a, _, _ in self.calls if marker in a]


@pytest.fixture
def setup_env(setup_done, monkeypatch):
    monkeypatch.setattr(trt, "_free_bytes", lambda _path: 10**15)
    (IndexTts2TrtSynthesizer({}, "cuda:0").engine_dir() / impl.MANIFEST).unlink()
    return setup_done


@pytest.mark.usefixtures("setup_env")
def test_setup_names_the_build_container_and_removes_it_under_the_lock(tmp_path, monkeypatch):
    held_at_rm: list[tuple[int, ...]] = []
    original = IndexTts2TrtSynthesizer._remove_container

    def remove(self, name):
        held_at_rm.append(gpulock.held_fds())
        return original(self, name)

    monkeypatch.setattr(IndexTts2TrtSynthesizer, "_remove_container", remove)
    run = _Recorder(fail_on="fukikae.backends.tts.impl_indextts2_trt")
    with pytest.raises(subprocess.CalledProcessError):
        IndexTts2TrtSynthesizer.setup(run=run)
    (build,) = [a for a in run.runs("build") if a[1] == "run"]
    name = _flag(build, "--name")[0]
    assert name.startswith(f"fukikae-indextts2_trt-{os.getpid()}-")
    assert f"{gpulock.GPU_INDEX_ENV}=0" in _flag(build, "-e")
    assert ["rm", "-f", name] in _docker_calls(tmp_path)  # even though the build failed
    assert held_at_rm and all(held_at_rm)
    assert not gpulock.held_fds()


@pytest.mark.usefixtures("setup_env")
def test_setup_skips_engines_already_built_for_this_gpu_unless_forced():
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    (s.engine_dir() / impl.MANIFEST).write_text(
        json.dumps({"build_id": "b1d0000000ab", "compute_capability": "8.6"})
    )
    run = _Recorder()
    IndexTts2TrtSynthesizer.setup(run=run)
    assert not [a for a in run.runs("build") if a[1] == "run"]
    run = _Recorder()
    IndexTts2TrtSynthesizer.setup(run=run, force=True)
    assert [a for a in run.runs("build") if a[1] == "run"]


@pytest.mark.usefixtures("setup_env")
def test_setup_builds_the_fp32_engines_its_options_name_beside_the_default_set():
    run = _Recorder()
    IndexTts2TrtSynthesizer.setup(run=run, options={"fp32": ["dit", "campplus"]})
    (build,) = [a for a in run.runs("build") if a[1] == "run"]
    variant = IndexTts2TrtSynthesizer({"fp32": ["campplus", "dit"]}, "cuda:0").engine_dir()
    assert build[-4:] == ["build", str(variant), "--fp32", "campplus,dit"]
    run = _Recorder()
    IndexTts2TrtSynthesizer.setup(run=run)
    (build,) = [a for a in run.runs("build") if a[1] == "run"]
    assert build[-2:] == ["build", str(IndexTts2TrtSynthesizer({}, "cuda:0").engine_dir())]


@pytest.mark.usefixtures("setup_env")
def test_setup_keeps_an_fp32_set_already_built():
    s = IndexTts2TrtSynthesizer({"fp32": ["dit"]}, "cuda:0")
    _variant_manifest(s, ["dit"], "b1d0000000ab")
    run = _Recorder()
    IndexTts2TrtSynthesizer.setup(run=run, options={"fp32": ["dit"]})
    assert not [a for a in run.runs("build") if a[1] == "run"]


@pytest.mark.usefixtures("setup_env")
def test_setup_cli_reads_the_workspace_s_engine_options(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from fukikae.cli import app

    seen: list[dict] = []
    monkeypatch.setattr(
        IndexTts2TrtSynthesizer, "setup",
        classmethod(lambda _cls, **kw: seen.append(kw.get("options") or {})),
    )  # fmt: skip
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "fukikae.toml").write_text('[tts.indextts2_trt]\nfp32 = ["dit"]\n')
    result = CliRunner().invoke(app, ["backends", "setup", "indextts2_trt", "--workspace", str(ws)])
    assert result.exit_code == 0, result.output
    assert seen == [{"fp32": ["dit"]}]
    result = CliRunner().invoke(app, ["backends", "setup", "indextts2_trt"])
    assert result.exit_code == 0, result.output
    assert seen[-1] == {}


@pytest.mark.usefixtures("setup_env")
def test_setup_rebuilds_engines_built_for_another_gpu():
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    (s.engine_dir() / impl.MANIFEST).write_text(
        json.dumps({"build_id": "b1d0000000ab", "compute_capability": "8.9"})
    )
    run = _Recorder()
    IndexTts2TrtSynthesizer.setup(run=run)
    assert [a for a in run.runs("build") if a[1] == "run"]


@pytest.mark.usefixtures("setup_env")
def test_setup_refuses_without_the_nvidia_runtime(monkeypatch):
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_INFO", '/ {"runc":{}}')
    monkeypatch.setattr(trt.shutil, "which", lambda _name: None)
    with pytest.raises(BackendUnavailable, match="nvidia"):
        IndexTts2TrtSynthesizer.setup(run=_Recorder())


@pytest.mark.usefixtures("setup_env")
def test_setup_refuses_an_unreachable_docker(monkeypatch):
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_DAEMON", "denied")
    with pytest.raises(BackendUnavailable, match="permission denied"):
        IndexTts2TrtSynthesizer.setup(run=_Recorder())


@pytest.mark.usefixtures("setup_env")
def test_setup_states_the_disk_it_needs(monkeypatch):
    monkeypatch.setenv("FUKIKAE_FAKE_DOCKER_IMAGE", "missing")
    monkeypatch.setattr(trt, "_free_bytes", lambda _path: 10 * 10**9)
    with pytest.raises(BackendUnavailable, match="24 GB"):
        IndexTts2TrtSynthesizer.setup(run=_Recorder())
    monkeypatch.delenv("FUKIKAE_FAKE_DOCKER_IMAGE")
    monkeypatch.setattr(trt, "_free_bytes", lambda _path: 2 * 10**9)
    with pytest.raises(BackendUnavailable, match="3 GB"):
        IndexTts2TrtSynthesizer.setup(run=_Recorder())


@pytest.mark.usefixtures("setup_env")
def test_setup_fetch_gets_the_proxy_and_ca_but_never_shows_the_token(tmp_path, monkeypatch):
    ca = tmp_path / "ca.pem"
    ca.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    monkeypatch.setenv("HF_TOKEN", "hf_secret_do_not_leak")
    run = _Recorder()
    IndexTts2TrtSynthesizer.setup(run=run)
    (fetch,) = run.runs("--fetch")
    env = next(e for a, e, _ in run.calls if a is fetch or a == fetch)
    assert "HTTPS_PROXY" in _flag(fetch, "-e") and env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert "SSL_CERT_FILE" in _flag(fetch, "-e") and f"{ca}:{ca}:ro" in _flag(fetch, "-v")
    assert "hf_secret_do_not_leak" not in " ".join(fetch)
    for argv, env, _ in run.calls:
        if "--fetch" not in argv:
            assert "HF_TOKEN" not in env and "HTTPS_PROXY" not in _flag(argv, "-e")


@pytest.mark.usefixtures("setup_env")
def test_setup_cli_reports_docker_missing_without_a_traceback(monkeypatch):
    from typer.testing import CliRunner

    from fukikae.cli import app

    monkeypatch.setattr(trt, "DOCKER", "/nonexistent/docker")
    result = CliRunner().invoke(app, ["backends", "setup", "indextts2_trt"])
    assert result.exit_code == 1
    assert "error:" in result.output and "docker" in result.output
    assert "Traceback" not in result.output


# ---- the stage the container can write -----------------------------------------------------


@pytest.fixture
def staged(setup_done, tmp_path):
    del setup_done  # the checkout and engines it sets up
    s = IndexTts2TrtSynthesizer({}, "cuda:0")
    stage = tmp_path / "stage"
    stage.mkdir()
    s._cwd = stage
    secret = tmp_path / "secret.txt"
    secret.write_text("host secret")
    return s, secret


def test_a_render_swapped_for_a_symlink_is_refused(staged, tmp_path):
    s, secret = staged
    out = s._subdir("out")
    (out / "000001.wav").symlink_to(secret)
    dest = tmp_path / "line.wav"
    with pytest.raises(remote.BackendError, match="render"):
        s._take_out(out / "000001.wav", dest)
    assert not dest.exists()


def test_a_render_is_copied_out_and_the_staged_one_removed(staged, tmp_path):
    s, _ = staged
    out = s._subdir("out")
    _wav(out / "000001.wav")
    dest = tmp_path / "line.wav"
    dest.symlink_to(tmp_path / "secret.txt")  # an old link at the destination is replaced
    s._take_out(out / "000001.wav", dest)
    assert not dest.is_symlink() and dest.read_bytes()[:4] == b"RIFF"
    assert (tmp_path / "secret.txt").read_text() == "host secret"
    assert not (out / "000001.wav").exists()


def test_a_stage_directory_swapped_for_a_symlink_is_refused(staged, tmp_path):
    s, _ = staged
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (s._cwd / "refs").symlink_to(elsewhere)
    with pytest.raises(remote.BackendError, match="refs"):
        s._stage(_wav(tmp_path / "clip.wav"), "refs", keep=True)
    assert not list(elsewhere.iterdir())


def test_a_staged_copy_never_writes_through_a_planted_link(staged, tmp_path):
    s, secret = staged
    clip = _wav(tmp_path / "clip.wav")
    import hashlib

    digest = hashlib.sha256(clip.read_bytes()).hexdigest()[:16]
    refs = s._subdir("refs")
    (refs / f"{digest}.wav").symlink_to(secret)
    with pytest.raises(remote.BackendError, match="not a regular file"):
        s._stage(clip, "refs", keep=True)
    assert secret.read_text() == "host secret"


def test_the_build_reads_the_gpu_from_nvidia_smi(monkeypatch):
    """Read at the start of the build, not by a CUDA init at its end: a host `systemctl
    daemon-reload` mid-build revokes a container's device access for new opens."""

    def smi(argv, **_kw):
        assert argv[0] == "nvidia-smi" and "--query-gpu=name,compute_cap" in argv
        return subprocess.CompletedProcess(argv, 0, "NVIDIA GeForce RTX 3090, 8.6\n", "")

    monkeypatch.setattr(impl.subprocess, "run", smi)
    assert impl._gpu_identity() == ("NVIDIA GeForce RTX 3090", "8.6")
