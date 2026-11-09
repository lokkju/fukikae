import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fukikae.artifacts import read_json
from fukikae.config import Config
from fukikae.models import Segment, Transcript, Word
from fukikae.stages import asr
from fukikae.stages.asr import AsrStage
from fukikae.workspace import Episode

LOG: list[Any] = []


def _tr(word_timing: bool) -> Transcript:
    words = [Word(start=0.0, end=1.0, text="hi", score=0.9)] if word_timing else None
    return Transcript(
        language="ko",
        backend="fake",
        word_timing=word_timing,
        segments=[Segment(start=0.0, end=1.0, text="hi", words=words)],
    )


class FakeAsr:
    name: ClassVar[str] = "fake-asr"
    kind: ClassVar[str] = "asr"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 10
    supports_lid: ClassVar[bool] = False
    timed: ClassVar[bool] = False

    def __init__(self, options: dict[str, Any], device: str) -> None:
        LOG.append(("asr-init", options, device))

    def load(self) -> None:
        LOG.append("asr-load")

    def unload(self) -> None:
        LOG.append("asr-unload")

    def transcribe(self, _audio: Path, language: str | None) -> Transcript:
        LOG.append(("transcribe", language))
        return _tr(FakeAsr.timed)


class FakeAligner:
    name: ClassVar[str] = "fake-align"
    kind: ClassVar[str] = "align"
    languages: ClassVar = frozenset({"ko"})
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 10

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.options = options
        LOG.append(("align-init", options, device))

    def load(self) -> None:
        LOG.append("align-load")

    def unload(self) -> None:
        LOG.append("align-unload")

    def align(self, _audio: Path, transcript: Transcript) -> Transcript:
        LOG.append("align")
        return _tr(True).model_copy(update={"backend": transcript.backend})


class EnOnlyAligner(FakeAligner):
    languages: ClassVar = frozenset({"en"})


class OtherAligner(FakeAligner):
    pass


@pytest.fixture
def ep(tmp_path) -> Episode:
    root = tmp_path / "ws"
    work = root / "episodes" / "e1"
    (work / "01-separate").mkdir(parents=True)
    (work / "01-separate" / "vocals_16k.wav").write_bytes(b"x")
    (work / "02-language").mkdir()
    (work / "02-language" / "language.json").write_text(
        json.dumps({"language": "ko", "confidence": 0.9, "source": "detected"})
    )
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


@pytest.fixture
def fakes(monkeypatch):
    LOG.clear()
    FakeAsr.timed = False
    classes = {"asr": {"fake": FakeAsr, "fa": FakeAsr}, "align": {"fa": FakeAligner}}
    classes["align"]["en"] = EnOnlyAligner
    classes["align"]["wav2vec2"] = FakeAligner

    def create(kind, bc, dev):
        return classes[kind][bc.backend](bc.options, dev)

    monkeypatch.setattr(asr, "create", create)
    monkeypatch.setattr(asr, "device_string", lambda _cfg: "cpu")
    return classes


def _cfg(**raw):
    return Config(raw=raw)


@pytest.mark.usefixtures("fakes")
def test_aligner_skipped_when_asr_has_word_timing(ep):
    FakeAsr.timed = True
    AsrStage().run(ep, _cfg(asr={"backend": "fake"}))
    t = read_json(ep.work / "03-asr" / "transcript.json", Transcript)
    assert t.word_timing
    assert "align" not in LOG
    assert LOG == [
        ("asr-init", {}, "cpu"), "asr-load", ("transcribe", "ko"), "asr-unload"
    ]  # fmt: skip


@pytest.mark.usefixtures("fakes")
def test_aligner_runs_after_asr_unloads(ep):
    AsrStage().run(ep, _cfg(asr={"backend": "fake"}, align={"backend": "wav2vec2", "model": "m"}))
    t = read_json(ep.work / "03-asr" / "transcript.json", Transcript)
    assert t.word_timing
    assert LOG.index("asr-unload") < LOG.index("align-load")
    assert LOG[-3:] == ["align-load", "align", "align-unload"]
    assert ("align-init", {"model": "m"}, "cpu") in LOG


@pytest.mark.usefixtures("fakes")
def test_falls_back_to_base_aligner_when_language_unsupported(ep):
    cfg = _cfg(
        asr={"backend": "fake"},
        align={"backend": "wav2vec2", "lang": {"ko": {"backend": "en", "x": 1}}},
    )
    AsrStage().run(ep, cfg)
    inits = [e for e in LOG if isinstance(e, tuple) and e[0] == "align-init"]
    assert [i[1] for i in inits] == [{"x": 1}, {}]
    assert LOG[-1] == "align-unload"
    assert read_json(ep.work / "03-asr" / "transcript.json", Transcript).word_timing


def test_defaults_when_no_tables(monkeypatch, ep):
    seen = []

    def create(kind, bc, dev):
        seen.append((kind, bc.backend, bc.options))
        return (FakeAsr if kind == "asr" else FakeAligner)(bc.options, dev)

    LOG.clear()
    monkeypatch.setattr(asr, "create", create)
    monkeypatch.setattr(asr, "device_string", lambda _cfg: "cpu")
    AsrStage().run(ep, Config())
    assert seen[0] == ("asr", "faster_whisper", {})
    assert seen[1] == ("align", "wav2vec2", {})


@pytest.mark.usefixtures("fakes")
def test_backend_override_keeps_options_only_when_names_match(ep):
    cfg = _cfg(asr={"backend": "fake", "model": "big"}, align={"backend": "wav2vec2"})
    AsrStage(backend="fake").run(ep, cfg)
    AsrStage(backend="fa").run(ep, cfg)
    inits = [e[1] for e in LOG if isinstance(e, tuple) and e[0] == "asr-init"]
    assert inits == [{"model": "big"}, {}]


@pytest.mark.usefixtures("fakes")
def test_out_dir_and_outputs(ep):
    s = AsrStage(out_dir="eval/asr-fake")
    assert s.name == "asr:eval/asr-fake"
    s.run(ep, _cfg(asr={"backend": "fake"}))
    assert (ep.work / "eval" / "asr-fake" / "transcript.json").exists()
    assert s.outputs(ep) == [ep.work / "eval" / "asr-fake" / "transcript.json"]
    assert AsrStage().outputs(ep) == [ep.work / "03-asr" / "transcript.json"]
    assert AsrStage().inputs(ep) == [
        ep.work / "01-separate" / "vocals_16k.wav",
        ep.work / "02-language" / "language.json",
    ]


@pytest.mark.usefixtures("fakes")
def test_unload_after_transcribe_failure(ep, monkeypatch):
    def bad(_s, _a, _l):
        raise RuntimeError("boom")

    monkeypatch.setattr(FakeAsr, "transcribe", bad)
    with pytest.raises(RuntimeError):
        AsrStage().run(ep, _cfg(asr={"backend": "fake"}))
    assert LOG[-1] == "asr-unload"
    assert not (ep.work / "03-asr" / "transcript.json").exists()


@pytest.mark.usefixtures("fakes")
def test_config_key_tracks_backend_aligner_and_language(ep):
    s = AsrStage()
    base = s.config_key(_cfg(asr={"backend": "fake"}), ep)
    assert s.config_key(_cfg(asr={"backend": "fake", "model": "x"}), ep) != base
    assert (
        s.config_key(_cfg(asr={"backend": "fake"}, align={"backend": "wav2vec2", "model": "m"}), ep)
        != base
    )
    assert AsrStage(backend="fa").config_key(_cfg(asr={"backend": "fake"}), ep) != base


@pytest.mark.gpu
def test_gpu_asr_and_align_on_clip(clip_ep):
    AsrStage().run(clip_ep, Config(source_language="ko"))
    t = read_json(clip_ep.work / "03-asr" / "transcript.json", Transcript)
    assert t.language == "ko" and t.word_timing and t.segments


@pytest.mark.usefixtures("fakes")
def test_eval_out_dir_gets_its_own_manifest_record(ep):
    from fukikae.job import load_manifest, run_stages

    assert AsrStage().name == "asr"
    evals = AsrStage(backend="fa", out_dir="eval/asr-fa")
    assert evals.name == "asr:eval/asr-fa"
    cfg = _cfg(asr={"backend": "fake"}, align={"backend": "fa"})
    assert run_stages(ep, cfg, [AsrStage()]) == ["asr"]
    pipeline = load_manifest(ep).stages["asr"]
    assert run_stages(ep, cfg, [evals]) == ["asr:eval/asr-fa"]
    assert load_manifest(ep).stages["asr"] == pipeline  # eval left it alone
    assert run_stages(ep, cfg, [AsrStage(backend="fa", out_dir="eval/asr-fa")]) == []
    assert run_stages(ep, cfg, [AsrStage()]) == []


@pytest.mark.usefixtures("fakes")
def test_asr_on_cuda_holds_the_gpu_lock(monkeypatch, ep):
    """Ruling CD15."""
    from fukikae import gpulock

    monkeypatch.setattr(asr, "device_string", lambda _cfg: "cuda:0")
    monkeypatch.setattr(asr, "free_mb", lambda _d: 99999)
    held: list[Any] = []
    monkeypatch.setattr(FakeAsr, "load", lambda _self: held.append(gpulock.held_fds()))
    FakeAsr.timed = True
    AsrStage().run(ep, _cfg(asr={"backend": "fake"}))
    assert held and held[0]
    assert gpulock.held_fds() == ()
