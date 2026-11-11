import json
import logging
import wave
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from fukikae import job
from fukikae.artifacts import read_json, write_json
from fukikae.config import Config
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    DubCues,
    Segment,
    Transcript,
    Turn,
    Word,
)
from fukikae.registry.store import Registry
from fukikae.workspace import Episode, Workspace

RATE = 16000
DUR = 20  # seconds: speech 0-8 s, silence after
STOCK = "ご視聴ありがとうございました"


def _mono(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _stems(work: Path) -> None:
    sep = work / "01-separate"
    sep.mkdir(parents=True, exist_ok=True)
    t = np.arange(DUR * RATE) / RATE
    voice = 0.2 * np.sin(2 * np.pi * 220 * t)
    voice[t >= 8] = 0.0  # no speech after 8 s
    bed = 0.001 * np.random.default_rng(0).standard_normal(len(t))  # about -60 dBFS
    _mono(sep / "background.wav", bed)
    _mono(sep / "vocals.wav", voice)
    _mono(sep / "vocals_16k.wav", voice)


def _words(start: float, text: str) -> list[Word]:
    return [
        Word(start=start + i * 0.3, end=start + i * 0.3 + 0.2, text=t) for i, t in enumerate(text)
    ]


def _seg(start: float, text: str) -> Segment:
    words = _words(start, text)
    return Segment(start=start, end=words[-1].end, text=text, words=words)


def _episode(tmp_path: Path, segments: list[Segment]) -> Episode:
    root = tmp_path / "ws"
    work = root / ".fukikae" / "episodes" / "e1"
    (work / "03-asr").mkdir(parents=True)
    _stems(work)
    write_json(
        work / "03-asr" / "transcript.json",
        Transcript(language="ja", backend="fake", word_timing=True, segments=segments),
    )
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


SPEECH = _seg(0.5, "燃えてきたぞ")
SAID_TO_CAMERA = _seg(3.0, STOCK)  # over speech: kept
OVER_SILENCE = _seg(11.0, STOCK)  # over no speech: dropped


@pytest.fixture
def stage():
    from fukikae.stages.asrclean import AsrCleanStage

    return AsrCleanStage()


def test_output_is_byte_identical_when_nothing_is_dropped(tmp_path, stage):
    ep = _episode(tmp_path, [SPEECH, SAID_TO_CAMERA])
    stage.run(ep, Config())
    cleaned, dropped = stage.outputs(ep)
    assert cleaned.read_bytes() == (ep.work / "03-asr" / "transcript.json").read_bytes()
    assert json.loads(dropped.read_text()) == []


def test_dropped_segments_are_listed_with_reasons(tmp_path, stage):
    ep = _episode(tmp_path, [SPEECH, SAID_TO_CAMERA, OVER_SILENCE])
    stage.run(ep, Config())
    cleaned, dropped = stage.outputs(ep)
    assert cleaned == ep.work / "03b-asrclean" / "transcript.json"
    assert dropped == ep.work / "03b-asrclean" / "dropped.json"
    kept = read_json(cleaned, Transcript)
    assert [s.text for s in kept.segments] == ["燃えてきたぞ", STOCK]
    assert kept.language == "ja" and kept.backend == "fake" and kept.word_timing
    (row,) = json.loads(dropped.read_text())
    assert row["text"] == STOCK and row["reason"] == "stock phrase over no speech"
    assert row["start"] == pytest.approx(11.0) and row["end"] == pytest.approx(OVER_SILENCE.end)


def test_only_the_runs_over_no_speech_are_dropped(tmp_path, stage):
    """A stock-phrase segment whose words sit in two runs, one over speech and one over
    silence 12 s later, loses only the silent run (Sol Bianca's 「お疲れ様でした」 layout)."""
    words = _words(1.0, STOCK[:3]) + _words(14.0, STOCK[3:])
    split = Segment(start=0.5, end=words[-1].end, text=STOCK, words=words)
    ep = _episode(tmp_path, [SPEECH, split])
    stage.run(ep, Config())
    cleaned, dropped = stage.outputs(ep)
    kept = read_json(cleaned, Transcript)
    assert [s.text for s in kept.segments] == ["燃えてきたぞ", STOCK[:3]]
    assert kept.segments[1].start == pytest.approx(1.0)
    assert [w.text for w in kept.segments[1].words or []] == list(STOCK[:3])
    (row,) = json.loads(dropped.read_text())
    assert row["text"] == STOCK[3:] and row["reason"] == "stock phrase over no speech"
    assert row["start"] == pytest.approx(14.0)


def test_inputs_outputs_and_key(tmp_path, stage):
    from fukikae import hallucinations, songs

    ep = _episode(tmp_path, [SPEECH])
    sep = ep.work / "01-separate"
    assert stage.name == "asrclean"
    assert stage.inputs(ep) == [
        ep.work / "03-asr" / "transcript.json",
        sep / "background.wav",
        sep / "vocals.wav",
        sep / "vocals_16k.wav",
    ]
    key = stage.config_key(Config(), ep)
    assert key["rule"] == hallucinations.rule_constants()
    assert key["songs"] == songs.detector_constants()
    tuned = Config(raw={"mix": {"songs": {"tail_window_s": 20}}})
    assert stage.config_key(tuned, ep) != key


def test_missing_stems_keep_every_segment(tmp_path, stage, caplog):
    ep = _episode(tmp_path, [SPEECH, OVER_SILENCE])
    (ep.work / "01-separate" / "background.wav").unlink()
    with caplog.at_level(logging.WARNING):
        stage.run(ep, Config())
    cleaned, _ = stage.outputs(ep)
    assert cleaned.read_bytes() == (ep.work / "03-asr" / "transcript.json").read_bytes()
    assert "stems" in caplog.text


def test_pipeline_order_puts_asrclean_after_asr():
    from fukikae.cli import pipeline_stages

    names = [s.name for s in pipeline_stages(Config())]
    assert names[names.index("asr") + 1] == "asrclean"


def test_transcript_path_prefers_the_cleaned_one(tmp_path, stage, caplog):
    from fukikae import transcripts
    from fukikae.transcripts import transcript_path

    ep = _episode(tmp_path, [SPEECH])
    transcripts._WARNED.clear()
    with caplog.at_level(logging.WARNING):
        assert transcript_path(ep) == ep.work / "03-asr" / "transcript.json"
        assert transcript_path(ep.work) == ep.work / "03-asr" / "transcript.json"
    assert caplog.text.count("predates the asrclean stage") == 1  # once per episode
    stage.run(ep, Config())
    assert transcript_path(ep) == ep.work / "03b-asrclean" / "transcript.json"
    assert transcript_path(ep.work) == ep.work / "03b-asrclean" / "transcript.json"


def test_existing_workspace_gets_the_stage_once_without_an_asr_rerun(tmp_path, stage, monkeypatch):
    from fukikae.stages import asr
    from fukikae.stages.asr import AsrStage

    monkeypatch.setattr(asr, "identity_key", lambda _k, _bc: {"model": "m", "revision": None})

    def no_asr(*_a: Any) -> None:
        raise AssertionError("asr must not rerun")

    monkeypatch.setattr(AsrStage, "run", no_asr)
    ep = _episode(tmp_path, [SPEECH, OVER_SILENCE])
    cfg = Config()
    # An episode transcribed before asrclean existed: asr done under its current key.
    manifest = job.Manifest()
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    manifest.stages["asr"] = job.StageRecord(
        key=job.cache_key(AsrStage(), ep, cfg), status="done", started=now, finished=now
    )
    job.save_manifest(ep, manifest)
    assert job.run_stages(ep, cfg, [AsrStage(), stage]) == ["asrclean"]
    assert job.run_stages(ep, cfg, [AsrStage(), stage]) == []


# -- consumers read the cleaned transcript --------------------------------------------------


class FakeTranslator:
    name: ClassVar[str] = "fake"
    kind: ClassVar[str] = "translate"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0

    def __init__(self, options: dict[str, Any], device: str) -> None:  # noqa: ARG002
        self.options = options

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def translate(self, req: Any) -> dict[str, Any]:
        from fukikae.translation import Rendering

        return {c.id: Rendering(literal="lit", text=f"ok {c.id}", direction=None) for c in req.cues}


def test_consumers_read_the_cleaned_transcript(tmp_path, stage, monkeypatch):
    from fukikae.stages import translate
    from fukikae.stages.translate import TranslateStage

    monkeypatch.setattr(translate, "create", lambda _k, bc, dev: FakeTranslator(bc.options, dev))
    monkeypatch.setattr(translate, "identity_key", lambda _k, _bc: {"model": "f", "revision": None})
    monkeypatch.setattr(translate, "device_string", lambda _cfg: "cpu")
    ep = _episode(tmp_path, [SPEECH, OVER_SILENCE])
    (ep.work / "05-characters").mkdir()
    with Registry(Workspace(ep.workspace_root).registry_path) as reg:
        _, natsu = reg.create_actor_with_character(np.ones(4, dtype=np.float32))
    write_json(
        ep.work / "05-characters" / "assignments.json",
        Assignments(
            episode="e1",
            threshold=0.6,
            clusters=[
                ClusterAssignment(
                    cluster="S0", actor_id=natsu, character_id=natsu, score=0.9,
                    clean_seconds=30, status="matched",
                )
            ],
            turns=[Turn(start=0.0, end=20.0, speaker="S0")],
        ),
    )  # fmt: skip
    cfg = Config(source_language="ja", raw={"translate": {"backend": "fake"}})
    t = TranslateStage()
    assert ep.work / "03b-asrclean" / "transcript.json" not in t.inputs(ep)
    stage.run(ep, cfg)
    assert ep.work / "03b-asrclean" / "transcript.json" in t.inputs(ep)
    t.run(ep, cfg)
    out = read_json(ep.work / "06-translate" / "cues.en.json", DubCues)
    assert [c.source_text for c in out.cues] == ["燃えてきたぞ"]

    from fukikae.cards.auto import episode_artifacts

    arts = episode_artifacts(Workspace(ep.workspace_root), "e1")
    assert arts is not None
    assert [s.text for s in arts.transcript.segments] == ["燃えてきたぞ"]
