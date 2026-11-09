from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from fukikae.artifacts import read_json, write_json
from fukikae.config import Config
from fukikae.job import cache_key
from fukikae.models import Delivery, DubCue, DubCues, FitReport, TtsPlan, TtsPlanCue
from fukikae.stages.timefit import TimefitStage
from fukikae.workspace import Episode

RATE = 24000


def _cue(i: int, start: float, end: float, text: str | None = "line", **kw) -> DubCue:
    return DubCue(
        id=f"c{i:04d}",
        start=start,
        end=end,
        character_id=1,
        extra=None,
        speaker="char1",
        source_text="src",
        text=text,
        budget=20,
        delivery=Delivery(reference=(start, end)),
        **kw,
    )


def _wav(path: Path, seconds: float) -> None:
    t = np.arange(round(seconds * RATE)) / RATE
    wavfile.write(path, RATE, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


# id: (slot start, slot end, rendered seconds, rate factor); None = skipped
LAYOUT: dict[int, tuple[float, float, float, float] | None] = {
    1: (1.0, 3.0, 1.2, 1.0),  # short: padded at the cue start
    2: (4.0, 5.5, 2.2, 1.0),  # stretch 1.25 gives 1.76 s; borrows 0.26 s after
    3: (7.0, 8.0, 2.5, 1.0),  # next cue adjacent: borrows before, then left long
    4: (8.0, 9.0, 0.8, 1.0),
    5: None,  # a skipped cue (song) still bounds its neighbours
    6: (12.0, 13.0, 1.3, 1.25),  # the rate fit used the whole budget: no stretch
}
SKIPPED_AT = (10.0, 11.9)


@pytest.fixture
def ep(tmp_path: Path) -> Episode:
    root = tmp_path / "ws"
    work = root / ".fukikae" / "episodes" / "e1"
    (work / "06-translate").mkdir(parents=True)
    (work / "07-tts").mkdir()
    cues: list[DubCue] = []
    plan: list[TtsPlanCue] = []
    for i, spec in LAYOUT.items():
        if spec is None:
            cues.append(_cue(i, *SKIPPED_AT, text=None, skip="song"))
            plan.append(TtsPlanCue(cue_id=f"c{i:04d}", character_id=1, extra=None, slot=1.9,
                                   skipped="skip: song"))  # fmt: skip
            continue
        a, b, secs, rate = spec
        cues.append(_cue(i, a, b))
        _wav(work / "07-tts" / f"c{i:04d}.wav", secs)
        plan.append(
            TtsPlanCue(
                cue_id=f"c{i:04d}", character_id=1, extra=None, card_id="k", backend="kokoro",
                key=f"key{i}", seconds=secs, rendered_seconds=secs * rate, slot=b - a,
                rate_factor=rate,
            )
        )  # fmt: skip
    write_json(
        work / "06-translate" / "cues.en.json",
        DubCues(episode="e1", source_language="ja", target_language="en", path="asr", cues=cues),
    )
    write_json(work / "07-tts" / "plan.json", TtsPlan(episode="e1", cues=plan))
    return Episode(source=tmp_path / "s.mkv", id="e1", work=work, workspace_root=root)


CFG = Config()


def _report(ep: Episode) -> FitReport:
    return read_json(ep.work / "08-timefit" / "report.json", FitReport)


@pytest.mark.ffmpeg
def test_timefit_places_stretches_borrows_and_reports(ep):
    TimefitStage(CFG).run(ep, CFG)
    report = _report(ep)
    by = {c.cue_id: c for c in report.cues}
    assert sorted(by) == ["c0001", "c0002", "c0003", "c0004", "c0006"]

    c1 = by["c0001"]
    assert c1.start == 1.0 and c1.stretch == 1.0 and c1.over_by == 0
    assert c1.seconds == pytest.approx(1.2, abs=0.01)

    c2 = by["c0002"]
    assert c2.stretch == pytest.approx(1.25)
    assert c2.borrowed_after == pytest.approx(2.2 / 1.25 - 1.5, abs=1e-3)
    assert c2.seconds == pytest.approx(1.76, abs=0.02) and c2.start == 4.0

    c3 = by["c0003"]  # previous placed end 5.76: borrows 0.4 s before; next cue at 8.0
    assert c3.borrowed_after == 0 and c3.borrowed_before == pytest.approx(0.4)
    assert c3.start == pytest.approx(6.6)
    assert c3.over_by == pytest.approx(2.5 / 1.25 - 1.0 - 0.4, abs=1e-3)
    # Placed whole: it overlaps the next cue, nothing is cut.
    assert c3.start + c3.seconds > 8.0 + 0.5

    c6 = by["c0006"]
    assert c6.stretch == 1.0 and c6.borrowed_after == pytest.approx(0.3)
    # The skipped cue before it ends at 11.9: only 0.1 s to borrow there, and none needed.
    assert c6.borrowed_before == 0

    assert report.over == ["c0003"]
    assert report.summary["placed"] == 5 and report.summary["stretched"] == 2
    assert report.summary["borrowed"] == 3 and report.summary["over"] == 1
    assert report.summary["worst"] == {"cue_id": "c0003", "over_by": pytest.approx(0.6, abs=1e-3)}
    for cid, c in by.items():
        out = ep.work / "08-timefit" / f"{cid}.wav"
        assert out.is_file()
        rate, data = wavfile.read(out)
        assert len(data) / rate == pytest.approx(c.seconds, abs=1e-3)


@pytest.mark.ffmpeg
def test_timefit_removes_stale_wavs_and_is_byte_stable(ep):
    stage = TimefitStage(CFG)
    (ep.work / "08-timefit").mkdir()
    (ep.work / "08-timefit" / "c0099.wav").write_bytes(b"old")
    stage.run(ep, CFG)
    first = (ep.work / "08-timefit" / "report.json").read_bytes()
    assert not (ep.work / "08-timefit" / "c0099.wav").exists()
    stage.run(ep, CFG)
    assert (ep.work / "08-timefit" / "report.json").read_bytes() == first


def test_timefit_inputs_name_every_rendered_wav(ep):
    inputs = TimefitStage(CFG).inputs(ep)
    names = [p.name for p in inputs]
    assert names[:2] == ["plan.json", "cues.en.json"]
    assert names[2:] == ["c0001.wav", "c0002.wav", "c0003.wav", "c0004.wav", "c0006.wav"]


def test_timefit_key_follows_settings_and_audio(ep):
    stage = TimefitStage(CFG)
    base = cache_key(stage, ep, CFG)
    cfg2 = Config(raw={"timefit": {"borrow_silence_ms": 200}})
    assert cache_key(TimefitStage(cfg2), ep, cfg2) != base
    _wav(ep.work / "07-tts" / "c0004.wav", 0.7)
    assert cache_key(stage, ep, CFG) != base
    key = stage.config_key(CFG, ep)
    assert key["settings"] == {"max_speedup": 1.25, "max_slowdown": 0.9, "borrow_silence_ms": 400}
    assert "ffmpeg" in key and "version" in key


def test_timefit_outputs(ep):
    assert TimefitStage(CFG).outputs(ep) == [ep.work / "08-timefit" / "report.json"]
