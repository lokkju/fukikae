import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from fukikae.artifacts import read_json, write_json
from fukikae.config import Config, ConfigError
from fukikae.models import (
    Delivery,
    DubCue,
    DubCues,
    FitReport,
    FitReportCue,
    MixReport,
    Tags,
    TagSpan,
)
from fukikae.stages.mix import MixStage, mix_settings
from fukikae.workspace import Episode

RATE = 16000
DUR = 12.0
VOICE_HZ = 1000.0
DUB_HZ = 300.0


def _stereo(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        st = np.stack([x, x], axis=1)
        w.writeframes((np.clip(st, -1, 1) * 32767).astype("<i2").tobytes())


def _cue(i: int, start: float, end: float, text: str | None, **kw) -> DubCue:
    return DubCue(
        id=f"c{i:04d}",
        start=start,
        end=end,
        character_id=1,
        extra=None,
        speaker=kw.pop("speaker", "Natsu"),
        source_text="src",
        text=text,
        budget=20,
        delivery=Delivery(reference=(start, end)),
        **kw,
    )


# Rendered cues: id -> (cue start, cue end, placed start, seconds)
PLACED = {
    1: (1.0, 3.0, 1.0, 1.5),
    2: (5.0, 7.0, 5.0, 1.8),  # laughter at 6.0-6.5 inside it
    4: (9.5, 10.5, 9.5, 1.0),  # lyrics translated as a line, inside a subtitle song cue
    5: (3.6, 4.4, 3.6, 0.8),  # inside a tag singing span
}


@pytest.fixture
def ep(tmp_path: Path) -> Episode:
    root = tmp_path / "ws"
    work = root / ".fukikae" / "episodes" / "e1"
    for d in ("00-extract", "01-separate", "04b-tag", "06-translate", "08-timefit"):
        (work / d).mkdir(parents=True)
    n = int(DUR * RATE)
    t = np.arange(n) / RATE
    rng = np.random.default_rng(3)
    bg = 0.02 * rng.standard_normal(n).clip(-3, 3)
    vo = np.zeros(n)
    for a, b in ((1, 3), (3.5, 4.5), (5, 7), (9, 11)):
        sel = (t >= a) & (t < b)
        vo[sel] = 0.2 * np.sin(2 * np.pi * VOICE_HZ * t[sel])
    _stereo(work / "01-separate" / "background.wav", bg)
    _stereo(work / "01-separate" / "vocals.wav", vo)
    _stereo(work / "00-extract" / "audio.wav", bg + vo)
    write_json(
        work / "04b-tag" / "tags.json",
        Tags(
            backend="sensevoice",
            events=[
                TagSpan(start=3.5, end=4.5, label="singing"),
                TagSpan(start=6.0, end=6.5, label="laughter"),
            ],
            emotions=[],
        ),
    )
    cues = [
        _cue(1, 1.0, 3.0, "First line."),
        _cue(2, 5.0, 7.0, "Second line.", speaker="extra:SPEAKER_01"),
        _cue(3, 9.0, 11.0, None, skip="song"),
        _cue(4, 9.5, 10.5, "Lyrics as a line."),
        _cue(5, 3.6, 4.4, "Sung words."),
    ]
    write_json(
        work / "06-translate" / "cues.en.json",
        DubCues(episode="e1", source_language="en", target_language="en", path="subs", cues=cues),
    )
    fit: list[FitReportCue] = []
    for i, (a, b, start, secs) in PLACED.items():
        x = 0.3 * np.sin(2 * np.pi * DUB_HZ * np.arange(int(secs * 24000)) / 24000)
        wavfile.write(work / "08-timefit" / f"c{i:04d}.wav", 24000, x.astype(np.float32))
        fit.append(
            FitReportCue(
                cue_id=f"c{i:04d}", slot_start=a, slot_end=b, start=start, seconds=secs,
                rendered_seconds=secs, rate_factor=1.0, stretch=1.0, borrowed_before=0.0,
                borrowed_after=0.0, over_by=0.0,
            )
        )  # fmt: skip
    write_json(work / "08-timefit" / "report.json", FitReport(episode="e1", cues=fit, over=[]))
    src = tmp_path / "src.mkv"
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nFirst line.\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={DUR}",
            "-i", str(work / "00-extract" / "audio.wav"),
            "-i", str(srt),
            "-map", "0:v", "-map", "1:a", "-map", "2:s",
            "-c:v", "mpeg4", "-c:a", "aac", "-c:s", "srt",
            "-metadata:s:a:0", "language=jpn", "-metadata:s:s:0", "language=eng",
            str(src),
        ],
        check=True,
    )  # fmt: skip
    return Episode(source=src, id="e1", work=work, workspace_root=root)


CFG = Config(raw={"tag": {"backend": "sensevoice", "enabled": True}})


def _band(x: np.ndarray, rate: int, hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    return float(spectrum[np.abs(freqs - hz) < 20].sum() / (spectrum.sum() + 1e-12))


@pytest.mark.ffmpeg
def test_mix_protects_songs_and_laughter_and_muxes(ep):
    MixStage(CFG, tags=True).run(ep, CFG)
    out = ep.work / "09-mix"
    report = read_json(out / "report.json", MixReport)
    assert report.dropped == {"c0004": "subs", "c0005": "tag:singing"}
    songs = {(s.start, s.end, s.source) for s in report.songs}
    assert (9.0, 11.0, "subs") in songs and (3.5, 4.5, "tag:singing") in songs
    assert [(s.start, s.end, s.source) for s in report.laughter] == [(6.0, 6.5, "tag:laughter")]
    assert report.placed == 2

    rate, y = wavfile.read(out / "dub.en.wav")
    assert rate == RATE
    mono = y.astype(np.float64).mean(axis=1)

    def seg(a: float, b: float) -> np.ndarray:
        return mono[int(a * rate) : int(b * rate)]

    # Dub window: the dub plays and the original voice is gone.
    assert _band(seg(1.1, 2.4), rate, DUB_HZ) > 0.5 and _band(seg(1.1, 2.4), rate, VOICE_HZ) < 0.02
    # The rest of the original line's slot stays clear after a short dub ends.
    assert _band(seg(2.6, 2.95), rate, VOICE_HZ) < 0.02
    # Song spans: the original voice untouched, no dub.
    for a, b in ((9.6, 10.4), (3.7, 4.3)):
        assert _band(seg(a, b), rate, VOICE_HZ) > 0.5 and _band(seg(a, b), rate, DUB_HZ) < 0.02
    # Laughter inside a dub cue: both audible.
    assert (
        _band(seg(6.05, 6.45), rate, VOICE_HZ) > 0.2 and _band(seg(6.05, 6.45), rate, DUB_HZ) > 0.2
    )

    loud = report.loudness
    assert set(loud) == {"original", "premix", "final"}
    assert set(loud["final"]) == {"i", "lra", "tp"}
    assert loud["final"]["i"] == pytest.approx(loud["original"]["i"], abs=1.0)
    assert loud["final"]["tp"] <= -1.0
    assert report.gain_db == pytest.approx(loud["original"]["i"] - loud["premix"]["i"], abs=0.05)
    assert report.normalization.startswith("linear")

    ass = (out / "dub.en.ass").read_text(encoding="utf-8")
    lines = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert (
        len(lines) == 2 and "Natsu: First line." in lines[0] and "Extra: Second line." in lines[1]
    )

    mkv = ep.work / "out" / "e1.fukikae.mkv"
    streams = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(mkv)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout  # fmt: skip
    )["streams"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    assert [a["tags"]["language"] for a in audio] == ["eng", "jpn"]
    assert [a["disposition"]["default"] for a in audio] == [1, 0]
    subs = [s for s in streams if s["codec_type"] == "subtitle"]
    assert [s["disposition"]["default"] for s in subs] == [1, 0]
    assert [s["disposition"]["forced"] for s in subs] == [0, 0]
    assert [s["codec_name"] for s in streams if s["codec_type"] == "subtitle"] == ["ass", "subrip"]


@pytest.mark.ffmpeg
def test_mix_without_tags_still_protects_subtitle_songs(ep):
    cfg = Config()
    MixStage(cfg, tags=False).run(ep, cfg)
    report = read_json(ep.work / "09-mix" / "report.json", MixReport)
    assert report.dropped == {"c0004": "subs"}
    assert report.laughter == [] and report.placed == 3


def test_mix_inputs_follow_the_tag_stage(ep):
    with_tags = [p.name for p in MixStage(CFG, tags=True).inputs(ep)]
    without = [p.name for p in MixStage(CFG, tags=False).inputs(ep)]
    assert "tags.json" in with_tags and "tags.json" not in without
    for name in ("report.json", "cues.en.json", "background.wav", "vocals.wav", "audio.wav",
                 "c0001.wav", "c0005.wav", "src.mkv"):  # fmt: skip
        assert name in without


def test_mix_outputs(ep):
    out = MixStage(CFG, tags=True).outputs(ep)
    assert out == [
        ep.work / "09-mix" / "dub.en.wav",
        ep.work / "09-mix" / "dub.en.ass",
        ep.work / "09-mix" / "report.json",
        ep.work / "out" / "e1.fukikae.mkv",
    ]


def test_mix_key_holds_the_settings(ep):
    key = MixStage(CFG, tags=True).config_key(CFG, ep)
    assert key["settings"]["keep_original_vocals"] is True
    assert key["settings"]["duck_db"] is None and key["settings"]["protect_bgm"] is False
    assert key["settings"]["detect_songs"] is True
    assert key["target_language"] == "en" and "ffmpeg" in key and "version" in key
    assert key["limiter_ceiling_db"] == -1.0
    assert "songs" in key  # the detector's constants


def test_mix_settings_read_and_validated():
    s = mix_settings(Config(raw={"mix": {"duck_db": -18, "protect_bgm": True}}))
    assert s.duck_db == -18 and s.protect_bgm is True and s.fade_ms == 40
    with pytest.raises(ConfigError, match="duck_db"):
        mix_settings(Config(raw={"mix": {"duck_db": 3}}))
    with pytest.raises(ConfigError, match="unknown"):
        mix_settings(Config(raw={"mix": {"duk_db": -10}}))
