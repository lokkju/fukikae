from pathlib import Path

import numpy as np
import pytest

from fukikae.backends.tag.sensevoice import (
    DEFAULT_REVISION,
    SenseVoiceTagger,
    merge_event_windows,
    parse_rich_tags,
    split_turn,
)
from fukikae.models import TagSpan, Turn


def test_parse_rich_tags():
    assert parse_rich_tags("<|ja|><|HAPPY|><|Laughter|><|withitn|>ははは") == ("happy", "laughter")
    assert parse_rich_tags("<|en|><|EMO_UNKNOWN|><|Speech|><|woitn|>hi") == (None, None)
    assert parse_rich_tags("<|en|><|NEUTRAL|><|BGM|>") == ("neutral", "bgm")


def test_parse_rich_tags_maps_cry_and_drops_unlisted_events():
    assert parse_rich_tags("<|en|><|SAD|><|Cry|><|woitn|>") == ("sad", "crying")
    assert parse_rich_tags("<|en|><|NEUTRAL|><|Cough|><|woitn|>") == ("neutral", None)
    assert parse_rich_tags("no tags at all") == (None, None)


def test_adjacent_event_windows_merge():
    spans = merge_event_windows([(0.0, 5.0, "bgm"), (2.5, 7.5, "bgm"), (5.0, 10.0, None)])
    assert spans == [TagSpan(start=0.0, end=7.5, label="bgm")]


def test_event_windows_with_a_gap_or_other_label_stay_apart():
    spans = merge_event_windows(
        [(0.0, 5.0, "bgm"), (2.5, 7.5, "laughter"), (7.5, 12.5, None), (10.0, 15.0, "bgm")]
    )
    assert [(s.start, s.end, s.label) for s in spans] == [
        (0.0, 5.0, "bgm"),
        (2.5, 7.5, "laughter"),
        (10.0, 15.0, "bgm"),
    ]


def test_split_turn_keeps_short_turns_whole_and_splits_long_ones():
    assert split_turn(Turn(start=1.0, end=9.0, speaker="S0")) == [(1.0, 9.0)]
    parts = split_turn(Turn(start=0.0, end=65.0, speaker="S0"))
    assert parts == [(0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]


def test_defaults_are_the_gpu_with_a_pinned_revision():
    t = SenseVoiceTagger({}, "cpu")
    assert (t.model, t.revision) == ("FunAudioLLM/SenseVoiceSmall", DEFAULT_REVISION)
    assert len(DEFAULT_REVISION) == 40
    # D26: the GPU by default (the tag stage falls back to the CPU without one)
    assert SenseVoiceTagger.default_device == "cuda" and SenseVoiceTagger.vram_mb > 0
    assert SenseVoiceTagger.devices == {"cpu", "cuda"}


@pytest.mark.slow
@pytest.mark.media
def test_sensevoice_on_fixture_clip(tmp_path: Path):
    import wave

    clip = Path(
        "/home/lokkju/fukikae-ws/fairytail/.fukikae/episodes/S01E11/01-separate/vocals_16k.wav"
    )
    if not clip.is_file():
        pytest.skip("no Fairy Tail E11 vocals on this machine")
    with wave.open(str(clip)) as w:
        w.setpos(int(60 * 16000))
        raw = w.readframes(40 * 16000)
    short = tmp_path / "clip.wav"
    with wave.open(str(short), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(16000)
        o.writeframes(raw)
    assert np.frombuffer(raw, dtype="<i2").size == 40 * 16000
    tagger = SenseVoiceTagger({}, "cpu")
    tagger.load()
    try:
        tags = tagger.tag(short, [Turn(start=19.07, end=22.75, speaker="S0")])
    finally:
        tagger.unload()
    allowed = {"neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"}
    assert tags.emotions  # E11 vocals 79.07-82.75 s read as happy; most turns read as unknown
    assert {s.label for s in tags.emotions} <= allowed
    assert {s.label for s in tags.events} <= {"bgm", "singing", "laughter", "crying", "applause"}
