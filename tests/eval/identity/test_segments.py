from pathlib import Path

import pytest

from fukikae.artifacts import write_json
from fukikae.eval.groundtruth import GroundTruthLine as G
from fukikae.eval.identity.segments import (
    Segment,
    diarization_segments,
    segments,
    subtitle_segments,
)
from fukikae.models import Diarization, Turn

TURNS = [
    Turn(start=10.0, end=15.0, speaker="SPEAKER_01"),
    Turn(start=0.0, end=4.0, speaker="SPEAKER_00"),
    Turn(start=3.5, end=6.0, speaker="SPEAKER_01"),  # overlaps SPEAKER_00: no clean span
    Turn(start=20.0, end=22.0, speaker="SPEAKER_00"),
    Turn(start=30.0, end=31.0, speaker="SPEAKER_02"),  # too short once padded
]


def test_diarization_segments_are_clean_spans_labeled_by_overlap():
    gt = [G(10.0, 13.0, "Lucy"), G(13.0, 15.0, "Natsu"), G(19.0, 23.0, "Natsu")]
    got = diarization_segments("S01E01", TURNS, gt)
    # Clusters in order of first appearance, each cluster's clean spans in time order.
    assert [(s.cluster, s.start, s.end) for s in got] == [
        ("SPEAKER_01", 10.1, 14.9),
        ("SPEAKER_00", 20.1, 21.9),
    ]
    # 10.1-14.9: Lucy covers 2.9 of 4.8 s (60.4%), enough at the 0.6 bar.
    assert [s.label for s in got] == ["Lucy", "Natsu"]
    assert all(s.episode == "S01E01" for s in got)


def test_diarization_segment_under_the_overlap_bar_is_unlabeled():
    gt = [G(10.0, 12.5, "Lucy"), G(12.5, 15.0, "Natsu")]
    got = diarization_segments("S01E01", TURNS, gt)
    assert got[0].label is None  # each name covers 2.4 of 4.8 s, under 0.6
    assert got[1].label is None  # no line at 20-22 s


def test_subtitle_segments_drop_overlap_and_short_pieces():
    gt = [
        G(0.0, 5.0, "A"),
        G(4.0, 8.0, "B"),  # 4-5 overlaps A: removed from both
        G(7.5, 9.0, "A"),  # overlaps B at 7.5-8: A keeps 8-9 (1.0 s), B keeps 5-7.5
        G(10.0, 10.8, "C"),  # too short
        G(12.0, 14.0, "D"),
        G(13.0, 16.0, "D"),  # same speaker overlapping: merged, not removed
    ]
    got = subtitle_segments("S01E01", gt)
    assert [(s.start, s.end, s.label) for s in got] == [
        (0.0, 4.0, "A"),
        (5.0, 7.5, "B"),
        (8.0, 9.0, "A"),
        (12.0, 16.0, "D"),
    ]
    assert all(s.cluster is None for s in got)


def test_segment_duration():
    assert Segment("E", 1.0, 3.5, None).duration == pytest.approx(2.5)


def test_segments_reads_the_diarize_stage(tmp_path: Path):
    work = tmp_path / ".fukikae" / "episodes" / "S01E01" / "04-diarize"
    work.mkdir(parents=True)
    write_json(work / "turns.json", Diarization(backend="fake", turns=TURNS))
    gt = [G(20.0, 22.0, "Natsu")]
    got = segments(tmp_path, "S01E01", "diarization", gt)
    assert [s.label for s in got] == [None, "Natsu"]
    assert segments(tmp_path, "S01E01", "subtitles", gt) == [Segment("S01E01", 20.0, 22.0, "Natsu")]
    with pytest.raises(ValueError, match="segmentation"):
        segments(tmp_path, "S01E01", "asr", gt)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        segments(tmp_path, "S01E02", "diarization", gt)
