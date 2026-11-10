from pathlib import Path

import numpy as np
import pytest

from fukikae.artifacts import write_json
from fukikae.eval.groundtruth import GroundTruthLine as G
from fukikae.eval.identity.segments import (
    Segment,
    asr_segments,
    asr_sentences,
    diarization_segments,
    segments,
    subtitle_segments,
)
from fukikae.models import Diarization, Transcript, Turn, Word
from fukikae.models import Segment as AsrSegment

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
        segments(tmp_path, "S01E01", "bogus", gt)  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        segments(tmp_path, "S01E02", "diarization", gt)


def _used_names(ep_dir: Path, turns: list[Turn]) -> None:
    from fukikae.names.models import NamesResult

    d = ep_dir / "04c-names"
    d.mkdir(parents=True)
    write_json(d / "turns.json", Diarization(backend="fake", turns=turns))
    write_json(
        d / "names.json",
        NamesResult(
            episode="S01E01", used=True, verdict="full", reason=None, source=None, offset_s=0.0,
            coverage={}, lines=[], hidden=[], pieces=[], labels={}, unvoiced=[], excluded=[],
            dropped_row_seconds=0.0, suggestions=[],
        ),
    )  # fmt: skip


def test_segments_read_the_names_stage_turns_when_names_were_used(tmp_path: Path):
    ep_dir = tmp_path / ".fukikae" / "episodes" / "S01E01"
    (ep_dir / "04-diarize").mkdir(parents=True)
    write_json(ep_dir / "04-diarize" / "turns.json", Diarization(backend="fake", turns=TURNS))
    refined = [
        t.model_copy(update={"speaker": "SPEAKER_00~natsu"}) if t.speaker == "SPEAKER_00" else t
        for t in TURNS
    ]
    _used_names(ep_dir, refined)
    gt = [G(20.0, 22.0, "Natsu")]
    got = segments(tmp_path, "S01E01", "diarization", gt)
    assert [s.cluster for s in got] == ["SPEAKER_01", "SPEAKER_00~natsu"]
    given = segments(tmp_path, "S01E01", "diarization", gt, turns=TURNS)  # e.g. assignments.turns
    assert [s.cluster for s in given] == ["SPEAKER_01", "SPEAKER_00"]


# ---- asr: sentences from the transcript, split where word-window voices differ ----


def wd(a: float, b: float, t: str) -> Word:
    return Word(start=a, end=b, text=t)


def transcript(*segs: AsrSegment) -> Transcript:
    return Transcript(language="en", backend="fake", word_timing=True, segments=list(segs))


def test_asr_sentences_split_at_terminal_punctuation_in_words_or_text():
    en = AsrSegment(
        start=0.0,
        end=2.5,
        text=" Hello there. How are you?",
        words=[
            wd(0.0, 0.5, " Hello"),
            wd(0.6, 1.2, " there."),
            wd(1.3, 1.6, " How"),
            wd(1.7, 1.9, " are"),
            wd(2.0, 2.5, " you?"),
        ],
    )
    # Japanese: one word per character, the punctuation only in the segment text.
    ja = AsrSegment(
        start=10.0,
        end=13.0,
        text="こいつ。ぐる",
        words=[
            wd(10.0, 10.2, "こ"),
            wd(10.3, 10.5, "い"),
            wd(10.6, 10.8, "つ"),
            wd(12.0, 12.2, "ぐ"),
            wd(12.4, 13.0, "る"),
        ],
    )
    got = asr_sentences(transcript(en, ja))
    assert [[w.text.strip() for w in s] for s in got] == [
        ["Hello", "there."],
        ["How", "are", "you?"],
        ["こ", "い", "つ"],
        ["ぐ", "る"],
    ]


def test_asr_sentences_split_at_long_gaps_and_skip_wordless_segments():
    seg = AsrSegment(
        start=0.0,
        end=5.0,
        text="a b c",
        words=[wd(0.0, 0.5, "a"), wd(0.6, 1.0, "b"), wd(3.0, 3.5, "c")],
    )
    bare = AsrSegment(start=6.0, end=8.0, text="no words", words=None)
    got = asr_sentences(transcript(seg, bare), max_gap=1.0)
    assert [[w.text for w in s] for s in got] == [["a", "b"], ["c"]]


def words_every(start: float, n: int, step: float = 0.5) -> list[Word]:
    return [wd(start + k * step, start + k * step + 0.4, f"w{k}") for k in range(n)]


def test_asr_segments_split_where_window_voices_differ():
    # One 8 s sentence (16 words, 0.5 s apart): 1.5 s windows of 4 words -> 4 windows; the
    # voice changes after the second window.
    seg = AsrSegment(start=0.0, end=8.0, text="", words=words_every(0.0, 16))
    calls: list[list[tuple[float, float]]] = []

    def embed(spans):
        calls.append(spans)
        a, b = np.eye(4, dtype=np.float32)[:2]
        return np.vstack([a, a, b, b][: len(spans)])

    got = asr_segments("S01E01", transcript(seg), [], embed, window=1.5, split=0.5)
    assert len(calls) == 1 and len(calls[0]) == 4
    assert [(s.start, s.end) for s in got] == [(0.0, 3.9), (4.0, 7.9)]
    assert all(s.cluster is None and s.label is None for s in got)


def test_asr_segments_short_tail_window_joins_the_previous_one():
    seg = AsrSegment(start=0.0, end=4.5, text="", words=words_every(0.0, 9))  # 0-4.4 s
    seen: list[list[tuple[float, float]]] = []

    def embed(spans):
        seen.append(spans)
        return np.tile(np.eye(4, dtype=np.float32)[:1], (len(spans), 1))

    got = asr_segments("S01E01", transcript(seg), [], embed, window=1.5)
    # Windows 0-1.9 and 2.0-4.4 (the 0.4 s tail at 4.0-4.4 joined the second).
    assert seen == [[(0.0, 1.9), (2.0, 4.4)]]
    assert [(s.start, s.end) for s in got] == [(0.0, 4.4)]


def test_asr_segments_drop_overlap_label_by_coverage_and_skip_short():
    seg1 = AsrSegment(start=0.0, end=4.0, text="", words=words_every(0.0, 8))  # 0-3.9
    seg2 = AsrSegment(start=10.0, end=10.6, text="", words=[wd(10.0, 10.6, "x")])  # too short
    gt = [G(0.0, 3.0, "Lucy"), G(2.5, 3.9, "Natsu")]  # 2.5-3.0 overlapped: removed
    turns = [
        Turn(start=0.0, end=1.0, speaker="SPEAKER_00"),
        Turn(start=0.5, end=1.2, speaker="SPEAKER_01"),  # 0.5-1.0 overlapped: removed
    ]

    def embed(spans):
        return np.tile(np.eye(4, dtype=np.float32)[:1], (len(spans), 1))

    got = asr_segments("S01E01", transcript(seg1, seg2), gt, embed, turns=turns)
    # Left after both overlaps go: 0-0.5, 1.0-2.5, 3.0-3.9; only 1.0-2.5 reaches 1.0 s.
    assert [(s.start, s.end, s.label) for s in got] == [(1.0, 2.5, "Lucy")]


def test_segments_reads_the_transcript_for_asr(tmp_path: Path):
    ep = tmp_path / ".fukikae" / "episodes" / "S01E01"
    (ep / "03-asr").mkdir(parents=True)
    (ep / "04-diarize").mkdir(parents=True)
    seg = AsrSegment(start=0.0, end=2.5, text="", words=words_every(0.0, 5))
    write_json(ep / "03-asr" / "transcript.json", transcript(seg))
    write_json(ep / "04-diarize" / "turns.json", Diarization(backend="fake", turns=[]))

    def embed(spans):
        return np.tile(np.eye(4, dtype=np.float32)[:1], (len(spans), 1))

    got = segments(tmp_path, "S01E01", "asr", [G(0.0, 2.4, "Lucy")], window_embed=embed)
    assert [(s.start, s.end, s.label) for s in got] == [(0.0, 2.4, "Lucy")]
    with pytest.raises(ValueError, match="window"):
        segments(tmp_path, "S01E01", "asr", [])
