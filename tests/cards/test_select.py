"""Reference selection (ruling CD5) and clone cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from scipy.io import wavfile

from fukikae.backends.base import Synthesizer
from fukikae.cards.card import card_id, dump_card, parse_card
from fukikae.cards.select import (
    RefPick,
    clone_card,
    reference_report,
    select_references,
)
from fukikae.cards.store import CardStore
from fukikae.models import Segment, Tags, TagSpan, Transcript, Turn, Word

RATE = 16000


def _vocals(path: Path, spans: list[tuple[float, float, float]], seconds: float = 120.0) -> Path:
    """Silence with, per (start, end, noise) span, a tone over its middle 70% and noise of
    amplitude ``noise`` over the rest, so a span's SNR is about 20 log10(0.3 / noise)."""
    rng = np.random.default_rng(0)
    x = np.zeros(int(seconds * RATE), np.float32)
    for start, end, noise in spans:
        a, b = int(start * RATE), int(end * RATE)
        x[a:b] = noise * rng.standard_normal(b - a)
        n = b - a
        lo, hi = a + int(0.15 * n), a + int(0.85 * n)
        t = np.arange(hi - lo) / RATE
        x[lo:hi] = 0.3 * np.sin(2 * np.pi * 200 * t)
    wavfile.write(path, RATE, (np.clip(x, -1, 1) * 32767).astype("<i2"))
    return path


def _transcript(turns: list[Turn]) -> Transcript:
    segs = [
        Segment(
            start=t.start, end=t.end, text=f"line at {t.start:g}",
            words=[Word(start=t.start + 0.1, end=t.end - 0.1, text=f"line at {t.start:g}")],
        )
        for t in turns
    ]  # fmt: skip
    return Transcript(language="ja", backend="x", word_timing=True, segments=segs)


def _tags(emotions: list[tuple[float, float, str, float | None]], backend: str | None) -> Tags:
    return Tags(
        backend="sensevoice",
        events=[],
        emotions=[TagSpan(start=a, end=b, label=lab, confidence=c) for a, b, lab, c in emotions],
        emotion_backend=backend,
    )


def _spans(picks: list[RefPick] | None) -> list[tuple[float, float]]:
    assert picks is not None
    return [(p.start, p.end) for p in picks]


def test_overlapping_turns_excluded(tmp_path):
    turns = [
        Turn(start=0, end=5, speaker="A"),
        Turn(start=4, end=6, speaker="B"),  # overlaps A's first turn
        Turn(start=10, end=15, speaker="A"),
        Turn(start=20, end=25, speaker="A"),
    ]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.001), (10, 15, 0.01), (20, 25, 0.01)])
    picks = select_references(turns, "A", _transcript(turns), None, vocals)
    assert (0.0, 5.0) not in _spans(picks)
    assert sorted(_spans(picks)) == [(10.0, 15.0), (20.0, 25.0)]


def test_ranked_by_snr(tmp_path):
    turns = [Turn(start=s, end=s + 5, speaker="A") for s in (0, 10, 20, 30)]
    noise = [0.03, 0.001, 0.01, 0.1]
    vocals = _vocals(
        tmp_path / "v.wav", [(t.start, t.end, n) for t, n in zip(turns, noise, strict=True)]
    )
    picks = select_references(turns, "A", _transcript(turns), None, vocals)
    assert _spans(picks) == [(10, 15), (20, 25), (0, 5)]
    assert picks is not None and picks[0].snr_db > picks[1].snr_db > picks[2].snr_db


def test_non_neutral_refs_rank_last_and_warn(tmp_path):
    """CD5: a scored non-neutral turn ranks after every other eligible turn, is picked only to
    reach the floor (D32), and the report says so."""
    turns = [
        Turn(start=0, end=5, speaker="A"),  # the cleanest turn, but angry at 0.8
        Turn(start=10, end=14, speaker="A"),  # neutral, 4 s: short of min_total alone
    ]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.0005), (10, 14, 0.01)])
    tags = _tags([(0, 5, "angry", 0.8), (10, 14, "neutral", 0.9)], "emotion2vec")
    picks = select_references(turns, "A", _transcript(turns), tags, vocals)
    assert _spans(picks) == [(10, 14), (0, 5)]
    assert picks is not None and [p.emotion for p in picks] == [None, "angry"]
    assert reference_report(picks) == ["references may carry emotion: angry"]

    # with enough neutral speech for the floor (D32: 2 clips, 10 s), the angry turn is not
    # used at all, clean as it is
    turns.append(Turn(start=20, end=26, speaker="A"))
    vocals = _vocals(tmp_path / "v2.wav", [(0, 5, 0.0005), (10, 14, 0.01), (20, 26, 0.01)])
    tags = _tags(
        [(0, 5, "angry", 0.8), (10, 14, "neutral", 0.9), (20, 26, "happy", 0.4)], "emotion2vec"
    )
    picks = select_references(turns, "A", _transcript(turns), tags, vocals)
    assert sorted(_spans(picks)) == [(10, 14), (20, 26)]
    assert picks is not None and reference_report(picks) == []


def test_unscored_or_low_confidence_labels_exclude_nothing(tmp_path):
    turns = [Turn(start=0, end=5, speaker="A"), Turn(start=10, end=15, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.0005), (10, 15, 0.01)])
    # SenseVoice labels carry no score: they demote nothing
    sv = _tags([(0, 5, "angry", None)], None)
    assert _spans(select_references(turns, "A", _transcript(turns), sv, vocals))[0] == (0, 5)
    # a scored label under 0.5 demotes nothing either
    low = _tags([(0, 5, "angry", 0.3)], "emotion2vec")
    picks = select_references(turns, "A", _transcript(turns), low, vocals)
    assert _spans(picks)[0] == (0, 5)
    assert picks is not None and picks[0].emotion is None and reference_report(picks) == []


def test_fills_to_count_and_total(tmp_path):
    starts = [0, 10, 20, 30, 40, 50]
    turns = [Turn(start=s, end=s + 6, speaker="A") for s in starts]
    turns += [Turn(start=60, end=62, speaker="A"), Turn(start=70, end=84, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(t.start, t.end, 0.01) for t in turns])
    picks = select_references(turns, "A", _transcript(turns), None, vocals)
    assert picks is not None and len(picks) == 3
    assert sum(p.end - p.start for p in picks) <= 20.0
    assert all(3.0 <= p.end - p.start <= 12.0 for p in picks)
    # 8 s turns: two fit under 20 s, a third would not
    turns8 = [Turn(start=s, end=s + 8, speaker="A") for s in (0, 10, 20)]
    vocals = _vocals(tmp_path / "v8.wav", [(t.start, t.end, 0.01) for t in turns8])
    assert len(_spans(select_references(turns8, "A", _transcript(turns8), None, vocals))) == 2


def test_too_little_clean_speech_returns_none(tmp_path):
    turns = [Turn(start=0, end=5.5, speaker="A"), Turn(start=10, end=12, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5.5, 0.01), (10, 12, 0.01)])
    assert select_references(turns, "A", _transcript(turns), None, vocals) is None


def test_turns_with_no_transcript_are_not_references(tmp_path):
    turns = [Turn(start=0, end=5, speaker="A"), Turn(start=10, end=15, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.001), (10, 15, 0.01)])
    tr = _transcript([turns[1]])  # ASR heard nothing in the first turn
    picks = select_references(turns, "A", tr, None, vocals, min_total=3.0)
    assert _spans(picks) == [(10, 15)]
    assert picks is not None and picks[0].transcript == "line at 10"


def test_several_clusters_of_one_character(tmp_path):
    turns = [
        Turn(start=0, end=5, speaker="S0"),
        Turn(start=10, end=15, speaker="S3"),
        Turn(start=20, end=25, speaker="S1"),
    ]
    vocals = _vocals(tmp_path / "v.wav", [(t.start, t.end, 0.01) for t in turns])
    picks = select_references(turns, {"S0", "S3"}, _transcript(turns), None, vocals)
    assert sorted(_spans(picks)) == [(0, 5), (10, 15)]


class CloneSynth(Synthesizer):
    name: ClassVar[str] = "clonefake"
    kind: ClassVar[str] = "tts"
    languages: ClassVar = frozenset({"en"})
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0
    sources: ClassVar = frozenset({"clone"})
    voice_keys: ClassVar = frozenset({"seed", "emotion_strength"})
    capabilities: ClassVar = frozenset({"emotion_audio"})
    sample_rate: ClassVar[int] = 22050
    speech_cps: ClassVar = {"en": 15.0}
    model_revision = "rev-9"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        pass

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        pass


def test_clone_card_stores_clips_with_origin(tmp_path):
    turns = [Turn(start=10, end=15, speaker="A"), Turn(start=20, end=26, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(10, 15, 0.01), (20, 26, 0.01)])
    picks = select_references(turns, "A", _transcript(turns), None, vocals)
    assert picks is not None
    store = CardStore(tmp_path / "ws")
    card = clone_card(CloneSynth, picks, store, vocals, "ja", "en", "S01E11")
    v = card.voice
    assert (v.backend, v.source, v.seed, v.model_revision) == ("clonefake", "clone", 1234, "rev-9")
    assert v.emotion_strength == 0.8 and card.target_language == "en"
    assert len(v.references) == len(picks)
    for ref, pick in zip(v.references, picks, strict=True):
        assert ref.language == "ja" and ref.transcript == pick.transcript
        assert ref.origin == {"episode": "S01E11", "start": pick.start, "end": pick.end}
        rate, data = wavfile.read(store.sidecar_path(ref.clip))
        assert rate == 24000 and data.ndim == 1
        assert len(data) / rate == pytest.approx(pick.end - pick.start, abs=0.01)
    # same inputs, same card (CD1): the clips' bytes and the id are reproducible
    again = clone_card(CloneSynth, picks, CardStore(tmp_path / "ws2"), vocals, "ja", "en", "S01E11")
    assert card_id(again) == card_id(card)
    assert parse_card(dump_card(card)) == card
    assert "seed = 1234" in dump_card(card) and "emotion_strength" not in dump_card(card)


def test_clone_card_records_reference_emotions_outside_the_id(tmp_path):
    turns = [Turn(start=0, end=5, speaker="A"), Turn(start=10, end=14, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.0005), (10, 14, 0.01)])
    tags = _tags([(0, 5, "angry", 0.8)], "emotion2vec")
    picks = select_references(turns, "A", _transcript(turns), tags, vocals)
    assert picks is not None
    store = CardStore(tmp_path / "ws")
    card = clone_card(CloneSynth, picks, store, vocals, "ja", "en", "S01E11")
    assert card.meta["reference_emotions"] == ["angry"]
    plain = clone_card(
        CloneSynth, [RefPick(p.start, p.end, p.transcript, p.snr_db) for p in picks], store,
        vocals, "ja", "en", "S01E11",
    )  # fmt: skip
    assert card_id(plain) == card_id(card) and "reference_emotions" not in plain.meta


def test_turns_inside_songs_are_not_references(tmp_path):
    """Separated theme-song vocals are clean and loud: without the song spans they would rank
    first (Fairy Tail E11's opening lyrics did, Task 9a)."""
    turns = [Turn(start=s, end=s + 5, speaker="A") for s in (0, 10, 20)]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.0005), (10, 15, 0.01), (20, 25, 0.01)])
    picks = select_references(turns, "A", _transcript(turns), None, vocals, exclude=[(-1, 6)])
    assert sorted(_spans(picks)) == [(10, 15), (20, 25)]


def test_reference_exclusions_are_songs_and_singing(tmp_path, monkeypatch):
    from fukikae import songs
    from fukikae.cards import select

    work = tmp_path / "ep"
    sep = work / "01-separate"
    sep.mkdir(parents=True)
    for n in ("background.wav", "vocals.wav", "vocals_16k.wav"):
        (sep / n).write_bytes(b"")
    monkeypatch.setattr(
        songs, "detect_theme_songs",
        lambda *_a, **_k: ([songs.SongSpan(65.0, 147.0, "detector", 0.9)], []),
    )  # fmt: skip
    tags = Tags(
        backend="sensevoice", emotions=[],
        events=[TagSpan(start=300, end=310, label="singing"), TagSpan(start=1, end=2, label="bgm")],
    )  # fmt: skip
    assert select.reference_exclusions(work, tags) == [(65.0, 147.0), (300.0, 310.0)]
    # no stems: only the tag stage's singing
    assert select.reference_exclusions(tmp_path / "none", tags) == [(300.0, 310.0)]


def test_eligible_references_are_exposed_ranked_and_picks_come_from_them(tmp_path):
    """Ruling CD17: the eligible list is public; select_references picks from it."""
    from fukikae.cards.select import eligible_references, pick_references

    starts = [0, 10, 20, 30, 40, 50]
    turns = [Turn(start=s, end=s + 6, speaker="A") for s in starts]
    turns.append(Turn(start=60, end=62, speaker="A"))  # too short: not eligible
    noise = [0.05, 0.01, 0.002, 0.03, 0.02, 0.04]
    vocals = _vocals(
        tmp_path / "v.wav", [(t.start, t.end, n) for t, n in zip(turns, noise, strict=False)]
    )
    tr = _transcript(turns)
    ranked = eligible_references(turns, "A", tr, None, vocals)
    assert [p.start for p in ranked] == [20, 10, 40, 30, 50, 0]  # quietest noise first
    assert pick_references(ranked) == select_references(turns, "A", tr, None, vocals)


def test_missing_stems_skip_song_exclusion_with_a_warning(tmp_path, caplog):
    from fukikae.cards.select import reference_exclusions

    work = tmp_path / "ep"
    (work / "01-separate").mkdir(parents=True)
    with caplog.at_level("WARNING", logger="fukikae.cards.select"):
        assert reference_exclusions(work, None) == []
    assert "theme songs are not excluded" in caplog.text and "background.wav" in caplog.text


def test_sparse_turns_are_not_references(tmp_path):
    """Task 12 (FT E12 character 3): a 5.2 s turn whose ASR text is one word ("くそ") is mostly
    someone else or no speech at all; a turn far sparser than the speaker's usual text rate,
    and sparse in absolute terms, is never a reference."""
    from fukikae.cards.select import eligible_references

    turns = [Turn(start=s, end=s + 5, speaker="A") for s in (0, 10, 20, 30)]
    vocals = _vocals(tmp_path / "v.wav", [(t.start, t.end, n) for t, n in zip(
        turns, (0.001, 0.01, 0.01, 0.01), strict=True)])  # fmt: skip
    texts = ["くそ", "俺の仲間はお前の器じゃないんだぞ", "あいつを倒せばこの眉取れるはずだから",
             "私のことはいいからさっきのやつ追いなさい"]  # fmt: skip
    segs = [
        Segment(start=t.start, end=t.end, text=x, words=[])
        for t, x in zip(turns, texts, strict=True)
    ]
    tr = Transcript(language="ja", backend="x", word_timing=False, segments=segs)
    ranked = eligible_references(turns, "A", tr, None, vocals)
    assert sorted(p.start for p in ranked) == [10, 20, 30]  # the cleanest turn, one word: gone


def test_sparse_check_needs_the_speaker_rate_too(tmp_path):
    """A speaker whose every turn is slow keeps them: the floor is relative and absolute."""
    from fukikae.cards.select import eligible_references

    turns = [Turn(start=s, end=s + 8, speaker="A") for s in (0, 10, 20)]
    vocals = _vocals(tmp_path / "v.wav", [(t.start, t.end, 0.01) for t in turns])
    segs = [Segment(start=t.start, end=t.end, text="ねえ", words=[]) for t in turns]
    tr = Transcript(language="ja", backend="x", word_timing=False, segments=segs)
    assert len(eligible_references(turns, "A", tr, None, vocals)) == 3


# -- D32: fragments joined, and the clone floor ------------------------------------------------


def _refs(turns, vocals, **kw):
    from fukikae.cards.select import eligible_references

    return eligible_references(turns, "A", _transcript(turns), None, vocals, **kw)


def test_fragments_of_one_speaker_are_joined_before_the_length_check(tmp_path):
    """D32 #1: two turns of the speaker 0.3 s apart, each under 3 s, are one 4.5 s turn."""
    turns = [Turn(start=0, end=2, speaker="A"), Turn(start=2.3, end=4.5, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 2, 0.01), (2.3, 4.5, 0.01)])
    (p,) = _refs(turns, vocals)
    assert (p.start, p.end) == (0.0, 4.5)
    assert p.transcript == "line at 0line at 2.3"
    assert _refs(turns, vocals, join_gap=0.0) == []  # the join is the gap setting's doing


def test_join_gap_is_0_6_s(tmp_path):
    from fukikae.cards.select import JOIN_GAP_S

    assert JOIN_GAP_S == 0.6
    turns = [Turn(start=0, end=2, speaker="A"), Turn(start=2.7, end=4.5, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 2, 0.01), (2.7, 4.5, 0.01)])
    assert _refs(turns, vocals) == []  # 0.7 s apart: two fragments, both too short


def test_another_speakers_turn_in_the_gap_blocks_the_join(tmp_path):
    turns = [
        Turn(start=0, end=2, speaker="A"),
        Turn(start=2.1, end=2.4, speaker="B"),
        Turn(start=2.5, end=4.5, speaker="A"),
    ]
    vocals = _vocals(tmp_path / "v.wav", [(0, 2, 0.01), (2.5, 4.5, 0.01)])
    assert _refs(turns, vocals) == []


def test_a_join_never_grows_past_the_longest_reference(tmp_path):
    """5 s and 8.7 s, 0.3 s apart, would be 14 s joined: they stay two references."""
    turns = [Turn(start=0, end=5, speaker="A"), Turn(start=5.3, end=14, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.01), (5.3, 14, 0.01)])
    assert sorted((p.start, p.end) for p in _refs(turns, vocals)) == [(0.0, 5.0), (5.3, 14.0)]


def test_a_fragment_overlapping_someone_else_never_sinks_its_clean_neighbour(tmp_path):
    turns = [
        Turn(start=0, end=4, speaker="A"),
        Turn(start=4.3, end=6, speaker="A"),
        Turn(start=5, end=7, speaker="B"),  # overlaps A's second fragment
    ]
    vocals = _vocals(tmp_path / "v.wav", [(0, 4, 0.01), (4.3, 6, 0.01)])
    assert [(p.start, p.end) for p in _refs(turns, vocals)] == [(0.0, 4.0)]


def test_a_joined_turn_that_fails_a_check_falls_back_to_its_pieces(tmp_path):
    """The gap is inside an excluded span (a song): the joined turn is refused, and the 4 s
    piece is still a reference on its own."""
    turns = [Turn(start=0, end=4, speaker="A"), Turn(start=4.4, end=6, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 4, 0.01), (4.4, 6, 0.01)])
    got = _refs(turns, vocals, exclude=[(4.1, 4.3)])
    assert [(p.start, p.end) for p in got] == [(0.0, 4.0)]


def test_foreign_speech_drops_a_turn_and_blocks_a_join(tmp_path):
    """``foreign`` (speech the subtitle names give someone else): a turn with 10% or more of
    it under that is no candidate, and a joined turn that much under it falls back."""
    turns = [Turn(start=0, end=4, speaker="A"), Turn(start=4.5, end=6, speaker="A"),
             Turn(start=10, end=15, speaker="A")]  # fmt: skip
    vocals = _vocals(tmp_path / "v.wav", [(0, 4, 0.01), (4.5, 6, 0.01), (10, 15, 0.01)])
    # 0.5 s of named speech in the gap: 8% of the 6 s joined turn passes
    assert [(p.start, p.end) for p in _refs(turns, vocals, foreign=[(4.0, 4.5)])][:1] == [
        (0.0, 6.0)
    ]
    # under someone else's name from 3.5 s: 0-4 alone is 12.5% foreign, so out, and with it
    # the join; 4.5-6 is too short alone
    got = _refs(turns, vocals, foreign=[(3.5, 4.5)])
    assert [(p.start, p.end) for p in got] == [(10.0, 15.0)]


def test_the_floor_takes_non_neutral_turns_until_two_clips_and_10_s(tmp_path):
    """D32 #2: one neutral 11 s turn is one clip; the angry turn makes the second."""
    from fukikae.cards.select import meets_floor

    turns = [Turn(start=0, end=11, speaker="A"), Turn(start=20, end=24, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 11, 0.01), (20, 24, 0.0005)])
    tags = _tags([(0, 11, "neutral", 0.9), (20, 24, "angry", 0.8)], "emotion2vec")
    picks = select_references(turns, "A", _transcript(turns), tags, vocals)
    assert _spans(picks) == [(0, 11), (20, 24)] and picks is not None and meets_floor(picks)


def test_the_floor_stops_taking_non_neutral_turns_once_reached(tmp_path):
    from fukikae.cards.select import meets_floor

    turns = [Turn(start=0, end=6, speaker="A"), Turn(start=10, end=15, speaker="A"),
             Turn(start=20, end=24, speaker="A")]  # fmt: skip
    vocals = _vocals(tmp_path / "v.wav", [(0, 6, 0.01), (10, 15, 0.01), (20, 24, 0.0005)])
    tags = _tags([(20, 24, "angry", 0.8)], "emotion2vec")
    picks = select_references(turns, "A", _transcript(turns), tags, vocals)
    assert sorted(_spans(picks)) == [(0, 6), (10, 15)] and meets_floor(picks or [])


def test_under_the_floor_the_best_picks_are_still_returned(tmp_path):
    """One 6.3 s turn is a clone below the floor: returned, and `meets_floor` says no."""
    from fukikae.cards.select import floor_shortfall, meets_floor

    turns = [Turn(start=0, end=6.3, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 6.3, 0.01)])
    picks = select_references(turns, "A", _transcript(turns), None, vocals)
    assert _spans(picks) == [(0, 6.3)] and picks is not None and not meets_floor(picks)
    assert (
        floor_shortfall([p.seconds for p in picks])
        == "1 clip and 6.3 s, under the floor of 2 clips and 10 s"
    )


def test_the_floor_is_judged_at_the_tenth_of_a_second_every_message_shows():
    """3 clips of 9.97 s print as 10.0 s: they reach the floor, as the report says."""
    from fukikae.cards.select import floor_shortfall

    assert floor_shortfall([3.526, 3.291, 3.156]) is None
    assert floor_shortfall([3.5, 3.0, 3.44]) == (
        "3 clips and 9.9 s, under the floor of 2 clips and 10 s"
    )
