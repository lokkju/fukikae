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
from tests.fixtures.speechlike import noise_bursts, speech, sung

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
    picks = select_references(turns, "A", _transcript(turns), None, vocals, scene_gap=None)
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
    picks = select_references(turns, "A", _transcript(turns), None, vocals, scene_gap=None)
    assert picks is not None and len(picks) == 3
    assert sum(p.end - p.start for p in picks) <= 20.0
    assert all(3.0 <= p.end - p.start <= 12.0 for p in picks)
    # 8 s turns: two fit under 20 s, a third would not
    turns8 = [Turn(start=s, end=s + 8, speaker="A") for s in (0, 10, 20)]
    vocals = _vocals(tmp_path / "v8.wav", [(t.start, t.end, 0.01) for t in turns8])
    picks8 = select_references(turns8, "A", _transcript(turns8), None, vocals, scene_gap=None)
    assert len(_spans(picks8)) == 2


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
    assert card_id(plain) == card_id(card) and plain.meta["reference_emotions"] == []
    # written empty, so a re-put of the same id over the emotional one (CardStore.put merges
    # [meta]) doesn't keep a stale label
    store.put(card)
    store.put(plain)
    assert store.get(card_id(card)).meta["reference_emotions"] == []


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


# -- D32 review: the clip cap, the in-fill floor test, joins split at the failing gap ---------


def _p(start, seconds, snr, emotion=None):
    from fukikae.cards.select import RefPick

    return RefPick(start, round(start + seconds, 3), "x", snr, emotion)


def _ranked(picks):
    from fukikae.cards.select import rank_key

    return sorted(picks, key=rank_key)


def test_the_clip_cap_never_stops_short_of_a_floor_the_episode_reaches():
    """Erza's shape: six non-neutral turns, the greedy three 9.6 s; swapping the third for the
    4.29 s turn reaches the floor, keeping the two best-ranked."""
    from fukikae.cards.select import meets_floor, pick_references

    ranked = _ranked([
        _p(948.457, 3.105, 44.1, "disgusted"), _p(917.086, 3.021, 31.75, "happy"),
        _p(935.075, 3.51, 28.98, "happy"), _p(939.243, 4.286, 27.7, "sad"),
        _p(711.61, 3.162, 25.83, "angry"), _p(900.38, 3.105, 22.21, "disgusted"),
    ])  # fmt: skip
    picks = pick_references(ranked, scene_gap=None)  # one scene: the D32 search alone
    assert picks is not None and meets_floor(picks)
    assert [p.start for p in picks] == [948.457, 917.086, 939.243]
    assert round(sum(p.seconds for p in picks), 1) == 10.4


def test_the_floor_wins_over_a_neutral_pick_when_nothing_else_reaches_it():
    """Gray's shape: no three clips that keep both neutral turns reach 10 s, so the neutral
    3.06 s turn makes way for the two longest angry ones."""
    from fukikae.cards.select import meets_floor, pick_references

    ranked = _ranked([
        _p(1167.595, 3.055, 52.54), _p(759.997, 3.341, 25.38),
        _p(941.352, 3.122, 46.45, "angry"), _p(965.973, 3.206, 37.92, "angry"),
        _p(753.28, 3.46, 28.57, "angry"),
    ])  # fmt: skip
    picks = pick_references(ranked)
    assert picks is not None and meets_floor(picks)
    assert [p.start for p in picks] == [759.997, 965.973, 753.28]


def test_as_few_neutral_picks_as_possible_give_way_the_lowest_ranked_first():
    """Three neutral 3 s turns fill the cap at 9 s: one of them (the lowest-ranked) makes way
    for the 4.5 s angry turn."""
    from fukikae.cards.select import pick_references

    ranked = _ranked([
        _p(0, 3.0, 50), _p(10, 3.0, 40), _p(20, 3.0, 30),  # neutral: 9 s greedy
        _p(30, 4.5, 45, "angry"),
    ])  # fmt: skip
    picks = pick_references(ranked)
    assert picks is not None and [p.start for p in picks] == [0, 10, 30]


def test_an_episode_short_of_the_floor_keeps_the_greedy_picks():
    from fukikae.cards.select import meets_floor, pick_references

    ranked = _ranked([_p(0, 3.0, 50), _p(10, 3.1, 40), _p(20, 3.2, 30, "angry"),
                      _p(30, 3.0, 20, "happy")])  # fmt: skip
    picks = pick_references(ranked)
    assert picks is not None and [p.start for p in picks] == [0, 10, 20]
    assert not meets_floor(picks)


def test_two_neutral_clips_at_9_96_s_reach_the_floor_and_take_no_filler():
    """The in-fill test judges the floor at a tenth, as `floor_shortfall` does."""
    from fukikae.cards.select import pick_references

    ranked = _ranked([_p(0, 5.0, 50), _p(10, 4.96, 40), _p(20, 3.0, 60, "angry")])
    picks = pick_references(ranked)
    assert picks is not None and [p.start for p in picks] == [0, 10]


def test_the_floor_search_runs_when_the_seconds_cap_skipped_a_turn():
    """12 s taken first leaves no room under 20 s for any other turn: one clip, under the
    floor. The search finds the 9 + 9.5 s pair that reaches it."""
    from fukikae.cards.select import meets_floor, pick_references

    ranked = _ranked([_p(0, 12.0, 50), _p(100, 9.0, 40), _p(200, 9.5, 30), _p(300, 11.0, 20)])
    picks = pick_references(ranked)
    assert picks is not None and meets_floor(picks)
    assert [p.start for p in picks] == [100, 200]


def test_a_run_of_three_that_fails_splits_at_the_failing_gap(tmp_path):
    """The gap between the second and third pieces is inside a song: the first two still join,
    and the third is a reference alone."""
    turns = [Turn(start=0, end=2, speaker="A"), Turn(start=2.3, end=4.5, speaker="A"),
             Turn(start=4.9, end=8, speaker="A")]  # fmt: skip
    vocals = _vocals(tmp_path / "v.wav", [(0, 2, 0.01), (2.3, 4.5, 0.01), (4.9, 8, 0.01)])
    got = _refs(turns, vocals, exclude=[(4.6, 4.8)])
    assert sorted((p.start, p.end) for p in got) == [(0.0, 4.5), (4.9, 8.0)]


def test_a_join_never_strands_a_short_third_piece_another_split_keeps(tmp_path):
    """5 s, 4.7 s and 2.5 s pieces 0.3 s apart are 12.8 s together: joining the last two keeps
    every piece, where joining the first two would leave 2.5 s alone."""
    turns = [Turn(start=0, end=5, speaker="A"), Turn(start=5.3, end=10, speaker="A"),
             Turn(start=10.3, end=12.8, speaker="A")]  # fmt: skip
    vocals = _vocals(tmp_path / "v.wav", [(0, 5, 0.01), (5.3, 10, 0.01), (10.3, 12.8, 0.01)])
    got = _refs(turns, vocals)
    assert sorted((p.start, p.end) for p in got) == [(0.0, 5.0), (5.3, 12.8)]
    assert _refs(turns, vocals) == got  # deterministic


def test_a_join_spans_two_labels_of_one_character(tmp_path):
    from fukikae.cards.select import eligible_references

    turns = [Turn(start=0, end=2, speaker="A"), Turn(start=2.3, end=4.5, speaker="B")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 2, 0.01), (2.3, 4.5, 0.01)])
    got = eligible_references(turns, {"A", "B"}, _transcript(turns), None, vocals)
    assert [(p.start, p.end) for p in got] == [(0.0, 4.5)]


def test_overlapping_turns_of_the_speaker_never_make_overlapping_references(tmp_path):
    """Two overlapping turns are one reference over both; when that would pass 12 s, one of
    them is a reference and the other is dropped, never both."""
    turns = [Turn(start=0, end=4, speaker="A"), Turn(start=3, end=7, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(0, 7, 0.01)])
    assert [(p.start, p.end) for p in _refs(turns, vocals)] == [(0.0, 7.0)]

    turns = [Turn(start=0, end=8, speaker="A"), Turn(start=6, end=13, speaker="A")]
    vocals = _vocals(tmp_path / "v2.wav", [(0, 13, 0.01)])
    assert [(p.start, p.end) for p in _refs(turns, vocals)] == [(0.0, 8.0)]


def test_speaker_f0_returns_the_median_now(tmp_path):
    """3 s at 130 Hz and 7 s at 260 Hz: the median reads 260 (the old 20th percentile read 130)."""
    from fukikae.cards.select import speaker_f0

    t3, t7 = np.arange(3 * RATE) / RATE, np.arange(7 * RATE) / RATE
    x = np.concatenate(
        [sum(np.sin(2 * np.pi * 130 * k * t3) / k for k in range(1, 6)),
         sum(np.sin(2 * np.pi * 260 * k * t7) / k for k in range(1, 6))]
    )  # fmt: skip
    wav_260 = tmp_path / "v.wav"
    wavfile.write(wav_260, RATE, (0.2 * x * 32767).astype("<i2"))
    f0, seconds = speaker_f0(wav_260, [(0, 10)])
    assert f0 is not None and abs(f0 - 260) < 8
    assert seconds == 10.0


# -- A1: reference gates (pitch register, voicing, song-like audio, one scene) ---------------


def _harmonic(hz: float, n: int) -> np.ndarray:
    t = np.arange(n) / RATE
    x = sum(a * np.sin(2 * np.pi * hz * k * t) for k, a in enumerate((1, 0.5, 0.33, 0.25), 1))
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def _voices(path: Path, spans: list[tuple[float, float, float | None, float]], seconds=600.0):
    """Silence with, per (start, end, hz, noise) span, noise of amplitude ``noise`` and over
    its middle 80% a harmonic tone at ``hz`` (loud noise instead when ``hz`` is None)."""
    rng = np.random.default_rng(1)
    x = np.zeros(int(seconds * RATE), np.float32)
    for start, end, hz, noise in spans:
        a, b = int(start * RATE), int(end * RATE)
        x[a:b] = noise * rng.standard_normal(b - a)
        lo, hi = a + int(0.1 * (b - a)), a + int(0.9 * (b - a))
        x[lo:hi] = _harmonic(hz, hi - lo) if hz else 0.3 * rng.standard_normal(hi - lo)
    wavfile.write(path, RATE, (np.clip(x, -1, 1) * 32767).astype("<i2"))
    return path


def _gate_case(tmp_path, spans):
    turns = [Turn(start=a, end=b, speaker="A") for a, b, _, _ in spans]
    return turns, _transcript(turns), _voices(tmp_path / "v.wav", spans)


def test_off_register_candidate_is_gated(tmp_path):
    """Character 1's shape: speech at 264 Hz, a full 4 st inside the female band; a clean,
    loud 175 Hz turn (the narration, between the bands) would rank first by SNR."""
    from fukikae.cards.select import Gated, eligible_references

    spans = [(10, 18, 175.0, 0.0005)]
    spans += [(100 + 20 * i, 106 + 20 * i, 264.0, 0.01) for i in range(4)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    assert eligible_references(turns, "A", tr, None, vocals)[0].start == 10  # ungated
    gated: list[Gated] = []
    picks = eligible_references(turns, "A", tr, None, vocals, register=("f", 264.0), gated=gated)
    assert sorted(p.start for p in picks) == [100, 120, 140, 160]
    assert gated == [Gated(off_register=1, unvoiced=0, song_like=0, top=1)]


def test_register_refuses_only_cross_class_candidates(tmp_path):
    """The register gate refuses the other band, and the ambiguous band only under a register
    a full 4 st inside its own; a same-band candidate passes however far from the median
    (the old 4 st rule refused the 360 Hz turn below)."""
    from fukikae.cards.select import RefVerdict, eligible_references

    spans = [(10, 16, 140.0, 0.01), (30, 36, 180.0, 0.01), (50, 56, 230.0, 0.01)]
    spans += [(70, 76, 360.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    verdicts: list[RefVerdict] = []
    eligible_references(turns, "A", tr, None, vocals, register=("f", 264.0), verdicts=verdicts)
    by = {v.pick.start: v.gates for v in verdicts}
    assert by == {10: ("off_register",), 30: ("off_register",), 50: (), 70: ()}
    verdicts = []  # a register near the band edge: 215 Hz is 1.3 st inside it
    eligible_references(turns, "A", tr, None, vocals, register=("f", 215.0), verdicts=verdicts)
    by = {v.pick.start: v.gates for v in verdicts}
    assert by == {10: ("off_register",), 30: (), 50: (), 70: ()}


def test_mixed_cluster_uses_its_majority_register(tmp_path):
    """One loud male turn (4 s, under the bimodal minimum) among five female ones: the
    register is the female majority's, so only female turns are references."""
    from fukikae.cards.select import eligible_references, speaker_register

    spans = [(10, 14, 120.0, 0.0005)]
    spans += [(100 + 20 * i, 106 + 20 * i, 240.0, 0.01) for i in range(5)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    register = speaker_register(vocals, [(t.start, t.end) for t in turns], (165.0, 200.0))
    assert register.gender == "f" and register.median == pytest.approx(240, rel=0.03)
    assert not register.bimodal
    picks = eligible_references(turns, "A", tr, None, vocals, register=register)
    assert picks and all(p.start >= 100 for p in picks)


def test_dominant_bimodal_cluster_keeps_the_gate_and_needs_a_split(tmp_path):
    """Controller ruling (a), character 7's shape: two male turns (12 s) against eight female
    (48 s), 4:1: bimodal, so flagged as needing a split, but the majority has at least twice
    the minority's seconds, so the cross-class gate refuses the male candidates."""
    from fukikae.cards.select import (
        SPLIT_GATED_NOTE,
        Gated,
        eligible_references,
        speaker_register,
    )

    spans = [(10, 16, 120.0, 0.0005), (30, 36, 125.0, 0.0005)]
    spans += [(100 + 20 * i, 106 + 20 * i, 240.0, 0.01) for i in range(8)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    register = speaker_register(vocals, [(t.start, t.end) for t in turns], (165.0, 200.0))
    assert register.gender == "f" and register.bimodal and not register.even
    gated: list[Gated] = []
    picks = eligible_references(turns, "A", tr, None, vocals, register=register, gated=gated)
    assert picks and not {10, 30} & {p.start for p in picks}
    assert gated[0].split and not gated[0].even and gated[0].off_register == 2
    assert SPLIT_GATED_NOTE in reference_report(picks, gated[0])


def test_near_even_bimodal_cluster_gates_no_pitch_and_needs_a_split(tmp_path):
    """FT E14 Nebaru's shape: four male turns (20 s) against three female (21 s), 1.05:1:
    two voices of about equal weight, so the register gate is skipped and the speaker is
    flagged as needing a split."""
    from fukikae.cards.select import SPLIT_NOTE, Gated, eligible_references, speaker_register

    spans = [(10 + 20 * i, 15 + 20 * i, 120.0, 0.0005) for i in range(4)]
    spans += [(200 + 20 * i, 207 + 20 * i, 240.0, 0.01) for i in range(3)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    register = speaker_register(vocals, [(t.start, t.end) for t in turns], (165.0, 200.0))
    assert register.gender == "f" and register.bimodal and register.even
    gated: list[Gated] = []
    picks = eligible_references(turns, "A", tr, None, vocals, register=register, gated=gated)
    assert {10, 30, 50, 70} & {p.start for p in picks}
    assert gated[0].split and gated[0].even and gated[0].off_register == 0
    assert SPLIT_NOTE in reference_report(picks, gated[0])


def test_unknown_register_gates_no_pitch(tmp_path):
    from fukikae.cards.select import eligible_references

    spans = [(10, 16, 120.0, 0.0005), (100, 106, 240.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    picks = eligible_references(turns, "A", tr, None, vocals, register=("u", 180.0))
    assert [p.start for p in picks] == [10, 100]


def test_unvoiced_candidate_is_gated(tmp_path):
    """Character 2's processed voice: a loud turn with a transcript and no voiced frames."""
    from fukikae.cards.select import Gated, eligible_references

    spans = [(10, 16, None, 0.0005), (100, 106, 240.0, 0.01), (130, 136, 240.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    assert eligible_references(turns, "A", tr, None, vocals)[0].start == 10
    gated: list[Gated] = []
    picks = eligible_references(turns, "A", tr, None, vocals, register=("f", 240.0), gated=gated)
    assert [p.start for p in picks] == [100, 130]
    assert all(p.voiced_fraction is not None and p.voiced_fraction >= 0.5 for p in picks)
    assert gated[0].unvoiced == 1 and gated[0].top == 1


def _clips(path: Path, clips: list[tuple[float, np.ndarray]], seconds: float = 600.0) -> Path:
    x = np.zeros(int(seconds * RATE), np.float32)
    for start, clip in clips:
        a = int(start * RATE)
        x[a : a + len(clip)] = clip
    wavfile.write(path, RATE, (np.clip(x, -1, 1) * 32767).astype("<i2"))
    return path


def test_song_like_candidate_is_gated_outside_theme_spans(tmp_path):
    """Character 3's insert song: a sung turn mid-episode, in no theme span, under the song
    mask; dialogue under the same mask (speech over a loud bed) is not sung and stays."""
    from fukikae.cards.select import RefVerdict, eligible_references

    turns = [Turn(start=s, end=s + 8, speaker="A") for s in (100, 300, 400)]
    vocals = _clips(
        tmp_path / "v.wav",
        [(100, speech(400, 8, seed=1)), (300, sung(400, 8)), (400, speech(400, 8, seed=2))],
    )
    mask = np.zeros(600, bool)
    mask[298:410] = True  # the song and the dialogue at 400 s are both over a loud bed
    verdicts: list[RefVerdict] = []
    picks = eligible_references(
        turns, "A", _transcript(turns), None, vocals, song_like=mask, verdicts=verdicts
    )
    assert sorted(p.start for p in picks) == [100, 400]
    by = {v.pick.start: v for v in verdicts}
    assert by[300].gates == ("song_like",) and by[300].pick.sung
    assert by[400].pick.song_share == 1.0 and not by[400].pick.sung


def test_candidate_mostly_inside_an_accepted_song_span_is_song_like(tmp_path):
    """Controller ruling (b): a candidate half or more inside a song span the song detector
    accepted is refused by gate 3, sung or not (the detector already judged the span sung);
    one under half inside is judged by the mask and sung-ness alone."""
    from fukikae.cards.select import RefVerdict, eligible_references

    turns = [Turn(start=s, end=s + 8, speaker="A") for s in (100, 300, 400)]
    vocals = _clips(
        tmp_path / "v.wav",
        [(100, speech(400, 8, seed=1)), (300, speech(400, 8, seed=3)),
         (400, speech(400, 8, seed=2))],
    )  # fmt: skip
    verdicts: list[RefVerdict] = []
    picks = eligible_references(
        turns, "A", _transcript(turns), None, vocals, song_spans=[(296.0, 304.0), (405.0, 420.0)],
        verdicts=verdicts,
    )  # fmt: skip
    by = {v.pick.start: v for v in verdicts}
    assert by[300].gates == ("song_like",) and not by[300].pick.sung  # 4 of 8 s inside
    assert by[400].gates == ()  # 3 of 8 s inside
    assert sorted(p.start for p in picks) == [100, 400]


def test_song_span_gate_reads_the_detectors_accepted_spans(tmp_path, monkeypatch):
    """``select.song_spans``: the theme detector's accepted spans, and nothing without the
    stems (the singing tags are no part of it)."""
    from fukikae import songs
    from fukikae.cards import select

    work = tmp_path
    sep = work / "01-separate"
    sep.mkdir()
    for name in ("background.wav", "vocals.wav", "vocals_16k.wav"):
        (sep / name).write_bytes(b"")
    monkeypatch.setattr(
        songs, "detect_theme_songs",
        lambda *_a, **_k: ([songs.SongSpan(65.0, 147.0, "detector", 0.9)], []),
    )  # fmt: skip
    assert select.song_spans(work) == [(65.0, 147.0)]
    assert select.song_spans(tmp_path / "none") == []


def test_verdicts_name_every_gate_per_turn(tmp_path):
    from fukikae.cards.select import RefVerdict, eligible_references

    spans = [(10, 16, None, 0.0005), (30, 38, 175.0, 0.0005), (100, 106, 264.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    verdicts: list[RefVerdict] = []
    eligible_references(turns, "A", tr, None, vocals, register=("f", 264.0), verdicts=verdicts)
    by = {v.pick.start: v for v in verdicts}
    assert by[10].gates == ("unvoiced",) and not by[10].passed
    assert by[30].gates == ("off_register",)
    assert by[100].gates == () and by[100].passed and by[100].pick.median is not None
    assert [v.rank for v in verdicts] == [0, 1, 2]


def test_gates_do_not_reorder_survivors(tmp_path):
    """CD5: the gates filter; survivors keep the ungated order."""
    from fukikae.cards.select import eligible_references

    spans = [(100 + 20 * i, 106 + 20 * i, 240.0, n) for i, n in enumerate((0.03, 0.001, 0.01))]
    spans += [(10, 16, 120.0, 0.0001)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    plain = [p.start for p in eligible_references(turns, "A", tr, None, vocals)]
    gated = [
        p.start for p in eligible_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    ]
    assert gated == [s for s in plain if s != 10]


def test_gates_keep_the_foreign_rule(tmp_path):
    """A turn a tenth under someone else's names stays out whatever the gates say."""
    from fukikae.cards.select import eligible_references

    spans = [(100, 106, 240.0, 0.001), (130, 136, 240.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    picks = eligible_references(
        turns, "A", tr, None, vocals, register=("f", 240.0), foreign=[(100.0, 101.0)]
    )
    assert [p.start for p in picks] == [130]


def test_the_floor_search_reads_only_gate_survivors(tmp_path):
    """The clip cap and floor search (D32) run on what the gates leave: three loud off-register
    turns would fill the cap; the survivors are searched for the floor instead."""
    from fukikae.cards.select import eligible_references, meets_floor, pick_references

    spans = [(10 + 70 * i, 14 + 70 * i, 150.0, 0.0002) for i in range(3)]
    spans += [(300 + 70 * i, 303.2 + 70 * i, 240.0, 0.01) for i in range(3)]
    spans += [(520, 525, 240.0, 0.03)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    ranked = eligible_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    picks = pick_references(ranked)
    assert picks is not None and meets_floor(picks)
    # 9.6 s from the three 3.2 s survivors; the floor search swaps one for the 5 s turn
    assert all(p.start >= 300 for p in picks) and any(p.start == 520 for p in picks)


def test_one_scene_cannot_fill_every_slot():
    from fukikae.cards.select import pick_references

    ranked = [_p(30, 7, 60), _p(40, 6, 50), _p(60, 6, 40), _p(400, 5, 30)]  # 3 within 60 s
    picks = pick_references(ranked, count=3)
    assert picks is not None and [p.start for p in picks][:2] == [30, 400]


def test_same_scene_is_used_when_the_floor_needs_it():
    from fukikae.cards.select import meets_floor, pick_references

    ranked = [_p(30, 6, 60), _p(40, 4, 50), _p(400, 3, 30)]  # 9 s from two scenes
    picks = pick_references(ranked, count=3)
    assert picks is not None and meets_floor(picks)
    assert [p.start for p in picks] == [30, 40, 400]  # in rank order


def test_scene_gate_off_for_narrators():
    from fukikae.cards.select import pick_references

    ranked = [_p(30, 7, 60), _p(40, 6, 50), _p(60, 6, 40), _p(400, 5, 30)]
    picks = pick_references(ranked, count=3, scene_gap=None)
    assert picks is not None and [p.start for p in picks] == [30, 40, 60]


def test_scene_gate_prefers_distinct_scenes_in_the_floor_search():
    """The cap stops the fill short: the floor combination takes distinct scenes when one
    reaches the floor, and a repeat only when none does."""
    from fukikae.cards.select import pick_references

    ranked = _ranked([
        _p(0, 3.0, 50, "angry"), _p(10, 4.0, 49, "angry"),
        _p(100, 3.0, 40, "sad"), _p(200, 3.5, 30, "sad"),
    ])  # fmt: skip
    # distinct scenes give 0, 100 and 200: 9.5 s; of the three-clip sets that reach 10 s,
    # only 10, 100 and 200 keeps one clip per scene
    picks = pick_references(ranked)
    assert picks is not None and [p.start for p in picks] == [10, 100, 200]
    found = pick_references(ranked, scene_gap=None)
    assert found is not None and [p.start for p in found] == [0, 10, 100]


def test_report_names_gated_top_candidates():
    from fukikae.cards.select import Gated

    assert reference_report([], Gated(off_register=2, unvoiced=0, song_like=1, top=2)) == [
        "gated: 2 off-register, 1 song-like"
    ]
    assert reference_report([], Gated(off_register=2, unvoiced=0, song_like=0, top=0)) == []
    assert reference_report([]) == []


def test_reference_gating_reads_the_song_mask_from_the_stems(tmp_path):
    from fukikae.cards.select import reference_gating

    sep = tmp_path / "01-separate"
    sep.mkdir()
    rate = 8000
    t = np.arange(4 * rate) / rate
    bg = 0.1 * np.sin(2 * np.pi * 200 * t)
    vo = np.where(t < 2, 0.1 * np.sin(2 * np.pi * 300 * t), 0.0)
    for name, x in (("background.wav", bg), ("vocals.wav", vo)):
        wavfile.write(sep / name, rate, (x * 32767).astype("<i2"))
    mask = reference_gating(tmp_path)
    assert mask is not None and list(mask) == [True, True, False, False]
    (sep / "background.wav").unlink()
    assert reference_gating(tmp_path) is None


def test_a_join_with_gaps_passes_the_voicing_gate(tmp_path):
    """Two turns of speech with pauses, 0.5 s apart, joined into one candidate (D32 #1): its
    voicing is read over its turns' loud frames, not over the join's gap and the pauses."""
    from fukikae.cards.select import RefVerdict, eligible_references

    turns = [Turn(start=100, end=103.2, speaker="A"), Turn(start=103.7, end=107.0, speaker="A")]
    vocals = _clips(
        tmp_path / "v.wav",
        [(100, speech(240, 3.2, seed=3, voiced_share=0.35)),
         (103.7, speech(240, 3.3, seed=4, voiced_share=0.35))],
    )  # fmt: skip
    verdicts: list[RefVerdict] = []
    picks = eligible_references(
        turns, "A", _transcript(turns), None, vocals, register=("f", 240.0), verdicts=verdicts
    )
    assert [(p.start, p.end) for p in picks] == [(100.0, 107.0)]
    from fukikae import pitch

    whole = pitch.summarize(pitch.f0_track(vocals, [(100.0, 107.0)]), 0)
    assert whole.voiced_fraction < 0.5  # what the gate read before: refused
    vf = verdicts[0].pick.voiced_fraction
    assert verdicts[0].passed and vf is not None and vf >= 0.7


def test_speech_with_pauses_passes_the_voicing_gate(tmp_path):
    """Erza's E13 shape: calm speech, mostly pauses, refused as unvoiced when the pauses
    counted against it."""
    from fukikae.cards.select import eligible_references

    turns = [Turn(start=100, end=106, speaker="A"), Turn(start=200, end=206, speaker="A")]
    vocals = _clips(
        tmp_path / "v.wav",
        [(100, speech(240, 6, seed=5, voiced_share=0.3)), (200, speech(240, 6, seed=6))],
    )
    picks = eligible_references(turns, "A", _transcript(turns), None, vocals, register=("f", 240.0))
    assert sorted(p.start for p in picks) == [100, 200]


def test_whisper_like_candidate_is_gated_as_unvoiced(tmp_path):
    from fukikae.cards.select import RefVerdict, eligible_references

    turns = [Turn(start=100, end=106, speaker="A"), Turn(start=200, end=206, speaker="A")]
    vocals = _clips(tmp_path / "v.wav", [(100, noise_bursts(6)), (200, speech(240, 6))])
    verdicts: list[RefVerdict] = []
    picks = eligible_references(
        turns, "A", _transcript(turns), None, vocals, register=("f", 240.0), verdicts=verdicts
    )
    assert [p.start for p in picks] == [200]
    assert {v.pick.start: v.gates for v in verdicts}[100] == ("unvoiced",)


def test_a_verdict_does_not_depend_on_the_other_candidates(tmp_path):
    """A quiet speaker's turn read alone, and beside a loud turn, measures the same: each
    candidate is its own loudness group."""
    from fukikae.cards.select import RefVerdict, eligible_references

    quiet = 0.02 * speech(240, 6, seed=7) / 0.25
    vocals = _clips(tmp_path / "v.wav", [(100, quiet), (200, speech(240, 6, seed=8))])
    alone = [Turn(start=100, end=106, speaker="A")]
    both = [*alone, Turn(start=200, end=206, speaker="A")]
    got: list[list[RefVerdict]] = []
    for turns in (alone, both):
        verdicts: list[RefVerdict] = []
        eligible_references(
            turns, "A", _transcript(turns), None, vocals, register=("f", 240.0),
            verdicts=verdicts,
        )  # fmt: skip
        got.append([v for v in verdicts if v.pick.start == 100])
    assert got[0][0].pick == got[1][0].pick and got[0][0].gates == got[1][0].gates == ()


def test_the_floor_search_prefers_neutral_clips_to_a_new_scene():
    """Fewest scored non-neutral clips first: an all-neutral set reaching the floor with a
    second clip of a scene beats one with an angry clip from a new scene."""
    from fukikae.cards.select import pick_references

    ranked = _ranked([
        _p(0, 4.0, 40), _p(20, 5.0, 39), _p(300, 3.0, 38), _p(600, 2.9, 37),
        _p(900, 6.0, 45, "angry"),
    ])  # fmt: skip
    picks = pick_references(ranked)
    assert picks is not None and all(p.emotion is None for p in picks)
    assert [p.start for p in picks] == [20, 300, 600]  # one per scene; the old key took 900


# -- A2: reference tiers (floor, short, thin; D41: never a stock voice) ----------------------


def _overlaps(p, span: tuple[float, float]) -> bool:
    return p.start < span[1] and span[0] < p.end


def test_floor_tier_when_the_floor_is_reached(tmp_path):
    from fukikae.cards.select import FLOOR_S, tiered_references

    spans = [(100 + 80 * i, 106 + 80 * i, 240.0, 0.01) for i in range(3)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    tier, picks = tiered_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    assert tier == "floor" and len(picks) >= 2 and sum(p.seconds for p in picks) >= FLOOR_S


def test_short_tier_between_three_seconds_and_the_floor(tmp_path):
    """4 s of clean, gate-passing speech: a clone from it, below the floor (no more 6 s
    minimum, which only chose between a clone and a stock voice)."""
    from fukikae.cards.select import CLONE_MIN_S, meets_floor, tiered_references

    turns, tr, vocals = _gate_case(tmp_path, [(100, 104, 240.0, 0.01)])
    tier, picks = tiered_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    assert tier == "short" and sum(p.seconds for p in picks) >= CLONE_MIN_S
    assert not meets_floor(picks)


def test_thin_tier_sets_the_gates_aside_but_not_songs_or_foreign_names(tmp_path):
    """2.1 s of speech in three short turns: an off-register one, one inside a theme song
    and one under another character's subtitle name. The thin tier takes the off-register
    turn (the gates are set aside) and never the song or the foreign one."""
    from fukikae.cards.select import tiered_references

    spans = [(10, 11.2, 120.0, 0.01), (30, 31.5, 240.0, 0.01), (50, 50.9, 240.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    theme, foreign = (29.0, 40.0), (50.0, 51.0)
    tier, picks = tiered_references(
        turns, "A", tr, None, vocals, register=("f", 240.0), theme_spans=[theme],
        foreign=[foreign],
    )  # fmt: skip
    assert tier == "thin"
    assert not any(_overlaps(p, theme) or _overlaps(p, foreign) for p in picks)
    assert [p.start for p in picks] == [10.0]


def test_thin_tier_never_returns_nothing_for_a_speaker_with_turns(tmp_path):
    """Every turn inside a song: the speaker's own turns are used regardless."""
    from fukikae.cards.select import tiered_references

    turns, tr, vocals = _gate_case(tmp_path, [(30, 31.5, 240.0, 0.01), (33, 34, 240.0, 0.01)])
    tier, picks = tiered_references(
        turns, "A", tr, None, vocals, register=("f", 240.0), theme_spans=[(0.0, 100.0)]
    )
    assert tier == "thin" and [(p.start, p.end) for p in picks] == [(30.0, 31.5), (33.0, 34.0)]


def test_thin_tier_stops_at_fifteen_seconds(tmp_path):
    """All off-register, so no gated pick: the longest turns first, whole, up to 15 s; a turn
    that doesn't fit is passed over for a shorter one that does."""
    from fukikae.cards.select import THIN_MAX_S, tiered_references

    spans = [(10, 18, 120.0, 0.01), (100, 107, 120.0, 0.01), (200, 206, 120.0, 0.01)]
    spans += [(300, 301, 120.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    tier, picks = tiered_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    assert tier == "thin" and THIN_MAX_S == 15.0
    # 8 + 7 s is 15 s of speech but 15.2 s of prompt with IndexTTS2's 0.2 s join
    assert [(p.start, p.end) for p in picks] == [(10.0, 18.0), (200.0, 206.0)]


def test_thin_tier_cuts_one_long_turn_to_fifteen_seconds(tmp_path):
    from fukikae.cards.select import tiered_references

    turns, tr, vocals = _gate_case(tmp_path, [(10, 30, 120.0, 0.01)])
    tier, picks = tiered_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    assert tier == "thin" and [(p.start, p.end) for p in picks] == [(10.0, 25.0)]


def test_thin_tier_young_rob(tmp_path):
    """Young Rob's shape (Fairy Tail E12, times less 500 s): six turns, two of them 17 ms
    blips cut from the end of a 0.79 s turn. The blips join that turn, and the three longest
    pieces are taken: at most 3 clips."""
    from fukikae.cards.select import THIN_MAX_CLIPS, tiered_references

    spans = [
        (20.355, 22.397, 120.0, 0.01), (58.104, 59.64, 120.0, 0.01),
        (64.01, 65.208, 120.0, 0.01), (39.44, 40.233, 120.0, 0.01),
        (40.233, 40.25, 120.0, 0.01), (40.25, 40.267, 120.0, 0.01),
    ]  # fmt: skip
    turns, tr, vocals = _gate_case(tmp_path, spans)
    tier, picks = tiered_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    assert tier == "thin" and THIN_MAX_CLIPS == 3
    assert [(p.start, p.end) for p in picks] == [
        (20.355, 22.397), (58.104, 59.64), (64.01, 65.208)
    ]  # fmt: skip


def test_thin_tier_joins_contiguous_turns_and_near_ones(tmp_path):
    """Turns touching or at most 0.6 s apart are one piece; another speaker's turn in the gap
    keeps them apart."""
    from fukikae.cards.select import thin_references

    spans = [(10, 10.4, 120.0, 0.01), (10.4, 11.0, 120.0, 0.01), (11.5, 12.0, 120.0, 0.01)]
    spans += [(30, 30.8, 120.0, 0.01), (31.2, 32.0, 120.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    turns.append(Turn(start=30.9, end=31.1, speaker="B"))
    picks = thin_references(turns, "A", tr, vocals)
    assert [(p.start, p.end) for p in picks] == [(10.0, 12.0), (30.0, 30.8), (31.2, 32.0)]


def test_thin_tier_skips_pieces_under_half_a_second_unless_nothing_else_is_left(tmp_path):
    from fukikae.cards.select import thin_references

    turns, tr, vocals = _gate_case(tmp_path, [(10, 10.3, 120.0, 0.01), (20, 21, 120.0, 0.01)])
    assert [(p.start, p.end) for p in thin_references(turns, "A", tr, vocals)] == [(20.0, 21.0)]
    turns, tr, vocals = _gate_case(tmp_path, [(10, 10.3, 120.0, 0.01), (20, 20.2, 120.0, 0.01)])
    picks = thin_references(turns, "A", tr, vocals)
    assert [(p.start, p.end) for p in picks] == [(10.0, 10.3), (20.0, 20.2)]


def test_thin_tier_counts_the_prompt_join_in_its_budget(tmp_path):
    """5 + 5 + 4.8 s is 14.8 s of speech and 15.2 s of prompt: the third piece doesn't fit."""
    from fukikae.cards.select import PROMPT_GAP_S, thin_references

    spans = [(10, 15, 120.0, 0.01), (20, 25, 120.0, 0.01), (30, 34.8, 120.0, 0.01)]
    turns, tr, vocals = _gate_case(tmp_path, spans)
    assert PROMPT_GAP_S == 0.2
    picks = thin_references(turns, "A", tr, vocals)
    assert [(p.start, p.end) for p in picks] == [(10.0, 15.0), (20.0, 25.0)]


def test_thin_picks_carry_their_transcript(tmp_path):
    from fukikae.cards.select import tiered_references

    turns, tr, vocals = _gate_case(tmp_path, [(10, 12, 240.0, 0.01)])
    _, (pick,) = tiered_references(turns, "A", tr, None, vocals, register=("f", 240.0))
    assert pick.transcript == "line at 10"


def test_clone_card_always_records_its_tier(tmp_path):
    """Review fix 3: every clone card names its tier, so a re-put of the same id (a thin card
    whose picks now pass the gates) replaces the stored tier instead of keeping it."""
    from fukikae.cards.auto import is_thin

    turns = [Turn(start=10, end=15, speaker="A"), Turn(start=20, end=26, speaker="A")]
    vocals = _vocals(tmp_path / "v.wav", [(10, 15, 0.01), (20, 26, 0.01)])
    picks = select_references(turns, "A", _transcript(turns), None, vocals)
    assert picks is not None
    store = CardStore(tmp_path / "ws")
    thin = clone_card(CloneSynth, picks, store, vocals, "ja", "en", "S01E11", "thin")
    cid = store.put(thin)
    assert is_thin(store.get(cid))
    gated = clone_card(CloneSynth, picks, store, vocals, "ja", "en", "S01E11", "floor")
    assert store.put(gated) == cid
    assert store.get(cid).meta["reference_tier"] == "floor" and not is_thin(store.get(cid))
    for tier in ("short", "thin"):
        again = clone_card(CloneSynth, picks, store, vocals, "ja", "en", "S01E11", tier)
        store.put(again)
        assert store.get(cid).meta["reference_tier"] == tier
    # with no tier given, the picks' own: 11 s in 2 clips reach the floor
    plain = clone_card(CloneSynth, picks, store, vocals, "ja", "en", "S01E11")
    assert plain.meta["reference_tier"] == "floor"
    short = clone_card(CloneSynth, picks[:1], store, vocals, "ja", "en", "S01E11")
    assert short.meta["reference_tier"] == "short"
