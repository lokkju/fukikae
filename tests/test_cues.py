from fukikae.cues import budget_chars, build_asr_cues, is_marked, windows
from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Delivery,
    DubCue,
    Segment,
    Tags,
    TagSpan,
    Transcript,
    Turn,
    Word,
)


def _transcript(segments: list[Segment], language: str = "ja") -> Transcript:
    return Transcript(language=language, backend="fake", word_timing=True, segments=segments)


def _assignments(turns: list[Turn], clusters: list[ClusterAssignment]) -> Assignments:
    return Assignments(episode="e1", threshold=0.6, clusters=clusters, turns=turns)


def _cluster(name: str, character: int | None, status: str = "matched") -> ClusterAssignment:
    return ClusterAssignment(
        cluster=name,
        actor_id=character,
        character_id=character,
        score=0.9,
        clean_seconds=30.0,
        status=status,  # type: ignore[arg-type]
    )


def _cue(i: int, start: float, end: float, **kw) -> DubCue:
    return DubCue(
        id=f"c{i:04d}",
        start=start,
        end=end,
        character_id=kw.pop("character_id", 1),
        extra=None,
        speaker=kw.pop("speaker", "Natsu"),
        source_text=kw.pop("source_text", "x"),
        budget=20,
        delivery=kw.pop("delivery", Delivery(reference=(start, end))),
        **kw,
    )


def test_budget_chars():
    assert budget_chars(2.0, "en", 1.25) == 37 and budget_chars(0.1, "en", 1.25) == 4
    assert budget_chars(2.0, "xx", 1.0) == 28  # unknown language: 14 cps


def test_asr_cues_split_on_speaker_change_and_gap():
    seg = Segment(
        start=0.0,
        end=3.0,
        text="あい。う",
        words=[
            Word(start=0.0, end=0.2, text="あ"),
            Word(start=0.4, end=0.6, text="い"),
            Word(start=0.8, end=1.0, text="う"),  # gap 0.2 but S1 speaks it
        ],
    )
    turns = [Turn(start=0.0, end=0.7, speaker="S0"), Turn(start=0.7, end=1.2, speaker="S1")]
    asg = _assignments(turns, [_cluster("S0", 3), _cluster("S1", 5)])
    cues = build_asr_cues(_transcript([seg]), asg, None, {3: "Natsu"}, "en", 1.25)
    assert [c.id for c in cues] == ["c0001", "c0002"]
    assert [c.source_text for c in cues] == ["あい。", "う"]  # punctuation from the segment text
    assert [c.character_id for c in cues] == [3, 5]
    assert [c.speaker for c in cues] == ["Natsu", "char5"]
    assert cues[0].start == 0.0 and cues[0].end == 0.6
    assert cues[0].delivery.reference == (0.0, 0.6)
    assert cues[0].budget == budget_chars(0.6, "en", 1.25)

    # one speaker, a gap over 0.6 s splits too
    gap = Segment(
        start=0.0,
        end=3.0,
        text="ab",
        words=[Word(start=0.0, end=0.2, text="a"), Word(start=1.0, end=1.2, text="b")],
    )
    one = _assignments([Turn(start=0.0, end=2.0, speaker="S0")], [_cluster("S0", 3)])
    assert len(build_asr_cues(_transcript([gap]), one, None, {}, "en", 1.25)) == 2


def test_long_cue_splits_after_sentence_final_punctuation():
    words = [Word(start=i * 0.5, end=i * 0.5 + 0.3, text=t) for i, t in enumerate("abcdefghij")]
    seg = Segment(start=0.0, end=5.0, text="abcdefg。hij", words=words)
    asg = _assignments([Turn(start=0.0, end=6.0, speaker="S0")], [_cluster("S0", 1)])
    cues = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25, max_cue_s=2.0)
    assert [c.source_text for c in cues] == ["abcdefg。", "hij"]


def test_space_separated_languages_join_with_spaces():
    seg = Segment(
        start=0.0,
        end=1.0,
        text="Hola, amigo.",
        words=[Word(start=0.0, end=0.3, text="Hola"), Word(start=0.4, end=0.8, text="amigo")],
    )
    asg = _assignments([Turn(start=0.0, end=1.0, speaker="S0")], [_cluster("S0", 1)])
    (cue,) = build_asr_cues(_transcript([seg], "es"), asg, None, {}, "en", 1.25)
    assert cue.source_text == "Hola, amigo."


def test_extra_cluster_gets_extra_label_and_no_character():
    seg = Segment(start=0.0, end=1.0, text="あ", words=[Word(start=0.1, end=0.3, text="あ")])
    for status in ("extra", "review"):
        asg = _assignments(
            [Turn(start=0.0, end=1.0, speaker="S7")], [_cluster("S7", None, status=status)]
        )
        (cue,) = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25)
        assert cue.character_id is None and cue.extra == "S7" and cue.speaker == "extra:S7"
    # a word under no turn at all is an unknown extra, never a crash
    asg = _assignments([Turn(start=5.0, end=6.0, speaker="S0")], [_cluster("S0", 1)])
    (cue,) = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25)
    assert cue.character_id is None and cue.extra == "unknown"


def test_cue_in_singing_span_is_skipped_as_song():
    seg = Segment(
        start=0.0,
        end=10.0,
        text="らら。はい",
        words=[
            Word(start=0.0, end=0.5, text="ら"),
            Word(start=0.6, end=1.0, text="ら"),
            Word(start=8.0, end=8.5, text="は"),
            Word(start=8.6, end=9.0, text="い"),
        ],
    )
    tags = Tags(
        backend="fake",
        events=[TagSpan(start=0.0, end=5.0, label="singing")],
        emotions=[TagSpan(start=7.5, end=9.5, label="angry", confidence=0.8)],
    )
    asg = _assignments([Turn(start=0.0, end=10.0, speaker="S0")], [_cluster("S0", 1)])
    song, line = build_asr_cues(_transcript([seg]), asg, tags, {}, "en", 1.25)
    assert song.skip == "song" and line.skip is None
    assert line.delivery.emotion == "angry" and line.delivery.emotion_confidence == 0.8
    assert song.delivery.emotion is None


def test_windows_break_on_gap_and_size():
    cues = [_cue(1, 0, 1), _cue(2, 1.5, 2), _cue(3, 5, 6), _cue(4, 6.1, 7)]
    assert [[c.id for c in w] for w in windows(cues)] == [["c0001", "c0002"], ["c0003", "c0004"]]
    many = [_cue(i, i, i + 0.5) for i in range(1, 30)]
    sizes = [len(w) for w in windows(many, max_cues=12)]
    assert sizes == [12, 12, 5]


def test_is_marked():
    angry = Delivery(reference=(0, 1), emotion="angry", emotion_confidence=0.8)
    weak = Delivery(reference=(0, 1), emotion="angry", emotion_confidence=0.3)
    neutral = Delivery(reference=(0, 1), emotion="neutral", emotion_confidence=0.9)
    assert is_marked(_cue(1, 0, 1, delivery=angry))
    assert not is_marked(_cue(1, 0, 1, delivery=weak))
    assert not is_marked(_cue(1, 0, 1, delivery=neutral))
    assert is_marked(_cue(1, 0, 1, source_text="(whispering) come here"))
    assert is_marked(_cue(1, 0, 1, source_text="（小声で）来て"))
    assert not is_marked(_cue(1, 0, 1, source_text="come here"))
    # SenseVoice gives no confidence; an unscored label counts as marked
    unscored = Delivery(reference=(0, 1), emotion="sad", emotion_confidence=None)
    assert is_marked(_cue(1, 0, 1, delivery=unscored))
