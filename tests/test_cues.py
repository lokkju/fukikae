import pytest

from fukikae.cues import (
    assign_subtitle_speakers,
    budget_chars,
    build_asr_cues,
    is_marked,
    name_map,
    windows,
)
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
from fukikae.subtitles import Cue


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
    assert budget_chars(2.0, "en", 1.25, cps=10.0) == 25  # the TTS backend's speech rate


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


def test_fragment_at_a_turn_boundary_joins_the_next_cue():
    # the turn change lands one character into "ロック": "ロ" must not become its own cue
    seg = Segment(
        start=0.0,
        end=3.0,
        text="ロックと言った。",
        words=[
            Word(start=0.5 + i * 0.25, end=0.52 + i * 0.25, text=t)
            for i, t in enumerate("ロックと言った")
        ],
    )
    turns = [Turn(start=0.0, end=0.6, speaker="S0"), Turn(start=0.6, end=3.0, speaker="S1")]
    asg = _assignments(turns, [_cluster("S0", 1), _cluster("S1", 2)])
    (cue,) = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25)
    assert cue.source_text == "ロックと言った。" and cue.character_id == 2
    assert cue.start == 0.5


def test_fragment_ending_a_sentence_joins_the_previous_cue():
    seg = Segment(
        start=0.0,
        end=3.0,
        text="行くぞ。",
        words=[
            Word(start=0.5 + i * 0.25, end=0.52 + i * 0.25, text=t) for i, t in enumerate("行くぞ")
        ],
    )
    turns = [Turn(start=0.0, end=0.95, speaker="S0"), Turn(start=0.95, end=3.0, speaker="S1")]
    asg = _assignments(turns, [_cluster("S0", 1), _cluster("S1", 2)])
    (cue,) = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25)
    assert cue.source_text == "行くぞ。" and cue.character_id == 1


def test_point_stamped_words_get_their_sound_until_the_next_word():
    # the aligner gives 20 ms per character; the slot runs past the last stamp
    words = [Word(start=i * 0.25, end=i * 0.25 + 0.02, text=t) for i, t in enumerate("はいはい")]
    seg = Segment(start=0.0, end=2.0, text="はいはい", words=words)
    asg = _assignments([Turn(start=0.0, end=3.0, speaker="S0")], [_cluster("S0", 1)])
    (cue,) = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25)
    assert cue.end == pytest.approx(0.75 + 0.3)


def test_a_genuine_interjection_by_another_speaker_survives():
    # A says two sentences; B's lone "え？" sits between them in the same segment
    text = "行くぞ。え？待て。"
    stamps = [("行", 0.0), ("く", 0.25), ("ぞ", 0.5), ("え", 0.8), ("待", 1.1), ("て", 1.35)]
    seg = Segment(
        start=0.0,
        end=2.0,
        text=text,
        words=[Word(start=t, end=t + 0.02, text=c) for c, t in stamps],
    )
    turns = [
        Turn(start=0.0, end=0.7, speaker="S0"),
        Turn(start=0.7, end=0.95, speaker="S1"),
        Turn(start=0.95, end=2.0, speaker="S0"),
    ]
    asg = _assignments(turns, [_cluster("S0", 1), _cluster("S1", 2)])
    cues = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25)
    assert [(c.source_text, c.character_id) for c in cues] == [
        ("行くぞ。", 1),
        ("え？", 2),
        ("待て。", 1),
    ]


def test_ideographic_comma_does_not_end_a_sentence_for_folding():
    # "偽" is cut off the end of a cue whose next word is "物": it joins forward even when
    # the cue before it ends in a comma
    stamps = [("弱", 0.0), ("点", 0.25), ("偽", 0.5), ("物", 0.75), ("だ", 1.0)]
    seg = Segment(
        start=0.0,
        end=2.0,
        text="弱点、偽物だ。",
        words=[Word(start=t, end=t + 0.02, text=c) for c, t in stamps],
    )
    turns = [
        Turn(start=0.0, end=0.4, speaker="S0"),
        Turn(start=0.4, end=0.6, speaker="S2"),
        Turn(start=0.6, end=2.0, speaker="S1"),
    ]
    asg = _assignments(turns, [_cluster("S0", 1), _cluster("S1", 2), _cluster("S2", 3)])
    cues = build_asr_cues(_transcript([seg]), asg, None, {}, "en", 1.25)
    assert [(c.source_text, c.character_id) for c in cues] == [("弱点、", 1), ("偽物だ。", 2)]


# ---- subtitle-first path ----


def _subs(
    cues, turns, clusters, transcript=None, tags=None, names=None, rules=("sdh", "song", "sign")
):
    return assign_subtitle_speakers(
        cues, turns, _assignments(turns, clusters), transcript, tags, names or {}, "en", 1.25, rules
    )


def test_subtitle_speaker_by_overlap():
    turns = [Turn(start=0.0, end=1.2, speaker="S0"), Turn(start=1.2, end=4.0, speaker="S1")]
    cues = [Cue(0.0, 2.0, "LUCY: (sighs) Fine."), Cue(2.0, 3.0, "[door slams]")]
    got = _subs(cues, turns, [_cluster("S0", 3), _cluster("S1", 5)], names={3: "Lucy"})
    assert [c.id for c in got] == ["c0001", "c0002"]
    assert got[0].character_id == 3 and got[0].speaker == "Lucy"  # 1.2 s of S0 against 0.8 s
    assert got[0].source_text == got[0].text == "Fine."
    assert got[0].literal is None and got[0].skip is None
    assert got[0].budget == budget_chars(2.0, "en", 1.25)
    assert got[0].delivery.reference == (0.0, 2.0)
    assert got[1].skip == "sdh" and got[1].character_id == 5


def test_overlap_sums_a_speakers_turns():
    turns = [
        Turn(start=0.0, end=0.5, speaker="S0"),
        Turn(start=0.5, end=1.1, speaker="S1"),
        Turn(start=1.1, end=1.6, speaker="S0"),
    ]
    got = _subs([Cue(0.0, 1.6, "Hi.")], turns, [_cluster("S0", 3), _cluster("S1", 5)])
    assert got[0].character_id == 3  # 1.0 s of S0 in two turns against 0.6 s of S1


def test_two_speaker_tie_broken_by_asr_words():
    turns = [Turn(start=0.0, end=1.1, speaker="S0"), Turn(start=1.1, end=2.0, speaker="S1")]
    clusters = [_cluster("S0", 3), _cluster("S1", 5)]
    # S1 holds three words, S0 one: S1 speaks the line, though S0's turn covers more of it
    seg = Segment(
        start=0.0,
        end=2.0,
        text="a b c d",
        words=[
            Word(start=0.1, end=0.3, text="a"),
            Word(start=1.2, end=1.4, text="b"),
            Word(start=1.5, end=1.7, text="c"),
            Word(start=1.75, end=1.95, text="d"),
        ],
    )
    cues = [Cue(0.0, 2.0, "Say it.")]
    assert _subs(cues, turns, clusters)[0].character_id == 3  # no transcript: overlap only
    got = _subs(cues, turns, clusters, transcript=_transcript([seg]))
    assert got[0].character_id == 5
    # under 30% of the cue for the runner-up: overlap stands
    turns2 = [Turn(start=0.0, end=1.5, speaker="S0"), Turn(start=1.5, end=2.0, speaker="S1")]
    assert _subs(cues, turns2, clusters, transcript=_transcript([seg]))[0].character_id == 3


def test_cue_with_no_turn_is_unknown_extra():
    turns = [Turn(start=0.0, end=1.0, speaker="S0")]
    clusters = [_cluster("S0", 3)]
    got = _subs([Cue(5.0, 6.0, "Who's there?"), Cue(1.5, 2.0, "Near.")], turns, clusters)
    assert [c.source_text for c in got] == ["Near.", "Who's there?"]  # in time order
    assert got[0].character_id == 3  # 0.5 s from a turn: the nearest within 1 s
    assert got[1].character_id is None and got[1].extra == "unknown"
    assert got[1].speaker == "extra:unknown"


def test_subtitle_extra_cluster_and_filtered_cues_keep_rule():
    turns = [Turn(start=0.0, end=4.0, speaker="S9")]
    cues = [Cue(0.0, 1.0, "♪ la la ♪"), Cue(1.0, 2.0, "Hello."), Cue(2.0, 3.0, "Ruby", ruby=True)]
    got = _subs(cues, turns, [_cluster("S9", None, status="extra")])
    assert [c.skip for c in got] == ["song", None, "ruby"]
    assert got[1].extra == "S9" and got[1].speaker == "extra:S9"
    off = _subs(cues, turns, [_cluster("S9", None, status="extra")], rules=())
    assert [c.skip for c in off] == [None, None, "ruby"]


def test_zero_length_subtitle_events_are_skipped_as_empty():
    """C2: an event with end <= start has no slot to speak in."""
    turns = [Turn(start=0.0, end=4.0, speaker="S0")]
    cues = [Cue(1.0, 1.0, "Hi."), Cue(2.0, 1.5, "Odd."), Cue(2.0, 3.0, "Fine.")]
    got = _subs(cues, turns, [_cluster("S0", 3)])
    assert [c.skip for c in got] == ["empty", "empty", None]


def test_layered_identical_subtitle_events_are_one_cue():
    """C8: typeset ASS draws a line as several overlapping events (an outline layer and a text
    layer); they are spoken once. An event that only touches the previous one is a new line."""
    turns = [Turn(start=0.0, end=10.0, speaker="S0")]
    cues = [
        Cue(1.0, 3.0, "Hi there."),
        Cue(1.0, 3.0, "{\\bord3}Hi there."),
        Cue(2.5, 3.5, "Hi there."),  # overlaps: same line
        Cue(3.5, 4.0, "Hi there."),  # touches only: said again
        Cue(2.0, 2.5, "Other."),
        Cue(6.0, 7.0, "Hi there."),  # later and apart: said again
    ]
    got = _subs(cues, turns, [_cluster("S0", 3)])
    assert [(c.start, c.end, c.source_text) for c in got] == [
        (1.0, 3.5, "Hi there."),
        (2.0, 2.5, "Other."),
        (3.5, 4.0, "Hi there."),
        (6.0, 7.0, "Hi there."),
    ]


def test_abutting_identical_subtitle_events_stay_two_cues():
    """C8: "Wait!" 10-11 then "Wait!" 11-12 is the line said twice, not one layered line."""
    turns = [Turn(start=9.0, end=13.0, speaker="S0")]
    cues = [Cue(10.0, 11.0, "Wait!"), Cue(11.0, 12.0, "Wait!")]
    got = _subs(cues, turns, [_cluster("S0", 3)])
    assert [(c.start, c.end, c.source_text) for c in got] == [
        (10.0, 11.0, "Wait!"),
        (11.0, 12.0, "Wait!"),
    ]


# ---- names (Task 8): a cue's own Name picks its speaker ----


def _named(
    name: str, character: int | None, source: str = "name", status: str = "matched", secs=30.0
) -> ClusterAssignment:
    return ClusterAssignment.model_validate(
        {
            "cluster": name,
            "actor_id": character,
            "character_id": character,
            "score": 0.9,
            "clean_seconds": secs,
            "status": status,
            "name": name.split("~")[1].replace("-", " ") if "~" in name else None,
            "source": source,
        }
    )


S12 = [Turn(start=1299.0, end=1304.0, speaker="SPEAKER_12")]
JELLAL = Cue(1300.3, 1302.8, "I'd even settle for Jellal!", name="NATSU")


def _by_name(cues, turns, clusters, **kw):
    return assign_subtitle_speakers(
        cues, turns, _assignments(turns, clusters), None, None, {9: "Natsu"}, "en", 1.25,
        ("sdh", "song", "sign"), **kw,
    )  # fmt: skip


def test_cue_name_beats_turn_overlap():
    (got,) = _by_name([JELLAL], S12, [_cluster("SPEAKER_12", 2)], name_map={"natsu": (9, None)})
    assert (got.character_id, got.extra, got.speaker) == (9, None, "Natsu")


def test_without_a_name_map_the_overlap_decides():
    (got,) = _by_name([JELLAL], S12, [_cluster("SPEAKER_12", 2)])
    assert got.character_id == 2


def test_hidden_cue_falls_back_to_turn_overlap():
    (got,) = _by_name(
        [JELLAL], S12, [_cluster("SPEAKER_12", 2)],
        name_map={"natsu": (9, None)}, hidden=[(1300.3, 1302.8, "natsu")],
    )  # fmt: skip
    assert got.character_id == 2


def test_hidden_lines_are_matched_after_moving_the_cue_back_by_the_offset():
    late = Cue(1303.3, 1305.8, JELLAL.text, name="NATSU")  # the track runs 3 s late
    clusters = [_cluster("SPEAKER_12", 2)]
    nm = {"natsu": (9, None)}
    hidden = [(1300.3, 1302.8, "natsu")]
    (hid,) = _by_name([late], S12, clusters, name_map=nm, hidden=hidden, offset_s=3.0)
    (shown,) = _by_name([late], S12, clusters, name_map=nm, hidden=hidden)
    assert (hid.character_id, shown.character_id) == (2, 9)


@pytest.mark.parametrize("name", ["GRAY", "NATSU/HAPPY", "", "SIGN"])
def test_unknown_or_multi_name_falls_back(name):
    cue = Cue(JELLAL.start, JELLAL.end, JELLAL.text, name=name)
    (got,) = _by_name([cue], S12, [_cluster("SPEAKER_12", 2)], name_map={"natsu": (9, None)})
    assert got.character_id == 2


def test_named_extra_gets_its_extra_label():
    cue = Cue(JELLAL.start, JELLAL.end, "Halt!", name="Wizard A")
    nm = {"wizard a": (None, "SPEAKER_04~wizard-a")}
    (got,) = _by_name([cue], S12, [_cluster("SPEAKER_12", 2)], name_map=nm)
    assert (got.character_id, got.extra, got.speaker) == (
        None, "SPEAKER_04~wizard-a", "extra:SPEAKER_04~wizard-a"
    )  # fmt: skip


def test_a_lock_kept_sub_cluster_keeps_its_speech():
    """A cue under a sub-cluster a lock kept (natsu's speech in a locked actor's cluster) stays
    with the locked character, whatever its Name says (operator state beats names)."""
    turns = [Turn(start=1299.0, end=1304.0, speaker="SPEAKER_12~natsu")]
    clusters = [_named("SPEAKER_12~natsu", 2, source="lock")]
    (got,) = _by_name([JELLAL], turns, clusters, name_map={"natsu": (9, None)})
    assert got.character_id == 2


def test_name_map_takes_name_decisions_only():
    asg = _assignments(
        [],
        [
            _named("SPEAKER_04~natsu", 9),
            _named("SPEAKER_12~natsu", 2, source="lock"),
            _named("SPEAKER_04~wizard-a", None, status="extra", secs=2.0),
            _named("SPEAKER_12~wizard-a", None, status="extra", secs=5.0),
            _named("SPEAKER_05~wizard-a", None, status="extra", secs=5.0),
            _cluster("SPEAKER_12", 2),
        ],
    )
    assert name_map(asg) == {"natsu": (9, None), "wizard a": (None, "SPEAKER_05~wizard-a")}


def test_a_hidden_line_of_another_name_never_hides_a_simultaneous_cue():
    """Task 8 review: a hidden HAPPY line said over NATSU's visible one hides only cues with
    its own key; NATSU's cue keeps its name."""
    happy = Cue(JELLAL.start, JELLAL.end, "Aye!", name="HAPPY")
    nm = {"natsu": (9, None), "happy": (18, None)}
    hidden = [(1300.3, 1302.8, "happy")]
    natsu, hap = _by_name([JELLAL, happy], S12, [_cluster("SPEAKER_12", 2)], name_map=nm,
                          hidden=hidden)  # fmt: skip
    assert (natsu.character_id, hap.character_id) == (9, 2)


def test_a_cue_over_a_locked_parents_remainder_stays_with_the_lock():
    """Task 8 review: the turn under the cue is the unnamed remainder SPEAKER_12, whose parent
    also holds a lock-kept sub-cluster; operator state beats names, so the NATSU cue stays with
    the locked character."""
    turns = [Turn(start=1299.0, end=1304.0, speaker="SPEAKER_12"),
             Turn(start=1310.0, end=1312.0, speaker="SPEAKER_12~charle")]  # fmt: skip
    clusters = [_cluster("SPEAKER_12", 2), _named("SPEAKER_12~charle", 2, source="lock")]
    (got,) = _by_name([JELLAL], turns, clusters, name_map={"natsu": (9, None)})
    assert got.character_id == 2


def test_the_overlap_and_lock_lookups_move_the_cue_back_by_the_offset():
    """Task 8 review: with a names offset the cue is matched to turns in audio time: the track
    runs 10 s late, so the cue at 1310.3 sits over the lock-kept turn at 1300.3."""
    late = Cue(1310.3, 1312.8, JELLAL.text, name="NATSU")
    turns = [Turn(start=1299.0, end=1304.0, speaker="SPEAKER_12~natsu"),
             Turn(start=1309.0, end=1314.0, speaker="SPEAKER_04")]  # fmt: skip
    clusters = [_named("SPEAKER_12~natsu", 2, source="lock"), _cluster("SPEAKER_04", 5)]
    nm = {"natsu": (9, None)}
    (moved,) = _by_name([late], turns, clusters, name_map=nm, offset_s=10.0)
    (plain,) = _by_name([late], turns, clusters, name_map=nm)
    assert (moved.character_id, plain.character_id) == (2, 9)
    assert (moved.start, moved.end) == (late.start, late.end)  # the cue keeps its own times
