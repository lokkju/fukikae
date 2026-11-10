from pathlib import Path

import pytest

from fukikae.config import Config
from fukikae.media import MediaError, ProbeResult, SubTrack
from fukikae.models import Turn
from fukikae.names.detect import (
    assess,
    assess_path,
    candidate_tracks,
    dialogue_lines,
)
from fukikae.names.normalize import NameClass
from fukikae.subtitles import parse_ass_cues, parse_ass_styles

STYLES = {"Default", "Italics", "Flashback"}
SOURCE = Path("/media/show/S01E01.mkv")

# Irregular timing inside ~32 s, so a 30 s shift has nothing to line up with inside +-10 s.
_DURS = [1.2, 0.8, 1.6, 1.0, 1.4, 0.9, 1.3, 1.1, 1.5, 0.7, 1.2, 1.0, 1.6, 0.8, 1.1, 1.4, 0.9, 1.3]
_GAPS = [0.5, 0.9, 0.4, 0.7, 1.0, 0.6, 0.8, 0.5, 0.4, 0.9, 0.7, 0.6, 1.0, 0.5, 0.8, 0.4, 0.6, 0.7]
_WHO = ["NATSU", "LUCY", "HAPPY", "NATSU", "ERZA", "LUCY", "WENDY", "NATSU", "CHARLE"]


def _timeline() -> list[tuple[float, float]]:
    t, out = 0.5, []
    for d, g in zip(_DURS, _GAPS, strict=True):
        out.append((round(t, 3), round(t + d, 3)))
        t += d + g
    return out


Line = tuple[float, float, str, str, str]
NAMED_EPISODE: list[Line] = [
    (a, b, "Default", _WHO[i % len(_WHO)], f"Line number {i}.")
    for i, (a, b) in enumerate(_timeline())
]
BLANK_EPISODE: list[Line] = [(a, b, s, "", t) for a, b, s, _, t in NAMED_EPISODE]


def _stamp(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def ass(lines: list[Line], styles: set[str] = STYLES) -> str:
    head = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour",
        *(f"Style: {s},Arial,24,&H00FFFFFF" for s in sorted(styles)),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    rows = [f"Dialogue: 0,{_stamp(a)},{_stamp(b)},{s},{n},0,0,0,,{t}" for a, b, s, n, t in lines]
    return "\n".join([*head, *rows, ""])


def cues(lines: list[Line]):
    return parse_ass_cues(ass(lines))


def speech_under(lines: list[Line]) -> list[Turn]:
    return [Turn(start=a, end=b, speaker="SPEAKER_00") for a, b, *_ in lines]


SPEECH = speech_under(NAMED_EPISODE)


def all_named(name: str) -> list[Line]:
    return [(a, b, s, name, t) for a, b, s, _, t in NAMED_EPISODE]


def names_are_styles() -> list[Line]:
    out = []
    for i, (a, b, _, _, t) in enumerate(NAMED_EPISODE):
        style = "Main" if i % 2 else "Flashback"
        out.append((a, b, style, style, t))
    return out


def shift(lines: list[Line], by: float) -> list[Line]:
    return [(a + by, b + by, s, n, t) for a, b, s, n, t in lines]


def named_fraction(frac: float) -> list[Line]:
    k = round(len(NAMED_EPISODE) * frac)
    return [ln if i < k else (*ln[:3], "", ln[4]) for i, ln in enumerate(NAMED_EPISODE)]


def test_crunchyroll_style_track_is_full():
    v, _ = assess(dialogue_lines(cues(NAMED_EPISODE)), STYLES, speech_under(NAMED_EPISODE))
    assert v.level == "full" and v.distinct >= 2
    assert v.reason is None and v.dialogue == v.named == len(NAMED_EPISODE)


def test_blank_names_are_none():
    v, _ = assess(dialogue_lines(cues(BLANK_EPISODE)), STYLES, SPEECH)
    assert v.level == "none" and v.reason is not None and "share" in v.reason


def test_one_name_everywhere_is_refused():
    v, _ = assess(dialogue_lines(cues(all_named("Default"))), STYLES, SPEECH)
    assert v.level == "none" and v.reason is not None and "dominant" in v.reason


def test_one_name_on_most_lines_is_refused():
    lines = [ln if i % 9 else (*ln[:3], "LUCY", ln[4]) for i, ln in enumerate(all_named("NATSU"))]
    v, _ = assess(dialogue_lines(cues(lines)), STYLES, SPEECH)
    assert v.level == "none" and v.top_share > 0.8


def test_names_equal_to_styles_are_refused():
    v, _ = assess(dialogue_lines(cues(names_are_styles())), {"Main", "Flashback"}, SPEECH)
    assert v.level == "none" and v.reason is not None and "style" in v.reason


def test_misaligned_track_is_refused_unless_align_off():
    shifted = shift(NAMED_EPISODE, 30.0)  # beyond the 10 s search
    v, _ = assess(dialogue_lines(cues(shifted)), STYLES, SPEECH)
    assert v.level == "none" and v.reason is not None and "line up" in v.reason
    assert (
        assess(dialogue_lines(cues(shifted)), STYLES, SPEECH, use="on", align=False)[0].level
        != "none"
    )


def test_small_offset_is_applied_to_lines():
    v, lines = assess(
        dialogue_lines(cues(shift(NAMED_EPISODE, 1.5))), STYLES, speech_under(NAMED_EPISODE)
    )
    assert v.offset_s == pytest.approx(1.5, abs=0.1)
    assert lines[0].start == pytest.approx(NAMED_EPISODE[0][0], abs=0.1)
    assert v.align_share > 0.9


def test_offset_moves_unnamed_lines_too():
    lines = named_fraction(0.6)
    _, got = assess(dialogue_lines(cues(shift(lines, 1.5))), STYLES, SPEECH)
    assert got[-1].start == pytest.approx(NAMED_EPISODE[-1][0], abs=0.1)


def test_partial_between_half_and_ninety_percent():
    assert assess(dialogue_lines(cues(named_fraction(0.6))), STYLES, SPEECH)[0].level == "partial"


def test_min_coverage_is_configurable():
    lines = dialogue_lines(cues(named_fraction(0.6)))
    v, _ = assess(lines, STYLES, SPEECH, min_coverage=0.7)
    assert v.level == "none" and v.reason is not None and "share" in v.reason


def test_use_on_skips_share_and_dominance_but_needs_two_names():
    blankish = dialogue_lines(cues(named_fraction(0.2)))
    assert assess(blankish, STYLES, SPEECH, use="on")[0].level == "partial"
    one = dialogue_lines(cues(all_named("NATSU")))
    v, _ = assess(one, STYLES, SPEECH, use="on")
    assert v.level == "none" and v.reason is not None and "two" in v.reason


def test_use_off_reads_nothing():
    v, lines = assess(dialogue_lines(cues(NAMED_EPISODE)), STYLES, SPEECH, use="off")
    assert v.level == "none" and lines == []


def test_role_labels_count_as_named():
    lines = [
        (*ln[:3], "WIZARD A" if i % 2 else "WIZARD B", ln[4]) for i, ln in enumerate(NAMED_EPISODE)
    ]
    v, got = assess(dialogue_lines(cues(lines)), STYLES, SPEECH)
    assert v.level == "full" and {ln.cls for ln in got} == {NameClass.ROLE}


def test_signs_and_next_episode_title_are_not_dialogue():
    lines = dialogue_lines(
        cues(
            [
                (1, 4, "White-Standard", "SIGN", "{\\pos(320,238)}Title"),
                (40, 44, "Next Ep Title", "SIGN", "The Next One"),
                *NAMED_EPISODE,
            ]
        )
    )
    assert all(ln.raw != "SIGN" for ln in lines) and len(lines) == len(NAMED_EPISODE)


def test_layers_of_one_line_are_one_line_and_ruby_is_dropped():
    lines = dialogue_lines(
        cues(
            [
                (10, 12, "Default", "NATSU", "{\\bord6}Hey!"),
                (10, 12.2, "Default", "NATSU", "Hey!"),
                (10, 12, "Default", "NATSU", "{\\fscy50}hei"),
                (13, 14, "Default", "LUCY", "{\\i1}{\\i0}"),
            ]
        )
    )
    assert [(ln.key, ln.start, ln.end, ln.text) for ln in lines] == [("natsu", 10, 12.2, "Hey!")]


def test_lines_carry_key_display_inner_and_ids():
    lines = dialogue_lines(
        cues([(1, 2, "Italics", "WENDY/I", "Hm."), (3, 4, "Default", "YOUNG MAKAROV", "Ha!")])
    )
    assert [(ln.id, ln.key, ln.display, ln.inner, ln.cls) for ln in lines] == [
        (0, "wendy", "Wendy", True, NameClass.NAMED),
        (1, "young makarov", "Young Makarov", False, NameClass.NAMED),
    ]


def test_simultaneous_different_names_become_multi():
    lines = dialogue_lines(
        cues([(10, 12, "Italics", "CHARLE", "x"), (10, 12, "Italics", "TOUKA", "x")])
    )
    assert {ln.cls for ln in lines} == {NameClass.MULTI} and len(lines) == 2
    assert all(ln.key is None for ln in lines)


def test_simultaneous_same_name_stays_named():
    lines = dialogue_lines(
        cues([(10, 12, "Default", "LUCY", "One."), (10, 12, "Default", "LUCY", "Two.")])
    )
    assert {ln.cls for ln in lines} == {NameClass.NAMED}


def test_named_line_serialises():
    from pydantic import TypeAdapter

    from fukikae.names.models import NamedLine

    (ln,) = dialogue_lines(cues([(1, 2, "Default", "NATSU", "Hey.")]))
    ta = TypeAdapter(NamedLine)
    assert ta.validate_json(ta.dump_json(ln)) == ln


def test_parse_ass_styles():
    assert parse_ass_styles(ass(NAMED_EPISODE)) == STYLES
    assert parse_ass_styles("1\n00:00:01,000 --> 00:00:02,000\nhi\n") == set()


# ---- candidates ----


def _track(index: int, lang: str, codec: str = "ass", **kw) -> SubTrack:
    return SubTrack(index, codec, lang, kw.pop("title", None), kw.pop("forced", False), **kw)


@pytest.fixture
def probe_with_en_and_de() -> ProbeResult:
    return ProbeResult(
        subtitles=[
            _track(2, "eng", cues=400),
            _track(3, "ger", cues=380),
            _track(4, "jpn", "subrip", cues=390),
        ],
        duration=1440.0,
    )


def cfg(target_language: str = "en", **raw) -> Config:
    return Config(target_language=target_language, raw=raw)


def test_candidate_order_prefers_target_dialogue_track_then_other_languages(probe_with_en_and_de):
    got = candidate_tracks(SOURCE, probe_with_en_and_de, cfg(target_language="de"))
    assert [c.how for c in got] == ["target", "other-language", "other-language"]
    assert got[0].track is not None and got[0].track.index == 3 and got[0].language == "de"
    assert got[1].language == "en" and got[2].language == "ja"


def test_configured_file_and_track_come_first(tmp_path: Path, probe_with_en_and_de):
    src = tmp_path / "S01E02.mkv"
    (tmp_path / "names").mkdir()
    named = tmp_path / "names" / "S01E02.ass"
    named.write_text(ass(NAMED_EPISODE))
    got = candidate_tracks(
        src,
        probe_with_en_and_de,
        cfg("en", names={"file": "names/{episode}.ass", "track": 3}),
        episode="S01E02",
    )
    assert [c.how for c in got] == ["file", "track", "target", "other-language"]
    assert got[0].file == named and got[1].track is not None and got[1].track.index == 3
    assert [c.track.index for c in got[2:] if c.track] == [2, 4]  # track 3 is not offered twice


def test_missing_names_file_is_an_error(tmp_path: Path, probe_with_en_and_de):
    with pytest.raises(MediaError, match="names"):
        candidate_tracks(tmp_path / "x.mkv", probe_with_en_and_de, cfg(names={"file": "nope.ass"}))


def test_target_from_subs_sidecar(tmp_path: Path):
    src = tmp_path / "S01E03.mkv"
    side = tmp_path / "S01E03.en.ass"
    side.write_text(ass(NAMED_EPISODE))
    got = candidate_tracks(src, ProbeResult(), cfg("en"))
    assert [(c.how, c.file) for c in got] == [("target", side)]


def test_srt_candidate_assesses_none(tmp_path: Path):
    srt = tmp_path / "S01E01.en.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nNATSU: Hey!\n")
    v, lines = assess_path(srt, SPEECH)
    assert v.level == "none" and v.reason is not None and "format" in v.reason
    assert "srt" in v.reason and lines == []


def test_ass_path_assesses(tmp_path: Path):
    p = tmp_path / "S01E01.en.ass"
    p.write_text(ass(NAMED_EPISODE))
    v, lines = assess_path(p, SPEECH)
    assert v.level == "full" and len(lines) == len(NAMED_EPISODE)


# ---- review carries (Task 2) ----


def test_blank_outline_layer_joins_the_named_text_layer():
    lines = [
        *[(a, b, s, "", "{\\bord6}" + t) for a, b, s, _, t in NAMED_EPISODE[:1]],
        *NAMED_EPISODE,
    ]
    got = dialogue_lines(cues(lines))
    assert len(got) == len(NAMED_EPISODE) and got[0].key == "natsu"
    v, _ = assess(got, STYLES, SPEECH)
    assert v.share == 1.0


def test_layers_group_by_key_not_raw_name():
    got = dialogue_lines(
        cues([(10, 12, "Default", "NATSU", "Hey!"), (10, 12.2, "Default", "Natsu ", "Hey!")])
    )
    assert [(ln.key, ln.end) for ln in got] == [("natsu", 12.2)]


def test_configured_track_equal_to_target_is_offered_once(probe_with_en_and_de):
    got = candidate_tracks(SOURCE, probe_with_en_and_de, cfg("en", names={"track": 2}))
    assert [c.track.index for c in got if c.track] == [2, 3, 4]
    assert [c.how for c in got] == ["track", "other-language", "other-language"]


def test_configured_file_equal_to_target_sidecar_is_offered_once(tmp_path: Path):
    src = tmp_path / "S01E03.mkv"
    side = tmp_path / "S01E03.en.ass"
    side.write_text(ass(NAMED_EPISODE))
    got = candidate_tracks(src, ProbeResult(), cfg("en", names={"file": "S01E03.en.ass"}))
    assert [(c.how, c.file) for c in got] == [("file", side)]


def test_non_numeric_names_track_is_a_media_error(probe_with_en_and_de):
    with pytest.raises(MediaError, match="track"):
        candidate_tracks(SOURCE, probe_with_en_and_de, cfg(names={"track": "eng"}))


def test_episode_placeholder_without_an_episode_is_an_error(tmp_path: Path, probe_with_en_and_de):
    with pytest.raises(MediaError, match="no episode was given"):
        candidate_tracks(
            tmp_path / "x.mkv", probe_with_en_and_de, cfg(names={"file": "{episode}.ass"})
        )


def test_unapplied_offset_reports_the_share_the_decision_accepted():
    # Lines 0.1 s late: the best offset gains under 0.1 of overlap, so nothing is applied, and
    # the verdict reports the share the pass decision accepted (the best offset's).
    from fukikae.names.align import alignment

    lines = dialogue_lines(cues(shift(NAMED_EPISODE, 0.1)))
    rep = alignment(lines, SPEECH)
    assert rep.best_offset_s == pytest.approx(0.1) and rep.share_at_offset > rep.overlap_share
    v, got = assess(lines, STYLES, SPEECH)
    assert v.offset_s == 0.0 and got[0].start == lines[0].start
    assert v.align_share == pytest.approx(rep.share_at_offset)
