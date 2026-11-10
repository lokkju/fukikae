import subprocess
import sys

import pytest

from fukikae.subtitles import (
    Cue,
    filter_cue,
    parse_ass_cues,
    parse_text_cues,
    read_cues,
    spoken_text,
    strip_markup,
)

ASS_HEAD = (
    "[Script Info]\nScriptType: v4.00+\n\n[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
)


def _ass(*rows: tuple[str, str, str]) -> str:
    """rows of (style, name, text)"""
    return ASS_HEAD + "".join(
        f"Dialogue: 0,0:00:{i:02d}.00,0:00:{i + 1:02d}.00,{s},{n},0,0,0,,{t}\n"
        for i, (s, n, t) in enumerate(rows)
    )


def cue(text: str, style: str = "", **kw) -> Cue:
    return Cue(0.0, 1.0, text, style, **kw)


def test_groundtruth_imports_still_work():
    from fukikae import subtitles
    from fukikae.eval.groundtruth import base

    for name in (
        "Cue", "read_cues", "read_text", "parse_text_cues", "parse_ass_cues",
        "parse_ttml_cues", "strip_markup", "COLOURS", "colour_from_hex",
    ):  # fmt: skip
        assert getattr(base, name) is getattr(subtitles, name), name


def test_pipeline_modules_never_import_eval():
    code = (
        "import sys\n"
        "import fukikae.stages.translate, fukikae.cues, fukikae.subtitles, fukikae.media\n"
        "bad = sorted(m for m in sys.modules if m.startswith('fukikae.eval'))\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_parsers_keep_their_ground_truth_fixes(tmp_path):
    # bidi marks at a line start, UTF-16 by BOM, ARIB 8-digit colours, furigana \fscy
    assert strip_markup("‪<i>Hi</i>‬") == "Hi"
    p = tmp_path / "u16.srt"
    p.write_bytes(
        b"\xff\xfe" + "1\r\n00:00:01,000 --> 00:00:02,000\r\nこんにちは\r\n".encode("utf-16-le")
    )
    assert read_cues(p) == [Cue(1.0, 2.0, "こんにちは")]
    got = parse_ass_cues(_ass(("Default", "", "{\\c&H0000FF00}a"), ("Rubi", "", "{\\fscy50}ふり")))
    assert got[0].text == "<c.green>a" and not got[0].ruby
    assert got[1].ruby
    assert (
        parse_text_cues("1\n00:00:01,000 --> 00:00:02,000\n<font color=#ffff00>Y</font>\n")[0].text
        == "<c.yellow>Y</c>"
    )


def test_ass_cues_carry_name_and_placement():
    got = parse_ass_cues(
        _ass(
            ("Default", "LUCY", "Fine."),
            ("Default", "SIGN", "{\\fad(400,1)\\pos(320,238)}A Card in the Hand"),
            ("Default", "NATSU", "{\\an8}Up here!"),
            ("Default", "", "{\\move(1,2,3,4)}Moving"),
        )
    )
    assert [c.name for c in got] == ["LUCY", "SIGN", "NATSU", ""]
    assert [c.placement for c in got] == ["", "pos", "top", "pos"]
    assert got[1].text == "A Card in the Hand"


def test_filter_cue_rules():
    assert filter_cue(cue("[door slams]")) == "sdh"
    assert filter_cue(cue("(sighs)")) == "sdh"
    assert filter_cue(cue("- [gasps]\n- (laughing)")) == "sdh"
    assert filter_cue(cue("♪ la la ♪")) == "song"
    assert filter_cue(cue("la la", style="OP-Romaji")) == "song"
    assert filter_cue(cue("la la", style="Karaoke")) == "song"
    assert filter_cue(cue("la la", style="ED")) == "song"
    assert filter_cue(cue("Next Time", style="Default", name="SIGN")) == "sign"
    assert filter_cue(cue("A Card in the Hand", placement="pos")) == "sign"
    assert filter_cue(cue("Guild Hall", style="Signs")) == "sign"
    assert filter_cue(cue("Scarlet Showdown", style="Next Ep Title")) == "sign"
    assert filter_cue(cue("Plain line.")) is None
    assert filter_cue(cue("I'm (really) fine.")) is None
    # styles that only contain the letters are not songs
    assert filter_cue(cue("Line.", style="Flashback")) is None
    assert filter_cue(cue("Line.", style="Default-Top")) is None
    assert filter_cue(cue("Line.", style="Narration")) is None


def test_filter_top_line_with_a_speaker_is_dialogue():
    # Crunchyroll raises dialogue with \an8 when a sign holds the bottom; an unnamed one is a sign
    assert filter_cue(cue("Up here!", name="NATSU", placement="top")) is None
    assert filter_cue(cue("Magnolia", placement="top")) == "sign"


def test_filter_rules_are_configurable():
    assert filter_cue(cue("[door slams]"), rules=("song", "sign")) is None
    assert filter_cue(cue("♪ la la ♪"), rules=("sdh",)) is None
    with pytest.raises(ValueError, match="unknown"):
        filter_cue(cue("x"), rules=("sdh", "lyrics"))


def test_spoken_text_drops_prefix_and_notes():
    assert spoken_text(cue("LUCY: (sighs) Fine.")) == "Fine."
    assert spoken_text(cue("- NATSU: Let's go!\n- HAPPY: Aye, sir!")) == "Let's go! Aye, sir!"
    assert spoken_text(cue("I'm <i>so</i> [whispering] tired.")) == "I'm so tired."
    assert spoken_text(cue("Wait: what?")) == "Wait: what?"  # not an upper-case name
    assert spoken_text(cue("<c.yellow>Line one\nline two")) == "Line one line two"
    assert spoken_text(cue("[door slams]")) == ""


# ---- fix round 1: real dialogue that the filter dropped ----


def test_speaker_names_ed_op_insert_are_dialogue():
    assert filter_cue(cue("Hey!", name="Ed")) is None  # Edward Elric
    assert filter_cue(cue("Hey!", name="Op")) is None
    assert filter_cue(cue("Hey!", name="Insert")) is None
    assert filter_cue(cue("la la", style="ED")) == "song"
    assert filter_cue(cue("la la", style="OP")) == "song"
    assert filter_cue(cue("la la", name="Lyrics")) == "song"
    assert filter_cue(cue("Up here!", name="Ed", placement="top")) is None


def test_positioned_line_with_a_speaker_is_dialogue():
    assert filter_cue(cue("Over here!", name="NATSU", placement="pos")) is None
    assert filter_cue(cue("Magnolia", placement="pos")) == "sign"
    assert filter_cue(cue("Magnolia", name="SIGN", placement="pos")) == "sign"


def test_music_mark_makes_a_song_only_when_it_leads_or_is_all():
    assert filter_cue(cue("♪ la la ♪")) == "song"
    assert filter_cue(cue("♪ Fly away to the sky")) == "song"
    assert filter_cue(cue("- ♫ la la\nla la ♫")) == "song"
    assert filter_cue(cue("♪♪♪")) == "song"
    assert filter_cue(cue("Good morning♪")) is None
    assert spoken_text(cue("Good morning♪")) == "Good morning"


def test_common_words_before_a_colon_are_not_speakers():
    assert spoken_text(cue("OK: fine.")) == "OK: fine."
    assert spoken_text(cue("WAIT: what?")) == "WAIT: what?"
    assert spoken_text(cue("NOTE: this is it.")) == "NOTE: this is it."
    assert spoken_text(cue("LUCY: Fine.")) == "Fine."
