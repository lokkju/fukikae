from pathlib import Path

import pytest
from typer.testing import CliRunner

from fukikae.eval.cli import eval_app
from fukikae.eval.groundtruth import GroundTruthLine, read_ground_truth
from fukikae.eval.groundtruth.base import Cue, Namer
from fukikae.eval.groundtruth.transcript import (
    NamedLine,
    TranscriptLine,
    align_transcript,
    read_transcript,
    write_named_ass,
)

BIGBANG_HTML = """<div class="entrytext"><p>Scene: A corridor at a sperm bank.</p>
<p>Sheldon: So if a photon is directed through a plane with two slits in it&#8230;</p>
<p>Leonard: Agreed, what&#8217;s your point?</p><p>(They sit and begin to fill in forms).</p>
<p>Receptionist (off): Hang on.</p></div>"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _tl(*pairs: tuple[str, str]) -> list[TranscriptLine]:
    return [TranscriptLine(s, t) for s, t in pairs]


def test_read_bigbangtrans_html(tmp_path: Path) -> None:
    lines = read_transcript(_write(tmp_path, "S01E01.html", BIGBANG_HTML))
    assert [ln.speaker for ln in lines] == ["Sheldon", "Leonard", "Receptionist"]
    assert lines[1].text == "Agreed, what’s your point?"


def test_read_txt_drops_directions_and_parentheticals(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "S01E01.txt",
        "Scene: Lab\nPenny (on phone): Hi (waves) there.\n(A door closes)\nWritten by: Bill\n",
    )
    assert read_transcript(p) == [TranscriptLine("Penny", "Hi there.")]


def test_unknown_suffix_names_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\.json"):
        read_transcript(_write(tmp_path, "a.json", "{}"))


def test_align_assigns_speakers_across_rewrapped_cues() -> None:
    lines = _tl(
        ("Sheldon", "If a photon is directed through a plane with two slits in it"),
        ("Leonard", "Agreed, what is your point?"),
        ("Sheldon", "There is no point, I just think it is a good idea for a tee shirt."),
    )
    cues = [
        Cue(1.0, 3.0, "If a photon is directed through\na plane with two slits in it"),
        Cue(3.2, 4.5, "Agreed, what is your point?"),
        Cue(5.0, 6.0, "There is no point, I just think"),
        Cue(6.0, 7.5, "it is a good idea for a tee shirt."),
    ]
    res = align_transcript(lines, cues, Namer())
    assert [n.speaker for n in res.named] == ["Sheldon", "Leonard", "Sheldon", "Sheldon"]
    assert res.cue_share == 1.0 and res.word_share == pytest.approx(1.0)
    assert res.speakers == {"Sheldon": 3, "Leonard": 1}


def test_dash_cue_with_two_speakers_is_split() -> None:
    lines = _tl(
        ("Sheldon", "Quantum mechanics describes the wave function of every particle"),
        ("Leonard", "What is your point?"),
        ("Sheldon", "There is no point."),
    )
    cues = [
        Cue(0.0, 2.0, "Quantum mechanics describes the wave function of every particle"),
        Cue(2.0, 4.0, "- What is your point?\n- There is no point."),
    ]
    res = align_transcript(lines, cues, Namer())
    assert [(n.speaker, n.text) for n in res.named[1:]] == [
        ("Leonard", "What is your point?"),
        ("Sheldon", "There is no point."),
    ]
    a, b = res.named[1], res.named[2]
    assert a.start == pytest.approx(2.0) and a.end == pytest.approx(b.start)
    assert b.end == pytest.approx(4.0)


def test_scene_missing_from_subtitles_does_not_bleed() -> None:
    cut = [
        ("Howard", "My mother made brisket for the whole neighborhood yesterday"),
        ("Raj", "Nobody asked for a brisket from the old country at all"),
        ("Howard", "She never asks anybody about anything before cooking"),
        ("Raj", "The oven in that kitchen runs forever and ever"),
        ("Howard", "Then we eat leftovers for eleven days straight"),
        ("Raj", "Eleven days is longer than the war lasted"),
    ]
    lines = _tl(
        ("Leonard", "The universe is expanding faster than anyone expected today"),
        ("Sheldon", "Dark energy is the likely explanation for that acceleration"),
        *cut,
        ("Penny", "Hey guys, I brought pizza from the place downstairs"),
        ("Leonard", "You are an angel sent from heaven above"),
    )
    cues = [
        Cue(1, 3, "The universe is expanding faster than anyone expected today"),
        Cue(3, 5, "Dark energy is the likely explanation for that acceleration"),
        Cue(60, 62, "Hey guys, I brought pizza from the place downstairs"),
        Cue(62, 64, "You are an angel sent from heaven above"),
    ]
    res = align_transcript(lines, cues, Namer())
    assert [n.speaker for n in res.named] == ["Leonard", "Sheldon", "Penny", "Leonard"]


def test_subtitle_lines_missing_from_transcript_stay_unlabelled() -> None:
    lines = _tl(("Leonard", "Good morning, did you sleep at all last night?"))
    cues = [
        Cue(0, 2, "Good morning, did you sleep at all last night?"),
        Cue(2, 4, "Completely unrelated sentence about zebras and quartz"),
    ]
    res = align_transcript(lines, cues, Namer())
    assert len(res.named) == 1 and res.cue_share == 0.5


def test_group_speakers_are_dropped() -> None:
    lines = _tl(("All", "Hello Penny, welcome to the neighborhood!"), ("Both", "Hello again"))
    cues = [Cue(0, 2, "Hello Penny, welcome to the neighborhood!"), Cue(2, 3, "Hello again")]
    assert align_transcript(lines, cues, Namer()).named == []


@pytest.mark.parametrize(
    ("speaker", "kept"),
    [
        ("All together", False),
        ("Everyone", False),
        ("Both", False),
        ("Leonard, Sheldon and Howard together", False),
        ("Leonard", True),
        ("Mrs Cooper", True),
    ],
)
def test_group_speaker_words_anywhere_in_the_name(speaker: str, kept: bool) -> None:
    lines = _tl((speaker, "Hello Penny, welcome to the neighborhood!"))
    cues = [Cue(0, 2, "Hello Penny, welcome to the neighborhood!")]
    assert bool(align_transcript(lines, cues, Namer()).named) is kept


def test_aliases_apply() -> None:
    lines = _tl(("Dr. Cooper", "Bazinga, that was a good joke indeed"))
    cues = [Cue(0, 2, "Bazinga, that was a good joke indeed")]
    res = align_transcript(lines, cues, Namer({"Dr. Cooper": "Sheldon"}))
    assert res.named[0].speaker == "Sheldon"


def test_named_ass_round_trips_through_the_harness_reader(tmp_path: Path) -> None:
    p = tmp_path / "x.ass"
    write_named_ass(
        p, [NamedLine(1.0, 2.5, "Penny", "Hi, there."), NamedLine(3.0, 4.0, "Leonard", "Yo")]
    )
    assert read_ground_truth(p, "ass", None) == [
        GroundTruthLine(1.0, 2.5, "Penny"),
        GroundTruthLine(3.0, 4.0, "Leonard"),
    ]


def test_cli_pairs_files_by_episode_code(tmp_path: Path) -> None:
    tr, sub = tmp_path / "tr", tmp_path / "sub"
    tr.mkdir()
    sub.mkdir()
    _write(
        tr,
        "S01E01.html",
        "<p>Leonard: Agreed, what is your point?</p><p>Sheldon: There is no point.</p>",
    )
    _write(tr, "S01E02.html", "<p>Penny: Not this one.</p>")
    _write(
        sub,
        "The.Show.S01E01.en.srt",
        "1\n00:00:01,000 --> 00:00:03,000\nAgreed, what is your point?\n\n"
        "2\n00:00:03,000 --> 00:00:05,000\nThere is no point.\n",
    )
    out = tmp_path / "out"
    r = CliRunner().invoke(
        eval_app,
        ["transcript-align", "--transcripts", str(tr), "--subtitles", str(sub),
         "--out", str(out), "--episodes", "S01E01"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "S01E01" in r.output and "Leonard" in r.output
    got = read_ground_truth(out / "S01E01.ass", "ass", None)
    assert [x.speaker for x in got] == ["Leonard", "Sheldon"]
    assert not (out / "S01E02.ass").exists()


def test_cli_reports_missing_episode(tmp_path: Path) -> None:
    r = CliRunner().invoke(
        eval_app,
        ["transcript-align", "--transcripts", str(tmp_path), "--subtitles", str(tmp_path),
         "--out", str(tmp_path / "o"), "--episodes", "S01E09"],
    )  # fmt: skip
    assert r.exit_code == 1 and "S01E09" in r.output
