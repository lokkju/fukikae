from pathlib import Path

import pytest
from typer.testing import CliRunner

from fukikae.eval.cli import eval_app
from fukikae.eval.groundtruth import GroundTruthLine, read_ground_truth
from fukikae.eval.groundtruth.bazinga import (
    Word,
    episode_code,
    is_pseudo,
    read_words,
    words_to_turns,
)

EP = "TheBigBangTheory.Season01.Episode01"


def _w(speaker: str, start: float, end: float, conf: float = 0.99, tok: str = "w") -> Word:
    return Word(EP, speaker, start, end, tok, conf)


def _row(speaker: str, start: float, end: float, tok: str, conf: float, ep: str = EP) -> str:
    return f"{ep} {speaker} {start} {end} {tok} {conf} _ leonard_hofstadter _\n"


def test_read_words_parses_the_annotation_columns(tmp_path: Path) -> None:
    p = tmp_path / f"{EP}.txt"
    p.write_text(
        _row("sheldon_cooper", 1.49, 1.61, "So", 0.99) + _row("penny", 2.0, 2.4, "Hi", 0.1)
    )
    assert read_words(p) == [
        Word(EP, "sheldon_cooper", 1.49, 1.61, "So", 0.99),
        Word(EP, "penny", 2.0, 2.4, "Hi", 0.1),
    ]


def test_episode_code_from_the_bazinga_episode_name() -> None:
    assert episode_code(EP) == "S01E01"
    assert episode_code("TheOffice.Season01.Episode05") == "S01E05"
    with pytest.raises(ValueError, match="nope"):
        episode_code("nope")


@pytest.mark.parametrize(
    ("speaker", "pseudo"),
    [
        ("all", True),
        ("multiple_persons", True),
        ("male_string_quartettist#unknown#TheBigBangTheory.Season01.Episode05", True),
        ("unknown", True),
        ("sheldon_cooper", False),
        ("penny", False),
    ],
)
def test_pseudo_speakers(speaker: str, pseudo: bool) -> None:
    assert is_pseudo(speaker) is pseudo


def test_consecutive_words_of_one_speaker_merge_into_a_turn() -> None:
    lines, stats = words_to_turns(
        [_w("sheldon_cooper", 1.0, 1.2, tok="So"), _w("sheldon_cooper", 1.3, 1.6, tok="if"),
         _w("penny", 2.0, 2.5, tok="Hi")]
    )  # fmt: skip
    assert [(ln.start, ln.end, ln.speaker, ln.text) for ln in lines] == [
        (1.0, 1.6, "sheldon_cooper", "So if"),
        (2.0, 2.5, "penny", "Hi"),
    ]
    assert (
        stats.turns == 2 and stats.kept == 3 and stats.speakers == {"sheldon_cooper": 1, "penny": 1}
    )


def test_a_gap_over_one_second_starts_a_new_turn() -> None:
    lines, _ = words_to_turns([_w("penny", 1.0, 1.5), _w("penny", 2.4, 2.8), _w("penny", 3.9, 4.2)])
    # 1.5 -> 2.4 is 0.9 s (same turn); 2.8 -> 3.9 is 1.1 s (new turn)
    assert [(ln.start, ln.end) for ln in lines] == [(1.0, 2.8), (3.9, 4.2)]


def test_a_turn_of_only_low_confidence_words_is_dropped() -> None:
    lines, stats = words_to_turns(
        [_w("penny", 1.0, 1.2, 0.1), _w("penny", 1.2, 1.4, 0.1), _w("leonard_hofstadter", 2.0, 2.3)]
    )
    assert [ln.speaker for ln in lines] == ["leonard_hofstadter"]
    assert stats.dropped["low-confidence"] == 2
    assert stats.turns_dropped == 1 and stats.kept == 1


def test_a_mixed_turn_is_trimmed_to_its_confident_words() -> None:
    lines, stats = words_to_turns(
        [_w("penny", 1.0, 1.2, 0.1, "uh"), _w("penny", 1.3, 1.6, 0.95, "hi"),
         _w("penny", 1.7, 2.0, 0.99, "there"), _w("penny", 2.1, 2.6, 0.1, "um")]
    )  # fmt: skip
    assert [(ln.start, ln.end, ln.text) for ln in lines] == [(1.3, 2.0, "hi there")]
    assert stats.kept == 2 and stats.dropped["low-confidence"] == 2


def test_min_confidence_is_a_threshold() -> None:
    words = [_w("penny", 1.0, 1.2, 0.95), _w("penny", 1.3, 1.6, 0.99)]
    lines, _ = words_to_turns(words, min_confidence=0.99)
    assert [(ln.start, ln.end) for ln in lines] == [(1.3, 1.6)]


def test_pseudo_speakers_are_dropped_and_split_turns() -> None:
    lines, stats = words_to_turns(
        [_w("penny", 1.0, 1.2), _w("all", 1.3, 1.5), _w("multiple_persons", 1.5, 1.6),
         _w("penny", 1.7, 1.9), _w("x#unknown#ep", 2.0, 2.2)]
    )  # fmt: skip
    assert [(ln.start, ln.end, ln.speaker) for ln in lines] == [
        (1.0, 1.2, "penny"),
        (1.7, 1.9, "penny"),
    ]
    assert stats.dropped["pseudo-speaker"] == 3 and stats.kept == 2


def test_time_scale_multiplies_every_time() -> None:
    lines, _ = words_to_turns([_w("penny", 10.0, 12.0)], time_scale=1.04271)
    assert lines[0].start == pytest.approx(10.4271)
    assert lines[0].end == pytest.approx(12.51252)


def test_turns_round_trip_through_the_harness_reader(tmp_path: Path) -> None:
    from fukikae.eval.groundtruth.transcript import write_named_ass

    lines, _ = words_to_turns(
        [_w("sheldon_cooper", 1.0, 1.25), _w("penny", 2.0, 2.5), _w("sheldon_cooper", 3.0, 3.5)]
    )
    p = tmp_path / "S01E01.ass"
    write_named_ass(p, lines)
    assert read_ground_truth(p, "ass", None) == [
        GroundTruthLine(1.0, 1.25, "sheldon_cooper"),
        GroundTruthLine(2.0, 2.5, "penny"),
        GroundTruthLine(3.0, 3.5, "sheldon_cooper"),
    ]


def test_cli_writes_one_named_ass_per_episode(tmp_path: Path) -> None:
    a = tmp_path / f"{EP}.txt"
    a.write_text(
        _row("sheldon_cooper", 1.0, 1.5, "So", 0.99)
        + _row("all", 2.0, 2.5, "Hey", 0.99)
        + _row("penny", 3.0, 3.5, "Hi", 0.1)
        + _row("penny", 3.6, 4.0, "there", 0.95)
    )
    ep2 = "TheBigBangTheory.Season01.Episode02"
    b = tmp_path / f"{ep2}.txt"
    b.write_text(_row("leonard_hofstadter", 5.0, 6.0, "Yo", 0.99, ep2))
    out = tmp_path / "labels"
    r = CliRunner().invoke(
        eval_app,
        ["bazinga-labels", str(a), str(b), "--out", str(out), "--time-scale", "2.0"],
    )
    assert r.exit_code == 0, r.output
    assert (
        "S01E01" in r.output and "pseudo-speaker 1" in r.output and "low-confidence 1" in r.output
    )
    assert read_ground_truth(out / "S01E01.ass", "ass", None) == [
        GroundTruthLine(2.0, 3.0, "sheldon_cooper"),
        GroundTruthLine(7.2, 8.0, "penny"),
    ]
    assert read_ground_truth(out / "S01E02.ass", "ass", None) == [
        GroundTruthLine(10.0, 12.0, "leonard_hofstadter")
    ]


def test_cli_refuses_two_files_for_one_episode(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text(_row("penny", 1.0, 1.5, "Hi", 0.99))
    b.write_text(_row("penny", 2.0, 2.5, "Hi", 0.99))
    r = CliRunner().invoke(
        eval_app, ["bazinga-labels", str(a), str(b), "--out", str(tmp_path / "o")]
    )
    assert r.exit_code == 1 and "S01E01" in r.output
    assert not (tmp_path / "o").exists()


def test_cli_names_a_malformed_line(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    a.write_text(_row("penny", 1.0, 1.5, "Hi", 0.99) + "garbage line\n")
    r = CliRunner().invoke(eval_app, ["bazinga-labels", str(a), "--out", str(tmp_path / "o")])
    assert r.exit_code == 1 and "a.txt:2" in r.output
