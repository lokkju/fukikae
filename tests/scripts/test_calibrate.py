"""The gender calibration script's tables and acceptance rule (identification Task 2)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gender-calibration" / "calibrate.py"
_spec = importlib.util.spec_from_file_location("gender_calibrate", _PATH)
assert _spec is not None and _spec.loader is not None
cal = importlib.util.module_from_spec(_spec)
sys.modules["gender_calibrate"] = cal
_spec.loader.exec_module(cal)
accept, Table = cal.accept, cal.Table


def table(main_in_other_class=0, unknown_share=0.0, line_agreement: float | None = 0.95):
    return Table(
        main_in_other_class=main_in_other_class,
        unknown_share=unknown_share,
        line_agreement=line_agreement,
    )


def test_acceptance_rule():
    assert not accept(table(main_in_other_class=1))
    assert not accept(table(unknown_share=0.12))
    assert accept(table(line_agreement=0.90))
    assert not accept(table(line_agreement=0.89))
    assert not accept(table(line_agreement=None))
    assert accept(table(unknown_share=0.10))


def _char(name, seconds, median, voiced=10.0):
    return {"kind": "character", "speaker": name, "seconds": seconds,
            "a0": {"median": median, "voiced_s": voiced},
            "pyin": {"median": median, "voiced_s": voiced}}  # fmt: skip


def _line(name, a0, pyin, dur=2.0, voiced=1.0):
    return {"kind": "line", "speaker": name, "start": 0.0, "end": dur,
            "a0": {"median": a0, "voiced_s": voiced},
            "pyin": {"median": pyin, "voiced_s": voiced}}  # fmt: skip


def test_table_counts_mains_in_the_other_class_and_unknown_seconds():
    truth = {"natsu": "m", "lucy": "f", "happy": "f", "erza": "f"}
    recs = [
        _char("Natsu", 100, 140),  # m, right
        _char("Lucy", 60, 150),  # f read m: other class
        _char("Happy", 40, 190),  # unknown band
        _char("Erza", 10, 150),  # not main (under 30 s): ignored
        _char("Nobody", 50, 140),  # no truth: listed, not counted
    ]
    t = cal.table(recs, truth, (165.0, 200.0))
    assert t.main_in_other_class == 1
    assert t.unknown_share == pytest.approx(40 / 200)
    assert t.confusion["f"] == {"m": 60, "u": 40, "f": 0}
    assert {m["speaker"] for m in t.mains} == {"Natsu", "Lucy", "Happy", "Nobody"}
    assert not accept(t)


def test_line_agreement_only_where_both_are_known():
    recs = [
        _line("A", 140, 145),  # m m
        _line("A", 250, 140),  # f m: disagree
        _line("A", 180, 140),  # A0 unknown: not compared
        _line("A", 250, 260, voiced=0.2),  # under the line minimum: unknown on both
        _line("A", 300, 310),  # f f
    ]
    t = cal.table(recs, {"a": "f"}, (165.0, 200.0))
    assert t.lines_compared == 3
    assert t.line_agreement == pytest.approx(2 / 3)
    assert t.line_unknown_share == pytest.approx(4 / 10)
    assert t.line_truth_agreement == pytest.approx(2 / 3)


def test_truth_file_takes_only_f_and_m(tmp_path: Path):
    good = tmp_path / "t.toml"
    good.write_text('[fairytail]\n"NATSU" = "m"\n"Lucy  Heartfilia" = "f"\n')
    assert cal.load_truth(good) == {"fairytail": {"natsu": "m", "lucy heartfilia": "f"}}
    bad = tmp_path / "b.toml"
    bad.write_text('[x]\n"A" = "u"\n')
    with pytest.raises(ValueError, match="use"):
        cal.load_truth(bad)


def test_shipped_truth_file_parses():
    truth = cal.load_truth(_PATH.parent / "truth.toml")
    assert {"fairytail", "bbt", "kaze"} <= set(truth)
