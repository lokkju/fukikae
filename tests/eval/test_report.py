import dataclasses
import json
from typing import Any

from typer.testing import CliRunner

from fukikae.cli import app
from fukikae.eval.ledger import LedgerRow, append_rows, make_row
from fukikae.eval.report import HEADLINE, Matrix, build_matrix, filter_rows, render


def row(
    language: str = "en", genre: str = "live-action", date: str | None = None, **over: Any
) -> LedgerRow:
    kw: dict[str, Any] = {
        "stage": "identity", "backend": "wespeaker", "model": "pyannote/w", "revision": None,
        "options": {"threshold": 0.6}, "language": language, "genre": genre, "series": "bbt",
        "test_set": "bbt", "split": "tune", "episodes": ["S01E01", "S01E02"],
        "metrics": {"fraction": 0.6, "clean": 5, "merges": 1},
    }  # fmt: skip
    r = make_row(**{**kw, **over})
    return r if date is None else dataclasses.replace(r, date=date)


def test_headline_metrics():
    assert HEADLINE == {
        "asr": "cer",
        "identity": "fraction",
        "embedder": "nc_overall",
        "tts": "speaker_similarity",
        "names": "cue_accuracy",
        "gender": "mismatch_rate",
    }


def test_matrix_columns_by_language_and_genre():
    m = build_matrix([
        row("en", metrics={"fraction": 0.6, "clean": 3, "merges": 0}),
        row("de", metrics={"fraction": 0.4, "clean": 2, "merges": 1}),
    ])  # fmt: skip
    assert m.columns == ["de/live-action", "en/live-action"]
    assert len(m.rows) == 1
    label, cells = m.rows[0]
    assert label.startswith("identity | bbt | tune | wespeaker pyannote/w | opts:")
    assert label.endswith("| -")
    assert cells == ["0.40 c2 m1", "0.60 c3 m0"]


def test_default_genre_is_its_own_column():
    m = build_matrix([row("en", "live-action"), row("en", "default")])
    assert m.columns == ["en/default", "en/live-action"]


def test_latest_row_wins_per_key():
    old = row(date="2026-11-08T10:00:00+00:00", metrics={"fraction": 0.2})
    new = row(date="2026-11-09T10:00:00+00:00", metrics={"fraction": 0.7})
    assert build_matrix([new, old]).rows[0][1] == ["0.70"]
    assert build_matrix([old, new]).rows[0][1] == ["0.70"]


def test_rows_split_by_key_and_missing_cells():
    m = build_matrix([
        row("en"),
        row("de", options={"threshold": 0.5}),
        row("en", split="holdout", episodes=["S01E03"], tag="pick"),
    ])  # fmt: skip
    assert len(m.rows) == 3
    assert sum(cells.count("-") for _, cells in m.rows) == 3
    assert any(label.endswith("| pick") for label, _ in m.rows)


def test_series_sharing_a_test_set_name_stay_apart():
    m = build_matrix([row("en", series="Grid"), row("en", series="Fairy Tail")])
    assert len(m.rows) == 2
    labels = sorted(label for label, _ in m.rows)
    assert " | bbt [Fairy Tail] | " in labels[0] and " | bbt [Grid] | " in labels[1]
    # One series alone keeps the plain label.
    assert " | bbt | " in build_matrix([row("en", series="Grid")]).rows[0][0]


def test_rows_grouped_by_stage():
    m = build_matrix([
        row(stage="identity"),
        row(stage="asr", backend="qwen3_asr", metrics={"cer": 0.1234}),
        row(stage="embedder", metrics={"nc_overall": 0.91}),
    ])  # fmt: skip
    assert [label.split(" | ")[0] for label, _ in m.rows] == ["asr", "identity", "embedder"]
    assert m.rows[0][1] == ["0.1234"]
    assert m.rows[2][1] == ["0.91"]


def test_metric_override():
    m = build_matrix(
        [row(metrics={"fraction": 0.6, "coverage": 0.25, "verdict": "fail"})], "coverage"
    )
    assert m.rows[0][1] == ["0.25"]
    m = build_matrix([row(metrics={"fraction": 0.6, "verdict": "fail"})], "verdict")
    assert m.rows[0][1] == ["fail"]
    m = build_matrix([row(metrics={"fraction": 0.6})], "nope")
    assert m.rows[0][1] == ["-"]


def test_filters_stage_backend_set_split_tag():
    rows = [
        row(),
        row(stage="asr", backend="qwen3_asr", metrics={"cer": 0.1}),
        row(test_set="aib"),
        row(split="holdout", episodes=["S01E03"]),
        row(tag="t1"),
    ]
    assert len(filter_rows(rows)) == 5
    assert len(filter_rows(rows, stages=["asr"])) == 1
    assert len(filter_rows(rows, backends=["qwen3_asr", "nope"])) == 1
    assert len(filter_rows(rows, sets=["aib"])) == 1
    assert len(filter_rows(rows, splits=["holdout"])) == 1
    assert len(filter_rows(rows, tags=["t1"])) == 1
    assert len(filter_rows(rows, stages=["identity"], sets=["bbt"], splits=["tune"])) == 2


def test_identity_cell_shows_clean_and_merges():
    m = build_matrix([row(metrics={"fraction": 0.5625, "clean": 5, "merges": 1})])
    assert m.rows[0][1] == ["0.56 c5 m1"]


def test_gender_cell_shows_mismatched_over_known():
    m = build_matrix([row(stage="gender", metrics={
        "mismatch_rate": 0.0678, "mismatched_strict": 25, "known": 369,
    })])  # fmt: skip
    assert m.rows[0][1] == ["0.07 25/369"]
    m = build_matrix([row(stage="gender", metrics={"mismatch_rate": 0.0678})])
    assert m.rows[0][1] == ["0.07"]  # an older row without the counts


def test_render_text_aligns_columns():
    m = Matrix(["de/live-action", "en/live-action"], [("identity | bbt", ["0.40", "0.60"])])
    lines = render(m).splitlines()
    assert len(lines) == 2
    assert lines[0].index("de/live-action") == lines[1].index("0.40")


def test_render_markdown_is_a_table():
    m = Matrix(["de/live-action"], [("identity | bbt", ["0.40"])])
    lines = render(m, "md").splitlines()
    assert lines[0] == "| row | de/live-action |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| identity \\| bbt | 0.40 |"


def test_report_cli_warns_on_bad_lines_and_still_renders(tmp_path):
    p = tmp_path / "results.jsonl"
    append_rows([row("en"), row("de")], p)
    with p.open("a") as f:
        f.write('{"schema": 1, "stage": "iden')
    r = CliRunner().invoke(app, ["eval", "report", "--ledger", str(p)])
    assert r.exit_code == 0, r.output
    assert "results.jsonl:3: not JSON, skipped" in r.stderr
    assert "de/live-action" in r.stdout and "0.60 c5 m1" in r.stdout


def test_report_cli_reads_fukikae_results_and_filters(tmp_path, monkeypatch):
    p = tmp_path / "env.jsonl"
    monkeypatch.setenv("FUKIKAE_RESULTS", str(p))
    append_rows([row("en"), row("ja", stage="asr", backend="qwen3_asr", metrics={"cer": 0.2})], p)
    r = CliRunner().invoke(app, ["eval", "report", "--stage", "asr", "--format", "md"])
    assert r.exit_code == 0, r.output
    assert r.stdout.startswith("| row | ja/live-action |")
    assert "identity" not in r.stdout


def test_report_cli_exports_filtered_rows(tmp_path):
    p = tmp_path / "results.jsonl"
    rows = [row("en"), row("en", test_set="aib")]
    append_rows(rows, p)
    out = tmp_path / "export" / "x.jsonl"
    r = CliRunner().invoke(
        app, ["eval", "report", "--ledger", str(p), "--set", "aib", "--export", str(out)]
    )
    assert r.exit_code == 0, r.output
    lines = out.read_text().splitlines()
    assert [json.loads(x)["run_id"] for x in lines] == [rows[1].run_id]


def test_report_cli_empty_ledger(tmp_path):
    r = CliRunner().invoke(app, ["eval", "report", "--ledger", str(tmp_path / "none.jsonl")])
    assert r.exit_code == 0
    assert "no ledger rows" in r.stdout
