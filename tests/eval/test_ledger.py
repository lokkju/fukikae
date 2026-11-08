import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fukikae.config import BackendConfig
from fukikae.eval import ledger
from fukikae.eval.ledger import (
    SCHEMA,
    LedgerRow,
    append_rows,
    git_commit,
    ledger_path,
    make_row,
    model_identity,
    options_hash,
    read_rows,
)


def row(**over: Any) -> LedgerRow:
    kw: dict[str, Any] = {
        "stage": "identity", "backend": "wespeaker", "model": "m", "revision": None,
        "options": {"threshold": 0.6}, "language": "ja", "genre": "live-action",
        "series": "Alice", "test_set": "aib", "split": "tune",
        "episodes": ["S01E01", "S01E02"], "metrics": {"fraction": 0.5},
    }  # fmt: skip
    return make_row(**{**kw, **over})


def test_ledger_path_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("FUKIKAE_RESULTS", str(tmp_path / "env.jsonl"))
    assert ledger_path(tmp_path / "flag.jsonl") == tmp_path / "flag.jsonl"
    assert ledger_path() == tmp_path / "env.jsonl"
    monkeypatch.delenv("FUKIKAE_RESULTS")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert ledger_path() == tmp_path / "xdg" / "fukikae" / "results.jsonl"
    monkeypatch.delenv("XDG_DATA_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert ledger_path() == tmp_path / "home" / ".local" / "share" / "fukikae" / "results.jsonl"


def test_tests_never_touch_the_real_ledger():
    assert not str(ledger_path()).startswith(str(Path.home() / ".local"))


def test_options_hash_ignores_key_order():
    assert options_hash({"a": 1, "b": {"c": 2}}) == options_hash({"b": {"c": 2}, "a": 1})
    assert options_hash({"a": 1}) != options_hash({"a": 2})
    assert len(options_hash({})) == 12


def test_append_then_read_round_trips(tmp_path):
    r = row()
    assert r.schema == SCHEMA and len(r.run_id) == 32 and r.date.endswith("+00:00")
    assert r.options_hash == options_hash({"threshold": 0.6})
    path = append_rows([r], tmp_path / "sub" / "r.jsonl")
    assert path == tmp_path / "sub" / "r.jsonl"
    rows, warnings = read_rows(path)
    assert rows == [r] and warnings == []


def test_append_adds_to_the_end(tmp_path):
    p = tmp_path / "r.jsonl"
    a, b, c = row(), row(split="holdout", episodes=["S01E03"]), row(test_set="x")
    append_rows([a, b], p)
    append_rows([c], p)
    assert read_rows(p)[0] == [a, b, c]
    assert p.read_text().count("\n") == 3


def test_append_after_a_truncated_line_starts_a_new_line(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"schema": 1, "stage": "iden')
    r = row()
    append_rows([r], p)
    rows, warnings = read_rows(p)
    assert rows == [r]
    assert warnings == ["r.jsonl:1: not JSON, skipped"]


def test_make_row_rejects_unknown_stage_and_split():
    with pytest.raises(ValueError, match="diarize"):
        row(stage="diarize")
    with pytest.raises(ValueError, match="dev"):
        row(split="dev")


def test_read_skips_truncated_and_future_rows(tmp_path):
    p = tmp_path / "results.jsonl"
    good = row()
    future = {**json.loads(json.dumps(good.__dict__)), "schema": 2}
    p.write_text(
        json.dumps(good.__dict__) + "\n"
        + '{"schema": 1, "stage": "iden' + "\n"
        + json.dumps(future) + "\n"
        + "\n"
    )  # fmt: skip
    rows, warnings = read_rows(p)
    assert len(rows) == 1 and len(warnings) == 2 and ":2:" in warnings[0]
    assert warnings == ["results.jsonl:2: not JSON, skipped", "results.jsonl:3: schema 2, skipped"]


def test_read_skips_rows_with_missing_or_unknown_fields(tmp_path):
    p = tmp_path / "results.jsonl"
    d = json.loads(json.dumps(row().__dict__))
    missing = {k: v for k, v in d.items() if k != "metrics"}
    p.write_text("\n".join(json.dumps(x) for x in (missing, {**d, "extra": 1}, [1, 2])) + "\n")
    rows, warnings = read_rows(p)
    assert rows == []
    assert [w.split(":")[1] for w in warnings] == ["1", "2", "3"]
    assert all(w.endswith("skipped") for w in warnings)


def test_read_missing_ledger_is_empty(tmp_path):
    assert read_rows(tmp_path / "none.jsonl") == ([], [])


def test_git_commit_outside_a_checkout_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, "_PACKAGE_DIR", tmp_path)
    assert git_commit() == "unknown"


def test_git_commit_in_a_checkout(monkeypatch, tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "f").write_text("x")
    git("add", "f")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    monkeypatch.setattr(ledger, "_PACKAGE_DIR", tmp_path)
    clean = git_commit()
    assert len(clean) == 12 and "dirty" not in clean
    (tmp_path / "f").write_text("y")
    assert git_commit() == clean + "-dirty"


def test_model_identity_reads_the_backend_and_hf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    refs = tmp_path / "hub" / "models--pyannote--wespeaker-voxceleb-resnet34-LM" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("abc123\n")
    assert model_identity("embed", BackendConfig("wespeaker", {})) == (
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "abc123",
    )
    assert model_identity("embed", BackendConfig("wespeaker", {"revision": "r1"}))[1] == "r1"
    # ASR backends keep the name in model_name.
    assert model_identity("asr", BackendConfig("faster_whisper", {})) == ("large-v3", None)


def test_model_identity_falls_back_on_any_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert model_identity("embed", BackendConfig("nope", {"model": "x/y"})) == ("x/y", None)
    assert model_identity("embed", BackendConfig("nope", {})) == (None, None)


def test_model_identity_honours_hf_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    refs = tmp_path / "hf" / "hub" / "models--a--b" / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("deadbeef")
    assert model_identity("embed", BackendConfig("wespeaker", {"model": "a/b"})) == (
        "a/b",
        "deadbeef",
    )
