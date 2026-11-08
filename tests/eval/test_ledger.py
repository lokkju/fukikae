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


def test_read_skips_rows_that_are_not_ledger_rows(tmp_path):
    p = tmp_path / "results.jsonl"
    d = json.loads(json.dumps(row().__dict__))
    missing = {k: v for k, v in d.items() if k != "metrics"}
    bad = (missing, [1, 2], {**d, "schema": 0}, {**d, "schema": "1"}, {**d, "schema": True})
    p.write_text("\n".join(json.dumps(x) for x in bad) + "\n")
    rows, warnings = read_rows(p)
    assert rows == []
    assert [w.split(":")[1] for w in warnings] == ["1", "2", "3", "4", "5"]
    assert all(w.endswith("skipped") for w in warnings)


def test_read_ignores_unknown_fields(tmp_path):
    """Ruling GT2: an additive field (a later tts row's, say) never costs the row."""
    p = tmp_path / "results.jsonl"
    r = row()
    p.write_text(json.dumps({**json.loads(json.dumps(r.__dict__)), "voice": "v1"}) + "\n")
    rows, warnings = read_rows(p)
    assert rows == [r] and warnings == []


def test_read_keeps_rows_with_unicode_line_separators(tmp_path):
    """Rows are written with ensure_ascii=False; U+2028, U+0085, \\x0b and \\x0c stay raw
    and must not split a line."""
    p = tmp_path / "results.jsonl"
    odd = row(tag="a\u2028b\u0085c\x0bd\x0ce")
    plain = row()
    append_rows([odd, plain], p)
    assert "\u2028" in p.read_text(encoding="utf-8")
    with p.open("a", encoding="utf-8") as f:
        f.write("not json\n")
    rows, warnings = read_rows(p)
    assert rows == [odd, plain]
    assert warnings == ["results.jsonl:3: not JSON, skipped"]


def _append_worker(args: tuple[str, int, int]) -> None:
    path, worker, m = args
    for k in range(m):
        append_rows([row(tag=f"w{worker}-{k}", detail={"pad": "x" * 5000})], Path(path))


def test_concurrent_appends_never_interleave(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    p = tmp_path / "results.jsonl"
    n, m = 6, 25
    with ProcessPoolExecutor(max_workers=n) as pool:
        list(pool.map(_append_worker, [(str(p), w, m) for w in range(n)]))
    rows, warnings = read_rows(p)
    assert warnings == []
    assert len(rows) == n * m
    assert {r.tag for r in rows} == {f"w{w}-{k}" for w in range(n) for k in range(m)}


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
    (tmp_path / "untracked").write_text("z")
    assert git_commit() == clean  # untracked files don't mark a row dirty
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
