import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.artifacts import write_json
from fukikae.cli import app
from fukikae.models import Diarization, Turn

HEADER = """[Script Info]
Title: x

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def stamp(t: float) -> str:
    return f"0:{int(t // 60):02d}:{t % 60:05.2f}"


def setup_ws(tmp_path: Path, episodes: list[str], turns_each: int = 6) -> Path:
    """Two speakers alternating 4 s turns: LUCY at 0, 10, 20 s ..., LAXUS at 5, 15, 25 s ..."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "fukikae.toml").write_text('source_language = "ja"\n', encoding="utf-8")
    gt = tmp_path / "gt"
    gt.mkdir()
    for ep in episodes:
        work = root / ".fukikae" / "episodes" / ep
        (work / "04-diarize").mkdir(parents=True)
        (work / "01-separate").mkdir(parents=True)
        (work / "01-separate" / "vocals_16k.wav").write_bytes(f"audio {ep}".encode())
        turns, lines = [], []
        for k in range(2 * turns_each):
            start = 5.0 * k
            spk, name = ("SPEAKER_00", "LUCY") if k % 2 == 0 else ("SPEAKER_01", "LAXUS")
            turns.append(Turn(start=start, end=start + 4.0, speaker=spk))
            lines.append(
                f"Dialogue: 0,{stamp(start)},{stamp(start + 4.0)},Default,{name},0,0,0,,hi\n"
            )
        write_json(work / "04-diarize" / "turns.json", Diarization(backend="x", turns=turns))
        (gt / f"Show - {ep}.ass").write_text(HEADER + "".join(lines), encoding="utf-8")
    return root


class FakeResident:
    """Stands in for the GPU embedder: LUCY's turns embed to e0, LAXUS's to e1."""

    calls = 0

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def __call__(self, _emb_cfg, _audio, spans):
        FakeResident.calls += 1
        rows = np.zeros((len(spans), 4), dtype=np.float32)
        for i, (s, _) in enumerate(spans):
            rows[i, int(s // 5) % 2] = 1.0
        return rows


def files_under(path: Path) -> set[Path]:
    return {p for p in path.rglob("*") if p.is_file()}


def test_identity_cli_tune_and_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fukikae.eval.identity import cache as mod

    root = setup_ws(tmp_path, ["S01E01", "S01E02", "S01E03"])
    episodes_dir = root / ".fukikae" / "episodes"
    before = files_under(episodes_dir)
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    FakeResident.calls = 0
    out = tmp_path / "result.json"
    args = [
        "eval", "identity", str(root), "--tune", "S01E01,S01E02", "--holdout", "S01E03",
        "--gt", "ass", "--gt-file", str(tmp_path / "gt"), "--json", str(out),
    ]  # fmt: skip
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "S01E01" in r.output and "LUCY" in r.output
    assert "held out: S01E03" in r.output
    assert r.output.count("verdict: pass") == 2
    data = json.loads(out.read_text())
    assert data["tune"]["score"]["verdict"] == "pass"
    assert data["tune"]["score"]["recurring"] == 2
    assert data["holdout"]["episode"] == "S01E03"
    assert data["holdout"]["score"]["verdict"] == "pass"
    # Held-out shares are over the held-out speech alone (6 clean spans of 3.8 s each).
    lucy = next(c for c in data["holdout"]["score"]["characters"] if c["name"] == "LUCY")
    assert lucy["speech"] == pytest.approx(22.8)
    assert lucy["by_episode"]["tune"] == pytest.approx(45.6)
    assert data["settings"]["embedder"] == {"backend": "ecapa", "options": {}}
    assert data["settings"]["threshold"] == 0.6
    assert data["settings"]["min_speech"] == 5.0
    # Nearest-centroid accuracy: E1 trains, E2 tests; the tuning episodes train the holdout.
    assert data["tune"]["nearest_centroid"] == {"LUCY": 1.0, "LAXUS": 1.0, "(overall)": 1.0}
    assert data["holdout"]["nearest_centroid"]["(overall)"] == 1.0
    assert "nearest-centroid accuracy" in r.output
    # Nothing written beside the pipeline's artifacts; the embeddings sit in the cache.
    assert files_under(episodes_dir) == before
    assert len(list((root / ".fukikae" / "cache" / "embeddings").rglob("*.npy"))) == 3
    assert FakeResident.calls == 3
    # A second run is served from the cache.
    r2 = CliRunner().invoke(app, args)
    assert r2.exit_code == 0, r2.output
    assert FakeResident.calls == 3


def test_identity_cli_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fukikae.eval.identity import cache as mod

    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    out = tmp_path / "r.json"
    r = CliRunner().invoke(
        app,
        ["eval", "identity", str(root), "--tune", "S01E01,S01E02", "--gt-file",
         str(tmp_path / "gt"), "--segmentation", "subtitles", "--embedder", "ecapa",
         "--embedder-option", "model=x/y", "--embedder-option", "batch=8",
         "--threshold", "0.7", "--min-speech", "20", "--json", str(out)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    s = json.loads(out.read_text())["settings"]
    assert s["segmentation"] == "subtitles"
    assert s["embedder"] == {"backend": "ecapa", "options": {"model": "x/y", "batch": 8}}
    assert s["threshold"] == 0.7
    assert s["matcher"] == "centroid"
    assert s["min_speech"] == 20.0
    assert json.loads(out.read_text())["tune"]["score"]["recurring"] == 2


def test_identity_cli_refuses_misaligned_ground_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fukikae.eval.identity import cache as mod

    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    late = HEADER + "".join(
        f"Dialogue: 0,{stamp(200.0 + 5 * k)},{stamp(204.0 + 5 * k)},Default,LUCY,0,0,0,,hi\n"
        for k in range(6)
    )
    (tmp_path / "gt" / "Show - S01E02.ass").write_text(late, encoding="utf-8")
    args = ["eval", "identity", str(root), "--tune", "S01E01,S01E02", "--gt-file",
            str(tmp_path / "gt")]  # fmt: skip
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1
    assert "line up" in r.output
    r = CliRunner().invoke(app, [*args, "--force"])
    assert r.exit_code == 0, r.output


def test_identity_cli_bad_arguments(tmp_path: Path):
    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    gt = str(tmp_path / "gt")
    base = ["eval", "identity", str(root), "--gt-file", gt]
    r = CliRunner().invoke(app, [*base, "--tune", "S01E01,S01E02", "--matcher", "nope"])
    assert r.exit_code == 1 and "matcher" in r.output
    r = CliRunner().invoke(app, [*base, "--tune", "S01E01,S01E02", "--holdout", "S01E02"])
    assert r.exit_code == 1 and "held-out" in r.output
    r = CliRunner().invoke(app, [*base, "--tune", "S01E01", "--embedder-option", "novalue"])
    assert r.exit_code == 1 and "k=v" in r.output


def test_identity_help_documents_the_pooled_holdout():
    r = CliRunner().invoke(app, ["eval", "identity", "--help"], env={"COLUMNS": "200"})
    assert r.exit_code == 0
    assert "pooled" in r.output and "--min-speech" in r.output
    assert "modal id in the tuning episodes" in r.output
