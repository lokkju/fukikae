"""`fukikae registry adapt`: fit the series' LDA projection from labeled episodes."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from typer.testing import CliRunner

from fukikae.artifacts import write_json
from fukikae.cli import app
from fukikae.job import Manifest, StageRecord, save_manifest
from fukikae.models import Diarization, Turn
from fukikae.registry.store import Registry
from fukikae.stages.diarize import clean_spans
from fukikae.workspace import Episode, Workspace

runner = CliRunner()
HEADER = """[Script Info]
Title: x

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
NAMES = {"SPEAKER_00": "LUCY", "SPEAKER_01": "LAXUS", "SPEAKER_02": "ERZA"}
DIM = 6


def stamp(t: float) -> str:
    return f"0:{int(t // 60):02d}:{t % 60:05.2f}"


def voice(rng: np.random.Generator, k: int) -> np.ndarray:
    v = np.eye(DIM)[0] + 0.2 * np.eye(DIM)[k + 1] + rng.normal(0, 0.05, DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_ws(tmp_path: Path, episodes: list[str], series: str = "show") -> Workspace:
    """Three speakers taking 4 s turns in rotation, 6 turns each; clusters.npz rows are
    the clean spans, each embedded near its speaker's voice."""
    root = tmp_path / "ws"
    root.mkdir()
    (root / "fukikae.toml").write_text(
        f'source_language = "ja"\nseries = "{series}"\n', encoding="utf-8"
    )
    gt = tmp_path / "gt"
    gt.mkdir()
    rng = np.random.default_rng(4)
    ws = Workspace(root)
    for eid in episodes:
        ep = Episode(source=Path(eid), id=eid, work=ws.work_dir(eid), workspace_root=root)
        d = ep.work / "04-diarize"
        d.mkdir(parents=True)
        turns, lines = [], []
        for k in range(18):
            spk = f"SPEAKER_{k % 3:02d}"
            turns.append(Turn(start=5.0 * k, end=5.0 * k + 4.0, speaker=spk))
            lines.append(
                f"Dialogue: 0,{stamp(5.0 * k)},{stamp(5.0 * k + 4.0)},Default,{NAMES[spk]},"
                "0,0,0,,hi\n"
            )
        write_json(d / "turns.json", Diarization(backend="x", turns=turns))
        arrays: dict[str, Any] = {"labels": np.array(sorted(NAMES), dtype=str)}
        for i, spk in enumerate(sorted(NAMES)):
            spans = clean_spans(turns, spk)
            arrays[f"{spk}/embeddings"] = np.vstack([voice(rng, i) for _ in spans])
            arrays[f"{spk}/spans"] = np.array(spans, dtype=np.float64)
            arrays[f"{spk}/clean_seconds"] = np.array(sum(b - a for a, b in spans))
        np.savez(d / "clusters.npz", **arrays)
        (gt / f"Show - {eid}.ass").write_text(HEADER + "".join(lines), encoding="utf-8")
        now = datetime.now(UTC)
        save_manifest(
            ep, Manifest(stages={"diarize": StageRecord(key="k", status="done", started=now)})
        )
    return ws


def adapt(ws: Workspace, episodes: str, *extra: str):
    gt = ws.root.parent / "gt"
    return runner.invoke(
        app,
        ["registry", "adapt", str(ws.root), "--episodes", episodes, "--gt", "ass"]
        + ["--gt-file", str(gt), *extra],
    )


def test_adapt_writes_one_projection_per_series(tmp_path):
    ws = make_ws(tmp_path, ["S01E12", "S01E13"])
    r = adapt(ws, "S01E12,S01E13")
    assert r.exit_code == 0, r.output
    assert "show" in r.output and "3 labels" in r.output
    with Registry(ws.registry_path) as reg:
        p = reg.load_projection("show", "lda")
        assert p is not None and p.matrix.shape == (DIM, 2)
        first = p.created
    r = adapt(ws, "S01E12", "--dims", "1")
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        p = reg.load_projection("show", "lda")
        assert p is not None and p.matrix.shape == (DIM, 1) and p.created >= first
        rows = reg._con.execute("SELECT COUNT(*) FROM projections").fetchone()[0]
        assert rows == 1  # the previous one is replaced


def test_adapt_refuses_episodes_outside_the_workspace(tmp_path):
    ws = make_ws(tmp_path, ["S01E12"])
    elsewhere = tmp_path / "Other Show S02E01.mkv"
    elsewhere.write_bytes(b"x")
    for ref in ("S01E12,S01E99", f"S01E12,{elsewhere}"):
        r = adapt(ws, ref)
        assert r.exit_code == 1
        assert "not an episode of this workspace" in r.output
    with Registry(ws.registry_path) as reg:
        assert reg.load_projection("show", "lda") is None


def test_adapt_refuses_unlabeled_episodes(tmp_path):
    ws = make_ws(tmp_path, ["S01E12"])
    gt = tmp_path / "gt" / "Show - S01E12.ass"
    gt.write_text(HEADER, encoding="utf-8")
    r = adapt(ws, "S01E12", "--force")
    assert r.exit_code == 1
    assert "labels" in r.output


def test_adapt_refuses_episodes_of_different_dimensions(tmp_path):
    ws = make_ws(tmp_path, ["S01E12", "S01E13"])
    path = ws.work_dir("S01E13") / "04-diarize" / "clusters.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    for lb in NAMES:
        n = len(arrays[f"{lb}/embeddings"])
        arrays[f"{lb}/embeddings"] = np.ones((n, DIM + 2), dtype=np.float32) / 3
    np.savez(path, **arrays)
    r = adapt(ws, "S01E12,S01E13")
    assert r.exit_code == 1
    assert "S01E13" in r.output and f"{DIM + 2}-dimensional" in r.output
    assert "registry reembed" in r.output
