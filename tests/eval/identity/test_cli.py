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
    assert data["settings"]["embedder"] == {
        "backend": "wespeaker",
        "options": {},
    }
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


def test_identity_cli_matcher_options_and_joint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fukikae.eval.identity import cache as mod

    root = setup_ws(tmp_path, ["S01E01", "S01E02", "S01E03"])
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    out = tmp_path / "r.json"
    base = ["eval", "identity", str(root), "--tune", "S01E01,S01E02", "--gt-file",
            str(tmp_path / "gt")]  # fmt: skip
    r = CliRunner().invoke(app, [*base, "--matcher", "knn", "--matcher-option", "k=3",
                                 "--json", str(out)])  # fmt: skip
    assert r.exit_code == 0, r.output
    s = json.loads(out.read_text())["settings"]
    assert s["matcher"] == "knn" and s["matcher_options"] == {"k": 3}
    r = CliRunner().invoke(app, [*base, "--matcher", "joint", "--threshold", "0.5",
                                 "--json", str(out)])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text())["tune"]["score"]["verdict"] == "pass"
    r = CliRunner().invoke(app, [*base, "--matcher", "joint", "--holdout", "S01E03"])
    assert r.exit_code == 1 and "joint" in r.output
    r = CliRunner().invoke(app, [*base, "--matcher", "knn", "--matcher-option", "bogus=1"])
    assert r.exit_code == 1 and "bogus" in r.output


def test_identity_cli_asr_segmentation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fukikae.eval.identity import cache as mod
    from fukikae.models import Segment as AsrSegment
    from fukikae.models import Transcript, Word

    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    for ep in ["S01E01", "S01E02"]:
        work = root / ".fukikae" / "episodes" / ep
        segs = []
        for k in range(12):
            start = 5.0 * k
            words = [Word(start=start + 0.5 * j, end=start + 0.5 * j + 0.4, text=f"w{j}.")
                     if j == 7 else Word(start=start + 0.5 * j, end=start + 0.5 * j + 0.4,
                                         text=f"w{j}") for j in range(8)]  # fmt: skip
            segs.append(AsrSegment(start=start, end=start + 4.0, text="", words=words))
        (work / "03-asr").mkdir()
        write_json(
            work / "03-asr" / "transcript.json",
            Transcript(language="ja", backend="x", word_timing=True, segments=segs),
        )
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    out = tmp_path / "r.json"
    r = CliRunner().invoke(
        app,
        ["eval", "identity", str(root), "--tune", "S01E01,S01E02", "--gt-file",
         str(tmp_path / "gt"), "--segmentation", "asr", "--segment-embedder", "wespeaker",
         "--segmentation-option", "window=1.5", "--json", str(out)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    assert data["settings"]["segmentation"] == "asr"
    assert data["settings"]["segment_embedder"] == {"backend": "wespeaker", "options": {}}
    assert data["settings"]["segmentation_options"] == {"window": 1.5}
    assert data["tune"]["score"]["verdict"] == "pass"
    assert data["episodes"][0]["segments"] == 12


def write_assignments(root: Path, ep: str, chars: dict[str, int | None]) -> None:
    from fukikae.models import Assignments, ClusterAssignment

    path = root / ".fukikae" / "episodes" / ep / "05-characters" / "assignments.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        Assignments(
            episode=ep,
            threshold=0.6,
            clusters=[
                ClusterAssignment(
                    cluster=c,
                    actor_id=cid,
                    character_id=cid,
                    score=None,
                    clean_seconds=20.0,
                    status="matched" if cid else "extra",
                )
                for c, cid in chars.items()
            ],
        ),
    )


def test_identity_cli_scores_the_pipelines_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fukikae.eval.identity import cache as mod

    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    write_assignments(root, "S01E01", {"SPEAKER_00": 1, "SPEAKER_01": 2})
    write_assignments(root, "S01E02", {"SPEAKER_00": 1, "SPEAKER_01": 1})
    out = tmp_path / "r.json"
    base = ["eval", "identity", str(root), "--tune", "S01E01,S01E02", "--gt-file",
            str(tmp_path / "gt"), "--from-assignments", "--json", str(out)]  # fmt: skip
    r = CliRunner().invoke(app, base)
    assert r.exit_code == 0, r.output
    data = json.loads(out.read_text())
    assert data["settings"]["matcher"] == "assignments"
    assert [m[:2] for m in data["tune"]["score"]["merges"]] == [["LAXUS", "LUCY"]]
    write_assignments(root, "S01E02", {"SPEAKER_00": 1, "SPEAKER_01": 2})
    r = CliRunner().invoke(app, base)
    assert r.exit_code == 0, r.output
    assert json.loads(out.read_text())["tune"]["score"]["verdict"] == "pass"
    r = CliRunner().invoke(app, [*base, "--segmentation", "subtitles"])
    assert r.exit_code == 1 and "diarization" in r.output


# ---- ledger rows ----


def char(name: str, passing: bool, recurring: bool = True):
    from fukikae.eval.identity.metrics import CharacterScore

    return CharacterScore(
        name=name, speech=10.0, by_episode={}, modal_id=1, share=0.9 if passing else 0.1,
        share_by_episode={}, ids={1: 10.0}, recurring=recurring, passing=passing,
        labeled_everywhere=True, strict=passing,
    )  # fmt: skip


def fake_score(chars, merges, fraction: float):
    from fukikae.eval.identity.metrics import IdentityScore

    return IdentityScore(
        recurring=sum(c.recurring for c in chars), passing=sum(c.passing for c in chars),
        fraction=fraction, merges=merges, strict_fraction=0.5, purity_by_id={}, coverage=0.4,
        verdict="fail", labeled_recurring=3, weighted_share=0.7, episodes=[], characters=chars,
    )  # fmt: skip


def fake_run(root: Path, holdout: bool = True):
    from fukikae.eval.identity.cli import IdentityRun

    tune = fake_score(
        [char("A", True), char("B", True), char("C", True), char("D", False)],
        [("B", "C", 3)],
        0.75,
    )
    hold = fake_score([char("A", True), char("B", False)], [], 0.5)
    settings = {
        "workspace": str(root), "tune": ["S01E01", "S01E02"],
        "holdout": "S01E03" if holdout else None, "gt": "ass", "segmentation": "diarization",
        "embedder": {"backend": "wespeaker", "options": {}}, "matcher": "centroid",
        "matcher_options": {}, "threshold": 0.6, "min_new_seconds": 20.0, "min_speech": 5.0,
        "profile": "live-action",
    }  # fmt: skip
    return IdentityRun(
        settings, [], tune, hold if holdout else None, "S01E03" if holdout else None, 0, 0,
        {"A": 1.0, "(overall)": 0.8}, {"(overall)": 0.6} if holdout else {},
    )  # fmt: skip


def test_clean_count_excludes_merged_and_failing():
    from fukikae.eval.identity.metrics import clean_count

    s = fake_score(
        [
            char("A", True),
            char("B", True),
            char("C", True),
            char("D", False),
            char("E", False, False),
        ],
        [("B", "C", 3)],
        0.75,
    )
    assert clean_count(s) == 1


def test_identity_rows_tune_and_holdout(tmp_path: Path):
    from fukikae.config import load_config
    from fukikae.eval.identity.cli import identity_rows
    from fukikae.eval.identity.metrics import clean_count
    from fukikae.eval.ledger import options_hash

    root = tmp_path / "bbt-en"
    root.mkdir()
    (root / "fukikae.toml").write_text(
        'profile = "live-action"\nsource_language = "en"\n[series]\ntitle = "BBT"\n'
    )
    cfg = load_config(root, user_config=None)
    run = fake_run(root)
    rows = identity_rows(run, cfg, "bbt", "pick")
    assert [r.split for r in rows] == ["tune", "holdout"]
    assert [r.episodes for r in rows] == [["S01E01", "S01E02"], ["S01E03"]]
    t, h = rows
    assert t.stage == "identity" and t.backend == "wespeaker"
    assert t.model == "pyannote/wespeaker-voxceleb-resnet34-LM"
    assert (t.language, t.genre, t.series, t.test_set, t.tag) == (
        "en", "live-action", "BBT", "bbt", "pick",
    )  # fmt: skip
    assert t.workspace == str(root)
    assert t.metrics["clean"] == clean_count(run.tune) == 1
    assert t.metrics == {
        "fraction": 0.75, "passing": 3, "recurring": 4, "clean": 1, "merges": 1,
        "strict_fraction": 0.5, "weighted_share": 0.7, "coverage": 0.4, "verdict": "fail",
        "nc_overall": 0.8,
    }  # fmt: skip
    assert {k: v for k, v in t.detail.items() if k != "characters"} == {
        "merges": [["B", "C", 3]], "nc": {"A": 1.0, "(overall)": 0.8},
    }  # fmt: skip
    # each recurring character's share on its modal id (the results docs' per-character tables)
    assert t.detail["characters"] == [
        {"name": c.name, "modal_id": 1, "share": c.share, "passing": c.passing}
        for c in run.tune.characters
        if c.recurring
    ]
    assert h.metrics["fraction"] == 0.5 and h.metrics["nc_overall"] == 0.6
    assert h.metrics["clean"] == 1 and h.metrics["merges"] == 0
    assert "workspace" not in t.options and "tune" not in t.options
    assert "holdout" not in t.options
    assert t.options["profile"] == "live-action"
    assert t.options["profile_hash"] == options_hash(cfg.profile_raw)
    # The diarize stage's resolved config: the live-action profile clusters at 0.5.
    assert t.options["diarize"]["backend"] == "pyannote"
    assert t.options["diarize"]["options"]["clustering_threshold"] == 0.5
    assert set(t.options["diarize"]["identity"]) == {"model", "revision"}
    # Tune and holdout of one run share the options hash, so they sit in one matrix row.
    assert t.options_hash == h.options_hash
    assert identity_rows(fake_run(root, holdout=False), cfg, "bbt", None)[0].split == "tune"
    assert len(identity_rows(fake_run(root, holdout=False), cfg, "bbt", None)) == 1


def test_identity_rows_hash_follows_the_workspace_diarize_override(tmp_path: Path):
    from fukikae.config import load_config
    from fukikae.eval.identity.cli import identity_rows

    base = 'profile = "live-action"\nsource_language = "en"\n[series]\ntitle = "BBT"\n'
    hashes = []
    for name, extra in (("a", ""), ("b", "[diarize]\nclustering_threshold = 0.6\n")):
        root = tmp_path / name
        root.mkdir()
        (root / "fukikae.toml").write_text(base + extra)
        cfg = load_config(root, user_config=None)
        (row, _) = identity_rows(fake_run(root), cfg, "bbt", None)
        hashes.append(row.options_hash)
        assert row.options["diarize"]["options"]["clustering_threshold"] == (0.6 if extra else 0.5)
    assert hashes[0] != hashes[1]


def test_identity_rows_without_nearest_centroid(tmp_path: Path):
    import dataclasses

    from fukikae.config import load_config
    from fukikae.eval.identity.cli import identity_rows

    cfg = load_config(tmp_path, user_config=None)
    run = dataclasses.replace(fake_run(tmp_path, holdout=False), tune_nc={})
    (row,) = identity_rows(run, cfg, "x", None)
    assert "nc_overall" not in row.metrics and row.detail["nc"] == {}
    assert row.language == "und"


def test_identity_cli_appends_unless_no_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fukikae.eval.identity import cli as mod
    from fukikae.eval.ledger import read_rows

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "aib"
    root.mkdir()
    (root / "fukikae.toml").write_text('source_language = "ja"\n')
    monkeypatch.setattr(mod, "run_identity", lambda *_a, **_k: fake_run(root))
    ledger = tmp_path / "results.jsonl"
    base = ["eval", "identity", str(root), "--tune", "S01E01,S01E02", "--holdout", "S01E03"]
    r = CliRunner().invoke(app, [*base, "--ledger", str(ledger)])
    assert r.exit_code == 0, r.output
    rows, warnings = read_rows(ledger)
    assert warnings == [] and len(rows) == 2
    assert {x.test_set for x in rows} == {"aib"} and {x.tag for x in rows} == {None}
    assert {x.genre for x in rows} == {"default"}
    r = CliRunner().invoke(app, [*base, "--ledger", str(ledger), "--no-ledger"])
    assert r.exit_code == 0, r.output
    assert len(read_rows(ledger)[0]) == 2
    r = CliRunner().invoke(app, [*base, "--set", "kaze", "--tag", "t1"])
    assert r.exit_code == 0, r.output
    rows, _ = read_rows()  # FUKIKAE_RESULTS from the autouse fixture
    assert [(x.test_set, x.tag) for x in rows] == [("kaze", "t1"), ("kaze", "t1")]


def test_identity_cli_ledger_failure_warns_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fukikae.eval.identity import cli as mod

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    root = tmp_path / "aib"
    root.mkdir()
    monkeypatch.setattr(mod, "run_identity", lambda *_a, **_k: fake_run(root))
    blocker = tmp_path / "file"
    blocker.write_text("x")
    r = CliRunner().invoke(
        app,
        [
            "eval",
            "identity",
            str(root),
            "--tune",
            "S01E01,S01E02",
            "--ledger",
            str(blocker / "r.jsonl"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "fraction: 0.75" in r.stdout
    assert "warning: ledger not written:" in r.stderr


def _names_json(root: Path, ep: str, verdict: str) -> None:
    from fukikae.names.models import NamesResult

    work = root / ".fukikae" / "episodes" / ep / "04c-names"
    work.mkdir(parents=True)
    write_json(
        work / "names.json",
        NamesResult(
            episode=ep,
            used=verdict in ("full", "partial"),
            verdict=verdict,  # type: ignore[arg-type]
            reason=None,
            source=None,
            offset_s=0.0,
            coverage={},
            lines=[],
            hidden=[],
            pieces=[],
            labels={},
            unvoiced=[],
            excluded=[],
            dropped_row_seconds=0.0,
            suggestions=[],
        ),
    )


@pytest.mark.parametrize("verdict", ["full", "partial", "none"])
def test_identity_refuses_a_holdout_whose_names_were_read(tmp_path: Path, verdict: str):
    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    _names_json(root, "S01E02", verdict)
    r = CliRunner().invoke(
        app,
        ["eval", "identity", str(root), "--tune", "S01E01", "--holdout", "S01E02",
         "--gt-file", str(tmp_path / "gt")],
    )  # fmt: skip
    assert r.exit_code != 0
    read = "its subtitles (verdict none)" if verdict == "none" else "its subtitle names"
    assert f"held-out episode S01E02 read {read}" in r.output


@pytest.mark.parametrize("verdict", ["holdout", "off"])
def test_check_held_out_accepts_unread_names(tmp_path: Path, verdict: str):
    from fukikae.names.inputs import check_held_out

    root = setup_ws(tmp_path, ["S01E02"])
    _names_json(root, "S01E02", verdict)
    check_held_out(root / ".fukikae" / "episodes" / "S01E02", "S01E02")
    check_held_out(root / ".fukikae" / "episodes" / "S01E09", "S01E09")  # no names.json


def test_from_assignments_scores_sub_cluster_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """S01E02's assignments label the names stage's sub-cluster SPEAKER_00~lucy: the harness
    segments the turns the assignments recorded, so its speech is scored, not left unassigned."""
    from fukikae.artifacts import read_json
    from fukikae.eval.identity import cache as mod
    from fukikae.models import Assignments

    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    write_assignments(root, "S01E01", {"SPEAKER_00": 1, "SPEAKER_01": 2})
    write_assignments(root, "S01E02", {"SPEAKER_00~lucy": 1, "SPEAKER_01": 2})
    work = root / ".fukikae" / "episodes" / "S01E02"
    turns = read_json(work / "04-diarize" / "turns.json", Diarization).turns
    path = work / "05-characters" / "assignments.json"
    a = read_json(path, Assignments)
    sub = [
        t.model_copy(update={"speaker": "SPEAKER_00~lucy"}) if t.speaker == "SPEAKER_00" else t
        for t in turns
    ]
    write_json(path, a.model_copy(update={"turns": sub}))
    out = tmp_path / "r.json"
    r = CliRunner().invoke(
        app,
        ["eval", "identity", str(root), "--tune", "S01E01,S01E02", "--gt-file",
         str(tmp_path / "gt"), "--from-assignments", "--json", str(out)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    score = json.loads(out.read_text())["tune"]["score"]
    assert score["verdict"] == "pass" and score["merges"] == []


def test_identity_refuses_a_holdout_whose_assignments_read_names(tmp_path: Path):
    """The reviewer's reproduction: names.json says holdout, but the assignments graded come
    from a characters run that read the names (a sub-cluster with a key)."""
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments

    root = setup_ws(tmp_path, ["S01E01", "S01E02"])
    write_assignments(root, "S01E01", {"SPEAKER_00": 1, "SPEAKER_01": 2})
    write_assignments(root, "S01E02", {"SPEAKER_00~lucy": 1, "SPEAKER_01": 2})
    path = root / ".fukikae" / "episodes" / "S01E02" / "05-characters" / "assignments.json"
    asg = read_json(path, Assignments)
    asg.clusters[0] = asg.clusters[0].model_copy(update={"name": "lucy", "source": "name"})
    write_json(path, asg)
    _names_json(root, "S01E02", "holdout")
    r = CliRunner().invoke(
        app,
        ["eval", "identity", str(root), "--tune", "S01E01", "--holdout", "S01E02",
         "--from-assignments", "--gt-file", str(tmp_path / "gt"), "--no-ledger"],
    )  # fmt: skip
    assert r.exit_code != 0
    assert "held-out episode S01E02's assignments hold name decisions" in r.output


def test_identity_ceiling_scores_a_names_read_episode_without_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """B2: S01E02 read its names (verdict full), so --holdout refuses it; --ceiling scores it
    the same way, against the tuning episodes' registry and ids, in a ledger row with split
    ceiling, labelled in its detail and in the printed title."""
    from fukikae.eval.identity import cache as mod

    root = setup_ws(tmp_path, ["S01E01", "S01E02", "S01E03"])
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    for ep in ("S01E01", "S01E02", "S01E03"):
        write_assignments(root, ep, {"SPEAKER_00": 1, "SPEAKER_01": 2})
    for ep in ("S01E02", "S01E03"):
        _names_json(root, ep, "full")
        work = root / ".fukikae" / "episodes" / ep
        (work / "04c-names" / "turns.json").write_bytes(
            (work / "04-diarize" / "turns.json").read_bytes()
        )
    base = ["eval", "identity", str(root), "--tune", "S01E01", "--gt-file", str(tmp_path / "gt"),
            "--from-assignments"]  # fmt: skip
    r = CliRunner().invoke(app, [*base, "--holdout", "S01E02", "--no-ledger"])
    assert r.exit_code != 0 and "read its subtitle names" in r.output
    ledger = tmp_path / "results.jsonl"
    out = tmp_path / "r.json"
    r = CliRunner().invoke(
        app, [*base, "--ceiling", "S01E02,S01E03", "--ledger", str(ledger), "--json", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert "ceiling: S01E02 (graded on its own names" in r.output
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert [(x["split"], x["episodes"], x["detail"].get("ceiling")) for x in rows] == [
        ("tune", ["S01E01"], None), ("ceiling", ["S01E02"], True), ("ceiling", ["S01E03"], True)
    ]  # fmt: skip
    assert rows[1]["options"] == rows[0]["options"]  # one matrix configuration
    got = json.loads(out.read_text())["ceiling"]
    assert [c["episode"] for c in got] == ["S01E02", "S01E03"]
    assert got[0]["score"]["verdict"] == "pass"


def test_identity_ceiling_episodes_resolve_against_tune_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With a matcher, each ceiling episode is resolved against the registry the tuning
    episodes built, never against an earlier ceiling episode."""
    from fukikae.eval.identity import cache as mod

    root = setup_ws(tmp_path, ["S01E01", "S01E02", "S01E03"])
    monkeypatch.setattr(mod, "ResidentEmbedder", FakeResident)
    from fukikae.eval.identity import simulate as sim

    seen: list[tuple[list[str], object]] = []
    orig = sim.simulate

    def spy(order, matcher, segs, embs, state=None):
        seen.append((list(order), state))
        return orig(order, matcher, segs, embs, state=state)

    monkeypatch.setattr(sim, "simulate", spy)
    r = CliRunner().invoke(app, ["eval", "identity", str(root), "--tune", "S01E01", "--ceiling",
                                 "S01E02,S01E03", "--gt-file", str(tmp_path / "gt"),
                                 "--no-ledger"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert [o for o, _ in seen] == [["S01E01"], ["S01E02"], ["S01E03"]]
    tune_state, one, two = (st for _, st in seen)
    assert len({id(tune_state), id(one), id(two)}) == 3  # a copy of the tune state each
