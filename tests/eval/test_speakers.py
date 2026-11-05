import wave
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from fukikae.artifacts import write_json
from fukikae.eval.cli import eval_app
from fukikae.eval.speakers import LabeledClip, load_labels, sample, score, verdict
from fukikae.models import Assignments, ClusterAssignment


def _assign(episode: str, mapping: dict[str, int | None]) -> Assignments:
    return Assignments(
        episode=episode,
        threshold=0.5,
        clusters=[
            ClusterAssignment(
                cluster=c,
                actor_id=None,
                character_id=cid,
                score=None,
                clean_seconds=30.0,
                status="matched" if cid is not None else "extra",
            )
            for c, cid in mapping.items()
        ],
    )


def _clip(ep: str, cluster: str, label: str, k: int = 0) -> LabeledClip:
    return LabeledClip(ep, cluster, f"{ep}/{cluster}-{k}.wav", 0.0, 1.0, label)


def test_score_consistent_merged_and_fraction() -> None:
    assignments = {
        "S01E01": _assign("S01E01", {"SPEAKER_00": 1, "SPEAKER_01": 2, "SPEAKER_02": 2}),
        "S01E02": _assign("S01E02", {"SPEAKER_00": 1, "SPEAKER_01": 3}),
    }
    labels = [
        _clip("S01E01", "SPEAKER_00", "detective"),
        _clip("S01E02", "SPEAKER_00", "detective"),
        _clip("S01E01", "SPEAKER_01", "profiler"),
        _clip("S01E02", "SPEAKER_01", "profiler"),
        _clip("S01E01", "SPEAKER_02", "surgeon"),
        _clip("S01E01", "SPEAKER_02", "?", 1),
        _clip("S01E01", "SPEAKER_02", "", 2),
        _clip("S01E01", "SPEAKER_02", "extra", 3),
    ]
    s = score(labels, assignments)
    assert s.recurring == 2
    assert s.consistent == 1
    assert s.fraction == 0.5
    assert s.merged_pairs == [("profiler", "surgeon")]
    assert verdict(s) == "rework"


def test_null_character_is_inconsistent_and_not_merged() -> None:
    assignments = {
        "S01E01": _assign("S01E01", {"A": None, "B": None}),
        "S01E02": _assign("S01E02", {"A": None}),
    }
    labels = [_clip("S01E01", "A", "x"), _clip("S01E02", "A", "x"), _clip("S01E01", "B", "y")]
    s = score(labels, assignments)
    assert (s.recurring, s.consistent, s.merged_pairs) == (1, 0, [])


def test_verdict_bar() -> None:
    from fukikae.eval.speakers import SpeakerScore

    assert verdict(SpeakerScore(10, 9, [], 0.9)) == "pass"
    assert verdict(SpeakerScore(10, 9, [("a", "b")], 0.9)) == "tune"
    assert verdict(SpeakerScore(4, 3, [], 0.75)) == "tune"
    assert verdict(SpeakerScore(4, 2, [], 0.5)) == "rework"


def _make_episode(root: Path, eid: str) -> Path:
    work = root / ".fukikae" / "episodes" / eid
    (work / "01-separate").mkdir(parents=True)
    (work / "04-diarize").mkdir(parents=True)
    (work / "05-characters").mkdir(parents=True)
    sr = 16000
    pcm = (np.arange(sr * 40) % 100).astype(np.int16)
    with wave.open(str(work / "01-separate" / "vocals_16k.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    spans = np.array([[1.0, 3.0], [5.0, 20.0], [22.0, 25.0], [30.0, 32.0], [33.0, 34.5]])
    np.savez(
        work / "04-diarize" / "clusters.npz",
        labels=np.array(["SPEAKER_00", "SPEAKER_01"]),
        **{
            "SPEAKER_00/embeddings": np.ones((5, 4), dtype=np.float32),
            "SPEAKER_00/spans": spans,
            "SPEAKER_00/clean_seconds": np.array(float((spans[:, 1] - spans[:, 0]).sum())),
            "SPEAKER_01/embeddings": np.zeros((0, 4), dtype=np.float32),
            "SPEAKER_01/spans": np.zeros((0, 2)),
            "SPEAKER_01/clean_seconds": np.array(0.0),
        },
    )
    write_json(work / "05-characters" / "assignments.json", _assign(eid, {"SPEAKER_00": 7}))
    return work


def _ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "fukikae.toml").write_text("")
    return root


def test_sample_exports_longest_spans_capped_and_blind(tmp_path: Path) -> None:
    root = _ws(tmp_path)
    _make_episode(root, "S01E01")
    out = tmp_path / "out"
    clips = sample(root, ["S01E01"], 3, out)
    assert [c.cluster for c in clips] == ["SPEAKER_00"] * 3
    # longest three: 15 s (capped to 8), 3 s, 2 s (the earlier of the 2 s ties); by start
    assert [round(c.end - c.start, 3) for c in clips] == [2.0, 8.0, 3.0]
    starts = [c.start for c in clips]
    assert starts == sorted(starts)
    big = next(c for c in clips if c.end - c.start == 8.0)
    assert (big.start, big.end) == (8.5, 16.5)
    for c in clips:
        with wave.open(str(out / c.clip), "rb") as w:
            assert (w.getframerate(), w.getnchannels()) == (16000, 1)
            assert w.getnframes() == round((c.end - c.start) * 16000)
    text = (out / "labels.yaml").read_text()
    assert text.lstrip().startswith("#")
    data = yaml.safe_load(text)
    assert set(data[0]) == {"episode", "cluster", "clip", "start", "end", "label"}
    assert all(d["label"] == "" for d in data)
    assert "character" not in text
    assert [c.label for c in load_labels(out / "labels.yaml")] == [""] * 3


def test_cli_sample_refuses_overwrite_then_speakers(tmp_path: Path) -> None:
    root = _ws(tmp_path)
    _make_episode(root, "S01E01")
    out = tmp_path / "out"
    r = CliRunner()
    args = ["sample", str(root), "--episodes", "S01E01", "--per-cluster", "2", "--out", str(out)]
    assert r.invoke(eval_app, args).exit_code == 0
    path = out / "labels.yaml"
    data = yaml.safe_load(path.read_text())
    data[0]["label"] = "detective"
    data[1]["label"] = "detective"
    path.write_text(yaml.safe_dump(data))
    res = r.invoke(eval_app, args)
    assert res.exit_code == 1
    assert "detective" in path.read_text()
    assert r.invoke(eval_app, [*args, "--force"]).exit_code == 0
    path.write_text(yaml.safe_dump(data))
    res = r.invoke(eval_app, ["speakers", str(root), "--labels", str(path)])
    assert res.exit_code == 0, res.output
    assert "recurring" in res.output
    assert "rework" in res.output  # no label appears in two episodes: nothing recurring


def _labels_file(tmp_path: Path, label_text: str) -> Path:
    path = tmp_path / "labels.yaml"
    path.write_text(
        "- episode: S01E01\n  cluster: SPEAKER_00\n  clip: S01E01/SPEAKER_00-0.wav\n"
        f"  start: 1.0\n  end: 2.0\n  label: {label_text}\n"
    )
    return path


def test_written_labels_are_quoted_and_quoted_question_mark_loads(tmp_path: Path) -> None:
    path = tmp_path / "labels.yaml"
    from fukikae.eval.speakers import write_labels

    write_labels(path, [_clip("S01E01", "SPEAKER_00", "")])
    assert 'label: ""' in path.read_text()
    assert load_labels(_labels_file(tmp_path, '"?"'))[0].label == "?"


def test_bare_question_mark_and_bare_bool_are_clean_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    for text in ("?", "no"):
        path = _labels_file(tmp_path, text)
        res = runner.invoke(eval_app, ["speakers", str(tmp_path), "--labels", str(path)])
        assert res.exit_code == 1, text
        assert res.output.startswith("error:") or "error:" in res.output
        assert "Traceback" not in res.output
    path = _labels_file(tmp_path, "no")
    res = runner.invoke(eval_app, ["speakers", str(tmp_path), "--labels", str(path)])
    assert "SPEAKER_00" in res.output and "quote" in res.output
    res = runner.invoke(
        eval_app,
        ["sample", str(tmp_path), "--episodes", "S01E01", "--out", str(tmp_path)],
    )
    assert res.exit_code == 1


def test_labels_compare_case_and_space_insensitively() -> None:
    assignments = {
        "S01E01": _assign("S01E01", {"A": 1, "B": 2}),
        "S01E02": _assign("S01E02", {"A": 1}),
    }
    labels = [
        _clip("S01E01", "A", "Detective"),
        _clip("S01E02", "A", "detective "),
        _clip("S01E01", "B", "Extra"),
        _clip("S01E02", "B", "EXTRA"),
    ]
    s = score(labels, assignments)
    assert (s.recurring, s.consistent, s.merged_pairs) == (1, 1, [])
    shared = {"S01E01": _assign("S01E01", {"A": 1, "B": 1})}
    s = score([_clip("S01E01", "A", "Surgeon"), _clip("S01E01", "B", "profiler")], shared)
    assert s.merged_pairs == [("Surgeon", "profiler")]


def _video_episode(tmp_path: Path, eid: str = "S01E01") -> tuple[Path, Path]:
    import subprocess

    root = _ws(tmp_path)
    work = root / ".fukikae" / "episodes" / eid
    (work / "01-separate").mkdir(parents=True)
    (work / "04-diarize").mkdir(parents=True)
    src = tmp_path / "media" / f"Show - {eid} (x).mkv"
    src.parent.mkdir()
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=640x360:r=24:d=12",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src),
        ],
        check=True,
    )  # fmt: skip
    sr = 44100
    t = np.arange(sr * 12) / sr
    pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    for name, rate, data in (
        ("vocals.wav", sr, pcm),
        ("vocals_16k.wav", 16000, pcm[:: sr // 16000][: 16000 * 12]),
    ):
        with wave.open(str(work / "01-separate" / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(data.tobytes())
    arrays = {"labels": np.array(["SPEAKER_00"]), "SPEAKER_00/spans": np.array([[2.0, 5.0]])}
    np.savez(work / "04-diarize" / "clusters.npz", **arrays)  # type: ignore[arg-type]
    return root, src


def _probe(path: Path) -> dict:
    import json
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,height:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)


@pytest.mark.ffmpeg
def test_sample_video_writes_mp4_with_vocals(tmp_path: Path) -> None:
    root, src = _video_episode(tmp_path)
    out = tmp_path / "out"
    clips = sample(root, [str(src)], 2, out, video=True)
    assert [c.clip for c in clips] == ["S01E01/SPEAKER_00-0.mp4"]
    info = _probe(out / clips[0].clip)
    assert abs(float(info["format"]["duration"]) - 3.0) <= 0.1
    kinds = sorted(s["codec_type"] for s in info["streams"])
    assert kinds == ["audio", "video"]
    assert next(s for s in info["streams"] if s["codec_type"] == "video")["height"] == 480
    assert "SPEAKER_00-0.mp4" in (out / "labels.yaml").read_text()
    assert not list((out / "S01E01").glob(".*"))


@pytest.mark.ffmpeg
def test_sample_video_missing_source_exits_1(tmp_path: Path) -> None:
    root, src = _video_episode(tmp_path)
    ref = str(src.with_name("Show - S01E01 (gone).mkv"))
    res = CliRunner().invoke(
        eval_app,
        ["sample", str(root), "--episodes", ref, "--out", str(tmp_path / "out"), "--video"],
    )
    assert res.exit_code == 1
    assert "S01E01" in res.output


@pytest.mark.ffmpeg
def test_force_removes_stale_clips_of_other_kind(tmp_path: Path) -> None:
    root, src = _video_episode(tmp_path)
    out = tmp_path / "out"
    r = CliRunner()
    base = ["sample", str(root), "--episodes", str(src), "--out", str(out)]
    assert r.invoke(eval_app, base).exit_code == 0
    assert (out / "S01E01" / "SPEAKER_00-0.wav").is_file()
    assert r.invoke(eval_app, [*base, "--video", "--force"]).exit_code == 0
    assert not list((out / "S01E01").glob("*.wav"))
    assert (out / "S01E01" / "SPEAKER_00-0.mp4").is_file()
    assert r.invoke(eval_app, [*base, "--force"]).exit_code == 0
    assert not list((out / "S01E01").glob("*.mp4"))
