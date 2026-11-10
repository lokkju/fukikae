import subprocess
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.artifacts import write_json
from fukikae.eval import autolabel
from fukikae.eval.autolabel import Line, label_spans, parse_ass
from fukikae.eval.cli import eval_app
from fukikae.eval.speakers import load_labels, score, verdict
from fukikae.media import MediaError, probe
from fukikae.models import Assignments, ClusterAssignment

HEADER = """[Script Info]
Title: x

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _dlg(start: str, end: str, name: str, text: str = "hi") -> str:
    return f"Dialogue: 0,{start},{end},Default,{name},0,0,0,,{text}\n"


def test_parse_ass_keeps_single_speaker_lines() -> None:
    text = (
        HEADER
        + _dlg("0:00:01.00", "0:00:02.50", "NATSU", "Hello, world")
        + _dlg("0:00:03.00", "0:00:04.00", "SIGN", "Fairy Tail")
        + _dlg("0:00:05.00", "0:00:06.00", "NATSU & HAPPY", "Aye")
        + _dlg("0:00:07.00", "0:00:08.00", "")
        + _dlg("0:00:09.00", "0:00:10.00", "Lucy")
        + _dlg("0:00:11.00", "0:00:12.00", "GRAY and ERZA")
    )
    assert parse_ass(text) == [
        Line(1.0, 2.5, "NATSU", "Hello, world"),
        Line(9.0, 10.0, "Lucy", "hi"),
    ]


def test_label_spans_threshold_and_spelling() -> None:
    lines = [
        Line(0.0, 5.0, "Natsu", ""),
        Line(5.0, 10.0, "LUCY", ""),
        Line(10.0, 20.0, "NATSU", ""),
    ]
    spans = [(0.0, 10.0), (4.0, 6.0), (10.0, 20.0), (30.0, 31.0)]
    # first span: 5s Natsu / 5s Lucy of 10 -> "?"; second: 1/1 -> "?"
    assert label_spans(spans, lines, 0.6) == ["?", "?", "Natsu", "?"]
    # 80% covered
    assert label_spans([(0.0, 5.0)], [Line(0.0, 4.0, "NATSU", "")], 0.6) == ["NATSU"]


@pytest.mark.ffmpeg
def test_choose_track_prefers_full_english(tmp_path: Path) -> None:
    def ass(path: Path, n: int) -> None:
        body = "".join(_dlg(f"0:00:{i:02d}.00", f"0:00:{i:02d}.50", "NATSU") for i in range(n))
        path.write_text(HEADER + body, encoding="utf-8")

    forced, full, de = tmp_path / "f.ass", tmp_path / "e.ass", tmp_path / "d.ass"
    ass(forced, 3)
    ass(full, 9)
    ass(de, 20)
    src = tmp_path / "S01E01.mkv"
    args = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64:d=12"]
    for p in (forced, full, de):
        args += ["-i", str(p)]
    args += ["-map", "0:v", "-map", "1", "-map", "2", "-map", "3", "-c:v", "libx264", "-c:s", "ass"]
    args += ["-metadata:s:s:0", "language=eng", "-disposition:s:0", "forced"]
    args += ["-metadata:s:s:1", "language=eng", "-metadata:s:s:2", "language=deu", str(src)]
    subprocess.run(args, check=True)
    pr = probe(src)
    assert [s.forced for s in pr.subtitles] == [True, False, False]
    work = tmp_path / "work"
    idx, path = autolabel.choose_subtitle_track(src, pr, work)
    assert idx == pr.subtitles[1].index
    assert path == work / "eval" / "subtitles.ass"
    assert len(parse_ass(path.read_text(encoding="utf-8"))) == 9
    idx2, _ = autolabel.choose_subtitle_track(src, pr, work, override=pr.subtitles[2].index)
    assert idx2 == pr.subtitles[2].index


def test_choose_subtitle_track_all_flagged_forced_picks_most_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fairy Tail S01E11 as ffprobe reports it: its one English ASS is flagged forced."""
    from fukikae.media import ProbeResult, SubTrack

    body = {2: 1, 3: 4}

    def fake(_source: Path, index: int, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join(
            _dlg(f"0:00:{i:02d}.00", f"0:00:{i:02d}.50", "NATSU") for i in range(body[index])
        )
        dest.write_text(HEADER + rows, encoding="utf-8")

    monkeypatch.setattr(autolabel, "_extract", fake)
    one = ProbeResult(subtitles=[SubTrack(2, "ass", "eng", "English", True)])
    idx, path = autolabel.choose_subtitle_track(Path("x.mkv"), one, tmp_path)
    assert idx == 2 and len(parse_ass(path.read_text(encoding="utf-8"))) == 1
    many = ProbeResult(
        subtitles=[
            SubTrack(2, "ass", "eng", "English", True),
            SubTrack(3, "ass", "eng", "Forced", True),
            SubTrack(4, "ass", "ger", None, True),
        ]
    )
    assert autolabel.choose_subtitle_track(Path("x.mkv"), many, tmp_path)[0] == 3
    none = ProbeResult(subtitles=[SubTrack(4, "ass", "ger", None, True)])
    with pytest.raises(MediaError, match="English ASS"):
        autolabel.choose_subtitle_track(Path("x.mkv"), none, tmp_path)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ws"
    spans = {
        "S01E01": {"SPEAKER_00": [[0.0, 4.0], [10.0, 14.0]], "SPEAKER_01": [[5.0, 9.0]]},
        "S01E02": {"SPEAKER_00": [[0.0, 4.0]], "SPEAKER_01": [[5.0, 9.0]]},
    }
    subs = {
        "S01E01": [("0:00:00.00", "0:00:04.00", "NATSU"), ("0:00:05.00", "0:00:09.00", "LUCY"),
                   ("0:00:10.00", "0:00:14.00", "NATSU")],
        "S01E02": [("0:00:00.00", "0:00:04.00", "Natsu"), ("0:00:05.00", "0:00:09.00", "Lucy")],
    }  # fmt: skip
    for eid, clusters in spans.items():
        (tmp_path / "media").mkdir(exist_ok=True)
        (tmp_path / "media" / f"Show - {eid}.mkv").write_bytes(b"")
        work = root / ".fukikae" / "episodes" / eid
        (work / "04-diarize").mkdir(parents=True)
        arrays: dict[str, np.ndarray] = {"labels": np.array(list(clusters))}
        for c, sp in clusters.items():
            arrays[f"{c}/spans"] = np.array(sp)
        np.savez(work / "04-diarize" / "clusters.npz", **arrays)  # type: ignore[arg-type]
        text = HEADER + "".join(_dlg(*row) for row in subs[eid])
        asses = work / "eval"
        asses.mkdir()
        (asses / "subtitles.ass").write_text(text, encoding="utf-8")
        a = Assignments(
            episode=eid,
            threshold=0.5,
            clusters=[
                ClusterAssignment(
                    cluster=c, actor_id=None, character_id=i + 1, score=None,
                    clean_seconds=8.0, status="matched",
                )
                for i, c in enumerate(clusters)
            ],
        )  # fmt: skip
        (work / "05-characters").mkdir()
        write_json(work / "05-characters" / "assignments.json", a)
    monkeypatch.setattr(
        autolabel, "prepare_subtitles", lambda ep, _track: ep.work / "eval/subtitles.ass"
    )
    return tmp_path / "media"


def test_end_to_end_labels_load_and_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = _setup(tmp_path, monkeypatch)
    root, out = tmp_path / "ws", tmp_path / "out"
    refs = [str(media / "Show - S01E01.mkv"), str(media / "Show - S01E02.mkv")]
    result = autolabel.auto_label(root, refs, out)
    clips = load_labels(out / "labels.yaml")
    assert [(c.episode, c.cluster, c.clip, c.label) for c in clips][:3] == [
        ("S01E01", "SPEAKER_00", "SPEAKER_00-0", "NATSU"),
        ("S01E01", "SPEAKER_00", "SPEAKER_00-1", "NATSU"),
        ("S01E01", "SPEAKER_01", "SPEAKER_01-0", "LUCY"),
    ]
    assert len(clips) == 5
    assert '"NATSU"' in (out / "labels.yaml").read_text()
    s = score(clips, {e: _a(root, e) for e in ("S01E01", "S01E02")})
    assert (s.recurring, s.consistent, verdict(s)) == (2, 2, "pass")
    report = autolabel.format_report(result)
    assert "SPEAKER_00" in report and "NATSU" in report and "purity" in report


def _a(root: Path, eid: str) -> Assignments:
    from fukikae.artifacts import read_json

    return read_json(
        root / ".fukikae/episodes" / eid / "05-characters/assignments.json", Assignments
    )


def test_cli_auto_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = _setup(tmp_path, monkeypatch)
    out = tmp_path / "out"
    r = CliRunner().invoke(
        eval_app,
        ["auto-label", str(tmp_path / "ws"), "--episodes",
         f"{media / 'Show - S01E01.mkv'},{media / 'Show - S01E02.mkv'}",
         "--out", str(out), "--report"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert (out / "labels.yaml").is_file()
    assert "purity" in r.output


def test_name_variants_collapse_and_thought_suffixes() -> None:
    text = (
        HEADER
        + _dlg("0:00:01.00", "0:00:02.00", "MAKAROV (WRAITH)")
        + _dlg("0:00:03.00", "0:00:04.00", "MAKAROV(WRAITH)")
        + _dlg("0:00:05.00", "0:00:06.00", "MAKAROV  (WRAITH)")
        + _dlg("0:00:07.00", "0:00:08.00", "MAKAROV")
        + _dlg("0:00:09.00", "0:00:10.00", "WRAITH/I")
        + _dlg("0:00:11.00", "0:00:12.00", "erza/os")
        + _dlg("0:00:13.00", "0:00:14.00", "GRAY (V.O.)")
        + _dlg("0:00:15.00", "0:00:16.00", "NATSU/HAPPY")
        + _dlg("0:00:17.00", "0:00:18.00", "NATSU/I/HAPPY")
    )
    names = [ln.name for ln in parse_ass(text)]
    assert names == [
        "MAKAROV (WRAITH)",
        "MAKAROV (WRAITH)",
        "MAKAROV (WRAITH)",
        "MAKAROV",
        "WRAITH",
        "erza",
        "GRAY",
    ]


def test_alias_applies_case_insensitively() -> None:
    text = (
        HEADER + _dlg("0:00:01.00", "0:00:02.00", "Nabaru") + _dlg("0:00:03.00", "0:00:04.00", "X")
    )
    assert [ln.name for ln in parse_ass(text, {"NABARU": "NEBARU"})] == ["NEBARU", "X"]


def _alias_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    media = _setup(tmp_path, monkeypatch)
    ass = tmp_path / "ws/.fukikae/episodes/S01E01/eval/subtitles.ass"
    ass.write_text(
        HEADER
        + _dlg("0:00:00.00", "0:00:04.00", "NABARU")
        + _dlg("0:00:05.00", "0:00:09.00", "LUCY")
        + _dlg("0:00:10.00", "0:00:14.00", "NEBARU")
        + _dlg("0:00:15.00", "0:00:16.00", "MAKAROV")
        + _dlg("0:00:17.00", "0:00:18.00", "MAKAROV (WRAITH)"),
        encoding="utf-8",
    )
    return media


def _labels(out: Path) -> list[str]:
    return [c.label for c in load_labels(out / "labels.yaml") if c.episode == "S01E01"]


def test_alias_option_and_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    media = _alias_setup(tmp_path, monkeypatch)
    refs = [str(media / "Show - S01E01.mkv")]
    root = tmp_path / "ws"
    autolabel.auto_label(root, refs, tmp_path / "o1")
    assert _labels(tmp_path / "o1") == ["NABARU", "NEBARU", "LUCY"]
    autolabel.auto_label(root, refs, tmp_path / "o2", aliases={"nabaru": "NEBARU"})
    assert _labels(tmp_path / "o2") == ["NEBARU", "NEBARU", "LUCY"]
    (root / "labels-aliases.yaml").write_text("NABARU: NEBARU\n", encoding="utf-8")
    autolabel.auto_label(root, refs, tmp_path / "o3")
    assert _labels(tmp_path / "o3") == ["NEBARU", "NEBARU", "LUCY"]


def test_report_lists_collapses_and_suggestions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = _alias_setup(tmp_path, monkeypatch)
    refs = [str(media / "Show - S01E01.mkv")]
    result = autolabel.auto_label(tmp_path / "ws", refs, tmp_path / "o")
    report = autolabel.format_report(result)
    assert "alias suggestions" in report
    assert "NABARU = NEBARU" in report or "NEBARU = NABARU" in report
    assert "MAKAROV = MAKAROV (WRAITH)" in report
    result = autolabel.auto_label(
        tmp_path / "ws", refs, tmp_path / "o", aliases={"NABARU": "NEBARU"}
    )
    report = autolabel.format_report(result)
    assert "NABARU -> NEBARU" in report


def test_auto_label_reads_the_names_stage_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fukikae.names.models import NamesResult

    media = _setup(tmp_path, monkeypatch)
    root = tmp_path / "ws"
    work = root / ".fukikae" / "episodes" / "S01E01"
    (work / "04c-names").mkdir()
    arrays: dict[str, np.ndarray] = {
        "labels": np.array(["SPEAKER_00~natsu", "SPEAKER_01"]),
        "SPEAKER_00~natsu/spans": np.array([[0.0, 4.0], [10.0, 14.0]]),
        "SPEAKER_01/spans": np.array([[5.0, 9.0]]),
    }
    np.savez(work / "04c-names" / "clusters.npz", **arrays)  # type: ignore[arg-type]
    write_json(
        work / "04c-names" / "names.json",
        NamesResult(
            episode="S01E01", used=True, verdict="full", reason=None, source=None, offset_s=0.0,
            coverage={}, lines=[], hidden=[], pieces=[], labels={}, unvoiced=[], excluded=[],
            dropped_row_seconds=0.0, suggestions=[],
        ),
    )  # fmt: skip
    autolabel.auto_label(root, [str(media / "Show - S01E01.mkv")], tmp_path / "out")
    clips = load_labels(tmp_path / "out" / "labels.yaml")
    assert {c.cluster for c in clips} == {"SPEAKER_00~natsu", "SPEAKER_01"}
    assert {c.label for c in clips if c.cluster == "SPEAKER_00~natsu"} == {"NATSU"}
