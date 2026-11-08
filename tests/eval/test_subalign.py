import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fukikae.eval.cli import eval_app
from fukikae.eval.groundtruth.base import AlignmentError, Cue, read_cues
from fukikae.eval.groundtruth.subalign import (
    MIN_ANCHORS,
    FitReport,
    TimeMap,
    check_fit,
    fit_to_reference,
    fit_to_speech,
    retime_file,
)
from fukikae.models import Turn

_WORDS = [
    "apple",
    "river",
    "stone",
    "cloud",
    "lantern",
    "violet",
    "copper",
    "meadow",
    "harbor",
    "falcon",
    "whisper",
    "ember",
    "canyon",
    "thistle",
    "marble",
    "orchard",
    "pepper",
    "glacier",
    "saddle",
    "tundra",
    "beacon",
    "willow",
    "cobalt",
    "quarry",
    "nectar",
    "pebble",
    "raven",
    "summit",
    "fennel",
    "lagoon",
    "walnut",
    "cinder",
    "prairie",
    "garnet",
]


def _texts(n: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)  # noqa: S311 - fixture texts, not secrets
    out: list[str] = []
    while len(out) < n:
        t = " ".join(rng.sample(_WORDS, 5))
        if t not in out:
            out.append(t)
    return out


def _labelled(n: int = 120) -> list[Cue]:
    return [Cue(10.0 + 7.0 * i, 12.5 + 7.0 * i, t) for i, t in enumerate(_texts(n))]


def _mapped(cues: list[Cue], scale: float, offset: float) -> list[Cue]:
    return [Cue(scale * c.start + offset, scale * c.end + offset, c.text) for c in cues]


def test_timemap_apply() -> None:
    assert TimeMap(1.5, -2.0).apply(4.0) == pytest.approx(4.0)


def test_text_fit_recovers_offset_and_drift() -> None:
    lab = _labelled()
    ref = _mapped(lab, 1.001, 12.5)
    rep = fit_to_reference(lab, ref, "en")
    assert rep.mode == "text"
    assert abs(rep.tmap.scale - 1.001) < 1e-4 and abs(rep.tmap.offset - 12.5) < 0.05
    assert rep.inliers >= 110
    assert rep.rms_s < 0.01 and rep.share is None
    check_fit(rep)


def test_text_fit_tolerates_rewrapped_lines_and_missing_cues() -> None:
    lab = _labelled(200)
    ref_full = _mapped(lab, 1.001, 12.5)
    ref: list[Cue] = []
    i = 0
    while i < len(ref_full):
        c = ref_full[i]
        if i % 10 == 0 and i + 1 < len(ref_full):  # re-wrapped: two cues joined into one
            d = ref_full[i + 1]
            ref.append(Cue(c.start, d.end, f"{c.text}\n{d.text}"))
            i += 2
            continue
        if i % 5 != 3:  # drop 20%
            ref.append(c)
        i += 1
    rep = fit_to_reference(lab, ref, "en")
    assert abs(rep.tmap.scale - 1.001) < 1e-4 and abs(rep.tmap.offset - 12.5) < 0.05
    check_fit(rep)


def test_text_fit_cjk_merges_split_lines_and_ignores_markup() -> None:
    # ASS-style labelled cues: one caption as two events at the same time, name prefix and
    # colour markers; the reference has the caption as one cue without them.
    lab: list[Cue] = []
    ref: list[Cue] = []
    for i in range(60):
        a, b = f"第{i}番目の台詞です", f"続きの言葉{i}号"
        t = 5.0 + 9.0 * i
        lab.append(Cue(t, t + 3, f"<c.yellow>（直美）{a}"))
        lab.append(Cue(t, t + 3, f"<c.yellow>{b}➡"))
        ref.append(Cue(t + 30.0, t + 33.0, f"{a}\n{b}"))
    rep = fit_to_reference(lab, ref, "ja")
    assert abs(rep.tmap.scale - 1.0) < 1e-4 and abs(rep.tmap.offset - 30.0) < 0.05
    assert rep.anchors >= 55


def test_cut_capture_refuses() -> None:
    lab = _labelled()
    ref = _mapped(lab, 1.0, 12.5)
    half = len(ref) // 2
    ref = ref[:half] + [Cue(c.start + 30.0, c.end + 30.0, c.text) for c in ref[half:]]
    rep = fit_to_reference(lab, ref, "en")
    with pytest.raises(AlignmentError, match="inlier"):
        check_fit(rep)
    check_fit(rep, force=True)


def test_too_few_anchors_refuses() -> None:
    lab = _labelled()
    ref = _mapped(lab[:10], 1.0, 4.0) + [
        Cue(2000.0 + 5 * i, 2002.0 + 5 * i, t) for i, t in enumerate(_texts(50, seed=99))
    ]
    rep = fit_to_reference(lab, ref, "en")
    assert rep.anchors < MIN_ANCHORS
    with pytest.raises(AlignmentError, match="anchor"):
        check_fit(rep)


def test_no_anchors_at_all_refuses() -> None:
    with pytest.raises(AlignmentError):
        fit_to_reference(_labelled(), _mapped(_labelled()[:1], 1.0, 0.0), "en")


def test_high_residual_refuses() -> None:
    lab = _labelled()
    ref = [
        Cue(c.start + 5.0 + (0.4 if i % 2 else -0.4), c.end + 5.0, c.text)
        for i, c in enumerate(lab)
    ]
    rep = fit_to_reference(lab, ref, "en")
    with pytest.raises(AlignmentError, match="rms"):
        check_fit(rep)


def test_speech_fit_recovers_frame_rate_drift() -> None:
    lab = [Cue(5.0 + 11.0 * i, 5.0 + 11.0 * i + 2.0 + (i % 4), "x") for i in range(80)]
    s = 25 / 23.976
    turns = [Turn(start=s * c.start + 3.0, end=s * c.end + 3.0, speaker="A") for c in lab]
    rep = fit_to_speech(lab, turns)
    assert rep.mode == "speech"
    assert rep.tmap.scale == pytest.approx(25 / 23.976)
    assert abs(rep.tmap.offset - 3.0) < 0.05
    assert rep.share is not None and rep.share > 0.95
    check_fit(rep)


def test_speech_fit_refuses_unrelated_speech() -> None:
    lab = [Cue(5.0 + 11.0 * i, 7.0 + 11.0 * i, "x") for i in range(80)]
    turns = [Turn(start=3000.0 + 20 * i, end=3001.0 + 20 * i, speaker="A") for i in range(20)]
    rep = fit_to_speech(lab, turns)
    with pytest.raises(AlignmentError, match="share"):
        check_fit(rep)
    check_fit(rep, force=True)


_ASS = (
    "﻿[Script Info]\r\nScriptType: v4.00+\r\n\r\n"
    "[V4+ Styles]\r\nFormat: Name, Fontname, Fontsize, PrimaryColour\r\n"
    "Style: Default,Yu Gothic,46,&H00FFFFFF\r\n\r\n"
    "[Events]\r\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\n"
    "Dialogue: 0,0:00:08.54,0:00:13.38,Default,Rin,0000,0000,0000,,"
    "{\\pos(140,496)\\c&H0000FF00}そうだねえ{\\fscx50}　{\\fscx100}あの２人, の話\r\n"
    "Comment: 0,0:00:20.00,0:00:21.00,Default,,0,0,0,,note\r\n"
    "Dialogue: 1,1:02:03.99,1:02:06.00,Default,,0000,0000,0000,,last\r\n"
)


def test_retime_ass_keeps_everything_but_times(tmp_path: Path) -> None:
    src = tmp_path / "in.ass"
    src.write_bytes(_ASS.encode("utf-8"))
    dst = tmp_path / "out.ass"
    assert retime_file(src, dst, TimeMap(1.001, 2.0)) == (2, 0)
    want = _ASS.replace("0:00:08.54,0:00:13.38", "0:00:10.55,0:00:15.39").replace(
        "1:02:03.99,1:02:06.00", "1:02:09.71,1:02:11.73"
    )
    assert dst.read_bytes() == want.encode("utf-8")


def test_retime_srt_and_clamps_negative(tmp_path: Path) -> None:
    text = (
        "1\n00:00:02,000 --> 00:00:04,500\n(Rin) hello\n\n"
        '2\n00:00:10,250 --> 00:00:12,000\n<font color="#ffff00">there</font>\n'
    )
    src = tmp_path / "in.srt"
    src.write_text(text, encoding="utf-8")
    dst = tmp_path / "out.srt"
    assert retime_file(src, dst, TimeMap(1.0, -5.0)) == (2, 1)
    assert dst.read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:00,000\n(Rin) hello\n\n"
        '2\n00:00:05,250 --> 00:00:07,000\n<font color="#ffff00">there</font>\n'
    )


def test_retime_vtt_keeps_settings(tmp_path: Path) -> None:
    text = "WEBVTT\n\n00:01.000 --> 00:03.000 align:start line:90%\nhi\n"
    src = tmp_path / "in.vtt"
    src.write_text(text, encoding="utf-8")
    dst = tmp_path / "out.vtt"
    assert retime_file(src, dst, TimeMap(1.0, 1.5)) == (1, 0)
    assert dst.read_text(encoding="utf-8") == (
        "WEBVTT\n\n00:02.500 --> 00:04.500 align:start line:90%\nhi\n"
    )


def _srt_text(cues: list[Cue]) -> str:
    def st(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    return "\n".join(
        f"{i}\n{st(c.start)} --> {st(c.end)}\n{c.text}\n" for i, c in enumerate(cues, 1)
    )


def test_cli_writes_retimed_file_and_refuses_bad_fit(tmp_path: Path) -> None:
    lab = _labelled()
    lab_p = tmp_path / "S01E01.srt"
    lab_p.write_text(_srt_text(lab), encoding="utf-8")
    good = tmp_path / "ref.srt"
    good.write_text(_srt_text(_mapped(lab, 1.001, 12.5)), encoding="utf-8")
    out = tmp_path / "aligned" / "S01E01.srt"
    runner = CliRunner()
    r = runner.invoke(
        eval_app,
        ["subalign", str(lab_p), "--reference", str(good), "--out", str(out), "--language", "en"],
    )
    assert r.exit_code == 0, r.output
    assert "text" in r.output and "1.0010" in r.output and "+12.50" in r.output
    got = read_cues(out)
    assert got[0].start == pytest.approx(1.001 * lab[0].start + 12.5, abs=0.002)
    assert got[0].text == lab[0].text

    half = len(lab) // 2
    cut = _mapped(lab, 1.0, 12.5)
    cut = cut[:half] + [Cue(c.start + 30.0, c.end + 30.0, c.text) for c in cut[half:]]
    bad = tmp_path / "cut.srt"
    bad.write_text(_srt_text(cut), encoding="utf-8")
    out2 = tmp_path / "aligned" / "S01E02.srt"
    args = ["subalign", str(lab_p), "--reference", str(bad), "--out", str(out2)]
    r = runner.invoke(eval_app, args)
    assert r.exit_code == 1
    assert "inlier" in r.output
    assert not out2.exists()
    r = runner.invoke(eval_app, [*args, "--force"])
    assert r.exit_code == 0, r.output
    assert out2.is_file()


def test_cli_speech_mode(tmp_path: Path) -> None:
    from fukikae.artifacts import write_json
    from fukikae.models import Diarization

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "fukikae.toml").write_text("", encoding="utf-8")
    lab = [Cue(5.0 + 11.0 * i, 8.0 + 11.0 * i, f"line {i}") for i in range(60)]
    turns = [Turn(start=c.start + 4.0, end=c.end + 4.0, speaker="A") for c in lab]
    d = ws / ".fukikae" / "episodes" / "S01E03" / "04-diarize"
    d.mkdir(parents=True)
    write_json(d / "turns.json", Diarization(backend="test", turns=turns))
    lab_p = tmp_path / "S01E03.srt"
    lab_p.write_text(_srt_text(lab), encoding="utf-8")
    out = tmp_path / "out.srt"
    r = CliRunner().invoke(
        eval_app,
        ["subalign", str(lab_p), "--speech", str(ws), "--episode", "s1e3", "--out", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert "speech" in r.output and "share" in r.output
    assert read_cues(out)[0].start == pytest.approx(9.0, abs=0.03)


def test_cli_needs_exactly_one_source(tmp_path: Path) -> None:
    lab_p = tmp_path / "S01E01.srt"
    lab_p.write_text(_srt_text(_labelled(5)), encoding="utf-8")
    r = CliRunner().invoke(eval_app, ["subalign", str(lab_p), "--out", str(tmp_path / "o.srt")])
    assert r.exit_code == 2


def test_fit_report_fields() -> None:
    rep = FitReport(TimeMap(1.0, 0.0), "text", 30, 29, 0.01, None)
    assert rep.tmap.apply(3.0) == 3.0
