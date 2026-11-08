from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.eval import autolabel
from fukikae.eval.cli import eval_app
from fukikae.eval.groundtruth import (
    AlignmentError,
    GroundTruthLine,
    alignment,
    check_alignment,
    read_ground_truth,
)
from fukikae.eval.groundtruth.sdh_prefix import extract_tag
from fukikae.eval.speakers import load_labels
from fukikae.media import MediaError, ProbeResult, SubTrack
from fukikae.models import Turn


def _srt(*cues: tuple[str, str, str]) -> str:
    return "\n".join(f"{i}\n{a} --> {b}\n{t}\n" for i, (a, b, t) in enumerate(cues, 1))


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _speakers(lines: list[GroundTruthLine]) -> list[str]:
    return [ln.speaker for ln in lines]


def test_sdh_latin_cyrillic_hangul_brackets(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "S01E01.srt",
        _srt(
            ("00:00:01,000", "00:00:02,000", "КОСТЯ: Привет"),
            ("00:00:03,000", "00:00:04,000", "(새하) 뭐야"),
            ("00:00:05,000", "00:00:06,000", "[music]"),
            ("00:00:07,000", "00:00:08,000", "(sighs)"),
            ("00:00:09,000", "00:00:10,000", "[Jane Doe] Hello there"),
            ("00:00:11,000", "00:00:12,000", "(Relinchos)"),
            ("00:00:13,000", "00:00:14,000", "[音乐]"),
            ("00:00:15,000", "00:00:16,000", "小明：你好"),
            ("00:00:17,000", "00:00:18,000", "Note this: not a speaker"),
            ("00:00:19,000", "00:00:20,000", "(HOMBRE) (TOSE)"),
        ),
    )
    lines = read_ground_truth(p, "sdh", {})
    assert _speakers(lines) == ["КОСТЯ", "새하", "Jane Doe", "小明", "HOMBRE"]
    assert lines[0].start == 1.0 and lines[0].end == 2.0


def test_sdh_name_is_kept_as_written_then_aliased(tmp_path: Path) -> None:
    p = _write(tmp_path, "S01E01.srt", _srt(("00:00:01,000", "00:00:02,000", "КОСТЯ: Привет")))
    assert _speakers(read_ground_truth(p, "sdh", {"костя": "Константин"})) == ["Константин"]


def test_sdh_dash_split_time_is_proportional(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "S01E01.srt",
        _srt(("00:00:10,000", "00:00:14,000", "- ARISU: Go!\n- KUINA: Wait, please.")),
    )
    a, k = read_ground_truth(p, "sdh", {})
    assert (a.speaker, k.speaker) == ("ARISU", "KUINA")
    assert a.start == pytest.approx(10.0) and k.end == pytest.approx(14.0)
    assert a.end == pytest.approx(k.start)
    # "Go!" (3 characters) against "Wait, please." (12)
    assert a.end - a.start == pytest.approx(4.0 * 3 / 15)


def test_sdh_tag_carries_until_dash(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        "S01E01.srt",
        _srt(("00:00:00,000", "00:00:06,000", "ARISU: One\ntwo three\n- (KUINA) four\nfive")),
    )
    lines = read_ground_truth(p, "sdh", {})
    assert _speakers(lines) == ["ARISU", "KUINA"]
    assert lines[0].end == pytest.approx(lines[1].start)


def test_sdh_untagged_dash_does_not_inherit(tmp_path: Path) -> None:
    p = _write(tmp_path, "S01E01.srt", _srt(("00:00:00,000", "00:00:06,000", "ARISU: One\n- Two")))
    assert _speakers(read_ground_truth(p, "sdh", {})) == ["ARISU"]


def test_extract_tag_configurable_sounds() -> None:
    assert extract_tag("(ПЕТР) да") == ("ПЕТР", "да")
    assert extract_tag("(LAUGHS) ok") is None
    assert extract_tag("(LAUGHS) ok", sounds=frozenset()) == ("LAUGHS", "ok")
    assert extract_tag("(sighs)") is None and extract_tag("[music]") is None


def test_sdh_reads_ass_text(tmp_path: Path) -> None:
    ass = (
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect,"
        " Text\nDialogue: 0,0:00:40.71,0:00:44.04,Default,,0,0,0,,{\\i1}[LU GUANG] The world\\N"
        "inside a photo\n"
        "Dialogue: 0,0:00:45.00,0:00:46.00,Default,,0,0,0,,[grunts]\n"
    )
    lines = read_ground_truth(_write(tmp_path, "S01E04.ass", ass), "sdh", {})
    assert [(ln.speaker, ln.start, ln.end) for ln in lines] == [("LU GUANG", 40.71, 44.04)]


VTT = """WEBVTT

00:00:01.000 --> 00:00:02.000
<c.vtt_yellow>(LOLA) Hola</c>

00:00:03.000 --> 00:00:04.000
<c.vtt_cyan>Porque vos lo ordenasteis.</c>

00:00:05.000 --> 00:00:06.000
<c.yellow>(LOLA) Otra vez</c>

00:00:07.000 --> 00:00:08.000
<c.yellow>Sin nombre</c>

00:00:09.000 --> 00:00:10.000
<c.vtt_yellow>(LOLA) Tercera</c>

00:00:11.000 --> 00:00:12.000
Blanco sin clase

00:00:13.000 --> 00:00:14.000 L:4% A:end
<c.vtt_blue>(Relinchos)</c>

00:00:15.000 --> 00:00:16.000
<c.vtt_cyan>Y otra.</c>
"""


def test_colour_vtt_names_a_colour_by_cooccurring_tag(tmp_path: Path) -> None:
    lines = read_ground_truth(_write(tmp_path, "S04E01.vtt", VTT), "colour", {})
    assert [(ln.start, ln.speaker) for ln in lines] == [
        (1.0, "LOLA"),
        (3.0, "cyan@S04E01"),
        (5.0, "LOLA"),
        (7.0, "LOLA"),
        (9.0, "LOLA"),
        (15.0, "cyan@S04E01"),
    ]


def test_colour_below_three_tags_keeps_colour_label(tmp_path: Path) -> None:
    text = VTT.replace("(LOLA) Tercera", "Tercera")
    lines = read_ground_truth(_write(tmp_path, "S04E01.vtt", text), "colour", {})
    assert {ln.speaker for ln in lines if ln.start in (1.0, 5.0, 7.0, 9.0)} == {
        "LOLA",
        "yellow@S04E01",
    }
    assert [ln.speaker for ln in lines if ln.start in (7.0, 9.0)] == ["yellow@S04E01"] * 2


def test_colour_font_tags_srt_and_ebu_tt(tmp_path: Path) -> None:
    srt = _srt(
        ("00:00:01,000", "00:00:02,000", '<font color="#ffff00">Hallo</font>'),
        ("00:00:03,000", "00:00:04,000", '<font color="#ffffff">Weiss</font>'),
        ("00:00:05,000", "00:00:06,000", '<font color="#00ffff">- Eins</font>\n- Zwei'),
    )
    got = read_ground_truth(_write(tmp_path, "S02E03.srt", srt), "colour", {})
    assert [ln.speaker for ln in got] == ["yellow@S02E03", "cyan@S02E03"]
    tt = """<?xml version="1.0"?>
<tt:tt xmlns:tt="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
<tt:head><tt:styling>
<tt:style xml:id="textYellow" tts:color="#ffff00"/>
<tt:style xml:id="textWhite" tts:color="#ffffff"/>
<tt:style xml:id="viaColor" tts:color="#00ff00"/>
</tt:styling></tt:head>
<tt:body><tt:div>
<tt:p begin="00:00:07.320" end="00:00:10.120"><tt:span style="textYellow">als Karin</tt:span></tt:p>
<tt:p begin="00:00:11.000" end="00:00:12.000"><tt:span style="textWhite">weiss</tt:span></tt:p>
<tt:p begin="00:00:13.000" end="00:00:14.000"><tt:span style="viaColor">gruen</tt:span><tt:br/>
<tt:span style="viaColor">zwei</tt:span></tt:p>
</tt:div></tt:body></tt:tt>"""
    got = read_ground_truth(_write(tmp_path, "S01E01.xml", tt), "colour", {})
    assert [(ln.speaker, ln.start, ln.end) for ln in got] == [
        ("yellow@S01E01", 7.32, 10.12),
        ("green@S01E01", 13.0, 14.0),
    ]


def test_colour_ass_override(tmp_path: Path) -> None:
    ass = (
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect,"
        " Text\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{\\c&H00FFFF&}Hi\n"
    )
    got = read_ground_truth(_write(tmp_path, "S01E01.ass", ass), "colour", {})
    assert [ln.speaker for ln in got] == ["yellow@S01E01"]  # &H00FFFF& is BGR: yellow


def test_aliases_apply_to_every_reader(tmp_path: Path) -> None:
    got = read_ground_truth(
        _write(tmp_path, "S04E01.vtt", VTT), "colour", {"lola": "Dolores", "CYAN@S04E01": "Pepe"}
    )
    assert {ln.speaker for ln in got} == {"Dolores", "Pepe"}


def _turns(spans: list[tuple[float, float]]) -> list[Turn]:
    return [Turn(start=a, end=b, speaker="S0") for a, b in spans]


def _speech(n: int = 30) -> list[tuple[float, float]]:
    # irregular speech so only one offset fits
    out, t = [], 5.0
    for i in range(n):
        d = 1.0 + (i % 4) * 0.7
        out.append((t, t + d))
        t += d + 1.5 + (i % 3) * 1.1
    return out


def test_alignment_finds_the_offset() -> None:
    spans = _speech()
    lines = [GroundTruthLine(a + 3.0, b + 3.0, "X") for a, b in spans]
    rep = alignment(lines, _turns(spans))
    assert rep.overlap_share < 0.6
    assert rep.best_offset_s == pytest.approx(3.0, abs=0.15)
    assert rep.share_at_offset > 0.95
    same = alignment([GroundTruthLine(a, b, "X") for a, b in spans], _turns(spans))
    assert same.overlap_share == pytest.approx(1.0)
    assert same.best_offset_s == 0.0


def test_check_alignment_applies_offset_and_refuses() -> None:
    spans = _speech()
    lines = [GroundTruthLine(a + 3.0, b + 3.0, "X") for a, b in spans]
    fixed, rep, applied = check_alignment(lines, _turns(spans))
    assert applied and rep.best_offset_s == pytest.approx(3.0, abs=0.15)
    assert fixed[0].start == pytest.approx(spans[0][0], abs=0.15)
    # 25 s late is outside the +-10 s window: nothing fits
    far = [GroundTruthLine(a + 25.0, b + 25.0, "X") for a, b in spans]
    with pytest.raises(AlignmentError, match="--force"):
        check_alignment(far, _turns(spans))
    check_alignment(far, _turns(spans), force=True)  # scored anyway


def test_check_alignment_no_speech_refuses() -> None:
    with pytest.raises(AlignmentError):
        check_alignment([GroundTruthLine(1.0, 2.0, "X")], [])


# ---- auto-label wiring ----

SUBS = _srt(
    ("00:00:00,000", "00:00:04,000", "NATSU: hi"),
    ("00:00:05,000", "00:00:09,000", "LUCY: hi"),
)


def _ws(tmp_path: Path, srt: str) -> tuple[Path, Path]:
    root = tmp_path / "ws"
    work = root / ".fukikae" / "episodes" / "S01E01"
    (work / "04-diarize").mkdir(parents=True)
    np.savez(
        work / "04-diarize" / "clusters.npz",
        labels=np.array(["C0", "C1"]),
        **{"C0/spans": np.array([[0.0, 4.0]]), "C1/spans": np.array([[5.0, 9.0]])},  # type: ignore[arg-type]
    )
    (tmp_path / "media").mkdir()
    (tmp_path / "media" / "Show - S01E01.mkv").write_bytes(b"")
    sub = tmp_path / "gt.srt"
    sub.write_text(srt, encoding="utf-8")
    return root, sub


def test_auto_label_sdh_with_gt_file(tmp_path: Path) -> None:
    root, sub = _ws(tmp_path, SUBS)
    ref = str(tmp_path / "media" / "Show - S01E01.mkv")
    out = tmp_path / "out"
    res = autolabel.auto_label(root, [ref], out, source="sdh", gt_file=sub)
    assert [c.label for c in load_labels(out / "labels.yaml")] == ["NATSU", "LUCY"]
    assert res.alignment and not res.alignment[0][2]


def test_auto_label_refuses_misaligned_without_force(tmp_path: Path) -> None:
    late = SUBS.replace("00:00:00,000", "00:00:30,000").replace("00:00:04,000", "00:00:34,000")
    late = late.replace("00:00:05,000", "00:00:35,000").replace("00:00:09,000", "00:00:39,000")
    root, sub = _ws(tmp_path, late)
    ref = str(tmp_path / "media" / "Show - S01E01.mkv")
    with pytest.raises(AlignmentError):
        autolabel.auto_label(root, [ref], tmp_path / "o1", source="sdh", gt_file=sub)
    autolabel.auto_label(root, [ref], tmp_path / "o2", source="sdh", gt_file=sub, force=True)


def test_auto_label_applies_small_offset_and_reports(tmp_path: Path) -> None:
    shifted = _srt(
        ("00:00:03,000", "00:00:07,000", "NATSU: hi"),
        ("00:00:08,000", "00:00:12,000", "LUCY: hi"),
    )
    root, sub = _ws(tmp_path, shifted)
    ref = str(tmp_path / "media" / "Show - S01E01.mkv")
    out = tmp_path / "out"
    res = autolabel.auto_label(root, [ref], out, source="sdh", gt_file=sub)
    assert [c.label for c in load_labels(out / "labels.yaml")] == ["NATSU", "LUCY"]
    ep, rep, applied = res.alignment[0]
    assert applied and rep.best_offset_s == pytest.approx(3.0, abs=0.15)
    assert "offset" in autolabel.format_report(res)


def test_cli_source_and_force(tmp_path: Path) -> None:
    root, sub = _ws(tmp_path, SUBS)
    ref = str(tmp_path / "media" / "Show - S01E01.mkv")
    r = CliRunner().invoke(
        eval_app,
        ["auto-label", str(root), "--episodes", ref, "--out", str(tmp_path / "o"),
         "--source", "sdh", "--gt-file", str(sub)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "2 labeled" in r.output
    bad = CliRunner().invoke(
        eval_app,
        ["auto-label", str(root), "--episodes", ref, "--out", str(tmp_path / "o"), "--source", "x"],
    )
    assert bad.exit_code != 0


# ---- track choice ----


def _subtitle_probe() -> ProbeResult:
    return ProbeResult(
        subtitles=[
            SubTrack(2, "subrip", "eng", "English", False),
            SubTrack(3, "subrip", "kor", "Korean [CC]", False),
            SubTrack(4, "hdmv_pgs_subtitle", "kor", "Korean", False),
            SubTrack(5, "subrip", "kor", "Korean", True),
            SubTrack(6, "subrip", "rus", "Russian", False),
        ]
    )


def _fake_extract(texts: dict[int, str], monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(_source: Path, index: int, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(texts[index], encoding="utf-8")

    monkeypatch.setattr(autolabel, "_extract_text", fake)


def test_choose_text_track_prefers_marked_most_tagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tagged = _srt(("00:00:01,000", "00:00:02,000", "(새하) 뭐야"))
    two = _srt(
        ("00:00:01,000", "00:00:02,000", "(새하) 뭐야"),
        ("00:00:03,000", "00:00:04,000", "(새벽) 응"),
    )
    _fake_extract({2: two, 3: tagged, 6: two}, monkeypatch)
    pr = _subtitle_probe()
    idx, out = autolabel.choose_text_track(Path("x.mkv"), pr, tmp_path, "sdh", "ko")
    assert (idx, out) == (3, tmp_path / "eval" / "subtitles.srt")  # [CC] and Korean; PGS skipped
    # no marker: the source-language tracks compete (forced and bitmap ones excluded)
    pr = ProbeResult(subtitles=[x for x in pr.subtitles if x.index != 3])
    _fake_extract({2: tagged, 6: two}, monkeypatch)
    idx, _ = autolabel.choose_text_track(Path("x.mkv"), pr, tmp_path, "sdh", "ru")
    assert idx == 6
    assert autolabel.choose_text_track(Path("x.mkv"), pr, tmp_path, "sdh", None, override=2)[0] == 2
    _fake_extract({2: "1\n00:00:01,000 --> 00:00:02,000\nplain\n", 6: "", 5: ""}, monkeypatch)
    with pytest.raises(MediaError, match="speaker information"):
        autolabel.choose_text_track(Path("x.mkv"), pr, tmp_path, "sdh", "en")


def test_choose_text_track_real_grid_probe_all_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grid S01E02 as ffprobe reports it: every track flagged forced, titles `Korean [CC]`."""
    pr = ProbeResult(
        subtitles=[
            SubTrack(2, "subrip", "kor", "Korean [CC]", True),
            SubTrack(3, "subrip", "jpn", "Japanese", True),
            SubTrack(4, "subrip", "eng", "English [CC]", True),
            SubTrack(5, "subrip", "chi", "Chinese (Traditional)", True),
            SubTrack(6, "subrip", "chi", "Chinese (Simplified)", True),
        ]
    )
    one = _srt(("00:00:01,000", "00:00:02,000", "(새하) 뭐야"))
    two = _srt(
        ("00:00:01,000", "00:00:02,000", "(Saeha) what"),
        ("00:00:03,000", "00:00:04,000", "(Saebyeok) yes"),
    )
    plain = _srt(("00:00:01,000", "00:00:02,000", "plain"))
    _fake_extract({2: one, 3: plain, 4: two, 5: plain, 6: plain}, monkeypatch)
    # [CC] in the title and the source language together beat a longer English [CC] track
    assert autolabel.choose_text_track(Path("x.mkv"), pr, tmp_path, "sdh", "ko")[0] == 2
    # without a source language the two [CC] tracks compete; the English one tags more
    assert autolabel.choose_text_track(Path("x.mkv"), pr, tmp_path, "sdh", None)[0] == 4


# ---- fix round 1 ----


@pytest.mark.parametrize(
    "line",
    [
        "(어두운 음악)", "(살짝 웃으며) 그래", "(피식 웃으며) 응", "(놀라며) 어",
        "(새하의 한숨) 휴",
        "(휴대전화 안내 음성) 안내", "(자막) 안녕", "(함께) 네", "(어진과 새하) 네",
        "(어진와 새하) 네", "(문이 쓱 열린다)", "(타이어 마찰음) 응", "(사이렌이 울린다) 응",
        "(Sam sighs) ok", "(Victor sighs) ok", "[in English] hello", "[in Korean] hello",
        "ACENTO FRANCÉS: hola", "EN ESPAÑOL: hola", "(RADIO) hola", "(TV) hola", "(CHUCKLES) ok",
        "(小明和小红) 你好", "(音乐声) 你好", "(笑) 你好", "[Sam gasps] hey",
    ],
)  # fmt: skip
def test_non_speaker_tags_are_rejected(line: str) -> None:
    assert extract_tag(line) is None


@pytest.mark.parametrize(
    "line,name",
    [("(새하) 뭐야", "새하"), ("КОСТЯ: Привет", "КОСТЯ"), ("(LOLA) hola", "LOLA"),
     ("[Jane Doe] hi", "Jane Doe"), ("(SAEHA) hi", "SAEHA")],
)  # fmt: skip
def test_speaker_tags_still_accepted(line: str, name: str) -> None:
    assert extract_tag(line) is not None and extract_tag(line)[0] == name  # type: ignore[index]


def test_zero_lines_refuse_unless_forced(tmp_path: Path) -> None:
    with pytest.raises(AlignmentError, match="no sdh speaker lines.*S01E01"):
        check_alignment([], _turns([(0.0, 1.0)]), label="sdh speaker lines in S01E01")
    assert check_alignment([], _turns([(0.0, 1.0)]), force=True)[0] == []
    root, sub = _ws(tmp_path, _srt(("00:00:00,000", "00:00:04,000", "no tags here")))
    ref = str(tmp_path / "media" / "Show - S01E01.mkv")
    with pytest.raises(AlignmentError, match="sdh"):
        autolabel.auto_label(root, [ref], tmp_path / "o", source="sdh", gt_file=sub)
    sub.write_text("", encoding="utf-8")
    with pytest.raises(AlignmentError, match="S01E01"):
        autolabel.auto_label(root, [ref], tmp_path / "o", source="ass", gt_file=sub)


@pytest.mark.parametrize("name", ["민서", "예서", "지서", "유다", "마음", "태음", "하서"])
def test_korean_given_names_ending_in_description_characters(name: str) -> None:
    got = extract_tag(f"({name}) 응")
    assert got is not None and got[0] == name


@pytest.mark.parametrize(
    "line",
    ["(놀라며) 어", "(살짝 웃으며) 응", "(피식 웃으며) 응", "(손뼉을 짝 치며) 응", "(풀벌레 울음)"],
)
def test_korean_action_endings_rejected(line: str) -> None:
    assert extract_tag(line) is None


LRE = "‪"


def _ja(tmp_path: Path, *texts: str) -> list[GroundTruthLine]:
    cues = [
        (f"00:00:{i * 2 + 1:02d},000", f"00:00:{i * 2 + 2:02d},000", t) for i, t in enumerate(texts)
    ]
    return read_ground_truth(_write(tmp_path, "ja.srt", _srt(*cues)), "sdh", None)


def test_netflix_bidi_marks_do_not_hide_tags(tmp_path: Path) -> None:
    lines = _ja(tmp_path, f"{LRE}（始）また行ってねえのか 面接")
    assert _speakers(lines) == ["始"]


def test_halfwidth_parens_and_tag_on_its_own_line(tmp_path: Path) -> None:
    lines = _ja(tmp_path, f"{LRE}(カルベ) おい", f"{LRE}（始）\n{LRE}俺も会社に頭 下げてやったのに")
    assert _speakers(lines) == ["カルベ", "始"]
    assert (lines[1].start, lines[1].end) == (3.0, 4.0)


def test_ruby_inside_tag_and_full_name_maps_to_given_name(tmp_path: Path) -> None:
    lines = _ja(
        tmp_path,
        f"{LRE}（{LRE}有栖(ありす){LRE} {LRE}始(はじめ){LRE}）\n{LRE}ホント クズ兄貴だな",
        f"{LRE}（始）また行ってねえのか",
    )
    assert _speakers(lines) == ["始", "始"]


def test_full_name_without_short_form_is_kept(tmp_path: Path) -> None:
    lines = _ja(tmp_path, f"{LRE}（有栖{LRE}良平(りょうへい){LRE}）あっ！")
    assert _speakers(lines) == ["有栖良平"]


@pytest.mark.parametrize(
    "tag",
    [
        "（通知音）", "（合成音声）", "（荒い息）", "（ふざけ合う声）", "（女性たちの悲鳴）",
        "（男性）", "（３人）", "（アリスたち）", "（クラクション）", "（足音）",
        "（アリス･チョータ）",
    ],
)  # fmt: skip
def test_japanese_descriptions_groups_and_pairs_are_not_speakers(tag: str) -> None:
    assert extract_tag(f"{tag}text") is None
    assert extract_tag(f"{LRE}{tag}text") is None


@pytest.mark.parametrize(
    ("tag", "name"),
    [("（チョータの母）", "チョータの母"), ("（琴音）", "琴音"), ("（さとみ）", "さとみ"),
     ("（シブキ）", "シブキ")],
)  # fmt: skip
def test_japanese_names_still_accepted(tag: str, name: str) -> None:
    assert extract_tag(f"{LRE}{tag}text") == (name, "text")


def test_pair_rule_needs_two_characters_each_side() -> None:
    assert extract_tag("（アリスとカルベ）text") is None
    assert extract_tag("（さとみ）text") == ("さとみ", "text")


def test_ass_override_blocks_split_parentheses(tmp_path: Path) -> None:
    ass = (
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect,"
        " Text\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,"
        "{\\fscx50}（{\\fscx100}薫{\\fscx50}）{\\fscx100}どうして？\n"
    )
    lines = read_ground_truth(_write(tmp_path, "ja.ass", ass), "sdh", None)
    assert _speakers(lines) == ["薫"]


def test_existing_pair_rule_still_rejects_korean_and_chinese_pairs() -> None:
    assert extract_tag("(새하와 새벽) text") is None
    assert extract_tag("(张三和李四) text") is None
