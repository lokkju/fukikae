from pathlib import Path

import pytest

from doppio.eval.asr import cer, normalize, parse_srt


def test_normalize_korean() -> None:
    assert normalize("<i>[문 닫히는 소리]</i> 안녕 하세요!", "ko") == "안녕하세요"


def test_normalize_strips_ass_music_and_parens() -> None:
    assert normalize("{\\an8}♪ la ♪ (sighs) Hello,  world.", "en") == "la hello world"


def test_cer_identical() -> None:
    assert cer("안녕하세요", "안녕하세요") == 0.0


def test_cer_one_substitution() -> None:
    assert cer("안녕하세요", "안녕하새요") == pytest.approx(0.2)


def test_parse_srt_concatenates_cues_in_order() -> None:
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\n<i>안녕</i>\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\n하세요\n두 줄\n"
    )
    assert parse_srt(srt) == ["<i>안녕</i>", "하세요 두 줄"]


def test_evaluate_reports_cached_backend_runtime_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doppio.eval.asr as mod
    import doppio.job
    from doppio.artifacts import write_json
    from doppio.config import Config
    from doppio.models import Segment, Transcript
    from doppio.workspace import Episode

    ep = Episode(tmp_path / "a.mkv", "S01E01", tmp_path / "work", tmp_path)
    path = ep.work / "eval/asr-fake/transcript.json"
    path.parent.mkdir(parents=True)
    write_json(
        path,
        Transcript(
            language="ko",
            backend="fake",
            word_timing=False,
            segments=[Segment(start=0, end=1, text="안녕하세요")],
        ),
    )
    monkeypatch.setattr(mod, "extract_subtitle_text", lambda *_: ["안녕하세요"])
    monkeypatch.setattr(doppio.job, "run_stages", lambda *_a, **_k: [])  # everything cached

    (score,) = mod.evaluate(ep, Config(), 2, ["fake"], "ko")
    assert score.runtime_s is None
    assert score.cer == 0.0

    monkeypatch.setattr(doppio.job, "run_stages", lambda *_a, **_k: ["asr"])
    (score,) = mod.evaluate(ep, Config(), 2, ["fake"], "ko")
    assert score.runtime_s is not None
