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
    srt = "1\n00:00:01,000 --> 00:00:02,000\n<i>안녕</i>\n\n2\n00:00:03,000 --> 00:00:04,000\n하세요\n두 줄\n"
    assert parse_srt(srt) == ["<i>안녕</i>", "하세요 두 줄"]
