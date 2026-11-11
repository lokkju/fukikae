import numpy as np
import pytest

from fukikae.hallucinations import (
    HALLUCINATION_ACTIVE_MAX,
    HALLUCINATION_MUSIC_SHARE,
    HALLUCINATION_PHRASES,
    drop_reason,
    fold,
    is_stock_phrase,
    phrase_table_digest,
    rule_constants,
    segment_span,
)
from fukikae.models import Segment, Word
from fukikae.songs import SongSpan

DUR = 60  # seconds
STOCK = "ご視聴ありがとうございました"


def _seg(start: float, end: float, text: str = STOCK, words: bool = False) -> Segment:
    ws = None
    if words:
        step = (end - start) / len(text)
        ws = [
            Word(start=start + i * step, end=start + i * step + 0.02, text=ch)
            for i, ch in enumerate(text)
        ]
    return Segment(start=start, end=end, text=text, words=ws)


def _quiet_mask() -> np.ndarray:
    return np.zeros(DUR, dtype=bool)  # no second song-like


def _active(share: float) -> np.ndarray:
    """Per 100 ms frame: vocals on for ``share`` of every second."""
    per = np.zeros(10, dtype=bool)
    per[: round(share * 10)] = True
    return np.tile(per, DUR)


def test_stock_phrase_folding():
    assert is_stock_phrase("ご視聴ありがとうございました。", "ja")
    assert is_stock_phrase("ご視聴ありがとうございました ご視聴ありがとうございました", "ja")
    assert is_stock_phrase(" Thanks for watching! ", "en")
    assert is_stock_phrase("THANK YOU FOR WATCHING.", "en-US")
    assert not is_stock_phrase("ありがとうございました", "ja")
    assert not is_stock_phrase("ご視聴ありがとうございました、また来週", "ja")
    assert not is_stock_phrase(STOCK, "en")  # the transcript's language only
    assert not is_stock_phrase("", "ja") and not is_stock_phrase("。", "ja")


def test_fold_strips_spaces_punctuation_and_symbols():
    assert fold("ＡＢＣ　ｄｅｆ！♪") == "abcdef"
    assert fold("Subtitles by the Amara.org community") == "subtitlesbytheamaraorgcommunity"


def test_every_language_has_phrases_that_fold_to_something():
    for lang, phrases in HALLUCINATION_PHRASES.items():
        assert phrases, lang
        assert all(fold(p) for p in phrases), lang
    assert {"ja", "en", "zh", "ko", "de", "es"} <= set(HALLUCINATION_PHRASES)


def test_phrase_over_a_theme_song_is_dropped():
    spans = [SongSpan(3.0, 40.0, "detector", 0.9)]
    reason = drop_reason(_seg(10, 30), "ja", spans, _quiet_mask(), _active(0.9))
    assert reason == "stock phrase over a theme song"


def test_phrase_over_music_is_dropped():
    mask = _quiet_mask()
    mask[10:30] = True
    reason = drop_reason(_seg(10, 30), "ja", [], mask, _active(0.9))
    assert reason == "stock phrase over music"


def test_phrase_over_silence_is_dropped():
    reason = drop_reason(_seg(10, 30), "ja", [], _quiet_mask(), _active(0.1))
    assert reason == "stock phrase over no speech"


def test_phrase_over_dialogue_is_kept():  # a character says it to camera, no bed
    seg = _seg(10, 14)
    assert drop_reason(seg, "ja", [], _quiet_mask(), _active(0.9)) is None


def test_other_text_over_music_is_kept():
    mask = _quiet_mask()
    mask[:] = True
    seg = _seg(10, 30, text="You're not alone")
    assert drop_reason(seg, "en", [SongSpan(0, 60, "detector")], mask, _active(0.1)) is None


def test_thresholds_are_the_edges():
    mask = _quiet_mask()
    mask[10 : 10 + round(10 * HALLUCINATION_MUSIC_SHARE)] = True  # exactly the share
    assert drop_reason(_seg(10, 20), "ja", [], mask, _active(0.9)) == "stock phrase over music"
    mask[10] = False
    assert drop_reason(_seg(10, 20), "ja", [], mask, _active(0.9)) is None
    at_max = _active(HALLUCINATION_ACTIVE_MAX)
    assert drop_reason(_seg(10, 20), "ja", [], _quiet_mask(), at_max) is None


def test_a_span_mostly_outside_the_theme_is_not_inside_it():
    spans = [SongSpan(0.0, 12.0, "detector")]
    assert drop_reason(_seg(10, 30), "ja", spans, _quiet_mask(), _active(0.9)) is None


def test_span_is_the_words_extent():
    """Whisper stretches a hallucinated segment over the whole stretch (Sol Bianca's
    「お疲れ様でした」 is 3517-3774 s with its words in 3564-3684 s); the words become cues."""
    seg = Segment(
        start=0.0,
        end=50.0,
        text="お疲れ様でした",
        words=[Word(start=20.0, end=20.02, text="お"), Word(start=29.0, end=29.02, text="た")],
    )
    assert segment_span(seg) == (20.0, 29.02)
    mask = _quiet_mask()
    mask[20:30] = True  # music only where the words are
    assert drop_reason(seg, "ja", [], mask, _active(0.9)) == "stock phrase over music"
    assert segment_span(_seg(1, 2)) == (1, 2)


def test_no_evidence_past_the_arrays_keeps_the_segment():
    seg = _seg(100, 110)  # past the 60 s of features
    assert drop_reason(seg, "ja", [], _quiet_mask(), _active(0.1)) is None


def test_rule_constants_follow_the_table(monkeypatch):
    from fukikae import hallucinations

    before = rule_constants()
    assert before["phrases"] == phrase_table_digest()
    monkeypatch.setitem(hallucinations.HALLUCINATION_PHRASES, "ja", (STOCK,))
    assert rule_constants()["phrases"] != before["phrases"]


@pytest.mark.parametrize("lang", ["ja", "en", "zh", "ko", "de", "es"])
def test_every_listed_phrase_matches_itself(lang):
    for p in HALLUCINATION_PHRASES[lang]:
        assert is_stock_phrase(p, lang)
        assert is_stock_phrase(f"{p} {p}", lang)
