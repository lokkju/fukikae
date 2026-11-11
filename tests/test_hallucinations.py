import numpy as np
import pytest

from fukikae.hallucinations import (
    FLAG_FRAGMENT,
    FLAG_SPOKEN,
    HALLUCINATION_ACTIVE_MAX,
    HALLUCINATION_MUSIC_SHARE,
    HALLUCINATION_PHRASES,
    RUN_GAP_S,
    clean_segment,
    drop_reason,
    fold,
    is_stock_phrase,
    phrase_table_digest,
    rule_constants,
    segment_span,
    word_runs,
)
from fukikae.models import Segment, Word
from fukikae.songs import SongSpan

DUR = 60  # seconds
STOCK = "ご視聴ありがとうございました"


def SUNG(lo: float, hi: float) -> bool:  # noqa: ARG001, N802
    return True


def SPOKEN(lo: float, hi: float) -> bool:  # noqa: ARG001, N802
    return False


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


def test_phrase_over_sung_music_is_dropped():
    mask = _quiet_mask()
    mask[10:30] = True
    reason = drop_reason(_seg(10, 30), "ja", [], mask, _active(0.9), SUNG)
    assert reason == "stock phrase over sung music"


def test_phrase_spoken_over_loud_music_is_kept():
    """The per-second mask reads a loud score under speech as song-like; without a sung signal
    the line is speech and stays (Task 10 review ruling)."""
    mask = _quiet_mask()
    mask[10:30] = True
    assert drop_reason(_seg(10, 30), "ja", [], mask, _active(0.9), SPOKEN) is None
    assert drop_reason(_seg(10, 30), "ja", [], mask, _active(0.9)) is None  # no sung test


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
    reason = drop_reason(_seg(10, 20), "ja", [], mask, _active(0.9), SUNG)
    assert reason == "stock phrase over sung music"
    mask[10] = False
    assert drop_reason(_seg(10, 20), "ja", [], mask, _active(0.9), SUNG) is None
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
    assert drop_reason(seg, "ja", [], mask, _active(0.9), SUNG) == "stock phrase over sung music"
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


# Sol Bianca S00E01's 「お疲れ様でした」: one segment over 3517-3774 s whose words sit in two
# runs, お疲 at 3564-3570 s and れ様でした at 3677-3684 s, the second inside the theme song
# (3658-3740 s).
SB_WORDS = [
    (3564.4337, "お"), (3569.6741, "疲"), (3677.2224, "れ"), (3678.5425, "様"),
    (3678.8026, "で"), (3678.9426, "し"), (3684.4430, "た"),
]  # fmt: skip
SB_THEME = [SongSpan(3658.0, 3740.0, "detector")]


def _sb_seg() -> Segment:
    words = [Word(start=t, end=t + 0.02, text=ch) for t, ch in SB_WORDS]
    return Segment(start=3517.0, end=3774.4, text="お疲れ様でした", words=words)


def _long(seconds: int = 3800) -> tuple[np.ndarray, np.ndarray]:
    """No second song-like and the vocals on everywhere: only the theme span is evidence."""
    return np.zeros(seconds, dtype=bool), np.ones(seconds * 10, dtype=bool)


def test_word_runs_split_at_gaps():
    runs = word_runs(_sb_seg().words or [])
    assert ["".join(w.text for w in r) for r in runs] == ["お疲", "れ様でした"]
    assert RUN_GAP_S > 5.5  # the run's own words sit up to 5.5 s apart


def test_only_the_run_inside_the_theme_is_dropped():
    song_like, active = _long()
    kept, dropped, flags = clean_segment(_sb_seg(), "ja", SB_THEME, song_like, active)
    assert kept is not None and kept.text == "お疲"
    assert [w.text for w in kept.words or []] == ["お", "疲"]
    assert kept.start == pytest.approx(3564.4337) and kept.end == pytest.approx(3569.6941)
    (d,) = dropped
    assert d.text == "れ様でした" and d.reason == "stock phrase over a theme song"
    assert (d.start, d.end) == (pytest.approx(3677.2224), pytest.approx(3684.463))
    (f,) = flags  # the surviving fragment is kept and flagged
    assert f.text == "お疲" and f.reason == "surviving fragment of a stock phrase"
    assert (f.start, f.end) == (pytest.approx(3564.4337), pytest.approx(3569.6941))


def test_a_segment_whose_every_run_passes_goes_whole():
    song_like, active = _long()
    theme = [SongSpan(3500.0, 3740.0, "detector")]
    kept, dropped, flags = clean_segment(_sb_seg(), "ja", theme, song_like, active)
    assert kept is None and flags == []
    (d,) = dropped
    assert d.text == "お疲れ様でした" and (d.start, d.end) == segment_span(_sb_seg())


def test_a_segment_with_no_run_passing_is_kept_as_is():
    song_like, active = _long()
    seg = _sb_seg()
    assert clean_segment(seg, "ja", [], song_like, active) == (seg, [], [])


def test_other_text_is_never_split():
    song_like, active = _long()
    seg = _sb_seg().model_copy(update={"text": "燃えてきたぞ"})
    assert clean_segment(seg, "ja", SB_THEME, song_like, active) == (seg, [], [])


def test_a_segment_without_words_is_judged_whole():
    seg = _seg(10, 30)
    spans = [SongSpan(0.0, 60.0, "detector")]
    kept, dropped, _ = clean_segment(seg, "ja", spans, _quiet_mask(), _active(0.9))
    assert kept is None and [d.text for d in dropped] == [STOCK]


def test_kept_words_join_with_spaces_in_a_spaced_language():
    words = [
        Word(start=t, end=t + 0.2, text=w)
        for t, w in [(1.0, " thanks"), (1.3, " for"), (1.6, " watching"),
                     (40.0, " thanks"), (40.3, " for"), (40.6, " watching")]
    ]  # fmt: skip
    seg = Segment(start=0.0, end=41.0, text="thanks for watching thanks for watching", words=words)
    kept, dropped, _ = clean_segment(seg, "en", [SongSpan(35.0, 50.0, "detector")],
                                  _quiet_mask(), _active(0.9))  # fmt: skip
    assert kept is not None and kept.text == "thanks for watching"
    assert dropped[0].text == "thanks for watching"


def test_a_run_spoken_over_music_is_kept_and_flagged():
    """Sol Bianca's お疲 run sits over music the per-second mask calls song-like; spoken, it is
    kept and flagged, while the run inside the theme still goes."""
    song_like, active = _long()
    song_like[3564:3571] = True
    kept, dropped, flags = clean_segment(_sb_seg(), "ja", SB_THEME, song_like, active, SPOKEN)
    assert kept is not None and kept.text == "お疲"
    assert [d.text for d in dropped] == ["れ様でした"]
    (f,) = flags
    assert f.reason == f"{FLAG_FRAGMENT}; {FLAG_SPOKEN}"
    kept, dropped, flags = clean_segment(_sb_seg(), "ja", SB_THEME, song_like, active, SUNG)
    assert kept is None and flags == []
    assert dropped[0].reason == "stock phrase over sung music; stock phrase over a theme song"


def test_a_whole_segment_spoken_over_music_is_kept_and_flagged():
    mask = _quiet_mask()
    mask[10:30] = True
    seg = _seg(10, 30)
    kept, dropped, flags = clean_segment(seg, "ja", [], mask, _active(0.9), SPOKEN)
    assert kept == seg and dropped == []
    assert [(f.text, f.reason) for f in flags] == [
        (STOCK, "stock phrase spoken over music (not sung)")
    ]


def test_a_phrase_said_over_quiet_dialogue_is_not_flagged():
    seg = _seg(10, 14, words=True)
    assert clean_segment(seg, "ja", [], _quiet_mask(), _active(0.9), SPOKEN) == (seg, [], [])
