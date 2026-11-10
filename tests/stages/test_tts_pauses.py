"""D33: the internal-pause cap, a post-pass on every cached render."""

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import wavfile

from fukikae.cards.card import Voice
from fukikae.config import Config, ConfigError
from fukikae.stages import tts
from fukikae.stages.tts import TtsStage
from tests.stages import test_tts as base
from tests.stages.test_tts import (
    CFG,
    CUES,
    FakeSynth,
    _cue,
    _plan,
    _run,
    _two_cards,
    _write_cues,
)

ep = base.ep  # the tts tests' episode fixture
fakes = base.fakes  # and their autouse fake backends

RATE = 16000


def _tone(seconds: float, amp: float = 0.25, freq: float = 440.0) -> np.ndarray:
    n = round(seconds * RATE)
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / RATE)


def _speech(*parts: float, lead: float = 0.0, trail: float = 0.0, noise: float = 0.0) -> np.ndarray:
    """Tones and gaps alternating (``parts`` seconds: tone, gap, tone, ...), with optional
    leading and trailing silence and a noise floor ``noise`` (peak) in every silence."""
    rng = np.random.default_rng(7)
    out = [np.zeros(round(lead * RATE))]
    for i, s in enumerate(parts):
        out.append(_tone(s) if i % 2 == 0 else np.zeros(round(s * RATE)))
    out.append(np.zeros(round(trail * RATE)))
    x = np.concatenate(out)
    if noise:
        quiet = np.abs(x) == 0
        x[quiet] = rng.uniform(-noise, noise, int(quiet.sum()))
    return x


def _int16(x: np.ndarray) -> np.ndarray:
    return (x * 32767).astype(np.int16)


def longest_pause_ms(data: np.ndarray, rate: int = RATE) -> int:
    """The longest silence between speech, by the trim's own frames and threshold."""
    found = tts._loud_frames(data, rate)
    assert found is not None
    mask, n = found
    idx = np.flatnonzero(mask)
    best = run = 0
    for v in mask[idx[0] : idx[-1] + 1]:
        run = 0 if v else run + 1
        best = max(best, run)
    return round(best * n * 1000 / rate)


def test_a_500_ms_internal_gap_becomes_200_ms():
    x = _int16(_speech(0.3, 0.5, 0.3))
    out, removed = tts.cap_pauses(x, RATE, 200)
    assert removed == round(0.3 * RATE)
    assert len(out) == len(x) - removed
    assert longest_pause_ms(out) == 200
    assert out.dtype == x.dtype


def test_pauses_at_or_under_the_cap_are_left_alone():
    x = _int16(_speech(0.3, 0.2, 0.3, 0.15, 0.3))
    out, removed = tts.cap_pauses(x, RATE, 200)
    assert removed == 0 and np.array_equal(out, x)


def test_leading_and_trailing_silence_are_untouched():
    lead, trail = round(0.4 * RATE), round(0.6 * RATE)
    x = _int16(_speech(0.3, 0.5, 0.3, lead=0.4, trail=0.6, noise=0.001))
    out, removed = tts.cap_pauses(x, RATE, 200)
    assert removed == round(0.3 * RATE)
    assert np.array_equal(out[:lead], x[:lead])
    assert np.array_equal(out[-trail:], x[-trail:])


def test_every_long_pause_is_capped():
    x = _int16(_speech(0.3, 0.5, 0.3, 0.7, 0.3, 0.1, 0.3))
    out, removed = tts.cap_pauses(x, RATE, 200)
    assert removed == round((0.3 + 0.5) * RATE)
    assert longest_pause_ms(out) == 200


def test_the_cut_keeps_both_edges_of_the_pause():
    """The middle goes: the speech's decay into the pause and the onset out of it stay."""
    x = _speech(0.3, 0.5, 0.3)
    out, _ = tts.cap_pauses(x, RATE, 200)
    first = round(0.3 * RATE)
    assert np.array_equal(out[: first + round(0.09 * RATE)], x[: first + round(0.09 * RATE)])
    tail = round((0.3 + 0.09) * RATE)
    assert np.array_equal(out[-tail:], x[-tail:])


def test_no_clicks_at_the_join():
    """A low hum under the pause, cut at different phases on either side: a hard splice
    would jump by up to twice its amplitude; the crossfade keeps every sample step at the
    join close to the hum's own (a 42.5 Hz sine moves under 2% of its amplitude a sample)."""
    hum = 0.002
    x = _speech(0.3, 0.5, 0.3)
    t = np.arange(len(x)) / RATE
    quiet = x == 0
    x[quiet] = hum * np.sin(2 * np.pi * 42.5 * t[quiet])
    out, removed = tts.cap_pauses(x, RATE, 200)
    assert removed > 0
    join = round(0.4 * RATE)  # the pause starts at 0.3 s; its first 100 ms are kept
    window = out[join - 400 : join + 400]
    assert np.abs(window).max() <= hum + 1e-9
    assert np.abs(np.diff(window)).max() <= 0.1 * hum
    spliced = np.concatenate([x[:join], x[join + removed :]])  # the same cut with no fade
    assert np.abs(np.diff(spliced[join - 400 : join + 400])).max() > 0.5 * hum


def test_stereo_and_float_renders_are_capped_alike():
    x = _speech(0.3, 0.5, 0.3).astype(np.float32)
    stereo = np.stack([x, x], axis=1)
    out, removed = tts.cap_pauses(stereo, RATE, 200)
    assert out.shape == (len(x) - removed, 2) and out.dtype == np.float32
    assert removed == round(0.3 * RATE)


def test_zero_disables_the_cap():
    x = _int16(_speech(0.3, 0.5, 0.3))
    out, removed = tts.cap_pauses(x, RATE, 0)
    assert removed == 0 and out is x


def test_a_silent_render_is_left_whole():
    x = np.zeros(RATE, dtype=np.int16)
    out, removed = tts.cap_pauses(x, RATE, 200)
    assert removed == 0 and out is x


# -- settings -------------------------------------------------------------------------------


def test_defaults_are_200_and_500_ms():
    assert tts.pause_caps(CFG) == (200, 500)


def test_ellipsis_lines_get_the_higher_cap():
    caps = (200, 500)
    assert tts.pause_cap_ms(caps, "Let's go, Happy!") == 200
    assert tts.pause_cap_ms(caps, "Wait... what?") == 500
    assert tts.pause_cap_ms(caps, "Natsu…") == 500
    assert tts.pause_cap_ms((0, 500), "Wait... what?") == 0  # 0 turns the cap off everywhere
    assert tts.pause_cap_ms((200, 0), "Wait... what?") == 0  # no cap on ellipsis lines only


def test_an_ellipsis_line_keeps_up_to_500_ms():
    x = _int16(_speech(0.3, 0.45, 0.3, 0.8, 0.3))
    cap = tts.pause_cap_ms((200, 500), "Well... I... guess")
    out, removed = tts.cap_pauses(x, RATE, cap)
    assert removed == round(0.3 * RATE)  # 450 ms kept, 800 ms down to 500
    assert longest_pause_ms(out) == 500


@pytest.mark.parametrize(
    ("table", "match"),
    [
        ({"max_pause_ms": -1}, "max_pause_ms"),
        ({"max_pause_ms": 10}, "max_pause_ms"),
        ({"max_pause_ms": 200.5}, "max_pause_ms"),
        ({"max_pause_ms": True}, "max_pause_ms"),
        ({"max_pause_ms_ellipsis": "500"}, "max_pause_ms_ellipsis"),
    ],
)
def test_bad_caps_are_refused(table, match):
    with pytest.raises(ConfigError, match=match):
        tts.pause_caps(Config(raw={"tts": {"backend": "fake", **table}}))


def test_the_caps_are_stage_keys_not_backend_options():
    cfg = Config(raw={"tts": {"backend": "fake", "max_pause_ms": 250}})
    assert tts.pause_caps(cfg) == (250, 500)
    assert "max_pause_ms" not in tts.backend_config(cfg, "fake").options
    assert "max_pause_ms" not in tts.stage_settings(cfg)[0].options


# -- keys -----------------------------------------------------------------------------------


def test_capped_key_follows_the_cap_and_zero_is_the_render_itself():
    assert tts.capped_key("abc", 0) == "abc"
    assert tts.capped_key("abc", 200) != "abc"
    assert tts.capped_key("abc", 200) != tts.capped_key("abc", 300)
    assert tts.capped_key("abc", 200) == tts.capped_key("abc", 200)


def test_the_cue_key_does_not_hold_the_cap(ep):
    """The cache keeps the raw render; changing the cap never renders again."""
    natsu, _ = _two_cards(ep)
    from fukikae.cards.store import CardStore

    store = CardStore(ep.workspace_root)
    vocals = ep.work / "01-separate" / "vocals.wav"
    a = tts.prepare_cue(CUES[0], natsu, CFG, store, who="Natsu", vocals=vocals)
    cfg = Config(raw={"tts": {"backend": "fake", "max_pause_ms": 300}})
    b = tts.prepare_cue(CUES[0], natsu, cfg, store, who="Natsu", vocals=vocals)
    assert a.key == b.key
    assert (a.max_pause_ms, b.max_pause_ms) == (200, 300)


def test_stage_key_holds_the_caps_only_while_capping(ep):
    _two_cards(ep)
    base = TtsStage(CFG).config_key(CFG, ep)
    other = Config(raw={"tts": {"backend": "fake", "max_pause_ms": 300}})
    off = Config(raw={"tts": {"backend": "fake", "max_pause_ms": 0}})
    assert base["pauses"] == {"max_pause_ms": 200, "max_pause_ms_ellipsis": 500, "version": 1}
    assert TtsStage(other).config_key(other, ep) != base
    assert "pauses" not in TtsStage(off).config_key(off, ep)


# -- the stage ------------------------------------------------------------------------------


GAP: dict[str, tuple[float, float, float]] = {}
_orig = FakeSynth.synthesize


def _gappy_synthesize(
    self: FakeSynth, text: str, voice: Voice, language: str, out: Path, **kw: Any
) -> float:
    """FakeSynth, but a text in GAP renders (seconds of tone, silent gap, tone)."""
    if text not in GAP:
        return _orig(self, text, voice, language, out, **kw)
    FakeSynth.CALLS.append((text, voice.voice, voice.rate))
    a, gap, b = (s / voice.rate for s in GAP[text])
    x = np.concatenate(
        [np.zeros(800), _tone(a), np.zeros(round(gap * RATE)), _tone(b), np.zeros(800)]
    )
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(_int16(x).astype("<i2").tobytes())
    return len(x) / RATE


@pytest.fixture
def gappy(monkeypatch):
    GAP.clear()
    monkeypatch.setattr(FakeSynth, "synthesize", _gappy_synthesize)
    return GAP


def _seconds(c: Any) -> float:
    assert c.seconds is not None
    return float(c.seconds)


def test_the_stage_plays_the_capped_render(ep, gappy):
    _two_cards(ep)
    gappy["natsu one"] = (0.5, 0.6, 0.5)
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    rate, data = wavfile.read(ep.work / "07-tts" / "c0001.wav")
    assert longest_pause_ms(data, rate) == 200
    # 0.5 + 0.2 + 0.5 s of sound, plus the trim's 30 ms at each end
    assert c.seconds == pytest.approx(1.2 + 2 * tts.TRIM_KEEP_S, abs=0.011)
    assert c.rate_factor == 1.0  # 1.6 s raw would have fit the 2 s slot too


def test_the_rate_fit_measures_the_capped_render(ep, gappy):
    """2.3 s raw overruns the 2 s slot; capped to 1.9 s it fits, so no faster render."""
    _two_cards(ep)
    gappy["natsu one"] = (0.8, 0.7, 0.8)
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert [r for t, _, r in FakeSynth.CALLS if t == "natsu one"] == [1.0]
    assert c.rate_factor == 1.0
    assert c.rendered_seconds == pytest.approx(c.seconds)
    assert _seconds(c) < 2.0


def test_changing_the_cap_rederives_without_rendering(ep, gappy):
    _two_cards(ep)
    gappy["natsu one"] = (0.5, 0.6, 0.5)
    TtsStage(CFG).run(ep, CFG)
    first = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    FakeSynth.CALLS.clear()
    cfg = Config(raw={"tts": {"backend": "fake", "max_pause_ms": 400}})
    TtsStage(cfg).run(ep, cfg)
    assert FakeSynth.CALLS == []
    again = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert again.key != first.key
    assert again.seconds == pytest.approx(_seconds(first) + 0.2, abs=0.011)
    assert "c0001" in _run(ep).cached
    off = Config(raw={"tts": {"backend": "fake", "max_pause_ms": 0}})
    TtsStage(off).run(ep, off)
    assert FakeSynth.CALLS == []
    raw = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert raw.seconds == pytest.approx(_seconds(first) + 0.4, abs=0.011)


def test_a_fitted_render_is_capped_too(ep, gappy):
    _two_cards(ep)
    gappy["natsu one"] = (1.5, 0.6, 1.5)  # 3.2 s capped: over the slot, rendered at 1.25
    TtsStage(CFG).run(ep, CFG)
    c = {c.cue_id: c for c in _plan(ep).cues}["c0001"]
    assert [r for t, _, r in FakeSynth.CALLS if t == "natsu one"] == [1.0, 1.25]
    rate, data = wavfile.read(ep.work / "07-tts" / "c0001.wav")
    assert longest_pause_ms(data, rate) == 200
    assert c.seconds == pytest.approx((3.0 / 1.25) + 0.2 + 2 * tts.TRIM_KEEP_S, abs=0.011)


def test_an_ellipsis_cue_keeps_its_longer_pause(ep, gappy):
    cues = [_cue(1, 1, "natsu... one"), _cue(2, 2, "lucy one")]
    _write_cues(ep, cues)
    _two_cards(ep)
    gappy["natsu... one"] = (0.5, 0.45, 0.5)
    TtsStage(CFG).run(ep, CFG)
    rate, data = wavfile.read(ep.work / "07-tts" / "c0001.wav")
    assert longest_pause_ms(data, rate) == 450
