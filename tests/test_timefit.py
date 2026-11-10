import wave
from pathlib import Path

import numpy as np
import pytest

from fukikae.config import Config, ConfigError
from fukikae.timefit import FitSettings, apply_fit, fit_settings, plan_fit

S = FitSettings()  # max_speedup 1.25, max_slowdown 0.9, borrow 400 ms


def test_fits_without_change():
    p = plan_fit(1.0, (0, 1.5), None, None, 1.0, S)
    assert p.stretch == 1.0 and p.start == 0 and p.over_by == 0
    assert p.borrowed_before == 0 and p.borrowed_after == 0


def test_stretch_capped_by_rate_already_used():
    p = plan_fit(3.0, (0, 2.0), None, None, 1.2, S)  # max_speedup 1.25
    assert p.stretch == pytest.approx(1.25 / 1.2)


def test_stretch_only_as_much_as_needed():
    p = plan_fit(2.2, (0, 2.0), None, None, 1.0, S)
    assert p.stretch == pytest.approx(1.1) and p.over_by == 0 and p.borrowed_after == 0


def test_no_stretch_left_when_rate_used_the_whole_budget():
    p = plan_fit(2.4, (1.0, 3.0), None, None, 1.25, S)
    assert p.stretch == 1.0
    assert p.borrowed_after == pytest.approx(0.4)


def test_borrows_after_then_before():
    # 3.0 s at most 1.25x is 2.4 s for a 1.5 s slot: 0.9 s to find.
    p = plan_fit(3.0, (10.0, 11.5), 9.0, 12.0, 1.0, S)
    assert p.stretch == pytest.approx(1.25)
    assert p.borrowed_after == pytest.approx(0.4)  # capped by borrow_silence_ms
    assert p.borrowed_before == pytest.approx(0.4)
    assert p.start == pytest.approx(9.6)
    assert p.over_by == pytest.approx(0.1)


def test_borrow_after_limited_by_next_cue_and_before_by_previous_end():
    p = plan_fit(2.5, (10.0, 12.0), 9.9, 12.2, 1.0, FitSettings(max_speedup=1.0))
    assert p.borrowed_after == pytest.approx(0.2)
    assert p.borrowed_before == pytest.approx(0.1)
    assert p.start == pytest.approx(9.9)
    assert p.over_by == pytest.approx(0.2)


def test_left_long_when_next_cue_is_adjacent():  # Review Focus 5
    p = plan_fit(4.0, (0, 2.0), None, 2.0, 1.0, S)
    assert p.over_by > 0 and p.borrowed_after == 0
    # Never before the start of the episode.
    assert p.borrowed_before == 0 and p.start == 0


def test_previous_cue_running_long_leaves_nothing_to_borrow_before():
    p = plan_fit(3.0, (5.0, 6.0), 5.5, 6.0, 1.0, S)
    assert p.borrowed_before == 0 and p.borrowed_after == 0
    assert p.over_by == pytest.approx(3.0 / 1.25 - 1.0)


def test_short_audio_is_padded_not_slowed():
    p = plan_fit(0.5, (0, 2.0), None, None, 1.0, S)
    assert p.stretch == 1.0 and p.start == 0 and p.over_by == 0


def test_settings_are_validated():
    with pytest.raises(ValueError, match="max_speedup"):
        FitSettings(max_speedup=0.8)
    with pytest.raises(ValueError, match="max_slowdown"):
        FitSettings(max_slowdown=1.2)
    with pytest.raises(ValueError, match="borrow_silence_ms"):
        FitSettings(borrow_silence_ms=-1)


def test_fit_settings_read_timefit_workspace_over_profile():
    cfg = Config(
        raw={"timefit": {"borrow_silence_ms": 250}},
        profile_raw={"timefit": {"max_speedup": 1.3, "borrow_silence_ms": 400}},
    )
    s = fit_settings(cfg)
    assert s == FitSettings(max_speedup=1.3, max_slowdown=0.9, borrow_silence_ms=250)
    assert fit_settings(Config()) == FitSettings()


def test_fit_settings_reject_unknown_keys_and_bad_values():
    with pytest.raises(ConfigError, match="max_speedup"):
        fit_settings(Config(raw={"timefit": {"max_speedup": 0.5}}))
    with pytest.raises(ConfigError, match="borrow_silnce_ms"):
        fit_settings(Config(raw={"timefit": {"borrow_silnce_ms": 300}}))
    with pytest.raises(ConfigError, match="max_speedup"):
        fit_settings(Config(raw={"timefit": {"max_speedup": True}}))


def _tone(path: Path, seconds: float, rate: int = 16000) -> None:
    n = round(seconds * rate)
    x = 0.3 * np.sin(2 * np.pi * 300 * np.arange(n) / rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((x * 32767).astype("<i2").tobytes())


@pytest.mark.ffmpeg
def test_apply_fit_changes_duration(tmp_path):
    src = tmp_path / "in.wav"
    _tone(src, 2.0)
    p = plan_fit(2.0, (0, 1.6), None, None, 1.0, S)
    assert p.stretch == pytest.approx(1.25)
    seconds = apply_fit(src, p, tmp_path / "out.wav")
    assert seconds == pytest.approx(1.6, abs=0.02)


@pytest.mark.ffmpeg
def test_apply_fit_keeps_pitch(tmp_path):
    src = tmp_path / "in.wav"
    _tone(src, 2.0)
    out = tmp_path / "out.wav"
    apply_fit(src, plan_fit(2.0, (0, 1.6), None, None, 1.0, S), out)
    from scipy.io import wavfile

    rate, data = wavfile.read(out)
    x = data.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x))
    peak = np.fft.rfftfreq(len(x), 1 / rate)[int(np.argmax(spectrum))]
    assert peak == pytest.approx(300, abs=5)


def test_apply_fit_without_stretch_copies(tmp_path):
    src = tmp_path / "in.wav"
    _tone(src, 0.5)
    out = tmp_path / "out.wav"
    seconds = apply_fit(src, plan_fit(0.5, (0, 2.0), None, None, 1.0, S), out)
    assert out.read_bytes() == src.read_bytes() and seconds == pytest.approx(0.5)


@pytest.mark.parametrize("slot", [(5.0, 5.0), (5.0, 4.9)])
def test_zero_or_negative_slot_places_at_start_and_borrows(slot):
    """C2: a zero-length cue never divides by zero; the render starts at the slot, is not
    stretched, borrows silence on both sides and plays over the rest."""
    p = plan_fit(2.0, slot, None, None, 1.0, S)
    assert p.stretch == 1.0
    assert p.borrowed_after == pytest.approx(0.4) and p.borrowed_before == pytest.approx(0.4)
    assert p.start == pytest.approx(4.6)
    assert p.over_by == pytest.approx(1.2)


def test_zero_slot_with_nothing_rendered_is_a_no_op():
    p = plan_fit(0.0, (5.0, 5.0), None, None, 1.0, S)
    assert (p.start, p.stretch, p.over_by) == (5.0, 1.0, 0.0)
