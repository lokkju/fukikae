"""Time fitting: what remains after translate's budget and the tts stage's rate fit.

Order, per the foundation spec: TTS rate control (done by the tts stage), bounded time-stretch,
then silence borrowed after the slot and before it. Rate and stretch share one budget: the
combined speed-up never exceeds ``max_speedup``, so a render the tts stage already sped up by
``rate_factor`` may be stretched by at most ``max_speedup / rate_factor``. Borrowing takes up to
``borrow_silence_ms`` after the slot, up to the next cue's start, then up to the same before it,
back to where the previous cue's audio ends. Whatever still does not fit is ``over_by``: the
audio is placed whole and plays past it (overlapping the next cue, both audible), never cut.

Audio shorter than its slot starts at the slot start and is padded with silence; it is never
slowed. ``max_slowdown`` bounds any slow-down, so with no slowing at all it holds by
construction; it is validated and kept in the stage key so a later slow-down of short lines
stays inside it.

Time-stretch is ffmpeg's ``atempo`` (pitch-preserving, in every ffmpeg build; rubberband is not
assumed).
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fukikae import media
from fukikae.artifacts import atomic_path
from fukikae.config import Config, ConfigError

FIT_KEYS = ("max_speedup", "max_slowdown", "borrow_silence_ms")


@dataclass(frozen=True)
class FitSettings:
    max_speedup: float = 1.25
    max_slowdown: float = 0.9
    borrow_silence_ms: int = 400

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_speedup) and self.max_speedup >= 1.0):
            raise ValueError(f"max_speedup must be >= 1.0, not {self.max_speedup!r}")
        if not (math.isfinite(self.max_slowdown) and 0.0 < self.max_slowdown <= 1.0):
            raise ValueError(f"max_slowdown must be in (0, 1], not {self.max_slowdown!r}")
        if self.borrow_silence_ms < 0:
            raise ValueError(f"borrow_silence_ms must be >= 0, not {self.borrow_silence_ms!r}")


@dataclass(frozen=True)
class FitPlan:
    """Where and how a render plays: ``start`` (seconds), ``stretch`` (speed factor, 1.0 for
    none), the silence borrowed before and after the slot, and ``over_by``, the seconds that
    still do not fit."""

    start: float
    stretch: float
    borrowed_before: float
    borrowed_after: float
    over_by: float

    def seconds(self, rendered_s: float) -> float:
        return rendered_s / self.stretch


def _number(table: dict[str, Any], key: str, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"[timefit] {key} must be a number, not {value!r}")
    return float(value)


def fit_settings(cfg: Config) -> FitSettings:
    """``[timefit]``, the workspace's keys over the profile's; the defaults for anything
    unset. Unknown keys and out-of-range values are ConfigErrors naming the key."""
    merged: dict[str, Any] = {}
    for source in (cfg.profile_raw, cfg.raw):
        table = source.get("timefit")
        if isinstance(table, dict):
            merged.update(table)
    unknown = sorted(set(merged) - set(FIT_KEYS))
    if unknown:
        raise ConfigError(
            f"[timefit] has unknown keys {', '.join(unknown)}; known: {', '.join(FIT_KEYS)}"
        )
    d = FitSettings()
    speedup = _number(merged, "max_speedup", d.max_speedup)
    slowdown = _number(merged, "max_slowdown", d.max_slowdown)
    borrow = _number(merged, "borrow_silence_ms", d.borrow_silence_ms)
    try:
        return FitSettings(speedup, slowdown, round(borrow))
    except ValueError as e:
        raise ConfigError(f"[timefit] {e}") from e


def plan_fit(
    rendered_s: float,
    slot: tuple[float, float],
    prev_end: float | None,
    next_start: float | None,
    rate_factor: float,
    s: FitSettings,
) -> FitPlan:
    """How to place ``rendered_s`` seconds of audio in ``slot``. ``prev_end`` is where the
    previous cue's audio ends (None: nothing before), ``next_start`` where the next cue starts
    (None: nothing after), ``rate_factor`` the speed-up the tts stage already applied."""
    a, b = slot
    length = b - a
    if rendered_s <= length:
        return FitPlan(a, 1.0, 0.0, 0.0, 0.0)
    if length <= 0.0:
        # a zero-length (or inverted) slot: no stretch can fit it, so the render starts at
        # the slot, unstretched, and borrows and plays over as usual
        b, length, stretch = a, 0.0, 1.0
        if rendered_s <= 0.0:
            return FitPlan(a, 1.0, 0.0, 0.0, 0.0)
    else:
        cap = max(1.0, s.max_speedup / max(rate_factor, 1e-9))
        stretch = min(rendered_s / length, cap)
        if stretch < 1.0 + 1e-6:
            stretch = 1.0
    remain = rendered_s / stretch - length
    borrow = s.borrow_silence_ms / 1000.0
    after_room = math.inf if next_start is None else next_start - b
    after = max(0.0, min(remain, borrow, after_room))
    remain -= after
    before_room = a - (0.0 if prev_end is None else max(prev_end, 0.0))
    before = max(0.0, min(remain, borrow, before_room))
    remain -= before
    over = remain if remain > 1e-9 else 0.0
    return FitPlan(a - before, stretch, before, after, over)


def apply_fit(src: Path, plan: FitPlan, out: Path) -> float:
    """Write ``src`` to ``out`` stretched by ``plan.stretch`` (ffmpeg ``atempo``, float
    samples, the source's rate), or copied when there is no stretch; returns the seconds
    written."""
    from scipy.io import wavfile

    out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(out) as tmp:
        if plan.stretch == 1.0:
            shutil.copyfile(src, tmp)
        else:
            media.run_ffmpeg(
                ["-i", str(src), "-filter:a", f"atempo={plan.stretch:.6f}",
                 "-c:a", "pcm_f32le", str(tmp)]
            )  # fmt: skip
    rate, data = wavfile.read(out, mmap=True)
    return len(data) / rate
