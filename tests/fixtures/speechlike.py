"""Synthetic audio shaped like speech, song and junk for the pitch and reference-gate tests:
no media file is committed, so these stand in for real clips."""

from __future__ import annotations

import numpy as np

RATE = 16000


def _syllable(f: np.ndarray, amp: float) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f) / RATE
    x = sum(a * np.sin(k * phase) for k, a in enumerate((1, 0.5, 0.33, 0.25), 1))
    env = np.hanning(len(f)) ** 0.3
    return (amp * env * x / 2).astype(np.float32)


def speech(
    hz: float, seconds: float, seed: int = 0, voiced_share: float = 0.5, amp: float = 0.25
) -> np.ndarray:
    """Speech-like audio: syllables of 0.08 to 0.22 s, each a harmonic tone gliding up to 3 st
    off ``hz`` with frame jitter, separated by silent pauses (about ``voiced_share`` voiced)."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    total, n = 0, int(seconds * RATE)
    while total < n:
        syl = int(rng.uniform(0.08, 0.22) * RATE)
        start, end = hz * 2 ** (rng.uniform(-3, 3) / 12), hz * 2 ** (rng.uniform(-3, 3) / 12)
        f = np.linspace(start, end, syl) * (1 + 0.004 * rng.standard_normal(syl))
        out.append(_syllable(f, amp))
        gap = int(syl * (1 - voiced_share) / voiced_share * rng.uniform(0.6, 1.4))
        out.append(np.zeros(gap, np.float32))
        total += syl + gap
    return np.concatenate(out)[:n]


def sung(hz: float, seconds: float, seed: int = 0, amp: float = 0.25) -> np.ndarray:
    """Song-like audio: held notes of 0.4 to 0.8 s on a scale around ``hz`` with a light
    5.5 Hz vibrato, joined by short breaths."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    total, n = 0, int(seconds * RATE)
    while total < n:
        note = int(rng.uniform(0.4, 0.8) * RATE)
        base = hz * 2 ** (int(rng.integers(-4, 5)) / 12)
        t = np.arange(note) / RATE
        f = base * 2 ** (0.25 * np.sin(2 * np.pi * 5.5 * t) / 12)
        out.append(_syllable(f, amp))
        gap = int(rng.uniform(0.03, 0.08) * RATE)
        out.append(np.zeros(gap, np.float32))
        total += note + gap
    return np.concatenate(out)[:n]


def noise_bursts(seconds: float, seed: int = 0, amp: float = 0.2) -> np.ndarray:
    """Whisper- or effect-like audio: bursts of noise shaped like syllables, with pauses."""
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    total, n = 0, int(seconds * RATE)
    while total < n:
        syl = int(rng.uniform(0.12, 0.3) * RATE)
        out.append((amp * np.hanning(syl) ** 0.3 * rng.standard_normal(syl)).astype(np.float32))
        gap = int(syl * rng.uniform(0.6, 1.4))
        out.append(np.zeros(gap, np.float32))
        total += syl + gap
    return np.concatenate(out)[:n]
