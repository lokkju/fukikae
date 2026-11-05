"""Shared helpers for the embedders: 16 kHz wav reading, span slicing, normalization."""

from __future__ import annotations

import gc
import sys
import wave
from pathlib import Path

import numpy as np

RATE = 16000
MIN_SAMPLES = 8000  # 0.5 s; shorter spans are wrap-padded so every network gets enough frames


def read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit pcm")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0


def cut(wav: np.ndarray, start: float, end: float, min_samples: int = MIN_SAMPLES) -> np.ndarray:
    """The span's samples, wrap-padded (repeated) up to `min_samples`."""
    seg = wav[max(0, int(start * RATE)) : int(end * RATE)]
    if len(seg) == 0:
        return np.zeros(min_samples, dtype=np.float32)
    if len(seg) < min_samples:
        seg = np.resize(seg, min_samples)
    return seg


def l2_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def free_gpu() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
