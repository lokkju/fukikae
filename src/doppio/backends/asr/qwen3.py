"""Qwen3-ASR speech recognition and language identification (transformers path, no vLLM).

The model has no segment timing of its own, so the audio is cut into chunks of at most
``max_chunk_s`` seconds at pauses found by a simple energy detector; each chunk becomes one
Segment carrying that chunk's start and end. Output has ``word_timing=False``; the
Qwen3 forced aligner (or any other aligner) supplies word times.
"""

from __future__ import annotations

import gc
import wave
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from doppio.models import Segment, Transcript

# Qwen's language names (the model card's 30) -> doppio's ISO 639-1 codes.
LANGUAGES: dict[str, str] = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "tl",
    "Persian": "fa",
    "Greek": "el",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Macedonian": "mk",
}
_BY_CODE = {code: name for name, code in LANGUAGES.items()}
_BY_LOWER = {name.lower(): code for name, code in LANGUAGES.items()}
_RATE = 16000
_FRAME_S = 0.02
_PAD_S = 0.1  # audio kept around a chunk so its first and last sounds are not clipped
# The model decodes at most 30 s; a chunk plus its padding on both sides stays under that.
MAX_CHUNK_S = 28.0


def name_to_code(name: str) -> str | None:
    """ISO 639-1 code for a Qwen language name (any case); None when unknown or empty."""
    return _BY_LOWER.get(name.strip().lower())


def code_to_name(code: str) -> str:
    try:
        return _BY_CODE[code]
    except KeyError:
        raise ValueError(f"qwen3_asr does not support language {code!r}") from None


def read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != _RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit pcm")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0


def plan_chunks(
    samples: np.ndarray, sr: int = _RATE, max_s: float = MAX_CHUNK_S, min_s: float = 0.2
) -> list[tuple[float, float]]:
    """Speech chunks ``(start, end)`` in seconds, each at most ``max_s`` long.

    Energy VAD: 20 ms frames are active above a tenth of the 90th-percentile frame level.
    A chunk that would exceed ``max_s`` is cut in the middle of the longest quiet run in the
    last two thirds of its window (the quietest frame when there is none). Chunks are then
    trimmed to their active frames; chunks with no speech are dropped.
    """
    hop = int(sr * _FRAME_S)
    n = len(samples) // hop
    if n == 0:
        return []
    frames = samples[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    thr = max(1e-3, 0.1 * float(np.percentile(rms, 90)))
    active = rms > thr
    max_f = int(max_s / _FRAME_S)
    bounds: list[tuple[int, int]] = []
    pos = 0
    while n - pos > max_f:
        lo, hi = pos + max_f // 3, pos + max_f
        cut = _quiet_cut(active[lo:hi], rms[lo:hi]) + lo
        bounds.append((pos, cut))
        pos = cut
    bounds.append((pos, n))
    out: list[tuple[float, float]] = []
    for a, b in bounds:
        idx = np.flatnonzero(active[a:b])
        if len(idx) == 0:
            continue
        start, end = (a + int(idx[0])) * _FRAME_S, (a + int(idx[-1]) + 1) * _FRAME_S
        if end - start >= min_s:
            out.append((start, end))
    return out


def _quiet_cut(active: np.ndarray, rms: np.ndarray) -> int:
    """Frame index within the window to cut at."""
    best_len, best_mid = 0, -1
    run_start = None
    for i, a in enumerate([*active.tolist(), True]):
        if not a and run_start is None:
            run_start = i
        elif a and run_start is not None:
            if i - run_start >= best_len:
                best_len, best_mid = i - run_start, (run_start + i) // 2
            run_start = None
    return best_mid if best_mid >= 0 else int(np.argmin(rms))


class Qwen3AsrTranscriber:
    name: ClassVar[str] = "qwen3_asr"
    kind: ClassVar[str] = "asr"
    languages: ClassVar[frozenset[str] | Literal["*"]] = frozenset(LANGUAGES.values())
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 5800  # measured peak 4.8 GB (bf16, batch 8) + 20%
    supports_lid: ClassVar[bool] = True
    extra: ClassVar[str] = "qwen"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model_name: str = str(options.get("model", "Qwen/Qwen3-ASR-1.7B"))
        self.device = device
        self.batch_size: int = int(options.get("batch_size", 8))
        self.max_chunk_s: float = float(options.get("max_chunk_s", MAX_CHUNK_S))
        self._model: Any = None

    @property
    def model(self) -> str:
        return self.model_name

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        cuda = self.device.startswith("cuda")
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16 if cuda else torch.float32,
            device_map=self.device if cuda else "cpu",
            max_inference_batch_size=self.batch_size,
            max_new_tokens=512,
        )

    def unload(self) -> None:
        self._model = None
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require(self) -> Any:
        if self._model is None:
            raise RuntimeError("Qwen3AsrTranscriber.load() has not been called")
        return self._model

    def _run(
        self, samples: np.ndarray, chunks: list[tuple[float, float]], language: str | None
    ) -> list[Any]:
        model = self._require()
        audio = [
            (samples[int(max(0.0, a - _PAD_S) * _RATE) : int((b + _PAD_S) * _RATE)], _RATE)
            for a, b in chunks
        ]
        results: list[Any] = []
        for i in range(0, len(audio), self.batch_size):
            part = audio[i : i + self.batch_size]
            results.extend(model.transcribe(audio=part, language=[language] * len(part)))
        return results

    def detect_language(self, audio: Path) -> tuple[str, float]:
        """Qwen's own guess over the three longest speech chunks; confidence is the share of
        those chunks that agree."""
        samples = read_wav16k(audio)
        chunks = sorted(plan_chunks(samples, max_s=self.max_chunk_s), key=lambda c: c[0] - c[1])[:3]
        votes: Counter[str] = Counter()
        for r in self._run(samples, chunks, None):
            code = name_to_code(str(r.language).split(",")[0])
            if code:
                votes[code] += 1
        if not votes:
            raise ValueError(f"{audio}: no speech to identify a language from")
        code, count = votes.most_common(1)[0]
        return code, count / len(chunks)

    def transcribe(self, audio: Path, language: str | None) -> Transcript:
        self._require()
        samples = read_wav16k(audio)
        chunks = plan_chunks(samples, max_s=self.max_chunk_s)
        results = self._run(samples, chunks, code_to_name(language) if language else None)
        segments: list[Segment] = []
        votes: Counter[str] = Counter()
        for (a, b), r in zip(chunks, results, strict=True):
            code = name_to_code(str(r.language).split(",")[0])
            if code:
                votes[code] += 1
            text = str(r.text).strip()
            if text:
                segments.append(Segment(start=a, end=b, text=text))
        detected = votes.most_common(1)[0][0] if votes else ""
        return Transcript(
            language=language or detected,
            backend=self.name,
            word_timing=False,
            segments=segments,
        )
