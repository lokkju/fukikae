"""faster-whisper (CTranslate2) speech recognition and language identification.

Whisper's own word timestamps are not trusted: the output has ``word_timing=False`` and the
aligner owns timing.
"""

from __future__ import annotations

import gc
import wave
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.models import Segment, Transcript


def _read_wav16k(path: Path) -> np.ndarray:
    """Samples as float32. Read here because faster-whisper's own decoder (PyAV) breaks on
    PyAV 19; the stage always hands over 16 kHz mono pcm."""
    with wave.open(str(path)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit pcm")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0


def _preload_cuda12_libs() -> None:
    """CTranslate2's wheel links CUDA 12's cuBLAS; torch here ships CUDA 13. The extra pulls
    ``nvidia-cublas-cu12``, whose library is loaded by path so no LD_LIBRARY_PATH is needed."""
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("nvidia.cublas")
    for base in (spec.submodule_search_locations or []) if spec else []:
        lib = Path(base) / "lib" / "libcublas.so.12"
        if lib.exists():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _device_parts(device: str) -> tuple[str, int]:
    if device.startswith("cuda"):
        _, _, idx = device.partition(":")
        return "cuda", int(idx) if idx else 0
    return "cpu", 0


def _decoding(options: dict[str, Any]) -> dict[str, Any]:
    """The decoding options set (none: faster-whisper's own defaults). ``temperature`` is one
    number (no fallback sampling, so a pass is deterministic) or a list; ``beam_size``,
    ``condition_on_previous_text`` and ``hallucination_silence_threshold`` (seconds) pass
    through."""
    out: dict[str, Any] = {}
    if "temperature" in options:
        t = options["temperature"]
        temps = list(t) if isinstance(t, (list, tuple)) else [t]
        if not temps or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and 0 <= x <= 1 for x in temps
        ):
            raise ValueError(f"faster_whisper temperature must be in [0, 1], got {t!r}")
        out["temperature"] = [float(x) for x in temps]
    if "beam_size" in options:
        b = options["beam_size"]
        if isinstance(b, bool) or not isinstance(b, int) or b < 1:
            raise ValueError(f"faster_whisper beam_size must be a positive integer, got {b!r}")
        out["beam_size"] = b
    if "condition_on_previous_text" in options:
        out["condition_on_previous_text"] = bool(options["condition_on_previous_text"])
    if "hallucination_silence_threshold" in options:
        h = options["hallucination_silence_threshold"]
        if isinstance(h, bool) or not isinstance(h, (int, float)) or h <= 0:
            raise ValueError(
                f"faster_whisper hallucination_silence_threshold must be > 0 s, got {h!r}"
            )
        out["hallucination_silence_threshold"] = float(h)
    return out


class FasterWhisperTranscriber:
    name: ClassVar[str] = "faster_whisper"
    kind: ClassVar[str] = "asr"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 5000
    supports_lid: ClassVar[bool] = True
    extra: ClassVar[str] = "whisper"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model_name: str = str(options.get("model", "large-v3"))
        self.device = device
        default = "float16" if device.startswith("cuda") else "int8"
        self.compute_type: str = str(options.get("compute_type", default))
        self.vad_filter: bool = bool(options.get("vad_filter", True))
        self.decoding: dict[str, Any] = _decoding(options)
        self._model: Any = None

    @property
    def model(self) -> str:
        return self.model_name

    def load(self) -> None:
        from faster_whisper import WhisperModel

        kind, index = _device_parts(self.device)
        if kind == "cuda":
            _preload_cuda12_libs()
        self._model = WhisperModel(
            self.model_name, device=kind, device_index=index, compute_type=self.compute_type
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
            raise RuntimeError("FasterWhisperTranscriber.load() has not been called")
        return self._model

    def detect_language(self, audio: Path) -> tuple[str, float]:
        """Whisper's language guess over up to four speech segments of the audio."""
        model = self._require()
        samples = _read_wav16k(audio)
        lang, prob, _ = model.detect_language(
            samples, vad_filter=True, language_detection_segments=4
        )
        return str(lang), float(prob)

    def transcribe(self, audio: Path, language: str | None) -> Transcript:
        model = self._require()
        segments, info = model.transcribe(
            _read_wav16k(audio),
            language=language,
            vad_filter=self.vad_filter,
            # faster-whisper applies the hallucination guard only with word timing; the
            # words are still not trusted or returned
            word_timestamps="hallucination_silence_threshold" in self.decoding,
            **self.decoding,
        )
        out = [
            Segment(start=float(s.start), end=float(s.end), text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]
        return Transcript(
            language=language or str(info.language),
            backend=self.name,
            word_timing=False,
            segments=out,
        )
