"""Speaker diarization with pyannote.audio (gated model; needs a Hugging Face token)."""

from __future__ import annotations

import gc
import wave
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.models import Diarization, Turn

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"
_RATE = 16000


def _read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != _RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit pcm")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0


class PyannoteDiarizer:
    name: ClassVar[str] = "pyannote"
    kind: ClassVar[str] = "diarize"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 3000  # measured peak 2498 MiB (nvidia-smi delta) + 20%
    extra: ClassVar[str] = "pyannote"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model: str = str(options.get("model", DEFAULT_MODEL))
        self.min_speakers: int | None = options.get("min_speakers")
        self.max_speakers: int | None = options.get("max_speakers")
        self.device = device
        self._pipeline: Any = None
        self.peak_vram_mb: float | None = None

    def load(self) -> None:
        from huggingface_hub import get_token

        token = get_token()
        if not token:
            raise RuntimeError(
                f"pyannote needs a Hugging Face token for the gated model {self.model}: "
                f"accept its terms at https://huggingface.co/{self.model} and run "
                "`huggingface-cli login` (or set HF_TOKEN)"
            )
        import torch
        from pyannote.audio import Pipeline

        try:
            pipeline = Pipeline.from_pretrained(self.model, token=token)
        except Exception as e:
            raise RuntimeError(
                f"could not load {self.model}: {e}. Accept the terms at "
                f"https://huggingface.co/{self.model} with the account that owns the token."
            ) from e
        if pipeline is None:
            raise RuntimeError(
                f"could not load {self.model}; accept its terms at https://huggingface.co/{self.model}"
            )
        self._pipeline = pipeline.to(torch.device(self.device))

    def unload(self) -> None:
        self._pipeline = None
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def diarize(self, audio: Path) -> Diarization:
        import torch

        if self._pipeline is None:
            raise RuntimeError("PyannoteDiarizer.load() has not been called")
        wav = torch.from_numpy(_read_wav16k(audio))[None]
        kwargs: dict[str, Any] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        out = self._pipeline({"waveform": wav, "sample_rate": _RATE}, **kwargs)
        if use_cuda:
            self.peak_vram_mb = torch.cuda.max_memory_allocated(torch.device(self.device)) / 2**20
        annotation = getattr(out, "speaker_diarization", out)
        turns = [
            Turn(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _track, label in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t.start, t.end))
        return Diarization(backend=self.name, turns=turns)
