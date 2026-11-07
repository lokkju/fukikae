"""Dialogue separation with DBS-TIGER (DnR v2): speech versus music and effects.

Trained on Divide and Remaster (cinematic soundtracks), so the speech stem is meant to drop
effects and shouts that a music-oriented separator keeps in its vocal stem. The model takes
44.1 kHz mono and returns music, sfx, speech.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.backends.separation.demucs import _read_wav, _write_wav
from fukikae.models import Separation

RATE = 44100


class DialogueSeparator:
    name: ClassVar[str] = "dialogue"
    kind: ClassVar[str] = "separation"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 3000
    extra: ClassVar[str] = "dialogue"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model_name: str = str(options.get("model", "JusperLee/DBS-TIGER-DnR"))
        self.background_mode: str = str(options.get("background", "residual"))
        if self.background_mode not in ("residual", "stems"):
            raise ValueError(
                f"background must be 'residual' or 'stems', got {self.background_mode!r}"
            )
        self.chunk_seconds = float(options.get("chunk_seconds", 6.0))
        self.hop_seconds = float(options.get("hop_seconds", 3.0))
        self.device = device
        self._model: Any = None
        self.peak_vram_mb: float | None = None

    @property
    def model(self) -> str:
        return self.model_name

    def load(self) -> None:
        from dbs.models import DBSTIGER

        self._model = DBSTIGER.from_pretrained(self.model_name).eval().to(self.device)

    def unload(self) -> None:
        self._model = None
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_model(self, mono: np.ndarray, rate: int) -> np.ndarray:
        """``(samples,)`` mono mix to ``(3, samples)``: music, sfx, speech."""
        import torch
        from dbs.engine import separate

        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        x = torch.from_numpy(mono.astype(np.float32)).view(1, 1, -1).to(self.device)
        protocol = {
            "mode": "chunked",
            "chunk_seconds": self.chunk_seconds,
            "hop_seconds": self.hop_seconds,
        }
        with torch.inference_mode():
            out = separate(self._model, x, protocol, sample_rate=rate)[0]
        if use_cuda:
            self.peak_vram_mb = torch.cuda.max_memory_allocated(torch.device(self.device)) / 2**20
        return out.reshape(3, -1).float().cpu().numpy()

    def separate(self, audio: Path, out_dir: Path) -> Separation:
        """Write ``vocals.wav`` (the speech stem) and ``background.wav`` to `out_dir`.

        Option ``background``: ``residual`` (default, mix minus speech, per Ruling R5) or
        ``stems`` (music plus effects). The returned ``vocals_16k`` is a placeholder; the
        stage derives the 16 kHz mono file.
        """
        if self._model is None:
            raise RuntimeError("DialogueSeparator.load() has not been called")
        wav, rate = _read_wav(audio)
        if rate != RATE:
            raise ValueError(f"{audio}: {rate} Hz, model needs {RATE} Hz")
        n = wav.shape[1]
        stems = self._run_model(wav.mean(0), rate)[:, :n]
        speech = stems[2]
        vocals = np.repeat(speech[None], wav.shape[0], axis=0)
        background = wav - vocals if self.background_mode == "residual" else None
        if background is None:
            background = np.repeat((stems[0] + stems[1])[None], wav.shape[0], axis=0)
        out_dir.mkdir(parents=True, exist_ok=True)
        vocals_path, bg_path = out_dir / "vocals.wav", out_dir / "background.wav"
        _write_wav(vocals_path, vocals, rate)
        _write_wav(bg_path, background, rate)
        return Separation(vocals=vocals_path, background=bg_path, vocals_16k=vocals_path)
