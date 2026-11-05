"""Demucs source separation: vocals versus the sum of every other stem."""

from __future__ import annotations

import gc
import wave
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.models import Separation


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read 16-bit pcm wav as float32 ``(channels, samples)``."""
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit pcm, got {w.getsampwidth() * 8}-bit")
        rate, ch = w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return (raw.reshape(-1, ch).T.astype(np.float32) / 32768.0), rate


def _write_wav(path: Path, data: np.ndarray, rate: int) -> None:
    pcm = (np.clip(data, -1.0, 1.0).T * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class DemucsSeparator:
    name: ClassVar[str] = "demucs"
    kind: ClassVar[str] = "separation"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 6000
    extra: ClassVar[str] = "demucs"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model_name: str = str(options.get("model", "htdemucs_ft"))
        self.background_mode: str = str(options.get("background", "residual"))
        if self.background_mode not in ("residual", "stems"):
            raise ValueError(
                f"background must be 'residual' or 'stems', got {self.background_mode!r}"
            )
        self.device = device
        self._model: Any = None
        self.peak_vram_mb: float | None = None

    @property
    def model(self) -> str:
        return self.model_name

    def load(self) -> None:
        from demucs.pretrained import get_model

        self._model = get_model(self.model_name)
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def separate(self, audio: Path, out_dir: Path) -> Separation:
        """Write ``vocals.wav`` and ``background.wav`` to `out_dir`.

        Option ``background``: ``residual`` (default, mix minus vocals) or ``stems`` (the
        non-vocal stems summed).

        The returned ``vocals_16k`` is a placeholder (the 44.1 kHz vocals path); the stage
        derives the 16 kHz mono file.
        """
        import torch
        from demucs.apply import apply_model

        if self._model is None:
            raise RuntimeError("DemucsSeparator.load() has not been called")
        model = self._model
        wav, rate = _read_wav(audio)
        if rate != model.samplerate:
            raise ValueError(f"{audio}: {rate} Hz, model needs {model.samplerate} Hz")
        if wav.shape[0] == 1:
            wav = np.repeat(wav, model.audio_channels, axis=0)
        x = torch.from_numpy(wav)
        ref = x.mean(0)
        x = (x - ref.mean()) / (ref.std() + 1e-8)
        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        with torch.no_grad():
            out = apply_model(
                model,
                x[None],
                device=self.device,
                shifts=1,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        out = out * (ref.std() + 1e-8) + ref.mean()
        if use_cuda:
            self.peak_vram_mb = torch.cuda.max_memory_allocated(torch.device(self.device)) / 2**20
        sources: list[str] = list(model.sources)
        stems = out.cpu().numpy()
        vi = sources.index("vocals")
        if self.background_mode == "stems":
            background = sum(stems[i] for i in range(len(sources)) if i != vi)
        else:
            # htdemucs_ft is a bag of per-stem specialists whose stems sum to the mix only to
            # about -19 dB (htdemucs: -37 dB). The mix minus vocals keeps everything vocals
            # did not take, so vocals + background rebuilds the input.
            background = wav - stems[vi]
        out_dir.mkdir(parents=True, exist_ok=True)
        vocals_path, bg_path = out_dir / "vocals.wav", out_dir / "background.wav"
        _write_wav(vocals_path, stems[vi], rate)
        _write_wav(bg_path, np.asarray(background), rate)
        return Separation(vocals=vocals_path, background=bg_path, vocals_16k=vocals_path)
