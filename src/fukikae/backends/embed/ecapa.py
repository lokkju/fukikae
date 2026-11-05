"""Speaker embeddings with SpeechBrain's ECAPA-TDNN (192-dim, L2-normalized)."""

from __future__ import annotations

import gc
import wave
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
_RATE = 16000
_BATCH = 16
_MIN_SAMPLES = 1600  # 0.1 s; shorter spans are zero-padded so the network gets a frame


def _read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != _RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit pcm")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0


class EcapaEmbedder:
    name: ClassVar[str] = "ecapa"
    kind: ClassVar[str] = "embed"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 1900  # measured peak 1570 MiB (nvidia-smi delta) + 20%
    dim: ClassVar[int] = 192
    extra: ClassVar[str] = "pyannote"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model: str = str(options.get("model", DEFAULT_MODEL))
        self.device = device
        self._loaded = False
        self._net: Any = None
        self.peak_vram_mb: float | None = None

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from speechbrain.inference.speaker import EncoderClassifier

        # Keep the checkpoint in the Hugging Face cache, not in ./pretrained_models.
        path = snapshot_download(self.model)
        self._net = EncoderClassifier.from_hparams(
            source=path, savedir=path, run_opts={"device": self.device}
        )
        self._loaded = True

    def unload(self) -> None:
        self._net = None
        self._loaded = False
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        """One L2-normalized row per span, shape ``(len(spans), 192)``."""
        if not self._loaded:
            raise RuntimeError("EcapaEmbedder.load() has not been called")
        if not spans:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        wav = _read_wav16k(audio)
        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        rows: list[np.ndarray] = []
        for i in range(0, len(spans), _BATCH):
            chunk = [wav[max(0, int(s * _RATE)) : int(e * _RATE)] for s, e in spans[i : i + _BATCH]]
            chunk = [np.pad(c, (0, max(0, _MIN_SAMPLES - len(c)))) for c in chunk]
            lens = torch.tensor([len(c) for c in chunk], dtype=torch.float32)
            batch = torch.zeros(len(chunk), int(lens.max()))
            for j, c in enumerate(chunk):
                batch[j, : len(c)] = torch.from_numpy(c)
            with torch.inference_mode():
                emb = self._net.encode_batch(
                    batch.to(self.device), (lens / lens.max()).to(self.device)
                )
            rows.append(emb.squeeze(1).float().cpu().numpy())
        if use_cuda:
            self.peak_vram_mb = torch.cuda.max_memory_allocated(torch.device(self.device)) / 2**20
        out = np.concatenate(rows, axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-12)).astype(np.float32)
