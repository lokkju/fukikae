"""Speaker embeddings with WavLM-base-plus-sv (x-vector head, 512-dim, L2-normalized)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.backends.embed._common import RATE, cut, free_gpu, l2_rows, read_wav16k

DEFAULT_MODEL = "microsoft/wavlm-base-plus-sv"


class WavlmSvEmbedder:
    name: ClassVar[str] = "wavlm_sv"
    kind: ClassVar[str] = "embed"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 1500
    dim: ClassVar[int] = 512
    extra: ClassVar[str] = "wavlm"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model: str = str(options.get("model", DEFAULT_MODEL))
        self.device = device
        self._loaded = False
        self._net: Any = None
        self._fe: Any = None
        self.peak_vram_mb: float | None = None

    def load(self) -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self._fe = AutoFeatureExtractor.from_pretrained(self.model)
        self._net = WavLMForXVector.from_pretrained(self.model).eval().to(self.device)  # pyright: ignore[reportArgumentType]
        self._loaded = True

    def unload(self) -> None:
        self._net = None
        self._fe = None
        self._loaded = False
        free_gpu()

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        """One L2-normalized row per span, shape ``(len(spans), 512)``."""
        if not self._loaded:
            raise RuntimeError("WavlmSvEmbedder.load() has not been called")
        if not spans:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        wav = read_wav16k(audio)
        dev = torch.device(self.device)
        use_cuda = dev.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        rows: list[np.ndarray] = []
        for s, e in spans:
            inp = self._fe(
                [cut(wav, s, e)], sampling_rate=RATE, return_tensors="pt", padding=True
            ).to(dev)
            with torch.inference_mode():
                rows.append(self._net(**inp).embeddings.float().cpu().numpy())
        if use_cuda:
            self.peak_vram_mb = torch.cuda.max_memory_allocated(dev) / 2**20
        return l2_rows(np.concatenate(rows, axis=0))
