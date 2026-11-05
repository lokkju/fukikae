"""Speaker embeddings with pyannote's WeSpeaker ResNet34 (256-dim, L2-normalized)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.backends.embed._common import cut, free_gpu, l2_rows, read_wav16k

DEFAULT_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


class WespeakerEmbedder:
    name: ClassVar[str] = "wespeaker"
    kind: ClassVar[str] = "embed"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 1200  # measured peak in the report; see docs/results
    dim: ClassVar[int] = 256
    extra: ClassVar[str] = "pyannote"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model: str = str(options.get("model", DEFAULT_MODEL))
        self.device = device
        self._loaded = False
        self._net: Any = None
        self.peak_vram_mb: float | None = None

    def load(self) -> None:
        import torch
        from pyannote.audio import Model

        net = Model.from_pretrained(self.model)
        if net is None:
            raise RuntimeError(f"cannot load {self.model}; accept its terms on huggingface.co")
        self._net = net.eval().to(torch.device(self.device))
        self._loaded = True

    def unload(self) -> None:
        self._net = None
        self._loaded = False
        free_gpu()

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        """One L2-normalized row per span, shape ``(len(spans), 256)``."""
        if not self._loaded:
            raise RuntimeError("WespeakerEmbedder.load() has not been called")
        if not spans:
            return np.zeros((0, self.dim), dtype=np.float32)
        import torch

        wav = read_wav16k(audio)
        dev = torch.device(self.device)
        use_cuda = dev.type == "cuda"
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        rows: list[np.ndarray] = []
        for s, e in spans:  # one at a time: the network pools over the whole input
            x = torch.from_numpy(cut(wav, s, e)).reshape(1, 1, -1).to(dev)
            with torch.inference_mode():
                rows.append(self._net(x).float().cpu().numpy())
        if use_cuda:
            self.peak_vram_mb = torch.cuda.max_memory_allocated(dev) / 2**20
        return l2_rows(np.concatenate(rows, axis=0))
