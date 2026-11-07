"""Speaker embeddings from ONNX models run by sherpa-onnx (CPU): CAM++, ERes2NetV2, TitaNet-L.

The three backends differ only in the model file. The files are downloaded once from the
sherpa-onnx ``speaker-recongition-models`` release (the spelling is upstream's) into
``$XDG_CACHE_HOME/fukikae/models`` (``~/.cache/fukikae/models``), or taken from the ``path``
option. They run on the CPU whatever device is passed; nothing is held on the GPU.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.artifacts import atomic_path
from fukikae.backends.embed._common import RATE, cut, l2_rows, read_wav16k

RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"


def models_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "fukikae" / "models"


def fetch_model(filename: str) -> Path:
    """The model file in the cache, downloaded on first use."""
    dest = models_dir() / filename
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(f"{RELEASE}/{filename}", timeout=120) as r,  # noqa: S310
        atomic_path(dest) as tmp,
        tmp.open("wb") as f,
    ):
        while block := r.read(1 << 20):
            f.write(block)
    return dest


class SherpaEmbedder:
    """Base class; subclasses set `name`, `dim` and `default_file`."""

    name: ClassVar[str]
    kind: ClassVar[str] = "embed"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu", "cuda"})
    vram_mb: ClassVar[int] = 0
    dim: ClassVar[int]
    extra: ClassVar[str] = "sherpa"
    default_file: ClassVar[str]

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.file: str = str(options.get("file", self.default_file))
        self.path: str | None = str(options["path"]) if "path" in options else None
        self.threads: int = int(options.get("threads", 4))
        self.device = device
        self._loaded = False
        self._ex: Any = None
        self.peak_vram_mb: float | None = 0.0

    def load(self) -> None:
        import sherpa_onnx

        model = Path(self.path) if self.path else fetch_model(self.file)
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model), num_threads=self.threads, provider="cpu"
        )
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._loaded = True

    def unload(self) -> None:
        self._ex = None
        self._loaded = False

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        """One L2-normalized row per span, shape ``(len(spans), dim)``."""
        if not self._loaded:
            raise RuntimeError(f"{type(self).__name__}.load() has not been called")
        if not spans:
            return np.zeros((0, self.dim), dtype=np.float32)
        wav = read_wav16k(audio)
        rows: list[list[float]] = []
        for s, e in spans:
            stream = self._ex.create_stream()
            stream.accept_waveform(sample_rate=RATE, waveform=cut(wav, s, e))
            stream.input_finished()
            rows.append(self._ex.compute(stream))
        return l2_rows(np.asarray(rows, dtype=np.float32))


class CampplusEmbedder(SherpaEmbedder):
    name: ClassVar[str] = "campplus"
    dim: ClassVar[int] = 512
    default_file: ClassVar[str] = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"


class Eres2netv2Embedder(SherpaEmbedder):
    name: ClassVar[str] = "eres2netv2"
    dim: ClassVar[int] = 192
    default_file: ClassVar[str] = "3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx"


class TitanetEmbedder(SherpaEmbedder):
    name: ClassVar[str] = "titanet"
    dim: ClassVar[int] = 192
    default_file: ClassVar[str] = "nemo_en_titanet_large.onnx"
