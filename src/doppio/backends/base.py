"""Protocols every backend implements. Model libraries are never imported here."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import numpy as np

from doppio.models import Diarization, Separation, Transcript

KINDS = ("separation", "asr", "align", "diarize", "embed", "lid")


class Backend(Protocol):
    """Common surface: identity, capabilities and the load/unload lifecycle."""

    name: ClassVar[str]
    kind: ClassVar[str]
    languages: ClassVar[frozenset[str] | Literal["*"]]
    devices: ClassVar[frozenset[str]]  # subset of {"cuda", "cpu"}
    vram_mb: ClassVar[int]

    def __init__(self, options: dict[str, Any], device: str) -> None: ...

    def load(self) -> None: ...

    def unload(self) -> None:
        """Free GPU memory: drop references, gc.collect(), and torch.cuda.empty_cache()
        when torch has been imported."""
        ...


class Separator(Backend, Protocol):
    def separate(self, audio: Path, out_dir: Path) -> Separation: ...


class Transcriber(Backend, Protocol):
    supports_lid: ClassVar[bool]

    def transcribe(self, audio: Path, language: str | None) -> Transcript: ...

    def detect_language(self, audio: Path) -> tuple[str, float]:
        """Optional; only meaningful when ``supports_lid`` is true."""
        ...


class Aligner(Backend, Protocol):
    def align(self, audio: Path, transcript: Transcript) -> Transcript:
        """Return the transcript with ``word_timing=True``."""
        ...


class Diarizer(Backend, Protocol):
    def diarize(self, audio: Path) -> Diarization: ...


class Embedder(Backend, Protocol):
    dim: ClassVar[int]

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        """Shape ``(len(spans), dim)``, rows L2-normalized."""
        ...
