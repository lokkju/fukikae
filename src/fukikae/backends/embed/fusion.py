"""Concatenate the L2-normalized embeddings of several embedders, renormalized.

Options: ``members`` (a list of backend names, at least two) and optional
``member_options`` (a table of options per member name). Members load one after another, so
peak GPU memory is the largest member's, not their sum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.backends.embed._common import l2_rows
from fukikae.backends.loader import create
from fukikae.config import BackendConfig


class FusionEmbedder:
    name: ClassVar[str] = "fusion"
    kind: ClassVar[str] = "embed"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 1900  # the largest member (ecapa)
    dim: ClassVar[int] = 0  # the sum of the members' dims; see `.dim`
    extra: ClassVar[str] = "pyannote"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        members = options.get("members")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("fusion: options.members must list at least two embedder names")
        if "fusion" in members:
            raise ValueError("fusion: members cannot include fusion")
        self.members: list[str] = [str(m) for m in members]
        self.member_options: dict[str, Any] = dict(options.get("member_options", {}))
        self.device = device
        self._loaded = False
        self.dim = 0  # type: ignore[misc]

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("FusionEmbedder.load() has not been called")
        parts: list[np.ndarray] = []
        for m in self.members:
            backend = create(
                "embed", BackendConfig(m, dict(self.member_options.get(m, {}))), self.device
            )
            backend.load()
            try:
                parts.append(l2_rows(np.asarray(backend.embed(audio, spans))))  # type: ignore[attr-defined]
            finally:
                backend.unload()
        self.dim = sum(p.shape[1] for p in parts)  # type: ignore[misc]
        if not spans:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_rows(np.concatenate(parts, axis=1))
