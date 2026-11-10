"""Segment embeddings, computed once per (segments, embedder, audio) and kept on disk.

The key is the sha256 of the segment spans, the embedder's backend name and options, and the
sha256 of the audio file's bytes, hashed in full on every lookup: a re-separation that
rewrites the audio, or any changed embedder option, misses (Review Focus 4). Segment labels
are not in the key, so relabeling the same spans hits. Files live in
``<workspace>/.fukikae/cache/embeddings/<episode>/<key>.npy``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import cast

import numpy as np

from fukikae import runctx
from fukikae.artifacts import atomic_path
from fukikae.backends.base import Backend, Embedder
from fukikae.backends.identity import identity_key
from fukikae.backends.loader import create
from fukikae.config import BackendConfig, Config
from fukikae.eval.identity.segments import Segment
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.gpulock import gpu_job

KEY_VERSION = 2

EmbedFn = Callable[[BackendConfig, Path, list[tuple[float, float]]], np.ndarray]


def cache_root(workspace: Path) -> Path:
    return Path(workspace) / ".fukikae" / "cache" / "embeddings"


def audio_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _spans(segs: Sequence[Segment]) -> list[tuple[float, float]]:
    return [(float(s.start), float(s.end)) for s in segs]


class EmbeddingCache:
    def __init__(self, root: Path, embed: EmbedFn | None = None) -> None:
        self.root = Path(root)
        self.embed = embed
        self.hits = 0
        self.misses = 0

    def key(self, segs: Sequence[Segment], embedder_cfg: BackendConfig, audio: Path) -> str:
        payload = {
            "version": KEY_VERSION,
            "segments": [[round(a, 6), round(b, 6)] for a, b in _spans(segs)],
            "embedder": {
                "backend": embedder_cfg.backend,
                "options": embedder_cfg.options,
                "identity": identity_key("embed", embedder_cfg),
            },
            "audio": audio_hash(audio),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()

    def get_or_compute(
        self,
        episode: str,
        segs: Sequence[Segment],
        embedder_cfg: BackendConfig,
        audio: Path,
        embed: EmbedFn | None = None,
    ) -> np.ndarray:
        """One embedding row per segment, from the cache or computed with `embed` (else the
        cache's own) and stored."""
        if not segs:
            return np.zeros((0, 0), dtype=np.float32)
        path = self.root / episode / f"{self.key(segs, embedder_cfg, audio)}.npy"
        if path.is_file():
            try:
                rows = np.load(path, allow_pickle=False)
            except (OSError, ValueError):
                rows = None
            if rows is not None and rows.ndim == 2 and len(rows) == len(segs):
                self.hits += 1
                return rows
        fn = embed or self.embed
        if fn is None:
            raise ValueError("embedding cache miss and no embed function given")
        rows = np.asarray(fn(embedder_cfg, Path(audio), _spans(segs)), dtype=np.float32)
        if rows.ndim != 2 or len(rows) != len(segs):
            raise ValueError(
                f"{embedder_cfg.backend}: {len(segs)} spans gave embeddings of shape {rows.shape}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(path) as tmp, tmp.open("wb") as f:
            np.save(f, rows, allow_pickle=False)
        self.misses += 1
        return rows


class ResidentEmbedder:
    """An embed function that keeps one embedder loaded across calls, the way the stages run
    models: on the configured GPU (`device_string`), after `wait_for_vram` with the run's
    `runctx.vram_timeout_s`, and never two at once: a call for another embedder unloads the
    current one first. Use as a context manager; leaving it unloads."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device: str | None = None
        self._backend: Backend | None = None
        self._key: str | None = None
        self._lock = contextlib.ExitStack()  # the GPU lock while a model is loaded (CD15)

    def __enter__(self) -> ResidentEmbedder:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        backend, self._backend, self._key = self._backend, None, None
        try:
            if backend is not None:
                backend.unload()
        finally:
            self._lock.close()

    def _ensure(self, emb_cfg: BackendConfig) -> Backend:
        key = json.dumps([emb_cfg.backend, emb_cfg.options], sort_keys=True, default=str)
        if self._backend is not None and self._key == key:
            return self._backend
        self.close()
        if self.device is None:
            self.device = device_string(self.cfg)
        backend = create("embed", emb_cfg, self.device)
        self._lock.enter_context(gpu_job(self.cfg.gpu_device, self.device, "eval embed"))
        try:
            if self.device.startswith("cuda"):
                wait_for_vram(
                    self.cfg.gpu_device, backend.vram_mb, runctx.vram_timeout_s, query=free_mb
                )
            backend.load()
        except BaseException:
            try:
                backend.unload()
            finally:
                self._lock.close()
            raise
        self._backend, self._key = backend, key
        return backend

    def __call__(
        self, emb_cfg: BackendConfig, audio: Path, spans: list[tuple[float, float]]
    ) -> np.ndarray:
        backend = self._ensure(emb_cfg)
        return cast(Embedder, backend).embed(audio, spans)
