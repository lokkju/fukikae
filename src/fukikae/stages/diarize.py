"""Diarize stage: split the vocals into speaker turns and embed each speaker's clean speech.

Two models run in sequence, never together: the diarizer produces ``turns.json``, is
unloaded, then the embedder is loaded to embed each cluster's clean spans.

Outputs under ``04-diarize/``:

``turns.json``
    A ``Diarization``.
``clusters.npz``
    ``labels``: array of the cluster labels in order of first appearance. For each label
    ``L``: ``L/embeddings`` ``(n, dim)`` float32, L2-normalized rows; ``L/spans`` ``(n, 2)``
    float64 start/end seconds, one row per embedding, longest first; ``L/clean_seconds`` a
    0-d float, the total length of *all* the cluster's clean spans (before the cap below).
    A cluster with no clean span has ``n == 0``.

A clean span is the speaker's turn minus any overlap with another speaker's turn, shrunk by
0.1 s at each end and at least 1.5 s long. At most ``MAX_SPANS_PER_CLUSTER`` (50) spans are
embedded per cluster (the longest), so a long episode costs a bounded number of passes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from fukikae import runctx
from fukikae.artifacts import atomic_path, write_json
from fukikae.backends.base import Diarizer, Embedder
from fukikae.backends.identity import identity_key
from fukikae.backends.loader import create
from fukikae.config import BackendConfig, Config
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.gpulock import gpu_job
from fukikae.models import Turn
from fukikae.stages.asr import _resolve
from fukikae.workspace import Episode

_sleep = time.sleep
MAX_SPANS_PER_CLUSTER = 50


def clean_spans(
    turns: list[Turn], speaker: str, min_len: float = 1.5, pad: float = 0.1
) -> list[tuple[float, float]]:
    """The speaker's turns that overlap no other speaker, shrunk by `pad` at each end,
    kept when at least `min_len` long afterwards."""
    others = [t for t in turns if t.speaker != speaker]
    out: list[tuple[float, float]] = []
    for t in turns:
        if t.speaker != speaker:
            continue
        if any(o.start < t.end and t.start < o.end for o in others):
            continue
        start, end = round(t.start + pad, 6), round(t.end - pad, 6)
        if end - start >= min_len - 1e-9:
            out.append((start, end))
    return out


def diarize_key(cfg: Config) -> dict[str, Any]:
    """The ``diarize`` part of the stage key: the resolved backend, its options and its model
    identity. The identity ledger rows record the same dict."""
    diar = _resolve(cfg, "diarize", None, "pyannote")
    return {
        "backend": diar.backend,
        "options": diar.options,
        "identity": identity_key("diarize", diar),
    }


class DiarizeStage:
    name = "diarize"

    def inputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "01-separate" / "vocals_16k.wav"]

    def outputs(self, ep: Episode) -> list[Path]:
        out = ep.work / "04-diarize"
        return [out / "turns.json", out / "clusters.npz"]

    @staticmethod
    def _cfgs(cfg: Config) -> tuple[BackendConfig, BackendConfig]:
        # Every profile names its embedder (Ruling R3), so no fallback: a missing or
        # ambiguous [embed] is a ConfigError, never a silent ECAPA.
        return _resolve(cfg, "diarize", None, "pyannote"), cfg.resolve("embed")

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        _, emb = self._cfgs(cfg)
        return {
            "diarize": diarize_key(cfg),
            "embed": {
                "backend": emb.backend,
                "options": emb.options,
                "identity": identity_key("embed", emb),
            },
        }

    @staticmethod
    def _gpu_wait(cfg: Config, device: str, vram_mb: int) -> None:
        if device.startswith("cuda"):
            wait_for_vram(
                cfg.gpu_device, vram_mb, runctx.vram_timeout_s, query=free_mb, sleep=_sleep
            )

    def run(self, ep: Episode, cfg: Config) -> None:
        # Ruling CD15: on a GPU the stage holds the card's lock for its whole life.
        with gpu_job(cfg.gpu_device, device_string(cfg), self.name):
            self._run(ep, cfg)

    def _run(self, ep: Episode, cfg: Config) -> None:
        audio = self.inputs(ep)[0]
        turns_path, clusters_path = self.outputs(ep)
        device = device_string(cfg)
        diar_cfg, emb_cfg = self._cfgs(cfg)
        turns_path.parent.mkdir(parents=True, exist_ok=True)

        diarizer = create("diarize", diar_cfg, device)
        self._gpu_wait(cfg, device, diarizer.vram_mb)
        try:
            diarizer.load()
            diarization = cast(Diarizer, diarizer).diarize(audio)
        finally:
            diarizer.unload()
        write_json(turns_path, diarization)

        labels: list[str] = []
        for t in diarization.turns:
            if t.speaker not in labels:
                labels.append(t.speaker)
        spans: dict[str, list[tuple[float, float]]] = {}
        clean_seconds: dict[str, float] = {}
        for label in labels:
            clean = clean_spans(diarization.turns, label)
            clean_seconds[label] = float(sum(e - s for s, e in clean))
            spans[label] = sorted(clean, key=lambda s: s[1] - s[0], reverse=True)[
                :MAX_SPANS_PER_CLUSTER
            ]

        embedder = create("embed", emb_cfg, device)
        flat = [s for label in labels for s in spans[label]]
        dim = cast(Embedder, embedder).dim
        vectors = np.zeros((0, dim), dtype=np.float32)
        if flat:
            self._gpu_wait(cfg, device, embedder.vram_mb)
            try:
                embedder.load()
                vectors = cast(Embedder, embedder).embed(audio, flat)
            finally:
                embedder.unload()

        arrays: dict[str, Any] = {"labels": np.array(labels, dtype=str)}
        at = 0
        for label in labels:
            n = len(spans[label])
            arrays[f"{label}/embeddings"] = vectors[at : at + n].astype(np.float32)
            arrays[f"{label}/spans"] = np.array(spans[label], dtype=np.float64).reshape(n, 2)
            arrays[f"{label}/clean_seconds"] = np.array(clean_seconds[label], dtype=np.float64)
            at += n
        with atomic_path(clusters_path) as tmp:
            np.savez(tmp, **arrays)
