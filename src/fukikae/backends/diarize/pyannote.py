"""Speaker diarization with pyannote.audio (gated model; needs a Hugging Face token).

Tuning options, each left at the model's own value when unset:

- `clustering_threshold`: the clustering step's distance threshold (community-1's VBx uses
  it for the agglomerative initialisation, default 0.6, range 0.5 to 0.8; the 3.x
  agglomerative clustering uses it as the merge distance).
- `min_cluster_size`: the 3.x agglomerative clustering's smallest cluster; VBx pipelines
  (community-1) have no such parameter and refuse it.
- `exclude_overlap`: compute the clustering embeddings without overlapped frames
  (`embedding_exclude_overlap`; community-1 ships with it on).
"""

from __future__ import annotations

import gc
import wave
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.models import Diarization, Turn

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"
_RATE = 16000


def _read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != _RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit pcm")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0


def _number(options: dict[str, Any], key: str) -> float | None:
    v = options.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"pyannote option {key} must be a number, not {v!r}")
    return float(v)


def _count(options: dict[str, Any], key: str) -> int | None:
    v = options.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"pyannote option {key} must be a whole number of at least 1, not {v!r}")
    return v


class PyannoteDiarizer:
    name: ClassVar[str] = "pyannote"
    kind: ClassVar[str] = "diarize"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 3000  # measured peak 2498 MiB (nvidia-smi delta) + 20%
    extra: ClassVar[str] = "pyannote"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model: str = str(options.get("model", DEFAULT_MODEL))
        self.min_speakers: int | None = options.get("min_speakers")
        self.max_speakers: int | None = options.get("max_speakers")
        self.clustering_threshold = _number(options, "clustering_threshold")
        self.min_cluster_size = _count(options, "min_cluster_size")
        eo = options.get("exclude_overlap")
        if eo is not None and not isinstance(eo, bool):
            raise ValueError(f"pyannote option exclude_overlap must be true or false, not {eo!r}")
        self.exclude_overlap: bool | None = eo
        self.device = device
        self._pipeline: Any = None
        self.peak_vram_mb: float | None = None

    def load(self) -> None:
        from huggingface_hub import get_token

        token = get_token()
        if not token:
            raise RuntimeError(
                f"pyannote needs a Hugging Face token for the gated model {self.model}: "
                f"accept its terms at https://huggingface.co/{self.model} and run "
                "`huggingface-cli login` (or set HF_TOKEN)"
            )
        import torch
        from pyannote.audio import Pipeline

        try:
            pipeline = Pipeline.from_pretrained(self.model, token=token)
        except Exception as e:
            raise RuntimeError(
                f"could not load {self.model}: {e}. Accept the terms at "
                f"https://huggingface.co/{self.model} with the account that owns the token."
            ) from e
        if pipeline is None:
            raise RuntimeError(
                f"could not load {self.model}; accept its terms at https://huggingface.co/{self.model}"
            )
        self.tune(pipeline)
        self._pipeline = pipeline.to(torch.device(self.device))

    def tune(self, pipeline: Any) -> None:
        """Apply the tuning options to a loaded pipeline (nothing when none is set)."""
        changes: dict[str, Any] = {}
        if self.clustering_threshold is not None:
            changes["threshold"] = self.clustering_threshold
        if self.min_cluster_size is not None:
            changes["min_cluster_size"] = self.min_cluster_size
        if changes:
            params = pipeline.parameters(instantiated=True)
            clustering = dict(params.get("clustering", {}))
            missing = [k for k in changes if k not in clustering]
            if missing:
                raise ValueError(
                    f"{self.model}: its clustering has no {', '.join(missing)} parameter "
                    f"(it has {', '.join(sorted(clustering)) or 'none'})"
                )
            clustering.update(changes)
            pipeline.instantiate({**params, "clustering": clustering})
        if self.exclude_overlap is not None:
            pipeline.embedding_exclude_overlap = self.exclude_overlap

    def unload(self) -> None:
        self._pipeline = None
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def diarize(self, audio: Path) -> Diarization:
        import torch

        if self._pipeline is None:
            raise RuntimeError("PyannoteDiarizer.load() has not been called")
        wav = torch.from_numpy(_read_wav16k(audio))[None]
        kwargs: dict[str, Any] = {}
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        use_cuda = self.device.startswith("cuda")
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        out = self._pipeline({"waveform": wav, "sample_rate": _RATE}, **kwargs)
        if use_cuda:
            self.peak_vram_mb = torch.cuda.max_memory_allocated(torch.device(self.device)) / 2**20
        annotation = getattr(out, "speaker_diarization", out)
        turns = [
            Turn(start=float(seg.start), end=float(seg.end), speaker=str(label))
            for seg, _track, label in annotation.itertracks(yield_label=True)
        ]
        turns.sort(key=lambda t: (t.start, t.end))
        return Diarization(backend=self.name, turns=turns)
