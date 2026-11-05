"""Find backends through the ``doppio.backends.<kind>`` entry point groups."""

from __future__ import annotations

import importlib.metadata
from typing import Any, cast

from doppio.backends.base import Backend
from doppio.config import BackendConfig, ConfigError

# Backend name -> extra that installs it. Read without importing the backend, so the
# hint survives a failed import. Each backend task adds its own line.
EXTRAS: dict[str, str] = {
    "demucs": "demucs",
    "faster_whisper": "whisper",
    "wav2vec2": "whisper",
    "qwen3_asr": "qwen",
    "qwen3_forced_aligner": "qwen",
    "pyannote": "pyannote",
    "ecapa": "pyannote",
}


class BackendUnavailable(Exception):  # noqa: N818
    """A backend is registered but cannot be imported (its extra is not installed)."""


def _entry_points(kind: str) -> dict[str, Any]:
    eps = importlib.metadata.entry_points(group=f"doppio.backends.{kind}")
    return {ep.name: ep for ep in eps}


def names(kind: str) -> list[str]:
    """Registered backend names for a kind, without importing any of them."""
    return sorted(_entry_points(kind))


def _load(kind: str, name: str, ep: Any) -> type[Backend]:
    try:
        return cast(type[Backend], ep.load())
    except ImportError as e:
        extra = EXTRAS.get(name)
        hint = f"pip install 'doppio[{extra}]'" if extra else "install its dependencies"
        raise BackendUnavailable(f"{kind} backend {name!r} cannot be imported ({e}); {hint}") from e


def available(kind: str) -> dict[str, type[Backend]]:
    """Backends of a kind that import cleanly; ones whose extra is missing are left out."""
    out: dict[str, type[Backend]] = {}
    for name, ep in _entry_points(kind).items():
        try:
            out[name] = _load(kind, name, ep)
        except BackendUnavailable:
            continue
    return out


def create(kind: str, cfg: BackendConfig, device: str) -> Backend:
    eps = _entry_points(kind)
    ep = eps.get(cfg.backend)
    if ep is None:
        installed = ", ".join(sorted(eps)) or "none"
        raise ConfigError(f"unknown {kind} backend {cfg.backend!r}; installed: {installed}")
    return _load(kind, cfg.backend, ep)(dict(cfg.options), device)


def supports(cls: type[Backend], language: str) -> bool:
    return cls.languages == "*" or language in cls.languages
