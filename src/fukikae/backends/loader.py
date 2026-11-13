"""Find backends through the ``fukikae.backends.<kind>`` entry point groups."""

from __future__ import annotations

import importlib.metadata
from typing import Any, cast

from fukikae.backends.base import Backend
from fukikae.config import BackendConfig, ConfigError

# Backend name -> extra that installs it. Read without importing the backend, so the
# hint survives a failed import. Each backend task adds its own line.
EXTRAS: dict[str, str] = {
    "demucs": "demucs",
    "dialogue": "dialogue",
    "faster_whisper": "whisper",
    "wav2vec2": "whisper",
    "qwen3_asr": "qwen",
    "qwen3_forced_aligner": "qwen",
    "pyannote": "pyannote",
    "ecapa": "pyannote",
    "wespeaker": "pyannote",
    "wavlm_sv": "wavlm",
    "campplus": "sherpa",
    "eres2netv2": "sherpa",
    "titanet": "sherpa",
    "fusion": "pyannote",
    "sensevoice": "sensevoice",
    "emotion2vec": "sensevoice",
    "kokoro": "kokoro",
    # Out-of-process backends import without their model; a missing worker environment
    # (`.venvs/<extra>`, from `workers/<extra>`) is reported at load with the setup command.
    "indextts2": "indextts2",
    "indextts2_trt": "indextts2_trt",
    "cosyvoice3": "cosyvoice3",
    "qwen3_tts": "qwen3tts",
    "zonos3": "zonos3",
}


class BackendUnavailable(Exception):  # noqa: N818
    """A backend is registered but cannot be imported (its extra is not installed)."""


def _entry_points(kind: str) -> dict[str, Any]:
    eps = importlib.metadata.entry_points(group=f"fukikae.backends.{kind}")
    return {ep.name: ep for ep in eps}


def names(kind: str) -> list[str]:
    """Registered backend names for a kind, without importing any of them."""
    return sorted(_entry_points(kind))


def _load(kind: str, name: str, ep: Any) -> type[Backend]:
    try:
        return cast(type[Backend], ep.load())
    except ImportError as e:
        extra = EXTRAS.get(name)
        hint = (
            f"install the extra: `uv sync --extra {extra}` in a checkout, "
            f"or `uv tool install 'fukikae[{extra}]'`"
            if extra
            else "install its dependencies"
        )
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


def backend_class(kind: str, name: str) -> type[Backend]:
    """The backend class registered under `name`, without instantiating it."""
    eps = _entry_points(kind)
    ep = eps.get(name)
    if ep is None:
        installed = ", ".join(sorted(eps)) or "none"
        raise ConfigError(f"unknown {kind} backend {name!r}; installed: {installed}")
    return _load(kind, name, ep)


def engine_problem(kind: str, name: str, where: str) -> str | None:
    """Why ``name`` can't be named at ``where`` (``[tts] backend``, say): it is an engine that
    renders another backend's cards (``card_backend``: indextts2_trt renders indextts2's) and
    makes none of its own. None for a backend of its own, or one that doesn't load here
    (whatever uses it reports that)."""
    try:
        cls = backend_class(kind, name)
    except Exception:  # noqa: BLE001 - unknown or broken here: its user reports that
        return None
    card_backend = getattr(cls, "card_backend", None)
    if not card_backend or card_backend == name:
        return None
    return (
        f"{where}: {name!r} renders {card_backend!r} cards and makes none of its own; name "
        f'{card_backend!r} there, and set [{kind}] engines = {{ {card_backend} = "{name}" }} to '
        f"render those cards with {name!r}"
    )


def create(kind: str, cfg: BackendConfig, device: str) -> Backend:
    eps = _entry_points(kind)
    ep = eps.get(cfg.backend)
    if ep is None:
        installed = ", ".join(sorted(eps)) or "none"
        raise ConfigError(f"unknown {kind} backend {cfg.backend!r}; installed: {installed}")
    return _load(kind, cfg.backend, ep)(dict(cfg.options), device)


def supports(cls: type[Backend], language: str) -> bool:
    return cls.languages == "*" or language in cls.languages
