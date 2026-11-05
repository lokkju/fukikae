"""Config loading and per-language backend resolution."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DEFAULT_USER_CONFIG = object()
_RESERVED = ("backend", "lang", "pair")


class ConfigError(Exception):
    """Raised for missing or invalid configuration."""


@dataclass(frozen=True)
class BackendConfig:
    backend: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrySettings:
    min_new_seconds: float = 20.0
    # Placeholder until Task 17 tunes it against real episodes.
    match_threshold: float = 0.6


@dataclass(frozen=True)
class Config:
    target_language: str = "en"
    source_language: str | None = None
    source: Path | None = None
    gpu_device: int = 0
    registry: RegistrySettings = field(default_factory=RegistrySettings)
    raw: dict[str, Any] = field(default_factory=dict)

    def resolve(
        self, kind: str, language: str | None = None, pair: str | None = None
    ) -> BackendConfig:
        """Pick the backend for a stage: pair table over language table over base."""
        table = self.raw.get(kind)
        if not isinstance(table, dict):
            table = {}
        merged: dict[str, Any] = {k: v for k, v in table.items() if k not in ("lang", "pair")}
        layers = (("lang", language), ("pair", pair))
        for group, key in layers:
            if key is None:
                continue
            override = table.get(group, {}).get(key)
            if not isinstance(override, dict):
                continue
            if "backend" in override and override["backend"] != merged.get("backend"):
                # A different backend gets only its own options, not the previous one's.
                merged = dict(override)
            else:
                merged.update(override)
        backend = merged.pop("backend", None)
        if not backend:
            raise ConfigError(f"no backend configured for {kind!r}")
        options = {k: v for k, v in merged.items() if k not in _RESERVED}
        return BackendConfig(str(backend), options)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e


def load_config(
    workspace: Path, user_config: Path | None | object = _DEFAULT_USER_CONFIG
) -> Config:
    """Deep-merge the user config, then the workspace's fukikae.toml over it.

    Omit ``user_config`` for ``~/.config/fukikae/config.toml``; pass None for none.
    """
    raw: dict[str, Any] = {}
    if user_config is _DEFAULT_USER_CONFIG:
        user_config = Path("~/.config/fukikae/config.toml").expanduser()
    if isinstance(user_config, Path):
        raw = _read(user_config)
    raw = _deep_merge(raw, _read(workspace / "fukikae.toml"))

    reg = raw.get("registry", {})
    defaults = RegistrySettings()
    source = raw.get("source")
    return Config(
        target_language=raw.get("target_language", "en"),
        source_language=raw.get("source_language"),
        source=workspace / Path(str(source)).expanduser() if source else None,
        gpu_device=int(raw.get("gpu_device", 0)),
        registry=RegistrySettings(
            min_new_seconds=float(reg.get("min_new_seconds", defaults.min_new_seconds)),
            match_threshold=float(reg.get("match_threshold", defaults.match_threshold)),
        ),
        raw=raw,
    )
