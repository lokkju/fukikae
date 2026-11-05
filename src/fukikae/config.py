"""Config loading and per-language backend resolution."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
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
    profile: str = "default"
    profile_raw: dict[str, Any] = field(default_factory=dict)

    def resolve(
        self, kind: str, language: str | None = None, pair: str | None = None
    ) -> BackendConfig:
        """Pick the backend for a stage: pair table over language table over base."""
        merged: dict[str, Any] = {}
        # Later layers win: profile, then the workspace's own tables over it.
        for source in (self.profile_raw, self.raw):
            table = source.get(kind)
            if not isinstance(table, dict):
                continue
            base = {k: v for k, v in table.items() if k not in ("lang", "pair")}
            merged = _layer(merged, base)
            for group, key in (("lang", language), ("pair", pair)):
                if key is None:
                    continue
                override = table.get(group, {}).get(key)
                if isinstance(override, dict):
                    merged = _layer(merged, override)
        backend = merged.pop("backend", None)
        if not backend:
            raise ConfigError(f"no backend configured for {kind!r}")
        options = {k: v for k, v in merged.items() if k not in _RESERVED}
        return BackendConfig(str(backend), options)


def _layer(merged: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Apply one layer; a different backend gets only its own options (Ruling R4)."""
    if "backend" in over and merged.get("backend") not in (None, over["backend"]):
        return dict(over)
    return {**merged, **over}


def profile_names() -> list[str]:
    return sorted(
        p.name.removesuffix(".toml")
        for p in resources.files("fukikae.profiles").iterdir()
        if p.name.endswith(".toml")
    )


def load_profile(name: str) -> dict[str, Any]:
    """Read a profile shipped as package data; unknown names list the available ones."""
    if name not in profile_names():
        raise ConfigError(f"unknown profile {name!r}; available: {', '.join(profile_names())}")
    text = resources.files("fukikae.profiles").joinpath(f"{name}.toml").read_text("utf-8")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"profile {name}: {e}") from e


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

    profile = str(raw.get("profile", "default"))
    profile_raw = load_profile(profile)
    reg = {**profile_raw.get("registry", {}), **raw.get("registry", {})}
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
        profile=profile,
        profile_raw=profile_raw,
    )
