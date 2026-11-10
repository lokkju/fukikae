"""Config loading and per-language backend resolution."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

_DEFAULT_USER_CONFIG = object()
_RESERVED = ("backend", "lang", "pair")
# Kinds whose profile backend differs from what a workspace table's options were written for
# before profiles existed (embed was ECAPA, now WeSpeaker): a workspace table that sets
# options must name its backend, or the options would silently move to another model.
_WORKSPACE_NEEDS_BACKEND = frozenset({"embed"})
# External ids `[series]` accepts now and nothing reads yet; the character-cards plan uses them.
SERIES_ID_KEYS = ("tmdb", "tvdb", "anidb")
_SERIES_KEYS = ("title", "genre", *SERIES_ID_KEYS)
_GENRE = re.compile(r"[a-z0-9-]+")
# A BCP 47-shaped language tag (a 2-3 letter language, then dash-separated subtags), the form
# packs carry and import accepts.
LANGUAGE_TAG = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{1,8})*")


def check_language(tag: object, what: str) -> str:
    """``tag`` when it is a language tag (``en``, ``pt-BR``), else ConfigError naming
    ``what``."""
    if not isinstance(tag, str) or not LANGUAGE_TAG.fullmatch(tag):
        hint = f" (use {tag.replace('_', '-')!r})" if isinstance(tag, str) and "_" in tag else ""
        raise ConfigError(f"{what} {tag!r} is not a language tag such as 'en' or 'pt-BR'{hint}")
    return tag


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
class SeriesInfo:
    """The workspace's `[series]` table: the title and genre the results ledger records."""

    title: str
    genre: str
    ids: dict[str, str | int] = field(default_factory=dict)


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
    series: SeriesInfo = field(default_factory=lambda: SeriesInfo("", "default", {}))

    def resolve(
        self,
        kind: str,
        language: str | None = None,
        pair: str | None = None,
        key: str = "backend",
        default: str | None = None,
    ) -> BackendConfig:
        """Pick the backend for a stage: pair table over language table over base. `key`
        names the table's selector (``matcher`` for ``[characters]``); the result's
        `backend` holds its value. With no selector set anywhere, `default` is used with the
        tables' options (ConfigError when `default` is None)."""
        merged: dict[str, Any] = {}
        ws_options: list[str] = []  # option keys the workspace sets
        ws_selects = False  # whether the workspace names the backend
        # Later layers win: profile, then the workspace's own tables over it.
        for source in (self.profile_raw, self.raw):
            table = source.get(kind)
            if not isinstance(table, dict):
                continue
            base = {k: v for k, v in table.items() if k not in ("lang", "pair")}
            layers = [base]
            for group, name in (("lang", language), ("pair", pair)):
                if name is None:
                    continue
                override = table.get(group, {}).get(name)
                if isinstance(override, dict):
                    layers.append(override)
            for layer in layers:
                merged = _layer(merged, layer, key)
                if source is self.raw:
                    ws_selects = ws_selects or key in layer
                    ws_options += [k for k in layer if k not in (*_RESERVED, key)]
        backend = merged.pop(key, None) or default
        if not backend:
            raise ConfigError(f"no {key} configured for {kind!r}")
        if kind in _WORKSPACE_NEEDS_BACKEND and ws_options and not ws_selects:
            raise ConfigError(
                f"[{kind}] in the workspace config sets {', '.join(sorted(set(ws_options)))} "
                f"without `{key}`, so it would apply to the {self.profile} profile's {kind} "
                f"backend {backend!r}. Set `{key}` in that table to the backend those options "
                f'are for (e.g. {key} = "{backend}" to keep it)'
            )
        options = {k: v for k, v in merged.items() if k not in (*_RESERVED, key)}
        return BackendConfig(str(backend), options)


def _layer(merged: dict[str, Any], over: dict[str, Any], key: str = "backend") -> dict[str, Any]:
    """Apply one layer; a different backend gets only its own options (Ruling R4)."""
    if key in over and merged.get(key) not in (None, over[key]):
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


def _series(raw: Any, workspace: Path, profile: str) -> SeriesInfo:
    """`[series]`: title defaults to the workspace directory's name, genre to the profile's."""
    if raw is None:
        raw = {}
    elif isinstance(raw, str):
        # The older top-level `series = "Name"`, which also keys the registry's LDA
        # projections (registry.projection.series_of), reads as the title.
        raw = {"title": raw}
    if not isinstance(raw, dict):
        raise ConfigError(f'[series] must be a table (or the older series = "Name"), not {raw!r}')
    unknown = sorted(set(raw) - set(_SERIES_KEYS))
    if unknown:
        raise ConfigError(
            f"[series] has unknown key(s) {', '.join(unknown)}; allowed: {', '.join(_SERIES_KEYS)}"
        )
    title = raw.get("title", workspace.resolve().name)
    if not isinstance(title, str):
        raise ConfigError(f"[series] title must be a string, not {title!r}")
    genre = raw.get("genre", profile)
    if not isinstance(genre, str) or not _GENRE.fullmatch(genre):
        raise ConfigError(
            f"[series] genre {genre!r} must be lowercase letters, digits and dashes "
            '(e.g. "live-action")'
        )
    ids: dict[str, str | int] = {}
    for key in SERIES_ID_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, str | int):
            raise ConfigError(f"[series] {key} must be a string or an integer, not {value!r}")
        ids[key] = value
    return SeriesInfo(title, genre, ids)


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
        target_language=check_language(raw.get("target_language", "en"), "target_language"),
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
        series=_series(raw.get("series"), workspace, profile),
    )
