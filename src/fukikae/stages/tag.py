"""Tag stage (optional): sound events and per-turn emotion from the vocals.

Reads ``01-separate/vocals_16k.wav`` and ``04-diarize/turns.json``; writes
``04b-tag/tags.json`` (a ``Tags``). Off unless ``[tag] enabled = true`` (the anime and
live-action profiles turn it on); the workspace's table wins over the profile's. Runs on the
CPU unless ``[tag] device = "cuda"``. ``enabled`` is not a backend option, so it stays out of
the cache key.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from fukikae import runctx
from fukikae.artifacts import read_json, write_json
from fukikae.backends.base import Tagger
from fukikae.backends.identity import identity_key
from fukikae.backends.loader import create
from fukikae.config import BackendConfig, Config
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.models import Diarization
from fukikae.workspace import Episode

_sleep = time.sleep


def tag_enabled(cfg: Config) -> bool:
    """`[tag] enabled`, workspace over profile; false when neither sets it."""
    value: Any = False
    for source in (cfg.profile_raw, cfg.raw):
        table = source.get("tag")
        if isinstance(table, dict) and "enabled" in table:
            value = table["enabled"]
    return value is True


class TagStage:
    name = "tag"

    def inputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "01-separate" / "vocals_16k.wav", ep.work / "04-diarize" / "turns.json"]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "04b-tag" / "tags.json"]

    @staticmethod
    def _backend_cfg(cfg: Config) -> tuple[BackendConfig, str]:
        bc = cfg.resolve("tag")
        options = {k: v for k, v in bc.options.items() if k not in ("enabled", "device")}
        return BackendConfig(bc.backend, options), str(bc.options.get("device", "cpu"))

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        bc, device = self._backend_cfg(cfg)
        return {
            "backend": bc.backend,
            "options": bc.options,
            "device": device,
            "identity": identity_key("tag", bc),
        }

    def run(self, ep: Episode, cfg: Config) -> None:
        audio, turns_path = self.inputs(ep)
        (out,) = self.outputs(ep)
        bc, want = self._backend_cfg(cfg)
        device = device_string(cfg) if want == "cuda" else "cpu"
        turns = read_json(turns_path, Diarization).turns
        backend = create("tag", bc, device)
        if device.startswith("cuda"):
            wait_for_vram(
                cfg.gpu_device, backend.vram_mb, runctx.vram_timeout_s, query=free_mb, sleep=_sleep
            )
        try:
            backend.load()
            tags = cast(Tagger, backend).tag(audio, turns)
        finally:
            backend.unload()
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, tags)
