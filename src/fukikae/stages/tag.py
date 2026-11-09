"""Tag stage (optional): sound events and per-turn emotion from the vocals.

Reads ``01-separate/vocals_16k.wav`` and ``04-diarize/turns.json``; writes
``04b-tag/tags.json`` (a ``Tags``). ``[tag] backend`` makes the events (and the emotions, unless
``emotion_backend`` names a second backend, with its options in ``[tag.emotion]``: then that
backend's scored emotions replace the first's, which is handed no turns). Off unless
``[tag] enabled = true`` (the anime and live-action profiles turn it on); the workspace's table
wins over the profile's. Runs on the CPU unless ``[tag] device = "cuda"``. ``enabled`` is not a
backend option, so it stays out of the cache key.
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
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.gpulock import gpu_job
from fukikae.models import Diarization, Tags, Turn
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
    def _backend_cfg(cfg: Config) -> tuple[BackendConfig, BackendConfig | None, str]:
        """(events backend, emotion backend or None, device); `device` serves both."""
        bc = cfg.resolve("tag")
        skip = ("enabled", "device", "emotion_backend", "emotion")
        options = {k: v for k, v in bc.options.items() if k not in skip}
        device = str(bc.options.get("device", "cpu"))
        name = bc.options.get("emotion_backend")
        if name is not None and not str(name).strip():
            raise ConfigError(
                "[tag] emotion_backend is empty; leave the key out to use no second backend"
            )
        emotion = None
        if name:
            sub = bc.options.get("emotion")
            emotion = BackendConfig(str(name), dict(sub) if isinstance(sub, dict) else {})
        return BackendConfig(bc.backend, options), emotion, device

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        bc, ebc, device = self._backend_cfg(cfg)
        return {
            "backend": bc.backend,
            "options": bc.options,
            "device": device,
            "identity": identity_key("tag", bc),
            "emotion_backend": ebc.backend if ebc else None,
            "emotion_options": ebc.options if ebc else None,
            "emotion_identity": identity_key("tag", ebc) if ebc else None,
        }

    @staticmethod
    def _run_one(
        bc: BackendConfig, device: str, cfg: Config, audio: Path, turns: list[Turn]
    ) -> Tags:
        backend = create("tag", bc, device)
        with gpu_job(cfg.gpu_device, device, f"tag {bc.backend}"):  # ruling CD15
            if device.startswith("cuda"):
                wait_for_vram(
                    cfg.gpu_device, backend.vram_mb, runctx.vram_timeout_s, query=free_mb,
                    sleep=_sleep,
                )  # fmt: skip
            try:
                backend.load()
                return cast(Tagger, backend).tag(audio, turns)
            finally:
                backend.unload()

    def run(self, ep: Episode, cfg: Config) -> None:
        audio, turns_path = self.inputs(ep)
        (out,) = self.outputs(ep)
        bc, ebc, want = self._backend_cfg(cfg)
        device = device_string(cfg) if want == "cuda" else "cpu"
        turns = read_json(turns_path, Diarization).turns
        # With a separate emotion backend the events backend decodes no turns.
        tags = self._run_one(bc, device, cfg, audio, [] if ebc else turns)
        if ebc:
            scored = self._run_one(ebc, device, cfg, audio, turns)
            tags = tags.model_copy(
                update={"emotions": scored.emotions, "emotion_backend": scored.backend}
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, tags)
