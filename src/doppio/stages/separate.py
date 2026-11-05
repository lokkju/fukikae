"""Separate stage: split extracted audio into vocals and a background bed.

This is the template for every GPU stage: resolve the backend, wait for free VRAM when the
device is CUDA, ``load()``, work, ``unload()`` in a ``finally``.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, cast

from doppio import media, runctx
from doppio.artifacts import atomic_path
from doppio.backends.base import Separator
from doppio.backends.loader import create
from doppio.config import BackendConfig, Config, ConfigError
from doppio.gpu import device_string, free_mb, wait_for_vram
from doppio.workspace import Episode

_sleep = time.sleep


class SeparateStage:
    name = "separate"

    def inputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "00-extract" / "audio.wav"]

    def outputs(self, ep: Episode) -> list[Path]:
        out = ep.work / "01-separate"
        return [out / "vocals.wav", out / "background.wav", out / "vocals_16k.wav"]

    @staticmethod
    def _backend_cfg(cfg: Config) -> BackendConfig:
        try:
            return cfg.resolve("separation")
        except ConfigError:
            return BackendConfig("demucs", {})

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        bc = self._backend_cfg(cfg)
        return {"backend": bc.backend, "options": bc.options}

    def run(self, ep: Episode, cfg: Config) -> None:
        bc = self._backend_cfg(cfg)
        device = device_string(cfg)
        backend = create("separation", bc, device)
        if device.startswith("cuda"):
            wait_for_vram(
                cfg.gpu_device,
                backend.vram_mb,
                runctx.vram_timeout_s,
                query=free_mb,
                sleep=_sleep,
            )
        final_vocals, final_bg, final_16k = self.outputs(ep)
        out_dir = final_vocals.parent
        scratch = out_dir / ".scratch"
        shutil.rmtree(scratch, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        backend.load()
        try:
            sep = cast(Separator, backend).separate(self.inputs(ep)[0], scratch)
            with atomic_path(final_vocals) as tmp:
                shutil.copyfile(sep.vocals, tmp)
            with atomic_path(final_bg) as tmp:
                shutil.copyfile(sep.background, tmp)
            with atomic_path(final_16k) as tmp:
                media.run_ffmpeg(
                    ["-i", str(final_vocals), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                     str(tmp)]
                )  # fmt: skip
        finally:
            backend.unload()
            shutil.rmtree(scratch, ignore_errors=True)
