"""Language stage: pin the source language from config, or detect it from the vocals."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from doppio import runctx
from doppio.artifacts import write_json
from doppio.backends.base import Transcriber
from doppio.backends.loader import create
from doppio.config import BackendConfig, Config, ConfigError
from doppio.gpu import device_string, free_mb, wait_for_vram
from doppio.workspace import Episode

_sleep = time.sleep


class LanguageResult(BaseModel):
    language: str
    confidence: float | None
    source: str  # "config" | "detected"


class LanguageStage:
    name = "language"

    def inputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "01-separate" / "vocals_16k.wav"]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / "02-language" / "language.json"]

    @staticmethod
    def _backend_cfg(cfg: Config) -> BackendConfig:
        try:
            return cfg.resolve("lid")
        except ConfigError:
            table = cfg.raw.get("lid")
            options = (
                {k: v for k, v in table.items() if k not in ("lang", "pair")}
                if isinstance(table, dict)
                else {}
            )
            return BackendConfig("faster_whisper", options or {"model": "large-v3"})

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        if cfg.source_language:
            return {"source_language": cfg.source_language}
        bc = self._backend_cfg(cfg)
        return {"backend": bc.backend, "options": bc.options}

    def run(self, ep: Episode, cfg: Config) -> None:
        (out,) = self.outputs(ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        if cfg.source_language:
            write_json(
                out, LanguageResult(language=cfg.source_language, confidence=None, source="config")
            )
            return
        bc = self._backend_cfg(cfg)
        device = device_string(cfg)
        backend = create("lid", bc, device)
        if not getattr(backend, "supports_lid", False):
            raise ValueError(f"lid backend {bc.backend!r} does not support language detection")
        if device.startswith("cuda"):
            wait_for_vram(
                cfg.gpu_device, backend.vram_mb, runctx.vram_timeout_s, query=free_mb, sleep=_sleep
            )
        try:
            backend.load()
            lang, conf = cast(Transcriber, backend).detect_language(self.inputs(ep)[0])
        finally:
            backend.unload()
        write_json(out, LanguageResult(language=lang, confidence=conf, source="detected"))
