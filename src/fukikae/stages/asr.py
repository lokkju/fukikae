"""ASR stage: transcribe the vocals, then align words when the backend gave none."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from fukikae import runctx
from fukikae.artifacts import read_json, write_json
from fukikae.backends.base import Aligner, Transcriber
from fukikae.backends.loader import create, supports
from fukikae.config import BackendConfig, Config, ConfigError
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.models import Transcript
from fukikae.stages.language import LanguageResult
from fukikae.workspace import Episode

_sleep = time.sleep


def _table_options(cfg: Config, kind: str) -> dict[str, Any]:
    table = cfg.raw.get(kind)
    if not isinstance(table, dict):
        return {}
    return {k: v for k, v in table.items() if k not in ("backend", "lang", "pair")}


def _resolve(cfg: Config, kind: str, language: str | None, default: str) -> BackendConfig:
    try:
        return cfg.resolve(kind, language)
    except ConfigError:
        # A table with options but no backend key keeps its options.
        opts = _table_options(cfg, kind)
        if language is not None:
            override = cfg.raw.get(kind, {}).get("lang", {}).get(language)
            if isinstance(override, dict):
                opts.update({k: v for k, v in override.items() if k != "backend"})
        return BackendConfig(default, opts)


DEFAULT_OUT_DIR = "03-asr"


class AsrStage:
    def __init__(self, backend: str | None = None, out_dir: str = DEFAULT_OUT_DIR) -> None:
        self.backend = backend
        self.out_dir = out_dir
        # Another out_dir (an eval run) keeps its own manifest record, so it never
        # overwrites the pipeline's "asr" record or another backend's.
        self.name = "asr" if out_dir == DEFAULT_OUT_DIR else f"asr:{out_dir}"

    def inputs(self, ep: Episode) -> list[Path]:
        return [
            ep.work / "01-separate" / "vocals_16k.wav",
            ep.work / "02-language" / "language.json",
        ]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / self.out_dir / "transcript.json"]

    def _language(self, ep: Episode) -> str | None:
        path = self.inputs(ep)[1]
        if not path.exists():
            return None
        return read_json(path, LanguageResult).language

    def _asr_cfg(self, cfg: Config, language: str | None) -> BackendConfig:
        bc = _resolve(cfg, "asr", language, "faster_whisper")
        if self.backend is not None and self.backend != bc.backend:
            return BackendConfig(self.backend, {})
        return bc

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        lang = self._language(ep)
        asr = self._asr_cfg(cfg, lang)
        align = _resolve(cfg, "align", lang, "wav2vec2")
        return {
            "language": lang,
            "asr": {"backend": asr.backend, "options": asr.options},
            "align": {"backend": align.backend, "options": align.options},
        }

    @staticmethod
    def _gpu_wait(cfg: Config, device: str, vram_mb: int) -> None:
        if device.startswith("cuda"):
            wait_for_vram(
                cfg.gpu_device, vram_mb, runctx.vram_timeout_s, query=free_mb, sleep=_sleep
            )

    def run(self, ep: Episode, cfg: Config) -> None:
        audio = self.inputs(ep)[0]
        lang = self._language(ep) or cfg.source_language
        device = device_string(cfg)

        asr = create("asr", self._asr_cfg(cfg, lang), device)
        self._gpu_wait(cfg, device, asr.vram_mb)
        try:
            asr.load()
            transcript = cast(Transcriber, asr).transcribe(audio, lang)
        finally:
            asr.unload()

        if not transcript.word_timing:
            transcript = self._align(cfg, device, audio, transcript, lang)
        (out,) = self.outputs(ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(out, transcript)

    def _align(
        self, cfg: Config, device: str, audio: Path, transcript: Transcript, lang: str | None
    ) -> Transcript:
        language = transcript.language or lang or ""
        bc = _resolve(cfg, "align", language, "wav2vec2")
        aligner = create("align", bc, device)
        if not supports(type(aligner), language):
            base = BackendConfig(
                str(cfg.raw.get("align", {}).get("backend", "wav2vec2")),
                _table_options(cfg, "align"),
            )
            aligner = create("align", base, device)
        self._gpu_wait(cfg, device, aligner.vram_mb)
        try:
            aligner.load()
            return cast(Aligner, aligner).align(audio, transcript)
        finally:
            aligner.unload()
