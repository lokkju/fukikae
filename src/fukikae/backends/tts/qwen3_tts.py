"""Qwen3-TTS as a fukikae Synthesizer: the declarations, in fukikae's environment.

The model runs in the worker (``impl_qwen3_tts``, under ``.venvs/qwen3tts``); this class is
what the tts stage, cards and eval see. Set the worker up with
``fukikae backends setup qwen3_tts``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fukikae.backends.tts import impl_qwen3_tts as _impl
from fukikae.backends.tts.remote import WorkerSynthesizer
from fukikae.config import ConfigError


class Qwen3TtsSynthesizer(WorkerSynthesizer):
    """Qwen3-TTS 1.7B Base (``Qwen/Qwen3-TTS-12Hz-1.7B-Base``): clone from the card's
    reference clips, with their transcripts (``icl``) or from the speaker embedding alone
    (``xvector``); no delivery input (``impl_qwen3_tts`` has why)."""

    name: ClassVar[str] = "qwen3_tts"
    extra: ClassVar[str] = "qwen3tts"
    languages: ClassVar[frozenset[str]] = frozenset(_impl.LANGUAGES)
    vram_mb: ClassVar[int] = 8000
    sources: ClassVar[frozenset[str]] = frozenset({"clone"})
    voice_keys: ClassVar[frozenset[str]] = frozenset({"seed", "temperature", "top_p"})
    capabilities: ClassVar[frozenset[str]] = frozenset()
    sample_rate: ClassVar[int] = _impl.SAMPLE_RATE
    speech_cps: ClassVar[dict[str, float]] = {"en": 15.0}
    option_keys: ClassVar[frozenset[str]] = frozenset({"mode"})
    model: ClassVar[str] = _impl.REPO
    # The model snapshot and the qwen-tts package that runs it: either can change a voice.
    revision: ClassVar[str] = f"{_impl.REVISION}+{_impl.CODE_DIST}=={_impl.CODE_VERSION}"
    model_revision: str = revision

    def __init__(self, options: dict[str, Any], device: str) -> None:
        super().__init__(options, device)
        mode = self.options.get("mode", "auto")
        if mode not in _impl.MODES:
            raise ConfigError(
                f"[tts.qwen3_tts] mode must be one of {', '.join(_impl.MODES)}, not {mode!r}"
            )
