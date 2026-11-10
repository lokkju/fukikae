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
    # nvidia-smi showed 4.7 GB over the idle card on GPU 0 for a bf16 render (Task 9c), CUDA
    # context and allocator included; longer prompts and lines add a little.
    vram_mb: ClassVar[int] = 6000
    sources: ClassVar[frozenset[str]] = frozenset({"clone"})
    voice_keys: ClassVar[frozenset[str]] = frozenset(
        {"seed", "temperature", "top_p", "qwen3_tts_mode"}
    )
    capabilities: ClassVar[frozenset[str]] = frozenset()
    sample_rate: ClassVar[int] = _impl.SAMPLE_RATE
    # Characters per second after silence trimming (ruling CD7), from Task 9c's eval tts clone
    # renders in the default icl mode (Japanese clips): median per render 12.3 on Fairy Tail E11
    # (40 renders, 5 cards) and 13.7 on E12 (72 renders, 9 cards; per card 11.8 to 15.3); 13.2
    # is those two medians averaged with weights 40 and 72 (not a median over all renders).
    # From German clips (BBT de, 24 renders) it was
    # 18.1; in xvector mode 14.8 (E11) and 15.3 (E12).
    speech_cps: ClassVar[dict[str, float]] = {"en": 13.2}
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
