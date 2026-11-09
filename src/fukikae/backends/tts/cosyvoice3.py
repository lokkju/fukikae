"""CosyVoice 3 as a fukikae Synthesizer: the declarations, in fukikae's environment.

The model runs in the worker (``impl_cosyvoice3``, under ``.venvs/cosyvoice3``); this class is
what the tts stage, cards and eval see. Set the worker up with
``fukikae backends setup cosyvoice3``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fukikae.backends.tts import impl_cosyvoice3 as _impl
from fukikae.backends.tts.remote import WorkerSynthesizer
from fukikae.config import ConfigError


class CosyVoice3Synthesizer(WorkerSynthesizer):
    """Fun-CosyVoice3 0.5B (``FunAudioLLM/Fun-CosyVoice3-0.5B-2512``): zero-shot or
    cross-lingual clone from the card's reference clips, with the translator's delivery
    direction through instruct mode (``impl_cosyvoice3`` has what it honours and why)."""

    name: ClassVar[str] = "cosyvoice3"
    extra: ClassVar[str] = "cosyvoice3"
    languages: ClassVar[frozenset[str]] = frozenset(_impl.LANGUAGES)
    # nvidia-smi showed 3.8 GB over the idle card on GPU 0 with five lines rendered (Task 9b
    # probe, fp32), CUDA context and allocator included.
    vram_mb: ClassVar[int] = 4500
    sources: ClassVar[frozenset[str]] = frozenset({"clone"})
    voice_keys: ClassVar[frozenset[str]] = frozenset({"seed", "rate", "emotion_strength"})
    capabilities: ClassVar[frozenset[str]] = frozenset({"instruction"})
    sample_rate: ClassVar[int] = _impl.SAMPLE_RATE
    # Characters per second after silence trimming (ruling CD7), from Task 9b's eval tts clone
    # renders (cross-lingual from Japanese clips): median per render 15.0 on Fairy Tail E11 (40
    # renders, 5 cards) and 16.4 on E12 (72 renders, 9 cards; per card 14.5 to 18.2); 15.9 is
    # the render-weighted mean of the two. From German clips (BBT de, 24 renders) it was 12.8.
    speech_cps: ClassVar[dict[str, float]] = {"en": 15.9}
    option_keys: ClassVar[frozenset[str]] = frozenset({"fp16", "mode"})
    model: ClassVar[str] = _impl.REPO
    # The model snapshot and the two code pins (CosyVoice, Matcha-TTS): any can change a voice.
    revision: ClassVar[str] = (
        f"{_impl.REVISION}+{_impl.CODE['CosyVoice'][1][:12]}+{_impl.CODE['Matcha-TTS'][1][:12]}"
    )
    model_revision: str = revision

    def __init__(self, options: dict[str, Any], device: str) -> None:
        super().__init__(options, device)
        mode = self.options.get("mode", "auto")
        if mode not in _impl.MODES:
            raise ConfigError(
                f"[tts.cosyvoice3] mode must be one of {', '.join(_impl.MODES)}, not {mode!r}"
            )
