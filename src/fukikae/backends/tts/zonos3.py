"""ZONOS3-Mini as a fukikae Synthesizer: the declarations, in fukikae's environment.

Spike backend (lokkju/fukikae#43, ``docs/results/2026-11-13-zonos3-mini-spike.md``). The model
runs in the worker (``impl_zonos3``, under ``.venvs/zonos3``); set it up with
``fukikae backends setup zonos3``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from fukikae.backends.tts import impl_zonos3 as _impl
from fukikae.backends.tts.remote import WorkerSynthesizer
from fukikae.config import ConfigError


class Zonos3Synthesizer(WorkerSynthesizer):
    """ZONOS3-Mini (``Zyphra/ZONOS3-Mini``) on Zyphra's plain-PyTorch reference: clone from the
    card's reference clips, as a raw reference prefix (``mode = "reference"``) or a MECT-B2
    speaker embedding (``"embedding"``). No capability is declared (CD20): the spike's emotion
    and pace arms reach the worker through ``emotion_vector`` and ``pace`` without one."""

    name: ClassVar[str] = "zonos3"
    extra: ClassVar[str] = "zonos3"
    languages: ClassVar[frozenset[str]] = frozenset(_impl.LANGUAGES)
    # nvidia-smi peak on GPU 0 during the spike's eval runs, CUDA context included.
    vram_mb: ClassVar[int] = 14000
    sources: ClassVar[frozenset[str]] = frozenset({"clone"})
    voice_keys: ClassVar[frozenset[str]] = frozenset({"seed", "temperature", "top_p"})
    capabilities: ClassVar[frozenset[str]] = frozenset()
    sample_rate: ClassVar[int] = _impl.SAMPLE_RATE
    # Placeholder until measured (CD7 asks for a measured figure; the spike's eval rows have it).
    speech_cps: ClassVar[dict[str, float]] = {"en": 14.0}
    option_keys: ClassVar[frozenset[str]] = frozenset({"mode", "dtype", "pace", "dose", "decoder"})
    cpu_refusal: ClassVar[str | None] = (
        "ZONOS3-Mini encodes reference clips with its dual codec, which needs a CUDA device."
    )
    model: ClassVar[str] = _impl.REPO
    revision: ClassVar[str] = f"{_impl.REVISION}+{_impl.CODE_REVISION[:12]}"
    model_revision: str = revision

    def __init__(self, options: dict[str, Any], device: str) -> None:
        super().__init__(options, device)
        mode = self.options.get("mode", "reference")
        if mode not in _impl.MODES:
            raise ConfigError(
                f"[tts.zonos3] mode must be one of {', '.join(_impl.MODES)}, not {mode!r}"
            )
        dtype = self.options.get("dtype", "bf16")
        if dtype not in _impl.DTYPES:
            raise ConfigError(
                f"[tts.zonos3] dtype must be one of {', '.join(_impl.DTYPES)}, not {dtype!r}"
            )
