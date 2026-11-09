"""IndexTTS2 as a fukikae Synthesizer: the declarations, in fukikae's environment.

The model runs in the worker (``impl_indextts2``, under ``.venvs/indextts2``); this class is
what the tts stage, cards and eval see. Set the worker up with
``UV_PROJECT_ENVIRONMENT=<checkout>/.venvs/indextts2 uv sync --directory
<checkout>/workers/indextts2 --frozen``.
"""

from __future__ import annotations

from typing import ClassVar

from fukikae.backends.tts import impl_indextts2 as _impl
from fukikae.backends.tts.remote import WorkerSynthesizer


class IndexTts2Synthesizer(WorkerSynthesizer):
    """IndexTTS2 (``IndexTeam/IndexTTS-2``): zero-shot clone from the card's reference clips,
    with the source line's audio as a separate emotion prompt (``impl_indextts2`` has what it
    honours and why it offers no emotion vector, instruction or duration)."""

    name: ClassVar[str] = "indextts2"
    extra: ClassVar[str] = "indextts2"
    languages: ClassVar[frozenset[str]] = frozenset(_impl.LANGUAGES)
    # Peak torch allocation 7.5 GB in fp32 on the first lines (Task 9a spike), plus the CUDA
    # context and allocator slack.
    vram_mb: ClassVar[int] = 9500
    sources: ClassVar[frozenset[str]] = frozenset({"clone"})
    voice_keys: ClassVar[frozenset[str]] = frozenset(
        {"seed", "temperature", "top_p", "emotion_strength"}
    )
    capabilities: ClassVar[frozenset[str]] = frozenset({"emotion_audio"})
    sample_rate: ClassVar[int] = _impl.SAMPLE_RATE
    # Characters per second after silence trimming (ruling CD7). Provisional: Kokoro's figure
    # until IndexTTS2's own renders are measured.
    speech_cps: ClassVar[dict[str, float]] = {"en": 15.5}
    option_keys: ClassVar[frozenset[str]] = frozenset({"fp16"})
    model: ClassVar[str] = _impl.REPO
    # The model snapshot and the code that runs it: either changing can change a voice.
    revision: ClassVar[str] = f"{_impl.REVISION}+{_impl.CODE_REVISION[:12]}"
    model_revision: str = revision
