"""Protocols every backend implements. Model libraries are never imported here."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

import numpy as np

from fukikae.models import Diarization, Separation, Tags, Transcript, Turn

if TYPE_CHECKING:
    from fukikae.cards.card import Voice
    from fukikae.translation import Rendering, TranslateRequest

KINDS = ("separation", "asr", "align", "diarize", "embed", "lid", "tag", "translate", "tts")

# Delivery inputs a Synthesizer may accept beyond the speaker voice.
TTS_CAPABILITIES = frozenset({"emotion_audio", "emotion_vector", "instruction", "duration"})


class Backend(Protocol):
    """Common surface: identity, capabilities and the load/unload lifecycle."""

    name: ClassVar[str]
    kind: ClassVar[str]
    languages: ClassVar[frozenset[str] | Literal["*"]]
    devices: ClassVar[frozenset[str]]  # subset of {"cuda", "cpu"}
    vram_mb: ClassVar[int]

    def __init__(self, options: dict[str, Any], device: str) -> None: ...

    def load(self) -> None: ...

    def unload(self) -> None:
        """Free GPU memory: drop references, gc.collect(), and torch.cuda.empty_cache()
        when torch has been imported."""
        ...


class Separator(Backend, Protocol):
    def separate(self, audio: Path, out_dir: Path) -> Separation: ...


class Transcriber(Backend, Protocol):
    supports_lid: ClassVar[bool]

    def transcribe(self, audio: Path, language: str | None) -> Transcript: ...

    def detect_language(self, audio: Path) -> tuple[str, float]:
        """Optional; only meaningful when ``supports_lid`` is true."""
        ...


class Aligner(Backend, Protocol):
    def align(self, audio: Path, transcript: Transcript) -> Transcript:
        """Return the transcript with ``word_timing=True``."""
        ...


class Diarizer(Backend, Protocol):
    def diarize(self, audio: Path) -> Diarization: ...


class Embedder(Backend, Protocol):
    dim: ClassVar[int]

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        """Shape ``(len(spans), dim)``, rows L2-normalized."""
        ...


class Tagger(Backend, Protocol):
    def tag(self, audio: Path, turns: list[Turn]) -> Tags:
        """Sound-event spans over `audio` and one emotion span per turn."""
        ...


class Translator(Backend, Protocol):
    def translate(self, req: TranslateRequest) -> dict[str, Rendering]:
        """One rendering per cue id of ``req``; ``TranslateFormatError`` when the model's
        answer is not the requested JSON."""
        ...


class Synthesizer(Backend, Protocol):
    """A TTS backend. A card's [voice] table is honored exactly or refused at load
    (``fukikae.cards.card.validate_for_backend``), never approximated.

    A backend class must subclass ``Synthesizer`` explicitly (Kokoro in Task 6 included) to
    inherit the protocol's method defaults; structural typing alone does not provide them."""

    sources: ClassVar[frozenset[str]]  # subset of cards.card.CARD_SOURCES
    voice_keys: ClassVar[frozenset[str]]  # [voice] keys it honors beyond the source's own field
    capabilities: ClassVar[frozenset[str]]  # subset of TTS_CAPABILITIES
    sample_rate: ClassVar[int]
    # Characters of text per second the backend speaks at rate 1.0, per target language,
    # measured on its renders; translate sizes each cue's budget with it (ruling CD7).
    speech_cps: ClassVar[dict[str, float]]
    model_revision: str
    # Optional, read with getattr (ruling CD14): ``default_device``, where it runs unless
    # [tts.<backend>] device says otherwise, and ``cpu_refusal``, why it must not run on the CPU.

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        """Raise ``CardError`` naming any ``[voice.options]`` key or value it can't honor,
        for a card of voice ``source`` when given."""
        ...

    @classmethod
    def capabilities_for(cls, source: str) -> frozenset[str]:  # noqa: ARG003 - per-source override
        """The delivery capabilities for a card of voice ``source``; ``capabilities`` unless a
        backend's sources differ (Qwen3-TTS takes an instruction only when designing)."""
        return cls.capabilities

    def synthesize(
        self,
        text: str,
        voice: Voice,
        language: str,
        out: Path,
        *,
        resolve: Callable[[str], Path],
        emotion_audio: Path | None = None,
        emotion_vector: dict[str, float] | None = None,
        instruction: str | None = None,
        emotion_strength: float = 1.0,
        target_seconds: float | None = None,
    ) -> float:
        """Write a mono wav at ``sample_rate`` to ``out`` and return its length in seconds.
        ``resolve`` maps a ``sha256:`` ref to its sidecar path."""
        ...
