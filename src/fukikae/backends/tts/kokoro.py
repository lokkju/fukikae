"""Kokoro-82M (hexgrad), in process: stock voices and weighted blends of them.

The model, its config and the voice tensors come from one pinned snapshot of
``hexgrad/Kokoro-82M``: the local Hugging Face cache first, a download only when it is missing
(never under ``HF_HUB_OFFLINE``). Kokoro's own ``KModel``/``KPipeline`` would fetch the newest
files instead, so they are given paths. A card's ``rate`` is Kokoro's ``speed``. A blend
(``source = "blend"``) is the weighted mean of its voices' tensors. Kokoro takes no emotion,
instruction or duration input, so every line renders neutral.

English G2P is misaki's, with spaCy's ``en_core_web_sm`` installed by the ``kokoro`` extra;
other languages need misaki's own extras (``ja``, ``zh``) or espeak-ng.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from fukikae.backends.base import Synthesizer
from fukikae.backends.embed._common import free_gpu
from fukikae.cards.card import CardError
from fukikae.config import ConfigError

if TYPE_CHECKING:
    from fukikae.cards.card import Voice

REPO = "hexgrad/Kokoro-82M"
REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
MODEL_FILE = "kokoro-v1_0.pth"
SAMPLE_RATE = 24000

# Characters per second at speed 1.0 after silence trimming: the median per-line rate over
# Fairy Tail E11's 393 translated cues in 12 stock voices (pooled 16.1, quartiles 11.1 and
# 18.6; per voice 10.1 bm_daniel to 18.6 am_fenrir). Character-cards Task 6 fix round 1.
SPEECH_CPS = {"en": 15.5}

# fukikae target language -> Kokoro's lang_code (American English for "en").
LANG_CODES = {
    "en": "a",
    "es": "e",
    "fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "ja": "j",
    "zh": "z",
}

# Every voice in the pinned snapshot's voices/ directory.
STOCK_VOICES = frozenset(
    {
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    }
)


def blend_packs(packs: Sequence[Any], weights: Sequence[float]) -> Any:
    """The weighted mean of voice tensors of one shape."""
    import torch

    total = float(sum(weights))
    if total <= 0:
        raise CardError("a Kokoro blend needs a positive total weight")
    stacked = torch.stack([p.float() for p in packs])
    w = torch.tensor([float(x) / total for x in weights]).view(-1, *([1] * (stacked.dim() - 1)))
    return (stacked * w).sum(dim=0)


class KokoroSynthesizer(Synthesizer):
    name: ClassVar[str] = "kokoro"
    kind: ClassVar[str] = "tts"
    languages: ClassVar[frozenset[str]] = frozenset(LANG_CODES)
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    default_device: ClassVar[str | None] = "cpu"  # 82M parameters: real time on the CPU (CD14)
    vram_mb: ClassVar[int] = 1500
    sources: ClassVar[frozenset[str]] = frozenset({"stock", "blend"})
    voice_keys: ClassVar[frozenset[str]] = frozenset({"rate"})
    capabilities: ClassVar[frozenset[str]] = frozenset()
    sample_rate: ClassVar[int] = SAMPLE_RATE
    stock_voices: ClassVar[frozenset[str]] = STOCK_VOICES
    speech_cps: ClassVar[dict[str, float]] = SPEECH_CPS
    model: ClassVar[str] = REPO
    revision: ClassVar[str] = REVISION
    model_revision: str = REVISION

    def __init__(self, options: dict[str, Any], device: str) -> None:
        if options:
            raise ConfigError(
                f"the kokoro TTS backend takes no options, got {', '.join(sorted(options))}"
            )
        self.device = device
        self._model: Any = None
        self._dir: Path | None = None
        self._pipelines: dict[str, Any] = {}
        self._packs: dict[str, Any] = {}

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        if options:
            raise CardError(
                f"kokoro takes no [voice.options] (a {source or 'kokoro'} card), "
                f"got {', '.join(sorted(options))}"
            )

    def _snapshot(self) -> Path:
        from huggingface_hub import constants, snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        patterns = ["config.json", MODEL_FILE, "voices/*.pt"]
        try:
            return Path(
                snapshot_download(
                    REPO, revision=REVISION, allow_patterns=patterns, local_files_only=True
                )
            )
        except LocalEntryNotFoundError:
            if constants.HF_HUB_OFFLINE:
                raise
        return Path(snapshot_download(REPO, revision=REVISION, allow_patterns=patterns))

    def load(self) -> None:
        from kokoro import KModel

        self._dir = self._snapshot()
        model = KModel(
            repo_id=REPO,
            config=str(self._dir / "config.json"),
            model=str(self._dir / MODEL_FILE),
        )
        self._model = model.to(self.device).eval()

    def unload(self) -> None:
        self._model = None
        self._pipelines.clear()
        self._packs.clear()
        free_gpu()

    def _pipeline(self, language: str) -> Any:
        code = LANG_CODES.get(language)
        if code is None:
            raise CardError(
                f"kokoro has no {language!r} voice; it speaks {', '.join(sorted(LANG_CODES))}"
            )
        if code not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[code] = KPipeline(lang_code=code, repo_id=REPO, model=self._model)
        return self._pipelines[code]

    def _pack(self, voice_id: str) -> Any:
        if voice_id not in STOCK_VOICES:
            raise CardError(f"kokoro has no stock voice {voice_id!r}")
        if voice_id not in self._packs:
            import torch

            assert self._dir is not None, "load() first"
            path = self._dir / "voices" / f"{voice_id}.pt"
            self._packs[voice_id] = torch.load(path, weights_only=True, map_location="cpu")
        return self._packs[voice_id]

    def _voice_tensor(self, voice: Voice) -> Any:
        if voice.source == "stock":
            assert voice.voice is not None
            return self._pack(voice.voice)
        if voice.source == "blend":
            ids = [i for i, _ in voice.voices]
            return blend_packs([self._pack(i) for i in ids], [w for _, w in voice.voices])
        raise CardError(f"kokoro does not support source {voice.source!r}")

    def synthesize(
        self,
        text: str,
        voice: Voice,
        language: str,
        out: Path,
        *,
        resolve: Callable[[str], Path],  # noqa: ARG002 - stock voices have no sidecars
        emotion_audio: Path | None = None,  # noqa: ARG002
        emotion_vector: dict[str, float] | None = None,  # noqa: ARG002
        instruction: str | None = None,  # noqa: ARG002
        emotion_strength: float = 1.0,  # noqa: ARG002
        target_seconds: float | None = None,  # noqa: ARG002
    ) -> float:
        from scipy.io import wavfile

        if self._model is None:
            raise RuntimeError("kokoro: load() before synthesize()")
        pipe = self._pipeline(language)
        pack = self._voice_tensor(voice)
        chunks = [
            r.audio.detach().cpu().numpy()
            for r in pipe(text, voice=pack, speed=voice.rate)
            if r.audio is not None
        ]
        audio = np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE // 10, np.float32)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        wavfile.write(out, SAMPLE_RATE, pcm)
        return len(pcm) / SAMPLE_RATE
