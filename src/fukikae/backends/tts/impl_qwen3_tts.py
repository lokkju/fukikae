"""Qwen3-TTS inside the TTS worker (``fukikae.backends.tts.worker``).

Runs under ``<checkout>/.venvs/qwen3tts`` (Python 3.12, ``workers/qwen3tts``), so the module
level imports only the standard library; ``fukikae.backends.tts.qwen3_tts`` imports this module
in fukikae's own environment to read the pins and to use the pure helpers below.

Pins: the model snapshot ``Qwen/Qwen3-TTS-12Hz-1.7B-Base`` at ``REVISION`` (its speech tokenizer
is the ``speech_tokenizer/`` folder of the same snapshot), and the ``qwen-tts`` package at
``CODE_VERSION`` (by sha256 in ``workers/qwen3tts/uv.lock``, and checked again at every load
against the installed distribution). ``fetch`` (``fukikae backends setup qwen3_tts``) is the
only step that downloads; ``load`` reads the Hugging Face cache with ``local_files_only`` and
hands qwen-tts the snapshot's local directory, so it never resolves a repository name (the
worker runs with ``HF_HUB_OFFLINE=1`` as well). Qwen3-TTS needs no external text normalizer.

What it honours (the class in ``fukikae.backends.tts.qwen3_tts`` declares the same):

- the speaker: the card's reference clips joined in card order, with 0.2 s of silence, into
  one prompt of whole clips up to ``PROMPT_MAX_S``, with their transcripts joined the same
  way. The prompt (speech codes and speaker embedding) is computed once per card and reused;
- the mode (option ``mode``): ``icl`` gives the model the prompt's transcript and speech codes,
  so it continues the speaker; ``xvector`` gives it only the speaker embedding. ``auto`` (the
  default) is ``icl`` when every clip has a transcript;
- ``seed`` (``random``, numpy and torch seeded before each line; 0 when the card sets none),
  ``temperature`` and ``top_p`` (the model's generation config when unset). The sub-talker
  (the codec's residual codebooks) samples with the model's own settings.

No delivery input is honoured. The Base model's public API takes no instruction (only the
CustomVoice and VoiceDesign models do), and Task 9c's A/B of an instruction pushed in through
the model's ``instruct_ids`` is recorded in the task report; there is no emotion audio, emotion
vector, speed or duration input.
"""

from __future__ import annotations

import random
import wave
from importlib import metadata
from typing import Any

REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
]
CODE_DIST = "qwen-tts"
CODE_VERSION = "0.1.1"
SAMPLE_RATE = 24000
PROMPT_MAX_S = 15.0
PROMPT_GAP_S = 0.2
# ISO 639-1 -> the model's language names (config.json talker_config.codec_language_id)
LANGUAGE_NAMES = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}
LANGUAGES = tuple(LANGUAGE_NAMES)
MODES = ("auto", "icl", "xvector")
# The Qwen tokenizer's added tokens at REVISION that aren't in the <|...|> form the worker
# strips for every backend (ruling CD20): <think>, <tool_call>, <tool_response> (and their
# closing tags) and the TTS tokens <tts_pad>, <tts_text_bos>, <tts_text_bos_single> and
# <tts_text_eod>. The worker strips these from text, transcripts and instructions too.
CONTROL_MARKUP: tuple[str, ...] = (
    r"</?\s*(?:think|tool_call|tool_response)\s*>",
    r"<\s*tts_[a-z_]*\s*>",
)

SETUP = "fukikae backends setup qwen3_tts"


# ---- pure helpers (also used in fukikae's environment, by the tests) -----------------------


def resolve_mode(mode: str, transcripts: list[str]) -> str:
    """``icl`` or ``xvector``: ``auto`` is ``icl`` when every clip has a transcript."""
    if mode not in MODES:
        raise ValueError(f"qwen3_tts mode must be one of {', '.join(MODES)}, not {mode!r}")
    have_all = bool(transcripts) and all(t.strip() for t in transcripts)
    if mode == "auto":
        return "icl" if have_all else "xvector"
    if mode == "icl" and not have_all:
        raise ValueError("icl mode needs a transcript for every reference clip")
    return mode


def language_name(code: str) -> str:
    try:
        return LANGUAGE_NAMES[code]
    except KeyError:
        raise ValueError(f"Qwen3-TTS speaks {', '.join(LANGUAGES)}, not {code!r}") from None


def prompt_plan(refs: list[tuple[str, float, str]]) -> tuple[list[str], str]:
    """(clips, joined transcript) of the whole clips, in card order, that fit ``PROMPT_MAX_S``
    with ``PROMPT_GAP_S`` between them; the first clip is always taken (cut to fit later).
    ``refs`` is (clip, seconds, transcript)."""
    clips: list[str] = []
    texts: list[str] = []
    total = 0.0
    for clip, seconds, text in refs:
        need = seconds + (PROMPT_GAP_S if clips else 0.0)
        if clips and total + need > PROMPT_MAX_S:
            break
        clips.append(clip)
        texts.append(text.strip())
        total += need
    return clips, " ".join(t for t in texts if t)


def installed_code_version() -> str | None:
    try:
        return metadata.version(CODE_DIST)
    except metadata.PackageNotFoundError:
        return None


def check_code() -> None:
    found = installed_code_version()
    if found != CODE_VERSION:
        raise RuntimeError(
            f"the worker environment has {CODE_DIST} {found!r}, and this fukikae pins "
            f"{CODE_VERSION}; set it up again with `{SETUP}`"
        )


# ---- the worker side -------------------------------------------------------------------------


def _snapshot() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(
            REPO, revision=REVISION, allow_patterns=MODEL_FILES, local_files_only=True
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"{REPO}@{REVISION[:12]} is not in the Hugging Face cache; download it with `{SETUP}`"
        ) from e


def fetch() -> None:
    """The setup step: the pinned model snapshot into the Hugging Face cache."""
    import sys

    from huggingface_hub import snapshot_download

    print(f"fetching {REPO}@{REVISION[:12]}", file=sys.stderr, flush=True)
    snapshot_download(REPO, revision=REVISION, allow_patterns=MODEL_FILES)


class Impl:
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.tts: Any = None
        self.mode = "auto"
        self._prompts: dict[tuple[str, ...], Any] = {}

    def load(self, options: dict[str, Any], device: str) -> None:
        self.mode = str(options.get("mode", "auto"))
        if self.mode not in MODES:
            raise ValueError(f"qwen3_tts mode must be one of {', '.join(MODES)}")
        check_code()
        model_dir = _snapshot()
        import torch
        from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tts = Qwen3TTSModel.from_pretrained(
            model_dir, device_map=device, dtype=dtype, local_files_only=True
        )
        if self.tts.model.tts_model_type != "base":
            raise RuntimeError(f"{REPO} loaded as {self.tts.model.tts_model_type}, not base")

    def unload(self) -> None:
        self.tts = None
        self._prompts.clear()
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _prompt(self, refs: list[dict[str, Any]], mode: str) -> Any:
        """The card's voice-clone prompt (computed once per clip list and mode)."""
        key = (mode, *(f"{r['clip']}\0{r.get('transcript') or ''}" for r in refs))
        if key in self._prompts:
            return self._prompts[key]
        import librosa
        import numpy as np

        timed = [
            (r["clip"], float(librosa.get_duration(path=r["clip"])), r.get("transcript") or "")
            for r in refs
        ]
        clips, transcript = prompt_plan(timed)
        parts: list[Any] = []
        gap = np.zeros(int(PROMPT_GAP_S * SAMPLE_RATE), np.float32)
        for clip in clips:
            y, _ = librosa.load(clip, sr=SAMPLE_RATE, mono=True)
            if parts:
                parts.append(gap)
            parts.append(y.astype(np.float32))
        audio = np.concatenate(parts)[: int(PROMPT_MAX_S * SAMPLE_RATE)]
        items = self.tts.create_voice_clone_prompt(
            ref_audio=(audio, SAMPLE_RATE),
            ref_text=transcript if mode == "icl" else None,
            x_vector_only_mode=mode == "xvector",
        )
        self._prompts[key] = items
        return items

    def _seed(self, seed: int) -> None:
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def synthesize(self, req: dict[str, Any]) -> float:
        if self.tts is None:
            raise RuntimeError("qwen3_tts: load() before synthesize()")
        language = language_name(req["language"])
        voice = req["voice"]
        refs = list(voice.get("references", []))
        if not refs:
            raise ValueError("Qwen3-TTS needs a clone card with reference clips")
        mode = resolve_mode(self.mode, [r.get("transcript") or "" for r in refs])
        prompt = self._prompt(refs, mode)
        gen: dict[str, Any] = {}
        if voice.get("temperature") is not None:
            gen["temperature"] = float(voice["temperature"])
        if voice.get("top_p") is not None:
            gen["top_p"] = float(voice["top_p"])
        self._seed(int(voice.get("seed") or 0))
        wavs, rate = self.tts.generate_voice_clone(
            text=req["text"], language=language, voice_clone_prompt=prompt, **gen
        )
        if int(rate) != SAMPLE_RATE:
            raise RuntimeError(f"Qwen3-TTS returned {rate} Hz, expected {SAMPLE_RATE}")
        import numpy as np

        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Qwen3-TTS returned no audio")
        _write_wav(req["out"], audio, SAMPLE_RATE)
        return len(audio) / SAMPLE_RATE


def _write_wav(path: str, audio: Any, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
