"""IndexTTS2 inside the TTS worker (``fukikae.backends.tts.worker``).

Runs under ``<checkout>/.venvs/indextts2`` (Python 3.11, ``workers/indextts2``), so the module
level imports only the standard library; ``fukikae.backends.tts.remote`` imports this module
in fukikae's own environment just to read the pins below.

Pins: the model snapshot ``IndexTeam/IndexTTS-2`` at ``REVISION``, the ``indextts`` code at git
commit ``CODE_REVISION`` (checked at load against the installed package's ``direct_url.json``),
and the four auxiliary models IndexTTS2 would otherwise fetch unpinned (``AUX``). All come from
the Hugging Face cache first; a download happens only when one is missing and never under
``HF_HUB_OFFLINE``. QwenEmotion (text to emotion) is not loaded: fukikae never asks for it.

What it honours (ruling recorded in the character-cards ledger, Task 9a):

- the speaker: the card's reference clips, joined with 0.2 s of silence in card order into one
  prompt of at most 15 s (IndexTTS2 reads only the first 15 s of its prompt);
- ``emotion_audio`` (the source line's span) as IndexTTS2's emotion prompt, with
  ``emotion_strength`` as its ``emo_alpha``; without it, the speaker prompt is the emotion
  prompt at alpha 1 (IndexTTS2's own default). IndexTTS2's emotion vector and emotion text
  replace the emotion prompt rather than add to it, and every cue has a source span, so they
  are not offered;
- ``seed`` (``torch.manual_seed`` before each line; 0 when the card sets none, so a render is
  reproducible), ``temperature`` and ``top_p`` (IndexTTS2's defaults, 0.8 and 0.8, when unset).

The released inference code has no duration control (its GPT's ``use_speed`` is fixed at zero
and no target token count is taken), so ``duration`` is not offered either.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import wave
from importlib import metadata
from typing import Any

REPO = "IndexTeam/IndexTTS-2"
REVISION = "740dcaff396282ffb241903d150ac011cd4b1ede"
CODE_REVISION = "d9e41aac89fd00b3d71497fddb287b7f24613712"
CHECKPOINT_FILES = [
    "config.yaml",
    "bpe.model",
    "feat1.pt",
    "feat2.pt",
    "gpt.pth",
    "s2mel.pth",
    "wav2vec2bert_stats.pt",
]
# (repo, revision, files): w2v-bert-2.0, the MaskGCT semantic codec, CAMPPlus, BigVGAN.
AUX = {
    "w2v_bert": (
        "facebook/w2v-bert-2.0",
        "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
        ["config.json", "preprocessor_config.json", "model.safetensors"],
    ),
    "semantic_codec": (
        "amphion/MaskGCT",
        "265c6cef07625665d0c28d2faafb1415562379dc",
        ["semantic_codec/model.safetensors"],
    ),
    "campplus": (
        "funasr/campplus",
        "e4b6ede7ce16997aff4ae69fbca1f0175e2afede",
        ["campplus_cn_common.bin"],
    ),
    "bigvgan": (
        "nvidia/bigvgan_v2_22khz_80band_256x",
        "633ff708ed5b74903e86ff1298cf4a98e921c513",
        ["config.json", "bigvgan_generator.pt"],
    ),
}
SAMPLE_RATE = 22050
PROMPT_RATE = 24000
PROMPT_MAX_S = 15.0
PROMPT_GAP_S = 0.2
LANGUAGES = ("en", "zh")


def _snapshot(repo: str, revision: str, files: list[str]) -> str:
    from huggingface_hub import constants, snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(
            repo, revision=revision, allow_patterns=files, local_files_only=True
        )
    except LocalEntryNotFoundError:
        if constants.HF_HUB_OFFLINE:
            raise
    return snapshot_download(repo, revision=revision, allow_patterns=files)


def installed_code_revision() -> str | None:
    """The git commit the worker env's ``indextts`` was installed from, when recorded."""
    try:
        text = metadata.distribution("indextts").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


class Impl:
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.tts: Any = None
        self._prompts: dict[tuple[str, ...], str] = {}
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def load(self, options: dict[str, Any], device: str) -> None:
        found = installed_code_revision()
        if found != CODE_REVISION:
            raise RuntimeError(
                f"the worker environment has indextts at {found!r}, and this fukikae pins "
                f"{CODE_REVISION}; re-sync it (`uv sync --frozen` in workers/indextts2)"
            )
        model_dir = _snapshot(REPO, REVISION, CHECKPOINT_FILES)
        aux = {}
        for key, (repo, rev, files) in AUX.items():
            snap = _snapshot(repo, rev, files)
            aux[key] = snap if key in ("w2v_bert", "bigvgan") else os.path.join(snap, files[0])
        from indextts.infer_v2 import IndexTTS2  # pyright: ignore[reportMissingImports]

        self.tts = IndexTTS2(
            cfg_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            use_fp16=bool(options.get("fp16", False)) and device != "cpu",
            device=device,
            use_cuda_kernel=False,
            use_qwen_emo=False,
            aux_paths=aux,
        )
        self._tmp = tempfile.TemporaryDirectory(prefix="fukikae-indextts2-")

    def unload(self) -> None:
        self.tts = None
        self._prompts.clear()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _prompt(self, clips: list[str]) -> str:
        """One speaker prompt from the reference clips, cached per clip list."""
        key = tuple(clips)
        if key in self._prompts:
            return self._prompts[key]
        import librosa
        import numpy as np

        parts: list[Any] = []
        total = 0
        limit = int(PROMPT_MAX_S * PROMPT_RATE)
        gap = np.zeros(int(PROMPT_GAP_S * PROMPT_RATE), np.float32)
        for clip in clips:
            y, _ = librosa.load(clip, sr=PROMPT_RATE, mono=True)
            if parts:
                parts.append(gap)
            parts.append(y.astype(np.float32))
            total = sum(len(p) for p in parts)
            if total >= limit:
                break
        audio = np.concatenate(parts)[:limit]
        assert self._tmp is not None
        path = os.path.join(self._tmp.name, f"prompt-{len(self._prompts)}.wav")
        _write_wav(path, audio, PROMPT_RATE)
        self._prompts[key] = path
        return path

    def synthesize(self, req: dict[str, Any]) -> float:
        if self.tts is None:
            raise RuntimeError("indextts2: load() before synthesize()")
        if req["language"] not in LANGUAGES:
            raise ValueError(f"IndexTTS2 speaks {', '.join(LANGUAGES)}, not {req['language']!r}")
        import numpy as np
        import torch

        voice = req["voice"]
        clips = [r["clip"] for r in voice.get("references", [])]
        if not clips:
            raise ValueError("IndexTTS2 needs a clone card with reference clips")
        prompt = self._prompt(clips)
        seed = int(voice.get("seed") or 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        gen: dict[str, Any] = {}
        if voice.get("temperature") is not None:
            gen["temperature"] = float(voice["temperature"])
        if voice.get("top_p") is not None:
            gen["top_p"] = float(voice["top_p"])
        emo = req.get("emotion_audio")
        result = self.tts.infer(
            spk_audio_prompt=prompt,
            text=req["text"],
            output_path=None,
            emo_audio_prompt=emo,
            emo_alpha=float(req.get("emotion_strength", 1.0)) if emo else 1.0,
            **gen,
        )
        if result is None:
            raise RuntimeError("IndexTTS2 returned no audio")
        rate, data = result
        if int(rate) != SAMPLE_RATE:
            raise RuntimeError(f"IndexTTS2 returned {rate} Hz, expected {SAMPLE_RATE}")
        pcm = np.asarray(data).reshape(-1).astype("<i2")
        with wave.open(req["out"], "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return len(pcm) / SAMPLE_RATE


def _write_wav(path: str, audio: Any, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
