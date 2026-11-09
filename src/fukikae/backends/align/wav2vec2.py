"""CTC forced alignment of a transcript against audio with per-language wav2vec2 models.

The de and es checkpoints have lowercase letter vocabularies, the English bundle uppercase;
a character the vocabulary lacks is looked up case-folded (German ß has no token and is
skipped, so its word is timed from its other letters).

ja and zh models have character vocabularies, so words there are single characters. The
Korean checkpoint's vocabulary is syllables; its characters are grouped back to the
transcript's whitespace-separated words.
"""

from __future__ import annotations

import gc
import json
import unicodedata
import wave
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.models import Segment, Transcript, Word

MODELS: dict[str, str] = {
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "en": "torchaudio:WAV2VEC2_ASR_BASE_960H",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
}
_CHAR_LANGS = frozenset({"ja", "zh"})
_RATE = 16000
_PAD_S = 0.25  # audio kept around a segment so its first and last sounds are not clipped


def split_words(text: str, language: str) -> list[str]:
    """Words of a segment: whitespace-separated, or single letters/digits for ja and zh."""
    if language in _CHAR_LANGS:
        return [c for c in text if unicodedata.category(c)[0] in "LN"]
    return text.split()


def ctc_align(emission: Any, tokens: list[int], blank: int) -> list[tuple[int, int, float]] | None:
    """Viterbi-align `tokens` to a ``(1, frames, vocab)`` log-prob emission.

    Returns ``(start_frame, end_frame, score)`` per token (end exclusive), or None when the
    target cannot fit in the frames.
    """
    import torch
    import torchaudio.functional as taf

    frames = emission.shape[1]
    repeats = sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b)
    if not tokens or len(tokens) + repeats > frames:
        return None
    targets = torch.tensor([tokens], dtype=torch.int32, device=emission.device)
    try:
        labels, scores = taf.forced_align(emission, targets, blank=blank)
    except (RuntimeError, ValueError):
        return None
    spans = taf.merge_tokens(labels[0], scores[0].exp(), blank=blank)
    if len(spans) != len(tokens):
        return None
    return [(int(s.start), int(s.end), float(s.score)) for s in spans]


def _read_wav16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != _RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit pcm")
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32768.0


class Wav2Vec2Aligner:
    name: ClassVar[str] = "wav2vec2"
    kind: ClassVar[str] = "align"
    languages: ClassVar[frozenset[str] | Literal["*"]] = frozenset(MODELS)
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 3000
    extra: ClassVar[str] = "whisper"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model_override: str | None = options.get("model")
        self.device = device
        self._loaded = False
        self._lang: str | None = None
        self._net: Any = None
        self._vocab: dict[str, int] = {}
        self._blank = 0
        self._normalize = True
        self._hf = True

    @property
    def model(self) -> str:
        return self.model_override or "per-language"

    def load(self) -> None:
        """Mark ready; the checkpoint for a language is fetched when a transcript names it."""
        self._loaded = True

    def unload(self) -> None:
        self._net = None
        self._lang = None
        self._loaded = False
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_for(self, language: str) -> None:
        name = self.model_override or MODELS.get(language)
        if name is None:
            raise ValueError(f"wav2vec2 has no alignment model for language {language!r}")
        if self._lang == language and self._net is not None:
            return
        self._net = None
        net: Any
        if name.startswith("torchaudio:"):
            import torchaudio.pipelines as pipelines

            bundle = getattr(pipelines, name.split(":", 1)[1])
            net = bundle.get_model().to(self.device).eval()
            labels = bundle.get_labels()
            self._vocab = {lab: i for i, lab in enumerate(labels)}
            self._blank = self._vocab["-"]
            self._normalize, self._hf = False, False
        else:
            from huggingface_hub import hf_hub_download
            from transformers import Wav2Vec2ForCTC

            with open(hf_hub_download(name, "vocab.json"), encoding="utf-8") as f:
                self._vocab = json.load(f)
            blank = next(self._vocab[k] for k in ("<pad>", "[PAD]") if k in self._vocab)
            self._blank = int(blank)
            net = Wav2Vec2ForCTC.from_pretrained(name).to(self.device).eval()  # pyright: ignore[reportArgumentType]
            self._normalize, self._hf = True, True
        self._net = net
        self._lang = language

    def _emission(self, samples: np.ndarray) -> Any:
        import torch

        x = torch.from_numpy(samples)
        if self._normalize:
            x = (x - x.mean()) / (x.std() + 1e-7)
        x = x[None].to(self.device)
        net: Any = self._net
        with torch.inference_mode():
            logits = net(x).logits if self._hf else net(x)[0]
        return torch.log_softmax(logits.float(), dim=-1)

    def _token_id(self, ch: str, language: str) -> int | None:
        if language == "en":
            ch = ch.upper()
        tid = self._vocab.get(ch)
        if tid is None:
            tid = self._vocab.get(ch.lower())
        if tid is None or tid == self._blank:
            return None
        return int(tid)

    def align(self, audio: Path, transcript: Transcript) -> Transcript:
        if not self._loaded:
            raise RuntimeError("Wav2Vec2Aligner.load() has not been called")
        language = transcript.language
        self.load_for(language)
        wav = _read_wav16k(audio)
        duration = len(wav) / _RATE
        segments = [
            self._align_segment(wav, duration, seg, language) for seg in transcript.segments
        ]
        return Transcript(
            language=language, backend=transcript.backend, word_timing=True, segments=segments
        )

    def _align_segment(
        self, wav: np.ndarray, duration: float, seg: Segment, language: str
    ) -> Segment:
        words = split_words(seg.text, language)
        whole = [Word(start=seg.start, end=seg.end, text=w, score=None) for w in words]
        c0 = max(0.0, seg.start - _PAD_S)
        c1 = min(duration, seg.end + _PAD_S)
        crop = wav[int(c0 * _RATE) : int(c1 * _RATE)]
        if not words or len(crop) < 400:
            return seg.model_copy(update={"words": whole})
        tokens: list[int] = []
        owner: list[int] = []
        for wi, w in enumerate(words):
            for ch in w:
                tid = self._token_id(ch, language)
                if tid is not None:
                    tokens.append(tid)
                    owner.append(wi)
        if not tokens:
            return seg.model_copy(update={"words": whole})
        emission = self._emission(crop)
        spans = ctc_align(emission, tokens, self._blank)
        if spans is None:
            return seg.model_copy(update={"words": whole})
        frame_s = (len(crop) / _RATE) / emission.shape[1]
        times: dict[int, list[tuple[float, float, float]]] = {}
        for wi, (a, b, score) in zip(owner, spans, strict=True):
            times.setdefault(wi, []).append((c0 + a * frame_s, c0 + b * frame_s, score))
        out: list[Word] = []
        cursor = seg.start
        for wi, text in enumerate(words):
            ts = times.get(wi)
            if ts is None:
                out.append(Word(start=cursor, end=cursor, text=text, score=None))
                continue
            start = min(max(ts[0][0], cursor), seg.end)
            end = min(max(ts[-1][1], start), seg.end)
            out.append(Word(start=start, end=end, text=text, score=sum(t[2] for t in ts) / len(ts)))
            cursor = end
        return seg.model_copy(update={"words": out})
