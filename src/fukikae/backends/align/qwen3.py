"""Qwen3-ForcedAligner: word timing from the model's timestamp tokens.

The model takes up to 5 minutes of audio per call, so a transcript is aligned in groups of
whole segments no longer than 300 s. Within a group all segment texts go in one call and the
returned words are handed back to the segments by count.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from fukikae.backends.asr.qwen3 import code_to_name, read_wav16k
from fukikae.models import Segment, Transcript, Word

log = logging.getLogger(__name__)

_RATE = 16000
MAX_GROUP_S = 300.0
_PAD_S = 0.2
# Aligner languages (the model card's 11), as ISO 639-1.
LANGUAGES = frozenset({"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"})


def group_segments(segments: list[Segment], max_s: float = MAX_GROUP_S) -> list[list[Segment]]:
    """Consecutive segments grouped so each group spans at most ``max_s`` seconds.

    Cuts fall only on segment boundaries; a single segment longer than ``max_s`` is its own
    group (and logged), since it cannot be split without losing its text.
    """
    groups: list[list[Segment]] = []
    for s in segments:
        if s.end - s.start > max_s:
            log.warning(
                "segment %.1f-%.1f s is %.0f s long, over the aligner's %.0f s limit",
                s.start,
                s.end,
                s.end - s.start,
                max_s,
            )
        if groups and s.end - groups[-1][0].start <= max_s:
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


class Qwen3ForcedAligner:
    name: ClassVar[str] = "qwen3_forced_aligner"
    kind: ClassVar[str] = "align"
    languages: ClassVar[frozenset[str] | Literal["*"]] = LANGUAGES
    devices: ClassVar[frozenset[str]] = frozenset({"cuda", "cpu"})
    vram_mb: ClassVar[int] = 3200  # measured peak 2.3 GB torch + context, + 20%
    extra: ClassVar[str] = "qwen"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model_name: str = str(options.get("model", "Qwen/Qwen3-ForcedAligner-0.6B"))
        self.device = device
        self._net: Any = None

    @property
    def model(self) -> str:
        return self.model_name

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ForcedAligner as Net

        cuda = self.device.startswith("cuda")
        self._net = Net.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16 if cuda else torch.float32,
            device_map=self.device if cuda else "cpu",
        )

    def unload(self) -> None:
        self._net = None
        gc.collect()
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def align(self, audio: Path, transcript: Transcript) -> Transcript:
        if self._net is None:
            raise RuntimeError("Qwen3ForcedAligner.load() has not been called")
        language = transcript.language
        if language not in LANGUAGES:
            raise ValueError(f"qwen3_forced_aligner does not support language {language!r}")
        lang_name = code_to_name(language)
        wav = read_wav16k(audio)
        out: list[Segment] = []
        for group in group_segments(transcript.segments):
            out.extend(self._align_group(wav, group, lang_name))
        return Transcript(
            language=language, backend=transcript.backend, word_timing=True, segments=out
        )

    def _words(self, text: str, lang_name: str) -> list[str]:
        words, _ = self._net.aligner_processor.encode_timestamp(text, lang_name)
        return list(words)

    def _align_group(self, wav: np.ndarray, group: list[Segment], lang_name: str) -> list[Segment]:
        duration = len(wav) / _RATE
        c0 = max(0.0, group[0].start - _PAD_S)
        c1 = min(duration, max(s.end for s in group) + _PAD_S)
        crop = wav[int(c0 * _RATE) : int(c1 * _RATE)]
        counts = [len(self._words(s.text, lang_name)) for s in group]
        if sum(counts) == 0 or len(crop) < 400:
            return [s.model_copy(update={"words": []}) for s in group]
        text = " ".join(s.text for s in group)
        items = self._net.align(audio=[(crop, _RATE)], text=[text], language=[lang_name])[0]
        if len(items) != sum(counts):
            # Joined text tokenised differently from its parts; align each segment alone.
            return [self._align_alone(wav, s, lang_name) for s in group]
        out: list[Segment] = []
        i = 0
        for s, n in zip(group, counts, strict=True):
            out.append(_with_words(s, items[i : i + n], c0))
            i += n
        return out

    def _align_alone(self, wav: np.ndarray, seg: Segment, lang_name: str) -> Segment:
        duration = len(wav) / _RATE
        c0 = max(0.0, seg.start - _PAD_S)
        c1 = min(duration, seg.end + _PAD_S)
        crop = wav[int(c0 * _RATE) : int(c1 * _RATE)]
        if not self._words(seg.text, lang_name) or len(crop) < 400:
            return seg.model_copy(update={"words": []})
        items = self._net.align(audio=[(crop, _RATE)], text=[seg.text], language=[lang_name])[0]
        return _with_words(seg, list(items), c0)


def _with_words(seg: Segment, items: list[Any], offset: float) -> Segment:
    """Words clamped inside the segment and kept in order."""
    words: list[Word] = []
    cursor = seg.start
    for it in items:
        start = min(max(offset + float(it.start_time), cursor), seg.end)
        end = min(max(offset + float(it.end_time), start), seg.end)
        words.append(Word(start=start, end=end, text=str(it.text), score=None))
        cursor = end
    return seg.model_copy(update={"words": words})
