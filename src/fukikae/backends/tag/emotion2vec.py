"""Scored per-turn emotion from emotion2vec+ large (funasr), on the CPU by default.

One span per diarize turn (a turn over 30 s is split), each with the winning class's score
as ``confidence``. The model's classes are angry, disgusted, fearful, happy, neutral, sad,
surprised, other and unknown; a turn whose winner is ``other`` or ``unknown`` gets no span.
It writes no events (SenseVoice keeps those).

The weights are the ``emotion2vec`` organisation's Hugging Face upload of
``iic/emotion2vec_plus_large`` (the ModelScope name), pinned to a commit and fetched into the
Hugging Face cache, never into the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from fukikae.backends.embed._common import cut, free_gpu, read_wav16k
from fukikae.backends.tag.sensevoice import MIN_TURN_S, split_turn
from fukikae.models import Tags, TagSpan, Turn

DEFAULT_MODEL = "emotion2vec/emotion2vec_plus_large"
DEFAULT_REVISION = "6c303ba987b86b93193de93e34bb2b077a6bedc4"
KEPT = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"}


def best_emotion(result: dict[str, Any]) -> tuple[str, float] | None:
    """(label, score) of the top class, or None when it is other, unknown or absent.

    Labels come as ``生气/angry``; the part after the slash is the name."""
    labels = [str(x).split("/")[-1].strip().lower() for x in result.get("labels", [])]
    values = [float(x) for x in result.get("scores", [])]
    if not labels or len(labels) != len(values):
        return None
    top = max(range(len(values)), key=values.__getitem__)
    return (labels[top], values[top]) if labels[top] in KEPT else None


class Emotion2vecTagger:
    name: ClassVar[str] = "emotion2vec"
    kind: ClassVar[str] = "tag"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu", "cuda"})
    vram_mb: ClassVar[int] = 0
    extra: ClassVar[str] = "sensevoice"  # funasr, shared with SenseVoice

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model: str = str(options.get("model", DEFAULT_MODEL))
        self.revision: str = str(options.get("revision", DEFAULT_REVISION))
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel
        from huggingface_hub import snapshot_download

        path = snapshot_download(self.model, revision=self.revision)
        self._model = AutoModel(
            model=path, device=self.device, disable_update=True, disable_pbar=True
        )

    def unload(self) -> None:
        self._model = None
        free_gpu()

    def _scores(self, samples: Any) -> dict[str, Any]:
        res = self._model.generate(
            samples, granularity="utterance", extract_embedding=False, disable_pbar=True
        )
        return dict(res[0]) if res else {}

    def tag(self, audio: Path, turns: list[Turn]) -> Tags:
        if self._model is None:
            raise RuntimeError("Emotion2vecTagger.tag called before load()")
        wav = read_wav16k(audio)
        emotions: list[TagSpan] = []
        for turn in sorted(turns, key=lambda t: (t.start, t.end)):
            for a, b in split_turn(turn):
                if b - a < MIN_TURN_S:
                    continue
                best = best_emotion(self._scores(cut(wav, a, b)))
                if best is not None:
                    emotions.append(
                        TagSpan(start=a, end=b, label=best[0], confidence=round(best[1], 4))
                    )
        return Tags(backend=self.name, events=[], emotions=emotions)
