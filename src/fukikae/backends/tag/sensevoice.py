"""Event and emotion tags from SenseVoiceSmall (funasr), on the CPU by default.

SenseVoice prefixes its transcript with rich tags, ``<|ja|><|HAPPY|><|Laughter|><|withitn|>``:
language, emotion, sound event, text normalization. This backend reads only the emotion and
the event tag. Emotions are one span per diarize turn (a turn over 30 s is split); events
come from 5 s windows with a 2.5 s hop over the vocals, adjacent windows with the same label
merged. ``<|EMO_UNKNOWN|>`` gives no emotion span. SenseVoice has no confidence score, so
spans carry none.

Event tags the model emits and this stage keeps: BGM, Laughter, Cry, Applause, and Singing
should a model revision add it. Speech, Sneeze, Breath and Cough are dropped.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar, Literal

from fukikae.backends.embed._common import RATE, cut, free_gpu, read_wav16k
from fukikae.models import Tags, TagSpan, Turn

DEFAULT_MODEL = "FunAudioLLM/SenseVoiceSmall"
DEFAULT_REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
MAX_TURN_S = 30.0
WINDOW_S = 5.0
HOP_S = 2.5
MIN_TURN_S = 0.2

EMOTIONS = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fearful",
    "disgusted": "disgusted",
    "surprised": "surprised",
}
EVENTS = {
    "bgm": "bgm",
    "singing": "singing",
    "laughter": "laughter",
    "cry": "crying",
    "applause": "applause",
}
_TAG = re.compile(r"<\|([^|>]*)\|>")


def parse_rich_tags(text: str) -> tuple[str | None, str | None]:
    """(emotion, event) from SenseVoice's tag prefix; None for unknown, absent or unlisted."""
    emotion: str | None = None
    event: str | None = None
    for raw in _TAG.findall(text):
        tag = raw.lower()
        if tag in EMOTIONS:
            emotion = EMOTIONS[tag]
        elif tag in EVENTS:
            event = EVENTS[tag]
    return emotion, event


def merge_event_windows(windows: list[tuple[float, float, str | None]]) -> list[TagSpan]:
    """One span per run of windows sharing a label; windows that overlap or touch join."""
    spans: list[TagSpan] = []
    open_by_label: dict[str, TagSpan] = {}
    for start, end, label in sorted(windows, key=lambda w: (w[0], w[1])):
        if label is None:
            continue
        cur = open_by_label.get(label)
        if cur is not None and start <= cur.end + 1e-9:
            cur.end = max(cur.end, end)
            continue
        cur = TagSpan(start=start, end=end, label=label)
        open_by_label[label] = cur
        spans.append(cur)
    return sorted(spans, key=lambda s: (s.start, s.end, s.label))


def split_turn(turn: Turn, limit: float = MAX_TURN_S) -> list[tuple[float, float]]:
    """The turn as one span, or consecutive pieces of at most `limit` seconds."""
    out: list[tuple[float, float]] = []
    start = turn.start
    while turn.end - start > limit + 1e-9:
        out.append((start, round(start + limit, 6)))
        start = round(start + limit, 6)
    out.append((start, turn.end))
    return out


class SenseVoiceTagger:
    name: ClassVar[str] = "sensevoice"
    kind: ClassVar[str] = "tag"
    languages: ClassVar[frozenset[str] | Literal["*"]] = "*"
    devices: ClassVar[frozenset[str]] = frozenset({"cpu", "cuda"})
    # D26: on the GPU by default (a small model; the CPU took most of the tag stage's 5 min);
    # the CPU when there is no usable GPU. VRAM measured on Fairy Tail E13 (see the perf report).
    default_device: ClassVar[str | None] = "cuda"
    vram_mb: ClassVar[int] = 1500
    extra: ClassVar[str] = "sensevoice"

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.model: str = str(options.get("model", DEFAULT_MODEL))
        self.revision: str = str(options.get("revision", DEFAULT_REVISION))
        self.language: str = str(options.get("language", "auto"))
        self.device = device
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from funasr import AutoModel

        from fukikae.backends.local import local_snapshot

        path = str(local_snapshot("sensevoice", self.model, self.revision))  # ruling CD24
        self._model = AutoModel(
            model=path, device=self.device, disable_update=True, disable_pbar=True
        )

    @classmethod
    def fetch(cls, options: dict[str, Any] | None = None) -> list[Path]:
        """Download the pinned snapshot (``fukikae backends setup sensevoice``)."""
        from fukikae.backends.local import fetch_snapshot

        inst = cls(options or {}, "cpu")
        return [fetch_snapshot(inst.model, inst.revision)]

    def unload(self) -> None:
        self._model = None
        free_gpu()

    def _decode(self, samples: Any) -> str:
        res = self._model.generate(
            input=samples, cache={}, language=self.language, use_itn=False, batch_size_s=60
        )
        return str(res[0]["text"]) if res else ""

    def tag(self, audio: Path, turns: list[Turn]) -> Tags:
        if self._model is None:
            raise RuntimeError("SenseVoiceTagger.tag called before load()")
        wav = read_wav16k(audio)
        duration = len(wav) / RATE

        windows: list[tuple[float, float, str | None]] = []
        start = 0.0
        while start < duration:
            end = min(start + WINDOW_S, duration)
            _, event = parse_rich_tags(self._decode(cut(wav, start, end)))
            windows.append((round(start, 6), round(end, 6), event))
            if end >= duration:
                break
            start += HOP_S

        emotions: list[TagSpan] = []
        for turn in sorted(turns, key=lambda t: (t.start, t.end)):
            for a, b in split_turn(turn):
                if b - a < MIN_TURN_S:
                    continue
                emotion, _ = parse_rich_tags(self._decode(cut(wav, a, b)))
                if emotion is not None:
                    emotions.append(TagSpan(start=a, end=b, label=emotion))
        return Tags(backend=self.name, events=merge_event_windows(windows), emotions=emotions)
