"""Shared data contract between pipeline stages. Times are float seconds."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Word(BaseModel):
    start: float
    end: float
    text: str
    score: float | None = None


class Segment(BaseModel):
    """A stretch of transcript as the ASR backend decoded it.

    A decoding chunk, not a dubbing cue: its length is whatever the backend chose (Qwen3-ASR
    gives chunks of up to ~28 s, faster-whisper much shorter ones) and it may span several
    speakers. Cue segmentation comes later, from the word timings and the diarization turns.
    """

    start: float
    end: float
    text: str
    words: list[Word] | None = None


class Transcript(BaseModel):
    schema_version: int = 1
    language: str
    backend: str
    word_timing: bool
    segments: list[Segment]


class Turn(BaseModel):
    start: float
    end: float
    speaker: str


class Diarization(BaseModel):
    schema_version: int = 1
    backend: str
    turns: list[Turn]


class TagSpan(BaseModel):
    start: float
    end: float
    label: str
    confidence: float | None = None


class Tags(BaseModel):
    """The tag stage's artifact: sound-event spans over the vocals and one emotion span per
    diarize turn (a turn over 30 s is split)."""

    schema_version: int = 1
    backend: str
    events: list[TagSpan]
    emotions: list[TagSpan]
    emotion_backend: str | None = None  # set when a second backend scored the emotions


class Separation(BaseModel):
    vocals: Path
    background: Path
    vocals_16k: Path


class ClusterAssignment(BaseModel):
    cluster: str
    actor_id: int | None
    character_id: int | None
    score: float | None
    clean_seconds: float
    status: Literal["matched", "new", "extra", "pinned", "review"]


class Assignments(BaseModel):
    schema_version: int = 1
    episode: str
    threshold: float
    clusters: list[ClusterAssignment]
    # The turns the cluster labels refer to; the next run pins against them. Written with
    # the assignments so the two can never disagree. Absent in older files.
    turns: list[Turn] | None = None


class Delivery(BaseModel):
    """How a cue was spoken, for the tts stage: ``reference`` is the cue's span on the
    separated vocals (the emotion reference audio), ``emotion`` comes from the tag stage when
    it ran, and ``direction`` is the translator's short free-text tag for marked delivery."""

    reference: tuple[float, float]
    emotion: str | None = None
    emotion_confidence: float | None = None
    direction: str | None = None


class DubCue(BaseModel):
    """One line to dub. ``speaker`` is the registry name, ``char<id>`` or ``extra:<cluster>``;
    ``budget`` is the most characters the slot can speak at ``max_speed``. ``text`` is None
    when the cue is skipped (``skip`` names why) or failed (``error`` says how)."""

    id: str
    start: float
    end: float
    character_id: int | None
    extra: str | None
    speaker: str
    source_text: str
    literal: str | None = None
    text: str | None = None
    budget: int
    delivery: Delivery
    skip: str | None = None
    error: str | None = None


class DubCues(BaseModel):
    """The translate stage's artifact, ``06-translate/cues.<target>.json``."""

    schema_version: int = 1
    episode: str
    source_language: str
    target_language: str
    path: Literal["asr", "subs"]
    cues: list[DubCue]
