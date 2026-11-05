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
