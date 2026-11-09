"""Shared data contract between pipeline stages. Times are float seconds."""

from pathlib import Path
from typing import Any, Literal

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
    ``budget`` is the most characters the slot can speak at ``max_speed``; ``over_budget`` and
    ``overrun`` flag a rendering left longer than that for time fitting. ``text`` is None
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
    # Set when ``text`` is still longer than ``budget`` after the retry: by how many characters.
    over_budget: bool = False
    overrun: int = 0
    # Why the line is as it is when that isn't the obvious reading, e.g. a subtitle line kept as
    # written because condensing it failed ("condense failed: ...") or came back no shorter.
    note: str | None = None


class DubCues(BaseModel):
    """The translate stage's artifact, ``06-translate/cues.<target>.json``."""

    schema_version: int = 1
    episode: str
    source_language: str
    target_language: str
    path: Literal["asr", "subs"]
    cues: list[DubCue]


class TtsPlanCue(BaseModel):
    """One cue in ``07-tts/plan.json``. A rendered cue has a ``card_id``, its cache ``key``
    (of the wav in ``07-tts``), ``seconds`` (that wav's length, silence trimmed) and ``slot``
    (the cue's length); ``rendered_seconds`` is the render at the card's own rate, before any
    rate fit, and ``rate_factor`` the speed-up the fit asked for (1.0 when none). ``passed`` is
    what the backend got beyond the card and the text, and ``neutral_only`` says the backend
    takes no delivery at all. ``skipped`` says why a cue has no audio (``skip: ...`` or
    ``error: ...`` from translate, or ``no text``). Nothing here depends on the run that wrote
    it, so the same audio always gives the same file."""

    cue_id: str
    character_id: int | None
    extra: str | None
    card_id: str | None = None
    backend: str | None = None
    key: str | None = None
    seconds: float | None = None
    rendered_seconds: float | None = None
    slot: float
    rate_factor: float = 1.0
    passed: dict[str, Any] = {}
    neutral_only: bool = False
    skipped: str | None = None


class TtsPlan(BaseModel):
    """The tts stage's artifact, ``07-tts/plan.json``."""

    schema_version: int = 1
    episode: str
    cues: list[TtsPlanCue]


class TtsRun(BaseModel):
    """What one tts run did, ``07-tts/run.json``; not a stage output, so nothing downstream
    depends on it. ``report`` holds the lines of the cards this run assigned (they are also
    appended to ``07-tts/assignments.log`` and kept as history notes)."""

    schema_version: int = 1
    episode: str
    finished: str
    renders: int
    rendered: list[str]
    cached: list[str]
    report: list[str] = []


class FitReportCue(BaseModel):
    """One placed cue in ``08-timefit/report.json``: the cue's slot, where its audio starts
    (``start``, before the slot when silence was borrowed), how long it plays (``seconds``,
    the fitted wav), the render's length and rate factor from the tts stage, the time-stretch
    applied (1.0 when none), the silence borrowed on each side, and ``over_by``, what still
    did not fit (the audio plays whole, past the borrowed silence)."""

    cue_id: str
    slot_start: float
    slot_end: float
    start: float
    seconds: float
    rendered_seconds: float
    rate_factor: float
    stretch: float
    borrowed_before: float
    borrowed_after: float
    over_by: float


class FitReport(BaseModel):
    """The timefit stage's artifact, ``08-timefit/report.json``. ``over`` lists the cues left
    longer than their slot plus the silence they could borrow; ``summary`` counts placed,
    stretched, borrowed and over cues, the median stretch of the stretched ones, and the
    worst overrun. Nothing in it depends on the run."""

    schema_version: int = 1
    episode: str
    cues: list[FitReportCue]
    over: list[str]
    summary: dict[str, Any] = {}


class ProtectedSpan(BaseModel):
    """A span where the mix keeps the original vocals untouched, and what said so:
    ``detector`` (the theme-song detector), ``subs`` (a subtitle cue the filter marked as a
    song), ``tag`` (an ASR cue the translate stage skipped as a song from tag spans),
    ``tag:singing``, ``tag:bgm`` or ``tag:laughter``. ``score`` is the detector's."""

    start: float
    end: float
    source: str
    score: float | None = None


class MixReport(BaseModel):
    """What the mix stage did, ``09-mix/report.json``. ``dropped`` maps each fitted cue that
    was not placed, because it falls in a song span, to that span's source. ``loudness`` holds
    the EBU R128 measurements (``i`` LUFS, ``lra`` LU, ``tp`` dBTP) of the original mix, the dub
    mix before normalization (``premix``) and after (``final``); ``gain_db`` is the one linear
    gain applied before the true-peak limiter."""

    schema_version: int = 1
    episode: str
    target_language: str
    songs: list[ProtectedSpan]
    laughter: list[ProtectedSpan]
    dropped: dict[str, str]
    dropped_by_source: dict[str, int] = {}
    # The theme-song detector's best window per region, accepted or not, with its scores.
    song_candidates: list[dict[str, Any]] = []
    placed: int
    removed_seconds: float
    dub_gain_db: float | None
    loudness: dict[str, dict[str, float]]
    gain_db: float | None = None
    normalization: str
