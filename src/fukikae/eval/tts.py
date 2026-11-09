"""`fukikae eval tts`: score TTS renders per character and write results-ledger rows.

A sample of each character's translated cues (``sample_cues``; extras, skipped, errored and
empty cues, and slots under ``--min-slot``, default 1.0 s, left out and counted as
``excluded_short``) is rendered with the character's card at the card's own rate, with no
rate fit and no time-stretch, through the tts stage's own ``prepare_cue`` and ``render_cue``
(ruling CD12), so check, key, delivery, trim and gain are the stage's (a render already in
``.fukikae/tts-cache`` is copied, not made again). Renders live in
``<ws>/.fukikae/eval/tts/<backend>-<revision>/`` as ``<episode>-<cue-id>.<key>.wav``: the
key in the name means a swapped card or changed text renders again instead of reusing a stale
file.

Metrics, per character and overall:

- ``speaker_similarity``: WeSpeaker cosine between each render and the normalized mean of the
  character's references: the clips ``cards.select.select_references`` picks from the sampled
  episodes (``similarity_source = "references"``), the same spans a clone card of this run is
  cut from, for every backend, so stock and clone renders are on one scale (the plan's
  decision). A character with too little clean speech for that is compared with the clips of
  its card when it has some (``"card"``), else with its three longest diarize turns no other
  speaker overlaps (``"original"``). Across languages the original voice is another actor in another
  language, so stock numbers are low by construction; they are a floor, not a verdict.
  Each source has its own metric (``speaker_similarity_references``, ``_card``, ``_original``);
  ``speaker_similarity``, the report's headline, exists only when every sampled character
  shares one source, so the two kinds are never pooled.
- ``wer``: word error rate of an English faster-whisper large-v3 pass over the render against
  the cue's text, both normalized by ``fukikae.eval.asr.normalize``; pooled over words, each
  cue's errors capped at its reference words (ruling CD16), with ``wer_median``, the median
  per-cue WER, beside it. The pass is deterministic (``ASR``: temperature 0 with no fallback,
  beam 5, no conditioning on earlier text, the hallucination guard; no VAD, so short lines are
  not cut), so the same renders always score the same. It is read
  against ``wer_floor``, the same ASR on natural English speech against its subtitle text
  (``--floor-subs``): the original track at the sampled cues when there is a reference
  workspace, else a fixed clip set (``--floor-workspace``, ``--floor-episode``). The floor
  carries the subtitles' own condensation and OCR errors, so a render WER near it is as good
  as this ASR can tell.
- ``fit_rate``: share of renders no longer than their slot before any rate fit or stretch;
  ``stretch_median`` and ``stretch_p90`` of the speed-up each would need (1.0 when it fits);
  ``fit_budget_rate``: share within ``[timefit] max_speedup``.
- ``emotion_agreement`` (when the tag stage labelled the source lines): share of the cues
  where both the source line and the render carry a label (same backend for both) whose
  labels match; an unlabelled render is not a miss. Coverage of each side (labelled share of
  the sampled cues) is recorded; under 50% on either side the agreement is kept out of the
  metrics and ``emotion_low_coverage`` is set. Episodes tagged by different backends are
  refused. A backend that takes no delivery (Kokoro) renders neutral by design, so its
  agreement tracks the neutral share of the source labels (noted in the row).
- per card (ruling CD8): ``cps_median`` and quartiles, characters of text per second of the
  trimmed render, for per-card translate budgets later.

With ``--backend`` naming a backend that clones and has no stock voices (IndexTTS2), each
sampled character gets a clone card built from those references (``clone_card``; mode
``clone``), put in the card store and never assigned; a character with too little clean speech
is skipped and noted, not rendered with a stock voice. A backend with stock voices renders
stock cards (mode ``stock``), as before. The CD5 warning ("references may carry emotion")
is a note of the run and of the row.

With ``--reference-workspace`` (BBT: the English original track's workspace, on the same
timeline as the dubbed track), each cue's slot is matched to the English words whose midpoint
falls inside it, within 0.1 s (``reference_span``; ``ref_unmatched`` counts cues with none),
and the row adds ``ref_duration_ratio`` (median render
seconds over the original line's spoken seconds), ``ref_emotion_agreement`` (the render's
emotion against the original line's, both by the tag backend) and ``ref_speaker_similarity``
(the render against the original actor at that line).

Models load one at a time, each unloaded before the next: the TTS backend, WeSpeaker,
faster-whisper, then the tagger (on the CPU, as the tag stage runs it).
"""

from __future__ import annotations

import contextlib
import math
import random
import shutil
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from fukikae import runctx
from fukikae.artifacts import atomic_path, read_json
from fukikae.backends import loader
from fukikae.backends.identity import identity_key, model_identity
from fukikae.cards.card import Card
from fukikae.cards.select import (
    RefPick,
    clone_card,
    gender_of,
    reference_report,
    select_references,
    speaker_f0,
)
from fukikae.cards.stock import pick_stock, revision, stock_card
from fukikae.cards.store import CardStore
from fukikae.config import BackendConfig, Config
from fukikae.eval.asr import normalize
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.gpulock import gpu_job
from fukikae.models import Assignments, DubCue, DubCues, Tags, Transcript, Turn, Word
from fukikae.workspace import Workspace

EMBEDDER = BackendConfig("wespeaker", {})
# Ruling CD16: a deterministic pass. One temperature (no fallback sampling, which is unseeded),
# beam 5, no conditioning on the previous window, and the hallucination guard (a repetition
# loop over silence is dropped). No VAD, so short lines are not cut.
ASR = BackendConfig(
    "faster_whisper",
    {
        "model": "large-v3",
        "vad_filter": False,
        "temperature": 0.0,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
    },
)
DEFAULT_TAGGER = "emotion2vec"
EXEMPLARS = 3
EXEMPLAR_MIN_S = 1.0
MIN_SLOT_S = 1.0  # shorter slots say more about the slot than the voice
REF_PAD_S = 0.1  # a reference word's midpoint may sit this far outside the slot
COVERAGE_MIN = 0.5  # emotion agreement is a headline only above this coverage on both sides
FLOOR_CLIPS = 40
FLOOR_MIN_S = 1.0
FLOOR_MIN_WORDS = 3
RATE_16K = 16000


# ---- sampling -------------------------------------------------------------------------------


def eligible(c: DubCue) -> bool:
    """A cue the tts stage renders for a registry character."""
    return (
        c.character_id is not None
        and c.skip is None
        and c.error is None
        and bool(c.text and c.text.strip())
    )


def count_short(cues: Mapping[str, DubCues], min_slot: float = MIN_SLOT_S) -> int:
    """Eligible cues left out of the sample for a slot under ``min_slot``."""
    return sum(
        1 for d in cues.values() for c in d.cues if eligible(c) and c.end - c.start < min_slot
    )


def sample_cues(
    cues: Mapping[str, DubCues],
    per_character: int = 8,
    seed: int = 0,
    min_slot: float = MIN_SLOT_S,
) -> list[tuple[str, DubCue]]:
    """Up to ``per_character`` eligible cues per character over every episode, slots of at
    least ``min_slot``, drawn with a generator seeded by ``seed`` and the character, returned
    in (episode, start) order."""
    by_char: dict[int, list[tuple[str, DubCue]]] = {}
    for episode in sorted(cues):
        for c in sorted(cues[episode].cues, key=lambda c: (c.start, c.id)):
            if eligible(c) and c.end - c.start >= min_slot:
                by_char.setdefault(cast(int, c.character_id), []).append((episode, c))
    out: list[tuple[str, DubCue]] = []
    for cid, items in sorted(by_char.items()):
        if len(items) <= per_character:
            out += items
        else:
            rng = random.Random(f"{seed}:{cid}")  # noqa: S311 - a reproducible sample
            out += [items[i] for i in sorted(rng.sample(range(len(items)), per_character))]
    return sorted(out, key=lambda p: (p[0], p[1].start, p[1].id))


# ---- metrics --------------------------------------------------------------------------------


def duration_fit(seconds: float, slot: float) -> tuple[bool, float]:
    """(fits its slot before rate fit or stretch, the speed-up it would need: 1.0 when it
    fits, ``inf`` for an empty slot)."""
    if slot <= 0:
        return False, math.inf
    ratio = seconds / slot
    return ratio <= 1.0, max(1.0, ratio)


def _words_in(transcript: Transcript, slot: tuple[float, float], pad: float) -> list[Word]:
    lo, hi = slot[0] - pad, slot[1] + pad
    return [
        w
        for seg in transcript.segments
        for w in (seg.words or [])
        if lo <= (w.start + w.end) / 2 <= hi
    ]


def reference_span(
    transcript: Transcript, slot: tuple[float, float], pad: float = REF_PAD_S
) -> tuple[float, float] | None:
    """First start to last end of the words whose midpoint lies in the slot widened by
    ``pad``; None when there are none."""
    words = _words_in(transcript, slot, pad)
    if not words:
        return None
    return min(w.start for w in words), max(w.end for w in words)


def reference_speech(
    transcript: Transcript, slot: tuple[float, float], pad: float = REF_PAD_S
) -> float | None:
    """Seconds the original line takes inside the slot (``reference_span``'s length)."""
    span = reference_span(transcript, slot, pad)
    return None if span is None else span[1] - span[0]


def similarity(render_vecs: np.ndarray, ref_vecs: np.ndarray) -> float:
    """Mean cosine of the render rows to the normalized mean of the reference rows."""
    mean = np.asarray(ref_vecs, dtype=np.float64).mean(axis=0)
    mean /= max(float(np.linalg.norm(mean)), 1e-12)
    r = np.asarray(render_vecs, dtype=np.float64)
    r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12)
    return float((r @ mean).mean())


def word_errors(hyp: str, ref: str, language: str) -> tuple[int, int]:
    """(substitutions + deletions + insertions, capped at the reference words; reference words)
    after normalization. The cap (ruling CD16) keeps one looping hypothesis from outweighing
    the rest of a pooled WER; an empty reference counts the hypothesis words."""
    import jiwer

    h, r = normalize(hyp, language), normalize(ref, language)
    if not r:
        return len(h.split()), 0
    words = len(r.split())
    if not h:
        return words, words
    out = jiwer.process_words(r, h)
    return min(out.substitutions + out.deletions + out.insertions, words), words


def word_error_rate(hyp: str, ref: str, language: str) -> float:
    errors, words = word_errors(hyp, ref, language)
    if words == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / words


def clean_exemplars(
    asg: Assignments, character_id: int, n: int = EXEMPLARS, min_s: float = EXEMPLAR_MIN_S
) -> list[tuple[float, float]]:
    """The character's ``n`` longest turns of at least ``min_s`` that no other speaker's turn
    overlaps, in time order."""
    chars = {a.cluster: a.character_id for a in asg.clusters if a.status not in ("extra",)}
    turns = asg.turns or []
    mine = [t for t in turns if chars.get(t.speaker) == character_id]
    clean = [
        t
        for t in mine
        if t.end - t.start >= min_s
        and not any(o.speaker != t.speaker and o.start < t.end and t.start < o.end for o in turns)
    ]
    best = sorted(clean, key=lambda t: (-(t.end - t.start), t.start))[:n]
    return sorted((float(t.start), float(t.end)) for t in best)


def to_16k(src: Path, dest: Path) -> Path:
    """``src`` as 16 kHz mono 16-bit pcm at ``dest`` (what the embedders, faster-whisper and
    the taggers read)."""
    from math import gcd

    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, data = wavfile.read(src)
    x = data.astype(np.float64)
    if np.issubdtype(data.dtype, np.integer):
        x /= float(np.iinfo(data.dtype).max) + 1.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if rate != RATE_16K:
        g = gcd(RATE_16K, rate)
        x = resample_poly(x, RATE_16K // g, rate // g)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    with atomic_path(dest) as tmp:
        wavfile.write(tmp, RATE_16K, pcm)
    return dest


# ---- scores and summaries -------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedCue:
    episode: str
    cue: DubCue
    path: Path
    seconds: float
    card_id: str = ""
    backend: str = ""
    voice: str | None = None


@dataclass
class CueScore:
    episode: str
    cue_id: str
    character_id: int
    card_id: str
    backend: str
    voice: str | None
    text: str
    seconds: float
    slot: float
    similarity: float | None
    similarity_source: str
    hyp: str
    errors: int
    ref_words: int
    source_emotion: str | None = None
    render_emotion: str | None = None
    ref_seconds: float | None = None
    ref_emotion: str | None = None
    ref_similarity: float | None = None
    neutral_only: bool = False  # the backend takes no delivery at all
    ref_matched: bool | None = None  # None: no reference workspace

    @property
    def fits(self) -> bool:
        return duration_fit(self.seconds, self.slot)[0]

    @property
    def stretch(self) -> float:
        return duration_fit(self.seconds, self.slot)[1]

    @property
    def cps(self) -> float:
        return len(self.text) / self.seconds if self.seconds > 0 else 0.0


@dataclass
class TtsSummary:
    per_character: dict[int, dict[str, float]]
    overall: dict[str, float]
    cues: int
    per_card: dict[str, dict[str, Any]] = field(default_factory=dict)
    similarity_source: dict[int, str] = field(default_factory=dict)
    emotion: dict[str, Any] = field(default_factory=dict)


def _r(x: float) -> float:
    return round(float(x), 4)


def _agreement(
    m: dict[str, float], prefix: str, pairs: Sequence[tuple[str | None, str | None]]
) -> dict[str, Any]:
    """Agreement over the pairs where both labels exist, with each side's coverage (labelled
    share of all pairs). Under ``COVERAGE_MIN`` on either side the agreement stays out of the
    metrics and ``<prefix>_low_coverage`` is set instead."""
    n = len(pairs)
    src = sum(1 for a, _ in pairs if a)
    ren = sum(1 for _, b in pairs if b)
    both = [(a, b) for a, b in pairs if a and b]
    agree = _r(np.mean([a == b for a, b in both])) if both else None
    low = n == 0 or src / n < COVERAGE_MIN or ren / n < COVERAGE_MIN or not both
    m[f"{prefix}_source_coverage"] = _r(src / n) if n else 0.0
    m[f"{prefix}_render_coverage"] = _r(ren / n) if n else 0.0
    m[f"{prefix}_cues"] = len(both)
    if low:
        m[f"{prefix}_low_coverage"] = 1
    elif agree is not None:
        m[f"{prefix}_agreement"] = agree
    return {
        "agreement": agree,
        "cues": len(both),
        "source_coverage": m[f"{prefix}_source_coverage"],
        "render_coverage": m[f"{prefix}_render_coverage"],
        "low_coverage": low,
    }


def _metrics(scores: Sequence[CueScore], max_speedup: float) -> dict[str, float]:
    m: dict[str, float] = {"cues": len(scores)}
    sources = sorted({s.similarity_source for s in scores if s.similarity is not None})
    for src in sources:
        vals = [s.similarity for s in scores if s.similarity_source == src]
        m[f"speaker_similarity_{src}"] = _r(np.mean([v for v in vals if v is not None]))
    if len(sources) == 1:  # never pooled across reference kinds
        m["speaker_similarity"] = m[f"speaker_similarity_{sources[0]}"]
    words = sum(s.ref_words for s in scores)
    if words:
        m["wer"] = _r(sum(s.errors for s in scores) / words)
        m["wer_median"] = _r(np.median([s.errors / s.ref_words for s in scores if s.ref_words]))
    stretch = [s.stretch for s in scores]
    m["fit_rate"] = _r(np.mean([s.fits for s in scores]))
    m["fit_budget_rate"] = _r(np.mean([x <= max_speedup + 1e-9 for x in stretch]))
    finite = [x for x in stretch if math.isfinite(x)]
    if finite:
        m["stretch_median"] = _r(np.median(finite))
        m["stretch_p90"] = _r(np.percentile(finite, 90))
    m["cps_median"] = _r(np.median([s.cps for s in scores]))
    if any(s.source_emotion for s in scores):
        _agreement(m, "emotion", [(s.source_emotion, s.render_emotion) for s in scores])
    tagged = [s for s in scores if s.render_emotion]
    if tagged:
        m["render_neutral_share"] = _r(np.mean([s.render_emotion == "neutral" for s in tagged]))
        m["render_labelled"] = len(tagged)
    if any(s.ref_matched is not None for s in scores):
        m["ref_unmatched"] = sum(1 for s in scores if s.ref_matched is False)
    ref = [s for s in scores if s.ref_seconds]
    if ref:
        m["ref_cues"] = len(ref)
        m["ref_duration_ratio"] = _r(
            np.median([s.seconds / cast(float, s.ref_seconds) for s in ref])
        )
        if any(s.ref_emotion for s in ref):
            _agreement(m, "ref_emotion", [(s.ref_emotion, s.render_emotion) for s in ref])
        rsim = [s.ref_similarity for s in ref if s.ref_similarity is not None]
        if rsim:
            m["ref_speaker_similarity"] = _r(np.mean(rsim))
    return m


NEUTRAL_NOTE = (
    "the TTS backend takes no delivery, so its renders are neutral by design and agreement is "
    "expected to track the neutral share of the source labels"
)


def summarize(
    scores: Sequence[CueScore], max_speedup: float, tagger: str | None = None
) -> TtsSummary:
    """Per-character and overall metrics, each card's speech rate (ruling CD8) and the emotion
    scoring's backends and coverage. ``tagger`` is the backend that labelled both the source
    lines and the renders."""
    by_char: dict[int, list[CueScore]] = {}
    by_card: dict[str, list[CueScore]] = {}
    for s in scores:
        by_char.setdefault(s.character_id, []).append(s)
        by_card.setdefault(s.card_id, []).append(s)
    per_character = {cid: _metrics(ss, max_speedup) for cid, ss in sorted(by_char.items())}
    overall: dict[str, float] = _metrics(scores, max_speedup) if scores else {"cues": 0}
    overall["characters"] = len(by_char)
    per_card: dict[str, dict[str, Any]] = {}
    for cid, ss in sorted(by_card.items()):
        cps = [s.cps for s in ss]
        per_card[cid] = {
            "backend": ss[0].backend,
            "voice": ss[0].voice,
            "characters": sorted({s.character_id for s in ss}),
            "cues": len(ss),
            "cps_median": _r(np.median(cps)),
            "cps_p25": _r(np.percentile(cps, 25)),
            "cps_p75": _r(np.percentile(cps, 75)),
        }
    sources = {cid: ss[0].similarity_source for cid, ss in sorted(by_char.items())}
    emotion: dict[str, Any] = {}
    if any(s.source_emotion for s in scores):
        emotion = _agreement({}, "emotion", [(s.source_emotion, s.render_emotion) for s in scores])
        emotion |= {"source_backend": tagger, "render_backend": tagger}
        if scores and all(s.neutral_only for s in scores):
            emotion["note"] = NEUTRAL_NOTE
    return TtsSummary(per_character, overall, len(scores), per_card, sources, emotion)


HEADLINE_METRICS = (
    "speaker_similarity", "speaker_similarity_references", "speaker_similarity_card",
    "speaker_similarity_original", "wer", "wer_median",
    "wer_floor", "wer_floor_clips", "fit_rate", "fit_budget_rate", "stretch_median",
    "stretch_p90", "cps_median", "cues", "characters", "excluded_short", "emotion_agreement",
    "emotion_cues", "emotion_source_coverage", "emotion_render_coverage", "emotion_low_coverage",
    "render_neutral_share", "render_labelled", "ref_duration_ratio", "ref_emotion_agreement",
    "ref_emotion_cues", "ref_emotion_source_coverage", "ref_emotion_render_coverage",
    "ref_emotion_low_coverage", "ref_speaker_similarity", "ref_cues", "ref_unmatched",
)  # fmt: skip


def tts_rows(
    summary: TtsSummary,
    *,
    cfg: Config,
    backend_cfg: BackendConfig,
    source_language: str,
    test_set: str,
    split: str,
    episodes: list[str],
    tag: str | None,
    workspace: Path,
    options: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
) -> list[Any]:
    """One ``tts`` ledger row for one TTS backend's renders."""
    from fukikae.eval.ledger import make_row

    model, rev = model_identity("tts", backend_cfg)
    metrics = {k: summary.overall[k] for k in HEADLINE_METRICS if k in summary.overall}
    detail: dict[str, Any] = {
        "per_character": {str(k): v for k, v in summary.per_character.items()},
        "per_card": summary.per_card,
        "similarity_source": {str(k): v for k, v in summary.similarity_source.items()},
    }
    if summary.emotion:
        detail["emotion"] = summary.emotion
    if notes:
        detail["notes"] = list(notes)
    return [
        make_row(
            stage="tts",
            backend=backend_cfg.backend,
            model=model,
            revision=rev,
            options={**backend_cfg.options, **(options or {})},
            language=f"{source_language}-{cfg.target_language}",
            genre=cfg.series.genre,
            series=cfg.series.title,
            test_set=test_set,
            split=split,
            episodes=episodes,
            metrics=metrics,
            detail=detail,
            workspace=str(workspace),
            tag=tag,
        )
    ]


# ---- the run --------------------------------------------------------------------------------


@dataclass
class EvalResult:
    scores: list[CueScore]
    summaries: dict[str, TtsSummary]
    backend_cfgs: dict[str, BackendConfig]
    source_language: str
    episodes: list[str]
    names: dict[int, str]
    mode: str
    notes: list[str] = field(default_factory=list)
    tagger: str | None = None
    floor: str | None = None  # how the WER floor was measured


@contextlib.contextmanager
def _model(kind: str, bc: BackendConfig, device: str, cfg: Config) -> Iterator[Any]:
    inst = loader.create(kind, bc, device)
    with gpu_job(cfg.gpu_device, device, f"eval tts {kind}"):  # ruling CD15
        if device.startswith("cuda"):
            wait_for_vram(cfg.gpu_device, inst.vram_mb, runctx.vram_timeout_s, query=free_mb)
        try:
            inst.load()
            yield inst
        finally:
            inst.unload()


def _cues(ws: Workspace, episodes: Sequence[str], target: str) -> dict[str, DubCues]:
    out: dict[str, DubCues] = {}
    for e in episodes:
        path = ws.work_dir(e) / "06-translate" / f"cues.{target}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{e}: no {path}; run `fukikae run <workspace> {e} --to translate` first"
            )
        out[e] = read_json(path, DubCues)
    return out


def _stock_cards(
    ws: Workspace,
    cfg: Config,
    store: CardStore,
    backend: str,
    picked: list[tuple[str, DubCue]],
    assigned: dict[int, str],
) -> dict[int, str]:
    """A card of ``backend`` per character: its assigned card when that is on ``backend``,
    else a stock voice of its estimated gender (put in the store, never assigned)."""
    from fukikae.cards.auto import character_spans, gender_thresholds

    male_below, female_above = gender_thresholds(cfg)
    rev = revision(backend)
    out: dict[int, str] = {}
    used: list[str] = []
    spans: dict[int, list[tuple[str, tuple[float, float]]]] = {}
    for e in sorted({e for e, _ in picked}):
        path = ws.work_dir(e) / "05-characters" / "assignments.json"
        if path.exists():
            for cid, ss in character_spans(read_json(path, Assignments)).items():
                spans.setdefault(cid, []).extend((e, s) for s in ss)
    chars = sorted({cast(int, c.character_id) for _, c in picked})
    for cid in chars:
        a = assigned.get(cid)
        if a is not None and store.has(a) and store.get(a).voice.backend == backend:
            out[cid] = a
            if store.get(a).voice.voice:
                used.append(cast(str, store.get(a).voice.voice))
    order = sorted(
        (c for c in chars if c not in out),
        key=lambda c: (-sum(b - a for _, (a, b) in spans.get(c, [])), c),
    )
    for cid in order:
        mine = spans.get(cid, [])
        f0 = None
        if mine:
            e = mine[0][0]
            f0, _ = speaker_f0(
                ws.work_dir(e) / "01-separate" / "vocals.wav", [s for x, s in mine if x == e]
            )
        voice = pick_stock(
            backend, cfg.target_language, gender_of(f0, male_below, female_above), used
        )
        used.append(voice)
        out[cid] = store.put(stock_card(backend, voice, cfg.target_language, rev))
    return out


def character_references(
    ws: Workspace,
    episodes: Sequence[str],
    cid: int,
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[str, list[RefPick]] | None:
    """The character's reference clips (``select_references``, songs and singing excluded) in
    the episode where they add up to the most speech, with that episode; None when no episode
    has enough. ``exclusions`` caches each episode's excluded spans."""
    from fukikae.cards.select import reference_exclusions
    from fukikae.stages.mix import mix_settings

    cache = exclusions if exclusions is not None else {}
    best: tuple[str, list[RefPick]] | None = None
    for e in episodes:
        work = ws.work_dir(e)
        asg_path = work / "05-characters" / "assignments.json"
        tr_path = work / "03-asr" / "transcript.json"
        vocals = work / "01-separate" / "vocals.wav"
        if not (asg_path.exists() and tr_path.exists() and vocals.exists()):
            continue
        asg = read_json(asg_path, Assignments)
        speakers = {
            a.cluster
            for a in asg.clusters
            if a.character_id == cid and a.status not in ("extra", "review")
        }
        if not speakers or not asg.turns:
            continue
        tags_path = work / "04b-tag" / "tags.json"
        tags = read_json(tags_path, Tags) if tags_path.exists() else None
        if e not in cache:
            cache[e] = reference_exclusions(work, tags, mix_settings(ws.config).songs)
        picks = select_references(
            asg.turns, speakers, read_json(tr_path, Transcript), tags, vocals, exclude=cache[e]
        )
        if picks and (
            best is None or sum(p.seconds for p in picks) > sum(p.seconds for p in best[1])
        ):
            best = (e, picks)
    return best


def _clone_cards(
    ws: Workspace,
    cfg: Config,
    store: CardStore,
    cls: type[Any],
    refs: Mapping[int, tuple[str, list[RefPick]] | None],
    source_language: str,
) -> dict[int, str]:
    """A clone card of ``cls`` per character with references (put in the store, never
    assigned)."""
    out: dict[int, str] = {}
    for cid, found in sorted(refs.items()):
        if found is None:
            continue
        e, picks = found
        vocals = ws.work_dir(e) / "01-separate" / "vocals.wav"
        card = clone_card(cls, picks, store, vocals, source_language, cfg.target_language, e)
        out[cid] = store.put(card)
    return out


def _render(
    ws: Workspace,
    cfg: Config,
    store: CardStore,
    picked: list[tuple[str, DubCue]],
    cards: dict[int, str],
    names: dict[int, str],
) -> tuple[list[RenderedCue], dict[str, BackendConfig], dict[str, bool]]:
    """Each picked cue rendered through the tts stage's own render path (ruling CD12)."""
    from fukikae.stages.tts import (
        DELIVERY_CAPS,
        LazySynth,
        backend_config,
        check_card,
        prepare_cue,
        render_cue,
        synth_device,
        wav_seconds,
    )

    def who(cid: int) -> str:
        return f"character {cid}" + (f" ({names[cid]})" if cid in names else "")

    checked = {
        card: check_card(store, card, who(cid), cfg.target_language)
        for cid, card in sorted(cards.items())
    }
    backends = sorted({card.voice.backend for card, _ in checked.values()})
    bcs = {b: backend_config(cfg, b) for b in backends}
    idents = {b: identity_key("tts", bcs[b]) for b in backends}
    neutral: dict[str, bool] = {}
    base = ws.root / ".fukikae" / "eval" / "tts"
    cache = ws.root / ".fukikae" / "tts-cache"
    out: list[RenderedCue] = []
    for b in backends:
        jobs = [(e, c) for e, c in picked if checked[cards[_cid(c)]][0].voice.backend == b]
        cls = checked[cards[_cid(jobs[0][1])]][1]
        neutral[b] = not (cls.capabilities & DELIVERY_CAPS)
        rev = idents[b].get("revision") or getattr(cls, "model_revision", None) or "unknown"
        folder = base / f"{b}-{rev}"
        folder.mkdir(parents=True, exist_ok=True)
        lazy = LazySynth(bcs[b], synth_device(cfg, cls), cfg)
        try:
            for e, c in jobs:
                card_ref = cards[_cid(c)]
                vocals = ws.work_dir(e) / "01-separate" / "vocals.wav"
                job = prepare_cue(
                    c, card_ref, cfg, store, who=who(_cid(c)), checked=checked[card_ref],
                    identity=idents[b], vocals=vocals,
                )  # fmt: skip
                dest = folder / f"{e}-{c.id}.{job.key}.wav"
                if not dest.exists():
                    if (cache / f"{job.key}.wav").exists():
                        with atomic_path(dest) as tmp:
                            shutil.copyfile(cache / f"{job.key}.wav", tmp)
                    else:
                        render_cue(job, lazy, dest, store, ref_dir=folder / "ref" / e)
                out.append(
                    RenderedCue(e, c, dest, wav_seconds(dest), card_ref, b, _voice_name(job.card))
                )
        finally:
            lazy.close()
    return out, bcs, neutral


def _cid(c: DubCue) -> int:
    return cast(int, c.character_id)


def _voice_name(card: Card) -> str | None:
    v = card.voice
    if v.voice:
        return v.voice
    if v.voices:
        return "+".join(f"{n}:{w:g}" for n, w in v.voices)
    return v.source


def sidecar_stem(ref: str) -> str:
    """A short file stem for a ``sha256:<hex>`` sidecar ref: its first 16 hex digits."""
    return ref.removeprefix("sha256:")[:16]


def _sixteen(path: Path) -> Path:
    dest = path.with_name(path.name[: -len(".wav")] + ".16k.wav")
    if not dest.exists() or dest.stat().st_mtime < path.stat().st_mtime:
        to_16k(path, dest)
    return dest


def _tagger_name(ws: Workspace, episode: str) -> str | None:
    path = ws.work_dir(episode) / "04b-tag" / "tags.json"
    if not path.exists():
        return None
    tags = read_json(path, Tags)
    return tags.emotion_backend or tags.backend


def _label(tags: Tags, start: float) -> str | None:
    for e in tags.emotions:
        if abs(e.start - start) < 1e-6:
            return e.label
    return None


def _write_clip(wav: np.ndarray, span: tuple[float, float], dest: Path) -> Path:
    from scipy.io import wavfile

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        seg = wav[max(0, int(span[0] * RATE_16K)) : int(span[1] * RATE_16K)]
        with atomic_path(dest) as tmp:
            wavfile.write(tmp, RATE_16K, (np.clip(seg, -1, 1) * 32767).astype("<i2"))
    return dest


def _read16k(path: Path) -> np.ndarray:
    from fukikae.backends.embed._common import read_wav16k

    return read_wav16k(path)


def floor_clips(
    out_dir: Path,
    subs: Sequence[Any],
    vocals: Path,
    *,
    episode: str,
    slots: Sequence[tuple[float, float]] | None = None,
    limit: int = FLOOR_CLIPS,
) -> list[tuple[Path, str]]:
    """Natural-speech clips with their subtitle text, for the WER floor: with ``slots``, the
    subtitle lines whose midpoint falls in each slot (within ``REF_PAD_S``), one clip per slot
    that has any; without, up to ``limit`` lines of at least ``FLOOR_MIN_S`` and
    ``FLOOR_MIN_WORDS`` words, evenly spaced over the episode (a fixed set)."""
    wav = _read16k(vocals)
    picks: list[tuple[tuple[float, float], str]] = []
    if slots is not None:
        for a, b in slots:
            hit = [c for c in subs if a - REF_PAD_S <= (c.start + c.end) / 2 <= b + REF_PAD_S]
            if hit:
                picks.append(
                    ((min(c.start for c in hit), max(c.end for c in hit)),
                     " ".join(c.text for c in hit))
                )  # fmt: skip
    else:
        ok = [
            c
            for c in subs
            if c.end - c.start >= FLOOR_MIN_S
            and len(normalize(c.text, "en").split()) >= FLOOR_MIN_WORDS
        ]
        step = max(1, len(ok) // limit) if ok else 1
        picks = [((c.start, c.end), c.text) for c in ok[::step][:limit]]
    return [
        (_write_clip(wav, span, out_dir / f"{episode}-{span[0]:.2f}.16k.wav"), text)
        for span, text in picks
    ]


def evaluate(
    root: Path,
    episodes: Sequence[str],
    *,
    backend: str | None = None,
    per_character: int = 8,
    seed: int = 0,
    min_slot: float = MIN_SLOT_S,
    reference_workspace: Path | None = None,
    floor_subs: Path | None = None,
    floor_workspace: Path | None = None,
    floor_episode: str | None = None,
) -> EvalResult:
    from fukikae.eval.autolabel import gt_path_for
    from fukikae.registry.store import Registry
    from fukikae.stages.tts import max_speedup
    from fukikae.subtitles import read_cues

    ws = Workspace(root)
    cfg = ws.config
    target = cfg.target_language
    docs = _cues(ws, episodes, target)
    sources = {d.source_language for d in docs.values()}
    if len(sources) != 1:
        raise ValueError(f"episodes are translated from different languages: {sorted(sources)}")
    source_language = sources.pop()
    taggers = {e: _tagger_name(ws, e) for e in episodes}
    named = sorted({t for t in taggers.values() if t})
    if len(named) > 1:
        raise ValueError(
            "the episodes' emotions come from different taggers ("
            + ", ".join(f"{e} {t}" for e, t in taggers.items() if t)
            + "); evaluate them separately"
        )
    tagger = named[0] if named else None
    if floor_subs is not None and floor_workspace is None and reference_workspace is None:
        raise ValueError("--floor-subs needs --reference-workspace or --floor-workspace")
    if (
        floor_subs is not None
        and reference_workspace is None
        and floor_workspace is not None
        and not floor_episode
    ):
        raise ValueError("a fixed WER floor set needs --floor-episode")
    picked = sample_cues(docs, per_character, seed, min_slot)
    excluded = count_short(docs, min_slot)
    if not picked:
        raise ValueError(
            f"no cue to render: every cue is an extra, skipped, failed, empty or shorter than "
            f"{min_slot:g} s"
        )
    store = CardStore(ws.root)
    chars = sorted({_cid(c) for _, c in picked})
    with Registry(ws.registry_path) as reg:
        names = {c.id: c.name for c in reg.characters() if c.name}
        assigned: dict[int, str] = {}
        for cid in chars:
            a = reg.card_for(cid, target)
            if a is not None and a.card_id is not None:
                assigned[cid] = a.card_id
    notes: list[str] = []
    if excluded:
        notes.append(f"left out {excluded} cue(s) with a slot under {min_slot:g} s")
    if backend is None:
        missing = [cid for cid in chars if cid not in assigned]
        if missing:
            notes.append(
                f"left out characters with no {target!r} card: {', '.join(map(str, missing))} "
                "(run the tts stage, or pass --backend)"
            )
            picked = [(e, c) for e, c in picked if c.character_id in assigned]
            if not picked:
                raise ValueError(f"no sampled character has a {target!r} card")
        cards = {cid: assigned[cid] for cid in chars if cid in assigned}
        mode = "assigned"
    refs = {
        cid: character_references(ws, sorted({e for e, c in picked if c.character_id == cid}), cid)
        for cid in sorted({_cid(c) for _, c in picked})
    }
    for cid, found in refs.items():
        for line in reference_report(found[1]) if found else []:
            notes.append(f"character {cid}: {line}")
    if backend is not None:
        cls = loader.backend_class("tts", backend)
        sources = getattr(cls, "sources", frozenset())
        if "clone" in sources and "stock" not in sources:
            mode = "clone"
            cards = _clone_cards(ws, cfg, store, cls, refs, source_language)
            skipped = [cid for cid in refs if cid not in cards]
            if skipped:
                notes.append(
                    "skipped (too little clean speech for a clone, under 6 s): "
                    + ", ".join(f"character {cid}" for cid in skipped)
                )
                picked = [(e, c) for e, c in picked if _cid(c) in cards]
                if not picked:
                    raise ValueError("no sampled character has enough clean speech for a clone")
        else:
            cards = _stock_cards(ws, cfg, store, backend, picked, assigned)
            mode = "stock"

    # 1. TTS backend(s).
    renders, bcs, neutral = _render(ws, cfg, store, picked, cards, names)
    small = {(r.episode, r.cue.id): _sixteen(r.path) for r in renders}
    eval_dir = ws.root / ".fukikae" / "eval" / "tts"

    # Original-voice references per character, or the card's clips.
    ref_spans: dict[int, list[tuple[Path, tuple[float, float]]]] = {}
    sim_source: dict[int, str] = {}
    for cid, card_ref in cards.items():
        card = store.get(card_ref)
        found = refs.get(cid)
        if found is not None:
            e, picks = found
            sim_source[cid] = "references"
            vocals16 = ws.work_dir(e) / "01-separate" / "vocals_16k.wav"
            if not vocals16.exists():
                to_16k(ws.work_dir(e) / "01-separate" / "vocals.wav", vocals16)
            ref_spans[cid] = [(vocals16, (p.start, p.end)) for p in picks]
            continue
        if card.voice.references:
            sim_source[cid] = "card"
            clips = []
            for ref in card.voice.references:
                dest = eval_dir / "refs" / f"{sidecar_stem(ref.clip)}.16k.wav"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    to_16k(store.sidecar_path(ref.clip), dest)
                clips.append((dest, (0.0, wav_seconds_16k(dest))))
            ref_spans[cid] = clips
            continue
        best: list[tuple[float, Path, tuple[float, float]]] = []
        for e in sorted({e for e, c in picked if c.character_id == cid}):
            path = ws.work_dir(e) / "05-characters" / "assignments.json"
            vocals = ws.work_dir(e) / "01-separate" / "vocals_16k.wav"
            if path.exists() and vocals.exists():
                for s in clean_exemplars(read_json(path, Assignments), cid):
                    best.append((s[1] - s[0], vocals, s))
        best.sort(key=lambda x: (-x[0], str(x[1]), x[2]))
        if best:
            sim_source[cid] = "original"
            ref_spans[cid] = [(p, s) for _, p, s in best[:EXEMPLARS]]
        else:
            sim_source[cid] = "none"
            notes.append(f"character {cid}: no clean original turn, so no speaker similarity")

    # The original line of each cue, from the reference workspace.
    ref_ws = Workspace(reference_workspace) if reference_workspace else None
    ref_line: dict[tuple[str, str], tuple[float, float]] = {}
    ref_vocals: dict[str, Path] = {}
    if ref_ws is not None:
        for e in sorted({r.episode for r in renders}):
            tpath = ref_ws.work_dir(e) / "03-asr" / "transcript.json"
            vocals = ref_ws.work_dir(e) / "01-separate" / "vocals_16k.wav"
            if not tpath.exists() or not vocals.exists():
                notes.append(f"{e}: the reference workspace has no transcript or vocals")
                continue
            ref_vocals[e] = vocals
            tr = read_json(tpath, Transcript)
            for r in renders:
                if r.episode == e:
                    span = reference_span(tr, (r.cue.start, r.cue.end))
                    if span is not None and span[1] > span[0]:
                        ref_line[(e, r.cue.id)] = span

    # The WER floor's natural-speech clips.
    floor: list[tuple[Path, str]] = []
    floor_desc: str | None = None
    if floor_subs is not None:
        if ref_ws is not None and (
            floor_workspace is None or Path(floor_workspace).resolve() == ref_ws.root.resolve()
        ):
            eps = sorted({r.episode for r in renders})
            for e in eps:
                vocals = ref_ws.work_dir(e) / "01-separate" / "vocals_16k.wav"
                if not vocals.exists():
                    continue
                subs = read_cues(gt_path_for(floor_subs, e, len(eps)))
                slots = [(r.cue.start, r.cue.end) for r in renders if r.episode == e]
                floor += floor_clips(eval_dir / "floor", subs, vocals, episode=e, slots=slots)
            floor_desc = f"reference:{ref_ws.root}:{floor_subs}"
        else:
            fws = Workspace(cast(Path, floor_workspace))
            ep = cast(str, floor_episode).upper()
            vocals = fws.work_dir(ep) / "01-separate" / "vocals_16k.wav"
            if not vocals.exists():
                raise FileNotFoundError(f"{ep}: no {vocals} for the WER floor")
            subs = read_cues(gt_path_for(floor_subs, ep, 1))
            floor = floor_clips(eval_dir / "floor-fixed", subs, vocals, episode=ep)
            floor_desc = f"fixed:{fws.root}:{ep}:{floor_subs}"

    gpu = device_string(cfg)
    # 2. WeSpeaker.
    sims: dict[tuple[str, str], float] = {}
    ref_sims: dict[tuple[str, str], float] = {}
    with _model("embed", EMBEDDER, gpu, cfg) as emb:
        ref_vecs: dict[int, np.ndarray] = {}
        for cid, items in ref_spans.items():
            ref_vecs[cid] = np.concatenate([emb.embed(p, [s]) for p, s in items], axis=0)
        for r in renders:
            k = (r.episode, r.cue.id)
            vec = emb.embed(small[k], [(0.0, r.seconds)])
            cid = _cid(r.cue)
            if cid in ref_vecs:
                sims[k] = similarity(vec, ref_vecs[cid])
            if k in ref_line:
                orig = emb.embed(ref_vocals[r.episode], [ref_line[k]])
                ref_sims[k] = similarity(vec, orig)

    # 3. faster-whisper: the renders, then the natural-speech floor clips.
    hyps: dict[tuple[str, str], str] = {}
    floor_errors = floor_words = 0
    with _model("asr", ASR, gpu, cfg) as asr:
        for r in renders:
            k = (r.episode, r.cue.id)
            tr = asr.transcribe(small[k], target)
            hyps[k] = " ".join(s.text for s in tr.segments).strip()
        for path, text in floor:
            tr = asr.transcribe(path, target)
            errs, words = word_errors(" ".join(s.text for s in tr.segments), text, target)
            floor_errors += errs
            floor_words += words

    # 4. The tagger, on the CPU: every render of a tagged episode, and every matched line.
    render_emotion: dict[tuple[str, str], str | None] = {}
    ref_emotion: dict[tuple[str, str], str | None] = {}
    to_tag = [r for r in renders if taggers.get(r.episode) or (r.episode, r.cue.id) in ref_line]
    if to_tag:
        name = tagger or DEFAULT_TAGGER
        with _model("tag", BackendConfig(name, {}), "cpu", cfg) as tg:
            for r in to_tag:
                k = (r.episode, r.cue.id)
                tags = tg.tag(small[k], [Turn(start=0.0, end=r.seconds, speaker="render")])
                render_emotion[k] = _label(tags, 0.0)
            for e in sorted(ref_vocals):
                mine = [r for r in to_tag if r.episode == e and (e, r.cue.id) in ref_line]
                if not mine:
                    continue
                turns = [
                    Turn(start=ref_line[(e, r.cue.id)][0], end=ref_line[(e, r.cue.id)][1],
                         speaker="original")
                    for r in mine
                ]  # fmt: skip
                tags = tg.tag(ref_vocals[e], turns)
                for r in mine:
                    ref_emotion[(e, r.cue.id)] = _label(tags, ref_line[(e, r.cue.id)][0])

    scores: list[CueScore] = []
    for r in renders:
        k = (r.episode, r.cue.id)
        cid = _cid(r.cue)
        errors, words = word_errors(hyps.get(k, ""), cast(str, r.cue.text), target)
        line = ref_line.get(k)
        scores.append(
            CueScore(
                episode=r.episode,
                cue_id=r.cue.id,
                character_id=cid,
                card_id=r.card_id,
                backend=r.backend,
                voice=r.voice,
                text=cast(str, r.cue.text),
                seconds=r.seconds,
                slot=r.cue.end - r.cue.start,
                similarity=sims.get(k),
                similarity_source=sim_source.get(cid, "none"),
                hyp=hyps.get(k, ""),
                errors=errors,
                ref_words=words,
                source_emotion=r.cue.delivery.emotion if taggers.get(r.episode) else None,
                render_emotion=render_emotion.get(k),
                ref_seconds=None if line is None else line[1] - line[0],
                ref_emotion=ref_emotion.get(k),
                ref_similarity=ref_sims.get(k),
                neutral_only=neutral.get(r.backend, False),
                ref_matched=None if ref_ws is None else line is not None,
            )
        )
    speedup = max_speedup(cfg)
    summaries: dict[str, TtsSummary] = {}
    for b in sorted(bcs):
        s = summarize([x for x in scores if x.backend == b], speedup, tagger=tagger)
        s.overall["excluded_short"] = excluded
        if floor_words:
            s.overall["wer_floor"] = _r(floor_errors / floor_words)
            s.overall["wer_floor_clips"] = len(floor)
        summaries[b] = s
    return EvalResult(
        scores, summaries, bcs, source_language, list(episodes), names, mode, notes, tagger,
        floor_desc,
    )  # fmt: skip


def wav_seconds_16k(path: Path) -> float:
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# ---- output ---------------------------------------------------------------------------------

_COLUMNS = (
    ("speaker_similarity", "sim"), ("speaker_similarity_references", "sim_refs"),
    ("speaker_similarity_card", "sim_card"),
    ("speaker_similarity_original", "sim_orig"), ("wer", "wer"), ("wer_median", "wer50"),
    ("fit_rate", "fit"),
    ("stretch_median", "str50"), ("stretch_p90", "str90"), ("cps_median", "cps"),
    ("emotion_agreement", "emo"), ("emotion_render_coverage", "emo_cov"),
    ("ref_duration_ratio", "ref_dur"), ("ref_emotion_agreement", "ref_emo"),
    ("ref_speaker_similarity", "ref_sim"), ("cues", "n"),
)  # fmt: skip


def format_summary(summary: TtsSummary, names: Mapping[int, str], backend: str) -> str:
    cols = [(k, h) for k, h in _COLUMNS if k in summary.overall]
    header = f"{'character':<24}" + "".join(f"{h:>9}" for _, h in cols) + "  similarity to"
    lines = [f"tts backend {backend}", header]

    def cell(m: Mapping[str, float], k: str) -> str:
        v = m.get(k)
        if v is None:
            return f"{'-':>9}"
        return f"{int(v):>9}" if k == "cues" else f"{v:>9.3f}"

    for cid, m in summary.per_character.items():
        label = f"{cid} {names.get(cid, '')}".strip()[:23]
        src = summary.similarity_source.get(cid, "none")
        lines.append(f"{label:<24}" + "".join(cell(m, k) for k, _ in cols) + f"  {src}")
    lines.append(
        f"{'overall':<24}" + "".join(cell(summary.overall, k) for k, _ in cols)
        + f"  ({summary.overall.get('characters', 0):.0f} characters)"
    )  # fmt: skip
    o = summary.overall
    if "wer_floor" in o:
        lines.append(
            f"WER floor (natural speech, same ASR): {o['wer_floor']:.3f} over "
            f"{o['wer_floor_clips']:.0f} clips; render WER {o.get('wer', 0):.3f} is read against it"
        )
    if summary.emotion:
        e = summary.emotion
        state = "LOW COVERAGE, not a headline" if e["low_coverage"] else "headline"
        lines.append(
            f"emotion ({e.get('source_backend')}): source coverage {e['source_coverage']:.2f}, "
            f"render coverage {e['render_coverage']:.2f}, {e['cues']} cues with both labels, "
            f"agreement {e['agreement'] if e['agreement'] is not None else '-'} ({state})"
        )
        if e.get("note"):
            lines.append(f"  note: {e['note']}")
    if o.get("excluded_short"):
        lines.append(f"cues left out for a short slot: {o['excluded_short']:.0f}")
    lines.append("")
    lines.append(f"{'card':<18}{'voice':<14}{'cues':>5}{'cps p25':>9}{'median':>8}{'p75':>7}")
    for cid, c in summary.per_card.items():
        lines.append(
            f"{cid:<18}{str(c['voice'] or '-'):<14}{c['cues']:>5}{c['cps_p25']:>9.2f}"
            f"{c['cps_median']:>8.2f}{c['cps_p75']:>7.2f}"
        )
    return "\n".join(lines)


__all__ = [
    "ASR",
    "EMBEDDER",
    "CueScore",
    "EvalResult",
    "RenderedCue",
    "TtsSummary",
    "clean_exemplars",
    "count_short",
    "duration_fit",
    "evaluate",
    "floor_clips",
    "format_summary",
    "reference_span",
    "reference_speech",
    "sample_cues",
    "sidecar_stem",
    "similarity",
    "summarize",
    "to_16k",
    "tts_rows",
    "word_error_rate",
    "word_errors",
]
