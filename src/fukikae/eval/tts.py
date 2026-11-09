"""`fukikae eval tts`: score TTS renders per character and write results-ledger rows.

A sample of each character's translated cues (``sample_cues``; extras, skipped, errored and
empty cues left out) is rendered with the character's card at the card's own rate, with no
rate fit and no time-stretch, trimmed of silence exactly as the tts stage trims (the render's
cache key is the stage's ``cue_key``, so a render already in ``.fukikae/tts-cache`` is linked,
not made again). Renders live in ``<ws>/.fukikae/eval/tts/<backend>-<revision>/`` as
``<episode>-<cue-id>.<key>.wav``: the key in the name means a swapped card or changed text
renders again instead of reusing a stale file.

Metrics, per character and overall:

- ``speaker_similarity``: WeSpeaker cosine between each render and the normalized mean of the
  character's references. A card with reference clips (a clone) is compared with those clips
  (``similarity_source = "card"``). A stock card has none, so it is compared with the
  character's original-voice exemplars, the three longest diarize turns no other speaker
  overlaps, from the separated vocals of the sampled episodes (``"original"``): that is the
  voice a clone would try to reproduce, so stock voices give the baseline on the same scale
  (the plan's decision). Across languages the original voice is another actor in another
  language, so stock numbers are low by construction; they are a floor, not a verdict.
- ``wer``: word error rate of an English faster-whisper large-v3 pass over the render (no VAD,
  so short lines are not cut) against the cue's text, both normalized by
  ``fukikae.eval.asr.normalize``; pooled over words, not averaged over cues.
- ``fit_rate``: share of renders no longer than their slot before any rate fit or stretch;
  ``stretch_median`` and ``stretch_p90`` of the speed-up each would need (1.0 when it fits);
  ``fit_budget_rate``: share within ``[timefit] max_speedup``.
- ``emotion_agreement`` (when the tag stage labelled the source lines): share of labelled
  cues whose render the same emotion backend labels the same way. A backend that takes no
  delivery (Kokoro) scores only its neutral delivery against the labels.
- per card (ruling CD8): ``cps_median`` and quartiles, characters of text per second of the
  trimmed render, for per-card translate budgets later.

With ``--reference-workspace`` (BBT: the English original track's workspace, on the same
timeline as the dubbed track), each cue's slot is matched to the English words whose midpoint
falls inside it (``reference_span``), and the row adds ``ref_duration_ratio`` (median render
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
from fukikae.cards.select import gender_of, speaker_f0
from fukikae.cards.stock import pick_stock, revision, stock_card
from fukikae.cards.store import CardStore
from fukikae.config import BackendConfig, Config
from fukikae.eval.asr import normalize
from fukikae.gpu import device_string, free_mb, wait_for_vram
from fukikae.models import Assignments, DubCue, DubCues, Tags, Transcript, Turn, Word
from fukikae.workspace import Workspace

EMBEDDER = BackendConfig("wespeaker", {})
ASR = BackendConfig("faster_whisper", {"model": "large-v3", "vad_filter": False})
DEFAULT_TAGGER = "emotion2vec"
EXEMPLARS = 3
EXEMPLAR_MIN_S = 1.0
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


def sample_cues(
    cues: Mapping[str, DubCues], per_character: int = 8, seed: int = 0
) -> list[tuple[str, DubCue]]:
    """Up to ``per_character`` eligible cues per character over every episode, drawn with a
    generator seeded by ``seed`` and the character, returned in (episode, start) order."""
    by_char: dict[int, list[tuple[str, DubCue]]] = {}
    for episode in sorted(cues):
        for c in sorted(cues[episode].cues, key=lambda c: (c.start, c.id)):
            if eligible(c):
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
    transcript: Transcript, slot: tuple[float, float], pad: float = 0.3
) -> tuple[float, float] | None:
    """First start to last end of the words whose midpoint lies in the slot widened by
    ``pad``; None when there are none."""
    words = _words_in(transcript, slot, pad)
    if not words:
        return None
    return min(w.start for w in words), max(w.end for w in words)


def reference_speech(
    transcript: Transcript, slot: tuple[float, float], pad: float = 0.3
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
    """(substitutions + deletions + insertions, reference words) after normalization."""
    import jiwer

    h, r = normalize(hyp, language), normalize(ref, language)
    if not r:
        return len(h.split()), 0
    if not h:
        return len(r.split()), len(r.split())
    out = jiwer.process_words(r, h)
    return out.substitutions + out.deletions + out.insertions, len(r.split())


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


def _r(x: float) -> float:
    return round(float(x), 4)


def _metrics(scores: Sequence[CueScore], max_speedup: float) -> dict[str, float]:
    m: dict[str, float] = {"cues": len(scores)}
    sims = [s.similarity for s in scores if s.similarity is not None]
    if sims:
        m["speaker_similarity"] = _r(np.mean(sims))
    words = sum(s.ref_words for s in scores)
    if words:
        m["wer"] = _r(sum(s.errors for s in scores) / words)
    stretch = [s.stretch for s in scores]
    m["fit_rate"] = _r(np.mean([s.fits for s in scores]))
    m["fit_budget_rate"] = _r(np.mean([x <= max_speedup + 1e-9 for x in stretch]))
    finite = [x for x in stretch if math.isfinite(x)]
    if finite:
        m["stretch_median"] = _r(np.median(finite))
        m["stretch_p90"] = _r(np.percentile(finite, 90))
    m["cps_median"] = _r(np.median([s.cps for s in scores]))
    labelled = [s for s in scores if s.source_emotion]
    if labelled:
        m["emotion_agreement"] = _r(
            np.mean([s.render_emotion == s.source_emotion for s in labelled])
        )
        m["emotion_cues"] = len(labelled)
    if any(s.render_emotion for s in scores):
        m["render_neutral_share"] = _r(np.mean([s.render_emotion == "neutral" for s in scores]))
    ref = [s for s in scores if s.ref_seconds]
    if ref:
        m["ref_cues"] = len(ref)
        m["ref_duration_ratio"] = _r(
            np.median([s.seconds / cast(float, s.ref_seconds) for s in ref])
        )
        emo = [s for s in ref if s.ref_emotion]
        if emo:
            m["ref_emotion_agreement"] = _r(
                np.mean([s.render_emotion == s.ref_emotion for s in emo])
            )
        rsim = [s.ref_similarity for s in ref if s.ref_similarity is not None]
        if rsim:
            m["ref_speaker_similarity"] = _r(np.mean(rsim))
    return m


def summarize(scores: Sequence[CueScore], max_speedup: float) -> TtsSummary:
    """Per-character and overall metrics, and each card's speech rate (ruling CD8)."""
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
    return TtsSummary(per_character, overall, len(scores), per_card, sources)


HEADLINE_METRICS = (
    "speaker_similarity", "wer", "fit_rate", "fit_budget_rate", "stretch_median",
    "stretch_p90", "cps_median", "cues", "characters", "emotion_agreement", "emotion_cues",
    "render_neutral_share", "ref_duration_ratio", "ref_emotion_agreement",
    "ref_speaker_similarity", "ref_cues",
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
) -> list[Any]:
    """One ``tts`` ledger row for one TTS backend's renders."""
    from fukikae.eval.ledger import make_row

    model, rev = model_identity("tts", backend_cfg)
    metrics = {k: summary.overall[k] for k in HEADLINE_METRICS if k in summary.overall}
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
            detail={
                "per_character": {str(k): v for k, v in summary.per_character.items()},
                "per_card": summary.per_card,
                "similarity_source": {str(k): v for k, v in summary.similarity_source.items()},
            },
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


@contextlib.contextmanager
def _model(kind: str, bc: BackendConfig, device: str, cfg: Config) -> Iterator[Any]:
    inst = loader.create(kind, bc, device)
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


def _render(
    ws: Workspace,
    cfg: Config,
    store: CardStore,
    picked: list[tuple[str, DubCue]],
    cards: dict[int, str],
    names: dict[int, str],
) -> tuple[list[RenderedCue], dict[str, BackendConfig]]:
    from fukikae.stages.tts import TtsStage, _Lazy, cue_key, trim_silence, wav_seconds

    def who(cid: int) -> str:
        return f"character {cid}" + (f" ({names[cid]})" if cid in names else "")

    checked = {
        card: TtsStage._check(store, card, who(cid), cfg.target_language)
        for cid, card in sorted(cards.items())
    }
    backends = sorted({card.voice.backend for card, _ in checked.values()})
    bcs = {b: TtsStage._backend_cfg(cfg, b) for b in backends}
    idents = {b: identity_key("tts", bcs[b]) for b in backends}
    _, device_pref = TtsStage._settings(cfg)
    base = ws.root / ".fukikae" / "eval" / "tts"
    cache = ws.root / ".fukikae" / "tts-cache"
    out: list[RenderedCue] = []
    for b in backends:
        jobs = [
            (e, c)
            for e, c in picked
            if checked[cards[cast(int, c.character_id)]][0].voice.backend == b
        ]
        cls = checked[cards[cast(int, jobs[0][1].character_id)]][1]
        rev = idents[b].get("revision") or getattr(cls, "model_revision", None) or "unknown"
        folder = base / f"{b}-{rev}"
        folder.mkdir(parents=True, exist_ok=True)
        lazy = _Lazy(bcs[b], TtsStage._device(cfg, cls, device_pref), cfg)
        try:
            for e, c in jobs:
                cid = cards[cast(int, c.character_id)]
                card, cls = checked[cid]
                slot = c.end - c.start
                key = cue_key(
                    cid, cast(str, c.text), c.delivery, bcs[b],
                    capabilities=cls.capabilities, identity=idents[b], slot=slot,
                )  # fmt: skip
                dest = folder / f"{e}-{c.id}.{key}.wav"
                if not dest.exists():
                    if (cache / f"{key}.wav").exists():
                        with atomic_path(dest) as tmp:
                            shutil.copyfile(cache / f"{key}.wav", tmp)
                    else:
                        _synth(lazy, card, cls, c, slot, dest, store, trim_silence)
                out.append(RenderedCue(e, c, dest, wav_seconds(dest), cid, b, _voice_name(card)))
        finally:
            lazy.close()
    return out, bcs


def _voice_name(card: Card) -> str | None:
    v = card.voice
    if v.voice:
        return v.voice
    if v.voices:
        return "+".join(f"{n}:{w:g}" for n, w in v.voices)
    return v.source


def _synth(
    lazy: Any, card: Card, cls: Any, c: DubCue, slot: float, dest: Path, store: CardStore, trim: Any
) -> None:
    from fukikae.stages.tts import TtsStage, _apply_gain

    p = TtsStage._passed(c, card, cls, slot)
    with atomic_path(dest) as tmp:
        lazy.get().synthesize(
            cast(str, c.text),
            card.voice,
            card.target_language,
            tmp,
            resolve=store.sidecar_path,
            emotion_vector=p.get("emotion_vector"),
            instruction=p.get("instruction"),
            emotion_strength=p.get("emotion_strength", card.voice.emotion_strength),
            target_seconds=p.get("target_seconds"),
        )
        trim(tmp)
        gain = float(card.voice.post.get("gain_db", 0.0))
        if gain:
            _apply_gain(tmp, gain)


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


def evaluate(
    root: Path,
    episodes: Sequence[str],
    *,
    backend: str | None = None,
    per_character: int = 8,
    seed: int = 0,
    reference_workspace: Path | None = None,
) -> EvalResult:
    from fukikae.registry.store import Registry
    from fukikae.stages.tts import max_speedup

    ws = Workspace(root)
    cfg = ws.config
    target = cfg.target_language
    docs = _cues(ws, episodes, target)
    sources = {d.source_language for d in docs.values()}
    if len(sources) != 1:
        raise ValueError(f"episodes are translated from different languages: {sorted(sources)}")
    source_language = sources.pop()
    picked = sample_cues(docs, per_character, seed)
    if not picked:
        raise ValueError("no cue to render: every cue is an extra, skipped, failed or empty")
    store = CardStore(ws.root)
    chars = sorted({cast(int, c.character_id) for _, c in picked})
    with Registry(ws.registry_path) as reg:
        names = {c.id: c.name for c in reg.characters() if c.name}
        assigned: dict[int, str] = {}
        for cid in chars:
            a = reg.card_for(cid, target)
            if a is not None and a.card_id is not None:
                assigned[cid] = a.card_id
    notes: list[str] = []
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
    else:
        cards = _stock_cards(ws, cfg, store, backend, picked, assigned)
        mode = "stock"

    # 1. TTS backend(s).
    renders, bcs = _render(ws, cfg, store, picked, cards, names)
    small = {(r.episode, r.cue.id): _sixteen(r.path) for r in renders}

    # Original-voice references per character, or the card's clips.
    ref_spans: dict[int, list[tuple[Path, tuple[float, float]]]] = {}
    sim_source: dict[int, str] = {}
    for cid, card_ref in cards.items():
        card = store.get(card_ref)
        if card.voice.references:
            sim_source[cid] = "card"
            clips = []
            for ref in card.voice.references:
                src = store.sidecar_path(ref.clip)
                dest = ws.root / ".fukikae" / "eval" / "tts" / "refs" / f"{ref.clip[7:23]}.16k.wav"
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    to_16k(src, dest)
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

    # The original English line of each cue, from the reference workspace.
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
            cid = cast(int, r.cue.character_id)
            if cid in ref_vecs:
                sims[k] = similarity(vec, ref_vecs[cid])
            if k in ref_line:
                orig = emb.embed(ref_vocals[r.episode], [ref_line[k]])
                ref_sims[k] = similarity(vec, orig)

    # 3. faster-whisper.
    hyps: dict[tuple[str, str], str] = {}
    with _model("asr", ASR, gpu, cfg) as asr:
        for r in renders:
            k = (r.episode, r.cue.id)
            tr = asr.transcribe(small[k], target)
            hyps[k] = " ".join(s.text for s in tr.segments).strip()

    # 4. The tagger(s), on the CPU.
    render_emotion: dict[tuple[str, str], str | None] = {}
    ref_emotion: dict[tuple[str, str], str | None] = {}
    taggers: dict[str, list[RenderedCue]] = {}
    for r in renders:
        name = _tagger_name(ws, r.episode)
        wants = (name and r.cue.delivery.emotion) or (r.episode, r.cue.id) in ref_line
        if wants:
            taggers.setdefault(name or DEFAULT_TAGGER, []).append(r)
    for name, items in sorted(taggers.items()):
        with _model("tag", BackendConfig(name, {}), "cpu", cfg) as tagger:
            for r in items:
                k = (r.episode, r.cue.id)
                tags = tagger.tag(small[k], [Turn(start=0.0, end=r.seconds, speaker="render")])
                render_emotion[k] = _label(tags, 0.0)
            for e in sorted({r.episode for r in items if (r.episode, r.cue.id) in ref_line}):
                mine = [r for r in items if r.episode == e and (e, r.cue.id) in ref_line]
                turns = [
                    Turn(start=ref_line[(e, r.cue.id)][0], end=ref_line[(e, r.cue.id)][1],
                         speaker="original")
                    for r in mine
                ]  # fmt: skip
                tags = tagger.tag(ref_vocals[e], turns)
                for r in mine:
                    ref_emotion[(e, r.cue.id)] = _label(tags, ref_line[(e, r.cue.id)][0])

    scores: list[CueScore] = []
    for r in renders:
        k = (r.episode, r.cue.id)
        cid = cast(int, r.cue.character_id)
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
                source_emotion=r.cue.delivery.emotion if _tagger_name(ws, r.episode) else None,
                render_emotion=render_emotion.get(k),
                ref_seconds=None if line is None else line[1] - line[0],
                ref_emotion=ref_emotion.get(k),
                ref_similarity=ref_sims.get(k),
            )
        )
    speedup = max_speedup(cfg)
    summaries = {b: summarize([s for s in scores if s.backend == b], speedup) for b in sorted(bcs)}
    return EvalResult(scores, summaries, bcs, source_language, list(episodes), names, mode, notes)


def wav_seconds_16k(path: Path) -> float:
    import wave

    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# ---- output ---------------------------------------------------------------------------------

_COLUMNS = (
    ("speaker_similarity", "sim"), ("wer", "wer"), ("fit_rate", "fit"),
    ("stretch_median", "str50"), ("stretch_p90", "str90"), ("cps_median", "cps"),
    ("emotion_agreement", "emo"), ("ref_duration_ratio", "ref_dur"),
    ("ref_emotion_agreement", "ref_emo"), ("ref_speaker_similarity", "ref_sim"), ("cues", "n"),
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
    "duration_fit",
    "evaluate",
    "format_summary",
    "reference_span",
    "reference_speech",
    "sample_cues",
    "similarity",
    "summarize",
    "to_16k",
    "tts_rows",
    "word_error_rate",
    "word_errors",
]
