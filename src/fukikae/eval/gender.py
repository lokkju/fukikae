"""`fukikae eval gender`: how well a dubbed episode keeps each line's voice register (Part A
evaluation, the Sol Bianca diagnostic's measurements on the A0 estimator).

Every pitch figure comes from `fukikae.pitch` (the median of a span's voiced frames, 165 and
200 Hz bands), on the source line's ``delivery.reference`` span of the separated vocals, on the
render in ``07-tts/<cue>.wav`` and on each clone card's reference clips:

- cue gender agreement, source line against render, with the strict bands (``u`` between them
  or under ``min_voiced_s`` voiced, ``pitch.LINE_MIN_VOICED_S`` by default) and with a single
  cut at ``CUT_HZ`` (180 Hz), overall, per speaker and for the extras (extras now render from
  their own clones, so the render is what counts); the stock voices' nominal gender against
  the source line is reported beside it (``stock_voice_disagree``, the diagnostic's extras
  measure, which only a workspace from before D41 has anything for);
- reference purity per clone card by the pipeline's own gate rule (controller ruling, Task
  11): each clip's origin span in this episode gets the verdict `cards.auto.reference_gates`
  gives, as the tts stage's purity check does: refused when its class crosses the speaker's
  register (the other band, or between the bands under a register a full 4 st inside its
  band; no pitch gate for a near-even bimodal register, under ``pitch.BIMODAL_GATE_RATIO``;
  none for a clip inside the character's name-confirmed turns), under
  ``pitch.REF_VOICED_MIN`` voiced, or song (song-like and sung, or inside the accepted song
  spans). A clip with no origin in this episode isn't judged and passes, as the pipeline
  leaves it, and neither is any clip of a thin card (``[meta] reference_tier = "thin"``: the
  thin tier sets the gates aside, A2), so a thin card is never counted in ``impure_cards``.
  Beside it, as a diagnostic only, the spec's distance measure on the clip file itself: not
  song (inside the accepted song spans for ``pitch.REF_SONG_SHARE`` of its origin,
  or sung by `pitch.is_sung`), at least ``pitch.REF_VOICED_MIN`` of its loud frames voiced,
  and its median within ``pitch.REF_PITCH_ST`` semitones of the speaker's majority-register
  median; plus the pooled references' class against the speaker's majority (``card≠source``);
- each speaker's reference tier and seconds (``[meta] reference_tier``, or inferred from the
  clip lengths by the tier rules for a card written before tiers existed);
- gated lines (``gender_gate`` in the plan), and lines whose ``emotion_audio`` was passed
  against a source line in the other band from its speaker's established gender (the tts
  stage's own `gender_gate` inputs; must be 0 after A3);
- stock-voiced cues (must be 0 under D41), cues inside song spans, cues whose source text is a
  stock phrase (`hallucinations.is_stock_phrase`, whatever the audio under it), and the
  asrclean stage's dropped stretches (whole segments and word runs, counted apart);
- with ``similarity``, the WeSpeaker cosine (on the CPU) of each render against its own source
  line, for clone and stock cues and, given a ``before`` plan, for the cues that plan voiced
  with a stock card and this one doesn't. A ``before`` cue is matched to a cue here by its
  source text and a start within ``BEFORE_START_TOL_S``, never by cue id (cue ids are
  renumbered when asrclean drops a segment).

Read-only on the workspace: the pitch cache and the similarity's 16 kHz scratch (the renders
and, without ``vocals_16k.wav``, the vocals, resampled for the embedder) go under
``cache_root`` (the workspace's own by default, as the pipeline's), and the registry, when a
lock-kept assignment needs it, is read from a temporary copy. R1: this module imports
pipeline code; the pipeline never imports it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from fukikae import pitch
from fukikae.artifacts import read_json
from fukikae.cards.card import Card, CardError
from fukikae.cards.select import CLONE_MIN_S, Tier, reaches_floor, share_inside
from fukikae.cards.store import CardStore
from fukikae.models import Assignments, DubCue, DubCues, TtsPlan, TtsPlanCue
from fukikae.pitch import Gender

CUT_HZ = 180.0
BANDS = (pitch.MALE_BELOW, pitch.FEMALE_ABOVE)
WORST = 20
WHOLE = (0.0, 1e7)  # a span covering any render or clip file (pitch clips it to the file)
STOCK = "stock"
# A --before cue matches a cue here with the same source text starting within this.
BEFORE_START_TOL_S = 0.25


class Embedder(Protocol):
    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray: ...


def band(median: float | None, voiced_s: float, bands: tuple[float, float],
         min_voiced_s: float) -> Gender:  # fmt: skip
    """The class of a median: ``m`` under the lower band edge, ``f`` over the upper, else
    ``u`` (also with no median or under ``min_voiced_s`` voiced)."""
    if median is None or voiced_s < min_voiced_s:
        return "u"
    if median < bands[0]:
        return "m"
    if median > bands[1]:
        return "f"
    return "u"


def stock_voice_gender(voice: str | None) -> Gender:
    """A Kokoro voice id's nominal gender (``af_heart`` f, ``bm_daniel`` m)."""
    if voice and len(voice) > 1 and voice[1] in "fm":
        return "f" if voice[1] == "f" else "m"
    return "u"


@dataclass
class CueRow:
    cue_id: str
    start: float
    speaker: str  # "character 3", "extra SPEAKER_08", "extra unknown"
    character_id: int | None
    extra: str | None
    card_id: str | None
    card_source: str | None  # clone, stock, ... (None without a card)
    stock_voice: str | None
    source_hz: float | None
    source_voiced_s: float
    render_hz: float | None
    render_voiced_s: float
    source: Gender  # strict bands
    render: Gender
    source_cut: Gender  # the single cut
    render_cut: Gender
    voice: Gender  # the speaker's established gender (the tts stage's line gate input)
    emotion_audio: bool
    gated: bool
    text: str | None

    @property
    def known(self) -> bool:
        return self.source in "mf" and self.render in "mf"

    @property
    def mismatched(self) -> bool:
        return self.known and self.source != self.render

    @property
    def known_cut(self) -> bool:
        return self.source_cut in "mf" and self.render_cut in "mf"

    @property
    def mismatched_cut(self) -> bool:
        return self.known_cut and self.source_cut != self.render_cut


@dataclass
class WhoRow:
    speaker: str
    cues: int = 0
    known: int = 0
    mismatched: int = 0
    m_to_f: int = 0  # source m, render f
    f_to_m: int = 0
    known_cut: int = 0
    mismatched_cut: int = 0

    @property
    def rate(self) -> float | None:
        return self.mismatched / self.known if self.known else None


@dataclass
class ClipVerdict:
    clip: str
    seconds: float
    median: float | None  # the clip file's
    loud_voiced_fraction: float  # the clip file's
    passed: bool  # by the pipeline's gate rule (the headline)
    gates: list[str]  # the gates refusing its origin span (off_register, unvoiced, song_like)
    judged: bool  # its origin is in this episode, so the gate rule read it
    passed_4st: bool  # by the 4-semitone distance measure (diagnostic)
    why: str | None  # "song", "voicing" or "pitch" when the distance measure refuses it


@dataclass
class ReferenceRow:
    speaker: str
    card_id: str
    tier: Tier
    seconds: float
    majority: Gender
    majority_hz: float | None
    card_class: Gender
    card_hz: float | None
    clips: list[ClipVerdict]
    flags: list[str] = field(default_factory=list)

    @property
    def passing(self) -> int:
        return sum(c.passed for c in self.clips)

    @property
    def passing_4st(self) -> int:
        return sum(c.passed_4st for c in self.clips)


@dataclass
class GenderReport:
    workspace: str
    episode: str
    min_voiced_s: float
    cues: int
    known: int
    mismatched_strict: int
    known_cut: int
    mismatched_cut: int
    per_speaker: list[WhoRow]
    extras_known: int
    extras_disagree: int
    stock_voice_disagree: tuple[int, int]  # (disagreeing, known source) on stock-voiced cues
    purity: dict[int, tuple[int, int]]  # character -> (passing references, total): gate rule
    purity_4st: dict[int, tuple[int, int]]  # the same by the 4-semitone measure (diagnostic)
    references: list[ReferenceRow]
    tiers: dict[str, tuple[Tier, float]]
    gated_lines: int
    wrong_gender_prompts: int
    wrong_gender_cues: list[str]
    dropped_segments: int | None  # whole segments asrclean dropped; None: no asrclean output
    dropped_runs: int | None  # word runs it dropped from segments that keep other runs
    stock_phrase_cues: int  # cues whose source text is a stock phrase, whatever is under it
    song_cues: int
    stock_cues: int
    former_stock: list[str]  # cue ids a --before plan voiced with a stock card, not stock now
    before: bool  # a --before plan was given
    similarity: dict[str, Any] | None
    worst: list[CueRow]
    rows: list[CueRow]
    notes: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {
            "cues": self.cues,
            "known": self.known,
            "mismatched_strict": self.mismatched_strict,
            "mismatch_rate": _share(self.mismatched_strict, self.known),
            "known_cut": self.known_cut,
            "mismatched_cut": self.mismatched_cut,
            "mismatch_rate_cut": _share(self.mismatched_cut, self.known_cut),
            "extras_known": self.extras_known,
            "extras_disagree": self.extras_disagree,
            "stock_voice_disagree": self.stock_voice_disagree[0],
            "stock_voice_known": self.stock_voice_disagree[1],
            "purity_passing": sum(p for p, _ in self.purity.values()),
            "purity_total": sum(t for _, t in self.purity.values()),
            "purity_4st_passing": sum(p for p, _ in self.purity_4st.values()),
            "purity_4st_total": sum(t for _, t in self.purity_4st.values()),
            "impure_cards": sum("impure" in r.flags for r in self.references),
            "short_speakers": sum(t == "short" for t, _ in self.tiers.values()),
            "thin_speakers": sum(t == "thin" for t, _ in self.tiers.values()),
            "gated_lines": self.gated_lines,
            "wrong_gender_prompts": self.wrong_gender_prompts,
            "stock_phrase_cues": self.stock_phrase_cues,
            "song_cues": self.song_cues,
            "stock_cues": self.stock_cues,
            "former_stock_cues": len(self.former_stock),
        }
        if self.dropped_segments is not None:
            out["dropped_segments"] = self.dropped_segments
        if self.dropped_runs is not None:
            out["dropped_runs"] = self.dropped_runs
        for k, v in (self.similarity or {}).items():
            if isinstance(v, dict) and v.get("mean") is not None:
                out[f"similarity_{k}"] = round(float(v["mean"]), 4)
        return out


def _share(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def _speaker(c: DubCue) -> str:
    if c.character_id is not None:
        return f"character {c.character_id}"
    return f"extra {c.extra or 'unknown'}"


def _tier_label(c: DubCue) -> str:
    """The speaker a card is keyed by: a character, an extras cluster, or (a cluster-less
    line, cloned from itself) ``extra unknown~<cue>``."""
    from fukikae.cards.auto import extras_label

    return (
        f"character {c.character_id}" if c.character_id is not None else f"extra {extras_label(c)}"
    )


def _cues_path(work: Path, target: str | None) -> Path:
    tr = work / "06-translate"
    if target and (tr / f"cues.{target}.json").exists():
        return tr / f"cues.{target}.json"
    found = sorted(tr.glob("cues.*.json"))
    if not found:
        raise FileNotFoundError(f"{tr}: no cues.<language>.json (run the translate stage)")
    return found[0]


def _work_dir(root: Path, episode: str) -> Path:
    from fukikae.workspace import Workspace

    ws = Workspace(root)
    work = ws.work_dir(episode.strip().upper())
    if work.is_dir():
        return work
    work = ws.work_dir(episode.strip())
    if work.is_dir():
        return work
    return ws.find_episode(episode).work


def _pooled(tracks: Sequence[pitch.F0Track]) -> pitch.F0Summary:
    if not tracks:
        return pitch.summarize(pitch.F0Track(np.empty(0), np.empty(0, np.int64), 0.0, np.empty(0)))
    f0 = np.concatenate([t.f0 for t in tracks])
    loud = np.concatenate(
        [t.loud if t.loud is not None else np.ones(len(t.f0), bool) for t in tracks]
    )
    seconds = float(sum(t.seconds for t in tracks))
    return pitch.summarize(
        pitch.F0Track(f0, np.zeros(len(f0), np.int64), seconds, np.array([seconds]), loud)
    )


def _semitones(a: float, b: float) -> float:
    return abs(12 * math.log2(a / b))


def _song_spans(work: Path, cfg: Any) -> list[tuple[float, float]]:
    from fukikae.cards.select import song_spans
    from fukikae.stages.mix import mix_settings

    try:
        return song_spans(work, mix_settings(cfg).songs)
    except Exception:  # noqa: BLE001 - a measurement aid; no spans is an honest answer
        return []


def _before_cues(before: Path) -> list[DubCue]:
    """The cues of the ``before`` plan's own episode: ``06-translate/cues.<language>.json``
    beside its ``07-tts``."""
    tr = before.resolve().parent.parent / "06-translate"
    found = sorted(tr.glob("cues.*.json")) if tr.is_dir() else []
    if not found:
        raise FileNotFoundError(
            f"{before}: no cues.<language>.json in {tr}; --before needs the plan inside its "
            "episode (07-tts/plan.json beside 06-translate)"
        )
    return list(read_json(found[0], DubCues).cues)


def match_before(cues: Sequence[DubCue], before: Sequence[DubCue],
                 tol: float = BEFORE_START_TOL_S) -> dict[str, str]:  # fmt: skip
    """Cue id here -> the ``before`` cue id with the same source text whose start is nearest
    and within ``tol`` seconds; each ``before`` cue matches at most once."""
    by_text: dict[str, list[DubCue]] = defaultdict(list)
    for b in before:
        by_text[b.source_text].append(b)
    taken: set[str] = set()
    out: dict[str, str] = {}
    for c in sorted(cues, key=lambda c: c.start):
        near = [
            b for b in by_text.get(c.source_text, [])
            if b.id not in taken and abs(b.start - c.start) <= tol
        ]  # fmt: skip
        if near:
            b = min(near, key=lambda b: abs(b.start - c.start))
            taken.add(b.id)
            out[c.id] = b.id
    return out


def _before_stock(before: Path, store: CardStore) -> set[str]:
    """Cue ids of the ``before`` plan voiced with a stock card: by that plan's own workspace
    card store when the plan sits in a workspace (``<ws>/.fukikae/episodes/<ep>/07-tts``),
    else by this workspace's, else by its backend (``kokoro``)."""
    plan = read_json(before, TtsPlan)
    stores = [store]
    parents = before.resolve().parents
    if len(parents) > 4 and (parents[3] / "cards").is_dir():
        stores.insert(0, CardStore(parents[4]))
    out = set()
    for p in plan.cues:
        if p.card_id is None or p.skipped:
            continue
        source = None
        for s in stores:
            try:
                source = s.get(p.card_id).voice.source
                break
            except (CardError, OSError):
                continue
        if source == STOCK or (source is None and p.backend == "kokoro"):
            out.add(p.cue_id)
    return out


def _gate_owner(root: Path, scratch: Path) -> Callable[[str], int | None]:
    """`cards.auto.key_owner` over a copy of the workspace's registry made under ``scratch``
    on first use (the registry opens read-write, so the workspace's own is never opened)."""
    import shutil
    import tempfile

    from fukikae.cards.auto import key_owner
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

    path = Workspace(root).registry_path
    cache: dict[str, int | None] = {}

    def owner(key: str) -> int | None:
        if key not in cache:
            if not path.exists():
                cache[key] = None
            else:
                scratch.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=scratch) as tmp:
                    for f in path.parent.glob(path.name + "*"):
                        shutil.copy2(f, Path(tmp) / f.name)
                    with Registry(Path(tmp) / path.name) as reg:
                        cache[key] = key_owner(reg)(key)
        return cache[key]

    return owner


def _similarity(
    vocals: Path,
    rows: Sequence[CueRow],
    cues: Mapping[str, DubCue],
    renders: Mapping[str, Path],
    former: Sequence[str],
    embedder: Embedder,
    scratch: Path,
) -> dict[str, Any]:
    """The embedder reads 16 kHz mono: the separate stage's ``vocals_16k.wav`` when present
    (else a 16 kHz copy under ``scratch``), and 16 kHz copies of the renders under ``scratch``
    (under the pitch cache root; the workspace is never written)."""
    from fukikae.eval.tts import to_16k

    scratch.mkdir(parents=True, exist_ok=True)
    have = [r for r in rows if r.cue_id in renders]
    spans = [(float(cues[r.cue_id].delivery.reference[0]),
              float(cues[r.cue_id].delivery.reference[1])) for r in have]  # fmt: skip
    v16 = vocals.with_name("vocals_16k.wav")
    if not v16.exists():
        v16 = to_16k(vocals, scratch / "vocals.16k.wav")
    src = embedder.embed(v16, spans) if have else np.zeros((0, 1))
    cos: dict[str, float] = {}
    for i, r in enumerate(have):
        wav = to_16k(renders[r.cue_id], scratch / f"{r.cue_id}.wav")
        rend = embedder.embed(wav, [WHOLE])[0]
        cos[r.cue_id] = float(np.dot(np.asarray(src[i]), np.asarray(rend)))

    def group(ids: Sequence[str]) -> dict[str, Any]:
        vals = [cos[i] for i in ids if i in cos]
        return {"cues": len(vals), "mean": float(np.mean(vals)) if vals else None}

    by_speaker: dict[str, list[str]] = defaultdict(list)
    for r in have:
        by_speaker[r.speaker].append(r.cue_id)
    return {
        "clone": group([r.cue_id for r in have if r.card_source != STOCK]),
        "stock": group([r.cue_id for r in have if r.card_source == STOCK]),
        "former_stock": group(list(former)),
        "by_speaker": {k: group(v) for k, v in sorted(by_speaker.items())},
        "per_cue": {k: round(v, 4) for k, v in cos.items()},
    }


def wespeaker_cpu() -> Embedder:
    """fukikae's embed backend (WeSpeaker ResNet34) loaded on the CPU."""
    from fukikae.backends.loader import create
    from fukikae.config import BackendConfig

    emb = create("embed", BackendConfig("wespeaker", {}), "cpu")
    emb.load()
    return emb  # type: ignore[return-value]


def measure(
    ws: Path,
    episode: str,
    *,
    similarity: bool = False,
    before: Path | None = None,
    cache_root: Path | None = None,
    min_voiced_s: float = pitch.LINE_MIN_VOICED_S,
    embedder: Embedder | None = None,
    embedder_factory: Callable[[], Embedder] | None = None,
) -> GenderReport:
    """Score one dubbed episode of workspace ``ws`` (see the module docstring)."""
    from fukikae import hallucinations
    from fukikae.cards.auto import (
        _spoken,  # pyright: ignore[reportPrivateUsage]
        character_spans,
        checked_names,
        confirmed_labels,
        reference_gates,
    )
    from fukikae.cards.select import reference_gating, speaker_register
    from fukikae.stages import tts
    from fukikae.stages.mix import mix_settings
    from fukikae.workspace import Workspace

    root = Path(ws)
    cfg = Workspace(root).config
    work = _work_dir(root, episode)
    cache = Path(cache_root) if cache_root is not None else root
    vocals = work / "01-separate" / "vocals.wav"
    doc = read_json(_cues_path(work, cfg.target_language), DubCues)
    cues = list(doc.cues)
    by_id = {c.id: c for c in cues}
    plan_path = work / "07-tts" / "plan.json"
    plan: dict[str, TtsPlanCue] = (
        {p.cue_id: p for p in read_json(plan_path, TtsPlan).cues} if plan_path.exists() else {}
    )
    notes: list[str] = []
    if not plan:
        notes.append(f"{plan_path}: no tts plan; nothing rendered to score")
    asg_path = work / "05-characters" / "assignments.json"
    asg = read_json(asg_path, Assignments) if asg_path.exists() else None
    store = CardStore(root)
    cards: dict[str, Card | None] = {}

    def card_of(cid: str | None) -> Card | None:
        if cid is None:
            return None
        if cid not in cards:
            try:
                cards[cid] = store.get(cid)
            except (CardError, OSError) as e:
                notes.append(f"card {cid}: {e}")
                cards[cid] = None
        return cards[cid]

    gate = tts.gender_gate(work, cues, cache)

    # -- per cue ---------------------------------------------------------------------------
    renders: dict[str, Path] = {}
    rows: list[CueRow] = []
    for c in cues:
        p = plan.get(c.id)
        card = card_of(p.card_id if p else None)
        line = gate.line(c.id)
        src_hz, src_v = (line.median, line.voiced_s) if line is not None else (None, 0.0)
        wav = work / "07-tts" / f"{c.id}.wav"
        rend_hz, rend_v = None, 0.0
        if p is not None and p.key and wav.exists():
            renders[c.id] = wav
            s = pitch.summarize(pitch.cached_track(cache, wav, [WHOLE]))
            rend_hz, rend_v = s.median, s.voiced_s
        cut = (CUT_HZ, CUT_HZ)
        rows.append(CueRow(
            cue_id=c.id, start=float(c.start), speaker=_speaker(c), character_id=c.character_id,
            extra=c.extra, card_id=p.card_id if p else None,
            card_source=card.voice.source if card else None,
            stock_voice=card.voice.voice if card and card.voice.source == STOCK else None,
            source_hz=src_hz, source_voiced_s=src_v, render_hz=rend_hz, render_voiced_s=rend_v,
            source=band(src_hz, src_v, BANDS, min_voiced_s),
            render=band(rend_hz, rend_v, BANDS, min_voiced_s),
            source_cut=band(src_hz, src_v, cut, min_voiced_s),
            render_cut=band(rend_hz, rend_v, cut, min_voiced_s),
            voice=gate.voice(c.id),
            emotion_audio=bool(p and p.passed.get("emotion_audio")),
            gated=bool(p and p.gender_gate), text=c.text,
        ))  # fmt: skip

    who: dict[str, WhoRow] = {}
    for r in rows:
        w = who.setdefault(r.speaker, WhoRow(r.speaker))
        w.cues += 1
        w.known += r.known
        w.mismatched += r.mismatched
        w.m_to_f += r.mismatched and r.source == "m"
        w.f_to_m += r.mismatched and r.source == "f"
        w.known_cut += r.known_cut
        w.mismatched_cut += r.mismatched_cut
    per_speaker = sorted(who.values(), key=lambda w: (-w.cues, w.speaker))
    extras = [r for r in rows if r.character_id is None]
    stock_rows = [r for r in rows if r.card_source == STOCK]
    stock_known = [r for r in stock_rows if r.source in "mf"]
    stock_dis = [r for r in stock_known if stock_voice_gender(r.stock_voice) not in (r.source, "u")]
    wrong = [
        c.id for c in cues
        if plan.get(c.id) is not None and plan[c.id].passed.get("emotion_audio")
        and tts.wrong_gender(gate.line(c.id), gate.voice(c.id))
    ]  # fmt: skip

    # -- references, tiers -----------------------------------------------------------------
    songs = _song_spans(work, cfg)
    char_spans = character_spans(asg) if asg is not None else {}
    try:
        song_mask = reference_gating(work, mix_settings(cfg).songs)
    except Exception:  # noqa: BLE001 - a measurement aid, as _song_spans
        song_mask = None
    names, _why = checked_names(work, asg) if asg is not None else (None, None)
    owner = _gate_owner(root, pitch.cache_dir(cache) / "registry")
    turns = list(asg.turns or []) if asg is not None else []
    speakers: dict[str, tuple[str, list[tuple[float, float]]]] = {}
    for c in cues:
        p = plan.get(c.id)
        card = card_of(p.card_id if p else None)
        if p is None or p.card_id is None or card is None or card.voice.source == STOCK:
            continue
        label = _tier_label(c)
        if label in speakers:
            continue
        if c.character_id is not None:
            spans = char_spans.get(c.character_id) or [
                tuple(x.delivery.reference) for x in cues if x.character_id == c.character_id
            ]
        elif c.extra:
            spans = [(float(t.start), float(t.end)) for t in turns if t.speaker == c.extra]
            # a cluster with no turn here: its cues' own spans are its speech, as extras_cards
            # reads them
            spans = spans or [
                tuple(x.delivery.reference) for x in cues
                if x.character_id is None and x.extra == c.extra and _spoken(x)
            ]  # fmt: skip
        else:
            spans = []
        spans = spans or [(float(c.delivery.reference[0]), float(c.delivery.reference[1]))]
        speakers[label] = (p.card_id, [(float(a), float(b)) for a, b in spans])

    references: list[ReferenceRow] = []
    tiers: dict[str, tuple[Tier, float]] = {}
    purity: dict[int, tuple[int, int]] = {}
    purity_4st: dict[int, tuple[int, int]] = {}
    bands = (float(BANDS[0]), float(BANDS[1]))
    for label, (card_id, spans) in speakers.items():
        card = card_of(card_id)
        assert card is not None
        track = pitch.cached_track(cache, vocals, spans) if vocals.exists() else None
        if track is not None:
            majority, msum = pitch.majority_register(track, BANDS)
            mixed = pitch.bimodal(pitch.band_seconds(track, BANDS))
        else:
            majority, msum, mixed = "u", None, False
        maj_hz = msum.median if msum is not None else None
        # the pipeline's gate inputs: the speaker's turns, register and confirmed turns
        cid = int(label.split()[1]) if label.startswith("character ") else None
        own = char_spans.get(cid, []) if cid is not None else spans
        register = speaker_register(vocals, own, bands, cache) if own and vocals.exists() else None
        confirmed: list[tuple[float, float]] = []
        if cid is not None and names is not None and asg is not None:
            labels = confirmed_labels(asg, cid, owner)
            confirmed = [(float(t.start), float(t.end)) for t in turns if t.speaker in labels]
        # a thin clone sets the gates aside (A2), and the pipeline never judges its references
        # (cards.auto.is_thin reads [meta] alone, so a card from before tiers is judged)
        meta_tier = card.meta.get("reference_tier")
        thin = meta_tier == "thin"
        clips: list[ClipVerdict] = []
        clip_tracks = []
        for ref in card.voice.references:
            try:
                path = store.sidecar_path(ref.clip)
            except (CardError, OSError) as e:
                notes.append(f"card {card_id}: clip {ref.clip}: {e}")
                continue
            t = pitch.cached_track(cache, path, [WHOLE])
            clip_tracks.append(t)
            s = pitch.summarize(t)
            o = ref.origin or {}
            in_song = (
                o.get("episode") == work.name and "start" in o and "end" in o
                and share_inside(float(o["start"]), float(o["end"]), songs) >= pitch.REF_SONG_SHARE
            )  # fmt: skip
            why = None
            if in_song or pitch.is_sung(pitch.stability(t)):
                why = "song"
            elif s.median is None or s.loud_voiced_fraction < pitch.REF_VOICED_MIN:
                why = "voicing"
            elif maj_hz is not None and _semitones(s.median, maj_hz) > pitch.REF_PITCH_ST:
                why = "pitch"
            judged = (
                not thin
                and o.get("episode") == work.name and "start" in o and "end" in o
                and vocals.exists()
            )  # fmt: skip
            gates: list[str] = []
            if judged:
                _m, refused = reference_gates(
                    vocals, (float(o["start"]), float(o["end"])), own_spans=own,
                    confirmed_spans=confirmed, register=register, song_like=song_mask,
                    song_spans=songs, bands=bands, pitch_root=cache,
                )  # fmt: skip
                gates = list(refused)
            loud = round(s.loud_voiced_fraction, 3)
            clips.append(ClipVerdict(
                ref.clip, round(t.seconds, 3), s.median, loud, not gates, gates, judged,
                why is None, why,
            ))  # fmt: skip
        pooled = _pooled(clip_tracks)
        card_class = pitch.classify(pooled, *BANDS).gender
        seconds = round(sum(c.seconds for c in clips), 3)
        tier: Tier
        if meta_tier in ("floor", "short", "thin"):
            tier = meta_tier
        elif reaches_floor([c.seconds for c in clips]):
            tier = "floor"
        else:
            tier = "short" if seconds >= CLONE_MIN_S else "thin"
        flags = []
        if any(not c.passed for c in clips):
            flags.append("impure")
        if tier != "floor":
            flags.append(tier)
        if card_class in "mf" and majority in "mf" and card_class != majority:
            flags.append("card≠source")
        if mixed:
            flags.append("mixed")
        row = ReferenceRow(label, card_id, tier, seconds, majority, maj_hz, card_class,
                           pooled.median, clips, flags)  # fmt: skip
        references.append(row)
        tiers[label] = (tier, seconds)
        if cid is not None:
            purity[cid] = (row.passing, len(clips))
            purity_4st[cid] = (row.passing_4st, len(clips))

    # -- songs, hallucinations, stock ------------------------------------------------------
    song_cues = sum(any(a <= (c.start + c.end) / 2 <= b for a, b in songs) for c in cues)
    language = doc.source_language or cfg.source_language or ""
    stock_phrase = sum(hallucinations.is_stock_phrase(c.source_text, language) for c in cues)
    dropped_path = work / "03b-asrclean" / "dropped.json"
    dropped_segments = dropped_runs = None
    if dropped_path.exists():
        rows_d = json.loads(dropped_path.read_text(encoding="utf-8"))
        dropped_runs = sum(d.get("kind") == "run" for d in rows_d)
        dropped_segments = len(rows_d) - dropped_runs
    former: list[str] = []
    if before is not None:
        stock_before = _before_stock(Path(before), store)
        matched = match_before(cues, _before_cues(Path(before)))
        now_stock = {r.cue_id for r in rows if r.card_source == STOCK}
        former = [
            c.id for c in cues
            if matched.get(c.id) in stock_before and c.id not in now_stock
        ]  # fmt: skip

    sim = None
    if similarity:
        emb = embedder if embedder is not None else (embedder_factory or wespeaker_cpu)()
        scratch = pitch.cache_dir(cache) / f"16k-{work.name}"
        sim = _similarity(vocals, rows, by_id, renders, former, emb, scratch)

    mismatched = [r for r in rows if r.mismatched]
    worst = sorted(
        mismatched,
        key=lambda r: -abs(math.log2((r.render_hz or 1.0) / (r.source_hz or 1.0))),
    )[:WORST]
    return GenderReport(
        workspace=str(root), episode=work.name, min_voiced_s=min_voiced_s, cues=len(rows),
        known=sum(r.known for r in rows), mismatched_strict=len(mismatched),
        known_cut=sum(r.known_cut for r in rows),
        mismatched_cut=sum(r.mismatched_cut for r in rows), per_speaker=per_speaker,
        extras_known=sum(r.known for r in extras),
        extras_disagree=sum(r.mismatched for r in extras),
        stock_voice_disagree=(len(stock_dis), len(stock_known)), purity=purity,
        purity_4st=purity_4st, references=references, tiers=tiers,
        gated_lines=sum(r.gated for r in rows), wrong_gender_prompts=len(wrong),
        wrong_gender_cues=wrong, dropped_segments=dropped_segments, dropped_runs=dropped_runs,
        stock_phrase_cues=stock_phrase, song_cues=song_cues, stock_cues=len(stock_rows),
        former_stock=former, before=before is not None, similarity=sim, worst=worst, rows=rows,
        notes=notes,
    )  # fmt: skip


# -- output -----------------------------------------------------------------------------------


def _hz(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f}"


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "-"


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def format_report(r: GenderReport) -> str:
    """The report as Markdown."""
    gate_p = sum(p for p, _ in r.purity.values())
    gate_t = sum(t for _, t in r.purity.values())
    st_p = sum(p for p, _ in r.purity_4st.values())
    if r.dropped_segments is None:
        dropped = "-"
    else:
        dropped = f"{r.dropped_segments} whole segments, {r.dropped_runs} word runs"
    out = [
        f"# eval gender: {r.episode}",
        "",
        f"cue gender mismatch: strict {r.mismatched_strict}/{r.known} known "
        f"({_pct(r.mismatched_strict, r.known)}); single {CUT_HZ:.0f} Hz cut "
        f"{r.mismatched_cut}/{r.known_cut} ({_pct(r.mismatched_cut, r.known_cut)}); "
        f"{r.cues} cues, lines under {r.min_voiced_s:g} s voiced are u",
        f"character references passing the pipeline's gates: {gate_p}/{gate_t}; "
        f"within {pitch.REF_PITCH_ST:g} st of the majority (diagnostic): {st_p}/{gate_t}",
        f"extras: render disagrees with the source on {r.extras_disagree} of {r.extras_known}; "
        f"stock voice's gender against the source {r.stock_voice_disagree[0]} of "
        f"{r.stock_voice_disagree[1]}",
        f"stock-voiced cues: {r.stock_cues}",
        f"gated lines: {r.gated_lines}; emotion audio passed against a wrong-gender line: "
        f"{r.wrong_gender_prompts}",
        f"cues in song spans: {r.song_cues}; cues whose source text is a stock phrase: "
        f"{r.stock_phrase_cues}; asrclean dropped: {dropped}",
        "",
        "| who | cues | known | mismatched | rate | src m, render f | src f, render m | "
        f"{CUT_HZ:.0f} Hz cut |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for w in r.per_speaker:
        rate = "-" if w.rate is None else f"{100 * w.rate:.0f}%"
        out.append(f"| {w.speaker} | {w.cues} | {w.known} | {w.mismatched} | {rate} | "
                   f"{w.m_to_f} | {w.f_to_m} | {w.mismatched_cut}/{w.known_cut} |")  # fmt: skip
    out += [
        f"| **all** | {r.cues} | {r.known} | **{r.mismatched_strict}** | "
        f"{_pct(r.mismatched_strict, r.known)} | | | {r.mismatched_cut}/{r.known_cut} |",
        "",
        "| speaker | card | tier | s | refs passing (gates) | refused by | "
        f"within {pitch.REF_PITCH_ST:g} st (diagnostic) | ref medians (Hz) | card g | majority | "
        "majority Hz | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in r.references:
        refused = "; ".join(f"{_hz(c.median)} Hz: {', '.join(c.gates)}" for c in x.clips if c.gates)
        meds = ", ".join(_hz(c.median) + ("" if c.passed_4st else f" ({c.why})") for c in x.clips)
        out.append(f"| {x.speaker} | {x.card_id[:8]} | {x.tier} | {x.seconds:.1f} | "
                   f"{x.passing}/{len(x.clips)} | {refused or '-'} | "
                   f"{x.passing_4st}/{len(x.clips)} | {meds} | {x.card_class} | {x.majority} | "
                   f"{_hz(x.majority_hz)} | {' '.join(x.flags)} |")  # fmt: skip
    if r.worst:
        out += ["", "| cue | time | who | voice | src Hz | render Hz | text |",
                "|---|---|---|---|---|---|---|"]  # fmt: skip
        for c in r.worst:
            voice = c.stock_voice or c.card_source or "-"
            text = (c.text or "").replace("|", "/")[:60]
            out.append(f"| {c.cue_id} | {_clock(c.start)} | {c.speaker} | {voice} | "
                       f"{_hz(c.source_hz)} | {_hz(c.render_hz)} | {text} |")  # fmt: skip
    if r.similarity is not None:
        out += ["", "| group | cues | render vs source cosine |", "|---|---|---|"]
        for k in ("clone", "stock", "former_stock"):
            g = r.similarity[k]
            mean = "-" if g["mean"] is None else f"{g['mean']:.2f}"
            out.append(f"| {k.replace('_', ' ')} | {g['cues']} | {mean} |")
    for n in r.notes:
        out.append(f"\nnote: {n}")
    return "\n".join(out) + "\n"


def _cue_json(c: CueRow) -> dict[str, Any]:
    d = asdict(c)
    d.update(known=c.known, mismatched=c.mismatched, known_cut=c.known_cut,
             mismatched_cut=c.mismatched_cut)  # fmt: skip
    return d


def options(r: GenderReport) -> dict[str, Any]:
    return {
        "bands": list(BANDS),
        "cut": int(CUT_HZ),
        "min_voiced_s": r.min_voiced_s,
        "pitch_version": pitch.VERSION,
        "similarity": r.similarity is not None,
        "before": r.before or None,
    }


def report_json(r: GenderReport) -> dict[str, Any]:
    """The report as one JSON document (``docs/results/``-ready)."""
    return {
        "workspace": r.workspace,
        "episode": r.episode,
        "options": options(r),
        "metrics": r.metrics(),
        "per_speaker": [asdict(w) | {"rate": w.rate} for w in r.per_speaker],
        "references": [
            asdict(x) | {"passing": x.passing, "passing_4st": x.passing_4st} for x in r.references
        ],
        "purity": {str(k): list(v) for k, v in r.purity.items()},
        "purity_4st": {str(k): list(v) for k, v in r.purity_4st.items()},
        "tiers": {k: list(v) for k, v in r.tiers.items()},
        "wrong_gender_cues": r.wrong_gender_cues,
        "former_stock": r.former_stock,
        "similarity": r.similarity,
        "worst": [c.cue_id for c in r.worst],
        "cues": [_cue_json(c) for c in r.rows],
        "notes": r.notes,
    }


def gender_rows(r: GenderReport, *, cfg: Any, test_set: str, split: str,
                tag: str | None) -> list[Any]:  # fmt: skip
    """One ``gender`` ledger row for the episode (GT2-additive stage)."""
    from fukikae.eval.ledger import make_row

    return [
        make_row(
            stage="gender",
            backend="fukikae.pitch",
            model=None,
            revision=None,
            options={k: v for k, v in options(r).items() if v is not None},
            language=cfg.target_language or "",
            genre=cfg.series.genre,
            series=cfg.series.title,
            test_set=test_set,
            split=split,
            episodes=[r.episode],
            metrics=r.metrics(),
            detail={
                "per_speaker": {w.speaker: [w.mismatched, w.known] for w in r.per_speaker},
                "purity": {str(k): list(v) for k, v in r.purity.items()},
                "purity_4st": {str(k): list(v) for k, v in r.purity_4st.items()},
                "tiers": {k: list(v) for k, v in r.tiers.items()},
                "flags": {x.speaker: x.flags for x in r.references if x.flags},
                "wrong_gender_cues": r.wrong_gender_cues,
                "worst": [c.cue_id for c in r.worst],
                "source_language": cfg.source_language,
            },
            workspace=r.workspace,
            tag=tag,
        )
    ]
