"""The season index, ``<ws>/studio/season.json``: what the studio shows, derived from the
workspace and never edited (Task 16).

`build_season` reads every episode with a work dir. An episode whose characters stage has
written ``05-characters/assignments.json`` is ready; the others are listed with the command
that brings them in. Per character (every registry character speaking in a ready episode):
its speech per episode, sample clips spread over the season, A0's gender over the season and
per episode beside the stored one, its current card per target language, merge suggestions
from registry centroids (`MERGE_SUGGEST_COS`, narrators never paired with dialogue
characters, an ignored character never paired), split suggestions for a bimodal episode
cluster (`pitch.bimodal`), names evidence (`fukikae.names.flags`, shared with `names show`)
and flags. Extras clusters are listed per episode with their tier and samples. ``queue``
orders the characters for review (`order_queue`).

An episode without its vocals stem is still listed, with a note, and gives no samples or pitch.

The index detects nothing on its own and writes nothing but ``season.json``, the sample clips
under ``<ws>/studio/audio/`` (a build deletes the clips its document no longer refers to,
`prune_clips`) and the pitch and song caches: the registry is only read (opening an older one
migrates it, as every command does), and no stock voice is ever proposed (D41).
`load_or_build` rebuilds it when its inputs' digest changes: each episode's assignments,
turns, transcripts, names files, cues, tts plan and run, dropped hallucinations and manifest,
the vocals' size and time, the resolved config (profile, user config and ``fukikae.toml``,
`config_digest`) and the registry's contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fukikae.models import Assignments, DubCues, Transcript, TtsPlan, TtsRun, Turn
from fukikae.studio.model import (
    CardInfo,
    CharacterEntry,
    EpisodeEntry,
    EpisodeSpeech,
    ExtraEntry,
    Flag,
    GenderInfo,
    Sample,
    SeasonDoc,
    Suggestion,
)
from fukikae.workspace import Workspace

if TYPE_CHECKING:
    from fukikae.cards.store import CardStore
    from fukikae.config import Config
    from fukikae.names.models import NamesReport, NamesResult
    from fukikae.pitch import F0Track
    from fukikae.registry.store import Character, Registry

# Bumped when the index built from the same inputs changes shape or content.
SEASON_VERSION = 2
# Registry centroid cosine at or over which two characters are suggested as one (spec, Part B).
MERGE_SUGGEST_COS = 0.5
MERGE_NEAREST = 3  # merge suggestions per character, nearest first
SAMPLE_MAX_S = 12.0  # a sample clip is the turn's first this many seconds
PART_SAMPLES = 2  # samples per part of a split suggestion
EXTRA_CLIPS = 2  # samples per extras cluster
# flag kinds by severity for the queue (an operator gender the card contradicts first, by
# ruling; then the spec's thin, stock kept, impure, bimodal, gated lines, names disagreements,
# below the floor); other kinds rank after these, unflagged last
SEVERITY = ("gender", "thin", "stock", "impure", "bimodal", "gated", "names", "floor")
# the stages `fukikae run` goes through, in order, for an episode's last stage done
STAGE_ORDER = (
    "extract", "separate", "language", "asr", "asrclean", "diarize", "names", "characters",
    "tag", "translate", "tts", "timefit", "mix",
)  # fmt: skip
TIERS = ("floor", "short", "thin")
# a sample clip's file name under studio/audio (`sample_name`); pruning touches only these
CLIP_NAME = re.compile(r"[0-9a-f]{64}\.wav")
EXTRA_STATUSES = ("extra", "review")  # clusters with no character: their cues are extras


def season_path(ws: Workspace | Path) -> Path:
    return _workspace(ws).root / "studio" / "season.json"


def _workspace(ws: Workspace | Path) -> Workspace:
    return ws if isinstance(ws, Workspace) else Workspace(Path(ws))


def episode_ids(ws: Workspace) -> list[str]:
    """Every episode with a work dir, in id order."""
    base = ws.root / ".fukikae" / "episodes"
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


def _assignments_path(ws: Workspace, episode: str) -> Path:
    return ws.work_dir(episode) / "05-characters" / "assignments.json"


def _vocals(ws: Workspace, episode: str) -> Path:
    return ws.work_dir(episode) / "01-separate" / "vocals.wav"


def _last_stage(ws: Workspace, episode: str) -> str | None:
    from fukikae.job import load_manifest
    from fukikae.registry.invalidate import episode as episode_of

    done = {n for n, r in load_manifest(episode_of(ws, episode)).stages.items()
            if r.status == "done"}  # fmt: skip
    return next((s for s in reversed(STAGE_ORDER) if s in done), None)


# -- sample clips ---------------------------------------------------------------------------


def _audio_dir(ws: Workspace) -> Path:
    d = ws.root / "studio" / "audio"
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    return d


def sample_name(ws: Workspace | Path, episode: str, start: float, end: float) -> str:
    """``<sha>.wav``: the clip's file name, from the span and the vocals' size and time (a
    re-run separate stage makes new clips)."""
    from fukikae.cards.select import CLIP_RATE

    w = _workspace(ws)
    st = _vocals(w, episode).stat()
    key = {
        "episode": episode, "start": round(float(start), 3), "end": round(float(end), 3),
        "vocals": [st.st_size, st.st_mtime_ns], "rate": CLIP_RATE,
    }  # fmt: skip
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest() + ".wav"


def cut_sample(ws: Workspace | Path, episode: str, start: float, end: float) -> Path:
    """``<ws>/studio/audio/<sha>.wav``: ``[start, end]`` of the episode's vocals, 24 kHz mono
    (`cards.select.cut_clip`), in an owner-only directory; cut once."""
    from fukikae.artifacts import atomic_path
    from fukikae.cards.select import cut_clip

    w = _workspace(ws)
    dest = _audio_dir(w) / sample_name(w, episode, start, end)
    if not dest.is_file():
        with atomic_path(dest) as tmp:
            cut_clip(_vocals(w, episode), start, end, tmp)
    return dest


def _sample(ep: _Episode, ws: Workspace, start: float, end: float) -> Sample | None:
    """A sample clip of ``[start, end]`` (its first `SAMPLE_MAX_S`); None when the episode has
    no vocals stem to cut it from."""
    from fukikae.cards.select import turn_text

    if not ep.vocals.is_file():
        return None
    end = min(end, start + SAMPLE_MAX_S)
    text = turn_text(ep.transcript, start, end) if ep.transcript is not None else ""
    clip = cut_sample(ws, ep.id, start, end).name
    return Sample(episode=ep.id, start=start, end=end, transcript=text, clip=clip)


def _pick(turns: Sequence[Turn], n: int, clean: set[tuple[float, float]]) -> list[Turn]:
    """``n`` turns spread over ``turns``: the time-ordered turns cut into ``n`` slices, and of
    each slice the longest clean turn (overlapping no other cluster), else the longest."""
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    out: list[Turn] = []
    for idx in np.array_split(np.arange(len(ordered)), min(n, len(ordered))):
        if not len(idx):
            continue
        part = [ordered[i] for i in idx]
        out.append(min(part, key=lambda t: ((t.start, t.end) not in clean,
                                            -(t.end - t.start), t.start)))  # fmt: skip
    return out


def _allocate(capacity: Sequence[int], n: int) -> list[int]:
    """Clips per episode: round-robin in episode order, never more than an episode's turns;
    with more episodes than clips, evenly spaced episodes get one each."""
    k = len(capacity)
    counts = [0] * k
    if k > n:
        for i in sorted({int(round(x)) for x in np.linspace(0, k - 1, n)}):
            counts[i] = 1
        return counts
    left = n
    while left > 0 and any(counts[i] < capacity[i] for i in range(k)):
        for i in range(k):
            if left and counts[i] < capacity[i]:
                counts[i] += 1
                left -= 1
    return counts


def _spread_samples(
    ws: Workspace, eps: dict[str, _Episode], turns: dict[str, list[Turn]], n: int
) -> list[Sample]:
    """``n`` samples of one speaker spread over its episodes and over each episode's turns;
    an episode without its vocals stem gives none."""
    order = [e for e in sorted(turns) if turns[e] and eps[e].vocals.is_file()]
    counts = _allocate([len(turns[e]) for e in order], n)
    out: list[Sample] = []
    for e, k in zip(order, counts, strict=True):
        if k:
            out += _samples(eps[e], ws, _pick(turns[e], k, eps[e].clean))
    return out


def _samples(ep: _Episode, ws: Workspace, turns: Iterable[Turn]) -> list[Sample]:
    """A sample per turn; none when the episode has no vocals stem."""
    got = [_sample(ep, ws, t.start, t.end) for t in turns]
    return [s for s in got if s is not None]


# -- one episode's files --------------------------------------------------------------------


@dataclass
class _Episode:
    id: str
    asg: Assignments
    transcript: Transcript | None
    vocals: Path
    cues: DubCues | None
    plan: TtsPlan | None
    run: TtsRun | None
    dropped: list[dict[str, Any]]
    names: NamesResult | None  # used names only
    report: NamesReport | None
    clean: set[tuple[float, float]] = field(default_factory=set)
    # character -> cluster -> its turns
    speech: dict[int, dict[str, list[Turn]]] = field(default_factory=dict)


def _clean_turns(turns: Sequence[Turn]) -> set[tuple[float, float]]:
    """The turns overlapping no turn of another cluster."""
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    dirty: set[int] = set()
    for i, t in enumerate(ordered):
        for j in range(i + 1, len(ordered)):
            u = ordered[j]
            if u.start >= t.end:
                break
            if u.speaker != t.speaker:
                dirty |= {i, j}
    return {(t.start, t.end) for i, t in enumerate(ordered) if i not in dirty}


def _load(ws: Workspace, cfg: Config, episode: str) -> _Episode:
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import used_names
    from fukikae.names.inputs import names_report_path
    from fukikae.names.models import NamesReport
    from fukikae.transcripts import transcript_path

    work = ws.work_dir(episode)
    asg = read_json(_assignments_path(ws, episode), Assignments)
    tr = transcript_path(work)
    cues = work / "06-translate" / f"cues.{cfg.target_language}.json"
    plan, run = work / "07-tts" / "plan.json", work / "07-tts" / "run.json"
    dropped_path = work / "03b-asrclean" / "dropped.json"
    try:
        dropped = json.loads(dropped_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        dropped = []
    names = used_names(work)
    rpath = names_report_path(work)
    ep = _Episode(
        id=episode,
        asg=asg,
        transcript=read_json(tr, Transcript) if tr.exists() else None,
        vocals=_vocals(ws, episode),
        cues=read_json(cues, DubCues) if cues.exists() else None,
        plan=read_json(plan, TtsPlan) if plan.exists() else None,
        run=read_json(run, TtsRun) if run.exists() else None,
        dropped=dropped if isinstance(dropped, list) else [],
        names=names,
        report=read_json(rpath, NamesReport) if names is not None and rpath.exists() else None,
    )
    turns = asg.turns or []
    ep.clean = _clean_turns(turns)
    who = {a.cluster: a.character_id for a in asg.clusters
           if a.character_id is not None and a.status not in EXTRA_STATUSES}  # fmt: skip
    for t in sorted(turns, key=lambda t: (t.start, t.end)):
        cid = who.get(t.speaker)
        if cid is not None:
            ep.speech.setdefault(cid, {}).setdefault(t.speaker, []).append(t)
    return ep


# -- pitch ----------------------------------------------------------------------------------


def _join(tracks: Sequence[F0Track]) -> F0Track | None:
    """One track of several (the season's), spans renumbered in order."""
    from fukikae.pitch import F0Track

    if not tracks:
        return None
    offsets = np.cumsum([0] + [t.spans for t in tracks[:-1]])
    loud = None
    if all(t.loud is not None for t in tracks):
        loud = np.concatenate([t.loud for t in tracks if t.loud is not None])
    return F0Track(
        f0=np.concatenate([t.f0 for t in tracks]),
        span_index=np.concatenate([t.span_index + o for t, o in zip(tracks, offsets, strict=True)]),
        seconds=float(sum(t.seconds for t in tracks)),
        span_seconds=np.concatenate([t.span_seconds for t in tracks]),
        loud=loud,
    )


def _turn_track(ws: Workspace, ep: _Episode, turns: Sequence[Turn]) -> F0Track | None:
    from fukikae.pitch import cached_track

    if not turns or not ep.vocals.exists():
        return None
    return cached_track(ws.root, ep.vocals, [(t.start, t.end) for t in turns])


# -- registry -------------------------------------------------------------------------------


def registry_digest(reg: Registry) -> str:
    """sha256 of the registry contents the index reads: characters, aliases, actors (with a
    hash of each centroid), every current card assignment and the exemplar counts."""
    chars = reg.characters()
    actors = []
    for a in reg.actors():
        cen = None if a.centroid is None else hashlib.sha256(a.centroid.tobytes()).hexdigest()
        actors.append([a.id, a.name, a.external_id, a.locked, cen, reg.episode_counts(a.id)])
    cards = []
    for c in chars:
        for lang in reg.card_languages(c.id):
            cur = reg.card_for(c.id, lang)
            cards.append(asdict(cur) if cur is not None else [c.id, lang])
    payload = {
        "characters": [asdict(c) for c in chars],
        "aliases": [asdict(a) for a in reg.aliases()],
        "actors": actors,
        "cards": cards,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _file_sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _episode_files(work: Path) -> list[Path]:
    from fukikae.names.inputs import DIARIZE_DIR, NAMES_DIR

    fixed = [
        "manifest.json",
        "03-asr/transcript.json",
        "03b-asrclean/transcript.json",
        "03b-asrclean/dropped.json",
        f"{DIARIZE_DIR}/turns.json",
        f"{DIARIZE_DIR}/clusters.npz",
        f"{NAMES_DIR}/names.json",
        f"{NAMES_DIR}/turns.json",
        f"{NAMES_DIR}/clusters.npz",
        "05-characters/assignments.json",
        "05-characters/names-report.json",
        "07-tts/plan.json",
        "07-tts/run.json",
    ]
    cues = sorted((work / "06-translate").glob("cues.*.json"))
    return [work / f for f in fixed] + cues


def config_digest(cfg: Config, ws: Workspace) -> str:
    """sha256 of the resolved configuration the index reads: ``cfg`` (the profile merged with
    the user config and ``fukikae.toml``) and the workspace's own `Workspace.config`, which
    the names evidence and card flags read."""
    payload = {"cfg": asdict(cfg), "workspace": asdict(ws.config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def inputs_digest(ws: Workspace | Path, cfg: Config, *, clips: int = 6) -> str:
    """The digest `load_or_build` compares: every input the index reads (the module
    docstring), the target language, the clip count and `SEASON_VERSION`."""
    from fukikae.pitch import VERSION as PITCH_VERSION
    from fukikae.registry.store import Registry

    w = _workspace(ws)
    episodes: dict[str, Any] = {}
    for e in episode_ids(w):
        work = w.work_dir(e)
        files = {str(p.relative_to(work)): _file_sha(p) for p in _episode_files(work)}
        v = _vocals(w, e)
        st = v.stat() if v.exists() else None
        split = w.root / "studio" / "splits" / f"{e}.json"
        episodes[e] = {
            "files": files,
            "vocals": [st.st_size, st.st_mtime_ns] if st is not None else None,
            "split": _file_sha(split),
        }
    registry = None
    if w.registry_path.exists():
        with Registry(w.registry_path) as reg:
            registry = registry_digest(reg)
    payload = {
        "version": SEASON_VERSION,
        "pitch": PITCH_VERSION,
        "workspace": str(w.root),
        "config": config_digest(cfg, w),
        "target_language": cfg.target_language,
        "clips": clips,
        "episodes": episodes,
        "registry": registry,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# -- per character ---------------------------------------------------------------------------


def _gender(c: Character, tracks: dict[str, F0Track]) -> GenderInfo:
    from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW, band_seconds, classify, summarize

    bands = (MALE_BELOW, FEMALE_ABOVE)
    per = {e: classify(summarize(t), *bands).gender for e, t in sorted(tracks.items())}
    season = _join([tracks[e] for e in sorted(tracks)])
    if season is None:
        est, conf, med, secs = "u", 0.0, None, {"m": 0.0, "u": 0.0, "f": 0.0}
    else:
        g = classify(summarize(season), *bands)
        est, conf, med = g.gender, round(g.confidence, 3), g.median
        secs = {k: round(v, 3) for k, v in band_seconds(season, bands).items()}
    return GenderInfo(
        estimate=est, confidence=conf, median=None if med is None else round(med, 1),
        seconds=secs, per_episode=per, stored=c.gender, stored_by=c.gender_by,
    )  # fmt: skip


def _splits(
    ws: Workspace, ep: _Episode, clusters: dict[str, list[Turn]], track: F0Track | None
) -> tuple[list[Suggestion], list[Flag]]:
    """A split suggestion and a ``bimodal`` flag for each of the character's clusters in the
    episode whose turns are `pitch.bimodal`, each part with its seconds and longest turns."""
    from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW, bimodal, turn_classes

    if track is None:
        return [], []
    classes = turn_classes(track, (MALE_BELOW, FEMALE_ABOVE))
    spans = [(cl, t) for cl, ts in clusters.items() for t in ts]
    spans.sort(key=lambda x: (x[1].start, x[1].end))  # the track's span order
    by: dict[str, dict[str, list[Turn]]] = defaultdict(lambda: defaultdict(list))
    for (cl, t), est in zip(spans, classes, strict=True):
        by[cl][est.gender].append(t)
    sugg: list[Suggestion] = []
    flags: list[Flag] = []
    for cl in sorted(by):
        secs = {g: sum(t.end - t.start for t in by[cl].get(g, [])) for g in ("m", "u", "f")}
        if not bimodal(secs):
            continue
        parts: dict[str, Any] = {}
        for g in ("f", "m"):
            longest = sorted(by[cl][g], key=lambda t: (-(t.end - t.start), t.start))
            parts[g] = {
                "seconds": round(secs[g], 3),
                "samples": [x.model_dump() for x in _samples(ep, ws, longest[:PART_SAMPLES])],
            }  # fmt: skip
        parts["u"] = {"seconds": round(secs["u"], 3)}
        sugg.append(Suggestion(kind="split", episode=ep.id, cluster=cl, parts=parts))
        flags.append(Flag(
            kind="bimodal", episode=ep.id,
            text=f"{ep.id} {cl}: {secs['m']:.1f} s in the low register and {secs['f']:.1f} s "
                 "in the high one; a pitch split may separate two voices",
        ))  # fmt: skip
    return sugg, flags


def _pairable(a: str | None, b: str | None) -> bool:
    """Whether characters of kinds ``a`` and ``b`` may be suggested as one: neither ignored
    nor unknown, and a narrator only with a narrator."""
    if a is None or b is None or "ignored" in (a, b):
        return False
    return (a == "narrator") == (b == "narrator")


def _merges(entries: dict[int, Character], reg: Registry) -> dict[int, list[Suggestion]]:
    """For each character, up to `MERGE_NEAREST` other characters of other actors whose
    registry centroids are at `MERGE_SUGGEST_COS` or more, nearest first; a narrator is paired
    only with a narrator, an ignored character never."""
    cen = {a.id: a.centroid for a in reg.actors() if a.centroid is not None}
    out: dict[int, list[Suggestion]] = {cid: [] for cid in entries}
    usable = [c for c in entries.values() if c.kind != "ignored" and c.actor_id in cen]
    for c in usable:
        x = cen[c.actor_id]
        got: list[tuple[float, int]] = []
        for d in usable:
            if d.id == c.id or d.actor_id == c.actor_id or not _pairable(c.kind, d.kind):
                continue
            y = cen[d.actor_id]
            denom = float(np.linalg.norm(x) * np.linalg.norm(y))
            if not denom:
                continue
            cos = float(np.dot(x, y)) / denom
            if cos >= MERGE_SUGGEST_COS:
                got.append((cos, d.id))
        got.sort(key=lambda p: (-p[0], p[1]))
        out[c.id] = [Suggestion(kind="merge", other=d, cosine=round(cos, 6))
                     for cos, d in got[:MERGE_NEAREST]]  # fmt: skip
    return out


@dataclass
class _Names:
    lines: list[str] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    hints: list[Suggestion] = field(default_factory=list)


def _names_evidence(ws: Workspace, eps: dict[str, _Episode], reg: Registry) -> dict[int, _Names]:
    """Per character: its key decisions, named lines and voice cosines in each names-used
    episode, the stale and disagreement lines `names show` computes (``names`` flags), and
    the names merge hints (`fukikae.names.flags.merge_hints`) as merge suggestions."""
    from fukikae.names import flags as nflags

    kind = {c.id: c.kind for c in reg.characters()}
    out: dict[int, _Names] = defaultdict(_Names)
    for e, ep in sorted(eps.items()):
        if ep.names is None:
            continue
        keys_of: dict[int, set[str]] = defaultdict(set)
        if ep.report is not None:
            for d in ep.report.keys:
                holders = {d.character_id} if d.how not in ("extra", "ambiguous") else set()
                holders |= set(d.locked.values())
                stale = nflags.stale_decision(d, ep.asg, reg, ws, e)
                for cid in sorted(h for h in holders if h is not None):
                    keys_of[cid].add(d.key)
                    out[cid].lines.append(
                        f"{e}: {d.key!r} {d.seconds:.1f} s ({', '.join(d.labels)}), {d.how}"
                    )
                    if stale is not None:
                        out[cid].lines.append(f"{e}: {d.key!r} {stale}")
                        out[cid].flags.append(Flag(kind="names", text=f"{e}: {d.key!r} {stale}",
                                                   episode=e))  # fmt: skip
            for line in ep.report.lines:
                if "acoustics disagree" not in line:
                    continue
                for d in ep.report.keys:
                    if d.character_id is not None and line.startswith(f"{d.key!r} -> "):
                        out[d.character_id].lines.append(f"{e}: {line}")
                        out[d.character_id].flags.append(
                            Flag(kind="names", text=f"{e}: {line}", episode=e)
                        )
        visible = ep.names.visible_lines()
        for cid, keys in sorted(keys_of.items()):
            n = sum(1 for ln in visible if ln.key in keys)
            out[cid].lines.append(f"{e}: {n} named line{'s' if n != 1 else ''}")
        for dis in nflags.disagreement_rows(ws, e, ep.asg, reg):
            out[dis.character_id].lines.append(
                f"{e}: {dis.cluster} ({dis.name}) at {dis.cosine:.3f} to the character's "
                "other-episode centroid"
            )
        for h in nflags.merge_hints(ws, ep.names, ep.asg, reg):
            if _pairable(kind.get(h.drop), kind.get(h.keep)):
                out[h.drop].hints.append(Suggestion(kind="merge", other=h.keep, episode=e))
    return out


def _card_infos(
    ws: Workspace, reg: Registry, store: CardStore, ids: list[int]
) -> tuple[dict[int, list[CardInfo]], dict[int, list[Flag]]]:
    """Each character's current card per target language, and the card flags (stale, thin
    or under the floor, impure, stock, operator gender) as character flags."""
    from fukikae.cards import auto
    from fukikae.cards.card import CardError
    from fukikae.registry.store import replace_refusal
    from fukikae.studio.apply import row_digest

    infos: dict[int, list[CardInfo]] = defaultdict(list)
    flags: dict[int, list[Flag]] = defaultdict(list)
    langs: dict[str, list[int]] = defaultdict(list)
    for cid in ids:
        for lang in reg.card_languages(cid):
            langs[lang].append(cid)
    operator = auto.operator_genders(reg)
    for lang in sorted(langs):
        chars = langs[lang]
        stale = auto.stale_cards(ws, reg, store, chars, lang)
        short = auto.below_floor(reg, store, chars, lang)
        impure = auto.impure_cards(ws, reg, store, lang, chars, genders=operator)
        stock = auto.stock_cards(reg, store, chars, lang)
        genders = auto.gender_mismatches(ws, reg, store, lang, chars, operator=operator)
        for cid in chars:
            a = reg.card_for(cid, lang)
            if a is None:
                continue
            source = tier = None
            if a.card_id is not None and store.has(a.card_id):
                try:
                    card = store.get(a.card_id)
                    source = card.voice.source
                    t = card.meta.get("reference_tier")
                    tier = t if t in TIERS else None
                except CardError:
                    pass
            kinds: list[str] = []
            where = f"{lang} card {a.card_id}"

            def add(kind: str, text: str, cid: int = cid, kinds: list[str] = kinds) -> None:
                kinds.append(kind)
                flags[cid].append(Flag(kind=kind, text=text))

            if cid in short:
                s = short[cid]
                add("thin" if s.thin else "floor", f"{where}: {s.phrase}")
            if cid in impure:
                add("impure", f"{where}: references fail the gates ({impure[cid].phrase})")
            if cid in stock:
                why = replace_refusal(a, "auto")
                then = ("the next tts run re-picks it as a clone" if why is None
                        else f"kept ({why}); a clone from the studio replaces it")  # fmt: skip
                add("stock", f"{where}: a stock voice, retired (D41); {then}")
            if cid in stale:
                add("stale", f"{where}: references the subtitle names now give someone else")
            if cid in genders:
                add("gender", f"{where}: {genders[cid].phrase}")
            infos[cid].append(CardInfo(
                language=lang, card_id=a.card_id, source=source, chosen_by=a.chosen_by,
                locked=a.locked, tier=tier, flags=kinds, digest=row_digest(reg, cid, lang),
            ))  # fmt: skip
    return infos, flags


def _gated(ep: _Episode, cid: int) -> Flag | None:
    if ep.plan is None:
        return None
    n = sum(1 for c in ep.plan.cues if c.character_id == cid and c.gender_gate)
    if not n:
        return None
    return Flag(
        kind="gated", episode=ep.id,
        text=f"{ep.id}: {n} line{'s' if n != 1 else ''} rendered without emotion audio, the "
             "source line being the other register (A3)",
    )  # fmt: skip


def _dropped(ep: _Episode, turns: Iterable[Turn]) -> Flag | None:
    spans = [(t.start, t.end) for t in turns]
    hit = []
    for d in ep.dropped:
        try:
            a, b = float(d["start"]), float(d["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(min(b, y) > max(a, x) for x, y in spans):
            hit.append(d)
    if not hit:
        return None
    texts = ", ".join(repr(str(d.get("text", ""))) for d in hit[:3])
    return Flag(
        kind="dropped", episode=ep.id,
        text=f"{ep.id}: {len(hit)} stretch{'es' if len(hit) != 1 else ''} over its turns "
             f"dropped as ASR hallucinations ({texts})",
    )  # fmt: skip


def _cue_count(ep: _Episode, cid: int) -> int:
    return 0 if ep.cues is None else sum(1 for c in ep.cues.cues if c.character_id == cid)


def _extras(ws: Workspace, ep: _Episode) -> list[ExtraEntry]:
    """The episode's extras clusters, and its cluster-less lines' labels from the last tts
    run, each with seconds, the run's tier and card, and samples."""
    run = ep.run.extras if ep.run is not None else {}
    labels = sorted({a.cluster for a in ep.asg.clusters if a.status in EXTRA_STATUSES}
                    | set(run))  # fmt: skip
    turns: dict[str, list[Turn]] = defaultdict(list)
    for t in ep.asg.turns or []:
        turns[t.speaker].append(t)
    cues = {c.id: c for c in ep.cues.cues} if ep.cues is not None else {}
    out: list[ExtraEntry] = []
    for label in labels:
        info = run.get(label, {})
        tier = info.get("tier") if info.get("tier") in TIERS else None
        mine = turns.get(label, [])
        if mine:
            seconds = sum(t.end - t.start for t in mine)
            samples = _samples(ep, ws, _pick(mine, EXTRA_CLIPS, ep.clean))
        else:
            seconds = float(info.get("seconds") or 0.0)
            cue = cues.get(label.split("~", 1)[1]) if "~" in label else None
            samples = []
            if cue is not None:
                a, b = cue.delivery.reference
                got = _sample(ep, ws, float(a), float(b))
                samples = [got] if got is not None else []
        out.append(ExtraEntry(
            episode=ep.id, label=label, seconds=round(seconds, 3), tier=tier,
            card_id=info.get("card"), samples=samples,
        ))  # fmt: skip
    return out


def order_queue(characters: Sequence[CharacterEntry]) -> list[int]:
    """Character ids for review: by their worst flag (`SEVERITY`; other kinds after those,
    unflagged last), then unreviewed before reviewed, then seconds of speech, most first.
    Ignored characters (decided: not a character) come last."""

    def rank(c: CharacterEntry) -> int:
        ranks = [SEVERITY.index(f.kind) if f.kind in SEVERITY else len(SEVERITY)
                 for f in c.flags]  # fmt: skip
        return min(ranks, default=len(SEVERITY) + 1)

    ordered = sorted(characters, key=lambda c: (
        c.kind == "ignored", rank(c), c.reviewed is not None, -c.seconds, c.id,
    ))  # fmt: skip
    return [c.id for c in ordered]


def build_season(ws: Workspace | Path, cfg: Config, *, clips: int = 6) -> SeasonDoc:
    """The season index of the workspace (the module docstring). Reads the registry without
    writing it; with no registry yet, no character is listed."""
    from fukikae.cards.store import CardStore
    from fukikae.registry.store import Registry, alias_key
    from fukikae.studio.apply import row_digest

    w = _workspace(ws)
    digest = inputs_digest(w, cfg, clips=clips)
    episodes: list[EpisodeEntry] = []
    eps: dict[str, _Episode] = {}
    for e in episode_ids(w):
        ready = _assignments_path(w, e).exists()
        note = None
        if ready and not _vocals(w, e).is_file():
            note = (f"no {_vocals(w, e).relative_to(w.work_dir(e))}, so no samples or pitch; "
                    f"fukikae run {w.root} {e} --to separate brings it back")  # fmt: skip
        episodes.append(EpisodeEntry(
            id=e, last_stage=_last_stage(w, e), ready=ready,
            command=None if ready else f"fukikae run {w.root} {e} --to characters", note=note,
        ))  # fmt: skip
        if ready:
            eps[e] = _load(w, cfg, e)
    extras = [x for e in sorted(eps) for x in _extras(w, eps[e])]
    characters: list[CharacterEntry] = []
    if w.registry_path.exists():
        with Registry(w.registry_path) as reg:
            chars = {c.id: c for c in reg.characters()}
            speaking = sorted({cid for ep in eps.values() for cid in ep.speech} & set(chars))
            index = {cid: chars[cid] for cid in speaking}
            store = CardStore(w.root)
            cards, card_flags = _card_infos(w, reg, store, speaking)
            merges = _merges(index, reg)
            names = _names_evidence(w, eps, reg)
            for cid in speaking:
                c = index[cid]
                per: dict[str, EpisodeSpeech] = {}
                tracks: dict[str, F0Track] = {}
                turns_by_ep: dict[str, list[Turn]] = {}
                suggestions = list(merges.get(cid, []))
                flags: list[Flag] = []
                for e, ep in sorted(eps.items()):
                    clusters = ep.speech.get(cid)
                    if not clusters:
                        continue
                    turns = sorted((t for ts in clusters.values() for t in ts),
                                   key=lambda t: (t.start, t.end))  # fmt: skip
                    turns_by_ep[e] = turns
                    clean = sum(a.clean_seconds for a in ep.asg.clusters
                                if a.cluster in clusters)  # fmt: skip
                    per[e] = EpisodeSpeech(
                        clusters=sorted(clusters),
                        seconds=round(sum(t.end - t.start for t in turns), 3),
                        clean_seconds=round(clean, 3), turns=len(turns), cues=_cue_count(ep, cid),
                    )  # fmt: skip
                    track = _turn_track(w, ep, turns)
                    if track is not None:
                        tracks[e] = track
                    if c.kind != "ignored":
                        s, f = _splits(w, ep, clusters, track)
                        suggestions += s
                        flags += f
                    for got in (_gated(ep, cid), _dropped(ep, turns)):
                        if got is not None:
                            flags.append(got)
                ev = names.get(cid, _Names())
                merged = {s.other for s in suggestions if s.kind == "merge"}
                other_kind = {h.other: getattr(chars.get(h.other or 0), "kind", None)
                              for h in ev.hints}  # fmt: skip
                suggestions += [h for h in ev.hints if h.other not in merged
                                and _pairable(c.kind, other_kind[h.other])]  # fmt: skip
                flags = card_flags.get(cid, []) + flags + ev.flags
                characters.append(CharacterEntry(
                    id=cid, actor_id=c.actor_id, name=c.name,
                    key=alias_key(c.name) if c.name else None, external_id=c.external_id,
                    kind=c.kind, locked=c.locked, reviewed=c.reviewed,
                    digest=row_digest(reg, cid), episodes=per,
                    seconds=round(sum(x.seconds for x in per.values()), 3),
                    clean_seconds=round(sum(x.clean_seconds for x in per.values()), 3),
                    turns=sum(x.turns for x in per.values()),
                    cues=sum(x.cues for x in per.values()),
                    samples=_spread_samples(w, eps, turns_by_ep, clips),
                    gender=_gender(c, tracks), cards=cards.get(cid, []),
                    suggestions=suggestions, names=ev.lines, flags=flags,
                ))  # fmt: skip
    doc = SeasonDoc(
        workspace=str(w.root),
        built=datetime.now(UTC).isoformat(timespec="seconds"),
        inputs_digest=digest,
        episodes=episodes,
        characters=characters,
        extras=extras,
        queue=order_queue(characters),
    )
    prune_clips(w, doc)
    return doc


def referenced_clips(doc: SeasonDoc) -> set[str]:
    """The clip names under ``<ws>/studio/audio`` that ``doc`` refers to: the only audio the
    studio server serves from the index."""
    return _clips(doc)


def _clips(doc: SeasonDoc) -> set[str]:
    """The clip names the document refers to: character, extras and split-part samples."""
    out = {s.clip for c in doc.characters for s in c.samples}
    out |= {s.clip for x in doc.extras for s in x.samples}
    for c in doc.characters:
        for sugg in c.suggestions:
            for part in (sugg.parts or {}).values():
                out |= {str(x["clip"]) for x in part.get("samples", []) if "clip" in x}
    return out


def prune_clips(ws: Workspace | Path, doc: SeasonDoc) -> list[str]:
    """Delete the clips under ``<ws>/studio/audio`` that ``doc`` no longer refers to: only
    regular files named like `sample_name` (``<sha>.wav``) directly in that directory, never
    a symlink, a subdirectory or anything else. Returns the names deleted."""
    d = _workspace(ws).root / "studio" / "audio"
    if d.is_symlink() or not d.is_dir():
        return []
    keep = _clips(doc)
    gone: list[str] = []
    with os.scandir(d) as it:
        for entry in it:
            if entry.name in keep or not CLIP_NAME.fullmatch(entry.name):
                continue
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
            try:
                os.unlink(os.path.join(d, entry.name))
            except FileNotFoundError:
                continue
            gone.append(entry.name)
    return sorted(gone)


def load_or_build(ws: Workspace | Path, cfg: Config, *, clips: int = 6) -> SeasonDoc:
    """``season.json`` when its ``inputs_digest`` still matches the workspace, else a fresh
    `build_season`, written there."""
    from fukikae.artifacts import read_json, write_json

    w = _workspace(ws)
    path = season_path(w)
    if path.exists():
        try:
            doc = read_json(path, SeasonDoc)
        except ValueError:
            doc = None
        if doc is not None and doc.inputs_digest == inputs_digest(w, cfg, clips=clips):
            return doc
    doc = build_season(w, cfg, clips=clips)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, doc)
    return doc


__all__ = [
    "MERGE_SUGGEST_COS",
    "SEASON_VERSION",
    "build_season",
    "config_digest",
    "cut_sample",
    "episode_ids",
    "inputs_digest",
    "load_or_build",
    "order_queue",
    "prune_clips",
    "referenced_clips",
    "registry_digest",
    "sample_name",
    "season_path",
]
