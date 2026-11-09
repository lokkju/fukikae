"""Automatic voice assignment (D12): a card for each character that has none in the target
language, and stock voices for extras.

Only characters with no assignment get one, ``by = "auto"``, unlocked. An automatic
assignment never replaces a card: should one appear between the check and the write, the
registry refuses (``ProtectedAssignmentError``, rulings CD2, CD3 and CD9) and the existing
card is kept.

With ``[tts] clone_backend`` set (IndexTTS2 in every profile since the operator's verdict,
D24), each such character gets a clone card of that backend (``cloning``): reference clips
from ``select.select_references``'s rules (CD5) over the character's turns in the
workspace's episodes (``workspace_cloning``), in the episode where they add up to the most
speech. A character with less clean speech than the minimum (6 s) gets a stock voice from
the gender-matched pool of the ``[tts]`` backend instead, and the report says how much it
had. Without ``clone_backend`` every automatic card is a stock voice. Extras always speak
with stock voices.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fukikae.cards.select import FEMALE_ABOVE, MALE_BELOW, gender_of, speaker_f0
from fukikae.cards.stock import check_stock_voice, pick_stock, revision, stock_card
from fukikae.cards.store import CardStore
from fukikae.registry.store import ProtectedAssignmentError

if TYPE_CHECKING:
    from fukikae.backends.base import Synthesizer
    from fukikae.cards.select import RefPick
    from fukikae.config import Config
    from fukikae.models import Assignments, DubCue
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

__all__ = [
    "CharacterRefs",
    "Cloning",
    "ProtectedAssignmentError",
    "character_candidates",
    "character_references",
    "character_spans",
    "clone_backend",
    "ensure_assignments",
    "extras_voices",
    "gender_thresholds",
    "tts_backend",
    "workspace_cloning",
]

UNKNOWN_EXTRA = "unknown"
MIN_CLONE_S = 6.0  # the spec's minimum of clean speech for a clone (pick_references' default)


@dataclass(frozen=True)
class CharacterRefs:
    """What a character's turns offer a clone: ``seconds`` of eligible reference speech in
    the episode that has the most, and the picks there (with that episode and its vocals),
    or no picks under the minimum."""

    seconds: float
    picks: list[RefPick] | None = None
    episode: str | None = None
    vocals: Path | None = None


@dataclass(frozen=True)
class Cloning:
    """The clone branch of ``ensure_assignments``: the backend clone cards are made for, the
    language the references speak, and each character's references."""

    backend: type[Synthesizer]
    source_language: str
    references: Callable[[int], CharacterRefs]


def clone_backend(cfg: Config) -> str | None:
    """``[tts] clone_backend``, the workspace's over the profile's; None when unset or empty
    (automatic cards are then stock voices)."""
    from fukikae.config import ConfigError

    name = None
    for source in (cfg.profile_raw, cfg.raw):
        table = source.get("tts")
        if isinstance(table, dict) and "clone_backend" in table:
            name = table["clone_backend"]
    if name is not None and not isinstance(name, str):
        raise ConfigError(f"[tts] clone_backend must be a backend name, not {name!r}")
    return name or None


def tts_backend(cfg: Config) -> str:
    """The ``[tts]`` backend: the one automatic cards and extras are made for."""
    return cfg.resolve("tts").backend


def gender_thresholds(cfg: Config) -> tuple[float, float]:
    """(male_below, female_above) in Hz from ``[tts] gender``, the workspace's over the
    profile's; F0 between them is an unknown gender."""
    from fukikae.config import ConfigError

    g: dict = {}
    for source in (cfg.profile_raw, cfg.raw):
        table = source.get("tts")
        if isinstance(table, dict) and isinstance(table.get("gender"), dict):
            g = {**g, **table["gender"]}
    unknown = sorted(set(g) - {"male_below", "female_above"})
    if unknown:
        raise ConfigError(f"unknown key(s) in [tts] gender: {', '.join(unknown)}")
    lo, hi = g.get("male_below", MALE_BELOW), g.get("female_above", FEMALE_ABOVE)
    for v in (lo, hi):
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise ConfigError(f"[tts] gender thresholds must be numbers (Hz), not {v!r}")
    if lo > hi:
        raise ConfigError(f"[tts] gender male_below ({lo}) is above female_above ({hi})")
    return float(lo), float(hi)


def character_spans(asg: Assignments) -> dict[int, list[tuple[float, float]]]:
    """Each character's diarize turns in the episode (extras and review clusters left out)."""
    by_cluster = {
        a.cluster: a.character_id
        for a in asg.clusters
        if a.character_id is not None and a.status not in ("extra", "review")
    }
    out: dict[int, list[tuple[float, float]]] = {}
    for t in asg.turns or []:
        cid = by_cluster.get(t.speaker)
        if cid is not None:
            out.setdefault(cid, []).append((float(t.start), float(t.end)))
    return out


def _stock_voices_in_use(reg: Registry, store: CardStore, backend: str, language: str) -> list[str]:
    used: list[str] = []
    for c in reg.characters():
        a = reg.card_for(c.id, language)
        if a is None or a.card_id is None or not store.has(a.card_id):
            continue
        v = store.get(a.card_id).voice
        if v.backend == backend and v.source == "stock" and v.voice:
            used.append(v.voice)
    return used


def _label(reg: Registry, cid: int) -> str:
    name = next((c.name for c in reg.characters() if c.id == cid), None)
    return f"{cid} ({name})" if name else str(cid)


def ensure_assignments(
    reg: Registry,
    store: CardStore,
    cfg: Config,
    character_spans: dict[int, list[tuple[float, float]]],
    vocals: Path,
    *,
    cloning: Cloning | None = None,
) -> list[str]:
    """Give every character in ``character_spans`` without a ``cfg.target_language`` card
    one: a clone card when ``cloning`` is given and the character has enough clean speech,
    else a stock card from its gender's pool, most speech first (so the main cast gets the
    first voices of each pool). Returns one report line per character handled."""
    language = cfg.target_language
    todo = []
    for cid, spans in character_spans.items():
        a = reg.card_for(cid, language)
        if a is None or a.card_id is None:
            todo.append((cid, spans))
    if not todo:
        return []
    backend = tts_backend(cfg)
    rev = revision(backend)
    male_below, female_above = gender_thresholds(cfg)
    used = _stock_voices_in_use(reg, store, backend, language)
    todo.sort(key=lambda item: (-sum(e - s for s, e in item[1]), item[0]))
    lines: list[str] = []
    for cid, spans in todo:
        found = cloning.references(cid) if cloning is not None else None
        voice: str | None = None
        if found is not None and found.picks:
            new, note = _clone(cloning, found, store, language)  # type: ignore[arg-type]
        else:
            f0, seconds = speaker_f0(vocals, spans)
            gender = gender_of(f0, male_below, female_above)
            voice = pick_stock(backend, language, gender, used)
            check_stock_voice(backend, voice)
            new = store.put(stock_card(backend, voice, language, rev))
            estimate = (
                f"gender estimate {gender}, F0 {f0:.0f} Hz (20th percentile) over {seconds:.1f} s"
                if f0 is not None
                else f"gender estimate u, too little voiced speech in {seconds:.1f} s"
            )
            note = f"stock {backend}:{voice}, card {new} ({estimate}); auto and unlocked"
            if found is not None:
                note = (
                    f"{found.seconds:.1f} s of clean speech, under {MIN_CLONE_S:.0f} s; "
                    f"stock voice {voice}: {note}"
                )
        try:
            reg.assign_card(cid, language, new, by="auto", locked=False, note=note)
        except ProtectedAssignmentError as e:
            lines.append(f"character {_label(reg, cid)}: kept its card ({e})")
            continue
        if voice is not None:
            used.append(voice)
        lines.append(f"character {_label(reg, cid)}: {note}")
    return lines


def _clone(
    cloning: Cloning, found: CharacterRefs, store: CardStore, language: str
) -> tuple[str, str]:
    """(card id, note) of a clone card from ``found``'s picks, put in the store."""
    from fukikae.cards.select import clone_card, reference_report

    picks, episode, vocals = found.picks or [], found.episode, found.vocals
    assert episode is not None and vocals is not None
    card = clone_card(
        cloning.backend, picks, store, vocals, cloning.source_language, language, episode
    )
    new = store.put(card)
    clips = ", ".join(f"{p.start:.2f}-{p.end:.2f}" for p in picks)
    seconds = sum(p.seconds for p in picks)
    note = (
        f"clone {cloning.backend.name}, card {new}, from {len(picks)} reference(s) in "
        f"{episode} ({clips}; {seconds:.1f} s of {found.seconds:.1f} s eligible); "
        "auto and unlocked"
    )
    for warning in reference_report(picks):  # ruling CD5
        note += f"; {warning}"
    return new, note


def extras_voices(cues: Iterable[DubCue], cfg: Config) -> dict[str, str]:
    """Extra label -> stock voice id from the ``[tts]`` backend's extras pools, in order of
    first appearance over every cue (spoken or not), so a label keeps its voice when a
    translation changes. Unknown genders, so the f and m extras pools alternate."""
    backend = tts_backend(cfg)
    out: dict[str, str] = {}
    used: list[str] = []
    for c in cues:
        if c.character_id is not None:
            continue
        label = c.extra or UNKNOWN_EXTRA
        if label in out:
            continue
        voice = pick_stock(backend, cfg.target_language, "u", used, extras=True)
        out[label] = voice
        used.append(voice)
    return out


# -- the production clone source -------------------------------------------------------------


def character_candidates(
    ws: Workspace,
    episodes: Sequence[str],
    cid: int,
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
) -> dict[str, list[RefPick]]:
    """Per episode, the character's eligible reference turns (``eligible_references``, songs
    and singing excluded), ranked. ``exclusions`` caches each episode's excluded spans."""
    from fukikae.artifacts import read_json
    from fukikae.cards.select import eligible_references, reference_exclusions
    from fukikae.models import Assignments, Tags, Transcript
    from fukikae.stages.mix import mix_settings

    cache = exclusions if exclusions is not None else {}
    out: dict[str, list[RefPick]] = {}
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
        out[e] = eligible_references(
            asg.turns, speakers, read_json(tr_path, Transcript), tags, vocals, exclude=cache[e]
        )
    return out


def character_references(
    ws: Workspace,
    episodes: Sequence[str],
    cid: int,
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
    candidates: Mapping[str, list[RefPick]] | None = None,
) -> tuple[str, list[RefPick]] | None:
    """The character's reference clips (``pick_references`` over ``character_candidates``)
    in the episode where they add up to the most speech, with that episode; None when no
    episode has enough."""
    from fukikae.cards.select import pick_references

    found = (
        candidates
        if candidates is not None
        else character_candidates(ws, episodes, cid, exclusions)
    )
    best: tuple[str, list[RefPick]] | None = None
    for e in episodes:
        picks = pick_references(found[e]) if e in found else None
        if picks and (
            best is None or sum(p.seconds for p in picks) > sum(p.seconds for p in best[1])
        ):
            best = (e, picks)
    return best


def workspace_episodes(ws: Workspace, current: str) -> list[str]:
    """``current`` first, then every other episode of the workspace with a work dir."""
    base = ws.root / ".fukikae" / "episodes"
    others = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    return [current, *(e for e in others if e != current)]


def workspace_cloning(root: Path, cfg: Config, current: str) -> Cloning | None:
    """The clone branch for an episode of the workspace at ``root``, or None when ``[tts]
    clone_backend`` is unset: references from the character's turns in every episode the
    workspace has processed so far (the current one first), the source language the current
    episode's transcript names (else ``source_language``)."""
    from typing import cast

    from fukikae.artifacts import read_json
    from fukikae.backends import loader
    from fukikae.backends.base import Synthesizer
    from fukikae.config import ConfigError
    from fukikae.models import Transcript
    from fukikae.workspace import Workspace

    name = clone_backend(cfg)
    if name is None:
        return None
    cls = cast(type[Synthesizer], loader.backend_class("tts", name))
    if "clone" not in getattr(cls, "sources", ()):
        raise ConfigError(f"[tts] clone_backend {name!r} makes no clone voices")
    ws = Workspace(root)
    episodes = workspace_episodes(ws, current)
    tr_path = ws.work_dir(current) / "03-asr" / "transcript.json"
    language = read_json(tr_path, Transcript).language if tr_path.exists() else None
    language = language or cfg.source_language
    if not language:
        raise ConfigError(
            f"{current}: no transcript language and no source_language for the clone references"
        )
    exclusions: dict[str, list[tuple[float, float]]] = {}

    def references(cid: int) -> CharacterRefs:
        candidates = character_candidates(ws, episodes, cid, exclusions)
        seconds = max((sum(p.seconds for p in c) for c in candidates.values()), default=0.0)
        best = character_references(ws, episodes, cid, candidates=candidates)
        if best is None:
            return CharacterRefs(seconds=round(seconds, 3))
        e, picks = best
        vocals = ws.work_dir(e) / "01-separate" / "vocals.wav"
        return CharacterRefs(round(seconds, 3), picks, e, vocals)

    return Cloning(backend=cls, source_language=language, references=references)
