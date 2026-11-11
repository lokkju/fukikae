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

Subtitle names (D27, D29 #4). In an episode whose names stage used names, a character's
reference turns are its name-confirmed ones (`confirmed_labels`: sub-clusters it got by name,
or by a lock on a key it answers to), never a span of an inner-voice or multi line
(``names.excluded``); names-used episodes are tried first. Only when no episode's
name-confirmed turns yield picks does the character's own acoustic speech count (its unnamed
remainders included, and acoustic-only episodes), before a stock voice. In both, a turn with
`STALE_OVERLAP` or more of it under speech the names give someone else (`foreign_spans`) is
never a candidate. A clone card with a reference that much under another character's names is
stale (`stale_cards`); unnamed speech never makes a card stale. ``ensure_assignments``
re-picks a stale card as it would make a new card (a new card id and a new ``auto`` row) when
the card is automatic (``replace_refusal`` lets ``auto`` replace it: ``auto``,
``auto-fallback`` or ``restock``, unlocked); a curated or imported card is only reported.

An episode counts as names-used only while its ``assignments.json`` answers the names stage's
refined clusters (the same turns as ``04c-names/turns.json``, and every names label among its
clusters; `checked_names`, the guard ``registry resolve`` applies too). One whose names were
refined after its characters stage ran is skipped, with a report line, until that stage
reruns. Without names nothing here changes.

The clone floor (D32): a clone wants 2 clips and 10 s of reference speech (``select``'s
``FLOOR_CLIPS`` and ``FLOOR_S``). Episodes whose picks reach it are preferred over ones with
more seconds that don't; when none does, the one closest to it (`_floor_key`). A character
whose best picks fall short still gets a clone from them (6 s is still the minimum for any
clone), and the clone is flagged for review. An
automatic clone card under the floor (`below_floor`; curated and imported cards are never
judged) is re-picked like a stale one when the current rules give picks that beat it
(`_floor_key`: reaching the floor, then clips up to 2, then seconds); otherwise, or when it is
locked, it is kept and flagged in the tts run's flags, ``fukikae voices list`` and ``fukikae
names show`` (`floor_lines`), naming the shortfall and pointing at ``fukikae voices clone``.
It never falls back to a stock voice or another character's card.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fukikae.cards.select import (
    FLOOR_CLIPS,
    FOREIGN_MAX,
    Gated,
    floor_shortfall,
    gender_of,
    meets_floor,
    share_inside,
    speaker_f0,
)
from fukikae.cards.stock import check_stock_voice, pick_stock, revision, stock_card
from fukikae.cards.store import CardStore
from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW
from fukikae.registry.store import AUTOMATIC_BY, ProtectedAssignmentError, replace_refusal

if TYPE_CHECKING:
    import numpy as np

    from fukikae.backends.base import Synthesizer
    from fukikae.cards.card import Card
    from fukikae.cards.select import RefPick, Register
    from fukikae.config import Config
    from fukikae.models import Assignments, ClusterAssignment, DubCue, Tags, Transcript
    from fukikae.names.models import NamesResult
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

log = logging.getLogger(__name__)

__all__ = [
    "AUTOMATIC_BY",
    "FALLBACK_BY",
    "FLOOR_NOTE",
    "REPICK_NOTE",
    "STALE_OVERLAP",
    "CharacterRefs",
    "Cloning",
    "ProtectedAssignmentError",
    "Shortfall",
    "below_floor",
    "character_candidates",
    "character_references",
    "character_spans",
    "checked_names",
    "clone_backend",
    "confirmed_labels",
    "ensure_assignments",
    "extras_voices",
    "find_references",
    "floor_lines",
    "floor_suffix",
    "foreign_spans",
    "gender_thresholds",
    "key_owner",
    "repick_blocker",
    "replace_refusal",
    "stale_cards",
    "stale_references",
    "tts_backend",
    "workspace_cloning",
]

UNKNOWN_EXTRA = "unknown"
FALLBACK_BY = "auto-fallback"  # a stock stand-in while the clone backend can't run here
MIN_CLONE_S = 6.0  # the spec's minimum of clean speech for a clone (pick_references' default)
STALE_OVERLAP = FOREIGN_MAX  # this share of a reference under someone else's names: stale
REPICK_NOTE = "re-picked: references not name-confirmed"
FLOOR_NOTE = "re-picked: below the clone floor"
FLOOR_FIX = "`fukikae voices clone` with a better clip replaces it"


@dataclass(frozen=True)
class CharacterRefs:
    """What a character's turns offer a clone: the picks (with their episode and its vocals)
    and ``seconds``, the eligible reference speech of the picked episode in the tier the picks
    come from (name-confirmed or acoustic); with no picks (under the minimum), ``seconds`` is
    the most any episode offers in either tier."""

    seconds: float
    picks: list[RefPick] | None = None
    episode: str | None = None
    vocals: Path | None = None
    language: str | None = None  # the picks' episode's transcript language
    confirmed: bool = False  # the picks are name-confirmed turns of a names-used episode
    names_used: bool = False  # some episode searched used subtitle names
    gated: Gated | None = None  # what the A1 gates refused in the picks' episode and tier


@dataclass(frozen=True)
class Cloning:
    """The clone branch of ``ensure_assignments``: the backend clone cards are made for, the
    language the references speak, and each character's references."""

    backend: type[Synthesizer]
    source_language: str
    references: Callable[[int], CharacterRefs]
    unavailable: str | None = None  # why the backend can't run here: stock for everyone


@dataclass(frozen=True)
class Shortfall:
    """An automatic clone card under the clone floor (D32 #2): its id and its clips' lengths
    in seconds (from each reference's origin)."""

    card_id: str
    clips: tuple[float, ...]

    @property
    def phrase(self) -> str:
        return floor_shortfall(self.clips) or ""


def _floor_key(clips: Sequence[float]) -> tuple[bool, int, float]:
    """How good a set of clips is against the floor: reaching it, then clips up to
    `FLOOR_CLIPS`, then seconds. A re-pick must beat the current card on this."""
    return (
        floor_shortfall(clips) is None,
        min(len(clips), FLOOR_CLIPS),
        round(sum(clips), 2),
    )


def _better(found: CharacterRefs | None, short: Shortfall) -> bool:
    """Whether ``found``'s picks beat the card's clips (`_floor_key`)."""
    if found is None or not found.picks:
        return False
    return _floor_key([p.seconds for p in found.picks]) > _floor_key(short.clips)


def clip_lengths(card: Card) -> tuple[float, ...] | None:
    """The card's reference lengths in seconds, from their origins; None when a reference has
    no usable origin (the floor can't be judged)."""
    out: list[float] = []
    for ref in card.voice.references:
        o = ref.origin or {}
        start, end = o.get("start"), o.get("end")
        if not isinstance(start, int | float) or not isinstance(end, int | float):
            return None
        out.append(float(end) - float(start))
    return tuple(out)


def below_floor(
    reg: Registry, store: CardStore, characters: Iterable[int], language: str
) -> dict[int, Shortfall]:
    """Character -> its ``language`` card's `Shortfall`, for the characters whose card is an
    automatic clone card (``auto``, ``auto-fallback`` or ``restock``, locked or not) under the
    clone floor. Curated and imported cards are the operator's choice and never listed."""
    from fukikae.cards.card import CardError

    out: dict[int, Shortfall] = {}
    for cid in characters:
        a = reg.card_for(cid, language)
        if a is None or a.card_id is None or a.chosen_by not in AUTOMATIC_BY:
            continue
        if not store.has(a.card_id):
            continue
        try:
            card = store.get(a.card_id)
        except CardError:
            continue  # stale_cards and the tts stage's checks name a card that can't be read
        if card.voice.source != "clone" or not card.voice.references:
            continue
        clips = clip_lengths(card)
        if clips is not None and floor_shortfall(clips) is not None:
            out[cid] = Shortfall(a.card_id, clips)
    return out


def floor_suffix(short: Mapping[int, Shortfall] | None, cid: int) -> str:
    """What a kept stale card's line appends when the card is under the clone floor too (its
    floor line is not printed separately); empty otherwise."""
    sf = (short or {}).get(cid)
    return f"; below the clone floor ({sf.phrase})" if sf is not None else ""


def _floor_flag(reg: Registry, cid: int, card: str, phrase: str, why: str | None = None) -> str:
    """The flag for a clone card under the floor that stays: no better pick, or ``why``."""
    head = f"character {_label(reg, cid)}: clone card {card} is below the clone floor ({phrase})"
    if why is None:
        return f"{head} and no better reference pick exists; kept, flagged for review: {FLOOR_FIX}"
    return f"{head}; {why}: {FLOOR_FIX}"


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
    if name:
        from fukikae.backends.loader import engine_problem

        problem = engine_problem("tts", name, "[tts] clone_backend")
        if problem:
            raise ConfigError(problem)
    return name or None


def tts_backend(cfg: Config) -> str:
    """The ``[tts]`` backend: the one automatic cards and extras are made for (never an engine
    of another backend's cards)."""
    from fukikae.backends.loader import engine_problem
    from fukikae.config import ConfigError

    name = cfg.resolve("tts").backend
    problem = engine_problem("tts", name, "[tts] backend")
    if problem:
        raise ConfigError(problem)
    return name


MEDIAN_WARNING = (
    "[tts.gender] now reads the median F0, not the 20th percentile; "
    "check male_below and female_above"
)
_MEDIAN_WARNED: set[tuple[tuple[str, object], ...]] = set()


def gender_thresholds(cfg: Config) -> tuple[float, float]:
    """(male_below, female_above) in Hz from ``[tts] gender``, the workspace's over the
    profile's, else ``fukikae.pitch``'s ``MALE_BELOW`` and ``FEMALE_ABOVE`` (165 and 200 Hz,
    calibrated on the median); F0 between them is an unknown gender. A workspace (not a
    profile) that sets either key was tuned on the old 20th-percentile estimate, which gets a
    warning once per run."""
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
    ws = cfg.raw.get("tts")
    ws_gender = ws.get("gender") if isinstance(ws, dict) else None
    if isinstance(ws_gender, dict) and {"male_below", "female_above"} & set(ws_gender):
        key = tuple(sorted(ws_gender.items()))
        if key not in _MEDIAN_WARNED:
            _MEDIAN_WARNED.add(key)
            log.warning(MEDIAN_WARNING)
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


def _where(refs: Sequence[dict]) -> str:
    """The stale references as a report phrase: episode, span and the names said over it."""
    parts = []
    for r in refs:
        said = ", ".join(r.get("names") or []) or "no name"
        parts.append(f"{r['episode']} {r['start']:.2f}-{r['end']:.2f} ({said})")
    return "; ".join(parts)


def ensure_assignments(
    reg: Registry,
    store: CardStore,
    cfg: Config,
    character_spans: dict[int, list[tuple[float, float]]],
    vocals: Path,
    *,
    cloning: Cloning | None = None,
    stale: Mapping[int, Sequence[dict]] | None = None,
    short: Mapping[int, Shortfall] | None = None,
    flags: list[str] | None = None,
) -> list[str]:
    """Give every character in ``character_spans`` without a ``cfg.target_language`` card
    one: a clone card when ``cloning`` is given and the character has enough clean speech,
    else a stock card from its gender's pool, most speech first (so the main cast gets the
    first voices of each pool). Returns one report line per card assigned: the lines that
    changed the registry.

    ``stale`` (`stale_cards`) lists the characters whose clone card has references the
    subtitle names give someone else. While ``cloning`` is usable, an automatic card
    (``replace_refusal`` lets ``auto`` replace it) is re-picked the way a new card is made, a
    new card with a new ``auto`` row noted `REPICK_NOTE`: a clone from name-confirmed
    references when there are enough, else from the character's own acoustic speech, else a
    stock voice, so a character whose speech the names gave to others stops speaking in their
    voice. A curated or imported card, and an automatic one while no clone can be made, is
    only flagged (D29 #4): the flag lines go to ``flags`` (when given), never to the returned
    lines, because they change nothing and would repeat on every run.

    ``short`` (`below_floor`) lists automatic clone cards under the clone floor (D32). An
    unlocked one is re-picked the same way (noted `FLOOR_NOTE`) when ``cloning`` gives picks
    that beat it (`_floor_key`); otherwise, or when locked, it is kept and flagged, never
    replaced by a stock voice. A new clone card under the floor is made, from the best picks
    there are, and flagged too."""
    language = cfg.target_language
    usable = cloning is not None and cloning.unavailable is None
    todo = []
    fallbacks: set[int] = set()
    repick: dict[int, str] = {}  # character -> its current card id
    floor: dict[int, Shortfall] = {}  # characters re-picked for the clone floor
    lines: list[str] = []
    flagged = flags if flags is not None else []
    for cid, spans in character_spans.items():
        a = reg.card_for(cid, language)
        refs = (stale or {}).get(cid)
        if a is None or a.card_id is None:
            todo.append((cid, spans))
        elif usable and a.chosen_by == FALLBACK_BY and not a.locked:
            todo.append((cid, spans))  # the stock stand-in for a clone that couldn't run
            fallbacks.add(cid)
        elif refs:
            where = _where(refs)
            also = floor_suffix(short, cid)
            if a.chosen_by not in AUTOMATIC_BY or replace_refusal(a, "auto") is not None:
                flagged.append(
                    f"character {_label(reg, cid)}'s card {a.card_id} ({a.chosen_by}"
                    f"{', locked' if a.locked else ''}) was cloned from turns the subtitles "
                    f"now name otherwise: {where}; `fukikae voices clone` replaces it{also}"
                )
            elif usable:
                todo.append((cid, spans))
                repick[cid] = a.card_id
            elif cloning is None:
                flagged.append(
                    f"character {_label(reg, cid)}: card {a.card_id} is cloned from turns the "
                    f"subtitles now name otherwise ({where}); kept, because no [tts] "
                    "clone_backend is set to re-pick it with: set one, or replace it with "
                    f"`fukikae voices clone` or `fukikae voices assign`{also}"
                )
            else:
                name = cloning.backend.name
                flagged.append(
                    f"character {_label(reg, cid)}: card {a.card_id} is cloned from turns the "
                    f"subtitles now name otherwise ({where}); kept until the clone backend "
                    f"{name} can run ({cloning.unavailable}; `fukikae backends setup {name}`), "
                    f"and the next tts run then re-picks it{also}"
                )
        elif cid in (short or {}):
            sf = (short or {})[cid]
            refusal = replace_refusal(a, "auto")
            if refusal is not None:
                flagged.append(_floor_flag(reg, cid, sf.card_id, sf.phrase, f"kept: {refusal}"))
            elif usable:
                todo.append((cid, spans))
                floor[cid] = sf
            elif cloning is None:
                flagged.append(_floor_flag(
                    reg, cid, sf.card_id, sf.phrase,
                    "kept, because no [tts] clone_backend is set to re-pick it with",
                ))  # fmt: skip
            else:
                name = cloning.backend.name
                flagged.append(_floor_flag(
                    reg, cid, sf.card_id, sf.phrase,
                    f"kept until the clone backend {name} can run ({cloning.unavailable}; "
                    f"`fukikae backends setup {name}`), when the next tts run re-picks it if "
                    "a better pick exists",
                ))  # fmt: skip
    if not todo:
        return lines
    backend = tts_backend(cfg)
    rev = revision(backend)
    male_below, female_above = gender_thresholds(cfg)
    used = _stock_voices_in_use(reg, store, backend, language)
    todo.sort(key=lambda item: (-sum(e - s for s, e in item[1]), item[0]))
    for cid, spans in todo:
        found = cloning.references(cid) if usable else None  # type: ignore[union-attr]
        voice: str | None = None
        by = "auto"
        if cid in fallbacks and not (found is not None and found.picks):
            continue  # still too little speech: the stand-in stays until a clone can be made
        if cid in floor and not _better(found, floor[cid]):
            sf = floor[cid]
            flagged.append(_floor_flag(reg, cid, sf.card_id, sf.phrase))
            continue  # the best picks there are: kept, flagged (never a stock voice)
        below: str | None = None
        if found is not None and found.picks:
            new, note = _clone(cloning, found, store, language)  # type: ignore[arg-type]
            below = floor_shortfall([p.seconds for p in found.picks])
            if below is not None:
                note += f"; below the clone floor ({below})"
            if cid in fallbacks:
                note += "; replaces the stock fallback made while the clone backend couldn't run"
        else:
            f0, seconds = speaker_f0(vocals, spans)
            gender = gender_of(f0, male_below, female_above)
            voice = pick_stock(backend, language, gender, used)
            check_stock_voice(backend, voice)
            new = store.put(stock_card(backend, voice, language, rev))
            estimate = (
                f"gender estimate {gender}, F0 {f0:.0f} Hz (median) over {seconds:.1f} s"
                if f0 is not None
                else f"gender estimate u, too little voiced speech in {seconds:.1f} s"
            )
            note = f"stock {backend}:{voice}, card {new} ({estimate}); auto and unlocked"
            if found is not None:
                note = (
                    f"{found.seconds:.1f} s of clean speech, under {MIN_CLONE_S:.0f} s; "
                    f"stock voice {voice}: {note}"
                )
            elif cloning is not None and cloning.unavailable is not None:
                name = cloning.backend.name
                note = (
                    f"clone backend {name} is not usable ({cloning.unavailable}); stock voice "
                    f"{voice}: {note}. Run `fukikae backends setup {name}`: the next run "
                    "replaces this stand-in with a clone"
                )
                by = FALLBACK_BY
        if cid in repick:
            if new == repick[cid]:
                continue
            note = f"{REPICK_NOTE} ({_where((stale or {})[cid])}); {note}"
        elif cid in floor:
            note = f"{FLOOR_NOTE} ({floor[cid].phrase}); {note}"
        try:
            reg.assign_card(cid, language, new, by=by, locked=False, note=note)
        except ProtectedAssignmentError as e:
            lines.append(f"character {_label(reg, cid)}: kept its card ({e})")
            continue
        if voice is not None:
            used.append(voice)
        lines.append(f"character {_label(reg, cid)}: {note}")
        if below is not None:
            flagged.append(_floor_flag(reg, cid, new, below))
    return lines


def _clone(
    cloning: Cloning, found: CharacterRefs, store: CardStore, language: str
) -> tuple[str, str]:
    """(card id, note) of a clone card from ``found``'s picks, put in the store."""
    from fukikae.cards.select import clone_card, reference_report

    picks, episode, vocals = found.picks or [], found.episode, found.vocals
    assert episode is not None and vocals is not None
    card = clone_card(
        cloning.backend, picks, store, vocals, found.language or cloning.source_language,
        language, episode,
    )  # fmt: skip
    new = store.put(card)
    clips = ", ".join(f"{p.start:.2f}-{p.end:.2f}" for p in picks)
    seconds = sum(p.seconds for p in picks)
    note = (
        f"clone {cloning.backend.name}, card {new}, from {len(picks)} reference(s) in "
        f"{episode} ({clips}; {seconds:.1f} s of {found.seconds:.1f} s eligible); "
        "auto and unlocked"
    )
    for warning in reference_report(picks, found.gated):  # rulings CD5 and A1
        note += f"; {warning}"
    if found.names_used and not found.confirmed:
        note += "; references are not name-confirmed"
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


@dataclass(frozen=True)
class EpisodeArtifacts:
    """What reference selection reads from one episode's work dir."""

    assignments: Assignments
    transcript: Transcript
    tags: Tags | None
    vocals: Path
    names: NamesResult | None = None  # the names stage's result when it used names
    skipped: str | None = None  # why a names-used episode is left out (`checked_names`)


def used_names(work: Path) -> NamesResult | None:
    """The episode's ``names.json`` when its names stage used names, else None."""
    from fukikae.artifacts import read_json
    from fukikae.names.inputs import names_path
    from fukikae.names.models import NamesResult

    path = names_path(work)
    if not path.exists():
        return None
    result = read_json(path, NamesResult)
    return result if result.used else None


SKIP_HINT = (
    "skipped for clone references and stale cards until its characters stage reruns "
    "(`fukikae run <ws> {episode} --from characters`)"
)


def names_mismatch(work: Path, names: NamesResult, asg: Assignments | None) -> str | None:
    """Why the episode's ``assignments.json`` doesn't answer its names stage's refined
    clusters (other turns than ``04c-names/turns.json``, or a names label missing from its
    clusters), or None when it does: resolve's guard (B1), for reference picking."""
    from fukikae.artifacts import read_json
    from fukikae.models import Diarization
    from fukikae.names.inputs import NAMES_DIR

    if asg is None:
        return "its names were used but it has no assignments.json"
    path = Path(work) / NAMES_DIR / "turns.json"
    turns = read_json(path, Diarization).turns if path.exists() else None
    if asg.turns != turns or not set(names.labels) <= {c.cluster for c in asg.clusters}:
        return "its assignments.json is older than its names (the names stage refined the clusters after the characters stage ran)"  # noqa: E501
    return None


def checked_names(work: Path, asg: Assignments | None) -> tuple[NamesResult | None, str | None]:
    """(names, why skipped) for an episode: its used ``names.json`` and None when its
    assignments answer the refined clusters (`names_mismatch`), (None, the reason) when they
    don't, (None, None) when its names stage used no names."""
    names = used_names(work)
    if names is None:
        return None, None
    why = names_mismatch(work, names, asg)
    return (None, why) if why is not None else (names, None)


def episode_artifacts(ws: Workspace, episode: str) -> EpisodeArtifacts | None:
    """The episode's assignments, transcript, tags, vocals and used names; None when one of
    the first four (tags aside) is missing. A names-used episode whose assignments are older
    than its names comes back with ``skipped`` set and no names (`checked_names`)."""
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments, Tags, Transcript

    work = ws.work_dir(episode)
    asg_path = work / "05-characters" / "assignments.json"
    tr_path = work / "03-asr" / "transcript.json"
    vocals = work / "01-separate" / "vocals.wav"
    if not (asg_path.exists() and tr_path.exists() and vocals.exists()):
        return None
    tags_path = work / "04b-tag" / "tags.json"
    asg = read_json(asg_path, Assignments)
    names, why = checked_names(work, asg)
    return EpisodeArtifacts(
        asg,
        read_json(tr_path, Transcript),
        read_json(tags_path, Tags) if tags_path.exists() else None,
        vocals,
        names,
        why,
    )


# key -> the character id it answers to (None for nobody, or for several at one rank)
type KeyOwner = Callable[[str], int | None]


def key_owner(reg: Registry) -> KeyOwner:
    """`Registry.character_for_key` as a cached id lookup; an ambiguous key answers nobody."""
    cache: dict[str, int | None] = {}

    def owner(key: str) -> int | None:
        if key not in cache:
            try:
                c = reg.character_for_key(key)
            except ValueError:
                c = None
            cache[key] = None if c is None else c.id
        return cache[key]

    return owner


def _workspace_owner(ws: Workspace) -> KeyOwner:
    """`key_owner` over the workspace's registry, opened for each new key; nobody without one."""
    from fukikae.registry.store import Registry

    cache: dict[str, int | None] = {}

    def owner(key: str) -> int | None:
        if key not in cache:
            if not ws.registry_path.exists():
                cache[key] = None
            else:
                with Registry(ws.registry_path) as reg:
                    cache[key] = key_owner(reg)(key)
        return cache[key]

    return owner


def _confirmed(c: ClusterAssignment, owner: KeyOwner) -> bool:
    """A name-confirmed assignment: a sub-cluster its character got by name, or a lock-kept
    one whose key that character answers to."""
    if c.character_id is None or c.status in ("extra", "review"):
        return False
    return c.source == "name" or (
        c.source == "lock" and c.name is not None and owner(c.name) == c.character_id
    )


def confirmed_labels(asg: Assignments, cid: int, owner: KeyOwner) -> set[str]:
    """The character's name-confirmed labels in a names-used episode: sub-clusters it got by
    name (``source = "name"``), and lock-kept ones whose key it answers to (a lock on a key
    that names someone else keeps the speech but confirms nothing). Unnamed remainders never
    count."""
    return {c.cluster for c in asg.clusters if c.character_id == cid and _confirmed(c, owner)}


type Spans = list[tuple[float, float]]


def _union(spans: Iterable[tuple[float, float]]) -> Spans:
    out: Spans = []
    for a, b in sorted((float(a), float(b)) for a, b in spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _subtract(spans: Iterable[tuple[float, float]], cut: Iterable[tuple[float, float]]) -> Spans:
    """``spans`` (merged) less ``cut``."""
    out: Spans = []
    holes = _union(cut)
    for a, b in _union(spans):
        edge = a
        for x, y in holes:
            if y <= edge or x >= b:
                continue
            if x > edge:
                out.append((edge, x))
            edge = max(edge, y)
        if edge < b:
            out.append((edge, b))
    return out


def _answers(asg: Assignments, cid: int, owner: KeyOwner) -> Callable[[str], bool]:
    """Whether the character answers to a key in this episode: a sub-cluster of that key went
    to it by name, or the registry gives it the key. A sub-cluster a lock kept on it doesn't
    count: the lock keeps that speech whatever its name, but the key stays someone else's
    (its turns are subtracted from `foreign_spans` instead)."""
    mine = {
        c.name
        for c in asg.clusters
        if c.character_id == cid
        and c.name
        and c.source == "name"
        and c.status not in ("extra", "review")
    }
    return lambda key: key in mine or owner(key) == cid


def foreign_spans(asg: Assignments, names: NamesResult, cid: int, owner: KeyOwner) -> Spans:
    """Speech the episode's names give someone other than the character, merged: turns
    name-confirmed for another character, and visible named lines (named or role) of a key the
    character doesn't answer to (`_answers`), less the character's own lock-kept turns (a lock
    keeps speech whatever its names say). Unnamed speech is nobody's here."""
    theirs = {c.cluster for c in asg.clusters if c.character_id != cid and _confirmed(c, owner)}
    kept = {c.cluster for c in asg.clusters if c.character_id == cid and c.source == "lock"}
    answers = _answers(asg, cid, owner)
    turns = asg.turns or []
    spans = [(t.start, t.end) for t in turns if t.speaker in theirs]
    spans += [(ln.start, ln.end) for ln in names.visible_lines() if ln.key and not answers(ln.key)]
    return _subtract(spans, [(t.start, t.end) for t in turns if t.speaker in kept])


def character_candidates(
    ws: Workspace,
    episodes: Sequence[str],
    cid: int,
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
    loaded: dict[str, EpisodeArtifacts | None] | None = None,
    owner: KeyOwner | None = None,
    *,
    acoustic: bool = False,
    gated: dict[str, Gated] | None = None,
) -> dict[str, list[RefPick]]:
    """Per episode, the character's eligible reference turns (``eligible_references``, songs
    and singing excluded), ranked. ``exclusions`` caches each episode's excluded spans and
    ``loaded`` its artifacts, so a caller asking for many characters reads each file once.

    The A1 gates run on every episode's candidates. The register gate runs only in the
    acoustic tier (no names, or ``acoustic``): a name-confirmed turn is the character's by its
    subtitle name, and a wide-ranging character's confirmed speech has a majority register
    that refuses its calmer turns (Fairy Tail E13: Erza's 143 to 179 Hz alto against a
    shouted 261 Hz register left her no reference at all). There the register is the
    character's majority register over the turns searched in that episode
    (``select.speaker_register``, on ``fukikae.pitch``'s bands, not
    `gender_thresholds`: a profile band tuned on the old estimator, such as the anime
    profile's 300 Hz pending D44, would keep only the turns above it and push the register
    up by 2 to 4 semitones on Sol Bianca). The voicing gate runs in both tiers, gate 3 reads
    the song detector's per-second mask (``select.reference_gating``) and each candidate's
    sung-ness, and the pitch tracks go through the workspace's pitch cache. ``gated``
    receives each episode's `Gated` counts (``split`` when the register is bimodal).

    In an episode whose names were used (``owner`` answers keys, the workspace's registry by
    default), the candidates are the character's name-confirmed turns (`confirmed_labels`),
    or with ``acoustic`` all of its turns, unnamed remainders included; either way never a
    span of the names stage's inner-voice and multi lines, nor a turn with `STALE_OVERLAP` of
    it under someone else's names (`foreign_spans`). A names-used episode whose assignments
    are older than its names (``skipped``) offers nothing."""
    from fukikae.cards.select import (
        Register,
        eligible_references,
        reference_exclusions,
        reference_gating,
        speaker_register,
    )
    from fukikae.stages.mix import mix_settings

    bands = (float(MALE_BELOW), float(FEMALE_ABOVE))
    cache = exclusions if exclusions is not None else {}
    arts = loaded if loaded is not None else {}
    out: dict[str, list[RefPick]] = {}
    for e in episodes:
        if e not in arts:
            arts[e] = episode_artifacts(ws, e)
        a = arts[e]
        if a is None or a.skipped is not None:
            continue
        own = {
            c.cluster
            for c in a.assignments.clusters
            if c.character_id == cid and c.status not in ("extra", "review")
        }
        foreign: Spans = []
        confirmed = False
        if a.names is not None:
            if owner is None:
                owner = _workspace_owner(ws)
            confirmed = not acoustic
            speakers = own if acoustic else confirmed_labels(a.assignments, cid, owner)
            foreign = foreign_spans(a.assignments, a.names, cid, owner)
        else:
            speakers = own
        if not speakers or not a.assignments.turns:
            continue
        work = ws.work_dir(e)
        if e not in cache:
            songs = mix_settings(ws.config).songs
            key = (str(work), repr(songs), _stamp(*_exclusion_files(work)))
            if key not in _EXCLUSIONS_MEMO:
                _remember(_EXCLUSIONS_MEMO, key, reference_exclusions(work, a.tags, songs))
            cache[e] = list(_EXCLUSIONS_MEMO[key])
        exclude = cache[e]
        if a.names is not None:
            exclude = [*exclude, *((float(x), float(y)) for x, y in a.names.excluded)]
        songs = mix_settings(ws.config).songs
        gkey = (str(work), repr(songs), _stamp(*_exclusion_files(work)))
        if gkey not in _GATING_MEMO:
            _remember(_GATING_MEMO, gkey, reference_gating(work, songs))
        song_like = _GATING_MEMO[gkey]
        spans = [(t.start, t.end) for t in a.assignments.turns if t.speaker in speakers]
        rkey = (str(work), tuple(spans), bands, _stamp(a.vocals))
        if rkey not in _REGISTER_MEMO:
            _remember(_REGISTER_MEMO, rkey, speaker_register(a.vocals, spans, bands, ws.root))
        register = _REGISTER_MEMO[rkey]
        if confirmed:
            register = Register("u", register.median)  # names vouch: no gate, no split flag
        key = (
            str(work), tuple(sorted(speakers)), tuple(exclude), tuple(foreign), register,
            gkey, _stamp(*_candidate_files(work)),
        )  # fmt: skip
        if key not in _CANDIDATES_MEMO:
            counts: list[Gated] = []
            picks = eligible_references(
                a.assignments.turns, speakers, a.transcript, a.tags, a.vocals, exclude=exclude,
                foreign=foreign, register=register, song_like=song_like, bands=bands,
                pitch_root=ws.root, gated=counts,
            )  # fmt: skip
            _remember(_CANDIDATES_MEMO, key, (picks, counts[0] if counts else Gated()))
        picks, found = _CANDIDATES_MEMO[key]
        out[e] = list(picks)
        if gated is not None:
            gated[e] = found
    return out


# Per process (a `fukikae run` of several episodes, or one CLI command): each episode's
# reference candidates and song exclusions, keyed by everything they're made from (the
# registry's part already resolved into speakers and foreign spans) and the files' mtimes
# and sizes, so every tts run of the batch and every character asks the files once (D32
# review: a flagged character with no better pick is looked up again on each run).
_CANDIDATES_MEMO: dict[tuple, tuple[list[RefPick], Gated]] = {}
_EXCLUSIONS_MEMO: dict[tuple, list[tuple[float, float]]] = {}
# each episode's gate 3 evidence (the song mask) and each speaker's register (A1)
_GATING_MEMO: dict[tuple, np.ndarray | None] = {}
_REGISTER_MEMO: dict[tuple, Register] = {}
_MEMO_MAX = 4096


def _remember(memo: dict, key: tuple, value: object) -> None:
    if len(memo) >= _MEMO_MAX:
        memo.clear()
    memo[key] = value


def _stamp(*paths: Path) -> tuple[tuple[str, int, int], ...]:
    """Each file's (path, mtime in ns, size); -1s for a missing one."""
    out = []
    for p in paths:
        try:
            st = p.stat()
        except OSError:
            out.append((str(p), -1, -1))
        else:
            out.append((str(p), st.st_mtime_ns, st.st_size))
    return tuple(out)


def _candidate_files(work: Path) -> tuple[Path, ...]:
    return (
        work / "05-characters" / "assignments.json",
        work / "03-asr" / "transcript.json",
        work / "04b-tag" / "tags.json",
        work / "01-separate" / "vocals.wav",
    )


def _exclusion_files(work: Path) -> tuple[Path, ...]:
    sep = work / "01-separate"
    return (
        sep / "background.wav", sep / "vocals.wav", sep / "vocals_16k.wav",
        work / "04b-tag" / "tags.json",
    )  # fmt: skip


def _best(
    episodes: Sequence[str], found: Mapping[str, list[RefPick]], allowed: Collection[str] | None
) -> tuple[str, list[RefPick]] | None:
    from fukikae.cards.select import pick_references

    best: tuple[str, list[RefPick]] | None = None
    for e in episodes:
        if allowed is not None and e not in allowed:
            continue
        picks = pick_references(found[e]) if e in found else None
        if picks and (best is None or _episode_key(picks) > _episode_key(best[1])):
            best = (e, picks)
    return best


def _episode_key(picks: Sequence[RefPick]) -> tuple[bool, int, float]:
    """An episode's picks rank by reaching the clone floor (D32). Among episodes that reach
    it, by the seconds picked before any floor filling: the picks up to the first scored
    non-neutral one taken past `MIN_CLONE_S`, which are the picks the rule before D32 made, so
    filling an episode with non-neutral turns never moves a character that already reached
    the floor to it. Below the floor, by how close the picks come (`_floor_key`: clips up to
    2, then seconds)."""
    if not meets_floor(picks):
        _, clips, seconds = _floor_key([p.seconds for p in picks])
        return False, clips, seconds
    used = 0.0
    for p in picks:
        if p.emotion is not None and used >= MIN_CLONE_S:
            break
        used += p.seconds
    return True, 0, used


def character_references(
    ws: Workspace,
    episodes: Sequence[str],
    cid: int,
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
    candidates: Mapping[str, list[RefPick]] | None = None,
    named: Collection[str] | None = None,
    acoustic: Mapping[str, list[RefPick]] | None = None,
    *,
    loaded: dict[str, EpisodeArtifacts | None] | None = None,
    owner: KeyOwner | None = None,
) -> tuple[str, list[RefPick], bool] | None:
    """The character's reference clips (``pick_references``) in the episode where they add
    up to the most speech, with that episode and whether the picks are name-confirmed; None
    when no episode has enough. The name-confirmed candidates (``candidates``, by default
    `character_candidates`) of episodes whose names were used (``named``, read from each
    episode's ``names.json`` by default) are tried first; only when none yields picks, the
    character's own acoustic speech in every episode (``acoustic``, by default
    `character_candidates` with ``acoustic``)."""
    loaded = loaded if loaded is not None else {}
    found = (
        candidates
        if candidates is not None
        else character_candidates(ws, episodes, cid, exclusions, loaded, owner)
    )
    if named is None:
        named = {e for e in episodes if used_names(ws.work_dir(e)) is not None}
    best = _best(episodes, found, named)
    if best is not None:
        return best[0], best[1], True
    if acoustic is None:
        acoustic = {
            **{e: c for e, c in found.items() if e not in named},
            **character_candidates(
                ws, [e for e in episodes if e in named], cid, exclusions, loaded, owner,
                acoustic=True,
            ),
        }  # fmt: skip
    best = _best(episodes, acoustic, None)
    return (best[0], best[1], False) if best is not None else None


# episode -> (assignments, used and current names, why a names-used episode is skipped)
type NamesCache = dict[str, tuple[Assignments | None, NamesResult | None, str | None]]


def _episode_names(
    ws: Workspace, episode: str, cache: NamesCache
) -> tuple[Assignments | None, NamesResult | None, str | None]:
    """The episode's assignments and names when its names were used and its assignments
    answer them (`checked_names`), read once per ``cache``; the assignments are read only for
    a names-used episode."""
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments

    if episode not in cache:
        work = ws.work_dir(episode)
        names = used_names(work)
        if names is None:
            cache[episode] = (None, None, None)
        else:
            path = work / "05-characters" / "assignments.json"
            asg = read_json(path, Assignments) if path.exists() else None
            why = names_mismatch(work, names, asg)
            cache[episode] = (asg, None if why else names, why)
    return cache[episode]


def _all_episodes(ws: Workspace) -> list[str]:
    base = ws.root / ".fukikae" / "episodes"
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []


_inside = share_inside


def stale_references(
    card: Card, foreign: Mapping[str, Sequence[tuple[float, float]]]
) -> list[dict]:
    """The card's reference origins in a names-used episode (a key of ``foreign``, each
    episode's `foreign_spans` for the card's character) with `STALE_OVERLAP` or more of them
    under speech the names give someone else, each as ``{episode, start, end, overlap}``;
    empty for a card with no such reference (a stock card, references from acoustic-only
    episodes, or from speech no name covers)."""
    out: list[dict] = []
    for ref in card.voice.references:
        o = ref.origin or {}
        e, start, end = o.get("episode"), o.get("start"), o.get("end")
        if not isinstance(e, str) or e not in foreign or start is None or end is None:
            continue
        share = _inside(float(start), float(end), foreign[e])
        if share >= STALE_OVERLAP:
            out.append({"episode": e, "start": float(start), "end": float(end),
                        "overlap": round(share, 3)})  # fmt: skip
    return out


def _names_over(
    names: NamesResult, start: float, end: float, answers: Callable[[str], bool]
) -> list[str]:
    """The keys of the visible named lines over ``start``-``end`` that the character doesn't
    answer to, most time first."""
    said: dict[str, float] = {}
    for ln in names.visible_lines():
        ov = min(ln.end, end) - max(ln.start, start)
        if ln.key and ov > 0 and not answers(ln.key):
            said[ln.key] = said.get(ln.key, 0.0) + ov
    return sorted(said, key=lambda k: (-said[k], k))


def stale_cards(
    ws: Workspace,
    reg: Registry,
    store: CardStore,
    characters: Iterable[int],
    language: str,
    *,
    notes: list[str] | None = None,
) -> dict[int, list[dict]]:
    """Character -> the stale references (`stale_references`, each with the other ``names``
    said over it) of its current ``language`` card, for the characters whose card has any.
    Reads nothing more than the registry when no episode of the workspace used names. A
    names-used episode whose assignments are older than its names is left out, and a card
    that can't be read is skipped; each gets a line in ``notes`` (when given)."""
    from fukikae.cards.card import CardError

    cache: NamesCache = {}
    named: list[str] = []
    for e in _all_episodes(ws):
        _, names, why = _episode_names(ws, e, cache)
        if why is not None and notes is not None:
            notes.append(f"{e}: {why}; {SKIP_HINT.format(episode=e)}")
        if names is not None:
            named.append(e)
    if not named:
        return {}
    owner = key_owner(reg)
    out: dict[int, list[dict]] = {}
    for cid in characters:
        a = reg.card_for(cid, language)
        if a is None or a.card_id is None or not store.has(a.card_id):
            continue
        try:
            card = store.get(a.card_id)
        except CardError as err:
            if notes is not None:
                notes.append(
                    f"character {_label(reg, cid)}: card {a.card_id} can't be read ({err}); "
                    "not checked against the subtitle names"
                )
            continue
        if not card.voice.references:
            continue
        foreign: dict[str, Spans] = {}
        for e in named:
            asg, names, _ = cache[e]
            assert asg is not None and names is not None
            foreign[e] = foreign_spans(asg, names, cid, owner)
        refs = stale_references(card, foreign)
        for r in refs:
            asg, names, _ = cache[r["episode"]]
            assert asg is not None and names is not None
            r["names"] = _names_over(names, r["start"], r["end"], _answers(asg, cid, owner))
        if refs:
            out[cid] = refs
    return out


def workspace_episodes(ws: Workspace, current: str) -> list[str]:
    """``current`` first, then every other episode of the workspace with a work dir."""
    return [current, *(e for e in _all_episodes(ws) if e != current)]


def clone_problem(cls: type[Synthesizer], cfg: Config) -> str | None:
    """Why ``cls`` can't render here, or None: a worker backend needs its environment set up
    from its current lock (``fukikae backends setup``), and every backend its device (for
    IndexTTS2, a usable GPU). Checked before any clone card is made or assigned."""
    from fukikae.backends.tts.remote import WorkerSynthesizer
    from fukikae.stages.tts import NoUsableGpuError, synth_device

    if issubclass(cls, WorkerSynthesizer):
        problem = cls({}, "cpu").environment_problem()
        if problem is not None:
            return problem
    try:
        synth_device(cfg, cls)  # a bad device value or a device the backend lacks raises
    except NoUsableGpuError as e:
        return str(e)
    return None


def repick_blocker(cfg: Config) -> str | None:
    """Why the next tts run can't re-pick a stale automatic card with a clone, or None when it
    can: no ``[tts] clone_backend``, or one that can't run here (``clone_problem``)."""
    from typing import cast

    from fukikae.backends import loader
    from fukikae.backends.base import Synthesizer

    name = clone_backend(cfg)
    if name is None:
        return "no [tts] clone_backend is set"
    problem = clone_problem(cast(type[Synthesizer], loader.backend_class("tts", name)), cfg)
    if problem is not None:
        return (
            f"the clone backend {name} can't run here ({problem}; `fukikae backends setup {name}`)"
        )
    return None


def workspace_cloning(root: Path, cfg: Config, current: str) -> Cloning | None:
    """The clone branch for an episode of the workspace at ``root``, or None when ``[tts]
    clone_backend`` is unset: references from the character's turns in every episode the
    workspace has processed so far (the current one first), each card's source language the
    transcript language of the episode its clips come from. When the backend can't run here
    (``clone_problem``), the result says why and every new character gets a stock voice."""
    from typing import cast

    from fukikae.backends import loader
    from fukikae.backends.base import Synthesizer
    from fukikae.config import ConfigError
    from fukikae.workspace import Workspace

    name = clone_backend(cfg)
    if name is None:
        return None
    cls = cast(type[Synthesizer], loader.backend_class("tts", name))
    if "clone" not in getattr(cls, "sources", ()):
        raise ConfigError(f"[tts] clone_backend {name!r} makes no clone voices")

    def nothing(_cid: int) -> CharacterRefs:
        return CharacterRefs(seconds=0.0)

    problem = clone_problem(cls, cfg)
    if problem is not None:
        return Cloning(cls, cfg.source_language or "", nothing, unavailable=problem)
    ws = Workspace(root)
    episodes = workspace_episodes(ws, current)
    loaded: dict[str, EpisodeArtifacts | None] = {}
    exclusions: dict[str, list[tuple[float, float]]] = {}
    here = episode_artifacts(ws, current)
    loaded[current] = here
    fallback = (here.transcript.language if here else None) or cfg.source_language
    if not fallback:
        raise ConfigError(
            f"{current}: no transcript language and no source_language for the clone references"
        )

    def references(cid: int) -> CharacterRefs:
        return find_references(ws, episodes, cid, fallback, loaded, exclusions)

    return Cloning(backend=cls, source_language=fallback, references=references)


def find_references(
    ws: Workspace,
    episodes: Sequence[str],
    cid: int,
    fallback: str | None = None,
    loaded: dict[str, EpisodeArtifacts | None] | None = None,
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
) -> CharacterRefs:
    """What the character's turns in ``episodes`` offer a clone (`CharacterRefs`): the
    name-confirmed tier first, the acoustic one only when that gives no picks
    (`character_references`). Reads no backend, so the CLI can ask it too. ``fallback`` is
    the picks' language when their episode's transcript has none; ``loaded`` and
    ``exclusions`` cache what each episode's files give, across characters."""
    loaded = loaded if loaded is not None else {}
    exclusions = exclusions if exclusions is not None else {}
    gated: dict[str, Gated] = {}
    candidates = character_candidates(ws, episodes, cid, exclusions, loaded, gated=gated)
    named = {e for e, a in loaded.items() if a is not None and a.names is not None}
    tiers = [candidates]
    acoustic_gated: dict[str, Gated] = {}
    best = character_references(ws, episodes, cid, candidates=candidates, named=named, acoustic={})
    if best is None:  # the acoustic tier is read only when confirmed turns give no picks
        acoustic = {
            **{e: c for e, c in candidates.items() if e not in named},
            **character_candidates(
                ws, [e for e in episodes if e in named], cid, exclusions, loaded,
                acoustic=True, gated=acoustic_gated,
            ),
        }  # fmt: skip
        acoustic_gated = {e: g for e, g in gated.items() if e not in named} | acoustic_gated
        tiers.append(acoustic)
        best = character_references(
            ws, episodes, cid, candidates={}, named=named, acoustic=acoustic
        )
    if best is None:
        seconds = max(
            (sum(p.seconds for p in c) for tier in tiers for c in tier.values()), default=0.0
        )
        return CharacterRefs(seconds=round(seconds, 3), names_used=bool(named))
    e, picks, confirmed = best
    pool = candidates if confirmed else tiers[-1]
    seconds = sum(p.seconds for p in pool.get(e, []))
    arts = loaded.get(e)
    language = (arts.transcript.language if arts else None) or fallback
    vocals = ws.work_dir(e) / "01-separate" / "vocals.wav"
    return CharacterRefs(
        round(seconds, 3), picks, e, vocals, language, confirmed, names_used=bool(named),
        gated=(gated if confirmed else acoustic_gated).get(e),
    )  # fmt: skip


def floor_lines(
    ws: Workspace, reg: Registry, store: CardStore, characters: Iterable[int], language: str
) -> list[str]:
    """One line per automatic clone card under the clone floor (`below_floor`, D32) for
    `voices list` and `names show`: whether the next tts run re-picks it (unlocked, a better
    pick exists, cloning can run here) or it stays flagged for review."""
    short = below_floor(reg, store, characters, language)
    if not short:
        return []
    loaded: dict[str, EpisodeArtifacts | None] = {}
    exclusions: dict[str, list[tuple[float, float]]] = {}
    episodes = _all_episodes(ws)
    blocker: list[str | None] = []
    out: list[str] = []
    for cid in sorted(short):
        sf = short[cid]
        a = reg.card_for(cid, language)
        assert a is not None
        head = (
            f"character {_label(reg, cid)}'s card {sf.card_id} ({a.chosen_by}"
            f"{', locked' if a.locked else ''}) is below the clone floor ({sf.phrase})"
        )
        refusal = replace_refusal(a, "auto")
        if refusal is not None:
            out.append(f"{head}; kept: {refusal}: {FLOOR_FIX}")
            continue
        found = find_references(ws, episodes, cid, None, loaded, exclusions)
        if not _better(found, sf):
            out.append(f"{head}; no better reference pick exists; flagged for review: {FLOOR_FIX}")
            continue
        picks = found.picks or []
        n = len(picks)
        better = f"{n} clip{'' if n == 1 else 's'}, {sum(p.seconds for p in picks):.1f} s"
        if not blocker:
            blocker.append(repick_blocker(ws.config))
        if blocker[0] is None:
            out.append(f"{head}; the next tts run re-picks it ({better})")
        else:
            out.append(f"{head}; a better pick exists ({better}), kept while {blocker[0]}")
    return out
