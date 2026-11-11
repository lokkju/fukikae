"""Automatic voice assignment (D12): a card for each character that has none in the target
language, and episode-local clone cards for extras (A2).

Only characters with no assignment get one, ``by = "auto"``, unlocked; the re-pick reasons
below replace automatic cards only. Should a card appear between the check and the write, the
registry refuses (``ProtectedAssignmentError``, rulings CD2, CD3 and CD9) and the existing
card is kept.

With ``[tts] clone_backend`` set (IndexTTS2 in every profile since the operator's verdict,
D24), each such character gets a clone card of that backend (``cloning``): reference clips
from ``select.select_references``'s rules (CD5) over the character's turns in the
workspace's episodes (``workspace_cloning``), in the episode where they add up to the most
speech. Every speaker gets a clone of its own speech (A2, D41: never a stock voice), in one
of three tiers (``select.tiered_references``): ``floor`` (the D32
floor), ``short`` (at least ``select.CLONE_MIN_S``, 3 s, of gated picks; flagged below the
floor) and ``thin`` (the character's longest turns up to 15 s with the gates set aside,
`character_thin`; flagged by `thin_lines`). Every clone card names its tier in
``[meta] reference_tier``.
Both tiers are searched at `PRE_FLOOR_S` (6 s) first, so a character that had that much
speech keeps the picks it had. An extras cluster gets an episode-local clone from its own
turns, and each cluster-less (``unknown``) line one from its own span (`extras_cards`); they
live in the tts stage's key, not the registry. A character that speaks but holds no turn of
its own in any episode is cloned the same way, thin, from its own cues' source spans
(`cue_references`), and flagged; when an episode it holds a cluster in was skipped because its
assignments are older than its names, the flag names the characters-stage rerun.
No path here makes or assigns a stock card: without a usable clone backend a character
without a card gets none and is flagged (the tts stage refuses first, ``stages.tts.
CloneBackendError``). An automatic stock card from before D41 (``stock_cards``: ``auto``,
``auto-fallback`` or ``restock``, unlocked) is re-picked as a clone, noted `STOCK_NOTE`; a
curated, imported or locked one is kept and flagged on every run.

Subtitle names (D27, D29 #4). In an episode whose names stage used names, a character's
reference turns are its name-confirmed ones (`confirmed_labels`: sub-clusters it got by name,
or by a lock on a key it answers to), never a span of an inner-voice or multi line
(``names.excluded``); names-used episodes are tried first. Only when no episode's
name-confirmed turns yield picks does the character's own acoustic speech count (its unnamed
remainders included, and acoustic-only episodes), before the thin tier. In both, a turn with
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
whose best picks fall short still gets a clone from them (the short or thin tier), and the
clone is flagged for review. An
automatic clone card under the floor (`below_floor`; curated and imported cards are never
judged) is re-picked like a stale one when the current rules give picks that beat it
(`_floor_key`: reaching the floor, then clips up to 2, then seconds); otherwise, or when it is
locked, it is kept and flagged in the tts run's flags, ``fukikae voices list`` and ``fukikae
names show`` (`floor_lines`), naming the shortfall and pointing at ``fukikae voices clone``.
It never falls back to a stock voice or another character's card.

Impure references (A1): `impure_cards` measures every clone card's references on their origin
spans with the verdict reference selection gives (`select.measure_candidates`,
`select.verdict`). An automatic card with a reference the gates refuse is re-picked through
the same path as a stale one (a new card and a new ``auto`` row, noted `PURITY_NOTE`, the old
one kept in history, CD1) when the current rules give picks; with none it is kept and
flagged, never made stock. A curated, imported or locked card is flagged only, in the tts
run's flags, ``fukikae voices list`` and ``fukikae names show`` (`impure_lines`). A card both
stale, under the floor and impure gets one re-pick with every reason in its note.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from fukikae.cards.select import (
    CLONE_MIN_S,
    FLOOR_CLIPS,
    FOREIGN_MAX,
    Gated,
    floor_shortfall,
    meets_floor,
    share_inside,
)
from fukikae.cards.store import CardStore
from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW
from fukikae.registry.store import AUTOMATIC_BY, ProtectedAssignmentError, replace_refusal
from fukikae.transcripts import transcript_path

if TYPE_CHECKING:
    import numpy as np

    from fukikae.backends.base import Synthesizer
    from fukikae.cards.card import Card
    from fukikae.cards.select import RefPick, Register, Tier
    from fukikae.config import Config
    from fukikae.models import Assignments, ClusterAssignment, DubCue, Tags, Transcript
    from fukikae.names.models import NamesResult
    from fukikae.registry.store import CardAssignment, Registry
    from fukikae.workspace import Episode, Workspace

log = logging.getLogger(__name__)

__all__ = [
    "AUTOMATIC_BY",
    "FALLBACK_BY",
    "FLOOR_NOTE",
    "REPICK_NOTE",
    "STOCK_NOTE",
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
    "extras_cards",
    "extras_label",
    "find_references",
    "floor_lines",
    "floor_suffix",
    "impure_cards",
    "impure_lines",
    "foreign_spans",
    "gender_thresholds",
    "key_owner",
    "repick_blocker",
    "replace_refusal",
    "stale_cards",
    "stale_references",
    "stock_cards",
    "thin_lines",
    "workspace_cloning",
]

UNKNOWN_EXTRA = "unknown"
# The stock stand-in made while the clone backend couldn't run, before D41: read in history
# rows and re-picked as a clone like any automatic stock card; never written again.
FALLBACK_BY = "auto-fallback"
# The seconds of picks the name-confirmed and acoustic tiers are searched at first, in that
# order, and the minimum `_episode_key` reads past before a scored non-neutral pick (the rule
# before D32 and D41). Only when neither tier reaches it are both searched again down to
# select.CLONE_MIN_S, so a character with this much speech anywhere keeps the picks it had,
# and the short tier only replaces what used to be a stock voice.
PRE_FLOOR_S = 6.0
THIN_NOTE = "thin reference"
THIN_FIX = "listen in `fukikae studio`, or `fukikae voices clone` with a better clip"
# a character with no turn of its own anywhere: cloned from its own cues' source spans
OWN_LINES_NOTE = "cloned from its own lines' source spans (no turn of its own in any episode)"
OWN_LINES_FLAG = "of its own lines' source spans: no turn of its own in any episode"
STALE_OVERLAP = FOREIGN_MAX  # this share of a reference under someone else's names: stale
REPICK_NOTE = "re-picked: references not name-confirmed"
FLOOR_NOTE = "re-picked: below the clone floor"
FLOOR_FIX = "`fukikae voices clone` with a better clip replaces it"
PURITY_NOTE = "re-picked: references fail the reference gates"
PURITY_FIX = "listen in `fukikae studio`, or replace it with `fukikae voices clone`"
STOCK_NOTE = "re-picked: stock voices are retired (D41)"
STOCK_FIX = (
    "stock voices are retired (D41): accept a clone in `fukikae studio` or use "
    "`fukikae voices clone`"
)
GATE_WORDS = {"off_register": "off-register", "unvoiced": "unvoiced", "song_like": "song-like"}


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
    tier: Tier | None = None  # A2: floor, short or thin (None: no picks)
    # thin picks from the character's own cues' source spans: it holds no turn anywhere
    from_cues: bool = False
    # episodes left out because their assignments are older than their names, in which the
    # character holds a cluster (its turns there wait for the characters stage to rerun)
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cloning:
    """The clone branch of ``ensure_assignments``: the backend clone cards are made for, the
    language the references speak, and each character's references."""

    backend: type[Synthesizer]
    source_language: str
    references: Callable[[int], CharacterRefs]
    unavailable: str | None = None  # why the backend can't run here: no clone is made


@dataclass(frozen=True)
class Shortfall:
    """An automatic clone card under the clone floor (D32 #2): its id and its clips' lengths
    in seconds (from each reference's origin)."""

    card_id: str
    clips: tuple[float, ...]
    thin: bool = False  # a thin-reference clone (A2), under the floor whatever its clips

    @property
    def phrase(self) -> str:
        if self.thin:
            return thin_phrase(self.clips)
        return floor_shortfall(self.clips) or ""


def thin_phrase(clips: Sequence[float]) -> str:
    """``thin reference: <seconds> s from <n> turns`` for a thin clone's clip lengths."""
    n = len(clips)
    return f"{THIN_NOTE}: {sum(clips):.1f} s from {n} turn{'' if n == 1 else 's'}"


def is_thin(card: Card) -> bool:
    """Whether the card is a thin-reference clone (``[meta] reference_tier``, A2)."""
    return card.meta.get("reference_tier") == "thin"


def _floor_key(clips: Sequence[float]) -> tuple[bool, int, float]:
    """How good a set of clips is against the floor: reaching it, then clips up to
    `FLOOR_CLIPS`, then seconds. A re-pick must beat the current card on this."""
    return (
        floor_shortfall(clips) is None,
        min(len(clips), FLOOR_CLIPS),
        round(sum(clips), 2),
    )


def _better(found: CharacterRefs | None, short: Shortfall) -> bool:
    """Whether ``found``'s picks beat the card's clips (`_floor_key`). Thin picks (A2: gates
    set aside) never beat a card, and any gated pick beats a thin card."""
    if found is None or not found.picks or found.tier == "thin":
        return False
    if short.thin:
        return True
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
    clone floor, a thin-reference clone (A2) always counting as under it. Curated and imported
    cards are the operator's choice and never listed."""
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
        if clips is not None and is_thin(card):
            out[cid] = Shortfall(a.card_id, clips, thin=True)
        elif clips is not None and floor_shortfall(clips) is not None:
            out[cid] = Shortfall(a.card_id, clips)
    return out


@dataclass(frozen=True)
class Impurity:
    """A clone card with references the A1 gates refuse, measured on their origin spans: its
    id and each failing reference as ``{episode, start, end, gate, median, voiced_fraction}``
    (``gate`` the first gate that refuses it, ``gates`` all of them)."""

    card_id: str
    refs: tuple[dict, ...]

    @property
    def gates(self) -> str:
        """The gates refusing any reference, as a report phrase (``off-register, unvoiced``)."""
        seen = {g for r in self.refs for g in r["gates"]}
        return ", ".join(GATE_WORDS[g] for g in GATE_WORDS if g in seen)

    @property
    def phrase(self) -> str:
        """Each failing reference: episode, span, gates and median."""
        parts = []
        for r in self.refs:
            med = f", {r['median']:.0f} Hz" if r.get("median") else ""
            words = ", ".join(GATE_WORDS[g] for g in r["gates"])
            parts.append(f"{r['episode']} {r['start']:.2f}-{r['end']:.2f} {words}{med}")
        return "; ".join(parts)


def impure_cards(
    ws: Workspace,
    reg: Registry,
    store: CardStore,
    language: str,
    characters: Iterable[int] | None = None,
) -> dict[int, Impurity]:
    """Character -> its ``language`` clone card's `Impurity`, for the characters (every one
    by default) whose card has a reference failing A1 gate 1, 2 or 3 measured on its origin
    span (the episode's vocals, ``origin = {episode, start, end}``), with the verdict
    selection gives (`select.measure_candidates`, `select.verdict`): the speaker's own turns
    inside the span, its register in that episode (skipped, as selection skips it, for a
    reference inside the character's name-confirmed turns of a names-used episode), the song
    mask and the song detector's accepted spans. The accepted-span part of gate 3 only ever
    refuses a card made before it: reference selection already leaves those spans out of
    every new pick (they are theme exclusions, ``select.reference_exclusions``), so it is here
    that it reads. Gate 4 is about the set, which the floor judges. Cards of any
    ``chosen_by`` are listed (the caller decides by `replace_refusal`:
    a curated, imported or locked card is only flagged); a reference without an episode
    origin (``voices clone``'s ``{"source": "voices clone"}``) or in an episode without
    artifacts is never judged."""
    from fukikae.cards.card import CardError
    from fukikae.cards.select import (
        Register,
        measure_candidates,
        reference_gating,
        song_share,
        song_spans,
        speaker_register,
        verdict,
    )
    from fukikae.stages.mix import mix_settings

    bands = (float(MALE_BELOW), float(FEMALE_ABOVE))
    if characters is None:
        characters = [c.id for c in reg.characters()]
    loaded: dict[str, EpisodeArtifacts | None] = {}
    owner: KeyOwner | None = None
    songs = mix_settings(ws.config).songs
    out: dict[int, Impurity] = {}
    for cid in characters:
        a = reg.card_for(cid, language)
        if a is None or a.card_id is None or not store.has(a.card_id):
            continue
        try:
            card = store.get(a.card_id)
        except CardError:
            continue
        if card.voice.source != "clone" or is_thin(card):
            continue  # a thin clone's references set the gates aside by design (A2)
        by_episode: dict[str, list[tuple[float, float]]] = {}
        for ref in card.voice.references:
            o = ref.origin or {}
            e, start, end = o.get("episode"), o.get("start"), o.get("end")
            if (
                isinstance(e, str)
                and isinstance(start, int | float)
                and isinstance(end, int | float)
            ):
                by_episode.setdefault(e, []).append((float(start), float(end)))
        failing: list[dict] = []
        for e, spans in by_episode.items():
            if e not in loaded:
                loaded[e] = episode_artifacts(ws, e)
            art = loaded[e]
            if art is None or art.skipped is not None or not art.assignments.turns:
                continue
            turns = art.assignments.turns
            own = {
                c.cluster
                for c in art.assignments.clusters
                if c.character_id == cid and c.status not in ("extra", "review")
            }
            own_spans = [(round(t.start, 3), round(t.end, 3)) for t in turns if t.speaker in own]
            confirmed_spans: list[tuple[float, float]] = []
            if art.names is not None:
                if owner is None:
                    owner = key_owner(reg)
                labels = confirmed_labels(art.assignments, cid, owner)
                confirmed_spans = [
                    (round(t.start, 3), round(t.end, 3)) for t in turns if t.speaker in labels
                ]
            work = ws.work_dir(e)
            gkey = (str(work), repr(songs), _stamp(*_exclusion_files(work)))
            if gkey not in _GATING_MEMO:
                _remember(_GATING_MEMO, gkey, reference_gating(work, songs))
            mask = _GATING_MEMO[gkey]
            if gkey not in _SONG_SPANS_MEMO:
                _remember(_SONG_SPANS_MEMO, gkey, song_spans(work, songs))
            accepted = _SONG_SPANS_MEMO[gkey]
            register: Register | None = None
            if own_spans:
                rspans = [(t.start, t.end) for t in turns if t.speaker in own]
                rkey = (str(work), tuple(rspans), bands, _stamp(art.vocals))
                if rkey not in _REGISTER_MEMO:
                    _remember(
                        _REGISTER_MEMO, rkey, speaker_register(art.vocals, rspans, bands, ws.root)
                    )
                register = _REGISTER_MEMO[rkey]
            for start, end in spans:
                vouched = (
                    bool(confirmed_spans)
                    and share_inside(start, end, confirmed_spans)
                    >= share_inside(start, end, own_spans) - 1e-6
                    > 0
                )
                pieces = confirmed_spans if vouched else own_spans
                (m,) = measure_candidates(art.vocals, [(start, end)], pieces, ws.root)
                reg_here = register
                if vouched and register is not None:
                    reg_here = Register("u", register.median)
                share = None if mask is None else song_share(start, end, mask)
                gates = verdict(
                    m, register=reg_here or Register("u", None), song=share, bands=bands,
                    in_song=share_inside(start, end, accepted) if accepted else None,
                )  # fmt: skip
                if gates:
                    failing.append({
                        "episode": e, "start": start, "end": end, "gate": gates[0],
                        "gates": list(gates),
                        "median": None if m.median is None else round(m.median, 1),
                        "voiced_fraction": m.voiced_fraction,
                    })  # fmt: skip
        if failing:
            out[cid] = Impurity(a.card_id, tuple(failing))
    return out


def _impure_flag(reg: Registry, cid: int, card: str, imp: Impurity, why: str) -> str:
    return (
        f"character {_label(reg, cid)}: clone card {card} has references that fail the "
        f"reference gates ({imp.phrase}); {why}: {PURITY_FIX}"
    )


def floor_suffix(short: Mapping[int, Shortfall] | None, cid: int) -> str:
    """What a kept stale card's line appends when the card is under the clone floor too (its
    floor line is not printed separately), a thin-reference clone named as one; empty
    otherwise."""
    sf = (short or {}).get(cid)
    if sf is None:
        return ""
    return _shortfall_suffix(None if sf.thin else sf.phrase, sf.phrase if sf.thin else None)


def _shortfall_suffix(below: str | None, thin: str | None) -> str:
    """``; below the clone floor (<below>)`` or ``; a thin-reference clone (<seconds> s from
    <n> turns)`` (``thin``: a `thin_phrase`), or empty with neither."""
    if thin is not None:
        return f"; a thin-reference clone ({thin.removeprefix(f'{THIN_NOTE}: ')})"
    return f"; below the clone floor ({below})" if below is not None else ""


def _thin_flag(
    reg: Registry,
    cid: int,
    card: str,
    phrase: str,
    why: str | None = None,
    *,
    own: CharacterRefs | None = None,
) -> str:
    """The flag for a thin-reference clone (A2): listen to it in the studio. With ``own``
    picks from the character's own lines (``from_cues``), says so, and names each episode
    skipped for assignments older than its names, with the rerun that brings its turns back."""
    seconds = phrase.removeprefix(f"{THIN_NOTE}: ")
    head = f"character {_label(reg, cid)}: card {card} is a thin-reference clone ({seconds})"
    if own is not None and own.from_cues:
        head += f" {OWN_LINES_FLAG}{_skip_hint(own.skipped)}"
    return f"{head}; {why}: {THIN_FIX}" if why else f"{head}; {THIN_FIX}"


def _skip_hint(skipped: Sequence[str]) -> str:
    """For each episode skipped because its assignments are older than its names: that it was
    skipped and the characters-stage rerun that brings its turns back."""
    return "".join(
        f"; {e} was skipped (its assignments are older than its names): rerun its characters "
        f"stage, `fukikae run <ws> {e} --from characters`"
        for e in skipped
    )


def _short_flag(
    reg: Registry,
    cid: int,
    sf: Shortfall,
    why: str | None = None,
    *,
    own: CharacterRefs | None = None,
) -> str:
    """The flag for a card under the floor that stays: thin or below the floor (``own``: the
    character's current picks, for `_thin_flag`)."""
    if sf.thin:
        return _thin_flag(reg, cid, sf.card_id, sf.phrase, why, own=own)
    return _floor_flag(reg, cid, sf.card_id, sf.phrase, why)


def _floor_flag(reg: Registry, cid: int, card: str, phrase: str, why: str | None = None) -> str:
    """The flag for a clone card under the floor that stays: no better pick, or ``why``."""
    head = f"character {_label(reg, cid)}: clone card {card} is below the clone floor ({phrase})"
    if why is None:
        return f"{head} and no better reference pick exists; kept, flagged for review: {FLOOR_FIX}"
    return f"{head}; {why}: {FLOOR_FIX}"


def clone_backend(cfg: Config) -> str | None:
    """``[tts] clone_backend``, the workspace's over the profile's; None when unset or empty
    (no clone can be made, and the tts stage refuses to make or re-pick a card, D41)."""
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
    *,
    cloning: Cloning | None = None,
    stale: Mapping[int, Sequence[dict]] | None = None,
    short: Mapping[int, Shortfall] | None = None,
    impure: Mapping[int, Impurity] | None = None,
    stock: Mapping[int, str] | None = None,
    flags: list[str] | None = None,
) -> list[str]:
    """Give every character in ``character_spans`` without a ``cfg.target_language`` card
    one: a clone card from its own speech in one of the three tiers (A2), most speech first,
    while ``cloning`` is usable. Never a stock voice (D41): without a usable clone backend a
    character without a card gets none and is flagged (the tts stage refuses before it gets
    here). Returns one report line per card assigned: the lines that changed the registry.

    ``stale`` (`stale_cards`) lists the characters whose clone card has references the
    subtitle names give someone else. While ``cloning`` is usable, an automatic card
    (``replace_refusal`` lets ``auto`` replace it) is re-picked the way a new card is made, a
    new card with a new ``auto`` row noted `REPICK_NOTE`: a clone from name-confirmed
    references when there are enough, else from the character's own acoustic speech, else
    the thin tier, so a character whose speech the names gave to others stops speaking in
    their voice. A curated or imported card, and an automatic one while no clone can be made, is
    only flagged (D29 #4): the flag lines go to ``flags`` (when given), never to the returned
    lines, because they change nothing and would repeat on every run.

    ``short`` (`below_floor`) lists automatic clone cards under the clone floor (D32). An
    unlocked one is re-picked the same way (noted `FLOOR_NOTE`) when ``cloning`` gives picks
    that beat it (`_floor_key`); otherwise, or when locked, it is kept and flagged, never
    replaced by a stock voice. A new clone card under the floor is made, from the best picks
    there are, and flagged too.

    ``impure`` (`impure_cards`) lists clone cards with references the A1 gates refuse. An
    automatic one is re-picked (noted `PURITY_NOTE` with the gates) when ``cloning`` gives
    picks, which pass every gate by construction; with none it is kept and flagged, never
    made stock. A curated, imported or locked one is flagged with the failing references,
    pointing at the studio and ``fukikae voices clone``. A card that is stale, under the
    floor and impure at once gets one re-pick whose note gives every reason.

    ``stock`` (`stock_cards`) lists the characters that speak with a stock card. An automatic
    one (``auto``, ``auto-fallback`` or ``restock``, unlocked) is re-picked as a clone, noted
    `STOCK_NOTE`, whatever tier its picks reach; with no speech of its own to clone from it
    is kept and flagged. A curated, imported or locked one is kept (CD2, CD3, CD9) and flagged
    on every run, pointing at the studio and ``fukikae voices clone``."""
    language = cfg.target_language
    usable = cloning is not None and cloning.unavailable is None
    todo = []
    restock: dict[int, str] = {}  # character -> its automatic stock card, re-picked (D41)
    repick: dict[int, str] = {}  # character -> its current card id
    floor: dict[int, Shortfall] = {}  # characters re-picked for the clone floor
    purity: dict[int, Impurity] = {}  # characters re-picked for impure references
    lines: list[str] = []
    flagged = flags if flags is not None else []
    for cid, spans in character_spans.items():
        a = reg.card_for(cid, language)
        refs = (stale or {}).get(cid)
        imp = (impure or {}).get(cid)
        if imp is not None and a is not None and a.card_id != imp.card_id:
            imp = None  # measured on another card
        if imp is not None and a is not None and (refs or cid in (short or {})):
            if a.chosen_by in AUTOMATIC_BY and replace_refusal(a, "auto") is None and usable:
                purity[cid] = imp  # one re-pick with the stale or floor reason
            else:
                why = _keep_why(a, cloning)
                flagged.append(_impure_flag(reg, cid, a.card_id or "", imp, why))
        held_stock = (stock or {}).get(cid)
        if held_stock is not None and (a is None or a.card_id != held_stock):
            held_stock = None  # read for another card
        if a is None or a.card_id is None:
            if usable:
                todo.append((cid, spans))
            else:
                flagged.append(
                    f"character {_label(reg, cid)} has no card, and {_no_clone(cloning)}; "
                    "fukikae makes no stock voices (D41)"
                )
        elif held_stock is not None:
            if a.chosen_by not in AUTOMATIC_BY or replace_refusal(a, "auto") is not None:
                flagged.append(_stock_flag(reg, cid, a))
            elif usable:
                todo.append((cid, spans))
                restock[cid] = held_stock
            else:
                flagged.append(_stock_flag(reg, cid, a, f"kept while {_no_clone(cloning)}"))
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
                flagged.append(_short_flag(reg, cid, sf, f"kept: {refusal}"))
            elif usable:
                todo.append((cid, spans))
                floor[cid] = sf
            elif cloning is None:
                flagged.append(_short_flag(
                    reg, cid, sf,
                    "kept, because no [tts] clone_backend is set to re-pick it with",
                ))  # fmt: skip
            else:
                name = cloning.backend.name
                flagged.append(_short_flag(
                    reg, cid, sf,
                    f"kept until the clone backend {name} can run ({cloning.unavailable}; "
                    f"`fukikae backends setup {name}`), when the next tts run re-picks it if "
                    "a better pick exists",
                ))  # fmt: skip
        elif imp is not None:
            if a.chosen_by in AUTOMATIC_BY and replace_refusal(a, "auto") is None and usable:
                todo.append((cid, spans))
                purity[cid] = imp
            else:
                flagged.append(_impure_flag(reg, cid, a.card_id, imp, _keep_why(a, cloning)))
    if not todo:
        return lines
    assert cloning is not None
    todo.sort(key=lambda item: (-sum(e - s for s, e in item[1]), item[0]))
    for cid, _spans in todo:
        found = cloning.references(cid)
        if cid in restock and not found.picks:
            a = reg.card_for(cid, language)
            assert a is not None
            hint = _skip_hint(found.skipped)
            flagged.append(_stock_flag(
                reg, cid, a, f"no speech of its own was found to clone a voice from{hint}",
            ))  # fmt: skip
            continue
        gated_picks = found is not None and bool(found.picks) and found.tier != "thin"
        if cid in purity and cid not in repick and not gated_picks:
            imp = purity[cid]
            why = "no gate-passing reference pick exists; kept, flagged for review"
            flagged.append(_impure_flag(reg, cid, imp.card_id, imp, why))
            if cid in floor:
                flagged.append(_short_flag(reg, cid, floor[cid]))
            continue  # never a stock voice
        if cid in floor and cid not in purity and not _better(found, floor[cid]):
            flagged.append(_short_flag(reg, cid, floor[cid], own=found))
            continue  # the best picks there are: kept, flagged (never a stock voice)
        below: str | None = None
        thin: str | None = None
        own_lines = found is not None and found.from_cues
        if found.picks:
            new, note = _clone(cloning, found, store, language)
            if found.tier == "thin":
                thin = thin_phrase([p.seconds for p in found.picks])
                note += f"; {thin}"
                if own_lines:
                    note += f"; {OWN_LINES_NOTE}"
            else:
                below = floor_shortfall([p.seconds for p in found.picks])
                if below is not None:
                    note += f"; below the clone floor ({below})"
        else:
            # D41: never a stock voice. A character with no speech of its own anywhere, its
            # cues included, has nothing to clone: a card it has stays, flagged, and one
            # without a card is named (the tts stage then stops on its cues)
            cur = reg.card_for(cid, language)
            hint = _skip_hint(found.skipped)
            if cur is not None and cur.card_id is not None:
                flagged.append(
                    f"character {_label(reg, cid)}: card {cur.card_id} is kept, flagged: no "
                    f"speech of its own was found to re-pick it from{hint}; `fukikae voices "
                    "clone` replaces it"
                )
            else:
                flagged.append(
                    f"character {_label(reg, cid)}: no speech of its own was found to clone a "
                    f"voice from, so it has no card{hint}; `fukikae voices clone` makes one"
                )
            continue
        again = "the current rules pick it again; kept, flagged for review"
        held = reg.card_for(cid, language)
        if (cid in repick or cid in floor or cid in purity) and held and new == held.card_id:
            # the re-pick is the card it has: nothing changes, and every flag stays, a stale
            # card's shortfall on its own line (one line per card)
            if cid in repick:
                flagged.append(
                    f"character {_label(reg, cid)}: card {new} is cloned from turns the "
                    f"subtitles now name otherwise ({_where((stale or {})[cid])}); {again}: "
                    f"`fukikae voices clone` replaces it{_shortfall_suffix(below, thin)}"
                )
            if cid in purity:
                flagged.append(_impure_flag(reg, cid, new, purity[cid], again))
            if cid not in repick and thin is not None:
                flagged.append(_thin_flag(reg, cid, new, thin, own=found))
            elif cid not in repick and below is not None:
                flagged.append(_floor_flag(reg, cid, new, below, again))
            continue
        reasons: list[str] = [STOCK_NOTE] if cid in restock else []
        if cid in repick:
            reasons.append(f"{REPICK_NOTE} ({_where((stale or {})[cid])})")
        if cid in floor or (cid in repick and cid in (short or {})):
            sf = floor.get(cid) or (short or {})[cid]
            reasons.append(f"{FLOOR_NOTE} ({sf.phrase})")
        if cid in purity:
            reasons.append(f"{PURITY_NOTE} ({purity[cid].gates})")
        if reasons:
            note = f"{'; '.join(reasons)}; {note}"
        try:
            reg.assign_card(cid, language, new, by="auto", locked=False, note=note)
        except ProtectedAssignmentError as e:
            lines.append(f"character {_label(reg, cid)}: kept its card ({e})")
            continue
        lines.append(f"character {_label(reg, cid)}: {note}")
        if below is not None:
            flagged.append(_floor_flag(reg, cid, new, below))
        if thin is not None:
            flagged.append(_thin_flag(reg, cid, new, thin, own=found))
    return lines


def _no_clone(cloning: Cloning | None) -> str:
    """Why no clone can be made here: no ``[tts] clone_backend``, or one that can't run."""
    if cloning is None:
        return "no [tts] clone_backend is set to clone one with"
    name = cloning.backend.name
    return (
        f"the clone backend {name} can't run here ({cloning.unavailable}; `fukikae backends "
        f"setup {name}`)"
    )


def _stock_flag(reg: Registry, cid: int, a: CardAssignment, why: str | None = None) -> str:
    """The flag for a stock card that stays (D41): curated, imported or locked, or ``why``."""
    state = f"{a.chosen_by}{', locked' if a.locked else ''}"
    head = f"character {_label(reg, cid)} speaks with a stock card ({state})"
    return f"{head}; {f'{why}; ' if why else ''}{STOCK_FIX}"


def stock_cards(
    reg: Registry, store: CardStore, ids: Iterable[int], language: str
) -> dict[int, str]:
    """Character -> its ``language`` card's id, for the characters whose card is a stock
    voice (``voice.source == "stock"``), whoever chose it; a card that can't be read is
    skipped (the tts stage's checks name it)."""
    from fukikae.cards.card import CardError

    out: dict[int, str] = {}
    for cid in ids:
        a = reg.card_for(cid, language)
        if a is None or a.card_id is None or not store.has(a.card_id):
            continue
        try:
            card = store.get(a.card_id)
        except CardError:
            continue
        if card.voice.source == "stock":
            out[cid] = a.card_id
    return out


def _keep_why(a: CardAssignment, cloning: Cloning | None) -> str:
    """Why an impure card stays: the operator's choice or a lock, or no clone to re-pick
    with yet."""
    refusal = replace_refusal(a, "auto")
    if a.chosen_by not in AUTOMATIC_BY or refusal is not None:
        state = f"{a.chosen_by}{', locked' if a.locked else ''}"
        return f"kept ({state}{f': {refusal}' if refusal else ''})"
    if cloning is None:
        return "kept, because no [tts] clone_backend is set to re-pick it with"
    name = cloning.backend.name
    return (
        f"kept until the clone backend {name} can run ({cloning.unavailable}; "
        f"`fukikae backends setup {name}`), when the next tts run re-picks it"
    )


def _clone(
    cloning: Cloning, found: CharacterRefs, store: CardStore, language: str
) -> tuple[str, str]:
    """(card id, note) of a clone card from ``found``'s picks, put in the store."""
    from fukikae.cards.select import clone_card, reference_report

    picks, episode, vocals = found.picks or [], found.episode, found.vocals
    assert episode is not None and vocals is not None
    card = clone_card(
        cloning.backend, picks, store, vocals, found.language or cloning.source_language,
        language, episode, found.tier,
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


def extras_label(cue: DubCue) -> str:
    """The label an extras cue's card is keyed by: its cluster, or for a cue with no cluster
    (``unknown``) ``unknown~<cue id>``, since each such line is cloned from itself (A2)."""
    label = cue.extra or UNKNOWN_EXTRA
    return f"{UNKNOWN_EXTRA}~{cue.id}" if label == UNKNOWN_EXTRA else label


@dataclass(frozen=True)
class ExtraCard:
    """An extras label's episode-local clone card (A2): its id, tier and reference clips, and
    for a cluster-less line whose span is no speech, what it is (``silence``, ``music`` or
    ``unvoiced``; `span_issue`)."""

    card_id: str
    tier: Tier
    clips: tuple[float, ...]
    issue: str | None = None

    @property
    def seconds(self) -> float:
        return round(sum(self.clips), 3)


def _spoken(c: DubCue) -> bool:
    return c.skip is None and c.error is None and bool(c.text and c.text.strip())


def extras_cards(
    ep: Episode,
    cues: Iterable[DubCue],
    cfg: Config,
    store: CardStore,
    cloning: Cloning,
    *,
    cards: dict[str, ExtraCard] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """(extras label -> clone card id, flag lines) for the episode's spoken cues with no
    character (A2, D41: never a stock voice). An extras cluster's card is cloned from its own
    turns in ``assignments.json`` through ``select.tiered_references`` (its own majority
    register, the song mask and accepted song spans, theme songs and tagged singing left
    out); a cluster with no turn there is cloned from its cues' own spans. Each cue with no
    cluster (``unknown``) gets ``unknown~<cue id>`` and a card cloned from its own source span
    (``delivery.reference``), always thin. Cards are made for ``cloning``'s backend and put in
    the store; they get no registry row, and the same turns give the same card ids on every
    run. Flags: one line per short or thin extras card, one per episode counting the
    cluster-less lines, and one per cluster-less line whose span is silence, music (the song
    mask or an accepted song span over half of it) or unvoiced (A0's loud-frame voiced
    fraction under ``REF_VOICED_MIN``): it is cloned anyway (D41) and surfaced (`span_issue`).
    ``cards`` receives each label's `ExtraCard`. The result is computed once per process for
    the same cues' spans, backend, settings and files (the stage key and the run share it),
    while every card it names is still in the store."""
    from fukikae.artifacts import read_json
    from fukikae.cards.select import (
        clone_card,
        reference_gating,
        song_spans,
        span_reference,
        speaker_register,
        tiered_references,
    )
    from fukikae.models import Assignments, Tags, Transcript, Turn
    from fukikae.stages.mix import mix_settings
    from fukikae.workspace import Workspace

    spoken = [c for c in cues if c.character_id is None and _spoken(c)]
    if not spoken:
        return {}, []
    work = ep.work
    songs = mix_settings(cfg).songs
    memo_key = (
        str(work), str(store.root), cloning.backend.name,
        getattr(cloning.backend, "model_revision", None), cloning.source_language,
        cfg.target_language, cfg.source_language, repr(songs),
        tuple((extras_label(c), tuple(c.delivery.reference), c.source_text) for c in spoken),
        _stamp(*_candidate_files(work), *_exclusion_files(work)),
    )  # fmt: skip
    hit = _EXTRAS_MEMO.get(memo_key)
    if hit is not None and all(store.has(cid) for cid in hit[0].values()):
        if cards is not None:
            cards.update(hit[2])
        return dict(hit[0]), list(hit[1])
    vocals = work / "01-separate" / "vocals.wav"
    asg_path, tr_path = (
        work / "05-characters" / "assignments.json",
        transcript_path(work),
    )
    tags_path = work / "04b-tag" / "tags.json"
    turns = list(read_json(asg_path, Assignments).turns or []) if asg_path.exists() else []
    transcript = (
        read_json(tr_path, Transcript)
        if tr_path.exists()
        else Transcript(
            language=cloning.source_language or cfg.source_language or "",
            backend="none",
            word_timing=False,
            segments=[],
        )
    )
    tags = read_json(tags_path, Tags) if tags_path.exists() else None
    language = transcript.language or cloning.source_language or cfg.source_language or ""
    ws = Workspace(ep.workspace_root)
    theme = _episode_exclusions(ws, ep.id, tags, {})
    gkey = (str(work), repr(songs), _stamp(*_exclusion_files(work)))
    if gkey not in _GATING_MEMO:
        _remember(_GATING_MEMO, gkey, reference_gating(work, songs))
    if gkey not in _SONG_SPANS_MEMO:
        _remember(_SONG_SPANS_MEMO, gkey, song_spans(work, songs))
    mask, accepted = _GATING_MEMO[gkey], _SONG_SPANS_MEMO[gkey]
    bands = (float(MALE_BELOW), float(FEMALE_ABOVE))
    target = cfg.target_language

    out: dict[str, str] = {}
    made: dict[str, ExtraCard] = {}
    flags: list[str] = []
    unknown: list[tuple[str, DubCue]] = []

    def put(label: str, tier: Tier, picks: list[RefPick]) -> None:
        card = clone_card(cloning.backend, picks, store, vocals, language, target, ep.id, tier)
        out[label] = store.put(card)
        made[label] = ExtraCard(out[label], tier, tuple(round(p.seconds, 3) for p in picks))

    for c in spoken:
        label = extras_label(c)
        if label in out:
            continue
        if label.startswith(f"{UNKNOWN_EXTRA}~"):
            a, b = (float(x) for x in c.delivery.reference)
            put(label, "thin", [span_reference(vocals, transcript, a, b, c.source_text)])
            unknown.append((label, c))
            continue
        mine = [t for t in turns if t.speaker == label]
        if mine:
            pool = turns
        else:  # a cluster with no turn here: its cues' own spans are its speech
            mine = [
                Turn(start=float(x.delivery.reference[0]), end=float(x.delivery.reference[1]),
                     speaker=label)
                for x in spoken if x.extra == label
            ]  # fmt: skip
            pool = [*turns, *mine]
        spans = [(t.start, t.end) for t in mine]
        rkey = (str(work), tuple(spans), bands, _stamp(vocals))
        if rkey not in _REGISTER_MEMO:
            _remember(_REGISTER_MEMO, rkey, speaker_register(vocals, spans, bands, ws.root))
        tier, picks = tiered_references(
            pool, label, transcript, tags, vocals, register=_REGISTER_MEMO[rkey],
            song_like=mask, song_spans=accepted, theme_spans=theme, bands=bands,
            pitch_root=ws.root,
        )  # fmt: skip
        put(label, tier, picks)
        ec = made[label]
        if tier == "thin":
            flags.append(
                f"extra {label}: card {ec.card_id} is a thin-reference clone "
                f"({thin_phrase(ec.clips).removeprefix(f'{THIN_NOTE}: ')}); {THIN_FIX}"
            )
        elif tier == "short":
            flags.append(
                f"extra {label}: card {ec.card_id} is below the clone floor "
                f"({floor_shortfall(ec.clips)}); {THIN_FIX}"
            )
    if unknown:
        n = len(unknown)
        flags.append(
            f"{ep.id}: {n} unknown line{'' if n == 1 else 's'} cloned from "
            "themselves (no cluster: each line's own source span is its voice)"
        )
        spans = [tuple(float(x) for x in c.delivery.reference) for _, c in unknown]
        issues = span_issues(vocals, spans, mask, accepted, ws.root)  # type: ignore[arg-type]
        for (label, c), (a, b), issue in zip(unknown, spans, issues, strict=True):
            if issue is None:
                continue
            made[label] = replace(made[label], issue=issue)
            flags.append(
                f"{ep.id}: unknown line {c.id} ({a:.2f}-{b:.2f}) is {issue}, not speech: "
                f"cloned from itself anyway (D41), so its voice may be no one's; {THIN_FIX}"
            )
    _remember(_EXTRAS_MEMO, memo_key, (dict(out), list(flags), dict(made)))
    if cards is not None:
        cards.update(made)
    return out, flags


# A span quieter than this RMS (about -70 dBFS) is silence, whatever its voicing reads.
SILENCE_RMS = 3e-4


def span_issues(
    vocals: Path,
    spans: Sequence[tuple[float, float]],
    song_like: np.ndarray | None,
    accepted: Sequence[tuple[float, float]],
    pitch_root: Path | None = None,
) -> list[str | None]:
    """What each span is when it is not speech, for a cluster-less line's own clone (review
    fix 6): ``silence`` (RMS under `SILENCE_RMS`), ``music`` (``REF_SONG_SHARE`` or more of it
    song-like by the song detector's mask, or inside its accepted song spans), ``unvoiced``
    (A0's loud-frame voiced fraction under ``REF_VOICED_MIN``), else None."""
    import numpy as np
    from scipy.io import wavfile

    from fukikae.cards.select import (
        _samples,
        measure_candidates,
        song_share,
    )
    from fukikae.pitch import REF_SONG_SHARE, REF_VOICED_MIN

    if not spans:
        return []
    rate, data = wavfile.read(vocals, mmap=True)
    measured = measure_candidates(vocals, spans, None, pitch_root)
    out: list[str | None] = []
    for (a, b), m in zip(spans, measured, strict=True):
        x = _samples(data, int(rate), a, b).astype(np.float64)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        music = (song_like is not None and song_share(a, b, song_like) >= REF_SONG_SHARE) or (
            bool(accepted) and share_inside(a, b, accepted) >= REF_SONG_SHARE
        )
        if rms < SILENCE_RMS:
            out.append("silence")
        elif music:
            out.append("music")
        elif m.voiced_fraction < REF_VOICED_MIN:
            out.append("unvoiced")
        else:
            out.append(None)
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
    tr_path = transcript_path(work)
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
    the song detector's per-second mask (``select.reference_gating``) with each candidate's
    sung-ness, and its accepted song spans (``select.song_spans``), and the pitch tracks go
    through the workspace's pitch cache. ``gated`` receives each episode's `Gated` counts
    (``split`` when the register is bimodal, ``even`` when no register gate ran for it).

    In an episode whose names were used (``owner`` answers keys, the workspace's registry by
    default), the candidates are the character's name-confirmed turns (`confirmed_labels`),
    or with ``acoustic`` all of its turns, unnamed remainders included; either way never a
    span of the names stage's inner-voice and multi lines, nor a turn with `STALE_OVERLAP` of
    it under someone else's names (`foreign_spans`). A names-used episode whose assignments
    are older than its names (``skipped``) offers nothing."""
    from fukikae.cards.select import (
        Register,
        eligible_references,
        reference_gating,
        song_spans,
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
        exclude = _episode_exclusions(ws, e, a.tags, cache)
        if a.names is not None:
            exclude = [*exclude, *((float(x), float(y)) for x, y in a.names.excluded)]
        songs = mix_settings(ws.config).songs
        gkey = (str(work), repr(songs), _stamp(*_exclusion_files(work)))
        if gkey not in _GATING_MEMO:
            _remember(_GATING_MEMO, gkey, reference_gating(work, songs))
        song_like = _GATING_MEMO[gkey]
        if gkey not in _SONG_SPANS_MEMO:
            _remember(_SONG_SPANS_MEMO, gkey, song_spans(work, songs))
        accepted = _SONG_SPANS_MEMO[gkey]
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
                pitch_root=ws.root, gated=counts, song_spans=accepted,
            )  # fmt: skip
            _remember(_CANDIDATES_MEMO, key, (picks, counts[0] if counts else Gated()))
        picks, found = _CANDIDATES_MEMO[key]
        out[e] = list(picks)
        if gated is not None:
            gated[e] = found
    return out


def _episode_exclusions(
    ws: Workspace, episode: str, tags: Tags | None, cache: dict[str, list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """The episode's reference exclusions (``select.reference_exclusions``: theme songs and
    tagged singing), read once per process while its files are unchanged and kept in
    ``cache`` per episode."""
    from fukikae.cards.select import reference_exclusions
    from fukikae.stages.mix import mix_settings

    if episode not in cache:
        work = ws.work_dir(episode)
        songs = mix_settings(ws.config).songs
        key = (str(work), repr(songs), _stamp(*_exclusion_files(work)))
        if key not in _EXCLUSIONS_MEMO:
            _remember(_EXCLUSIONS_MEMO, key, reference_exclusions(work, tags, songs))
        cache[episode] = list(_EXCLUSIONS_MEMO[key])
    return cache[episode]


def character_thin(
    ws: Workspace,
    episodes: Sequence[str],
    cid: int,
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
    loaded: dict[str, EpisodeArtifacts | None] | None = None,
    owner: KeyOwner | None = None,
) -> tuple[str, list[RefPick]] | None:
    """The thin tier for a character (A2): `select.thin_references` over its own turns in
    each episode (every cluster it holds, unnamed remainders included), leaving out theme
    songs and tagged singing (`_episode_exclusions`) and turns under someone else's names
    (`foreign_spans`), in the episode where they add up to the most seconds (the first such,
    in ``episodes`` order); only when no episode has any such turn, the same from its own
    turns regardless. None for a character with no turn in any episode."""
    from fukikae.cards.select import thin_references

    cache = exclusions if exclusions is not None else {}
    arts = loaded if loaded is not None else {}
    found: list[tuple[str, list[RefPick], list[RefPick]]] = []
    for e in episodes:
        if e not in arts:
            arts[e] = episode_artifacts(ws, e)
        a = arts[e]
        if a is None or a.skipped is not None or not a.assignments.turns:
            continue
        own = {
            c.cluster
            for c in a.assignments.clusters
            if c.character_id == cid and c.status not in ("extra", "review")
        }
        if not own:
            continue
        foreign: Spans = []
        if a.names is not None:
            if owner is None:
                owner = _workspace_owner(ws)
            foreign = foreign_spans(a.assignments, a.names, cid, owner)
        theme = _episode_exclusions(ws, e, a.tags, cache)
        turns, tr, vocals = a.assignments.turns, a.transcript, a.vocals
        strict = thin_references(
            turns, own, tr, vocals, theme_spans=theme, foreign=foreign, fallback=False
        )
        loose = strict or thin_references(turns, own, tr, vocals)
        found.append((e, strict, loose))
    for which in (1, 2):
        best: tuple[str, list[RefPick]] | None = None
        for e, *tiers in found:
            picks = tiers[which - 1]
            secs = round(sum(p.seconds for p in picks), 3)
            if picks and (best is None or secs > round(sum(p.seconds for p in best[1]), 3)):
                best = (e, picks)
        if best is not None:
            return best
    return None


# Per process (a `fukikae run` of several episodes, or one CLI command): each episode's
# reference candidates and song exclusions, keyed by everything they're made from (the
# registry's part already resolved into speakers and foreign spans) and the files' mtimes
# and sizes, so every tts run of the batch and every character asks the files once (D32
# review: a flagged character with no better pick is looked up again on each run).
_CANDIDATES_MEMO: dict[tuple, tuple[list[RefPick], Gated]] = {}
_EXCLUSIONS_MEMO: dict[tuple, list[tuple[float, float]]] = {}
# each episode's gate 3 evidence (the song mask) and each speaker's register (A1)
_GATING_MEMO: dict[tuple, np.ndarray | None] = {}
_SONG_SPANS_MEMO: dict[tuple, list[tuple[float, float]]] = {}
_REGISTER_MEMO: dict[tuple, Register] = {}
# each episode's extras cards (extras_cards): the stage key and the run ask once
_EXTRAS_MEMO: dict[tuple, tuple[dict[str, str], list[str], dict[str, ExtraCard]]] = {}
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
        transcript_path(work),
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
    episodes: Sequence[str],
    found: Mapping[str, list[RefPick]],
    allowed: Collection[str] | None,
    min_total: float = PRE_FLOOR_S,
) -> tuple[str, list[RefPick]] | None:
    from fukikae.cards.select import pick_references

    best: tuple[str, list[RefPick]] | None = None
    for e in episodes:
        if allowed is not None and e not in allowed:
            continue
        picks = pick_references(found[e], min_total=min_total) if e in found else None
        if picks and (best is None or _episode_key(picks) > _episode_key(best[1])):
            best = (e, picks)
    return best


def _episode_key(picks: Sequence[RefPick]) -> tuple[bool, int, float]:
    """An episode's picks rank by reaching the clone floor (D32). Among episodes that reach
    it, by the seconds picked before any floor filling: the picks up to the first scored
    non-neutral one taken past `PRE_FLOOR_S`, which are the picks the rule before D32 made, so
    filling an episode with non-neutral turns never moves a character that already reached
    the floor to it. Below the floor, by how close the picks come (`_floor_key`: clips up to
    2, then seconds)."""
    if not meets_floor(picks):
        _, clips, seconds = _floor_key([p.seconds for p in picks])
        return False, clips, seconds
    used = 0.0
    for p in picks:
        if p.emotion is not None and used >= PRE_FLOOR_S:
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
    min_total: float = PRE_FLOOR_S,
) -> tuple[str, list[RefPick], bool] | None:
    """The character's reference clips (``pick_references``) in the episode where they add
    up to the most speech, with that episode and whether the picks are name-confirmed; None
    when no episode has enough. The name-confirmed candidates (``candidates``, by default
    `character_candidates`) of episodes whose names were used (``named``, read from each
    episode's ``names.json`` by default) are tried first; only when none yields picks, the
    character's own acoustic speech in every episode (``acoustic``, by default
    `character_candidates` with ``acoustic``). Picks under ``min_total`` seconds don't
    count (`PRE_FLOOR_S`; `find_references` asks again down to ``select.CLONE_MIN_S``)."""
    loaded = loaded if loaded is not None else {}
    found = (
        candidates
        if candidates is not None
        else character_candidates(ws, episodes, cid, exclusions, loaded, owner)
    )
    if named is None:
        named = {e for e in episodes if used_names(ws.work_dir(e)) is not None}
    best = _best(episodes, found, named, min_total)
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
    best = _best(episodes, acoustic, None, min_total)
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


def workspace_cloning(
    root: Path, cfg: Config, current: str, *, cues: Sequence[DubCue] | None = None
) -> Cloning | None:
    """The clone branch for an episode of the workspace at ``root``, or None when ``[tts]
    clone_backend`` is unset: references from the character's turns in every episode the
    workspace has processed so far (the current one first), each card's source language the
    transcript language of the episode its clips come from. A character that holds no turn in
    any episode is cloned from its own spoken ``cues``' source spans (``delivery.reference``)
    in the current episode, thin (`cue_references`), the way a cluster-less line is. When the
    backend can't run here (``clone_problem``), the result says why and reads no
    references."""
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
        found = find_references(ws, episodes, cid, fallback, loaded, exclusions)
        if found.picks or not cues:
            return found
        picks = cue_references(ws, current, cid, cues, exclusions)
        if not picks:
            return found
        language = (here.transcript.language if here else None) or fallback
        return replace(
            found, seconds=round(sum(p.seconds for p in picks), 3), picks=picks,
            episode=current, vocals=ws.work_dir(current) / "01-separate" / "vocals.wav",
            language=language, confirmed=False, tier="thin", from_cues=True,
        )  # fmt: skip

    return Cloning(backend=cls, source_language=fallback, references=references)


def cue_references(
    ws: Workspace,
    episode: str,
    cid: int,
    cues: Iterable[DubCue],
    exclusions: dict[str, list[tuple[float, float]]] | None = None,
) -> list[RefPick]:
    """The thin tier from a character's own spoken cues in ``episode`` (review fix 1): their
    source spans (``delivery.reference``) as turns, through `select.thin_references` (longest
    first up to ``THIN_MAX_S``, theme songs and tagged singing left out unless nothing else
    is left). For a character that holds no turn of its own in any episode."""
    from fukikae.artifacts import read_json
    from fukikae.cards.select import thin_references
    from fukikae.models import Tags, Transcript, Turn

    work = ws.work_dir(episode)
    vocals = work / "01-separate" / "vocals.wav"
    turns = [
        Turn(start=float(c.delivery.reference[0]), end=float(c.delivery.reference[1]),
             speaker="cues")
        for c in cues if c.character_id == cid and _spoken(c)
    ]  # fmt: skip
    if not turns or not vocals.exists():
        return []
    tr_path, tags_path = transcript_path(work), work / "04b-tag" / "tags.json"
    transcript = (
        read_json(tr_path, Transcript)
        if tr_path.exists()
        else Transcript(language="", backend="none", word_timing=False, segments=[])
    )
    tags = read_json(tags_path, Tags) if tags_path.exists() else None
    theme = _episode_exclusions(ws, episode, tags, exclusions if exclusions is not None else {})
    return thin_references(turns, "cues", transcript, vocals, theme_spans=theme)


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
    (`character_references`), both at `PRE_FLOOR_S` and then again down to
    ``select.CLONE_MIN_S`` of gated picks (tier ``floor`` or ``short``); with neither, the
    thin tier (`character_thin`). A character with
    any turn always gets picks (A2, D41). Reads no backend, so the CLI can ask it too.
    ``fallback`` is the picks' language when their episode's transcript has none; ``loaded``
    and ``exclusions`` cache what each episode's files give, across characters."""
    loaded = loaded if loaded is not None else {}
    exclusions = exclusions if exclusions is not None else {}
    gated: dict[str, Gated] = {}
    candidates = character_candidates(ws, episodes, cid, exclusions, loaded, gated=gated)
    named = {e for e, a in loaded.items() if a is not None and a.names is not None}
    tiers = [candidates]
    acoustic_gated: dict[str, Gated] = {}
    best = None
    for minimum in (PRE_FLOOR_S, CLONE_MIN_S):
        best = character_references(
            ws, episodes, cid, candidates=candidates, named=named, acoustic={}, min_total=minimum
        )
        if best is not None:
            break
        if len(tiers) == 1:  # the acoustic tier is read only when confirmed turns give no picks
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
            ws, episodes, cid, candidates={}, named=named, acoustic=tiers[1], min_total=minimum
        )
        if best is not None:
            break
    if best is None:  # A2: the thin tier, from the character's longest turns
        thin = character_thin(ws, episodes, cid, exclusions, loaded)
        if thin is None:
            seconds = max(
                (sum(p.seconds for p in c) for tier in tiers for c in tier.values()), default=0.0
            )
            skipped = tuple(
                e
                for e in episodes
                if (a := loaded.get(e)) is not None
                and a.skipped is not None
                and any(c.character_id == cid for c in a.assignments.clusters)
            )
            return CharacterRefs(seconds=round(seconds, 3), names_used=bool(named), skipped=skipped)
        e, picks = thin
        arts = loaded.get(e)
        return CharacterRefs(
            round(sum(p.seconds for p in picks), 3), picks, e,
            ws.work_dir(e) / "01-separate" / "vocals.wav",
            (arts.transcript.language if arts else None) or fallback, False,
            names_used=bool(named), tier="thin",
        )  # fmt: skip
    e, picks, confirmed = best
    pool = candidates if confirmed else tiers[-1]
    seconds = sum(p.seconds for p in pool.get(e, []))
    arts = loaded.get(e)
    language = (arts.transcript.language if arts else None) or fallback
    vocals = ws.work_dir(e) / "01-separate" / "vocals.wav"
    return CharacterRefs(
        round(seconds, 3), picks, e, vocals, language, confirmed, names_used=bool(named),
        gated=(gated if confirmed else acoustic_gated).get(e),
        tier="floor" if meets_floor(picks) else "short",
    )  # fmt: skip


def floor_lines(
    ws: Workspace, reg: Registry, store: CardStore, characters: Iterable[int], language: str
) -> list[str]:
    """One line per automatic clone card under the clone floor (`below_floor`, D32) for
    `voices list` and `names show`: whether the next tts run re-picks it (unlocked, a better
    pick exists, cloning can run here) or it stays flagged for review."""
    short = {
        cid: sf for cid, sf in below_floor(reg, store, characters, language).items() if not sf.thin
    }  # a thin clone has its own line (`thin_lines`)
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


def thin_lines(reg: Registry, store: CardStore, ids: Iterable[int], language: str) -> list[str]:
    """One line per character whose ``language`` card is a thin-reference clone (A2), for
    ``voices list`` and ``names show``: listen to it in the studio."""
    from fukikae.cards.card import CardError

    out: list[str] = []
    for cid in sorted(ids):
        a = reg.card_for(cid, language)
        if a is None or a.card_id is None or not store.has(a.card_id):
            continue
        try:
            card = store.get(a.card_id)
        except CardError:
            continue
        clips = clip_lengths(card)
        if card.voice.source == "clone" and is_thin(card) and clips is not None:
            out.append(_thin_flag(reg, cid, a.card_id, thin_phrase(clips)))
    return out


def impure_lines(
    ws: Workspace,
    reg: Registry,
    store: CardStore,
    characters: Iterable[int],
    language: str,
    *,
    impure: Mapping[int, Impurity] | None = None,
) -> list[str]:
    """One line per clone card with references the A1 gates refuse (`impure_cards`, or the
    ``impure`` already measured) for `voices list` and `names show`: whether the next tts
    run re-picks it (automatic, a gate-passing pick exists, cloning can run here) or it stays
    flagged."""
    chars = set(characters)
    if impure is None:
        impure = impure_cards(ws, reg, store, language, chars)
    impure = {cid: imp for cid, imp in impure.items() if cid in chars}
    if not impure:
        return []
    loaded: dict[str, EpisodeArtifacts | None] = {}
    exclusions: dict[str, list[tuple[float, float]]] = {}
    episodes = _all_episodes(ws)
    blocker: list[str | None] = []
    out: list[str] = []
    for cid in sorted(impure):
        imp = impure[cid]
        a = reg.card_for(cid, language)
        assert a is not None
        head = (
            f"character {_label(reg, cid)}'s card {imp.card_id} ({a.chosen_by}"
            f"{', locked' if a.locked else ''}) has references that fail the reference gates "
            f"({imp.phrase})"
        )
        if a.chosen_by not in AUTOMATIC_BY or replace_refusal(a, "auto") is not None:
            out.append(f"{head}; kept: {PURITY_FIX}")
            continue
        found = find_references(ws, episodes, cid, None, loaded, exclusions)
        if not found.picks or found.tier == "thin":  # thin picks set the gates aside
            out.append(f"{head}; no gate-passing reference pick exists; flagged: {PURITY_FIX}")
            continue
        n = len(found.picks)
        better = f"{n} clip{'' if n == 1 else 's'}, {sum(p.seconds for p in found.picks):.1f} s"
        if not blocker:
            blocker.append(repick_blocker(ws.config))
        if blocker[0] is None:
            out.append(f"{head}; the next tts run re-picks it ({better})")
        else:
            out.append(f"{head}; a gate-passing pick exists ({better}), kept while {blocker[0]}")
    return out


def card_status_lines(
    ws: Workspace,
    reg: Registry,
    store: CardStore,
    characters: Iterable[int],
    language: str,
    *,
    notes: list[str] | None = None,
) -> list[str]:
    """The flagged cards of ``characters`` for ``voices list`` and ``names show``, one line
    per card with every reason it has (review fix 7): references the subtitle names now give
    someone else (`stale_cards`; a kept stale card under the floor says so too,
    `floor_suffix`), under the clone floor (`floor_lines`) or a thin-reference clone
    (`thin_lines`), and references the reference gates refuse (`impure_lines`), then stock
    cards (D41: an automatic one is re-picked as a clone by the next tts run, a curated,
    imported or locked one is kept). A card with one reason gets that reason's line; one with
    several gets a line naming each and what the next tts run does, as its single re-pick
    would. ``notes`` receives the episodes `stale_cards` skipped."""
    chars = sorted(set(characters))
    stale = stale_cards(ws, reg, store, chars, language, notes=notes)
    short = below_floor(reg, store, chars, language)
    impure = impure_cards(ws, reg, store, language, chars)
    multi = {cid for cid in impure if cid in stale or cid in short}
    blocker: list[str | None] = []

    def block() -> str | None:
        if not blocker:
            blocker.append(repick_blocker(ws.config))
        return blocker[0]

    out = [
        _stale_line(reg, cid, language, stale[cid], short, block)
        for cid in sorted(stale)
        if cid not in multi and reg.card_for(cid, language) is not None
    ]
    rest = [cid for cid in chars if cid not in stale and cid not in multi]
    out += floor_lines(ws, reg, store, rest, language)
    out += thin_lines(reg, store, rest, language)
    out += impure_lines(
        ws, reg, store, [cid for cid in chars if cid not in multi], language, impure=impure
    )
    loaded: dict[str, EpisodeArtifacts | None] = {}
    exclusions: dict[str, list[tuple[float, float]]] = {}
    for cid in sorted(multi):
        a = reg.card_for(cid, language)
        assert a is not None
        imp, sf, refs = impure[cid], short.get(cid), stale.get(cid)
        reasons: list[str] = []
        if refs:
            reasons.append(f"was cloned from turns {_now_named(refs)}")
        if sf is not None:
            reasons.append(
                f"is a thin-reference clone ({sf.phrase.removeprefix(f'{THIN_NOTE}: ')})"
                if sf.thin
                else f"is below the clone floor ({sf.phrase})"
            )
        reasons.append(f"has references that fail the reference gates ({imp.phrase})")
        head = (
            f"character {_label(reg, cid)}'s card {imp.card_id} ({a.chosen_by}"
            f"{', locked' if a.locked else ''}) {'; '.join(reasons)}"
        )
        refusal = replace_refusal(a, "auto")
        if a.chosen_by not in AUTOMATIC_BY or refusal is not None:
            out.append(f"{head}; kept{f': {refusal}' if refusal else ''}: {PURITY_FIX}")
            continue
        found = find_references(ws, _all_episodes(ws), cid, None, loaded, exclusions)
        gated = bool(found.picks) and found.tier != "thin"
        repicks = bool(refs and found.picks) or gated or (sf is not None and _better(found, sf))
        if not repicks:
            out.append(f"{head}; no better reference pick exists; flagged: {PURITY_FIX}")
            continue
        picks = found.picks or []
        n = len(picks)
        better = f"{n} clip{'' if n == 1 else 's'}, {sum(p.seconds for p in picks):.1f} s"
        why = block()
        if why is None:
            out.append(f"{head}; the next tts run re-picks it ({better})")
        else:
            out.append(f"{head}; a better pick exists ({better}), kept while {why}")
    for cid, card in sorted(stock_cards(reg, store, chars, language).items()):
        a = reg.card_for(cid, language)
        assert a is not None
        head = (
            f"character {_label(reg, cid)}'s card {card} ({a.chosen_by}"
            f"{', locked' if a.locked else ''}) is a stock card; stock voices are retired (D41)"
        )
        if a.chosen_by not in AUTOMATIC_BY or replace_refusal(a, "auto") is not None:
            out.append(
                f"{head}: kept; accept a clone in `fukikae studio` or use `fukikae voices clone`"
            )
        elif (why := block()) is None:
            out.append(f"{head}: the next tts run re-picks it as a clone")
        else:
            out.append(f"{head}: kept while {why}, then the next tts run re-picks it as a clone")
    return out


def _now_named(refs: Sequence[dict]) -> str:
    said = sorted({k for r in refs for k in r.get("names") or []})
    return f"now named {', '.join(k.upper() for k in said) or 'otherwise'}"


def _stale_line(
    reg: Registry,
    cid: int,
    language: str,
    refs: Sequence[dict],
    short: Mapping[int, Shortfall],
    block: Callable[[], str | None],
) -> str:
    """A stale card's line: what replaces it (a kept one under the floor says so too)."""
    a = reg.card_for(cid, language)
    assert a is not None
    if a.chosen_by in AUTOMATIC_BY and replace_refusal(a, "auto") is None:
        why = block()
        fix = (
            "the next tts run re-picks it"
            if why is None
            else f"kept while {why}; `fukikae voices clone` replaces it" + floor_suffix(short, cid)
        )
    else:
        fix = "`fukikae voices clone` replaces it" + floor_suffix(short, cid)
    return (
        f"character {_label(reg, cid)}'s card {a.card_id} ({a.chosen_by}) was cloned from "
        f"turns {_now_named(refs)}; {fix}"
    )
