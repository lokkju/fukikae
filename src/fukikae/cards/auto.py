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
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fukikae.cards.select import FEMALE_ABOVE, MALE_BELOW, gender_of, speaker_f0
from fukikae.cards.stock import check_stock_voice, pick_stock, revision, stock_card
from fukikae.cards.store import CardStore
from fukikae.registry.store import AUTOMATIC_BY, ProtectedAssignmentError, replace_refusal

if TYPE_CHECKING:
    from fukikae.backends.base import Synthesizer
    from fukikae.cards.card import Card
    from fukikae.cards.select import RefPick
    from fukikae.config import Config
    from fukikae.models import Assignments, ClusterAssignment, DubCue, Tags, Transcript
    from fukikae.names.models import NamesResult
    from fukikae.registry.store import Registry
    from fukikae.workspace import Workspace

__all__ = [
    "AUTOMATIC_BY",
    "FALLBACK_BY",
    "REPICK_NOTE",
    "STALE_OVERLAP",
    "CharacterRefs",
    "Cloning",
    "ProtectedAssignmentError",
    "character_candidates",
    "character_references",
    "character_spans",
    "checked_names",
    "clone_backend",
    "confirmed_labels",
    "confirmed_spans",
    "ensure_assignments",
    "extras_voices",
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
STALE_OVERLAP = 0.1  # this share of a reference under someone else's names makes it stale
REPICK_NOTE = "re-picked: references not name-confirmed"


@dataclass(frozen=True)
class CharacterRefs:
    """What a character's turns offer a clone: ``seconds`` of eligible reference speech in
    the episode that has the most, and the picks there (with that episode and its vocals),
    or no picks under the minimum."""

    seconds: float
    picks: list[RefPick] | None = None
    episode: str | None = None
    vocals: Path | None = None
    language: str | None = None  # the picks' episode's transcript language
    confirmed: bool = False  # the picks are name-confirmed turns of a names-used episode
    names_used: bool = False  # some episode searched used subtitle names


@dataclass(frozen=True)
class Cloning:
    """The clone branch of ``ensure_assignments``: the backend clone cards are made for, the
    language the references speak, and each character's references."""

    backend: type[Synthesizer]
    source_language: str
    references: Callable[[int], CharacterRefs]
    unavailable: str | None = None  # why the backend can't run here: stock for everyone


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
    lines, because they change nothing and would repeat on every run."""
    language = cfg.target_language
    usable = cloning is not None and cloning.unavailable is None
    todo = []
    fallbacks: set[int] = set()
    repick: dict[int, str] = {}  # character -> its current card id
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
            if a.chosen_by not in AUTOMATIC_BY or replace_refusal(a, "auto") is not None:
                flagged.append(
                    f"character {_label(reg, cid)}'s card {a.card_id} ({a.chosen_by}"
                    f"{', locked' if a.locked else ''}) was cloned from turns the subtitles "
                    f"now name otherwise: {where}; `fukikae voices clone` replaces it"
                )
            elif usable:
                todo.append((cid, spans))
                repick[cid] = a.card_id
            elif cloning is None:
                flagged.append(
                    f"character {_label(reg, cid)}: card {a.card_id} is cloned from turns the "
                    f"subtitles now name otherwise ({where}); kept, because no [tts] "
                    "clone_backend is set to re-pick it with: set one, or replace it with "
                    "`fukikae voices clone` or `fukikae voices assign`"
                )
            else:
                name = cloning.backend.name
                flagged.append(
                    f"character {_label(reg, cid)}: card {a.card_id} is cloned from turns the "
                    f"subtitles now name otherwise ({where}); kept until the clone backend "
                    f"{name} can run ({cloning.unavailable}; `fukikae backends setup {name}`), "
                    "and the next tts run then re-picks it"
                )
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
        if found is not None and found.picks:
            new, note = _clone(cloning, found, store, language)  # type: ignore[arg-type]
            if cid in fallbacks:
                note += "; replaces the stock fallback made while the clone backend couldn't run"
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
        try:
            reg.assign_card(cid, language, new, by=by, locked=False, note=note)
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
    for warning in reference_report(picks):  # ruling CD5
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
    to it (by name, or kept by a lock), or the registry gives it the key."""
    mine = {
        c.name
        for c in asg.clusters
        if c.character_id == cid and c.name and c.status not in ("extra", "review")
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
) -> dict[str, list[RefPick]]:
    """Per episode, the character's eligible reference turns (``eligible_references``, songs
    and singing excluded), ranked. ``exclusions`` caches each episode's excluded spans and
    ``loaded`` its artifacts, so a caller asking for many characters reads each file once.

    In an episode whose names were used (``owner`` answers keys, the workspace's registry by
    default), the candidates are the character's name-confirmed turns (`confirmed_labels`),
    or with ``acoustic`` all of its turns, unnamed remainders included; either way never a
    span of the names stage's inner-voice and multi lines, nor a turn with `STALE_OVERLAP` of
    it under someone else's names (`foreign_spans`). A names-used episode whose assignments
    are older than its names (``skipped``) offers nothing."""
    from fukikae.cards.select import eligible_references, reference_exclusions
    from fukikae.stages.mix import mix_settings

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
        if a.names is not None:
            if owner is None:
                owner = _workspace_owner(ws)
            speakers = own if acoustic else confirmed_labels(a.assignments, cid, owner)
            foreign = foreign_spans(a.assignments, a.names, cid, owner)
        else:
            speakers = own
        if not speakers or not a.assignments.turns:
            continue
        if e not in cache:
            cache[e] = reference_exclusions(ws.work_dir(e), a.tags, mix_settings(ws.config).songs)
        exclude = cache[e]
        turns = a.assignments.turns
        if a.names is not None:
            exclude = [*exclude, *((float(x), float(y)) for x, y in a.names.excluded)]
            turns = [
                t
                for t in turns
                if t.speaker not in speakers or _inside(t.start, t.end, foreign) < STALE_OVERLAP
            ]
        out[e] = eligible_references(
            turns, speakers, a.transcript, a.tags, a.vocals, exclude=exclude
        )
    return out


def _best(
    episodes: Sequence[str], found: Mapping[str, list[RefPick]], allowed: Collection[str] | None
) -> tuple[str, list[RefPick]] | None:
    from fukikae.cards.select import pick_references

    best: tuple[str, list[RefPick]] | None = None
    for e in episodes:
        if allowed is not None and e not in allowed:
            continue
        picks = pick_references(found[e]) if e in found else None
        if picks and (
            best is None or sum(p.seconds for p in picks) > sum(p.seconds for p in best[1])
        ):
            best = (e, picks)
    return best


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


def confirmed_spans(
    ws: Workspace,
    character_id: int,
    *,
    episodes: Sequence[str] | None = None,
    owner: KeyOwner | None = None,
    cache: NamesCache | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Per names-used episode of the workspace (or of ``episodes``), the character's
    name-confirmed turns (`confirmed_labels`), merged and in time order, less the names
    stage's inner-voice and multi spans; an episode that used names but confirms none of its
    turns maps to an empty list. ``cache`` keeps each episode's reads for the next call."""
    owner = owner if owner is not None else _workspace_owner(ws)
    cache = cache if cache is not None else {}
    out: dict[str, list[tuple[float, float]]] = {}
    for e in episodes if episodes is not None else _all_episodes(ws):
        asg, names, _ = _episode_names(ws, e, cache)
        if asg is None or names is None:
            continue
        labels = confirmed_labels(asg, character_id, owner)
        turns = [(t.start, t.end) for t in asg.turns or [] if t.speaker in labels]
        out[e] = _subtract(turns, names.excluded)
    return out


def _inside(start: float, end: float, spans: Sequence[tuple[float, float]]) -> float:
    """The share of ``start``-``end`` inside the union of ``spans``."""
    covered, edge = 0.0, start
    for a, b in sorted(spans):
        a, b = max(a, edge), min(b, end)
        if b > a:
            covered += b - a
            edge = b
    return covered / (end - start) if end > start else 1.0


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
        inst = cls({}, "cpu")
        if not inst.interpreter().exists():
            return f"no worker environment at {inst.venv()}"
        lock = inst.lock_problem()
        if lock is not None:
            return lock
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
        candidates = character_candidates(ws, episodes, cid, exclusions, loaded)
        named = {e for e, a in loaded.items() if a is not None and a.names is not None}
        acoustic = {
            **{e: c for e, c in candidates.items() if e not in named},
            **character_candidates(
                ws, [e for e in episodes if e in named], cid, exclusions, loaded, acoustic=True
            ),
        }
        seconds = max(
            (sum(p.seconds for p in c) for c in (*candidates.values(), *acoustic.values())),
            default=0.0,
        )
        best = character_references(
            ws, episodes, cid, candidates=candidates, named=named, acoustic=acoustic
        )
        if best is None:
            return CharacterRefs(seconds=round(seconds, 3), names_used=bool(named))
        e, picks, confirmed = best
        arts = loaded.get(e)
        language = (arts.transcript.language if arts else None) or fallback
        vocals = ws.work_dir(e) / "01-separate" / "vocals.wav"
        return CharacterRefs(
            round(seconds, 3), picks, e, vocals, language, confirmed, names_used=bool(named)
        )

    return Cloning(backend=cls, source_language=fallback, references=references)
