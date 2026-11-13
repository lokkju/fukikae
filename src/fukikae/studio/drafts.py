"""Reference candidates and voice drafts (Task 18): the voice-tuning loop's data.

`candidates` lists a character's own speech across the ready episodes as references: every
candidate the automatic rules judge eligible (``rank`` in the CD5 order, name-confirmed turns
first) and every turn they refuse, with each check's verdict (`fukikae.cards.select.REASONS`
and the A1 gates, measured as `fukikae.cards.auto.character_candidates` measures them: the
operator's gender beats the estimate, and name-confirmed turns are gated by class only). A
character with no turn anywhere is offered its own spoken cues' source spans, what the
cue-thin tier clones it from. Nothing else is ever offered: D41, every voice is a clone of
the character's own speech.

A draft (`Draft`, ``<ws>/studio/drafts/<character>.<language>.json``, owner-only) is one
character's tuning session for one target language. `open_draft` makes version 1 from the
character's current clone card for the language; else from an accepted card of another
language, copied the way ``fukikae voices derive`` copies one; else from the automatic
proposal (`fukikae.cards.auto.find_references`, the tiers included), assigned nowhere. A
retired stock card is never a starting point. `add_version` builds each new version's card
with `draft_card`, exactly as the automatic clone is built (`fukikae.cards.select.
clone_card_spans`), with the backend's own voice keys as parameters, and records the
references no eligible candidate covers as ``overrides`` (the operator chose a gated turn),
the digest of the language's card assignment (`fukikae.studio.apply.row_digest`, what an
accept carries) and the tier.

Drafts never write the registry. Draft cards go into the content-addressed card store,
because auditions render them and an accept assigns one from there; putting a card assigns
it to nobody.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fukikae.cards.select import (
    CLONE_MIN_S,
    CLONE_SEED,
    FLOOR_CLIPS,
    FLOOR_S,
    GATES,
    PROMPT_GAP_S,
    SCENE_GAP_S,
    THIN_MAX_S,
    reaches_floor,
)
from fukikae.studio.model import Candidate, Draft, DraftVersion, RefSpan
from fukikae.workspace import Workspace

if TYPE_CHECKING:
    from fukikae.backends.base import Synthesizer
    from fukikae.cards.auto import CandidateReport
    from fukikae.cards.card import Card
    from fukikae.cards.select import Tier
    from fukikae.config import Config
    from fukikae.pitch import Gender
    from fukikae.registry.store import Registry

# The pre-gate checks a turn's verdicts list (`select.REASONS` without ``repeat``, which is no
# check of its own).
CHECKS = ("length", "overlap", "excluded", "foreign", "no_text", "sparse")
# How far a reference may run past the character's speech (an origin's float rounding).
SPAN_SLACK_S = 0.01
# The most references a version may name, and the shortest one.
MAX_REFERENCES = 12
MIN_REFERENCE_S = 0.2
# The [voice] keys a draft's parameters may set, when the backend lists them in voice_keys.
VOICE_PARAMS = (
    "seed", "temperature", "top_p", "rate", "pitch_semitones", "emotion_strength",
    "qwen3_tts_mode", "cosyvoice3_mode",
)  # fmt: skip
_NULLABLE = frozenset({"seed", "temperature", "top_p"})
_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,3}")


class DraftError(Exception):
    """A draft request the rules refuse (the message says why)."""


class UnknownCharacter(DraftError):  # noqa: N818 (a 404, not an error of the caller's)
    """No such character in the registry."""


class NoRegistry(DraftError):  # noqa: N818
    """The workspace has no registry yet."""


def _workspace(ws: Workspace | Path) -> Workspace:
    return ws if isinstance(ws, Workspace) else Workspace(Path(ws))


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def language_of(tag: str) -> str:
    """``tag`` as a normalized target language; DraftError when it isn't a language tag."""
    from fukikae.config import normalize_language

    if not _LANGUAGE.fullmatch(tag):
        raise DraftError(f"not a language tag: {tag!r}")
    return normalize_language(tag)


# -- files ----------------------------------------------------------------------------------


def drafts_dir(ws: Workspace | Path) -> Path:
    return _workspace(ws).root / "studio" / "drafts"


def draft_path(ws: Workspace | Path, character_id: int, language: str) -> Path:
    return drafts_dir(ws) / f"{int(character_id)}.{language_of(language)}.json"


def load_draft(ws: Workspace | Path, character_id: int, language: str) -> Draft | None:
    """The saved draft, or None when there is none."""
    from fukikae.artifacts import read_json

    path = draft_path(ws, character_id, language)
    if not path.is_file():
        return None
    return read_json(path, Draft)


def save_draft(ws: Workspace | Path, draft: Draft) -> Path:
    """Write the draft to its file, atomically, owner-only (the directory too)."""
    from fukikae.artifacts import atomic_path

    d = drafts_dir(ws)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    path = draft_path(ws, draft.character_id, draft.language)
    with atomic_path(path) as tmp:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(draft.model_dump_json(indent=2))
    return path


# -- the registry ---------------------------------------------------------------------------


def _registry(ws: Workspace) -> Registry:
    from fukikae.registry.store import Registry

    if not ws.registry_path.is_file():
        raise NoRegistry(f"{ws.root} has no registry yet")
    return Registry(ws.registry_path)


def _character_state(ws: Workspace, character_id: int) -> tuple[Gender | None, bool]:
    """(the character's operator gender, whether it is a narrator); UnknownCharacter when the
    registry has no such character."""
    from fukikae.cards.auto import narrators, operator_genders

    with _registry(ws) as reg:
        if not any(c.id == character_id for c in reg.characters()):
            raise UnknownCharacter(f"no character {character_id} in the registry")
        return operator_genders(reg).get(character_id), character_id in narrators(reg)


def ready_episodes(ws: Workspace | Path) -> list[str]:
    """The episodes the studio can read speech from: assignments and a vocals stem."""
    from fukikae.studio.season import episode_ids

    w = _workspace(ws)
    return [
        e
        for e in episode_ids(w)
        if (w.work_dir(e) / "05-characters" / "assignments.json").is_file()
        and (w.work_dir(e) / "01-separate" / "vocals.wav").is_file()
    ]


# -- candidates -----------------------------------------------------------------------------


def _band(median: float | None) -> Gender | None:
    from fukikae.cards.select import gender_of

    return None if median is None else gender_of(median)


def _gate_verdicts(gates: Sequence[str], measured: bool) -> dict[str, bool]:
    return {g: g not in gates for g in GATES} if measured else {}


def _overlaps(start: float, end: float, spans: Iterable[tuple[float, float]]) -> bool:
    return any(a < end and start < b for a, b in spans)


def _report_items(
    rep: CandidateReport, skip: Sequence[tuple[float, float]] = ()
) -> list[tuple[Any, tuple[str, ...], tuple[str, ...], bool, bool]]:
    """(pick, pre-gate reasons, gates, eligible, confirmed) for a report's candidates and
    refused turns, leaving out any overlapping ``skip`` (turns another report covers)."""
    out: list[tuple[Any, tuple[str, ...], tuple[str, ...], bool, bool]] = []
    for v in rep.verdicts:
        if not _overlaps(v.pick.start, v.pick.end, skip):
            out.append((v.pick, (), tuple(v.gates), v.passed, rep.confirmed))
    for r in rep.refused:
        if not _overlaps(r.pick.start, r.pick.end, skip):
            out.append((r.pick, tuple(r.reasons), tuple(r.gates), False, rep.confirmed))
    return out


def candidates(
    ws: Workspace | Path,
    cfg: Config,  # noqa: ARG001 (the plan's signature: the stage settings come from ws)
    character_id: int,
) -> list[Candidate]:
    """Every stretch of the character's own speech in the ready episodes (`ready_episodes`)
    as a reference candidate: the eligible ones first, in rank order, then the refused ones
    by episode and time (see the module notes). In an episode whose subtitle names were used,
    the name-confirmed turns are judged as the confirmed tier judges them and the character's
    other turns as the acoustic tier does. Reads, never writes: the clips are named, not
    cut."""
    from fukikae.cards.auto import character_candidates
    from fukikae.cards.select import rank_key
    from fukikae.studio.season import sample_name

    w = _workspace(ws)
    gender, narrator = _character_state(w, character_id)
    episodes = ready_episodes(w)
    order = {e: i for i, e in enumerate(episodes)}
    loaded: dict[str, Any] = {}
    exclusions: dict[str, list[tuple[float, float]]] = {}
    first: dict[str, CandidateReport] = {}
    character_candidates(w, episodes, character_id, exclusions, loaded, gender=gender, report=first)
    named = [e for e in episodes if (a := loaded.get(e)) is not None and a.names is not None]
    acoustic: dict[str, CandidateReport] = {}
    if named:
        character_candidates(
            w, named, character_id, exclusions, loaded, acoustic=True, gender=gender,
            report=acoustic,
        )  # fmt: skip
    items: list[tuple[str, Any, tuple[str, ...], tuple[str, ...], bool, bool]] = []
    for e in episodes:
        mine: list[tuple[float, float]] = []
        if e in first:
            items += [(e, *x) for x in _report_items(first[e])]
            if first[e].confirmed:
                mine = first[e].turn_spans
        if e in acoustic:
            items += [(e, *x) for x in _report_items(acoustic[e], mine)]
    eligible = sorted(
        (x for x in items if x[4]), key=lambda x: (not x[5], rank_key(x[1]), order[x[0]])
    )
    refused = sorted((x for x in items if not x[4]), key=lambda x: (order[x[0]], x[1].start))
    out: list[Candidate] = []
    taken: dict[str, list[float]] = {}
    for rank, (e, p, _, gates, _, confirmed) in enumerate(eligible):
        verdicts = dict.fromkeys(CHECKS, True)
        verdicts |= _gate_verdicts(gates, p.voiced_fraction is not None)
        if not narrator:  # gate 4 informs, never refuses; narrators have none
            starts = taken.setdefault(e, [])
            verdicts["scene"] = not any(abs(p.start - s) < SCENE_GAP_S for s in starts)
            starts.append(p.start)
        out.append(_candidate(w, e, p, rank, verdicts, [], confirmed, sample_name))
    for e, p, reasons, gates, _, confirmed in refused:
        verdicts = {k: k not in reasons for k in CHECKS}
        verdicts |= _gate_verdicts(gates, p.voiced_fraction is not None)
        failed = [*reasons, *(g for g in gates if g not in reasons)]
        out.append(_candidate(w, e, p, None, verdicts, failed, confirmed, sample_name))
    if not out:
        out = _cue_candidates(w, episodes, character_id, sample_name)
    return out


def _candidate(
    ws: Workspace,
    episode: str,
    p: Any,
    rank: int | None,
    verdicts: dict[str, bool],
    reasons: list[str],
    confirmed: bool,
    sample_name: Any,
) -> Candidate:
    return Candidate(
        episode=episode, start=p.start, end=p.end, seconds=round(p.end - p.start, 3),
        transcript=p.transcript, snr=p.snr_db, median=p.median, gender=_band(p.median),
        rank=rank, verdicts=verdicts, reasons=reasons,
        clip=sample_name(ws, episode, p.start, p.end), emotion=p.emotion, confirmed=confirmed,
    )  # fmt: skip


def _cue_candidates(
    ws: Workspace, episodes: Sequence[str], character_id: int, sample_name: Any
) -> list[Candidate]:
    """For a character with no turn in any ready episode: its own spoken cues' source spans
    (joined as accept joins them), longest first."""
    from fukikae.artifacts import read_json
    from fukikae.cards.select import turn_text
    from fukikae.models import Transcript
    from fukikae.studio.apply import _cue_spans
    from fukikae.transcripts import transcript_path

    spans: list[tuple[str, float, float, str]] = []
    for e in episodes:
        work = ws.work_dir(e)
        found = _cue_spans(work, character_id)
        if not found:
            continue
        tr_path = transcript_path(work)
        tr = read_json(tr_path, Transcript) if tr_path.exists() else None
        spans += [(e, a, b, turn_text(tr, a, b) if tr else "") for a, b in found]
    spans.sort(key=lambda s: (-(s[2] - s[1]), s[0], s[1]))
    return [
        Candidate(
            episode=e,
            start=a,
            end=b,
            seconds=round(b - a, 3),
            transcript=text,
            snr=None,
            median=None,
            gender=None,
            rank=i,
            verdicts={},
            reasons=[],
            clip=sample_name(ws, e, a, b),
            source="cue",
        )  # fmt: skip
        for i, (e, a, b, text) in enumerate(spans)
    ]


def overrides(references: Sequence[RefSpan], offered: Sequence[Candidate]) -> list[RefSpan]:
    """The references no eligible candidate covers: gated or refused turns the operator chose
    anyway, recorded as overrides."""
    ok = [c for c in offered if c.rank is not None]
    return [
        r
        for r in references
        if not any(
            c.episode == r.episode
            and c.start - SPAN_SLACK_S <= r.start
            and r.end <= c.end + SPAN_SLACK_S
            for c in ok
        )
    ]


# -- the meter ------------------------------------------------------------------------------


def floor_meter(references: Sequence[Any]) -> dict[str, Any]:
    """The selection against the D32 floor (`FLOOR_CLIPS` clips, `FLOOR_S` seconds) and
    IndexTTS2's prompt (`THIN_MAX_S`, counting `PROMPT_GAP_S` between clips): clips, seconds,
    whether it runs over the prompt, and its tier (``floor``, ``short`` from `CLONE_MIN_S`,
    else ``thin``)."""
    clips = [float(r.end) - float(r.start) for r in references]
    seconds = round(sum(clips), 3)
    prompt = seconds + PROMPT_GAP_S * max(0, len(clips) - 1)
    tier: Tier = "floor" if reaches_floor(clips) else "short" if seconds >= CLONE_MIN_S else "thin"
    return {
        "clips": len(clips), "seconds": seconds, "floor_clips": FLOOR_CLIPS, "floor_s": FLOOR_S,
        "prompt_s": THIN_MAX_S, "over_prompt": prompt > THIN_MAX_S + 1e-9, "tier": tier,
    }  # fmt: skip


# -- backends and parameters ----------------------------------------------------------------


def clone_class(name: str) -> type[Synthesizer]:
    """The clone backend class named ``name``; DraftError for an unknown backend, one that
    makes no clone voices, or an engine of another backend."""
    from fukikae.backends import loader
    from fukikae.backends.base import Synthesizer

    try:
        cls = cast(type[Synthesizer], loader.backend_class("tts", name))
    except Exception as e:  # noqa: BLE001 (unknown, or its extra isn't installed)
        raise DraftError(f"no tts backend {name!r} here ({e})") from None
    if "clone" not in getattr(cls, "sources", ()):
        raise DraftError(f"{name} makes no clone voices (D41: every voice is a clone)")
    own = getattr(cls, "card_backend", None)
    if own and own != name:
        raise DraftError(f"{name} is an engine of {own}; a draft names {own}")
    return cls


def voice_key_spec(key: str) -> dict[str, Any]:
    """A voice key's control for the page: type, bounds, the clone default and choices."""
    from fukikae.cards.card import MODE_FIELDS, SEED_MAX, VOICE_RANGES, Voice

    if key == "seed":
        return {"key": key, "type": "int", "min": 0, "max": SEED_MAX, "default": CLONE_SEED,
                "choices": None}  # fmt: skip
    if key in MODE_FIELDS:
        default = getattr(Voice("", "", ""), key)
        return {"key": key, "type": "choice", "min": None, "max": None, "default": default,
                "choices": list(MODE_FIELDS[key])}  # fmt: skip
    bounds = VOICE_RANGES.get(key, {})
    spec: dict[str, Any] = {
        "key": key, "type": "float", "min": bounds.get("lo"), "max": bounds.get("hi"),
        "default": getattr(Voice("", "", ""), key, None), "choices": None,
    }  # fmt: skip
    if bounds.get("lo_open"):
        spec["min_open"] = True
    return spec


def backends(cfg: Config) -> list[dict[str, Any]]:
    """The clone backends this install can import, each with its voice keys (`voice_key_spec`),
    whether it is the workspace's ``[tts] clone_backend`` and whether it can render here
    (``cards.auto.clone_problem``: its worker environment and device)."""
    from fukikae.backends import loader
    from fukikae.cards.auto import clone_backend, clone_problem

    try:
        default = clone_backend(cfg)
    except Exception:  # noqa: BLE001 (a bad setting: no default)
        default = None
    out: list[dict[str, Any]] = []
    for name, cls in sorted(loader.available("tts").items()):
        if "clone" not in getattr(cls, "sources", ()):
            continue
        own = getattr(cls, "card_backend", None)
        if own and own != name:
            continue
        synth = cast("type[Synthesizer]", cls)
        try:
            problem = clone_problem(synth, cfg)
        except Exception as e:  # noqa: BLE001 (reported, never fatal to the listing)
            problem = str(e)
        out.append({
            "name": name, "default": name == default, "installed": problem is None,
            "problem": problem, "capabilities": sorted(getattr(cls, "capabilities", ())),
            "voice_keys": [voice_key_spec(k) for k in VOICE_PARAMS if k in synth.voice_keys],
        })  # fmt: skip
    return out


def check_params(cls: type[Synthesizer], params: Mapping[str, Any]) -> dict[str, Any]:
    """``params`` with each value in its [voice] type; DraftError for a key the backend
    doesn't list in its ``voice_keys`` (or no draft parameter), or a value of the wrong type.
    Ranges are checked when the card is built (the card parser's own bounds)."""
    from fukikae.cards.card import MODE_FIELDS

    if not isinstance(params, Mapping):
        raise DraftError("params must be an object of voice keys")
    out: dict[str, Any] = {}
    for key, value in params.items():
        if key not in VOICE_PARAMS or key not in cls.voice_keys:
            keys = ", ".join(k for k in VOICE_PARAMS if k in cls.voice_keys) or "none"
            raise DraftError(f"{cls.name} has no voice key {key!r}; its keys: {keys}")
        if value is None and key in _NULLABLE:
            out[key] = None
        elif key == "seed":
            if isinstance(value, bool) or not isinstance(value, int):
                raise DraftError(f"seed must be an integer, not {value!r}")
            out[key] = value
        elif key in MODE_FIELDS:
            if value not in MODE_FIELDS[key]:
                raise DraftError(f"{key} must be one of {', '.join(MODE_FIELDS[key])}")
            out[key] = value
        else:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DraftError(f"{key} must be a number, not {value!r}")
            if not math.isfinite(value):
                raise DraftError(f"{key} must be finite")
            out[key] = float(value)
    return out


def _params_of(card: Card) -> dict[str, Any]:
    """The draft parameters a card carries: every draft key its backend lists, else every
    one that differs from the default (and the seed)."""
    from fukikae.backends import loader
    from fukikae.cards.card import Voice

    try:
        keys = loader.backend_class("tts", card.voice.backend).voice_keys  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 (its backend isn't installed: read what the card says)
        plain = Voice("", "", "")
        keys = {k for k in VOICE_PARAMS if getattr(card.voice, k) != getattr(plain, k)}
        keys.add("seed")
    return {k: getattr(card.voice, k) for k in VOICE_PARAMS if k in keys}


# -- own speech -----------------------------------------------------------------------------


def check_own_speech(ws: Workspace | Path, character_id: int, refs: Sequence[RefSpan]) -> None:
    """DraftError unless every reference lies inside the character's own turns in its episode
    (joined as accept joins them, `fukikae.studio.apply._own_turns`) or its own spoken cues'
    source spans there. Overlap with another speaker is allowed here (a gated pick, recorded
    as an override); accept checks it again by the card's tier."""
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments
    from fukikae.studio.apply import _cue_spans, _own_turns

    w = _workspace(ws)
    turns: dict[str, list[tuple[float, float]]] = {}
    bad: list[str] = []
    for r in refs:
        if r.episode not in turns:
            path = w.work_dir(r.episode) / "05-characters" / "assignments.json"
            mine: list[tuple[float, float]] = []
            if path.is_file():
                asg = read_json(path, Assignments)
                if asg.turns is not None:
                    mine = _own_turns(asg, character_id)[0]
            turns[r.episode] = mine + _cue_spans(w.work_dir(r.episode), character_id)
        if not any(
            a - SPAN_SLACK_S <= r.start and r.end <= b + SPAN_SLACK_S for a, b in turns[r.episode]
        ):
            bad.append(r.label())
    if bad:
        raise DraftError(
            f"not character {character_id}'s own speech (D41: a draft clones only the "
            f"character's own turns or spoken cues): {', '.join(bad)}"
        )


# -- draft cards ----------------------------------------------------------------------------


def _check_refs(ws: Workspace, refs: Sequence[RefSpan]) -> None:
    if not refs:
        raise DraftError("a version needs at least one reference")
    if len(refs) > MAX_REFERENCES:
        raise DraftError(f"a version takes at most {MAX_REFERENCES} references")
    ready = set(ready_episodes(ws))
    for r in refs:
        if not (math.isfinite(r.start) and math.isfinite(r.end)) or r.start < 0:
            raise DraftError(f"not a span: {r.label()}")
        if r.end - r.start < MIN_REFERENCE_S:
            raise DraftError(f"{r.label()}: a reference is at least {MIN_REFERENCE_S:g} s")
        if r.episode not in ready:
            raise DraftError(f"{r.episode} has no assignments and vocals to cut a reference from")


def _thin_refs(draft: Draft, refs: Sequence[RefSpan]) -> bool:
    """Whether ``refs`` are exactly the references of a thin-tier version of the draft (a
    change of parameters only keeps the pipeline's thin tier, and accept's thin rule)."""
    want = sorted((r.episode, r.start, r.end) for r in refs)
    return any(
        v.tier == "thin" and sorted((r.episode, r.start, r.end) for r in v.references) == want
        for v in draft.versions
    )


def _build_card(
    ws: Workspace,
    cfg: Config,
    character_id: int,
    language: str,
    refs: Sequence[RefSpan],
    cls: type[Synthesizer],
    params: Mapping[str, Any],
    tier: Tier | None,
) -> str:
    from fukikae.artifacts import read_json
    from fukikae.cards.card import CardError, dump_card, parse_card, validate_for_backend
    from fukikae.cards.select import RefPick, SpanPick, _emotion, clone_card_spans, turn_text
    from fukikae.cards.store import CardStore
    from fukikae.models import Tags, Transcript
    from fukikae.transcripts import transcript_path

    norm = check_params(cls, params)
    episodes: dict[str, tuple[Transcript | None, Tags | None]] = {}
    items: list[SpanPick] = []
    for r in refs:
        if r.episode not in episodes:
            work = ws.work_dir(r.episode)
            tr_path, tags_path = transcript_path(work), work / "04b-tag" / "tags.json"
            episodes[r.episode] = (
                read_json(tr_path, Transcript) if tr_path.exists() else None,
                read_json(tags_path, Tags) if tags_path.exists() else None,
            )
        tr, tags = episodes[r.episode]
        text = turn_text(tr, r.start, r.end) if tr is not None else ""
        pick = RefPick(r.start, r.end, text, 0.0, _emotion(tags, r.start, r.end))
        lang = (tr.language if tr is not None else None) or cfg.source_language or ""
        items.append(SpanPick(pick, r.episode, ws.work_dir(r.episode) / "01-separate" /
                              "vocals.wav", lang))  # fmt: skip
    store = CardStore(ws.root)
    card = clone_card_spans(cls, items, store, language, tier)
    card = replace(
        card, voice=replace(card.voice, **norm), meta={**card.meta, "character": character_id}
    )
    try:
        parse_card(dump_card(card))  # the card parser's bounds (seed bits, emotion 0 to 1, ...)
        validate_for_backend(card, cls)
    except CardError as e:
        raise DraftError(str(e)) from None
    return store.put(card)


def draft_card(
    ws: Workspace | Path,
    cfg: Config,
    draft: Draft,
    references: Sequence[RefSpan],
    backend: str,
    params: Mapping[str, Any],
) -> str:
    """The clone card for ``references`` with ``backend`` and ``params``, built as
    ``cards.auto._clone`` builds a clone (a 24 kHz sidecar per reference cut from its own
    episode's vocals, its ASR text and origin; seed 1234 and the defaults unless ``params``
    sets them), put in the card store; its id. ``params`` may set only the backend's
    ``voice_keys``. Every reference must be the character's own speech (`check_own_speech`).
    The tier is the references' own (`floor_meter`), except that the exact references of a
    thin-tier version stay thin."""
    w = _workspace(ws)
    cls = clone_class(backend)
    check_params(cls, params)
    _check_refs(w, references)
    check_own_speech(w, draft.character_id, references)
    tier: Tier | None = "thin" if _thin_refs(draft, references) else None
    return _build_card(w, cfg, draft.character_id, draft.language, references, cls, params, tier)


def _digest(ws: Workspace, character_id: int, language: str) -> str:
    from fukikae.studio.apply import row_digest

    with _registry(ws) as reg:
        return row_digest(reg, character_id, language)


def add_version(
    ws: Workspace | Path,
    cfg: Config,
    draft: Draft,
    references: Sequence[RefSpan],
    backend: str,
    params: Mapping[str, Any],
    *,
    offered: Sequence[Candidate] | None = None,
) -> tuple[Draft, DraftVersion]:
    """A new version of the draft from ``references``, ``backend`` and ``params``
    (`draft_card`), its gated picks recorded as overrides (against ``offered``, by default
    `candidates`), saved; the updated draft and the version."""
    w = _workspace(ws)
    refs = list(references)
    card_id = draft_card(w, cfg, draft, refs, backend, params)
    cls = clone_class(backend)
    offered = candidates(w, cfg, draft.character_id) if offered is None else offered
    from fukikae.cards.card import CardError
    from fukikae.cards.store import CardStore

    try:
        tier = CardStore(w.root).get(card_id).meta.get("reference_tier")
    except CardError:
        tier = None
    version = DraftVersion(
        n=max((v.n for v in draft.versions), default=0) + 1, backend=backend, references=refs,
        params={**_defaults(cls), **check_params(cls, params)}, card_id=card_id,
        overrides=overrides(refs, offered), created=_now(),
        digest=_digest(w, draft.character_id, draft.language),
        tier=tier if tier in ("floor", "short", "thin") else floor_meter(refs)["tier"],
    )  # fmt: skip
    draft = draft.model_copy(update={"versions": [*draft.versions, version]})
    save_draft(w, draft)
    return draft, version


def _defaults(cls: type[Synthesizer]) -> dict[str, Any]:
    """A clone card's value of each draft key the backend lists (seed 1234, the defaults)."""
    from fukikae.cards.card import Voice

    plain = Voice("", "", "", seed=CLONE_SEED)
    return {k: getattr(plain, k) for k in VOICE_PARAMS if k in cls.voice_keys}


# -- opening a draft ------------------------------------------------------------------------


def _clone_of(store: Any, card_id: str | None) -> Card | None:
    """The stored card when it is a clone (never a retired stock card), else None."""
    from fukikae.cards.card import CardError

    if card_id is None or not store.has(card_id):
        return None
    try:
        card = store.get(card_id)
    except CardError:
        return None
    return card if card.voice.source == "clone" else None


def _spans_of(card: Card) -> tuple[list[RefSpan], int]:
    refs: list[RefSpan] = []
    bare = 0
    for r in card.voice.references:
        o = r.origin or {}
        try:
            refs.append(RefSpan(episode=str(o["episode"]), start=float(o["start"]),
                                end=float(o["end"])))  # fmt: skip
        except (KeyError, TypeError, ValueError):
            bare += 1
    return refs, bare


def _tier_of(card: Card, refs: Sequence[RefSpan]) -> Tier:
    t = card.meta.get("reference_tier")
    return t if t in ("floor", "short", "thin") else floor_meter(refs)["tier"]


def _derive(store: Any, old: Card, old_id: str, language: str, character_id: int) -> str:
    """``fukikae voices derive``'s copy of a card for another target language (no
    fingerprint, catchphrase renderings emptied, ``derived_from``), put in the store."""
    import datetime as dt

    from fukikae.cards.card import Card as CardT

    style = None
    if old.style:
        style = dict(old.style)
        if "catchphrases" in style:
            style["catchphrases"] = [{**p, "target": ""} for p in style["catchphrases"]]
    new = CardT(
        target_language=language, voice=replace(old.voice, fingerprint=None), style=style,
        meta={**old.meta, "derived_from": old_id, "created": dt.date.today(),
              "character": character_id},
    )  # fmt: skip
    return store.put(new)


def _accepted_elsewhere(reg: Registry, store: Any, character_id: int, language: str) -> Any:
    """(language, card id, card) of the character's accepted clone card in another target
    language: a studio accept first, then any other operator choice; None without one."""
    from fukikae.registry.store import OPERATOR_BY

    found: list[tuple[int, str, str, Card]] = []
    for lang in sorted(reg.card_languages(character_id)):
        if lang == language:
            continue
        a = reg.card_for(character_id, lang)
        if a is None or a.chosen_by not in OPERATOR_BY:
            continue
        card = _clone_of(store, a.card_id)
        if card is not None and a.card_id is not None:
            found.append((0 if a.chosen_by == "studio" else 1, lang, a.card_id, card))
    if not found:
        return None
    _, lang, cid, card = min(found, key=lambda f: (f[0], f[1]))
    return lang, cid, card


def _proposal(
    ws: Workspace, cfg: Config, character_id: int, language: str
) -> tuple[list[RefSpan], Tier]:
    """The automatic rules' references for the character, computed without assigning
    (`cards.auto.find_references`, then its own cues' source spans for a character with no
    turn anywhere, `cards.auto.cue_references`)."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import cue_references, find_references
    from fukikae.models import DubCues

    gender, narrator = _character_state(ws, character_id)
    episodes = ready_episodes(ws)
    found = find_references(
        ws, episodes, character_id, cfg.source_language, gender=gender, narrator=narrator
    )
    if found.picks and found.episode is not None:
        refs = [RefSpan(episode=found.episode, start=p.start, end=p.end) for p in found.picks]
        return refs, found.tier or floor_meter(refs)["tier"]
    best: tuple[str, list[Any]] | None = None
    for e in episodes:
        work = ws.work_dir(e) / "06-translate"
        paths = [work / f"cues.{language}.json", *sorted(work.glob("cues.*.json"))]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            continue
        picks = cue_references(ws, e, character_id, read_json(path, DubCues).cues)
        secs = sum(p.seconds for p in picks)
        if picks and (best is None or secs > sum(p.seconds for p in best[1])):
            best = (e, picks)
    if best is None:
        raise DraftError(
            f"character {character_id} has no speech of its own in any ready episode to clone "
            "(D41: never a stock voice)"
        )
    return [RefSpan(episode=best[0], start=p.start, end=p.end) for p in best[1]], "thin"


def open_draft(
    ws: Workspace | Path,
    cfg: Config,
    character_id: int,
    language: str,
    *,
    offered: Sequence[Candidate] | None = None,
) -> Draft:
    """A new draft for the character and target language, version 1 from its current clone
    card; else from an accepted card of another language (copied as ``voices derive`` does);
    else from the automatic proposal (see the module notes). Not saved (`save_draft`); the
    registry is only read."""
    from fukikae.cards.auto import clone_backend
    from fukikae.cards.store import CardStore
    from fukikae.config import ConfigError

    w = _workspace(ws)
    language = language_of(language)
    store = CardStore(w.root)
    _character_state(w, character_id)
    with _registry(w) as reg:
        cur = reg.card_for(character_id, language)
        card = _clone_of(store, cur.card_id if cur is not None else None)
        other = None if card is not None else _accepted_elsewhere(reg, store, character_id,
                                                                    language)  # fmt: skip
    derived_from: str | None = None
    unlisted = 0
    if card is not None and cur is not None and cur.card_id is not None:
        base, card_id = "card", cur.card_id
        refs, unlisted = _spans_of(card)
        backend, params, tier = card.voice.backend, _params_of(card), _tier_of(card, refs)
    elif other is not None:
        derived_from, old_id, old = other
        base, card_id = "derived", _derive(store, old, old_id, language, character_id)
        refs, unlisted = _spans_of(old)
        backend, params, tier = old.voice.backend, _params_of(old), _tier_of(old, refs)
    else:
        try:
            name = clone_backend(cfg)
        except ConfigError as e:
            raise DraftError(str(e)) from None
        if name is None:
            raise DraftError(
                "no [tts] clone_backend is set, so no clone can be proposed (D41: never a "
                "stock voice)"
            )
        cls = clone_class(name)
        refs, tier = _proposal(w, cfg, character_id, language)
        base, backend, params = "proposed", name, _defaults(cls)
        card_id = _build_card(w, cfg, character_id, language, refs, cls, {}, tier)
    offered = candidates(w, cfg, character_id) if offered is None else offered
    version = DraftVersion(
        n=1, backend=backend, references=refs, params=params, card_id=card_id,
        overrides=overrides(refs, offered), created=_now(),
        digest=_digest(w, character_id, language), tier=tier, unlisted=unlisted,
    )  # fmt: skip
    return Draft(
        character_id=character_id, language=language, base=base, derived_from=derived_from,
        versions=[version],
    )  # fmt: skip


__all__ = [
    "CHECKS",
    "DraftError",
    "NoRegistry",
    "UnknownCharacter",
    "add_version",
    "backends",
    "candidates",
    "check_own_speech",
    "check_params",
    "clone_class",
    "draft_card",
    "draft_path",
    "floor_meter",
    "load_draft",
    "open_draft",
    "overrides",
    "ready_episodes",
    "save_draft",
    "voice_key_spec",
]
