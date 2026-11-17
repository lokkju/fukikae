"""Reference candidates and voice drafts (Task 18): the voice-tuning loop's data.

`candidates` lists a character's own speech across the ready episodes as references: every
candidate the automatic rules judge eligible (``rank`` in the CD5 order, name-confirmed turns
first) and every turn they refuse, with each check's verdict (`fukikae.cards.select.REASONS`
and the A1 gates, measured as `fukikae.cards.auto.character_candidates` measures them: the
operator's gender beats the estimate, and name-confirmed turns are gated by class only). A
character with no turn anywhere is offered its own spoken cues' source spans, what the
cue-thin tier clones it from (theme songs and tagged singing ranked last, as
`fukikae.cards.auto.cue_references` leaves them out). Nothing else is ever offered: D41, every
voice is a clone of the character's own speech. An episode whose turns are out of date for
the character (`fukikae.studio.apply.unready_episodes`: its names or characters stage stale,
or older than an applied studio decision on the character, or its names newer than its
assignments) offers nothing, and a reference in it is refused until its characters stage
reruns.

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

Every reference must be the character's own clean speech (`check_own_speech`), the rule
accept applies (`fukikae.studio.apply._own_speech`): inside its own turns and overlapping no
other speaker, whatever its seconds. The one exemption is a version whose references are
exactly those of a thin card the pipeline itself built for the character
(`fukikae.studio.apply.thin_origin`); the version and its card record that card as
``thin_origin``. The tier (floor, short or thin, by seconds) is a separate property: a
version thin by its seconds is held to the clean rule like any other.

`accept_decision` turns a saved version into the accept the server applies (Task 20): its
card, references, overrides and audition count, and the digest of the language's card
assignment taken when the version was made. A second target language's draft opens from the
character's studio-accepted card first, then any other operator-chosen clone card
(``base = "derived"``, ``derived_from`` naming the language); an imported or automatic card is
never a starting point for another language.

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
from fukikae.studio.model import AcceptDecision, Candidate, Draft, DraftVersion, RefSpan
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
# The most references a version may name, the shortest one and the longest one.
MAX_REFERENCES = 12
MIN_REFERENCE_S = 0.2
MAX_REFERENCE_S = 12.0
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


class UnknownVersion(DraftError):  # noqa: N818 (a 404)
    """No such draft, or no such version of it."""


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
    """The saved draft, or None when there is none. ValueError when the file is another
    character's or language's draft (copied or renamed into place)."""
    from fukikae.artifacts import read_json

    path = draft_path(ws, character_id, language)
    if not path.is_file():
        return None
    draft = read_json(path, Draft)
    want = (int(character_id), language_of(language))
    if (draft.character_id, draft.language) != want:
        raise ValueError(
            f"{path.name} holds the draft of character {draft.character_id} "
            f"({draft.language}), not of character {want[0]} ({want[1]})"
        )
    return draft


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


def unready(ws: Workspace | Path, character_id: int, episodes: Iterable[str]) -> dict[str, str]:
    """The episodes of ``episodes`` whose turns are out of date for the character, each with
    why and the command that brings it back (`fukikae.studio.apply.unready_episodes`)."""
    from fukikae.studio.apply import LogUnreadable, unready_episodes

    w = _workspace(ws)
    eps = list(episodes)
    if not eps:
        return {}
    try:
        with _registry(w) as reg:
            return unready_episodes(w, reg, eps, character_id)
    except LogUnreadable as e:
        raise DraftError(str(e)) from None


def speech_episodes(ws: Workspace | Path, character_id: int) -> list[str]:
    """The ready episodes (`ready_episodes`) whose turns are current for the character
    (`unready`): the episodes candidates and proposals read."""
    w = _workspace(ws)
    eps = ready_episodes(w)
    stale = unready(w, character_id, eps)
    return [e for e in eps if e not in stale]


# -- candidates -----------------------------------------------------------------------------


def _candidate_inputs(ws: Workspace) -> list[Path]:
    """Every file `candidates` reads, or reads to decide which episodes it reads."""
    from fukikae.studio.apply import decisions_log
    from fukikae.studio.season import episode_ids
    from fukikae.transcripts import transcript_path

    reg = ws.registry_path
    paths = [ws.root / "fukikae.toml", reg, reg.with_name(reg.name + "-wal"), decisions_log(ws)]
    for e in episode_ids(ws):
        work = ws.work_dir(e)
        sep, names = work / "01-separate", work / "04c-names"
        paths += [
            work / "manifest.json", work / "05-characters" / "assignments.json",
            transcript_path(work), work / "04b-tag" / "tags.json", sep / "vocals.wav",
            sep / "background.wav", sep / "vocals_16k.wav", names / "names.json",
            names / "turns.json", *sorted((work / "06-translate").glob("cues.*.json")),
        ]  # fmt: skip
    return paths


def candidates_digest(ws: Workspace | Path) -> str:
    """A cheap digest of what `candidates` reads (each file's size and modification time:
    the registry and decisions log, the config, and per episode its manifest, assignments,
    transcript, tags, stems, names and cues), for a cache to compare. Reads no file."""
    import hashlib

    h = hashlib.sha256()
    for path in _candidate_inputs(_workspace(ws)):
        try:
            st = path.stat()
            h.update(f"{path}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
        except OSError:
            h.update(f"{path}\0-\n".encode())
    return h.hexdigest()


def _band(median: float | None) -> Gender | None:
    from fukikae.cards.select import gender_of

    return None if median is None else gender_of(median)


def _gate_verdicts(gates: Sequence[str], measured: bool) -> dict[str, bool]:
    return {g: g not in gates for g in GATES} if measured else {}


def _overlaps(start: float, end: float, spans: Iterable[tuple[float, float]]) -> bool:
    return any(a < end and start < b for a, b in spans)


def _subtract(
    start: float, end: float, cut: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``start``-``end`` less every span of ``cut``, as the pieces left."""
    pieces = [(start, end)]
    for a, b in sorted(cut):
        nxt: list[tuple[float, float]] = []
        for x, y in pieces:
            if b <= x or a >= y:
                nxt.append((x, y))
                continue
            if a > x:
                nxt.append((x, a))
            if b < y:
                nxt.append((b, y))
        pieces = nxt
    return pieces


def _report_items(
    rep: CandidateReport, skip: Sequence[tuple[float, float]] = (), transcript: Any = None
) -> list[tuple[Any, tuple[str, ...], tuple[str, ...], bool, bool]]:
    """(pick, pre-gate reasons, gates, eligible, confirmed) for a report's candidates and
    refused turns. A pick overlapping ``skip`` (the name-confirmed turns another report
    covers) is split: the pieces outside ``skip`` stay, refused with ``split`` (they were
    judged only as part of the whole), so no unconfirmed turn is lost."""
    from fukikae.cards.select import RefPick, turn_text

    out: list[tuple[Any, tuple[str, ...], tuple[str, ...], bool, bool]] = []
    rows = [(v.pick, (), tuple(v.gates), v.passed) for v in rep.verdicts]
    rows += [(r.pick, tuple(r.reasons), tuple(r.gates), False) for r in rep.refused]
    for pick, reasons, gates, passed in rows:
        if not _overlaps(pick.start, pick.end, skip):
            out.append((pick, reasons, gates, passed, rep.confirmed))
            continue
        for a, b in _subtract(pick.start, pick.end, skip):
            if b - a < MIN_REFERENCE_S:
                continue
            text = turn_text(transcript, a, b) if transcript is not None else ""
            piece = RefPick(round(a, 3), round(b, 3), text, pick.snr_db)
            out.append((piece, ("split", *reasons), (), False, rep.confirmed))
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
    other turns as the acoustic tier does; an acoustic candidate that overlaps a confirmed
    turn is split around it (`_report_items`). Episodes whose turns are out of date for the
    character are left out (`speech_episodes`). Assigns nothing and cuts no clip (the clips are
    named, not cut); only the reference selection's own caches (pitch tracks) may fill."""
    from fukikae.cards.auto import character_candidates
    from fukikae.cards.select import rank_key
    from fukikae.studio.season import sample_name

    w = _workspace(ws)
    gender, narrator = _character_state(w, character_id)
    episodes = speech_episodes(w, character_id)
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
            art = loaded.get(e)
            tr = art.transcript if art is not None else None
            items += [(e, *x) for x in _report_items(acoustic[e], mine, tr)]
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
    (joined as accept joins them), longest first. A span over a theme song or tagged singing
    (the episode's reference exclusions, what `fukikae.cards.auto.cue_references` leaves out)
    is refused with ``excluded``, unless every span is: then all are offered, as the thin
    tier falls back to them."""
    from fukikae.artifacts import read_json
    from fukikae.cards.auto import _episode_exclusions
    from fukikae.cards.select import turn_text
    from fukikae.models import Tags, Transcript
    from fukikae.studio.apply import _cue_spans
    from fukikae.transcripts import transcript_path

    cache: dict[str, list[tuple[float, float]]] = {}
    spans: list[tuple[str, float, float, str, bool]] = []
    for e in episodes:
        work = ws.work_dir(e)
        found = _cue_spans(work, character_id)
        if not found:
            continue
        tr_path, tags_path = transcript_path(work), work / "04b-tag" / "tags.json"
        tr = read_json(tr_path, Transcript) if tr_path.exists() else None
        tags = read_json(tags_path, Tags) if tags_path.exists() else None
        theme = _episode_exclusions(ws, e, tags, cache)
        spans += [
            (e, a, b, turn_text(tr, a, b) if tr else "", _overlaps(a, b, theme)) for a, b in found
        ]
    if all(x[4] for x in spans):
        spans = [(e, a, b, t, False) for e, a, b, t, _ in spans]
    spans.sort(key=lambda s: (s[4], -(s[2] - s[1]), s[0], s[1]))
    out: list[Candidate] = []
    for i, (e, a, b, text, excluded) in enumerate(spans):
        out.append(Candidate(
            episode=e, start=a, end=b, seconds=round(b - a, 3), transcript=text, snr=None,
            median=None, gender=None, rank=None if excluded else i,
            verdicts={"excluded": False} if excluded else {},
            reasons=["excluded"] if excluded else [], clip=sample_name(ws, e, a, b),
            source="cue",
        ))  # fmt: skip
    return out


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
    doesn't list in its ``voice_keys`` (or no draft parameter), a value of the wrong type, or
    one outside the card parser's own bounds (the seed's 32 bits, `cards.card.VOICE_RANGES`),
    checked here so that nothing is cut for a version that would be refused."""
    from fukikae.cards.card import MODE_FIELDS, SEED_MAX, VOICE_RANGES, CardError
    from fukikae.cards.card import _float as card_float

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
            if not 0 <= value <= SEED_MAX:
                raise DraftError(f"seed must be 0 to {SEED_MAX} (32 bits)")
            out[key] = value
        elif key in MODE_FIELDS:
            if value not in MODE_FIELDS[key]:
                raise DraftError(f"{key} must be one of {', '.join(MODE_FIELDS[key])}")
            out[key] = value
        else:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DraftError(f"{key} must be a number, not {str(value)[:40]!r}")
            try:
                number = float(value)
            except OverflowError:
                raise DraftError(f"{key} is too large a number") from None
            if not math.isfinite(number):
                raise DraftError(f"{key} must be finite")
            try:
                out[key] = card_float(number, key, "[voice]", **VOICE_RANGES.get(key, {}))
            except CardError as e:
                raise DraftError(str(e)) from None
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


def _within(r: RefSpan, spans: Iterable[tuple[float, float]]) -> bool:
    return any(a - SPAN_SLACK_S <= r.start and r.end <= b + SPAN_SLACK_S for a, b in spans)


def check_own_speech(
    ws: Workspace | Path, character_id: int, refs: Sequence[RefSpan], *, exempt: bool = False
) -> None:
    """DraftError unless every reference is the character's own clean speech, the rule
    accept applies (`fukikae.studio.apply._own_speech`). Its episode must be current for the
    character (`unready`: else "run `fukikae run <ws> <ep> --to characters` first"); it must
    lie inside the character's own turns there (joined as accept joins them,
    `fukikae.studio.apply._own_turns`) and overlap no other speaker's turn (the refusal names
    the reference and the speaker), whatever its seconds. Only in an episode where the
    character holds no turn do its own spoken cues' source spans count instead. ``exempt``
    (the exact references of the pipeline's thin card, `fukikae.studio.apply.thin_origin`)
    lifts the overlap rule alone."""
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments
    from fukikae.studio.apply import _cue_spans, _own_turns

    w = _workspace(ws)
    stale = unready(w, character_id, {r.episode for r in refs})
    if stale:
        raise DraftError(
            f"not current for character {character_id}'s references: "
            + "; ".join(stale[e] for e in sorted(stale))
        )
    seen: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float, str]],
                          list[tuple[float, float]]]] = {}  # fmt: skip
    bad: list[str] = []
    dirty: list[str] = []
    for r in refs:
        if r.episode not in seen:
            path = w.work_dir(r.episode) / "05-characters" / "assignments.json"
            mine: list[tuple[float, float]] = []
            others: list[tuple[float, float, str]] = []
            if path.is_file():
                asg = read_json(path, Assignments)
                if asg.turns is not None:
                    mine = _own_turns(asg, character_id)[0]
                    whose = {a.cluster: a for a in asg.clusters}
                    own = {
                        a.cluster for a in asg.clusters
                        if a.character_id == character_id and a.status not in ("extra", "review")
                    }  # fmt: skip
                    for t in asg.turns:
                        if t.speaker in own:
                            continue
                        a = whose.get(t.speaker)
                        who = (
                            f"character {a.character_id} ({t.speaker})"
                            if a is not None and a.character_id is not None
                            else f"cluster {t.speaker}"
                        )
                        others.append((float(t.start), float(t.end), who))
            cues = [] if mine else _cue_spans(w.work_dir(r.episode), character_id)
            seen[r.episode] = (mine, others, cues)
        mine, others, cues = seen[r.episode]
        if not mine:
            if not _within(r, cues):
                bad.append(r.label())
            continue
        if not _within(r, mine):
            bad.append(r.label())
            continue
        over = sorted({
            who for a, b, who in others if a < r.end - SPAN_SLACK_S and r.start + SPAN_SLACK_S < b
        })  # fmt: skip
        if over and not exempt:
            dirty.append(f"{r.label()} overlaps {', '.join(over)}")
    if bad:
        raise DraftError(
            f"not character {character_id}'s own speech (D41: a draft clones only the "
            f"character's own turns, or its spoken cues in an episode where it holds no turn): "
            f"{', '.join(bad)}"
        )
    if dirty:
        raise DraftError(
            f"not character {character_id}'s voice alone (D41: a reference overlaps no other "
            "speaker, whatever its seconds; only the exact references of the pipeline's own "
            f"thin card are exempt): {'; '.join(dirty)}"
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
        if r.end - r.start > MAX_REFERENCE_S + 1e-9:
            raise DraftError(
                f"{r.label()}: a reference is at most {MAX_REFERENCE_S:g} s; choose a shorter "
                "stretch of it"
            )
        if r.episode not in ready:
            raise DraftError(f"{r.episode} has no assignments and vocals to cut a reference from")
    if (problem := _duplicates(refs)) is not None:
        raise DraftError(problem)


def _duplicates(refs: Sequence[RefSpan]) -> str | None:
    """Why ``refs`` count a piece of audio more than once, or None."""
    ordered = sorted(refs, key=lambda r: (r.episode, r.start, r.end))
    for a, b in zip(ordered, ordered[1:], strict=False):
        if a.episode == b.episode and b.start < a.end - 1e-9:
            return (
                f"a version names the same audio more than once (duplicate, nested or "
                f"overlapping references): {a.label()} and {b.label()}"
            )
    return None


def accept_problems(
    ws: Workspace | Path, character_id: int, card_id: str, refs: Sequence[RefSpan]
) -> list[str]:
    """What accept would refuse a version's card for, as accept says it
    (`fukikae.studio.apply._own_speech`: strict own speech, the thin exemption, stale
    episodes, the cue-span rule), and any audio its references name more than once; empty
    when accept would take it. Changes nothing."""
    from fukikae.cards.card import CardError
    from fukikae.cards.store import CardStore
    from fukikae.studio.apply import DecisionRefused, _own_speech

    w = _workspace(ws)
    out: list[str] = []
    if (problem := _duplicates(refs)) is not None:
        out.append(problem)
    try:
        card = CardStore(w.root).get(card_id)
        with _registry(w) as reg:
            _own_speech(w, reg, character_id, card_id, card)
    except DecisionRefused as e:
        out.append(str(e))
    except CardError as e:
        out.append(f"card {card_id} can't be read: {e}")
    return out


def _ref_set(refs: Sequence[RefSpan], carried: Sequence[Any]) -> frozenset[tuple[Any, ...]]:
    """The references as `fukikae.studio.apply.reference_set` sees a card's."""
    spans = {(r.episode, round(r.start, 3), round(r.end, 3)) for r in refs}
    return frozenset(spans | {("clip", c.clip) for c in carried})


def _thin_origin(ws: Workspace, character_id: int, refs: Sequence[RefSpan],
                 carried: Sequence[Any] = ()) -> str | None:  # fmt: skip
    """The pipeline thin card whose references are exactly these, or None."""
    from fukikae.cards.store import CardStore
    from fukikae.studio.apply import thin_origin

    with _registry(ws) as reg:
        return thin_origin(reg, CardStore(ws.root), character_id, _ref_set(refs, carried))


def _carried(ws: Workspace, draft: Draft) -> list[Any]:
    """The references with no origin span of the draft's latest version's card: carried into
    the next version unless the operator removes them."""
    from fukikae.cards.card import CardError
    from fukikae.cards.store import CardStore

    if not draft.versions:
        return []
    store = CardStore(ws.root)
    cid = draft.versions[-1].card_id
    if not store.has(cid):
        return []
    try:
        card = store.get(cid)
    except CardError:
        return []
    return [r for r in card.voice.references if not _has_origin(r)]


def _has_origin(r: Any) -> bool:
    o = r.origin or {}
    try:
        str(o["episode"])
        float(o["start"])
        float(o["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _build_card(
    ws: Workspace,
    cfg: Config,
    character_id: int,
    language: str,
    refs: Sequence[RefSpan],
    cls: type[Synthesizer],
    params: Mapping[str, Any],
    tier: Tier | None,
    *,
    thin_origin: str | None = None,
    carried: Sequence[Any] = (),
) -> tuple[str, Tier]:
    """The card for ``refs`` (and the ``carried`` origin-less references after them), put in
    the store; its id and tier. The parameters are checked before anything is cut."""
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
    meta: dict[str, Any] = {**card.meta, "character": character_id}
    if thin_origin is not None:
        meta["thin_origin"] = thin_origin
    card = replace(
        card,
        voice=replace(card.voice, references=(*card.voice.references, *carried), **norm),
        meta=meta,
    )
    try:
        parse_card(dump_card(card))  # the card parser's bounds (seed bits, emotion 0 to 1, ...)
        validate_for_backend(card, cls)
    except CardError as e:
        raise DraftError(str(e)) from None
    built = meta.get("reference_tier")
    if built not in ("floor", "short", "thin"):
        built = floor_meter(refs)["tier"]
    return store.put(card), built


def _draft(
    w: Workspace,
    cfg: Config,
    draft: Draft,
    references: Sequence[RefSpan],
    backend: str,
    params: Mapping[str, Any],
    keep_unlisted: bool,
) -> tuple[str, Tier, str | None, int]:
    """(card id, tier, thin origin, carried origin-less references) for a new version."""
    cls = clone_class(backend)
    check_params(cls, params)
    _check_refs(w, references)
    carried = _carried(w, draft) if keep_unlisted else []
    origin = _thin_origin(w, draft.character_id, references, carried)
    check_own_speech(w, draft.character_id, references, exempt=origin is not None)
    card_id, tier = _build_card(
        w, cfg, draft.character_id, draft.language, references, cls, params,
        "thin" if origin is not None else None, thin_origin=origin, carried=carried,
    )  # fmt: skip
    return card_id, tier, origin, len(carried)


def draft_card(
    ws: Workspace | Path,
    cfg: Config,
    draft: Draft,
    references: Sequence[RefSpan],
    backend: str,
    params: Mapping[str, Any],
    *,
    keep_unlisted: bool = True,
) -> str:
    """The clone card for ``references`` with ``backend`` and ``params``, built as
    ``cards.auto._clone`` builds a clone (a 24 kHz sidecar per reference cut from its own
    episode's vocals, its ASR text and origin; seed 1234 and the defaults unless ``params``
    sets them), put in the card store; its id. ``params`` may set only the backend's
    ``voice_keys``, within the card's bounds, checked before anything is cut. The references
    are distinct pieces of audio (no duplicate, nested or overlapping ones), each at most
    `MAX_REFERENCE_S`, and every one is the character's own clean speech
    (`check_own_speech`), unless they are exactly the references of the pipeline's own thin
    card for the character, which only lifts the overlap rule and keeps that card's thin tier
    (``[meta] thin_origin`` names it). Otherwise the tier is the references' own. The latest
    version's origin-less references are carried after them unless ``keep_unlisted`` is
    off."""
    w = _workspace(ws)
    return _draft(w, cfg, draft, references, backend, params, keep_unlisted)[0]


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
    keep_unlisted: bool = True,
) -> tuple[Draft, DraftVersion]:
    """A new version of the draft from ``references``, ``backend`` and ``params``
    (`draft_card`), its gated picks recorded as overrides (against ``offered``, by default
    `candidates`), its thin origin and carried origin-less references recorded, saved; the
    updated draft and the version."""
    w = _workspace(ws)
    refs = list(references)
    card_id, tier, origin, unlisted = _draft(w, cfg, draft, refs, backend, params, keep_unlisted)
    cls = clone_class(backend)
    offered = candidates(w, cfg, draft.character_id) if offered is None else offered
    version = DraftVersion(
        n=max((v.n for v in draft.versions), default=0) + 1, backend=backend, references=refs,
        params={**_defaults(cls), **check_params(cls, params)}, card_id=card_id,
        overrides=overrides(refs, offered), created=_now(),
        digest=_digest(w, draft.character_id, draft.language), tier=tier, unlisted=unlisted,
        thin_origin=origin,
    )  # fmt: skip
    draft = draft.model_copy(update={"versions": [*draft.versions, version]})
    save_draft(w, draft)
    return draft, version


def _defaults(cls: type[Synthesizer]) -> dict[str, Any]:
    """A clone card's value of each draft key the backend lists (seed 1234, the defaults)."""
    from fukikae.cards.card import Voice

    plain = Voice("", "", "", seed=CLONE_SEED)
    return {k: getattr(plain, k) for k in VOICE_PARAMS if k in cls.voice_keys}


def version_of(draft: Draft, n: int) -> DraftVersion:
    """Version ``n`` of the draft; `UnknownVersion` without one."""
    v = next((v for v in draft.versions if v.n == n), None)
    if v is None:
        raise UnknownVersion(f"draft {draft.name} has no version {n}")
    return v


def accept_decision(draft: Draft, n: int) -> AcceptDecision:
    """The accept of version ``n`` (Task 20): its card, references, overrides and audition
    count, and the digest of the language's card assignment taken when the version was made
    (`DraftVersion.digest`), so a pipeline re-pick since makes it stale. Key and external id
    are left for `fukikae.studio.apply.fill`; a version with no digest gives a decision with
    none, which `apply_decision` refuses unchecked."""
    v = version_of(draft, n)
    return AcceptDecision(
        op="accept", character=draft.character_id, language=draft.language,
        card_id=v.card_id, draft=draft.name, version=v.n, references=v.references,
        overrides=v.overrides, auditions=len(v.auditions), digest=v.digest,
    )  # fmt: skip


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


def _card_origin(ws: Workspace, character_id: int, card: Card) -> str | None:
    """The pipeline thin card whose references are exactly ``card``'s, or None."""
    from fukikae.cards.store import CardStore
    from fukikae.studio.apply import reference_set, thin_origin

    with _registry(ws) as reg:
        return thin_origin(reg, CardStore(ws.root), character_id, reference_set(card))


def _tier_of(card: Card, refs: Sequence[RefSpan]) -> Tier:
    t = card.meta.get("reference_tier")
    return t if t in ("floor", "short", "thin") else floor_meter(refs)["tier"]


def _derive(store: Any, old: Card, old_id: str, language: str, character_id: int,
            thin_origin: str | None = None) -> str:  # fmt: skip
    """``fukikae voices derive``'s copy of a card for another target language (no
    fingerprint, catchphrase renderings emptied, ``derived_from``, and ``thin_origin`` when
    its references are exactly the pipeline's thin card's), put in the store."""
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
              "character": character_id,
              **({"thin_origin": thin_origin} if thin_origin is not None else {})},
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
    episodes = speech_episodes(ws, character_id)
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
    else from the automatic proposal (see the module notes). Version 1 is checked by accept's
    rules and never refused: ``acceptable`` and ``accept_problems`` say what accept would
    refuse (`accept_problems`). Not saved (`save_draft`); the
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
    origin: str | None = None
    if card is not None and cur is not None and cur.card_id is not None:
        base, card_id = "card", cur.card_id
        refs, unlisted = _spans_of(card)
        backend, params, tier = card.voice.backend, _params_of(card), _tier_of(card, refs)
        origin = _card_origin(w, character_id, card)
    elif other is not None:
        derived_from, old_id, old = other
        origin = _card_origin(w, character_id, old)
        base = "derived"
        card_id = _derive(store, old, old_id, language, character_id, origin)
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
        origin = _thin_origin(w, character_id, refs)
        card_id, _ = _build_card(w, cfg, character_id, language, refs, cls, {}, tier,
                                 thin_origin=origin)  # fmt: skip
    offered = candidates(w, cfg, character_id) if offered is None else offered
    problems = accept_problems(w, character_id, card_id, refs)
    version = DraftVersion(
        n=1, backend=backend, references=refs, params=params, card_id=card_id,
        overrides=overrides(refs, offered), created=_now(),
        digest=_digest(w, character_id, language), tier=tier, unlisted=unlisted,
        thin_origin=origin, acceptable=not problems, accept_problems=problems,
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
    "UnknownVersion",
    "accept_decision",
    "accept_problems",
    "add_version",
    "backends",
    "candidates",
    "candidates_digest",
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
    "speech_episodes",
    "unready",
    "version_of",
    "voice_key_spec",
]
