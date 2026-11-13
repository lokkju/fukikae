"""The `[review]` gates of `fukikae run` (Task 21): `needs_identity` before translate and
`needs_voices` before tts, each the reasons an episode must stop, or none.

A main character of an episode is one of kind ``character`` or ``narrator`` (never
``ignored``, never an extras cluster) with at least ``[review] min_seconds`` of speech in it:
its turns in clusters whose status is not ``extra`` or ``review``, the seconds the season
index lists (`fukikae.studio.season`).

`needs_identity` stops for a main character never reviewed, or with a flag raised since its
review: a names flag on this episode (`names_flags`: a stale key decision or acoustics that
disagree with the subtitle names) or impure references on its automatic card in the target
language. Flags carry no time of their own, so "since" reads the time of the flag's source: a
names flag is as new as the newest of the episode's ``assignments.json``, names report and
names file, and an impure card as new as its assignment row (``assigned``). A card that
`replace_refusal` protects (curated, imported, locked) never reopens the gate: operator
state beats every automatic rule. A pitch-split suggestion (a bimodal cluster, or a split the
names stage skipped) never stops a run (controller ruling, 2026-11-13: half of Fairy Tail's
were one speaker shouting); when the gate stops for something else, each is named as a note.
A new part of an operator's pitch split is a new character, never reviewed, so it stops the
episode as such.

`needs_voices` stops for a main character whose card in the target language is not one an
automatic pick may not replace (``replace_refusal(card, "auto") is None``): no card, or an
automatic one.

Both are season-wide by construction: ``reviewed`` is a time on the character and an accepted
card is a registry assignment, so a later episode stops only for characters that are new,
newly flagged or still unvoiced. The gates only read: the registry, the episode's files and
the pitch cache (which a bimodal note may fill).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fukikae.workspace import Episode, Workspace

if TYPE_CHECKING:
    from fukikae.config import Config
    from fukikae.models import Assignments, Turn
    from fukikae.registry.store import Character, Registry

MAIN_KINDS = ("character", "narrator")
# The stage each gate stops before, and its `[review]` key.
GATE_STAGES = {"translate": "identity", "tts": "voices"}
SPLIT_NOTE = "a split suggestion never stops a run"


def _assignments(ep: Episode) -> Assignments | None:
    from fukikae.artifacts import read_json
    from fukikae.models import Assignments

    path = ep.work / "05-characters" / "assignments.json"
    return read_json(path, Assignments) if path.exists() else None


def _speech(asg: Assignments) -> dict[int, dict[str, list[Turn]]]:
    """Character -> cluster -> its turns, extras clusters left out (as the season index)."""
    from fukikae.studio.season import EXTRA_STATUSES

    who = {a.cluster: a.character_id for a in asg.clusters
           if a.character_id is not None and a.status not in EXTRA_STATUSES}  # fmt: skip
    out: dict[int, dict[str, list[Turn]]] = defaultdict(lambda: defaultdict(list))
    for t in sorted(asg.turns or [], key=lambda t: (t.start, t.end)):
        cid = who.get(t.speaker)
        if cid is not None:
            out[cid][t.speaker].append(t)
    return out


def episode_seconds(ep: Episode) -> dict[int, float]:
    """Character -> its seconds of speech in the episode (none before the characters stage)."""
    asg = _assignments(ep)
    if asg is None:
        return {}
    return {cid: sum(t.end - t.start for ts in cl.values() for t in ts)
            for cid, cl in _speech(asg).items()}  # fmt: skip


def main_characters(ep: Episode, reg: Registry, cfg: Config) -> list[int]:
    """The episode's main characters (the module docstring), in id order."""
    kinds = {c.id: c.kind for c in reg.characters()}
    return sorted(cid for cid, s in episode_seconds(ep).items()
                  if kinds.get(cid) in MAIN_KINDS and s >= cfg.review.min_seconds)  # fmt: skip


def _label(c: Character) -> str:
    return f"character {c.id} ({c.name or 'unnamed'})"


def _after(when: datetime | str, reviewed: str) -> bool:
    t = datetime.fromisoformat(when) if isinstance(when, str) else when
    return t > datetime.fromisoformat(reviewed)


def _mtime(paths: list[Path]) -> datetime | None:
    times = [p.stat().st_mtime_ns for p in paths if p.exists()]
    return datetime.fromtimestamp(max(times) / 1e9, UTC) if times else None


def names_flags(ws: Workspace, cfg: Config, ep: Episode, reg: Registry) -> dict[int, list[str]]:
    """Character -> the texts of its ``names`` flags on this episode, as the season index
    raises them (`season._names_evidence`)."""
    from fukikae.studio import season

    if _assignments(ep) is None:
        return {}
    loaded = season._load(ws, cfg, ep.id)
    out: dict[int, list[str]] = {}
    for cid, ev in season._names_evidence(ws, {ep.id: loaded}, reg).items():
        texts = [f.text for f in ev.flags if f.kind == "names" and f.episode == ep.id]
        if texts:
            out[cid] = texts
    return out


def _names_time(ep: Episode) -> datetime | None:
    """When the episode's names flags last could have changed: its newest names input."""
    from fukikae.names.inputs import names_path, names_report_path

    work = ep.work
    return _mtime([work / "05-characters" / "assignments.json", names_report_path(work),
                   names_path(work)])  # fmt: skip


def _new_flags(
    ws: Workspace, cfg: Config, ep: Episode, reg: Registry, chars: list[Character]
) -> list[str]:
    """The flags raised since each reviewed character's review, as reasons."""
    from fukikae.cards import auto
    from fukikae.cards.store import CardStore
    from fukikae.registry.store import replace_refusal

    lang = cfg.target_language
    out: list[str] = []
    names = names_flags(ws, cfg, ep, reg)
    since = _names_time(ep)
    ids = [c.id for c in chars]
    impure = auto.impure_cards(ws, reg, CardStore(ws.root), lang, ids,
                               genders=auto.operator_genders(reg))  # fmt: skip
    for c in chars:
        assert c.reviewed is not None
        if since is not None and _after(since, c.reviewed):
            out += [
                f"{_label(c)}: {text} (raised since its review)" for text in names.get(c.id, [])
            ]
        imp = impure.get(c.id)
        a = reg.card_for(c.id, lang)
        if imp is None or a is None or a.card_id != imp.card_id:
            continue
        if replace_refusal(a, "auto") is not None:
            continue  # the operator's card: flagged in the studio, never a stop
        if _after(a.assigned, c.reviewed):
            out.append(f"{_label(c)}: its {lang} card {a.card_id}, picked since its review, has "
                       f"references that fail the reference gates ({imp.phrase})")  # fmt: skip
    return out


def bimodal_notes(ws: Workspace, ep: Episode, characters: list[int]) -> list[str]:
    """One line per cluster of ``characters`` in the episode whose turns are bimodal (a
    pitch-split suggestion), with each register's seconds; none without the vocals stem."""
    from fukikae.pitch import cached_track
    from fukikae.studio.season import bimodal_clusters

    asg = _assignments(ep)
    vocals = ep.work / "01-separate" / "vocals.wav"
    if asg is None or not vocals.is_file():
        return []
    speech = _speech(asg)
    out: list[str] = []
    for cid in characters:
        clusters = {cl: list(ts) for cl, ts in speech.get(cid, {}).items()}
        turns = sorted((t for ts in clusters.values() for t in ts), key=lambda t: (t.start, t.end))
        if not turns:
            continue
        track = cached_track(ws.root, vocals, [(t.start, t.end) for t in turns])
        for cl, secs in bimodal_clusters(track, clusters).items():
            out.append(
                f"{ep.id} {cl} (character {cid}): {secs['m']:.1f} s low, "
                f"{secs['f']:.1f} s high; a pitch split may separate two voices"
            )
    return out


def needs_identity(ep: Episode, cfg: Config, reg: Registry) -> list[str]:
    """Why the episode must stop before translate (the module docstring); empty: it may go
    on. Notes on split suggestions follow the reasons, only when there are reasons."""
    ws = Workspace(ep.workspace_root)
    chars = {c.id: c for c in reg.characters()}
    mains = main_characters(ep, reg, cfg)
    reasons = [f"{_label(chars[cid])}: never reviewed" for cid in mains
               if chars[cid].reviewed is None]  # fmt: skip
    reviewed = [chars[cid] for cid in mains if chars[cid].reviewed is not None]
    if reviewed:
        reasons += _new_flags(ws, cfg, ep, reg, reviewed)
    if reasons:
        reasons += [f"note: {n} ({SPLIT_NOTE})" for n in bimodal_notes(ws, ep, mains)]
    return reasons


def needs_voices(ep: Episode, cfg: Config, reg: Registry) -> list[str]:
    """Why the episode must stop before tts: each main character without a curated,
    imported or locked card in the target language; empty: it may go on."""
    from fukikae.registry.store import replace_refusal

    lang = cfg.target_language
    chars = {c.id: c for c in reg.characters()}
    out: list[str] = []
    for cid in main_characters(ep, reg, cfg):
        a = reg.card_for(cid, lang)
        if replace_refusal(a, "auto") is not None:
            continue
        what = (f"no {lang} card" if a is None or a.card_id is None
                else f"its {lang} card {a.card_id} is automatic ({a.chosen_by})")  # fmt: skip
        out.append(f"{_label(chars[cid])}: {what}; accept a voice for it")
    return out


def _with_registry(
    check: Callable[[Episode, Config, Registry], list[str]],
) -> Callable[[Episode, Config], list[str]]:
    def gate(ep: Episode, cfg: Config) -> list[str]:
        from fukikae.registry.store import Registry

        path = Workspace(ep.workspace_root).registry_path
        if not path.exists():  # no characters yet: nothing to review
            return []
        with Registry(path) as reg:
            return check(ep, cfg, reg)

    gate.__name__ = check.__name__
    return gate


identity_gate = _with_registry(needs_identity)
voices_gate = _with_registry(needs_voices)


def gates_for(cfg: Config) -> dict[str, Callable[[Episode, Config], list[str]]]:
    """The gates `[review]` turns on, by the stage each stops before (`job.run_stages`)."""
    out: dict[str, Callable[[Episode, Config], list[str]]] = {}
    if cfg.review.identity:
        out["translate"] = identity_gate
    if cfg.review.voices:
        out["tts"] = voices_gate
    return out


__all__ = [
    "GATE_STAGES",
    "bimodal_notes",
    "episode_seconds",
    "gates_for",
    "identity_gate",
    "main_characters",
    "names_flags",
    "needs_identity",
    "needs_voices",
    "voices_gate",
]
