"""Automatic voice assignment (D12): a card for each character that has none in the target
language, and stock voices for extras.

Only characters with no assignment get one, ``by = "auto"``, unlocked. An automatic
assignment never replaces a card: should one appear between the check and the write, the
registry refuses (``ProtectedAssignmentError``, rulings CD2 and CD3) and the existing card is
kept. Until a cloning backend is chosen (D15), every automatic card is a stock voice from the
gender-matched pool of the ``[tts]`` backend. Task 12 adds the clone branch.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from fukikae.cards.select import FEMALE_ABOVE, MALE_BELOW, gender_of, speaker_f0
from fukikae.cards.stock import check_stock_voice, pick_stock, revision, stock_card
from fukikae.cards.store import CardStore
from fukikae.registry.store import ProtectedAssignmentError

if TYPE_CHECKING:
    from fukikae.config import Config
    from fukikae.models import Assignments, DubCue
    from fukikae.registry.store import Registry

__all__ = [
    "ProtectedAssignmentError",
    "character_spans",
    "ensure_assignments",
    "extras_voices",
    "gender_thresholds",
    "tts_backend",
]

UNKNOWN_EXTRA = "unknown"


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
) -> list[str]:
    """Give every character in ``character_spans`` without a ``cfg.target_language`` card a
    stock card from its gender's pool, most speech first (so the main cast gets the first
    voices of each pool). Returns one report line per character handled."""
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
        try:
            reg.assign_card(cid, language, new, by="auto", locked=False, note=note)
        except ProtectedAssignmentError as e:
            lines.append(f"character {_label(reg, cid)}: kept its card ({e})")
            continue
        used.append(voice)
        lines.append(f"character {_label(reg, cid)}: {note}")
    return lines


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
