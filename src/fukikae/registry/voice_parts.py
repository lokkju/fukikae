"""Identities of an operator pitch split's voice parts (Task 15 review, controller ruling).

A pitch split (``<ws>/studio/splits/<episode>.json``) relabels a diarize cluster's unnamed
turns ``<cluster>~voice-f`` and ``<cluster>~voice-m``. Neither part inherits the parent's pin
by overlap in time. The characters stage and `registry resolve` both apply:

- a part with an identity of its own (its own label's earlier assignment: what the keeper rule
  gave it, an operator merge or remap, a name, a lock) keeps it in every path: characters
  reruns, ``--rematch`` and ``registry resolve``. Two parts on one character (only an operator
  merge does that) stay so, and a later stored-gender change never moves the keeper (the season
  index flags the conflict as ``gender`` instead);
- the keeper rule decides only for parts without one (the first run after a split): exactly one
  part keeps the parent cluster's character (`voice_keeper`): the part in the character's
  stored register (``f`` keeps ``~voice-f``, ``m`` keeps ``~voice-m``); with no stored gender,
  the part with more seconds of turns (``~voice-f`` on a tie, as
  `fukikae.names.align.split_voices` breaks its own). A stored gender whose part the split
  didn't produce keeps no part on the character;
- every other part never gets the parent's character: one matched to an existing character
  other than the keeper's (or a sibling's) keeps that match; otherwise it gets a new actor and
  character (status ``new``, ``split = "new"``), whatever its seconds: a rowless part is never
  an extra. A part whose siblings already have identities is decided the same way;
- an operator merge that puts a ``split = "new"`` part on its keeper's character clears the
  marker (`settle_split_markers`), and a pin onto a label that is not a voice part drops it.

Where the parent's character comes from is each caller's: the characters stage takes the
parent label's earlier assignment, else the parent's acoustic match over its parts' rows;
`registry resolve` takes the character its joint grouping gave the parts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping

from fukikae.models import ClusterAssignment, Turn
from fukikae.names.align import VOICE_PREFIX, voice_label
from fukikae.names.models import LabelInfo


def voice_parts(labels: Mapping[str, LabelInfo], present: Collection[str]) -> dict[str, list[str]]:
    """Each split parent cluster's voice parts (``cls == "voice"``) among ``present``, sorted."""
    out: dict[str, list[str]] = defaultdict(list)
    for lb, info in labels.items():
        if info.cls == "voice" and lb in present:
            out[info.parent].append(lb)
    return {p: sorted(v) for p, v in sorted(out.items())}


def label_seconds(turns: Iterable[Turn]) -> dict[str, float]:
    """Seconds of turns per label."""
    out: dict[str, float] = defaultdict(float)
    for t in turns:
        out[t.speaker] += t.end - t.start
    return dict(out)


def voice_keeper(
    parent: str, parts: Collection[str], gender: str | None, seconds: Mapping[str, float]
) -> str | None:
    """The one of ``parts`` that keeps the parent's character: the part in ``gender``'s
    register when the character has a stored gender (None when the split made no such part),
    else the part with more seconds, ``~voice-f`` on a tie."""
    if not parts:
        return None
    if gender in ("f", "m"):
        want = voice_label(parent, gender)
        return want if want in parts else None
    return max(sorted(parts), key=lambda lb: (round(seconds.get(lb, 0.0), 6), lb.endswith("-f")))


def settle_split_markers(clusters: list[ClusterAssignment]) -> list[ClusterAssignment]:
    """``clusters`` with the ``split = "new"`` marker cleared on every voice part that now
    holds the character a ``split = "kept"`` sibling (same parent cluster) holds: an operator
    merge put it back on the keeper's character, so it no longer needs an identity."""
    kept: dict[str, set[int]] = defaultdict(set)
    for c in clusters:
        if c.split == "kept" and c.character_id is not None:
            kept[_parent_of(c.cluster)].add(c.character_id)
    return [
        c.model_copy(update={"split": None})
        if c.split == "new" and c.character_id in kept.get(_parent_of(c.cluster), set())
        else c
        for c in clusters
    ]


def _parent_of(label: str) -> str:
    return label.split(f"~{VOICE_PREFIX}", 1)[0]


def keeper_why(gender: str | None) -> str:
    return f"its stored gender {gender}" if gender in ("f", "m") else "more seconds"


__all__ = ["keeper_why", "label_seconds", "settle_split_markers", "voice_keeper", "voice_parts"]
