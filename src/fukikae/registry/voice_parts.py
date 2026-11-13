"""Identities of an operator pitch split's voice parts (Task 15 review, controller ruling).

A pitch split (``<ws>/studio/splits/<episode>.json``) relabels a diarize cluster's unnamed
turns ``<cluster>~voice-f`` and ``<cluster>~voice-m``. The operator split it because it holds
two voices, so the parts never both stay with the parent's character, and neither inherits
the parent's pin by overlap in time. The characters stage and `registry resolve` both apply:

- exactly one part keeps the parent cluster's character (`voice_keeper`): the part in the
  character's stored register (``f`` keeps ``~voice-f``, ``m`` keeps ``~voice-m``); with no
  stored gender, the part with more seconds of turns (``~voice-f`` on a tie, as
  `fukikae.names.align.split_voices` breaks its own). A stored gender whose part the split
  didn't produce keeps no part on the character;
- every other part gets a new actor and character (status ``new``, ``split = "new"``), whatever
  it scores against the parent's actor, whose exemplars came from the mixed cluster, and
  whatever its seconds: a rowless part is never an extra;
- a part already holding its own character from an earlier run (pinned to its own label) keeps
  it, so reruns are stable; two parts pinned to one character are settled by the keeper rule.

Where the parent's character comes from is each caller's: the characters stage takes the
parent label's earlier assignment, else the parent's acoustic match over its parts' rows;
`registry resolve` takes the character its joint grouping gave the parts.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping

from fukikae.models import Turn
from fukikae.names.align import voice_label
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


def keeper_why(gender: str | None) -> str:
    return f"its stored gender {gender}" if gender in ("f", "m") else "more seconds"


__all__ = ["keeper_why", "label_seconds", "voice_keeper", "voice_parts"]
