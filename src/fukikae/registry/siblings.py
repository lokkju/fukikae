"""A rowless remainder's named siblings (Task 11 controller ruling, amending R1).

A remainder label (a names-used episode's unnamed speech in one diarize cluster) that kept no
embedding rows and has no pin can't be matched on its own. The characters stage and
`registry resolve` both decide it from the other sub-clusters of its parent diarize cluster:

- when one character holds at least ``SIBLING_SHARE`` of the parent's named seconds (the
  turns of its named sub-clusters, by the character each was assigned), the remainder takes
  that character (``follow``);
- when the parent has named sub-clusters but no character holds that share (they went to
  several characters, or to named extras), the remainder is ``extra`` (``split``);
- only when the parent has no named sub-clusters does it take the parent's acoustic match,
  as R1 had it (`sibling_verdict` returns None).

A named sub-cluster is one whose key is a name (``cls == "named"``); a role label names no
character (D29 #6), so a parent whose only keyed sub-clusters are role labels keeps R1's
acoustic path. A named sub-cluster that went extra (a key under ``[names] min_new_seconds``)
still counts its seconds in the parent's named total, held by no character.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from fukikae.models import ClusterAssignment
from fukikae.names.models import LabelInfo

SIBLING_SHARE = 0.6  # the Task 11 ruling's 60%, the same share as D29 #3's rename


@dataclass(frozen=True)
class SiblingVerdict:
    how: Literal["follow", "split"]
    character_id: int | None  # the character a ``follow`` takes; the top holder for ``split``
    share: float  # that character's share of the parent's named seconds
    named_seconds: float

    def line(self, label: str, parent: str, describe: str) -> str:
        """The report line: ``describe`` names the character (``character 9 (Natsu)``)."""
        if self.how == "follow":
            return (
                f"{label} follows its named siblings: {describe} holds {self.share:.0%} of "
                f"{parent}'s {self.named_seconds:.1f} s of named speech"
            )
        if self.character_id is None:
            return (
                f"{label} stays extra: {parent}'s {self.named_seconds:.1f} s of named speech "
                "went to no character"
            )
        return (
            f"{label} stays extra: {parent}'s {self.named_seconds:.1f} s of named speech is "
            f"split (most is {describe} at {self.share:.0%}, under {SIBLING_SHARE:.0%})"
        )


def has_named_siblings(label: str, labels: Mapping[str, LabelInfo]) -> bool:
    """Whether ``label``'s parent diarize cluster has a named (not role) sub-cluster."""
    parent = labels[label].parent
    return any(i.parent == parent and i.cls == "named" for i in labels.values())


def sibling_verdict(
    label: str,
    labels: Mapping[str, LabelInfo],
    assigned: Mapping[str, ClusterAssignment],
    seconds: Mapping[str, float],
) -> SiblingVerdict | None:
    """How the rowless remainder ``label`` goes by its named siblings' assignments
    (``assigned``, by label) and turn ``seconds`` (by label); None when its parent has no named
    sub-cluster (R1's acoustic match decides). A named sibling missing from ``assigned``
    counts as held by no character."""
    parent = labels[label].parent
    siblings = [lb for lb, i in labels.items() if i.parent == parent and i.cls == "named"]
    if not siblings:
        return None
    held: dict[int, float] = defaultdict(float)
    total = 0.0
    for lb in siblings:
        s = seconds.get(lb, 0.0)
        total += s
        a = assigned.get(lb)
        if a is not None and a.character_id is not None:
            held[a.character_id] += s
    if not held or total <= 0:
        return SiblingVerdict("split", None, 0.0, total)
    cid, s = min(held.items(), key=lambda kv: (-kv[1], kv[0]))
    share = s / total
    return SiblingVerdict("follow" if share >= SIBLING_SHARE else "split", cid, share, total)
