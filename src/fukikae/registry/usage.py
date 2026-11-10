"""The actors other episodes' ``assignments.json`` still name.

A rowless remainder (a names-used episode's unnamed speech with no embedding rows of its own,
Ruling R1) takes its parent cluster's acoustic match and adds no exemplar to that actor. When
every exemplar the actor has comes from another episode, rebuilding that episode (the
characters stage, or `registry resolve`) leaves it with none, and pruning it would leave the
remainder's episode naming an actor and character that no longer exist. Both prune only
orphans outside this set.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from fukikae.artifacts import read_json
from fukikae.models import Assignments


def actors_in_use(workspace_root: Path, skip: Collection[str]) -> set[int]:
    """Actor ids named by the assignments of every episode of the workspace except ``skip``
    (the episodes being rebuilt, whose assignments are about to be rewritten). An unreadable
    file names nothing."""
    used: set[int] = set()
    episodes = workspace_root / ".fukikae" / "episodes"
    for path in sorted(episodes.glob("*/05-characters/assignments.json")):
        if path.parent.parent.name in skip:
            continue
        try:
            a = read_json(path, Assignments)
        except (OSError, ValueError):
            continue
        used |= {c.actor_id for c in a.clusters if c.actor_id is not None}
    return used
