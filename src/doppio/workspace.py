"""Workspace layout types."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Episode:
    source: Path
    id: str
    work: Path
    workspace_root: Path
