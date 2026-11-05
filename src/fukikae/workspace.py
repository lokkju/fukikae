"""Workspace layout and episode resolution."""

import re
from dataclasses import dataclass
from pathlib import Path

from fukikae.config import Config, load_config

VIDEO_SUFFIXES = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".webm", ".ts"}
_CODE = re.compile(r"[Ss](\d+)[Ee](\d+)")


@dataclass(frozen=True)
class Episode:
    source: Path
    id: str
    work: Path
    workspace_root: Path


def episode_id(path: Path) -> str:
    """`S01E01` from the filename, else the slugified stem."""
    m = _CODE.search(path.name)
    if m:
        return f"S{int(m[1]):02d}E{int(m[2]):02d}"
    return re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-") or "episode"


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def config(self) -> Config:
        return load_config(self.root)

    def work_dir(self, episode_id: str) -> Path:
        return self.root / ".fukikae" / "episodes" / episode_id

    @property
    def registry_path(self) -> Path:
        return self.root / ".fukikae" / "registry" / "registry.sqlite"

    def _episode(self, source: Path) -> Episode:
        eid = episode_id(source)
        return Episode(source=source, id=eid, work=self.work_dir(eid), workspace_root=self.root)

    def find_episode(self, ref: str) -> Episode:
        """`ref` is a media file path, or an SxxEyy code searched under `config.source`."""
        path = Path(ref).expanduser()
        if path.is_file():
            return self._episode(path)
        m = _CODE.fullmatch(ref.strip())
        if not m:
            raise FileNotFoundError(f"{ref}: not a file, and not an SxxEyy episode code")
        source = self.config.source
        if source is None:
            raise FileNotFoundError(f"{ref}: no `source` set in {self.root / 'fukikae.toml'}")
        want = (int(m[1]), int(m[2]))
        for p in sorted(source.rglob("*")):
            if p.suffix.lower() not in VIDEO_SUFFIXES or not p.is_file():
                continue
            found = _CODE.search(p.name)
            if found and (int(found[1]), int(found[2])) == want:
                return self._episode(p)
        raise FileNotFoundError(f"{ref}: no matching episode under {source}")
