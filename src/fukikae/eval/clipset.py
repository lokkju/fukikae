"""Clip-set embedder check: Ruling R8's nearest-centroid accuracy on folder-per-character clips.

A clip set is a directory whose top-level folders are characters and whose audio files
(found recursively) are that character's clips. No episode media and no clustering are
involved, so the number isolates how well an embedder separates the characters.

Split rule (the plan's "Clip-set split"): when every clip's file name carries an episode
number, each character trains on the smallest prefix of its episodes holding at least half
its clips and tests on the rest, so no episode sits on both sides. Otherwise the first half
of each character's clips trains: by whole folders (a character's clips under two or more
folders, e.g. emotions, train on the smallest prefix of its sorted folders holding at least
half its clips, so no folder is on both sides and one take cannot leak across), else in sorted
path order. A character with no clip on one side
is not scored and is listed.
"""

from __future__ import annotations

import hashlib
import json
import re
import wave
from collections import defaultdict
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fukikae.artifacts import atomic_path
from fukikae.config import BackendConfig
from fukikae.eval.identity.metrics import nearest_centroid_accuracy
from fukikae.eval.identity.segments import Segment
from fukikae.media import run_ffmpeg

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
DEFAULT_EPISODE_PATTERN = r"(?<=\s)(\d{1,3})(?=_)"
CACHE_DIR = ".fukikae-cache"

EmbedFn = Callable[[BackendConfig, Path, list[tuple[float, float]]], np.ndarray]


@dataclass(frozen=True)
class Clip:
    character: str
    path: Path
    episode: int | None


def scan_clipset(
    root: Path,
    episode_pattern: str = DEFAULT_EPISODE_PATTERN,
    exclude: Collection[str] = (),
) -> list[Clip]:
    """Every audio file under each top-level directory of `root`, sorted by path. Hidden
    directories (the cache) are skipped at any depth. The episode is the first match of
    `episode_pattern` in the file name, else None."""
    root = Path(root)
    pat = re.compile(episode_pattern)
    clips: list[Clip] = []
    for char_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if char_dir.name in exclude:
            continue
        for path in sorted(char_dir.rglob("*")):
            rel = path.relative_to(char_dir)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            m = pat.search(path.name)
            clips.append(Clip(char_dir.name, path, int(m.group(1)) if m else None))
    return clips


def split_clips(clips: list[Clip]) -> tuple[list[Clip], list[Clip], str, list[str]]:
    """(train, test, rule, characters not scorable); rule is "episode", "folder" (whole
    folders), "half" (sorted path order) or "folder+half" (per character, both)."""
    by_char: dict[str, list[Clip]] = defaultdict(list)
    for c in clips:
        by_char[c.character].append(c)
    rule = "episode" if clips and all(c.episode is not None for c in clips) else "half"
    used_folder = used_half = False
    train: list[Clip] = []
    test: list[Clip] = []
    skipped: list[str] = []
    for name in sorted(by_char):
        mine = by_char[name]
        if rule == "episode":
            counts: dict[int, int] = defaultdict(int)
            for c in mine:
                counts[c.episode] += 1  # type: ignore[index]
            need = len(mine) / 2
            taken: set[int] = set()
            total = 0
            for ep in sorted(counts):
                if total >= need:
                    break
                taken.add(ep)
                total += counts[ep]
            side_train = [c for c in mine if c.episode in taken]
            side_test = [c for c in mine if c.episode not in taken]
        else:
            folders: dict[Path, int] = defaultdict(int)
            for c in mine:
                folders[c.path.parent] += 1
            if len(folders) >= 2:
                used_folder = True
                need = len(mine) / 2
                taken_dirs: set[Path] = set()
                total = 0
                for d in sorted(folders):
                    if total >= need:
                        break
                    taken_dirs.add(d)
                    total += folders[d]
                side_train = [c for c in mine if c.path.parent in taken_dirs]
                side_test = [c for c in mine if c.path.parent not in taken_dirs]
            else:
                used_half = True
                ordered = sorted(mine, key=lambda c: str(c.path))
                cut = (len(ordered) + 1) // 2
                side_train, side_test = ordered[:cut], ordered[cut:]
        if not side_train or not side_test:
            skipped.append(name)
            continue
        train += side_train
        test += side_test
    if rule == "half":
        rule = (
            "folder"
            if used_folder and not used_half
            else "half"
            if used_half and not used_folder
            else "folder+half"
        )
    return train, test, rule, skipped


def _clip_key(path: Path) -> str:
    st = path.stat()
    blob = json.dumps([str(path), st.st_size, st.st_mtime_ns], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prepare_audio(clips: list[Clip], cache: Path) -> dict[Path, tuple[Path, float]]:
    """A 16 kHz mono wav per clip under ``cache/16k/`` (named by a hash of the source path,
    size and mtime) and its duration in seconds, keyed by the source path."""
    out_dir = Path(cache) / "16k"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[Path, tuple[Path, float]] = {}
    for c in clips:
        if c.path in result:
            continue
        target = out_dir / f"{_clip_key(c.path)}.wav"
        if not target.is_file():
            with atomic_path(target) as tmp:
                run_ffmpeg(["-i", str(c.path), "-ac", "1", "-ar", "16000", "-f", "wav", str(tmp)])
        with wave.open(str(target), "rb") as w:
            duration = w.getnframes() / float(w.getframerate())
        result[c.path] = (target, duration)
    return result


def embed_clips(
    clips: list[Clip],
    audio: dict[Path, tuple[Path, float]],
    emb_cfg: BackendConfig,
    embed: EmbedFn,
    cache: Path,
    identity: str = "",
) -> np.ndarray:
    """One embedding row per clip over its whole span, cached as ``cache/emb-<sha>.npy`` keyed
    by the backend, its options, `identity` (the model file or revision the backend resolves
    to, which a changed default must miss) and the clips' keys."""
    if not clips:
        return np.zeros((0, 0), dtype=np.float32)
    keys = [audio[c.path][0].stem for c in clips]
    blob = json.dumps(
        [emb_cfg.backend, emb_cfg.options, identity, keys], sort_keys=True, default=str
    )
    path = Path(cache) / f"emb-{hashlib.sha256(blob.encode('utf-8')).hexdigest()}.npy"
    if path.is_file():
        try:
            rows = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            rows = None
        if rows is not None and rows.ndim == 2 and len(rows) == len(clips):
            return rows
    parts = []
    for c in clips:
        wav, duration = audio[c.path]
        row = np.asarray(embed(emb_cfg, wav, [(0.0, duration)]), dtype=np.float32)
        if row.ndim != 2 or len(row) != 1:
            raise ValueError(f"{emb_cfg.backend}: one span gave embeddings of shape {row.shape}")
        parts.append(row[0])
    rows = np.stack(parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp, tmp.open("wb") as f:
        np.save(f, rows, allow_pickle=False)
    return rows


def clipset_accuracy(
    train: list[Clip],
    test: list[Clip],
    train_embs: np.ndarray,
    test_embs: np.ndarray,
    audio: dict[Path, tuple[Path, float]] | None = None,
) -> dict[str, float]:
    """R8's per-character and overall nearest-centroid accuracy. With `audio` the accuracy
    is weighted by clip duration (speech-weighted, as R8); without it every clip counts 1."""

    def segs(side: str, clips: list[Clip]) -> list[Segment]:
        return [
            Segment(
                episode=side,
                start=0.0,
                end=audio[c.path][1] if audio is not None else 1.0,
                label=c.character,
            )
            for c in clips
        ]

    return nearest_centroid_accuracy(
        segs("train", train), train_embs, segs("test", test), test_embs
    )
