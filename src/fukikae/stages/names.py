"""Names stage: speaker names from subtitles onto the diarize turns (D27), CPU only.

Reads ``04-diarize/turns.json`` and ``clusters.npz`` and the subtitle tracks
``names.detect.candidate_tracks`` offers (``[names] file``, ``[names] track``, the target
language's track as translate finds it, then other languages' dialogue tracks). Each candidate
is copied, or extracted from the source, into ``04c-names/tracks/`` once per source file
version, and those copies are the stage's inputs, so the stage reruns when a track's content
changes and not when the media file is merely touched or remuxed with the same subtitles.

Writes under ``04c-names/``:

``names.json``
    A ``NamesResult``, always: the verdict (``off``, ``none``, ``holdout``, ``partial``,
    ``full``), the track chosen and every one tried, the lines, hidden line ids, pieces, labels,
    unvoiced lines, excluded spans, dropped row seconds and alias suggestions.
``turns.json``, ``clusters.npz``
    Only when names were used: the refined diarization (``<cluster>~<slug>`` labels) and the
    diarize stage's embedding rows moved onto those labels, in the diarize stage's format. A
    rerun that doesn't use names deletes them, and ``names.inputs.identity_inputs`` then
    returns ``04-diarize``'s pair.

Evaluation hooks (``[names]``): an episode in ``holdout_episodes`` reads no subtitles at all
(verdict ``holdout``, and no track copy stays under ``tracks/``); ``hide_fraction`` withholds
that share of each key's lines (keys with at least ``MIN_HIDE_LINES``), drawn per candidate
before detection with ``random.Random(f"{hide_seed}:{episode}:{key}")``. Hidden lines stay in
``names.json`` with their keys, but count as unnamed in detection and the offset, and name,
cut, label, exclude and suggest nothing; consumers read lines through
``NamesResult.visible_lines()``. ``names.inputs.check_held_out`` is the eval's guard.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fukikae import media
from fukikae.artifacts import atomic_path, read_json, write_json
from fukikae.config import Config, ConfigError
from fukikae.models import Diarization
from fukikae.names import detect
from fukikae.names.align import (
    attribute,
    excluded_spans,
    label_infos,
    refine_clusters,
    refined_turns,
)
from fukikae.names.detect import TrackChoice
from fukikae.names.inputs import DIARIZE_DIR, NAMES_DIR
from fukikae.names.models import NamedLine, NamesResult, Piece
from fukikae.names.normalize import suggest_aliases
from fukikae.workspace import Episode, Workspace

log = logging.getLogger(__name__)

# Bump when the stage writes something different for the same inputs and settings.
NAMES_VERSION = 1
MIN_HIDE_LINES = 5  # a key needs this many lines before hide_fraction withholds any
TRACKS_DIR = "tracks"


def hide_lines(lines: list[NamedLine], fraction: float, seed: int, episode: str) -> list[int]:
    """The ids withheld from the stage: per key with at least MIN_HIDE_LINES lines,
    ``round(fraction * n)`` of them, drawn with ``random.Random(f"{seed}:{episode}:{key}")``,
    so a key's draw never depends on the other keys."""
    if fraction <= 0:
        return []
    by_key: dict[str, list[int]] = defaultdict(list)
    for ln in lines:
        if ln.key is not None:
            by_key[ln.key].append(ln.id)
    out: list[int] = []
    for key in sorted(by_key):
        ids = sorted(by_key[key])
        if len(ids) >= MIN_HIDE_LINES:
            rng = random.Random(f"{seed}:{episode}:{key}")  # noqa: S311  # a reproducible draw
            out += rng.sample(ids, round(fraction * len(ids)))
    return sorted(out)


def _describe(c: TrackChoice) -> dict[str, Any]:
    t = c.track
    return {
        "how": c.how,
        "language": c.language,
        "track": None if t is None else t.index,
        "codec": None if t is None else t.codec,
        "title": None if t is None else t.title,
        "file": None if c.file is None else str(c.file),
    }


def _coverage(
    lines: list[NamedLine], hidden: list[int], pieces: list[Piece], v: detect.Verdict
) -> dict[str, float | int]:
    per_turn: dict[int, list[Piece]] = defaultdict(list)
    for p in pieces:
        per_turn[p.turn].append(p)
    return {
        "dialogue": v.dialogue,
        "named": v.named,
        "share": round(v.share, 6),
        "distinct": v.distinct,
        "top_share": round(v.top_share, 6),
        "align_share": round(v.align_share, 6),
        "turns": len(per_turn),
        "named_turns": sum(1 for ps in per_turn.values() if len(ps) == 1 and ps[0].key),
        "cut_turns": sum(1 for ps in per_turn.values() if len(ps) > 1),
        "pieces": len(pieces),
        "sub_clusters": len({p.label for p in pieces if p.key is not None}),
        "lines": len(lines),
        "hidden": len(hidden),
    }


class NamesStage:
    name = "names"

    def __init__(self, cfg: Config | None = None) -> None:
        # The config the CLI resolved; inputs() has no config of its own.
        self.cfg = cfg
        self._probes: dict[tuple[str, int, int], media.ProbeResult] = {}
        # inputs(), config_key() and run() all ask; probe and choose once per source version
        self._choices: dict[str, list[TrackChoice]] = {}

    def _config(self, ep: Episode) -> Config:
        return self.cfg if self.cfg is not None else Workspace(ep.workspace_root).config

    @staticmethod
    def _holdout(ep: Episode, cfg: Config) -> bool:
        return ep.id in cfg.names.holdout_episodes

    def _reads_subtitles(self, ep: Episode, cfg: Config) -> bool:
        return cfg.names.use != "off" and not self._holdout(ep, cfg)

    # -- candidates ------------------------------------------------------------------------

    def _probe(self, ep: Episode) -> media.ProbeResult:
        if not ep.source.is_file():
            return media.ProbeResult()
        st = ep.source.stat()
        key = (str(ep.source.resolve()), st.st_size, st.st_mtime_ns)
        if key not in self._probes:
            self._probes[key] = media.probe(ep.source)
        return self._probes[key]

    def _candidates(self, ep: Episode, cfg: Config) -> list[TrackChoice]:
        """The tracks to try; ConfigError for a missing ``[names]`` file or track, a
        non-numeric track or ``{episode}`` without an episode (as translate reports
        ``[subs]``)."""
        st = ep.source.stat() if ep.source.is_file() else None
        memo = json.dumps(
            [
                ep.id,
                str(ep.source),
                None if st is None else [st.st_size, st.st_mtime_ns],
                cfg.target_language,
                cfg.raw.get("names"),
                cfg.raw.get("subs"),
            ],
            sort_keys=True,
            default=str,
        )
        if memo not in self._choices:
            try:
                self._choices[memo] = detect.candidate_tracks(
                    ep.source, self._probe(ep), cfg, episode=ep.id
                )
            except media.MediaError as e:
                raise ConfigError(str(e)) from e
        return self._choices[memo]

    def _materialise(
        self, ep: Episode, choices: list[TrackChoice]
    ) -> list[tuple[TrackChoice, Path | None]]:
        """Each candidate with its copy under ``04c-names/tracks/`` (None for an embedded track
        that can't carry names). A copy is refreshed only when its source file's size or mtime
        changed, so the source is read once per version, and every embedded track due is
        extracted in one ffmpeg pass."""
        tracks = ep.work / NAMES_DIR / TRACKS_DIR
        index_path = tracks / "index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            index = {}
        if not isinstance(index, dict):
            index = {}
        changed = False
        out: list[tuple[TrackChoice, Path | None]] = []
        extract: list[tuple[media.SubTrack, Path, Path]] = []  # track, scratch, dest
        for c in choices:
            if c.file is not None:
                src = c.file.resolve()
                digest = hashlib.sha256(str(src).encode()).hexdigest()[:8]
                name = f"file-{digest}{src.suffix.casefold() or '.txt'}"
                extra: list[Any] = []
            elif c.track is not None and detect.no_name_field(c) is None:
                src = ep.source.resolve()
                name = f"track-{c.track.index}.ass"
                extra = [c.track.index]
            else:
                out.append((c, None))
                continue
            st = src.stat()
            # inode and ctime too: an edit that keeps the size and resets the mtime still
            # changes the ctime, and a replaced file has a new inode
            stamp = [str(src), st.st_size, st.st_mtime_ns, st.st_ino, st.st_ctime_ns, *extra]
            dest = tracks / name
            if index.get(name) != stamp or not dest.is_file():
                tracks.mkdir(parents=True, exist_ok=True)
                if c.file is not None:
                    with atomic_path(dest) as tmp:
                        shutil.copyfile(src, tmp)
                else:
                    assert c.track is not None
                    extract.append((c.track, tracks / f".{name}.partial", dest))
                index[name] = stamp
                changed = True
            out.append((c, dest))
        if extract:
            got = media.extract_subtitles(ep.source.resolve(), [(t, tmp) for t, tmp, _ in extract])
            for path, (_, _, dest) in zip(got, extract, strict=True):
                os.replace(path, dest)
        if changed:
            with atomic_path(index_path) as tmp:
                tmp.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        return out

    def _sources(self, ep: Episode, cfg: Config) -> list[tuple[TrackChoice, Path | None]]:
        if not self._reads_subtitles(ep, cfg):
            return []
        return self._materialise(ep, self._candidates(ep, cfg))

    # -- Stage protocol ----------------------------------------------------------------------

    def inputs(self, ep: Episode) -> list[Path]:
        d = ep.work / DIARIZE_DIR
        paths = [d / "turns.json", d / "clusters.npz"]
        try:
            sources = self._sources(ep, self._config(ep))
        except (ConfigError, media.MediaError, OSError):
            return paths  # run() raises the same error with its context
        return paths + [p for _, p in sources if p is not None]

    def outputs(self, ep: Episode) -> list[Path]:
        return [ep.work / NAMES_DIR / "names.json"]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:
        """What this episode's names stage reads: ``use``, whether the episode is held out, the
        thresholds, the hide settings (the seed only when lines are hidden) and the
        candidates. ``min_new_seconds`` (the characters stage's) and other episodes' holdout
        membership stay out, so changing them never reruns it."""
        s = cfg.names
        key: dict[str, Any] = {"version": NAMES_VERSION, "use": s.use}
        if s.use == "off":
            return key
        key["holdout"] = self._holdout(ep, cfg)
        if key["holdout"]:
            return key
        key |= {
            "min_coverage": s.min_coverage,
            "turn_share": s.turn_share,
            "align": s.align,
            "hide_fraction": s.hide_fraction,
        }
        if s.hide_fraction > 0:
            key["hide_seed"] = s.hide_seed
        try:
            sources = self._sources(ep, cfg)
        except (ConfigError, media.MediaError, OSError) as e:
            # Never raise outside run(): the error is part of the key, and run() raises it.
            key["candidates_error"] = f"{type(e).__name__}: {e}"
            return key
        key["candidates"] = [
            {**_describe(c), "file": None if p is None or c.file is None else p.name}
            for c, p in sources
        ]
        return key

    def run(self, ep: Episode, cfg: Config) -> None:
        s = cfg.names
        out = ep.work / NAMES_DIR
        out.mkdir(parents=True, exist_ok=True)
        turns_out, clusters_out = out / "turns.json", out / "clusters.npz"
        names_out = out / "names.json"
        # names.json goes last and says whether the pair beside it is the identity input, so a
        # run that fails part way must leave neither an old names.json nor a half-new pair.
        for p in (names_out, turns_out, clusters_out):
            p.unlink(missing_ok=True)
        if not self._reads_subtitles(ep, cfg):
            # held out or off: no copy of a track that could carry names stays behind
            shutil.rmtree(out / TRACKS_DIR, ignore_errors=True)
        diar_dir = ep.work / DIARIZE_DIR
        diar = read_json(diar_dir / "turns.json", Diarization)
        empty: dict[str, Any] = {
            "episode": ep.id,
            "used": False,
            "source": None,
            "offset_s": 0.0,
            "coverage": {},
            "lines": [],
            "hidden": [],
            "pieces": [],
            "labels": {},
            "unvoiced": [],
            "excluded": [],
            "dropped_row_seconds": 0.0,
            "suggestions": [],
        }
        result: NamesResult
        if s.use == "off":
            result = NamesResult(**empty, verdict="off", reason='[names] use = "off"')
        elif self._holdout(ep, cfg):
            result = NamesResult(
                **empty, verdict="holdout", reason=f"{ep.id} is in [names] holdout_episodes"
            )
        else:
            try:
                result = self._attribute(ep, cfg, diar, empty, turns_out, clusters_out)
            except BaseException:
                turns_out.unlink(missing_ok=True)
                clusters_out.unlink(missing_ok=True)
                raise
        if not result.used:
            turns_out.unlink(missing_ok=True)
            clusters_out.unlink(missing_ok=True)
        write_json(names_out, result)
        log.info(
            "names %s: %s%s",
            ep.id,
            result.verdict,
            f" ({result.reason})" if result.reason else "",
        )

    def _attribute(
        self,
        ep: Episode,
        cfg: Config,
        diar: Diarization,
        empty: dict[str, Any],
        turns_out: Path,
        clusters_out: Path,
    ) -> NamesResult:
        s = cfg.names
        tried: list[dict[str, Any]] = []
        chosen: tuple[dict[str, Any], detect.Verdict, list[NamedLine]] | None = None

        def hide(lines: list[NamedLine]) -> list[int]:
            return hide_lines(lines, s.hide_fraction, s.hide_seed, ep.id)

        for choice, path in self._sources(ep, cfg):
            desc = _describe(choice)
            why = detect.no_name_field(choice)
            if why is not None or path is None:
                tried.append({**desc, "verdict": "none", "reason": why})
                continue
            # Hidden lines are drawn from each candidate's own lines before detection and count
            # as unnamed there, so they never decide the verdict or the offset (spec step 1).
            v, lines = detect.assess_path(
                path,
                diar.turns,
                min_coverage=s.min_coverage,
                use=s.use,
                align=s.align,
                hide=hide,
            )
            tried.append({**desc, "verdict": v.level, "reason": v.reason})
            if v.level != "none":
                chosen = (desc, v, lines)
                break
        if chosen is None:
            if not tried:
                reason = "no subtitle track to read names from"
            elif len(tried) == 1:
                reason = tried[0]["reason"]
            else:
                reason = f"none of {len(tried)} tracks passed: " + "; ".join(
                    f"{t['how']} {t['language'] or t['file']}: {t['reason']}" for t in tried
                )
            return NamesResult(**{**empty, "tried": tried}, verdict="none", reason=reason)
        desc, v, lines = chosen
        hidden = hide(lines)  # the same draw: the offset moves lines, never their ids or keys
        hidden_set = set(hidden)
        visible = [ln for ln in lines if ln.id not in hidden_set]
        pieces, unvoiced = attribute(diar.turns, lines, turn_share=s.turn_share, hidden=hidden)
        rturns = refined_turns(pieces)
        write_json(turns_out, Diarization(backend=diar.backend, turns=rturns))
        dropped = refine_clusters(
            ep.work / DIARIZE_DIR / "clusters.npz", pieces, rturns, clusters_out
        )
        counts = Counter(ln.key for ln in visible if ln.key is not None)
        return NamesResult(
            episode=ep.id,
            used=True,
            verdict=v.level,
            reason=None,
            source=desc,
            tried=tried,
            offset_s=v.offset_s,
            coverage=_coverage(lines, hidden, pieces, v),
            lines=lines,
            hidden=hidden,
            pieces=pieces,
            labels=label_infos(pieces, visible),
            unvoiced=unvoiced,
            excluded=excluded_spans(visible),
            dropped_row_seconds=dropped,
            suggestions=suggest_aliases(counts),
        )
