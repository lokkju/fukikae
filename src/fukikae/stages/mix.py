"""Mix stage: the dub track, its subtitles and the dubbed episode.

Reads ``08-timefit/report.json`` and its wavs, ``06-translate/cues.<target>.json``,
``01-separate/background.wav`` and ``vocals.wav``, ``00-extract/audio.wav`` (the loudness
target), ``04b-tag/tags.json`` when the tag stage runs, and the source media; writes
``09-mix/dub.<target>.wav``, ``09-mix/dub.<target>.ass``, ``09-mix/report.json`` (a
``MixReport``) and ``out/<episode>.fukikae.mkv``.

Songs keep the original audio, from evidence that does not need the tag stage. A song span is
any of: a theme song found by ``fukikae.songs`` in the separation stems (``detector``), a cue
the translate stage marked ``skip = "song"`` (``subs`` on the subtitle path, from the subtitle
filter; ``tag`` on the ASR path, from tag spans), the tag stage's ``singing`` spans, and, with
``[mix] protect_bgm = true``, its ``bgm`` spans (off by default: SenseVoice marks BGM under
ordinary dialogue). A fitted cue whose slot's midpoint falls in a song span is not placed (its
source is in ``dropped``), and the original vocals stay untouched over every song span.
Laughter spans keep the original vocals even under a dub cue.

Elsewhere the original vocals are removed where a dub cue plays: over the cue's slot and over
the span its audio actually plays (borrowed silence included), faded over ``fade_ms``, or kept
``duck_db`` down instead. With ``keep_original_vocals = false`` they are removed everywhere but
song and laughter spans. The dub's level is matched to the original vocals' (one gain for the
episode, in the report), and the mix's integrated loudness to the original mix's (EBU R128).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from fukikae import media, songs
from fukikae.artifacts import read_json, write_json
from fukikae.config import Config, ConfigError
from fukikae.mix import (
    DUB_BITRATE,
    MixSettings,
    Span,
    build_dub,
    loudness_stats,
    mux,
    normalize,
    removal_windows,
    subtract,
    union,
    write_ass,
)
from fukikae.models import DubCue, DubCues, FitReport, MixReport, ProtectedSpan, Tags
from fukikae.workspace import Episode, Workspace

log = logging.getLogger(__name__)
MIX_VERSION = 2  # 2: undubbed cues keep their voice; ASS on the played span
MIX_KEYS = tuple(f.name for f in fields(MixSettings) if f.name != "songs")


def mix_settings(cfg: Config) -> MixSettings:
    """``[mix]``, the workspace's keys over the profile's; unknown keys and bad values are
    ConfigErrors naming the key."""
    merged: dict[str, Any] = {}
    for source in (cfg.profile_raw, cfg.raw):
        table = source.get("mix")
        if isinstance(table, dict):
            merged.update(table)
    song_table: dict[str, Any] = {}
    for source in (cfg.profile_raw, cfg.raw):
        table = source.get("mix")
        if isinstance(table, dict) and isinstance(table.get("songs"), dict):
            song_table.update(table["songs"])
    merged.pop("songs", None)
    unknown = sorted(set(merged) - set(MIX_KEYS))
    if unknown:
        raise ConfigError(
            f"[mix] has unknown keys {', '.join(unknown)}; known: {', '.join(MIX_KEYS)}"
        )
    for key in ("keep_original_vocals", "match_dialogue", "protect_bgm", "detect_songs"):
        if key in merged and not isinstance(merged[key], bool):
            raise ConfigError(f"[mix] {key} must be true or false, not {merged[key]!r}")
    for key in ("duck_db", "fade_ms", "true_peak_db"):
        value = merged.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
            raise ConfigError(f"[mix] {key} must be a number, not {value!r}")
    if "fade_ms" in merged:
        merged["fade_ms"] = round(merged["fade_ms"])
    bad = sorted(set(song_table) - set(songs.SONG_KEYS))
    if bad:
        raise ConfigError(
            f"[mix.songs] has unknown keys {', '.join(bad)}; known: {', '.join(songs.SONG_KEYS)}"
        )
    for key, value in song_table.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"[mix.songs] {key} must be a number, not {value!r}")
    try:
        song_settings = songs.SongSettings(**song_table)
    except (ValueError, TypeError) as e:
        raise ConfigError(f"[mix.songs] {e}") from e
    try:
        return MixSettings(**merged, songs=song_settings)
    except ValueError as e:
        raise ConfigError(f"[mix] {e}") from e


def _inside(t: float, spans: list[ProtectedSpan]) -> ProtectedSpan | None:
    return next((s for s in spans if s.start <= t <= s.end), None)


class MixStage:
    name = "mix"

    def __init__(self, cfg: Config | None = None, *, tags: bool = False) -> None:
        self.cfg = cfg
        self.tags = tags

    def _config(self, ep: Episode) -> Config:
        return self.cfg if self.cfg is not None else Workspace(ep.workspace_root).config

    # -- paths -------------------------------------------------------------------------------

    @staticmethod
    def _fit_path(ep: Episode) -> Path:
        return ep.work / "08-timefit" / "report.json"

    @staticmethod
    def _cues_path(ep: Episode, cfg: Config) -> Path:
        return ep.work / "06-translate" / f"cues.{cfg.target_language}.json"

    @staticmethod
    def _tags_path(ep: Episode) -> Path:
        return ep.work / "04b-tag" / "tags.json"

    def inputs(self, ep: Episode) -> list[Path]:
        cfg = self._config(ep)
        fit = self._fit_path(ep)
        wavs: list[Path] = []
        if fit.exists():
            wavs = [
                ep.work / "08-timefit" / f"{c.cue_id}.wav" for c in read_json(fit, FitReport).cues
            ]
        sep = ep.work / "01-separate"
        return [
            fit,
            self._cues_path(ep, cfg),
            sep / "background.wav",
            sep / "vocals.wav",
            sep / "vocals_16k.wav",
            ep.work / "00-extract" / "audio.wav",
            *([self._tags_path(ep)] if self.tags else []),
            *wavs,
            ep.source,
        ]

    def outputs(self, ep: Episode) -> list[Path]:
        target = self._config(ep).target_language
        out = ep.work / "09-mix"
        return [
            out / f"dub.{target}.wav",
            out / f"dub.{target}.ass",
            out / "report.json",
            ep.work / "out" / f"{ep.id}.fukikae.mkv",
        ]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        return {
            "settings": asdict(mix_settings(cfg)),
            "songs": songs.detector_constants(mix_settings(cfg).songs),
            "target_language": cfg.target_language,
            "bitrate": DUB_BITRATE,
            "ffmpeg": media.ffmpeg_version(),
            "version": MIX_VERSION,
        }

    # -- run ---------------------------------------------------------------------------------

    def _song_spans(
        self, s: MixSettings, doc: DubCues, tags: Tags | None, sep: Path
    ) -> tuple[list[ProtectedSpan], list[songs.Candidate]]:
        found: list[ProtectedSpan] = []
        cue_source = "subs" if doc.path == "subs" else "tag"
        for c in doc.cues:
            if c.skip == "song":
                found.append(ProtectedSpan(start=c.start, end=c.end, source=cue_source))
        for e in tags.events if tags else []:
            if e.label == "singing" or (s.protect_bgm and e.label == "bgm"):
                found.append(ProtectedSpan(start=e.start, end=e.end, source=f"tag:{e.label}"))
        candidates: list[songs.Candidate] = []
        if s.detect_songs:
            spans, candidates = songs.detect_theme_songs(
                sep / "background.wav", sep / "vocals.wav", sep / "vocals_16k.wav", s.songs
            )
            for sp in spans:
                found.append(
                    ProtectedSpan(start=sp.start, end=sp.end, source=sp.source, score=sp.score)
                )
        return found, candidates

    def run(self, ep: Episode, cfg: Config) -> None:
        from scipy.io import wavfile

        s = mix_settings(cfg)
        target = cfg.target_language
        dub_wav, ass, report_path, mkv = self.outputs(ep)
        doc = read_json(self._cues_path(ep, cfg), DubCues)
        fit = read_json(self._fit_path(ep), FitReport)
        tags_path = self._tags_path(ep)
        tags = read_json(tags_path, Tags) if self.tags and tags_path.exists() else None
        sep = ep.work / "01-separate"
        background, vocals = sep / "background.wav", sep / "vocals.wav"

        song_spans, candidates = self._song_spans(s, doc, tags, sep)
        for c in candidates:
            log.info(
                "mix %s: song candidate %s %.0f-%.0f s, score %.3f, voiced %s: %s",
                ep.id, c.region, c.start, c.end, c.score,
                "-" if c.voiced is None else f"{c.voiced:.3f}", c.reason,
            )  # fmt: skip
        laughter = [
            ProtectedSpan(start=e.start, end=e.end, source="tag:laughter")
            for e in (tags.events if tags else [])
            if e.label == "laughter"
        ]
        by_id = {c.id: c for c in doc.cues}
        placed: list[tuple[Path, float]] = []
        played: list[Span] = []
        slots: list[Span] = []
        spoken: list[DubCue] = []
        dropped: dict[str, str] = {}
        for fc in fit.cues:
            hit = _inside((fc.slot_start + fc.slot_end) / 2, song_spans)
            if hit is not None:
                dropped[fc.cue_id] = hit.source
                continue
            end = fc.start + fc.seconds
            placed.append((ep.work / "08-timefit" / f"{fc.cue_id}.wav", fc.start))
            played.append((fc.start, end))
            slots.append((fc.slot_start, fc.slot_end))
            if fc.cue_id in by_id:
                # The event follows the audio as it plays.
                spoken.append(by_id[fc.cue_id].model_copy(update={"start": fc.start, "end": end}))
        dropped_by_source: dict[str, int] = {}
        for source in dropped.values():
            dropped_by_source[source] = dropped_by_source.get(source, 0) + 1
        log.info(
            "mix %s: %d cues placed, %d dropped as song %s",
            ep.id, len(placed), len(dropped), dropped_by_source,
        )  # fmt: skip
        # A dubbed slot clears the original voice over the whole slot (ruling CD11), except
        # where a cue that is not dubbed (errored, untranslated, skipped) keeps its own voice:
        # there only the dub's played span is cleared.
        dubbed = {fc.cue_id for fc in fit.cues if fc.cue_id not in dropped}
        undubbed = [(c.start, c.end) for c in doc.cues if c.id not in dubbed]
        dub_spans = union([*played, *subtract(slots, undubbed)])

        keep = [(p.start, p.end) for p in song_spans]
        if s.keep_original_vocals:
            windows = removal_windows(dub_spans, tags, songs=keep, protect_bgm=s.protect_bgm)
        else:
            rate, data = wavfile.read(background, mmap=True)
            whole = max([len(data) / rate, *(b for _, b in dub_spans)])
            windows = removal_windows([(0.0, whole)], tags, songs=keep, protect_bgm=s.protect_bgm)

        out_dir = dub_wav.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        premix = out_dir / f".premix.{target}.wav"
        try:
            stats = build_dub(background, vocals, placed, windows, s, premix)
            original = loudness_stats(ep.work / "00-extract" / "audio.wav")
            result = normalize(premix, original["i"], dub_wav, s.true_peak_db)
            before = result["source"]
        finally:
            premix.unlink(missing_ok=True)
        after = loudness_stats(dub_wav)
        write_ass(spoken, ass)
        mux(ep.source, dub_wav, ass, target, mkv)
        encoded = loudness_stats(mkv, "0:a:0")  # the dub track as muxed, after the AAC encode
        write_json(
            report_path,
            MixReport(
                episode=ep.id,
                target_language=target,
                songs=sorted(song_spans, key=lambda p: (p.start, p.end, p.source)),
                laughter=laughter,
                dropped=dropped,
                dropped_by_source=dropped_by_source,
                song_candidates=[asdict(c) for c in candidates],
                placed=len(placed),
                removed_seconds=round(sum(b - a for a, b in union(windows)), 3),
                dub_gain_db=None if stats.dub_gain_db is None else round(stats.dub_gain_db, 2),
                loudness={"original": original, "premix": before, "final": after, "mkv": encoded},
                gain_db=result["gain_db"],
                normalization=f"linear gain, true-peak limiter at {s.true_peak_db} dBTP",
            ),
        )
