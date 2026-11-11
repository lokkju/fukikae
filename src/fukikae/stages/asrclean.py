"""asrclean stage: drop stock ASR hallucinations over music or silence (A4).

Reads ``03-asr/transcript.json`` and the separation stems, writes ``03b-asrclean/transcript.json``
(the transcript without the dropped stretches), ``03b-asrclean/dropped.json`` (one
``{start, end, text, reason}`` per dropped stretch: a whole segment, or one word run of a
segment that keeps its other runs) and ``03b-asrclean/flagged.json`` (the same shape, one per
kept stretch of a stock phrase the operator should check: a surviving fragment, or a phrase
spoken over music). The rule is `fukikae.hallucinations.clean_segment`: a segment whose text
is a stock outro phrase loses each of its word runs that lies over a theme song, sung music or
no speech by the song detector (`fukikae.songs`, with ``[mix.songs]``) and the vocals' F0
stability (`fukikae.pitch.is_sung`, through the workspace's pitch cache), and goes whole when
every run does. Phrases match the transcript's own language only (its ``language``, else
``source_language``), never another language's list.

A stage of its own, CPU only, so that changing the filter never changes the asr key and reruns
Whisper or Qwen3-ASR. When nothing is dropped the output is a byte copy of the input, so the
stages reading it (through `fukikae.transcripts.transcript_path`) keep their keys.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from fukikae import hallucinations, pitch, songs
from fukikae.artifacts import atomic_path, read_json, write_json
from fukikae.config import Config
from fukikae.models import Transcript
from fukikae.transcripts import ASR_TRANSCRIPT, CLEAN_TRANSCRIPT
from fukikae.workspace import Episode

log = logging.getLogger(__name__)

STEMS = ("background.wav", "vocals.wav", "vocals_16k.wav")


def _song_settings(cfg: Config) -> songs.SongSettings:
    from fukikae.stages.mix import mix_settings

    return mix_settings(cfg).songs


def sung_test(vocals: Path, pitch_root: Path | None) -> hallucinations.SungTest:
    """Whether ``vocals`` over a stretch holds notes (`pitch.is_sung` on its F0 track), read
    through ``pitch_root``'s pitch cache when given."""

    def sung(lo: float, hi: float) -> bool:
        spans = [(lo, hi)]
        track = (
            pitch.cached_track(pitch_root, vocals, spans)
            if pitch_root is not None
            else pitch.f0_track(vocals, spans)
        )
        return pitch.is_sung(pitch.stability(track))

    return sung


def _row(x: hallucinations.Drop | hallucinations.Flag) -> dict[str, Any]:
    return {"start": round(x.start, 3), "end": round(x.end, 3), "text": x.text, "reason": x.reason}


class AsrCleanStage:
    name = "asrclean"

    def inputs(self, ep: Episode) -> list[Path]:
        sep = ep.work / "01-separate"
        return [ep.work / ASR_TRANSCRIPT, *(sep / n for n in STEMS)]

    def outputs(self, ep: Episode) -> list[Path]:
        out = ep.work / CLEAN_TRANSCRIPT
        return [out, out.with_name("dropped.json"), out.with_name("flagged.json")]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        return {
            "rule": hallucinations.rule_constants(),
            "songs": songs.detector_constants(_song_settings(cfg)),
        }

    def run(self, ep: Episode, cfg: Config) -> None:
        src, *stems = self.inputs(ep)
        out, dropped_path, flagged_path = self.outputs(ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        transcript = read_json(src, Transcript)
        language = transcript.language or cfg.source_language or ""
        dropped: list[dict[str, Any]] = []
        flagged: list[dict[str, Any]] = []
        keep = list(transcript.segments)
        suspects = [s for s in keep if hallucinations.is_stock_phrase(s.text, language)]
        if suspects and not all(p.exists() for p in stems):
            log.warning(
                "asrclean %s: separation stems missing; keeping %d stock-phrase segment(s)",
                ep.id, len(suspects),
            )  # fmt: skip
        elif suspects:
            s = _song_settings(cfg)
            background, vocals, vocals_16k = stems
            feats = songs.frame_features(background, vocals, s.active_db)
            song_like = songs.song_like_mask(*feats, s)
            spans, _ = songs.detect_theme_songs(background, vocals, vocals_16k, s)
            active = songs.active_frames(vocals, s.active_db)
            sung = sung_test(vocals, ep.workspace_root)
            keep = []
            for seg in transcript.segments:
                kept, drops, flags = hallucinations.clean_segment(
                    seg, language, spans, song_like, active, sung
                )
                if kept is not None:
                    keep.append(kept)
                for d in drops:
                    dropped.append(_row(d))
                    log.info(
                        "asrclean %s: dropped %.1f-%.1f s %r: %s",
                        ep.id, d.start, d.end, d.text, d.reason,
                    )  # fmt: skip
                for f in flags:
                    flagged.append(_row(f))
                    log.warning(
                        "asrclean %s: kept and flagged %.1f-%.1f s %r: %s",
                        ep.id, f.start, f.end, f.text, f.reason,
                    )  # fmt: skip
        if dropped:
            write_json(out, transcript.model_copy(update={"segments": keep}))
        else:
            with atomic_path(out) as tmp:
                shutil.copyfile(src, tmp)
        for path, rows in ((dropped_path, dropped), (flagged_path, flagged)):
            with atomic_path(path) as tmp:
                tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", "utf-8")
        if dropped or flagged:
            log.info(
                "asrclean %s: %d stretch(es) dropped, %d kept and flagged",
                ep.id, len(dropped), len(flagged),
            )  # fmt: skip
