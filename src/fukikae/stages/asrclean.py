"""asrclean stage: drop stock ASR hallucinations over music or silence (A4).

Reads ``03-asr/transcript.json`` and the separation stems, writes ``03b-asrclean/transcript.json``
(the transcript without the dropped segments) and ``03b-asrclean/dropped.json``
(``[{start, end, text, reason}]``). The rule is `fukikae.hallucinations`: a segment goes when
its text is a stock outro phrase for the transcript's language and its span is a theme song,
music or no speech by the song detector (`fukikae.songs`, with ``[mix.songs]``).

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

from fukikae import hallucinations, songs
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


class AsrCleanStage:
    name = "asrclean"

    def inputs(self, ep: Episode) -> list[Path]:
        sep = ep.work / "01-separate"
        return [ep.work / ASR_TRANSCRIPT, *(sep / n for n in STEMS)]

    def outputs(self, ep: Episode) -> list[Path]:
        out = ep.work / CLEAN_TRANSCRIPT
        return [out, out.with_name("dropped.json")]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        return {
            "rule": hallucinations.rule_constants(),
            "songs": songs.detector_constants(_song_settings(cfg)),
        }

    def run(self, ep: Episode, cfg: Config) -> None:
        src, *stems = self.inputs(ep)
        out, dropped_path = self.outputs(ep)
        out.parent.mkdir(parents=True, exist_ok=True)
        transcript = read_json(src, Transcript)
        language = transcript.language or cfg.source_language or ""
        dropped: list[dict[str, Any]] = []
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
            keep = []
            for seg in transcript.segments:
                reason = hallucinations.drop_reason(seg, language, spans, song_like, active)
                if reason is None:
                    keep.append(seg)
                    continue
                start, end = hallucinations.segment_span(seg)
                dropped.append(
                    {"start": round(start, 3), "end": round(end, 3), "text": seg.text,
                     "reason": reason}
                )  # fmt: skip
                log.info(
                    "asrclean %s: dropped %.1f-%.1f s %r: %s",
                    ep.id, start, end, seg.text, reason,
                )  # fmt: skip
        if dropped:
            write_json(out, transcript.model_copy(update={"segments": keep}))
        else:
            with atomic_path(out) as tmp:
                shutil.copyfile(src, tmp)
        with atomic_path(dropped_path) as tmp:
            tmp.write_text(json.dumps(dropped, ensure_ascii=False, indent=2) + "\n", "utf-8")
