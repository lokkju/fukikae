"""Timefit stage: place each rendered cue in its slot (``fukikae.timefit``).

Reads ``07-tts/plan.json``, ``06-translate/cues.<target>.json`` (the cue times) and every
rendered ``07-tts/<cue-id>.wav``; writes ``08-timefit/<cue-id>.wav`` (stretched, or a copy) and
``08-timefit/report.json`` (a ``FitReport``). Cues go in time order; a cue may borrow silence
after its slot up to the next cue's start (any cue, spoken or not, so a dub never runs into a
line the mix keeps in the original) and before it back to where the previous cue's audio ends.
A cue left long is placed whole and listed in ``over``; nothing is cut.

The key holds ``[timefit]`` (``max_speedup``, ``max_slowdown``, ``borrow_silence_ms``), the
ffmpeg version (``atempo`` shapes the audio) and ``FIT_VERSION``; the inputs name each wav, so
re-rendered audio re-fits even when the plan's keys are unchanged.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fukikae import media
from fukikae.artifacts import read_json, write_json
from fukikae.config import Config
from fukikae.models import DubCues, FitReport, FitReportCue, TtsPlan
from fukikae.timefit import apply_fit, fit_settings, plan_fit
from fukikae.workspace import Episode, Workspace

FIT_VERSION = 1
OVER_EPS = 1e-3  # seconds; less is rounding


class TimefitStage:
    name = "timefit"

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg

    def _config(self, ep: Episode) -> Config:
        return self.cfg if self.cfg is not None else Workspace(ep.workspace_root).config

    @staticmethod
    def _plan_path(ep: Episode) -> Path:
        return ep.work / "07-tts" / "plan.json"

    @staticmethod
    def _cues_path(ep: Episode, cfg: Config) -> Path:
        return ep.work / "06-translate" / f"cues.{cfg.target_language}.json"

    def inputs(self, ep: Episode) -> list[Path]:
        cfg = self._config(ep)
        plan = self._plan_path(ep)
        wavs: list[Path] = []
        if plan.exists():
            wavs = [
                ep.work / "07-tts" / f"{c.cue_id}.wav"
                for c in read_json(plan, TtsPlan).cues
                if c.skipped is None
            ]
        return [plan, self._cues_path(ep, cfg), *wavs]

    def outputs(self, ep: Episode) -> list[Path]:
        """The report, then every fitted wav it lists, so a deleted wav makes the stage stale."""
        report = ep.work / "08-timefit" / "report.json"
        if not report.exists():
            return [report]
        try:
            cues = read_json(report, FitReport).cues
        except ValueError:
            return [report]
        return [report, *(ep.work / "08-timefit" / f"{c.cue_id}.wav" for c in cues)]

    def config_key(self, cfg: Config, ep: Episode) -> dict[str, Any]:  # noqa: ARG002
        return {
            "settings": asdict(fit_settings(cfg)),
            "ffmpeg": media.ffmpeg_version(),
            "version": FIT_VERSION,
        }

    def run(self, ep: Episode, cfg: Config) -> None:
        settings = fit_settings(cfg)
        plan = read_json(self._plan_path(ep), TtsPlan)
        cues = sorted(
            read_json(self._cues_path(ep, cfg), DubCues).cues, key=lambda c: (c.start, c.end, c.id)
        )
        rendered = {c.cue_id: c for c in plan.cues if c.skipped is None}
        out_dir = ep.work / "08-timefit"
        out_dir.mkdir(parents=True, exist_ok=True)

        placed: list[FitReportCue] = []
        prev_end: float | None = None
        for n, cue in enumerate(cues):
            next_start = next((c.start for c in cues[n + 1 :] if c.start >= cue.start), None)
            pc = rendered.get(cue.id)
            if pc is None:
                prev_end = cue.end if prev_end is None else max(prev_end, cue.end)
                continue
            src = ep.work / "07-tts" / f"{cue.id}.wav"
            rendered_s = pc.seconds if pc.seconds is not None else 0.0
            fp = plan_fit(
                rendered_s, (cue.start, cue.end), prev_end, next_start, pc.rate_factor, settings
            )
            seconds = apply_fit(src, fp, out_dir / f"{cue.id}.wav")
            placed.append(
                FitReportCue(
                    cue_id=cue.id,
                    slot_start=round(cue.start, 4),
                    slot_end=round(cue.end, 4),
                    start=round(fp.start, 4),
                    seconds=round(seconds, 4),
                    rendered_seconds=round(rendered_s, 4),
                    rate_factor=pc.rate_factor,
                    stretch=round(fp.stretch, 4),
                    borrowed_before=round(fp.borrowed_before, 4),
                    borrowed_after=round(fp.borrowed_after, 4),
                    over_by=round(fp.over_by, 4),
                )
            )
            end = fp.start + seconds
            prev_end = end if prev_end is None else max(prev_end, end)

        keep = {c.cue_id for c in placed}
        for stale in out_dir.glob("*.wav"):
            if stale.stem not in keep:
                stale.unlink()
        over = [c.cue_id for c in placed if c.over_by > OVER_EPS]
        write_json(
            out_dir / "report.json",
            FitReport(episode=ep.id, cues=placed, over=over, summary=self._summary(placed)),
        )

    @staticmethod
    def _summary(placed: list[FitReportCue]) -> dict[str, Any]:
        stretched = [c.stretch for c in placed if c.stretch != 1.0]
        over = [c for c in placed if c.over_by > OVER_EPS]
        worst = max(over, key=lambda c: c.over_by, default=None)
        return {
            "placed": len(placed),
            "stretched": len(stretched),
            "median_stretch": round(statistics.median(stretched), 4) if stretched else 1.0,
            "borrowed": sum(1 for c in placed if c.borrowed_before or c.borrowed_after),
            "borrowed_seconds": round(sum(c.borrowed_before + c.borrowed_after for c in placed), 3),
            "over": len(over),
            "over_seconds": round(sum(c.over_by for c in over), 3),
            "worst": None if worst is None else {"cue_id": worst.cue_id, "over_by": worst.over_by},
        }
