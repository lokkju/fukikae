"""ASR evaluation: character error rate against a subtitle track's text."""

import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from doppio.config import Config
from doppio.media import run_ffmpeg
from doppio.models import Transcript
from doppio.workspace import Episode

_ASS = re.compile(r"\{[^}]*\}")
_HTML = re.compile(r"<[^>]*>")
_SDH = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|［[^］]*］")
_CJK = {"zh", "ja", "ko"}
_TIME = re.compile(r"\d+:\d+:\d+[,.]\d+\s*-->")


def normalize(text: str, language: str) -> str:
    """Strip markup, SDH notes, music symbols and punctuation; collapse whitespace.

    Chinese, Japanese and Korean lose all whitespace (their scoring is per character).
    """
    text = _ASS.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _SDH.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c if c.isspace() or unicodedata.category(c)[0] in "LNM" else " " for c in text)
    if language.split("-")[0] in _CJK:
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of `hypothesis` against `reference`."""
    import jiwer

    return float(jiwer.cer(reference, hypothesis))


def parse_srt(srt: str) -> list[str]:
    """Cue texts in order, the lines of a cue joined by a space."""
    cues: list[str] = []
    for block in re.split(r"\n\s*\n", srt.replace("\r\n", "\n").strip()):
        lines = block.split("\n")
        for i, line in enumerate(lines):
            if _TIME.search(line):
                text = " ".join(x.strip() for x in lines[i + 1 :] if x.strip())
                if text:
                    cues.append(text)
                break
    return cues


def extract_subtitle_text(ep: Episode, stream_index: int) -> list[str]:
    """Cue texts of one subtitle stream (absolute index), via a temp file in the workspace."""
    tmp_dir = ep.work / "eval"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as d:
        out = Path(d) / "subtitle.srt"
        run_ffmpeg(["-i", str(ep.source), "-map", f"0:{stream_index}", "-f", "srt", str(out)])
        return parse_srt(out.read_text(encoding="utf-8", errors="replace"))


@dataclass(frozen=True)
class AsrScore:
    backend: str
    cer: float
    runtime_s: float


def transcript_text(transcript: Transcript) -> str:
    return " ".join(s.text for s in transcript.segments)


def evaluate(
    ep: Episode, cfg: Config, subtitle_track: int, backends: list[str], language: str
) -> list[AsrScore]:
    """Run the upstream stages once, then the ASR stage per backend into eval/asr-<backend>."""
    from doppio.artifacts import read_json
    from doppio.job import run_stages
    from doppio.stages.asr import AsrStage
    from doppio.stages.extract import ExtractStage
    from doppio.stages.language import LanguageStage
    from doppio.stages.separate import SeparateStage

    reference = normalize(" ".join(extract_subtitle_text(ep, subtitle_track)), language)
    run_stages(ep, cfg, [ExtractStage(), SeparateStage(), LanguageStage()])
    scores: list[AsrScore] = []
    for b in backends:
        stage = AsrStage(backend=b, out_dir=f"eval/asr-{b}")
        started = time.monotonic()
        run_stages(ep, cfg, [stage])
        runtime = time.monotonic() - started
        (path,) = stage.outputs(ep)
        hyp = normalize(transcript_text(read_json(path, Transcript)), language)
        scores.append(AsrScore(b, cer(reference, hyp), runtime))
    return scores
