"""Where an episode's transcript is read from: the asrclean stage's, or the ASR stage's for an
episode that predates asrclean.

Every consumer of the transcript (the translate stage's cue builder, reference selection in
`fukikae.cards.auto`, the extras cards, the evals) reads it through `transcript_path`, so a
stock ASR hallucination the asrclean stage dropped never becomes a cue, a reference or a
character.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fukikae.workspace import Episode

log = logging.getLogger(__name__)

ASR_TRANSCRIPT = Path("03-asr") / "transcript.json"
CLEAN_TRANSCRIPT = Path("03b-asrclean") / "transcript.json"

_WARNED: set[str] = set()


def transcript_path(ep: Episode | Path) -> Path:
    """``03b-asrclean/transcript.json`` of an episode (or its work directory); the ASR stage's
    ``03-asr/transcript.json``, with a warning once per episode, when the cleaned one isn't
    there (an episode from before the asrclean stage, or one whose asrclean hasn't run)."""
    work = Path(ep) if isinstance(ep, Path) else ep.work
    clean = work / CLEAN_TRANSCRIPT
    if clean.exists():
        return clean
    raw = work / ASR_TRANSCRIPT
    if raw.exists() and str(work) not in _WARNED:
        _WARNED.add(str(work))
        log.warning(
            "%s predates the asrclean stage; reading %s (stock ASR hallucinations not removed;"
            " `fukikae run` adds the stage without rerunning ASR)",
            work.name, ASR_TRANSCRIPT,
        )  # fmt: skip
    return raw
