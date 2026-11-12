"""Stock ASR hallucinations: the outro phrases Whisper-style models write over music or silence.

Trained on subtitled video, an ASR model fills a stretch with no speech (an ending song, a
silent credit roll) with what subtitles say there: 「ご視聴ありがとうございました」, "thanks for
watching", a subtitler's credit. Sol Bianca S00E01's transcript has 「ご視聴ありがとうございました」
twice over the ending song (3774-3801 s and 3808-3819 s by its words), which became a character
of its own (char 11, 12 cues), and one 「お疲れ様でした」 segment stretched over 3517-3774 s
whose seven words sit in two runs, お疲 at 3564-3570 s over sung music and れ様でした at
3677-3684 s inside the theme song (3658-3740 s). The asrclean stage drops all three.

A stretch is dropped only when both hold:

- the segment's text, folded (`fold`: NFKC, casefold, spaces, punctuation and symbols
  stripped), is a phrase of `HALLUCINATION_PHRASES` for the transcript's language or that
  phrase repeated (`is_stock_phrase`);
- the stretch is music or no speech (`span_reason`): at least ``HALLUCINATION_THEME_SHARE`` of
  it inside a theme span from the song detector, or at least ``HALLUCINATION_MUSIC_SHARE`` of
  its seconds song-like by the detector's per-second test with its vocals sung (held notes,
  `fukikae.pitch.is_sung`), or the vocals stem active on under ``HALLUCINATION_ACTIVE_MAX`` of
  its 100 ms frames.

The per-second test alone reads loud background music under speech as song-like (the bed
within a few dB of the voice), so a character saying 「お疲れ様でした」 over a loud score would
match it; the sung condition keeps that line. A stock phrase spoken over music is kept and
flagged, and so is the surviving fragment of a segment that lost some of its runs
(`clean_segment`'s flags), for the operator to check.

The phrase list alone would delete a character who says 「ご視聴ありがとうございました」 to camera
at the end of an episode; the music-or-silence condition keeps that line. A segment with word
timings is judged by word runs (`clean_segment`, `word_runs`: words split at gaps over
``RUN_GAP_S``), since Whisper stretches one hallucinated segment over a whole silence and its
words are what becomes cues: each run over music or no speech goes, the segment keeps the other
runs' words with its text rewritten from them, and goes whole when every run does. A segment
without words is judged on its own start and end.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fukikae.models import Segment, Word
from fukikae.songs import FRAME_S, SongSpan

# Each entry says where it was seen. "Seen" without an episode means the phrase is a known
# Whisper hallucination on silence or music in that language (the training subtitles' outros),
# reported widely; entries are added when a transcript shows one.
HALLUCINATION_PHRASES: dict[str, tuple[str, ...]] = {
    "ja": (
        "ご視聴ありがとうございました",  # Sol Bianca S00E01, twice over the ending song (char 11)
        "お疲れ様でした",  # Sol Bianca S00E01, 3564-3570 s (sung music), 3677-3684 s (theme)
        "ご覧いただきありがとうございました",  # the spec's list; Whisper outro on silence
        "チャンネル登録よろしくお願いします",  # the spec's list; YouTube outro on silence
        "最後までご視聴いただきありがとうございました",  # Whisper outro on silence
    ),
    "en": (
        "thank you for watching",  # the spec's list; Whisper outro on silence
        "thanks for watching",  # the spec's list; Whisper outro on silence
        "subtitles by the amara.org community",  # the spec's list; subtitle credit on music
        "please subscribe",  # the spec's list; YouTube outro on silence
        "thank you for watching and see you next time",  # Whisper outro on silence
    ),
    "zh": (
        "谢谢观看",  # Whisper outro on silence
        "感谢观看",  # Whisper outro on silence
        "请不吝点赞订阅转发打赏支持明镜与点点栏目",  # Whisper credit on music, widely reported
        "字幕由amara.org社区提供",  # subtitle credit on music
    ),
    "ko": (
        "시청해주셔서 감사합니다",  # Whisper outro on silence
        "구독과 좋아요 부탁드립니다",  # YouTube outro on silence
    ),
    "de": (
        "danke fürs zuschauen",  # Whisper outro on silence
        "untertitel im auftrag des zdf für funk",  # Whisper subtitle credit on music
        "untertitel der amara.org-community",  # subtitle credit on music
    ),
    "es": (
        "gracias por ver el video",  # Whisper outro on silence
        "subtítulos realizados por la comunidad de amara.org",  # subtitle credit on music
        "suscríbete al canal",  # YouTube outro on silence
    ),
}

HALLUCINATION_THEME_SHARE = 0.5
HALLUCINATION_MUSIC_SHARE = 0.6
HALLUCINATION_ACTIVE_MAX = 0.3
# A suspect segment's words split into runs at gaps longer than this, and each run is judged
# on its own span: Whisper spreads one hallucinated segment over minutes (Sol Bianca's
# 「お疲れ様でした」 has お疲 at 3564-3570 s and れ様でした at 3677-3684 s, the second inside the
# theme song), and words of one run there sit up to 5.5 s apart.
RUN_GAP_S = 10.0
# Languages whose kept words join with no space when a segment's text is rewritten.
NO_SPACE_LANGUAGES = frozenset({"ja", "zh", "yue"})
VERSION = 4

# Whether the vocals over ``lo``-``hi`` seconds are sung (`fukikae.pitch.is_sung` on their F0
# track); the asrclean stage builds one from the vocals stem.
SungTest = Callable[[float, float], bool]


def fold(text: str) -> str:
    """NFKC, casefolded, with spaces, punctuation, symbols and control characters stripped."""
    t = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in t if unicodedata.category(ch)[0] not in "PZSC")


def _language(language: str) -> str:
    return language.split("-")[0].split("_")[0].lower()


def _folded(language: str) -> tuple[str, ...]:
    return tuple(
        f for f in (fold(p) for p in HALLUCINATION_PHRASES.get(_language(language), ())) if f
    )


def is_stock_phrase(text: str, language: str) -> bool:
    """Whether the folded ``text`` equals a phrase for ``language`` or is that phrase repeated."""
    f = fold(text)
    if not f:
        return False
    return any(len(f) % len(p) == 0 and f == p * (len(f) // len(p)) for p in _folded(language))


def phrase_table_digest() -> str:
    """sha256 of the phrase table, for the asrclean stage's key."""
    blob = json.dumps(HALLUCINATION_PHRASES, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def rule_constants() -> dict[str, Any]:
    """The phrase table's digest and the music-or-silence thresholds."""
    return {
        "phrases": phrase_table_digest(),
        "theme_share": HALLUCINATION_THEME_SHARE,
        "music_share": HALLUCINATION_MUSIC_SHARE,
        "active_max": HALLUCINATION_ACTIVE_MAX,
        "run_gap_s": RUN_GAP_S,
        "version": VERSION,
    }


def segment_span(seg: Segment) -> tuple[float, float]:
    """Its words' extent when it has words, else its own start and end."""
    if seg.words:
        return min(w.start for w in seg.words), max(w.end for w in seg.words)
    return seg.start, seg.end


def _share(mask: np.ndarray, lo: float, hi: float, step: float) -> float | None:
    a = max(0, math.floor(lo / step))
    b = min(len(mask), max(a + 1, math.ceil(hi / step)))
    if b <= a:
        return None
    return float(np.asarray(mask[a:b], dtype=np.float64).mean())


def over_music(lo: float, hi: float, song_like: np.ndarray) -> bool:
    """Whether at least ``HALLUCINATION_MUSIC_SHARE`` of ``lo``-``hi``'s seconds are song-like
    by the detector's per-second mask (loud music under speech passes this too)."""
    music = _share(song_like, lo, hi, 1.0)
    return music is not None and music >= HALLUCINATION_MUSIC_SHARE


def span_reason(
    lo: float,
    hi: float,
    song_spans: Sequence[SongSpan],
    song_like: np.ndarray,
    active: np.ndarray,
    sung: SungTest | None = None,
) -> str | None:
    """Why the span ``lo``-``hi`` is music or no speech, or None when it is neither (or the
    arrays give no evidence there). The music branch needs the span `over_music` and its
    vocals sung by ``sung``; without a ``sung`` test nothing counts as sung, so that branch
    never drops."""
    length = max(hi - lo, 1e-6)
    inside = sum(max(0.0, min(hi, s.end) - max(lo, s.start)) for s in song_spans)
    if inside / length >= HALLUCINATION_THEME_SHARE:
        return "stock phrase over a theme song"
    if over_music(lo, hi, song_like) and sung is not None and sung(lo, hi):
        return "stock phrase over sung music"
    on = _share(active, lo, hi, FRAME_S)
    if on is not None and on < HALLUCINATION_ACTIVE_MAX:
        return "stock phrase over no speech"
    return None


def drop_reason(
    seg: Segment,
    language: str,
    song_spans: Sequence[SongSpan],
    song_like: np.ndarray,
    active: np.ndarray,
    sung: SungTest | None = None,
) -> str | None:
    """Why ``seg`` as a whole is a stock hallucination to drop, or None to keep it, on its
    `segment_span`. ``song_like`` is the detector's per-second mask (`songs.song_like_mask`),
    ``active`` the vocals stem's per 100 ms frame activity (`songs.active_frames`), ``sung``
    the sung test (`span_reason`); seconds or frames past their end give no evidence. The
    asrclean stage uses `clean_segment`, which judges each word run apart."""
    if not is_stock_phrase(seg.text, language):
        return None
    lo, hi = segment_span(seg)
    return span_reason(lo, hi, song_spans, song_like, active, sung)


def word_runs(words: Sequence[Word], gap: float = RUN_GAP_S) -> list[list[Word]]:
    """The words in time order, split where the next starts more than ``gap`` seconds after
    the latest end of the run so far."""
    runs: list[list[Word]] = []
    end = -math.inf
    for w in sorted(words, key=lambda w: (w.start, w.end)):
        if runs and w.start - end <= gap:
            runs[-1].append(w)
            end = max(end, w.end)
        else:
            runs.append([w])
            end = w.end
    return runs


@dataclass(frozen=True)
class Drop:
    """A dropped stretch of a segment: its span, text, why, and whether it is the whole
    segment (else one word run of a segment that keeps its other runs)."""

    start: float
    end: float
    text: str
    reason: str
    whole: bool = True


@dataclass(frozen=True)
class Flag:
    """A kept stretch of a stock-phrase segment the operator should check: its span, text and
    why it was flagged."""

    start: float
    end: float
    text: str
    reason: str


FLAG_FRAGMENT = "surviving fragment of a stock phrase"
FLAG_SPOKEN = "stock phrase spoken over music (not sung)"


def _join(words: Sequence[Word], language: str) -> str:
    if _language(language) in NO_SPACE_LANGUAGES:
        return "".join(w.text.strip() for w in words)
    return " ".join(w.text.strip() for w in words if w.text.strip())


def _flag_reasons(lo: float, hi: float, song_like: np.ndarray, fragment: bool) -> list[str]:
    reasons = [FLAG_FRAGMENT] if fragment else []
    if over_music(lo, hi, song_like):
        reasons.append(FLAG_SPOKEN)
    return reasons


def clean_segment(
    seg: Segment,
    language: str,
    song_spans: Sequence[SongSpan],
    song_like: np.ndarray,
    active: np.ndarray,
    sung: SungTest | None = None,
) -> tuple[Segment | None, list[Drop], list[Flag]]:
    """``seg`` with its stock-hallucination stretches removed, what went, and what was kept but
    flagged. Only a segment whose whole text is a stock phrase (`is_stock_phrase`) is touched.
    With word timings its words split into runs (`word_runs`) and each run over music or no
    speech (`span_reason` on the run's extent, with ``sung``) goes: every run going drops the
    segment (None, one `Drop` with its whole text); some going leaves the segment with the
    other runs' words, its text rewritten from them and its span their extent. Without words
    the segment is judged whole. Each kept run is flagged when it is a surviving fragment
    (another run of its segment went) or lies over music it doesn't sing (`over_music`
    without ``sung``), one `Flag` per run with its reasons joined by "; "."""
    if not is_stock_phrase(seg.text, language):
        return seg, [], []
    if not seg.words:
        reason = span_reason(seg.start, seg.end, song_spans, song_like, active, sung)
        if reason is None:
            notes = _flag_reasons(seg.start, seg.end, song_like, fragment=False)
            flags = [Flag(seg.start, seg.end, seg.text, "; ".join(notes))] if notes else []
            return seg, [], flags
        return None, [Drop(seg.start, seg.end, seg.text, reason)], []
    runs = word_runs(seg.words)
    extents = [(r[0].start, max(w.end for w in r)) for r in runs]
    verdicts = [span_reason(a, b, song_spans, song_like, active, sung) for a, b in extents]
    if all(v is not None for v in verdicts):
        lo, hi = segment_span(seg)
        reasons = list(dict.fromkeys(v for v in verdicts if v is not None))
        return None, [Drop(lo, hi, seg.text, "; ".join(reasons))], []
    fragment = any(v is not None for v in verdicts)
    flags = []
    for r, (a, b), v in zip(runs, extents, verdicts, strict=True):
        notes = _flag_reasons(a, b, song_like, fragment) if v is None else []
        if notes:
            flags.append(Flag(a, b, _join(r, language), "; ".join(notes)))
    if not fragment:
        return seg, [], flags
    kept = [w for r, v in zip(runs, verdicts, strict=True) if v is None for w in r]
    drops = [
        Drop(a, b, _join(r, language), v, whole=False)
        for r, (a, b), v in zip(runs, extents, verdicts, strict=True)
        if v is not None
    ]
    new = seg.model_copy(
        update={
            "words": kept,
            "text": _join(kept, language),
            "start": kept[0].start,
            "end": max(w.end for w in kept),
        }
    )
    return new, drops, flags
