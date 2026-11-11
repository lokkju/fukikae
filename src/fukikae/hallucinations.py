"""Stock ASR hallucinations: the outro phrases Whisper-style models write over music or silence.

Trained on subtitled video, an ASR model fills a stretch with no speech (an ending song, a
silent credit roll) with what subtitles say there: 「ご視聴ありがとうございました」, "thanks for
watching", a subtitler's credit. Sol Bianca S00E01's transcript has 「ご視聴ありがとうございました」
twice over the ending song (3774-3819 s) and 「お疲れ様でした」 with seven words spread over two
minutes of the credits; they became a character of their own (char 11, 12 cues) and five junk
cues on two others.

A segment is dropped (`drop_reason`) only when both hold:

- its text, folded (`fold`: NFKC, casefold, spaces, punctuation and symbols stripped), is a
  phrase of `HALLUCINATION_PHRASES` for the transcript's language or that phrase repeated
  (`is_stock_phrase`);
- its span is music or no speech: at least ``HALLUCINATION_THEME_SHARE`` of it inside a theme
  span from the song detector, or at least ``HALLUCINATION_MUSIC_SHARE`` of its seconds
  song-like by the detector's per-second test, or the vocals stem active on under
  ``HALLUCINATION_ACTIVE_MAX`` of its 100 ms frames.

The phrase list alone would delete a character who says 「ご視聴ありがとうございました」 to camera
at the end of an episode; the music-or-silence condition keeps that line. A segment's span is
its words' extent when it has word timings (Whisper stretches a hallucinated segment over the
whole silence; its words are what becomes cues), else the segment's own start and end.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Sequence
from typing import Any

import numpy as np

from fukikae.models import Segment
from fukikae.songs import FRAME_S, SongSpan

# Each entry says where it was seen. "Seen" without an episode means the phrase is a known
# Whisper hallucination on silence or music in that language (the training subtitles' outros),
# reported widely; entries are added when a transcript shows one.
HALLUCINATION_PHRASES: dict[str, tuple[str, ...]] = {
    "ja": (
        "ご視聴ありがとうございました",  # Sol Bianca S00E01, twice over the ending song
        "お疲れ様でした",  # Sol Bianca S00E01, 3517-3774 s, seven words over the credits
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
VERSION = 1


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


def drop_reason(
    seg: Segment,
    language: str,
    song_spans: Sequence[SongSpan],
    song_like: np.ndarray,
    active: np.ndarray,
) -> str | None:
    """Why ``seg`` is a stock hallucination to drop, or None to keep it. ``song_like`` is the
    detector's per-second mask (`songs.song_like_mask`), ``active`` the vocals stem's per
    100 ms frame activity (`songs.active_frames`); seconds or frames past their end give no
    evidence."""
    if not is_stock_phrase(seg.text, language):
        return None
    lo, hi = segment_span(seg)
    length = max(hi - lo, 1e-6)
    inside = sum(max(0.0, min(hi, s.end) - max(lo, s.start)) for s in song_spans)
    if inside / length >= HALLUCINATION_THEME_SHARE:
        return "stock phrase over a theme song"
    music = _share(song_like, lo, hi, 1.0)
    if music is not None and music >= HALLUCINATION_MUSIC_SHARE:
        return "stock phrase over music"
    on = _share(active, lo, hi, FRAME_S)
    if on is not None and on < HALLUCINATION_ACTIVE_MAX:
        return "stock phrase over no speech"
    return None
