"""Dub cues: what each line says, who says it, and how many characters its slot can hold.

The ASR path builds cues from the transcript's word timings and the diarization turns the
characters stage wrote with its assignments. A cue ends at a speaker change, at a pause over
``gap_s``, or after sentence-final punctuation once it has run ``max_cue_s``; a cue that
reaches twice ``max_cue_s`` with no such punctuation ends at the next word anyway.

Word-level ASR output (Qwen3's forced aligner especially) drops punctuation from the words,
though the segment text keeps it; the punctuation between two words of the segment text is
put back on the word before it, so cues keep their sentence ends and the translator sees them.
"""

from __future__ import annotations

import math
import re
import unicodedata

from fukikae.models import (
    Assignments,
    ClusterAssignment,
    Delivery,
    DubCue,
    Segment,
    Tags,
    TagSpan,
    Transcript,
    Turn,
    Word,
)

# Speakable characters per second of the target language at normal speed.
CPS = {"en": 15.0, "de": 14.0, "es": 15.0, "fr": 15.0, "ja": 8.0, "ko": 8.0, "zh": 6.0}
DEFAULT_CPS = 14.0
MIN_BUDGET = 4

# Languages written without spaces between words.
_NO_SPACE = frozenset({"ja", "zh", "yue", "th", "lo", "km", "my"})
_FINAL = frozenset("。．！？!?…‥.")  # sentence ends; the ideographic comma 、 is not one
_INTERJECTION = frozenset("！？!?〜～")  # marks of a spoken interjection
_MANNER = re.compile(r"[(（][^()（）]{1,40}[)）]")
# A word under no turn takes the nearest turn this close, else it is an unknown extra.
NEAREST_TURN_S = 1.0
# A word stamped shorter than this (a forced aligner's per-character point) sounds until the
# next word starts, for at most WORD_SOUND_S from its start.
POINT_STAMP_S = 0.1
WORD_SOUND_S = 0.3
# A cue of at most this many characters (punctuation aside) that touches a neighbour of the
# same ASR segment within FRAGMENT_GAP_S may be a piece of that neighbour (see _join_fragments)
# neighbour cut off by a turn boundary falling inside a word.
FRAGMENT_CHARS = 2
FRAGMENT_S = 0.35
FRAGMENT_GAP_S = 0.3


def budget_chars(seconds: float, language: str, max_speed: float) -> int:
    """``floor(seconds * CPS * max_speed)``, at least 4."""
    cps = CPS.get(language, DEFAULT_CPS)
    return max(MIN_BUDGET, math.floor(seconds * cps * max_speed))


def _is_punct(text: str) -> bool:
    chars = [ch for ch in text if not ch.isspace()]
    return bool(chars) and all(unicodedata.category(ch).startswith("P") for ch in chars)


def _segment_words(seg: Segment) -> list[Word]:
    """The segment's words with the segment text's punctuation put back on them."""
    if not seg.words:
        return [Word(start=seg.start, end=seg.end, text=seg.text)] if seg.text.strip() else []
    text = seg.text
    out: list[Word] = []
    pos = 0
    for w in seg.words:
        token = w.text.strip()
        at = text.find(token, pos) if token else -1
        if at >= 0:
            between = text[pos:at]
            if out and _is_punct(between):
                out[-1] = out[-1].model_copy(update={"text": out[-1].text + between.strip()})
            pos = at + len(token)
        out.append(w)
    tail = text[pos:]
    if out and _is_punct(tail):
        out[-1] = out[-1].model_copy(update={"text": out[-1].text + tail.strip()})
    return out


def _join(words: list[Word], language: str) -> str:
    if any(w.text[:1].isspace() for w in words):  # whisper-style tokens carry their spaces
        text = "".join(w.text for w in words)
    elif language in _NO_SPACE:
        text = "".join(w.text.strip() for w in words)
    else:
        text = " ".join(w.text.strip() for w in words)
    return re.sub(r"\s+", " ", text).strip()


def _turn_at(turns: list[Turn], t: float) -> Turn | None:
    covering = [u for u in turns if u.start <= t <= u.end]
    if covering:
        return min(covering, key=lambda u: abs((u.start + u.end) / 2 - t))
    near = [(min(abs(u.start - t), abs(u.end - t)), u) for u in turns]
    near = [(d, u) for d, u in near if d <= NEAREST_TURN_S]
    return min(near, key=lambda p: p[0])[1] if near else None


def _who(
    turn: Turn | None, by_cluster: dict[str, ClusterAssignment]
) -> tuple[int | None, str | None]:
    """(character_id, extra) for the turn's cluster."""
    if turn is None:
        return None, "unknown"
    a = by_cluster.get(turn.speaker)
    if a is None or a.status in ("extra", "review") or a.character_id is None:
        return None, turn.speaker
    return a.character_id, None


# A run of words for one speaker: ((character_id, extra), [(segment index, word), ...]).
type _Group = tuple[tuple[int | None, str | None], list[tuple[int, Word]]]


def _near(out: list[_Group], i: int, j: int) -> bool:
    """Whether group j touches group i (within FRAGMENT_GAP_S) inside one of i's segments."""
    if not 0 <= j < len(out):
        return False
    pairs, other = out[i][1], out[j][1]
    if j < i:
        gap, seg = pairs[0][1].start - other[-1][1].end, other[-1][0]
    else:
        gap, seg = other[0][1].start - pairs[-1][1].end, other[0][0]
    return gap <= FRAGMENT_GAP_S and seg in {p[0] for p in pairs}


def _text(group: _Group) -> str:
    return "".join(w.text for _, w in group[1]).strip()


def _can_fold(out: list[_Group], i: int, j: int) -> bool:
    """Whether fragment i may fold into neighbour j.

    Only a fragment under FRAGMENT_S, and never back into a sentence the neighbour has
    closed. Then either (a) a mid-word sliver: no punctuation on either side of the junction
    and no interjection mark in the fragment, which may end a sentence (it began in the cue
    before); or (b) a piece with no sentence-end or interjection mark at all. A lone "え？"
    between another speaker's lines is neither.
    """
    if not _near(out, i, j):
        return False
    frag, other = _text(out[i]), _text(out[j])
    if not frag or not other:
        return False
    junction = (other[-1], frag[0]) if j < i else (frag[-1], other[0])
    marked = any(ch in _FINAL or ch in _INTERJECTION for ch in frag)
    sliver = not any(_is_punct(ch) for ch in junction) and not any(
        ch in _INTERJECTION for ch in frag
    )
    ws = [w for _, w in out[i][1]]
    if ws[-1].end - ws[0].start >= FRAGMENT_S:
        return False  # a piece that long was spoken as its own line
    if j < i and other[-1] in _FINAL:
        return False  # never into a sentence the neighbour has closed
    return sliver or not marked


def _join_fragments(groups: list[_Group]) -> list[_Group]:
    """Fold turn-boundary fragments into a neighbour of the same segment (see _can_fold): one
    that ends a sentence only into the cue before it, any other forward first."""
    out = list(groups)
    i = 0
    while i < len(out):
        frag = _text(out[i])
        bare = "".join(ch for ch in frag if not _is_punct(ch) and not ch.isspace())
        if len(bare) > FRAGMENT_CHARS:
            i += 1
            continue
        order = [i - 1] if frag[-1:] in _FINAL else [i + 1, i - 1]
        j = next((j for j in order if _can_fold(out, i, j)), None)
        if j is None:
            i += 1
            continue
        key, other = out[j]
        pairs = out[i][1]
        merged = other + pairs if j < i else pairs + other
        out[min(i, j)] = (key, merged)
        del out[max(i, j)]
        i = max(min(i, j), 0)
    return out


def _overlap(span: TagSpan, start: float, end: float) -> float:
    return max(0.0, min(span.end, end) - max(span.start, start))


def build_asr_cues(
    transcript: Transcript,
    assignments: Assignments,
    tags: Tags | None,
    names: dict[int, str],
    target_language: str,
    max_speed: float,
    *,
    gap_s: float = 0.6,
    max_cue_s: float = 8.0,
) -> list[DubCue]:
    """Cues from word timings; see the module docstring for where a cue ends."""
    turns = assignments.turns or []
    by_cluster = {a.cluster: a for a in assignments.clusters}
    # (segment index, speaker-time midpoint, word with its sounding end)
    stamped = sorted(
        ((i, w) for i, seg in enumerate(transcript.segments) for w in _segment_words(seg)),
        key=lambda p: (p[1].start, p[1].end),
    )
    items: list[tuple[int, float, Word]] = []
    for n, (i, w) in enumerate(stamped):
        mid = (w.start + w.end) / 2
        if w.end - w.start < POINT_STAMP_S:
            nxt = stamped[n + 1][1].start if n + 1 < len(stamped) else math.inf
            w = w.model_copy(update={"end": max(w.end, min(nxt, w.start + WORD_SOUND_S))})
        items.append((i, mid, w))

    groups: list[_Group] = []
    for seg_i, mid, w in items:
        key = _who(_turn_at(turns, mid), by_cluster)
        if groups:
            cur_key, cur = groups[-1]
            prev = cur[-1][1]
            length = prev.end - cur[0][1].start
            split = (
                key != cur_key
                or w.start - prev.end > gap_s
                or (prev.text.strip()[-1:] in _FINAL and length >= max_cue_s)
                or w.end - cur[0][1].start > 2 * max_cue_s
            )
            if not split:
                cur.append((seg_i, w))
                continue
        groups.append((key, [(seg_i, w)]))
    groups = _join_fragments(groups)

    events = [e for e in (tags.events if tags else []) if e.label == "singing"]
    emotions = tags.emotions if tags else []
    cues: list[DubCue] = []
    for (character_id, extra), pairs in groups:
        ws = [w for _, w in pairs]
        start, end = ws[0].start, max(w.end for w in ws)
        mid = (start + end) / 2
        if character_id is not None:
            speaker = names.get(character_id) or f"char{character_id}"
        else:
            speaker = f"extra:{extra}"
        emotion = max(emotions, key=lambda s: _overlap(s, start, end), default=None)
        if emotion is not None and _overlap(emotion, start, end) <= 0:
            emotion = None
        cues.append(
            DubCue(
                id=f"c{len(cues) + 1:04d}",
                start=start,
                end=end,
                character_id=character_id,
                extra=extra,
                speaker=speaker,
                source_text=_join(ws, transcript.language),
                budget=budget_chars(end - start, target_language, max_speed),
                delivery=Delivery(
                    reference=(start, end),
                    emotion=None if emotion is None else emotion.label,
                    emotion_confidence=None if emotion is None else emotion.confidence,
                ),
                skip="song" if any(e.start <= mid <= e.end for e in events) else None,
            )
        )
    return cues


def windows(cues: list[DubCue], *, gap_s: float = 2.0, max_cues: int = 12) -> list[list[DubCue]]:
    """Scene-sized groups: a new window after a pause over ``gap_s`` or ``max_cues`` cues."""
    out: list[list[DubCue]] = []
    for c in cues:
        if out and len(out[-1]) < max_cues and c.start - out[-1][-1].end <= gap_s:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def is_marked(cue: DubCue) -> bool:
    """Marked delivery: a non-neutral emotion with confidence at least 0.5 (an unscored label,
    which is all SenseVoice gives, counts), or a parenthesized manner note in the source."""
    d = cue.delivery
    scored = d.emotion_confidence is None or d.emotion_confidence >= 0.5
    if d.emotion and d.emotion != "neutral" and scored:
        return True
    return bool(_MANNER.search(cue.source_text))
