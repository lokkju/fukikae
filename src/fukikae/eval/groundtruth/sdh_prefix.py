"""Speaker tags in SDH/closed captions: `NAME:`, `(NAME)`, `[NAME]`, `- NAME:`."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from fukikae.eval.groundtruth.base import (
    Cue,
    GroundTruthLine,
    Namer,
    split_times,
    strip_markup,
)

DEFAULT_SOUNDS = frozenset(
    {
        "music", "музыка", "음악", "音乐", "音樂", "laughs", "laughter", "laughing", "смеётся",
        "смеется", "웃음", "笑", "applause", "sighs", "gasps", "screams", "sfx", "silence",
        "тишина", "аплодисменты", "박수", "掌声", "掌聲",
        "grunts", "groans", "chuckles", "whispers", "cries", "sobs", "yells", "shouts",
        "ríe", "susurra", "susurrando", "asiente", "gime", "recita", "silabeando", "suspira",
        "llora", "grita", "todos", "ambos", "música", "risas", "aplausos",
        "sigh", "gasp", "scoffs", "mutters", "whimpers", "snores", "coughs", "yelling",
        "radio", "tv", "vídeo", "video", "смех", "вздох", "вздыхает", "кричит",
        "웃", "한숨", "음성", "소리", "울음", "신음", "경적", "자막", "함께", "音声", "声",
        "音楽", "拍手", "足音", "物音", "爆音", "雑音", "銃声", "歓声", "罵声", "悲鳴", "吐息",
        "ため息", "笑い", "クラクション", "ノック", "アナウンス", "男性", "女性", "一同", "全員",
    }
)  # fmt: skip

# A tag starting with one of these words is a fragment or a language note, not a name.
LEADING_WORDS = frozenset(
    {"in", "on", "over", "through", "with", "from", "at", "of", "for", "to", "en", "sur", "por",
     "по", "в", "acento", "voz", "sound", "sounds"}
)  # fmt: skip
_CJK = re.compile(r"[\u1100-\u11ff\u3040-\u30ff\u3130-\u318f\u3400-\u9fff\uac00-\ud7af]")
# `A과 B`, `A와 B`, `A랑 B` (Korean) and `AとB`, `A和B` (Japanese, Chinese): several speakers.
_PAIR = re.compile(
    r"[\uac00-\ud7af]+[과와랑]\s+[\uac00-\ud7af]+"
    r"|[\u3040-\u30ff\u3400-\u9fff]{2,}[と和与跟][\u3040-\u30ff\u3400-\u9fff]{2,}"
)
_SEVERAL = re.compile(r"[・･、／]")  # `アリス･チョータ`: several speakers
_SOUND_END = re.compile(r".{2,}[音声息鳴]$")  # 通知音, 荒い息, ふざけ合う声; 琴音 is two characters
_GROUP = re.compile(r"(?:[0-9０-９]+人|たち|達)$")  # ３人, アリスたち
_KANJI_RUBY = re.compile(r"(?<=[\u3400-\u9fff々])[(（][\u3040-\u30ffー]+[)）]")
_HANGUL_ENDING = re.compile(r"(?:하며|으며|면서|하다|이다)$")  # whole description endings
_HANGUL_SHORT_ENDING = re.compile(
    r"[며다음]$"
)  # only on 3+ syllables or after another word: 울린다, 조작음, 놀라며

MAX_NAME_CHARS = 24
MAX_NAME_WORDS = 4
_NAME_CHARS = re.compile(r"[\w .'’\-]+")
_DASH = re.compile(r"^\s*[-–—]\s*")
_BRACKET = re.compile(r"^[(\[（【]\s*([^)\]）】]{1,40}?)\s*[)\]）】]\s*(.*)$", re.DOTALL)
_COLON = re.compile(r"^([^:：(\[]{1,40}?)\s*[:：]\s*(.*)$", re.DOTALL)


def _is_description(name: str, sounds: frozenset[str]) -> bool:
    folded = name.casefold()
    entries = {x.casefold() for x in sounds}
    words = folded.split()
    if folded in entries or any(w in entries for w in words):
        return True
    if any(_CJK.search(e) and e in folded for e in entries):
        return True  # no spaces in Hangul/CJK: a listed word anywhere in the tag
    if words[0] in LEADING_WORDS or _PAIR.search(name):
        return True
    last = words[-1]
    if not _CJK.search(last):
        return False
    if _SEVERAL.search(name) or _SOUND_END.search(last) or _GROUP.search(last):
        return True
    return bool(
        _HANGUL_ENDING.search(last)
        or ((len(last) >= 3 or len(words) >= 2) and _HANGUL_SHORT_ENDING.search(last))
    )


def valid_name(name: str, sounds: frozenset[str], upper_only: bool = False) -> bool:
    """1-4 words, at most 24 characters, letters of any script (digits, dots, hyphens and
    apostrophes allowed), not a sound description, not all lowercase. `upper_only` demands
    capitals for cased scripts (the `NAME:` form, which plain sentences also resemble)."""
    name = name.strip()
    if not name or len(name) > MAX_NAME_CHARS or len(name.split()) > MAX_NAME_WORDS:
        return False
    if not _NAME_CHARS.fullmatch(name) or not any(c.isalpha() for c in name):
        return False
    if sounds and _is_description(name, sounds):
        return False
    cased = name.upper() != name.lower()
    if cased and name == name.lower():
        return False
    return not (cased and upper_only and name != name.upper())


def strip_ruby(text: str) -> str:
    """Remove a `(かな)` or `（かな）` run (hiragana, katakana, ー only) that directly follows a
    kanji: `有栖(ありす)` -> `有栖`."""
    return _KANJI_RUBY.sub("", text)


def ja_short_names(names: Iterable[str]) -> dict[str, str]:
    """A two-word CJK tag (`有栖 始`, family and given name) maps to its last word when that word
    is also a tag on its own in the same file. Hangul tags too (`김 철수` -> `철수`): Korean SDH
    uses given names as well."""
    pool = set(names)
    out: dict[str, str] = {}
    for name in pool:
        words = name.split()
        if len(words) == 2 and _CJK.search(name) and words[1] in pool:
            out[name] = words[1]
    return out


def joins_known_tags(name: str, pool: set[str]) -> bool:
    """`始と薫`: a joiner with a tag of the same file on each side. Single-character names escape
    the 2+ characters rule of `_PAIR`, so the file's own tags tell a pair from a compound name."""
    for m in re.finditer(r"[と和与跟]", name):
        a, b = name[: m.start()], name[m.end() :]
        if a in pool and b in pool:
            return True
    return False


def extract_tag(line: str, sounds: frozenset[str] = DEFAULT_SOUNDS) -> tuple[str, str] | None:
    """(name, rest of the line) when `line` starts with a speaker tag, after markup and a
    leading dash are removed."""
    text = _DASH.sub("", strip_ruby(strip_markup(line)).replace("\u3000", " ")).strip()
    m = _BRACKET.match(text)
    if m and valid_name(m.group(1), sounds):
        return m.group(1).strip(), m.group(2).strip()
    m = _COLON.match(text)
    if m and valid_name(m.group(1), sounds, upper_only=True):
        return m.group(1).strip(), m.group(2).strip()
    return None


@dataclass(frozen=True)
class Segment:
    """One speaker's share of a cue: the lines from a dash or tag to the next, as plain text
    with the tag and dash removed."""

    name: str | None
    text: str


def cue_segments(text: str, sounds: frozenset[str] = DEFAULT_SOUNDS) -> list[Segment]:
    """Split a cue where a dash starts a new speaker or a tag appears; a tag carries over to
    the lines that follow it. A tag with no text after it in its segment is not a tag (it is
    a sound description such as `(Relinchos)`)."""
    segs: list[tuple[str | None, list[str]]] = []
    for line in text.split("\n"):
        plain = strip_markup(line).strip()
        if not plain:
            continue
        tag = extract_tag(line, sounds)
        dash = bool(_DASH.match(plain))
        if dash or tag is not None or not segs:
            segs.append((tag[0] if tag else None, []))
        segs[-1][1].append(tag[1] if tag else _DASH.sub("", plain, count=1) if dash else plain)
    out = []
    for name, lines in segs:
        body = " ".join(x for x in lines if x).strip()
        out.append(Segment(name if body else None, body))
    return out


def read_sdh(
    cues: list[Cue], namer: Namer, sounds: frozenset[str] = DEFAULT_SOUNDS
) -> list[GroundTruthLine]:
    out: list[GroundTruthLine] = []
    per_cue = [cue_segments(cue.text, sounds) for cue in cues]
    pool = {s.name for segs in per_cue for s in segs if s.name}
    short = ja_short_names(pool)
    for cue, segs in zip(cues, per_cue, strict=True):
        spans = split_times(
            cue.start, cue.end, [max(1, len(re.sub(r"\s", "", s.text))) for s in segs]
        )
        for seg, (a, b) in zip(segs, spans, strict=True):
            name = namer(short.get(seg.name, seg.name)) if seg.name else None
            if seg.name and joins_known_tags(seg.name, pool):
                name = None  # several speakers
            if name:
                out.append(GroundTruthLine(a, b, name))
    return out
