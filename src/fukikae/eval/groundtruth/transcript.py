"""Speaker-named transcripts with no timing, aligned to a release's own timed subtitles.

The subtitle words carry the timing and the transcript carries the names. Both sides are
tokenized and matched with `difflib`; every matched subtitle token takes its transcript token's
speaker, and a subtitle segment is named when most of it matched and one speaker holds most of
the matched tokens. A scene the broadcast cut or a line the transcript lacks matches nothing, so
it stays unnamed and the cues around it keep their own speakers."""

import difflib
import html
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

from fukikae.eval.asr import normalize
from fukikae.eval.groundtruth.base import Cue, Namer, split_times
from fukikae.eval.groundtruth.sdh_prefix import cue_segments

MIN_MATCHED = 0.5
MIN_MAJORITY = 0.6
TRANSCRIPT_SKIP = {"scene", "credits sequence", "teleplay", "story", "written by"}
GROUP_SPEAKERS = {"all", "both", "everyone", "together"}

_WORDS = re.compile(r"\w+")
_LINE = re.compile(r"^([^:]{1,40}):\s*(.+)$")
_PAREN = re.compile(r"\([^)]*\)")
_WHOLLY_PAREN = re.compile(r"^\(.*\)[\W_]*$")
_CJK_LANGUAGES = {"zh", "ja", "ko"}


@dataclass(frozen=True)
class TranscriptLine:
    speaker: str
    text: str


@dataclass(frozen=True)
class NamedLine:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class AlignResult:
    named: list[NamedLine]
    cue_share: float  # segments that got a speaker, of all segments with words
    word_share: float  # subtitle words that matched a transcript word
    speakers: dict[str, int]  # named lines per speaker


class _Paragraphs(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._buf: list[str] | None = None

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            self._buf = []
        elif tag == "br" and self._buf is not None:
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._buf is not None:
            self.paragraphs.append("".join(self._buf))
            self._buf = None

    def handle_data(self, data: str) -> None:
        if self._buf is not None:
            self._buf.append(data)


def _raw_lines(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    suffix = Path(path).suffix.casefold()
    if suffix in {".html", ".htm"}:
        parser = _Paragraphs()
        parser.feed(text)
        parser.close()
        return parser.paragraphs
    if suffix == ".txt":
        return [html.unescape(ln) for ln in text.splitlines()]
    raise ValueError(f"{path}: unsupported transcript format {suffix!r}; use .html, .htm or .txt")


def read_transcript(path: Path) -> list[TranscriptLine]:
    """`Name: line` paragraphs of a transcript page (`.html`/`.htm`) or text file (`.txt`);
    scene headings, credits and parenthesised directions are dropped, and parentheticals are
    removed from the speaker and the text."""
    out: list[TranscriptLine] = []
    for raw in _raw_lines(path):
        line = re.sub(r"\s+", " ", raw.replace("\xa0", " ")).strip()
        if not line or _WHOLLY_PAREN.match(line):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        speaker = re.sub(r"\s+", " ", _PAREN.sub("", m.group(1))).strip()
        if not speaker or speaker.casefold() in TRANSCRIPT_SKIP:
            continue
        text = re.sub(r"\s+", " ", _PAREN.sub("", m.group(2))).strip()
        if text:
            out.append(TranscriptLine(speaker, text))
    return out


def _is_group(speaker: str) -> bool:
    """Several speakers: any word of the name is a group word (`All together`, `Everyone`)."""
    return any(w in GROUP_SPEAKERS for w in _WORDS.findall(speaker.casefold()))


def _tokens(text: str, language: str) -> list[str]:
    norm = normalize(text, language)
    return list(norm) if language.split("-")[0] in _CJK_LANGUAGES else norm.split()


def align_transcript(
    lines: list[TranscriptLine], cues: list[Cue], namer: Namer, language: str = "en"
) -> AlignResult:
    """Name the speaker of each subtitle segment (a cue, or one dash-separated part of it)."""
    ref: list[str] = []
    ref_speaker: list[str | None] = []
    for ln in lines:
        name = None if _is_group(ln.speaker) else namer(ln.speaker)
        toks = _tokens(ln.text, language)
        ref += toks
        ref_speaker += [name] * len(toks)

    segs: list[tuple[Cue, list[tuple[str, tuple[float, float], list[int]]]]] = []
    sub: list[str] = []
    for cue in cues:
        parts = cue_segments(cue.text)
        spans = split_times(
            cue.start, cue.end, [max(1, len(re.sub(r"\s", "", p.text))) for p in parts]
        )
        rows = []
        for p, span in zip(parts, spans, strict=True):
            toks = _tokens(p.text, language)
            if not toks:
                continue
            rows.append((p.text, span, list(range(len(sub), len(sub) + len(toks)))))
            sub += toks
        segs.append((cue, rows))

    owner: list[str | None] = [None] * len(sub)
    matched = [False] * len(sub)
    if sub and ref:
        sm = difflib.SequenceMatcher(None, sub, ref, autojunk=False)
        for block in sm.get_matching_blocks():
            for k in range(block.size):
                matched[block.a + k] = True
                owner[block.a + k] = ref_speaker[block.b + k]

    named: list[NamedLine] = []
    total = 0
    for _, rows in segs:
        for text, (a, b), idx in rows:
            total += 1
            hit = [owner[i] for i in idx if matched[i]]
            if len(hit) / len(idx) < MIN_MATCHED:
                continue
            top = Counter(hit).most_common(1)[0]
            if top[0] is None or top[1] / len(hit) < MIN_MAJORITY:
                continue
            named.append(NamedLine(a, b, top[0], text))
    return AlignResult(
        named,
        len(named) / total if total else 0.0,
        sum(matched) / len(sub) if sub else 0.0,
        dict(Counter(n.speaker for n in named)),
    )


def _stamp(t: float) -> str:
    cs = max(0, round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_named_ass(path: Path, lines: list[NamedLine]) -> None:
    """A minimal ASS file with the speaker in each Dialogue line's Name field."""
    head = (
        "[Script Info]\nScriptType: v4.00+\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,"
        "1,2,2,2,10,10,10,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    body = "".join(
        f"Dialogue: 0,{_stamp(ln.start)},{_stamp(ln.end)},Default,"
        f"{ln.speaker.replace(',', ' ')},0,0,0,,{ln.text.replace(chr(10), chr(92) + 'N')}\n"
        for ln in lines
    )
    Path(path).write_text(head + body, encoding="utf-8")
