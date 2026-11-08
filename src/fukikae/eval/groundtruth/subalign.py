"""Subtitle retiming: the offset and linear drift that map one capture's subtitles onto another
capture's timeline, found from a reference subtitle file of the target capture (cue onsets
matched by text) or from the target's detected speech, and a retimer that rewrites only the
timestamps of a subtitle file."""

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from fukikae.artifacts import atomic_path
from fukikae.eval.asr import normalize
from fukikae.eval.groundtruth.base import (
    MIN_SHARE,
    AlignmentError,
    Cue,
    _union,
    strip_markup,
)
from fukikae.models import Turn

ANCHOR_MIN_RATIO = 0.8
ANCHOR_MIN_GAP = 0.1  # the runner-up match must score at least this much lower
INLIER_TOL_S = 0.5
MAX_SCALE_DEV = 0.05
MIN_ANCHORS = 20
MIN_INLIER_SHARE = 0.6
MAX_RMS_S = 0.3
RANSAC_DRAWS = 2000
STEP_BAND_S = 4 * INLIER_TOL_S  # anchors this close to the fit count for the step and max checks
STEP_MIN_SIDE = 8  # anchors needed on each side of a candidate step
MAX_STEP_S = INLIER_TOL_S / 2
RANSAC_MIN_SPAN_S = 60.0
SPEECH_SCALES = (1.0, 1000 / 1001, 1001 / 1000, 25 / 23.976, 23.976 / 25, 24 / 23.976, 23.976 / 24)
SPEECH_MAX_OFFSET_S = 300.0
SPEECH_STEP_S = 0.5
SPEECH_REFINE_S = 1.0
SPEECH_FINE_STEP_S = 0.02

_CJK = {"zh", "ja", "ko"}


@dataclass(frozen=True)
class TimeMap:
    """Labelled-file time -> target time: `scale * t + offset`."""

    scale: float
    offset: float

    def apply(self, t: float) -> float:
        return self.scale * t + self.offset


@dataclass(frozen=True)
class Step:
    """The best single change point in a text fit: the same scale with one offset before
    `at_s` (labelled-file time of the first anchor after it) and another from there on."""

    at_s: float
    offset_before: float
    offset_after: float
    before: int
    after: int

    @property
    def size_s(self) -> float:
        return self.offset_after - self.offset_before


@dataclass(frozen=True)
class FitReport:
    """`anchors`: text-matched cue pairs (text) or cues scored (speech); `inliers`: anchors
    within INLIER_TOL_S of the fit (text) or cues at least half over speech (speech); `rms_s`:
    the inliers' residual (text; NaN in speech mode); `share`: the share of cue time over
    speech at the fit (speech; None in text mode). Text mode also gives `max_resid_s`, the
    largest residual among anchors within STEP_BAND_S of the fit, and `step`, the best single
    change point among them (None with too few anchors for one)."""

    tmap: TimeMap
    mode: Literal["text", "speech"]
    anchors: int
    inliers: int
    rms_s: float
    share: float | None
    max_resid_s: float = float("nan")
    step: Step | None = None


# ---- text mode ----


def _merge_events(cues: list[Cue]) -> list[Cue]:
    """Cues sharing exact start and end joined into one (ASS writes a two-line caption as two
    events), in time order."""
    groups: dict[tuple[float, float], list[str]] = {}
    for c in cues:
        groups.setdefault((c.start, c.end), []).append(c.text)
    return [Cue(a, b, "\n".join(ts)) for (a, b), ts in sorted(groups.items())]


def _units(cues: list[Cue], language: str) -> list[tuple[float, str]]:
    """(onset, normalized text) per caption, too-short texts dropped and consecutive repeats
    (roll-up captions written as two cues) collapsed onto the first onset."""
    cjk = language.split("-")[0].casefold() in _CJK
    out: list[tuple[float, str]] = []
    for c in _merge_events(cues):
        text = normalize(strip_markup(c.text), language)
        if (len(text) < 4) if cjk else (len(text.split()) < 2):
            continue
        if out and out[-1][1] == text:
            continue
        out.append((c.start, text))
    return out


def _anchors(labelled: list[Cue], reference: list[Cue], language: str) -> list[tuple[float, float]]:
    lab = _units(labelled, language)
    ref = _units(reference, language)
    lab_n = Counter(t for _, t in lab)
    ref_n = Counter(t for _, t in ref)
    ref_at = {t: s for s, t in ref}
    pool = [t for t in ref_n if ref_n[t] == 1]
    picks: list[tuple[float, str]] = []
    for onset, text in lab:
        if lab_n[text] != 1:
            continue
        if text in ref_n:
            if ref_n[text] == 1:
                picks.append((onset, text))
            continue
        close = difflib.get_close_matches(text, pool, n=2, cutoff=ANCHOR_MIN_RATIO)
        if not close:
            continue
        if len(close) == 2:
            r0 = difflib.SequenceMatcher(None, text, close[0]).ratio()
            r1 = difflib.SequenceMatcher(None, text, close[1]).ratio()
            if r0 - r1 < ANCHOR_MIN_GAP - 1e-9:
                continue
        picks.append((onset, close[0]))
    used = Counter(r for _, r in picks)
    return [(onset, ref_at[r]) for onset, r in picks if used[r] == 1]


def _fit_line(t: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    scale, offset = np.polyfit(t, r, 1)
    return float(scale), float(offset)


def fit_to_reference(labelled: list[Cue], reference: list[Cue], language: str) -> FitReport:
    """Fit the labelled file's timeline onto the reference file's by RANSAC over text anchors
    (onset pairs at least RANSAC_MIN_SPAN_S apart, scale within 1 +- MAX_SCALE_DEV), then least
    squares on the inliers. Raises AlignmentError when no candidate line exists at all."""
    pairs = _anchors(labelled, reference, language)
    if len(pairs) < 2:
        raise AlignmentError(
            f"only {len(pairs)} text anchor(s) between the two subtitle files; "
            "a different programme, or the wrong --language?"
        )
    t = np.array([p[0] for p in pairs])
    r = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(pairs), size=(RANSAC_DRAWS, 2))
    i, j = draws[:, 0], draws[:, 1]
    dt = t[j] - t[i]
    ok = np.abs(dt) >= RANSAC_MIN_SPAN_S
    i, j, dt = i[ok], j[ok], dt[ok]
    scale = (r[j] - r[i]) / dt
    ok = np.abs(scale - 1.0) <= MAX_SCALE_DEV
    i, scale = i[ok], scale[ok]
    if len(scale) == 0:
        raise AlignmentError(
            f"{len(pairs)} text anchors, but no two of them {RANSAC_MIN_SPAN_S:g} s apart agree "
            f"on a scale within 1 +- {MAX_SCALE_DEV:g}"
        )
    offset = r[i] - scale * t[i]
    resid = np.abs(scale[:, None] * t[None, :] + offset[:, None] - r[None, :])
    within = resid <= INLIER_TOL_S
    counts = within.sum(axis=1)
    err = np.where(within, resid, 0.0).sum(axis=1)
    best = int(np.lexsort((err, -counts))[0])
    s, o = float(scale[best]), float(offset[best])
    # A drawn line passes through two anchors' noise; refit once on a band twice as wide so
    # the least squares can settle between them, then tighten to INLIER_TOL_S.
    mask = np.abs(s * t + o - r) <= 2 * INLIER_TOL_S
    for _ in range(3):
        if mask.sum() >= 2:
            s, o = _fit_line(t[mask], r[mask])
        mask = np.abs(s * t + o - r) <= INLIER_TOL_S
    res = (s * t + o - r)[mask]
    rms = float(np.sqrt(np.mean(res**2))) if len(res) else float("inf")
    band = np.abs(s * t + o - r) <= STEP_BAND_S
    max_resid = float(np.abs(s * t + o - r)[band].max()) if band.any() else float("inf")
    step = _best_step(t[band], r[band])
    return FitReport(TimeMap(s, o), "text", len(pairs), int(mask.sum()), rms, None, max_resid, step)


def _best_step(t: np.ndarray, r: np.ndarray) -> Step | None:
    """The change point that best explains the anchors as one scale with two offsets
    (least squares over every split leaving STEP_MIN_SIDE anchors a side). A residual series
    can't show a step on its own: the single-line fit tilts to absorb most of it."""
    order = np.argsort(t)
    t, r = t[order], r[order]
    n = len(t)
    if n < 2 * STEP_MIN_SIDE:
        return None
    best: tuple[float, int, np.ndarray] | None = None
    for k in range(STEP_MIN_SIDE, n - STEP_MIN_SIDE + 1):
        if t[k] == t[k - 1]:
            continue
        a = np.column_stack([t, np.ones(n), (np.arange(n) >= k).astype(float)])
        coef, *_ = np.linalg.lstsq(a, r, rcond=None)
        sse = float(((a @ coef - r) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k, coef)
    if best is None:
        return None
    _, k, coef = best
    return Step(float(t[k]), float(coef[1]), float(coef[1] + coef[2]), k, n - k)


# ---- speech mode ----


def fit_to_speech(labelled: list[Cue], speech: list[Turn]) -> FitReport:
    """The (scale, offset) that puts the most cue time over detected speech: every scale in
    SPEECH_SCALES at offsets +-SPEECH_MAX_OFFSET_S in SPEECH_STEP_S steps, the best offset of
    each refined +-SPEECH_REFINE_S in SPEECH_FINE_STEP_S steps. Ties take the middle of the
    plateau, then the scale nearest 1."""
    spans = [(c.start, c.end) for c in _merge_events(labelled) if c.end > c.start]
    x, y = _union(speech)
    if not spans or len(x) == 0:
        raise AlignmentError("speech fit needs timed cues and detected speech; one is empty")
    s = np.array([a for a, _ in spans])
    e = np.array([b for _, b in spans])

    def shares(sc: float, offsets: np.ndarray) -> np.ndarray:
        a = sc * s[None, :] + offsets[:, None]
        b = sc * e[None, :] + offsets[:, None]
        over = np.interp(b, x, y) - np.interp(a, x, y)
        return over.sum(axis=1) / (sc * float((e - s).sum()))

    def plateau(offsets: np.ndarray, sh: np.ndarray) -> tuple[float, float]:
        top = float(sh.max())
        idx = np.flatnonzero(sh >= top - 1e-9)
        return float(offsets[idx[len(idx) // 2]]), top

    n = int(round(SPEECH_MAX_OFFSET_S / SPEECH_STEP_S))
    coarse = np.arange(-n, n + 1) * SPEECH_STEP_S
    m = int(round(SPEECH_REFINE_S / SPEECH_FINE_STEP_S))
    fine_steps = np.arange(-m, m + 1) * SPEECH_FINE_STEP_S
    found: list[tuple[float, float, float]] = []  # (share, scale, offset)
    for sc in SPEECH_SCALES:
        o0, _ = plateau(coarse, shares(sc, coarse))
        fine = o0 + fine_steps
        o1, top = plateau(fine, shares(sc, fine))
        found.append((top, sc, o1))
    top = max(f[0] for f in found)
    _, sc, off = min((f for f in found if f[0] >= top - 1e-9), key=lambda f: abs(f[1] - 1.0))
    a, b = sc * s + off, sc * e + off
    covered = (np.interp(b, x, y) - np.interp(a, x, y)) / (b - a)
    inliers = int((covered >= 0.5).sum())
    return FitReport(TimeMap(sc, off), "speech", len(spans), inliers, float("nan"), top)


# ---- the check ----


def check_fit(rep: FitReport, force: bool = False) -> None:
    """Refuse a fit a single linear map can't be trusted for (a cut, a jump, the wrong file):
    text mode needs MIN_ANCHORS anchors, MIN_INLIER_SHARE of them inliers, an inlier rms up to
    MAX_RMS_S and a scale within 1 +- MAX_SCALE_DEV; speech mode needs MIN_SHARE of cue time
    over speech. `force` skips the refusal."""
    problems: list[str] = []
    if rep.mode == "text":
        share = rep.inliers / rep.anchors if rep.anchors else 0.0
        if rep.anchors < MIN_ANCHORS:
            problems.append(f"only {rep.anchors} text anchors (need {MIN_ANCHORS})")
        if share < MIN_INLIER_SHARE:
            problems.append(
                f"inlier share {share:.0%} ({rep.inliers}/{rep.anchors}) is below "
                f"{MIN_INLIER_SHARE:.0%}; a cut or a jump between the captures"
            )
        if not rep.rms_s <= MAX_RMS_S:
            problems.append(f"inlier rms {rep.rms_s:.3f} s is above {MAX_RMS_S:g} s")
        if rep.max_resid_s > INLIER_TOL_S:
            problems.append(
                f"max residual {rep.max_resid_s:.3f} s is above {INLIER_TOL_S:g} s "
                f"(anchors within {STEP_BAND_S:g} s of the fit)"
            )
        st = rep.step
        if st is not None and abs(st.size_s) > MAX_STEP_S:
            problems.append(
                f"a {st.size_s:+.2f} s step at {st.at_s:.1f} s in the labelled file: offset "
                f"{st.offset_before:+.2f} s before ({st.before} anchors), "
                f"{st.offset_after:+.2f} s after ({st.after}); a cut or a jump between "
                "the captures"
            )
    else:
        sh = rep.share or 0.0
        if sh < MIN_SHARE:
            problems.append(f"speech share {sh:.0%} is below {MIN_SHARE:.0%}")
    if abs(rep.tmap.scale - 1.0) > MAX_SCALE_DEV:
        problems.append(f"scale {rep.tmap.scale:.6f} is outside 1 +- {MAX_SCALE_DEV:g}")
    if problems and not force:
        raise AlignmentError(
            "no trustworthy linear fit: " + "; ".join(problems) + ". Pass --force to write anyway"
        )


# ---- retiming ----

_ARROW = re.compile(r"^(\s*)(\S+)(\s*-->\s*)(\S+)(.*)$")
_TEXT_STAMP = re.compile(r"^(?:(\d+):)?(\d+):(\d+)([.,])(\d+)$")
_ASS_STAMP = re.compile(r"^(\d+):(\d+):(\d+)(?:\.(\d+))?$")


def _ass_stamp(t: float) -> str:
    cs = int(round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _text_stamp(t: float, like: re.Match[str]) -> str:
    hours, _, _, sep, frac = like.groups()
    digits = len(frac)
    units = int(round(t * 10**digits))
    whole, part = divmod(units, 10**digits)
    h, rest = divmod(whole, 3600)
    mi, se = divmod(rest, 60)
    tail = f"{mi:02d}:{se:02d}{sep}{part:0{digits}d}"
    if hours is not None or h:
        return f"{h:0{max(2, len(hours or ''))}d}:{tail}"
    return tail


def _seconds(m: re.Match[str]) -> float:
    groups = m.groups()
    if len(groups) == 5:  # text stamp
        h, mi, se, _, frac = groups
    else:
        h, mi, se, frac = groups
    return (
        int(h or 0) * 3600 + int(mi) * 60 + int(se) + (int(frac) / 10 ** len(frac) if frac else 0)
    )


def _mapped(t: float, tmap: TimeMap) -> tuple[float, bool]:
    v = tmap.apply(t)
    return (0.0, True) if v < 0 else (v, False)


def _retime_ass(lines: list[str], tmap: TimeMap) -> tuple[list[str], int, int]:
    fields = ["layer", "start", "end"]
    in_events = False
    out: list[str] = []
    done = clamped = 0
    for raw in lines:
        body = raw.rstrip("\r\n")
        nl = raw[len(body) :]
        row = body.strip().lstrip("﻿")
        low = row.casefold()
        if row.startswith("["):
            in_events = low == "[events]"
        elif in_events and low.startswith("format:"):
            fields = [f.strip().casefold() for f in row.split(":", 1)[1].split(",")]
        elif in_events and low.startswith("dialogue:") and {"start", "end"} <= set(fields):
            head, rest = body.split(":", 1)
            si, ei = fields.index("start"), fields.index("end")
            parts = rest.split(",", max(si, ei) + 1)
            ms, me = _ASS_STAMP.match(parts[si].strip()), _ASS_STAMP.match(parts[ei].strip())
            if ms and me and len(parts) > max(si, ei) + 1:
                a, ca = _mapped(_seconds(ms), tmap)
                b, cb = _mapped(_seconds(me), tmap)
                for k, v in ((si, a), (ei, b)):
                    lead = parts[k][: len(parts[k]) - len(parts[k].lstrip())]
                    trail = parts[k][len(parts[k].rstrip()) :]
                    parts[k] = f"{lead}{_ass_stamp(v)}{trail}"
                body = f"{head}:{','.join(parts)}"
                done += 1
                clamped += ca or cb
        out.append(body + nl)
    return out, done, clamped


def _retime_text(lines: list[str], tmap: TimeMap) -> tuple[list[str], int, int]:
    out: list[str] = []
    done = clamped = 0
    for raw in lines:
        body = raw.rstrip("\r\n")
        nl = raw[len(body) :]
        m = _ARROW.match(body) if "-->" in body else None
        ma = _TEXT_STAMP.match(m.group(2)) if m else None
        mb = _TEXT_STAMP.match(m.group(4)) if m else None
        if m and ma and mb:
            a, ca = _mapped(_seconds(ma), tmap)
            b, cb = _mapped(_seconds(mb), tmap)
            body = f"{m.group(1)}{_text_stamp(a, ma)}{m.group(3)}{_text_stamp(b, mb)}{m.group(5)}"
            done += 1
            clamped += ca or cb
        out.append(body + nl)
    return out, done, clamped


def _decode(raw: bytes) -> tuple[str, str, bytes]:
    """(text, codec, BOM) for UTF-16 by its BOM, else UTF-8 with undecodable bytes kept."""
    for bom, codec in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")):
        if raw.startswith(bom):
            return raw[len(bom) :].decode(codec, errors="surrogatepass"), codec, bom
    return raw.decode("utf-8", errors="surrogateescape"), "utf-8", b""


def _encode(text: str, codec: str, bom: bytes) -> bytes:
    errors = "surrogateescape" if codec == "utf-8" else "surrogatepass"
    return bom + text.encode(codec, errors=errors)


def retime_file(src: Path, dst: Path, tmap: TimeMap) -> tuple[int, int]:
    """Write `src` to `dst` with every cue time mapped through `tmap` (ASS/SSA `Dialogue` Start
    and End, SRT and WebVTT `-->` lines); every other byte is kept. Times that map below zero
    become zero. A UTF-16 file (by its BOM) is written back in the same encoding. Returns
    (cues rewritten, cues clamped at 0)."""
    text, codec, bom = _decode(Path(src).read_bytes())
    lines = text.splitlines(keepends=True)
    head = text.lstrip("﻿ \t\r\n")[:200].casefold()
    suffix = Path(src).suffix.casefold()
    if suffix in {".ass", ".ssa"} or head.startswith("[script info]") or "[events]" in head:
        out, done, clamped = _retime_ass(lines, tmap)
    elif "-->" in text:
        out, done, clamped = _retime_text(lines, tmap)
    else:
        raise ValueError(f"{src}: not an ASS/SSA, SRT or WebVTT file; can't retime it")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(dst) as tmp:
        tmp.write_bytes(_encode("".join(out), codec, bom))
    return done, clamped


def guess_language(cues: list[Cue]) -> str:
    """`ja` when the text has kana, `ko` for hangul, `zh` for han alone, else `en`: enough for
    `normalize`, which only tells CJK (no word spaces) from the rest."""
    text = "".join(strip_markup(c.text) for c in cues)
    if re.search(r"[぀-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    if re.search(r"[一-鿿]", text):
        return "zh"
    return "en"
