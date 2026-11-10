"""`fukikae eval tts-ab`: a blind A/B listening pack for the TTS backends (the Task 11
checkpoint, ruling D15: the operator picks the cloning backend by listening).

A spec (TOML) names one or more sets, each a workspace, an episode and the candidates
(``backend`` or ``backend:mode``, the mode a ``[tts.<backend>] mode`` override such as
``qwen3_tts:xvector``). For each set:

1. The cues ``eval tts`` samples (``sample_cues``, 8 per character, seed 0) are rendered for
   every candidate with the cards ``eval tts`` builds (clone cards from ``select_references``,
   stock cards for a stock backend), through the tts stage's own path, including the rate fit
   (``render_for_slot``, ruling CD12). Renders already in ``.fukikae/tts-cache`` are reused;
   anything missing is rendered on the workspace's GPU under ``gpu_lock`` (ruling CD15).
2. ``select_cues`` picks about ``cues`` of them over at least ``characters`` characters,
   mixing neutral and emotional source lines, and keeps at least one line each candidate
   marked suspect: the render guard's (ruling CD18) or, with ``ledger_runs``, a cue that
   candidate's ``eval tts`` row lists in ``high_wer_cues``.
3. Each render goes through timefit's own ``plan_fit`` and ``apply_fit`` for its slot (the
   previous cue's slot end and the next cue's start bound the borrowing, as the stage would
   with the neighbours in their slots). Every clip of a cue covers one window, the slot and
   every candidate's placed audio plus ``PAD_S`` each side, so all are the same length; the
   reference is the original mix (``00-extract/audio.wav``) over that window, plus the English
   original from ``reference_workspace`` when there is one (BBT).
4. With ``[set.scene]``, one 60-90 s scene goes through tts, timefit and mix per candidate
   (characters without enough clean speech for a clone fall back to a stock card, as Task 12
   will; extras get the stage's stock extras), muxed with the video as ``video/sceneNN_X.mkv``.

``write_pack`` blinds it all: per cue (and scene) the candidates are shuffled with
``random.Random(f"ab-blind:{seed}:{slot}")`` and written as ``audio/cueNN_A.wav``,
``cueNN_B.wav``, ...; the references are ``cueNN_ref.wav`` and ``cueNN_ref_en.wav``. Every
clip is mono 48 kHz (scenes stereo), brought to ``TARGET_LUFS`` (EBU R128 integrated) by one
linear gain with the true-peak limiter (``mix.normalize``, ruling CD10). ``key.toml`` holds the
label mapping, the suspect flags and each render's provenance; neither ``index.html`` nor
``README.txt`` names a backend or links the key. ``verdict.toml`` is the template the operator
fills, and ``read_verdict`` is how Task 12 reads it back (labels resolved through the key).
The folder is owner-only (0700 directories, 0600 files).
"""

from __future__ import annotations

import contextlib
import dataclasses
import html
import os
import random
import string
import tempfile
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from fukikae.cards.store import CardStore
    from fukikae.config import Config
    from fukikae.stages.tts import RenderJob
    from fukikae.workspace import Workspace

TARGET_LUFS = -23.0
RATE = 48000
PAD_S = 0.3
LABELS = string.ascii_uppercase
SAMPLE_PER_CHARACTER = 8  # eval tts's default sample, so its cached renders are found
SAMPLE_SEED = 0
SCENE_SECONDS = 75.0
STOCK_PICK = "stock"
EMOTIONAL = frozenset({"angry", "happy", "sad", "fearful", "surprised", "disgusted"})
REFERENCE_SUFFIX = {"original": "ref", "english": "ref_en"}
REFERENCE_TITLE = {"original": "Original line", "english": "English track"}

SPEC_KEYS = frozenset({"seed", "set", "ledger"})
SET_KEYS = frozenset(
    {
        "name", "workspace", "episode", "candidates", "cues", "characters",
        "reference_workspace", "ledger_runs", "scene",
    }
)  # fmt: skip
SCENE_KEYS = frozenset({"start", "seconds"})


# ---- candidates and the spec ----------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A backend, with a ``[tts.<backend>] mode`` override or None for its default."""

    backend: str
    mode: str | None = None

    @property
    def name(self) -> str:
        return self.backend if self.mode is None else f"{self.backend}:{self.mode}"

    @classmethod
    def parse(cls, text: str) -> Candidate:
        backend, sep, mode = text.strip().partition(":")
        if not backend or (sep and not mode) or ":" in mode:
            raise ValueError(f"candidate {text!r}: use backend or backend:mode")
        return cls(backend, mode or None)


@dataclass(frozen=True)
class SceneSpec:
    start: float | None = None  # None: the densest dialogue window (``pick_scene``)
    seconds: float = SCENE_SECONDS


@dataclass(frozen=True)
class SetSpec:
    name: str
    workspace: Path
    episode: str
    candidates: tuple[Candidate, ...]
    cues: int = 6
    characters: int = 3
    reference_workspace: Path | None = None
    ledger_runs: Mapping[str, str] = field(default_factory=dict)
    scene: SceneSpec | None = None


@dataclass(frozen=True)
class PackSpec:
    seed: int
    sets: tuple[SetSpec, ...]
    ledger: Path | None = None


def _unknown(table: Mapping[str, Any], known: frozenset[str], where: str) -> None:
    extra = sorted(set(table) - known)
    if extra:
        raise ValueError(
            f"{where}: unknown key(s) {', '.join(extra)}; known: {', '.join(sorted(known))}"
        )


def _path(value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a path")
    return Path(value).expanduser()


def load_spec(path: Path) -> PackSpec:
    """The pack spec at ``path``; unknown keys and malformed values are ValueErrors."""
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    _unknown(data, SPEC_KEYS, str(path))
    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path}: seed must be an integer")
    sets: list[SetSpec] = []
    for n, raw in enumerate(data.get("set") or [], start=1):
        where = f"{path}: [[set]] {n}"
        _unknown(raw, SET_KEYS, where)
        name = raw.get("name")
        if not isinstance(name, str) or not name or not name.replace("-", "").isalnum():
            raise ValueError(f"{where}: name must be letters, digits and dashes")
        cands = tuple(Candidate.parse(c) for c in raw.get("candidates") or [])
        if len(cands) < 2 or len({c.name for c in cands}) != len(cands):
            raise ValueError(f"{where}: candidates must name two or more distinct candidates")
        if len(cands) > len(LABELS):
            raise ValueError(f"{where}: at most {len(LABELS)} candidates")
        scene = raw.get("scene")
        if scene is not None:
            _unknown(scene, SCENE_KEYS, f"{where} scene")
            scene = SceneSpec(
                float(scene["start"]) if "start" in scene else None,
                float(scene.get("seconds", SCENE_SECONDS)),
            )
            if not 30.0 <= scene.seconds <= 180.0:
                raise ValueError(f"{where}: scene seconds must be 30 to 180")
        runs = raw.get("ledger_runs") or {}
        unknown_runs = sorted(set(runs) - {c.name for c in cands})
        if unknown_runs:
            raise ValueError(f"{where}: ledger_runs for unknown candidate(s) {unknown_runs}")
        sets.append(
            SetSpec(
                name=name,
                workspace=_path(raw.get("workspace"), f"{where} workspace"),
                episode=str(raw.get("episode", "")).upper(),
                candidates=cands,
                cues=int(raw.get("cues", 6)),
                characters=int(raw.get("characters", 3)),
                reference_workspace=(
                    _path(raw["reference_workspace"], f"{where} reference_workspace")
                    if "reference_workspace" in raw
                    else None
                ),
                ledger_runs={str(k): str(v) for k, v in runs.items()},
                scene=scene,
            )
        )
    if not sets:
        raise ValueError(f"{path}: no [[set]]")
    if len({s.name for s in sets}) != len(sets):
        raise ValueError(f"{path}: set names must be distinct")
    ledger = _path(data["ledger"], f"{path}: ledger") if "ledger" in data else None
    return PackSpec(seed, tuple(sets), ledger)


# ---- cue selection ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CueInfo:
    """A sampled cue as selection sees it; ``suspect`` maps candidate name to the reason."""

    episode: str
    cue_id: str
    character_id: int
    emotion: str | None
    suspect: Mapping[str, str] = field(default_factory=dict)

    @property
    def emotional(self) -> bool:
        return self.emotion in EMOTIONAL


def select_cues(
    items: Sequence[CueInfo], *, count: int, characters: int, seed: int
) -> list[CueInfo]:
    """About ``count`` cues: first one suspect line per candidate and kind of flag (the
    render guard's, any other), preferring lines several candidates share, then the
    most-sampled characters; then one line for each of the ``characters`` most-sampled
    characters, then more of theirs, alternating emotional and neutral source lines. Ties
    are broken by ``random.Random(f"ab-select:{seed}")``.
    Returned in (episode, cue) order."""
    rng = random.Random(f"ab-select:{seed}")  # noqa: S311 - a reproducible pick
    pool = sorted(items, key=lambda i: (i.episode, i.cue_id))
    per_char: dict[int, int] = {}
    for i in pool:
        per_char[i.character_id] = per_char.get(i.character_id, 0) + 1
    top = sorted(per_char, key=lambda c: (-per_char[c], c))[:characters]
    chosen: list[CueInfo] = []

    def take(options: list[CueInfo]) -> None:
        options = [o for o in options if o not in chosen]
        if options:
            chosen.append(options[rng.randrange(len(options))])

    def kinds(i: CueInfo) -> set[tuple[str, str]]:
        return {
            (b, "guard" if r.startswith("render guard") else "other") for b, r in i.suspect.items()
        }

    for kind in sorted({k for i in pool for k in kinds(i)}):
        if any(kind in kinds(c) for c in chosen):
            continue
        flagged = [i for i in pool if kind in kinds(i)]
        best = max((len(i.suspect), i.character_id in top) for i in flagged)
        take([i for i in flagged if (len(i.suspect), i.character_id in top) == best])

    def want_emotional() -> bool:
        return sum(c.emotional for c in chosen) <= sum(not c.emotional for c in chosen)

    def fill_for(cid: int) -> None:
        mine = [i for i in pool if i.character_id == cid and i not in chosen]
        wanted = [i for i in mine if i.emotional == want_emotional()]
        take(wanted or mine)

    for cid in top:
        if not any(c.character_id == cid for c in chosen):
            fill_for(cid)
    while len(chosen) < count and len(chosen) < len(pool):
        open_chars = [c for c in top if any(i.character_id == c and i not in chosen for i in pool)]
        if not open_chars:
            take(pool)
            continue
        fill_for(min(open_chars, key=lambda c: (sum(x.character_id == c for x in chosen), c)))
    return sorted(chosen, key=lambda i: (i.episode, i.cue_id))


def blind_order(candidates: Sequence[str], *, seed: int, slot: str) -> dict[str, str]:
    """Label -> candidate for one cue or scene: the names sorted, then shuffled by
    ``random.Random(f"ab-blind:{seed}:{slot}")``."""
    order = sorted(candidates)
    random.Random(f"ab-blind:{seed}:{slot}").shuffle(order)  # noqa: S311 - reproducible
    return {LABELS[n]: c for n, c in enumerate(order)}


# ---- materials and the pack ----------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """Audio whose first sample plays at ``start`` seconds on the episode's timeline."""

    path: Path
    start: float


@dataclass
class CueMaterial:
    set: str
    episode: str
    cue_id: str
    character_id: int
    character: str
    source_text: str
    text: str
    emotion: str | None
    slot: tuple[float, float]
    references: dict[str, Clip]  # "original", "english"
    renders: dict[str, Clip]  # candidate name -> its placed, fitted render
    suspect: dict[str, str] = field(default_factory=dict)
    info: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SceneMaterial:
    set: str
    episode: str
    window: tuple[float, float]
    original: Clip
    renders: dict[str, Clip]  # candidate name -> its scene mix
    info: dict[str, dict[str, Any]] = field(default_factory=dict)
    video: Path | None = None  # the episode's video, for muxed scene files


def _seconds(path: Path) -> float:
    from scipy.io import wavfile

    rate, data = wavfile.read(path, mmap=True)
    return len(data) / rate


def cue_window(m: CueMaterial) -> tuple[float, float]:
    """The slot and every candidate's placed audio, plus ``PAD_S`` each side (from 0)."""
    starts = [m.slot[0], *(c.start for c in m.renders.values())]
    ends = [m.slot[1], *(c.start + _seconds(c.path) for c in m.renders.values())]
    return max(0.0, min(starts) - PAD_S), max(ends) + PAD_S


def _as_float(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.float32) / (float(np.iinfo(data.dtype).max) + 1.0)
    return np.asarray(data, dtype=np.float32)


def _window_audio(clip: Clip, window: tuple[float, float], channels: int) -> np.ndarray:
    """``clip`` over ``window`` as float32 (n, channels) at ``RATE``, silence where the clip
    has no audio."""
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    rate, data = wavfile.read(clip.path, mmap=True)
    x: Any = data if data.ndim == 2 else data[:, None]
    a = round((window[0] - clip.start) * rate)
    b = round((window[1] - clip.start) * rate)
    seg = _as_float(np.asarray(x[max(0, a) : max(0, min(b, len(x)))]))
    lead = max(0, -a)
    total = max(0, b - a)
    buf = np.zeros((total, x.shape[1]), dtype=np.float32)
    buf[lead : lead + len(seg)] = seg[: max(0, total - lead)]
    if channels == 1:
        buf = buf.mean(axis=1, keepdims=True)
    elif buf.shape[1] == 1:
        buf = np.repeat(buf, channels, axis=1)
    elif buf.shape[1] > channels:
        buf = buf[:, :channels]
    if rate != RATE:
        g = gcd(RATE, rate)
        buf = resample_poly(buf, RATE // g, rate // g, axis=0).astype(np.float32)
    n = round((window[1] - window[0]) * RATE)
    if len(buf) < n:
        buf = np.concatenate([buf, np.zeros((n - len(buf), channels), dtype=np.float32)])
    return buf[:n]


def _write_clip(clip: Clip, window: tuple[float, float], dest: Path, channels: int) -> float:
    """``clip`` over ``window`` to ``dest`` (16-bit, ``RATE``) at ``TARGET_LUFS``; returns the
    gain applied (dB)."""
    from scipy.io import wavfile

    from fukikae.mix import normalize

    buf = _window_audio(clip, window, channels)
    n = len(buf)
    with tempfile.TemporaryDirectory(prefix="fukikae-ab-") as tmp:
        raw, level = Path(tmp) / "raw.wav", Path(tmp) / "level.wav"
        wavfile.write(raw, RATE, buf)
        gain = normalize(raw, TARGET_LUFS, level)["gain_db"]
        _, read = wavfile.read(level)
    out: Any = read if read.ndim == 2 else read[:, None]
    if len(out) < n:
        out = np.concatenate([out, np.zeros((n - len(out), out.shape[1]), dtype=out.dtype)])
    wavfile.write(dest, RATE, out[:n] if channels > 1 else out[:n, 0])
    return float(gain)


def _mux_scene(video: Path, window: tuple[float, float], audio: Path, dest: Path) -> None:
    from fukikae import media

    media.run_ffmpeg(
        ["-ss", f"{window[0]:.3f}", "-t", f"{window[1] - window[0]:.3f}", "-i", str(video),
         "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-sn", "-dn",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-vf", "scale=-2:720",
         "-c:a", "flac", "-shortest", str(dest)]
    )  # fmt: skip


# -- TOML out ------------------------------------------------------------------------------------


def _owner_only(root: Path) -> None:
    os.chmod(root, 0o700)
    for p in root.rglob("*"):
        os.chmod(p, 0o700 if p.is_dir() else 0o600)


@contextlib.contextmanager
def _umask(mask: int) -> Iterator[None]:
    old = os.umask(mask)
    try:
        yield
    finally:
        os.umask(old)


def write_pack(
    out: Path,
    *,
    seed: int,
    cues: Sequence[CueMaterial],
    scenes: Sequence[SceneMaterial] = (),
    numbers: str | None = None,
    notes: Sequence[str] = (),
) -> Path:
    """Write the blind pack to ``out`` (absent or empty): ``audio/``, ``video/`` (scenes with
    a video), ``index.html``, ``README.txt``, ``verdict.toml``, ``key.toml`` and, with
    ``numbers`` (an HTML fragment), ``numbers.html``. Owner-only. Returns ``out``."""
    import tomli_w

    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; give a new folder")
    with _umask(0o077):
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        audio = out / "audio"
        audio.mkdir(exist_ok=True)
        key: dict[str, Any] = {
            "seed": seed,
            "target_lufs": TARGET_LUFS,
            "candidates": sorted({c for m in [*cues, *scenes] for c in m.renders}),
            "notes": list(notes),
            "cue": {},
            "scene": {},
        }
        page_cues: list[tuple[str, CueMaterial, list[str], list[str]]] = []
        for n, m in enumerate(cues, start=1):
            slot = f"cue{n:02d}"
            labels = blind_order(list(m.renders), seed=seed, slot=slot)
            window = cue_window(m)
            gains: dict[str, float] = {}
            refs: list[str] = []
            for kind in ("original", "english"):
                if kind in m.references:
                    name = f"{slot}_{REFERENCE_SUFFIX[kind]}.wav"
                    gains[kind] = _write_clip(m.references[kind], window, audio / name, 1)
                    refs.append(kind)
            for label, cand in labels.items():
                gains[cand] = _write_clip(m.renders[cand], window, audio / f"{slot}_{label}.wav", 1)
            key["cue"][slot] = {
                "set": m.set,
                "episode": m.episode,
                "cue_id": m.cue_id,
                "character_id": m.character_id,
                "character": m.character,
                "emotion": m.emotion or "",
                "slot": [round(m.slot[0], 4), round(m.slot[1], 4)],
                "window": [round(window[0], 4), round(window[1], 4)],
                "labels": labels,
                "suspect": dict(sorted(m.suspect.items())),
                "gain_db": {k: round(v, 2) for k, v in sorted(gains.items())},
                "render": {c: _plain(m.info.get(c, {})) for c in sorted(m.renders)},
            }
            page_cues.append((slot, m, refs, list(labels)))
        page_scenes: list[tuple[str, SceneMaterial, list[str], bool]] = []
        for n, s in enumerate(scenes, start=1):
            slot = f"scene{n:02d}"
            labels = blind_order(list(s.renders), seed=seed, slot=slot)
            gains = {"original": _write_clip(s.original, s.window, audio / f"{slot}_ref.wav", 2)}
            for label, cand in labels.items():
                gains[cand] = _write_clip(
                    s.renders[cand], s.window, audio / f"{slot}_{label}.wav", 2
                )
            if s.video is not None:
                vdir = out / "video"
                vdir.mkdir(exist_ok=True)
                for suffix in ["ref", *labels]:
                    _mux_scene(
                        s.video,
                        s.window,
                        audio / f"{slot}_{suffix}.wav",
                        vdir / f"{slot}_{suffix}.mkv",
                    )
            key["scene"][slot] = {
                "set": s.set,
                "episode": s.episode,
                "window": [round(s.window[0], 3), round(s.window[1], 3)],
                "labels": labels,
                "gain_db": {k: round(v, 2) for k, v in sorted(gains.items())},
                "render": {c: _plain(s.info.get(c, {})) for c in sorted(s.renders)},
            }
            page_scenes.append((slot, s, list(labels), s.video is not None))
        (out / "key.toml").write_text(
            "# The blind labels' key. Open only after verdict.toml's [cue.*] and [scene.*] are\n"
            "# filled in. suspect: the render guard's flag (ruling CD18) or a cue the candidate's\n"
            "# eval tts row lists in high_wer_cues (per-render WER over 0.5).\n"
            + tomli_w.dumps(key),
            encoding="utf-8",
        )
        (out / "verdict.toml").write_text(
            verdict_template(key["candidates"], page_cues, page_scenes), encoding="utf-8"
        )
        (out / "index.html").write_text(
            _index_html(page_cues, page_scenes, numbers is not None), encoding="utf-8"
        )
        (out / "README.txt").write_text(_readme(len(page_cues), len(page_scenes)), encoding="utf-8")
        if numbers is not None:
            (out / "numbers.html").write_text(_page("Numbers", numbers), encoding="utf-8")
    _owner_only(out)
    return out


def _plain(d: Mapping[str, Any]) -> dict[str, Any]:
    """``d`` with None dropped and floats rounded, for TOML."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        out[k] = round(v, 4) if isinstance(v, float) else v
    return out


# -- verdict ------------------------------------------------------------------------------------


def verdict_template(
    candidates: Sequence[str],
    cues: Sequence[tuple[str, CueMaterial, list[str], list[str]]],
    scenes: Sequence[tuple[str, SceneMaterial, list[str], bool]],
) -> str:
    lines = [
        "# fukikae TTS A/B verdict (ruling D15). Fill in, save, and tell the controller.",
        "#",
        "# 1. Listen first. Fill every [cue.*] and [scene.*] table with the blind labels only:",
        '#    ranking = labels best first (partial is fine, e.g. ["C", "A"]), or pick = one label.',
        "# 2. Then open key.toml and fill [overall] by candidate name.",
        "#",
        "# Candidates in this pack: " + ", ".join(candidates),
        f'# pick = one of them, or "{STOCK_PICK}" (none passes: stock voices stay the default,',
        "#        and clone-by-default (D12) waits). Kokoro is the stock baseline, so picking it",
        f'#        means the same as "{STOCK_PICK}".',
        "# passed = every candidate whose clone you would accept, even if not your pick.",
        "",
        "[overall]",
        'pick = ""',
        "passed = []",
        'notes = ""',
        "",
    ]
    for slot, m, _, labels in cues:
        lines += [
            f"# {slot}: {m.set} {m.episode} {m.cue_id}, {m.character}; labels {', '.join(labels)}",
            f"[cue.{slot}]",
            "ranking = []",
            'pick = ""',
            'notes = ""',
            "",
        ]
    for slot, s, labels, _ in scenes:
        lines += [
            f"# {slot}: {s.set} {s.episode} {s.window[0]:.1f}-{s.window[1]:.1f} s; "
            f"labels {', '.join(labels)}",
            f"[scene.{slot}]",
            "ranking = []",
            'pick = ""',
            'notes = ""',
            "",
        ]
    lines += [
        "# Per-character overrides (ruling CD21), after opening key.toml. One table each:",
        "# [[override]]",
        '# set = "bbt-de"          # the set name in index.html',
        "# character = 2           # the registry id in index.html",
        '# backend = "qwen3_tts"   # a candidate backend, or "stock" for a stock voice',
        '# mode = "xvector"        # optional: qwen3_tts icl|xvector, cosyvoice3',
        "#                         #   cross_lingual|zero_shot",
        '# notes = ""',
        "",
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class Ranking:
    ranking: list[Candidate]
    pick: Candidate | None
    notes: str


@dataclass(frozen=True)
class Override:
    set: str
    character: int
    candidate: Candidate | None  # None: a stock voice
    notes: str


@dataclass(frozen=True)
class Verdict:
    """``verdict.toml`` resolved through ``key.toml``. ``pick`` is the chosen cloning
    candidate, None when ``stock`` (the operator picked ``stock`` or the stock baseline, or
    left it empty: then ``stock`` is False and the verdict is not given yet)."""

    pick: Candidate | None
    stock: bool
    passed: list[Candidate]
    notes: str
    cues: dict[str, Ranking]
    scenes: dict[str, Ranking]
    overrides: list[Override]

    @property
    def given(self) -> bool:
        return self.stock or self.pick is not None


MODES = {
    "qwen3_tts": ("auto", "icl", "xvector"),  # impl_qwen3_tts.MODES
    "cosyvoice3": ("auto", "zero_shot", "cross_lingual"),  # impl_cosyvoice3.MODES
}
STOCK_BACKENDS = frozenset({"kokoro"})


def _str(v: Any, where: str) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{where} must be a string, not {v!r}")
    return v


def _ranking(raw: Any, labels: Mapping[str, str], where: str) -> Ranking:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a table")
    _unknown(raw, frozenset({"ranking", "pick", "notes"}), where)
    order = raw.get("ranking", [])
    if not isinstance(order, list):
        raise ValueError(f"{where} ranking must be a list of labels")
    out: list[Candidate] = []
    for lab in order:
        if lab not in labels:
            raise ValueError(f"{where}: label {lab!r} is not one of {', '.join(labels)}")
        out.append(Candidate.parse(labels[lab]))
    if len({c.name for c in out}) != len(out):
        raise ValueError(f"{where}: a label appears twice in ranking")
    pick = _str(raw.get("pick", ""), f"{where} pick").strip()
    if pick and pick not in labels:
        raise ValueError(f"{where}: pick {pick!r} is not one of {', '.join(labels)}")
    return Ranking(out, Candidate.parse(labels[pick]) if pick else None, raw.get("notes", ""))


def read_verdict(pack: Path) -> Verdict:
    """``pack/verdict.toml`` with every label resolved through ``pack/key.toml`` and every
    name checked against the pack's candidates. Task 12 reads it with this function: a
    ``pick`` with a mode becomes ``[tts] clone_backend`` plus that mode as the card default
    (``qwen3_tts_mode`` / ``cosyvoice3_mode``); ``stock`` leaves the profiles alone (D12
    waits); each override applies to that character's card in that set's series."""
    pack = Path(pack)
    key = tomllib.loads((pack / "key.toml").read_text(encoding="utf-8"))
    data = tomllib.loads((pack / "verdict.toml").read_text(encoding="utf-8"))
    _unknown(data, frozenset({"overall", "cue", "scene", "override"}), "verdict.toml")
    names = set(key["candidates"])

    def candidate(text: str, where: str) -> Candidate:
        if text not in names:
            raise ValueError(
                f"{where}: {text!r} is not a candidate here ({', '.join(sorted(names))})"
            )
        return Candidate.parse(text)

    overall = data.get("overall", {})
    _unknown(overall, frozenset({"pick", "passed", "notes"}), "[overall]")
    pick_text = _str(overall.get("pick", ""), "[overall] pick").strip()
    stock = pick_text == STOCK_PICK
    pick = None if not pick_text or stock else candidate(pick_text, "[overall] pick")
    if pick is not None and pick.backend in STOCK_BACKENDS:
        pick, stock = None, True
    passed = [
        candidate(_str(p, "[overall] passed"), "[overall] passed")
        for p in overall.get("passed", [])
    ]
    cues = {
        slot: _ranking(raw, key["cue"][slot]["labels"], f"[cue.{slot}]")
        for slot, raw in data.get("cue", {}).items()
        if _known_slot(slot, key["cue"], "cue")
    }
    scenes = {
        slot: _ranking(raw, key["scene"][slot]["labels"], f"[scene.{slot}]")
        for slot, raw in data.get("scene", {}).items()
        if _known_slot(slot, key["scene"], "scene")
    }
    sets = {v["set"] for v in [*key["cue"].values(), *key["scene"].values()]}
    overrides: list[Override] = []
    for n, raw in enumerate(data.get("override", []), start=1):
        where = f"[[override]] {n}"
        _unknown(raw, frozenset({"set", "character", "backend", "mode", "notes"}), where)
        if raw.get("set") not in sets:
            raise ValueError(
                f"{where}: set {raw.get('set')!r} is not one of {', '.join(sorted(sets))}"
            )
        char = raw.get("character")
        if not isinstance(char, int) or isinstance(char, bool):
            raise ValueError(f"{where}: character must be the registry id (an integer)")
        backend = _str(raw.get("backend", ""), f"{where} backend")
        mode = raw.get("mode")
        cand: Candidate | None
        if backend == STOCK_PICK:
            if mode is not None:
                raise ValueError(f"{where}: a stock voice takes no mode")
            cand = None
        else:
            if backend not in {Candidate.parse(c).backend for c in names}:
                raise ValueError(f"{where}: backend {backend!r} is not a candidate here")
            if mode is not None and mode not in MODES.get(backend, ()):
                raise ValueError(f"{where}: {backend} has no mode {mode!r}")
            cand = Candidate(backend, mode)
        overrides.append(Override(raw["set"], char, cand, raw.get("notes", "")))
    return Verdict(pick, stock, passed, overall.get("notes", ""), cues, scenes, overrides)


def _known_slot(slot: str, table: Mapping[str, Any], kind: str) -> bool:
    if slot not in table:
        raise ValueError(f"[{kind}.{slot}] is not in this pack")
    return True


# -- the page and the readme ------------------------------------------------------------------

CSS = """
:root { color-scheme: light dark; --fg: #1d1d1f; --bg: #fafaf7; --muted: #6b6b6b;
  --card: #ffffff; --line: #ddd; }
@media (prefers-color-scheme: dark) { :root { --fg: #e8e8e3; --bg: #18181a; --muted: #a0a0a0;
  --card: #232326; --line: #3a3a3e; } }
body { font: 15px/1.45 system-ui, sans-serif; color: var(--fg); background: var(--bg);
  max-width: 1100px; margin: 0 auto; padding: 16px; }
h1 { font-size: 1.4em; } h2 { font-size: 1.15em; margin-top: 2em; }
section { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 16px; margin: 12px 0; }
.meta { color: var(--muted); font-size: 0.9em; }
.src { font-size: 1.05em; } .en { font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 8px 16px; margin-top: 8px; }
.grid div { display: flex; align-items: center; gap: 8px; }
.grid b { min-width: 3.5em; } audio { width: 100%; max-width: 230px; }
.ref b { color: var(--muted); }
"""


def _page(title: str, body: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>\n"
        f"<body>\n{body}\n</body></html>\n"
    )


def _player(label: str, src: str, css: str = "") -> str:
    cls = f' class="{css}"' if css else ""
    return (
        f"<div{cls}><b>{html.escape(label)}</b>"
        f'<audio controls preload="none" src="{html.escape(src)}"></audio></div>'
    )


def _index_html(
    cues: Sequence[tuple[str, CueMaterial, list[str], list[str]]],
    scenes: Sequence[tuple[str, SceneMaterial, list[str], bool]],
    numbers: bool,
) -> str:
    e = html.escape
    parts = [
        "<h1>TTS A/B listening</h1>",
        "<p>For each line: play the original, then every lettered render, and rank the "
        "letters in <code>verdict.toml</code>. Letters are shuffled per line. Every clip of a "
        "line is the same length and loudness; a render may start a little before its slot "
        "when timefit borrowed silence.</p>",
    ]
    current = None
    for slot, m, refs, labels in cues:
        if m.set != current:
            current = m.set
            parts.append(f"<h2>Set {e(m.set)} ({e(m.episode)})</h2>")
        emotion = f", source line tagged {e(m.emotion)}" if m.emotion else ""
        players = [
            _player(REFERENCE_TITLE[k], f"audio/{slot}_{REFERENCE_SUFFIX[k]}.wav", "ref")
            for k in refs
        ]
        players += [_player(lab, f"audio/{slot}_{lab}.wav") for lab in labels]
        parts.append(
            f'<section id="{slot}"><div class="meta">{slot} · {e(m.character)} '
            f"(character {m.character_id}) · {e(m.cue_id)} at {m.slot[0]:.1f} s{emotion}</div>"
            f'<div class="src">{e(m.source_text)}</div><div class="en">{e(m.text)}</div>'
            f'<div class="grid">{"".join(players)}</div></section>'
        )
    if scenes:
        parts.append("<h2>Scenes</h2>")
    for slot, s, labels, video in scenes:
        players = [_player("Original", f"audio/{slot}_ref.wav", "ref")]
        players += [_player(lab, f"audio/{slot}_{lab}.wav") for lab in labels]
        note = (
            f" Video with each track: <code>video/{slot}_&lt;letter&gt;.mkv</code> "
            "(play it with mpv)."
            if video
            else ""
        )
        parts.append(
            f'<section id="{slot}"><div class="meta">{slot} · set {e(s.set)} {e(s.episode)} · '
            f"{s.window[0]:.1f} to {s.window[1]:.1f} s · tts, timefit and mix in full.{note}</div>"
            f'<div class="grid">{"".join(players)}</div></section>'
        )
    if numbers:
        parts.append(
            '<p class="meta">After listening: <a href="numbers.html">the eval tts numbers</a> '
            "per candidate (they name candidates, not letters).</p>"
        )
    return _page("TTS A/B listening", "\n".join(parts))


def _readme(n_cues: int, n_scenes: int) -> str:
    return f"""TTS A/B listening pack
======================

{n_cues} lines and {n_scenes} scene(s). For each line there is the original audio
(cueNN_ref.wav), the show's own English track at the same moment where the set has one
(cueNN_ref_en.wav: the original English of a live-action show, or an anime's official
English dub), and one render per candidate under a shuffled letter (cueNN_A.wav,
cueNN_B.wav, ...). The letters are shuffled per line, so A on one line is not A on the
next. Every clip of a line has the same length and the same loudness (EBU R128,
{TARGET_LUFS:g} LUFS), so neither can tip the choice. The scenes (sceneNN_*.wav, and
video/sceneNN_*.mkv with the picture) run the whole dubbing path, tts, timefit and mix,
over 60 to 90 seconds.

Some lines were chosen because a render of them came out wrong in testing. Which ones, and
for which candidate, is in key.toml.

What to do
----------
1. Open index.html in a browser and listen. Write your ranking for every line and scene in
   verdict.toml, using the letters only (ranking = ["C", "A", ...], best first, or pick = "C").
2. Only then open key.toml. It maps each letter to a candidate.
3. Fill [overall] in verdict.toml by candidate name: pick (or "stock" if none is good
   enough, which keeps stock voices as the default), passed (every candidate you would
   accept), and any per-character [[override]] (for example one character on another mode).
4. Tell the controller the verdict is in. numbers.html has the eval tts numbers per
   candidate for afterwards.

Playing it over ssh
-------------------
The folder is self-contained: the page makes no network requests and every path in it is
relative, so it works wherever the folder goes.

  Copy it to your machine and open index.html there:
    rsync -a ml-01:{{this folder}}/ ~/ab-review/      (or scp -r)

  Or serve it from here, bound to localhost only, and tunnel the port:
    on ml-01:   python3 -m http.server 8765 --bind 127.0.0.1 --directory {{this folder}}
    locally:    ssh -N -L 8765:127.0.0.1:8765 ml-01
    then open   localhost:8765 in your browser; stop the server with Ctrl-C afterwards.

  Or play single files in the terminal session with mpv or ffplay -nodisp -autoexit
  (audio goes to the machine you are logged into, so this only helps on a local session).

The files are readable by your user only (0700 folders, 0600 files); keep it that way, since
the clips are cut from the episodes.
"""


# ---- gathering from a workspace (models; GPU for missing renders) ---------------------------


def _with_mode(cfg: Config, cand: Candidate) -> Config:
    """``cfg`` with ``[tts.<backend>] mode`` set to the candidate's mode, if any."""
    if cand.mode is None:
        return cfg
    raw = {**cfg.raw}
    tts = dict(raw.get("tts") or {})
    table = dict(tts.get(cand.backend) or {})
    table["mode"] = cand.mode
    tts[cand.backend] = table
    raw["tts"] = tts
    return dataclasses.replace(cfg, raw=raw)


@dataclass
class Rendered:
    path: Path
    seconds: float
    rate_factor: float
    key: str
    card: str
    voice: str | None
    retry_seed: int | None
    suspect: str | None
    renders_made: int


def _high_wer(ledger: Path | None, runs: Mapping[str, str]) -> dict[str, set[str]]:
    """Candidate -> the ``episode:cue`` its ledger row lists in ``high_wer_cues``."""
    from fukikae.eval.ledger import read_rows

    if not runs:
        return {}
    rows, _ = read_rows(ledger)
    out: dict[str, set[str]] = {}
    for cand, prefix in runs.items():
        hit = [r for r in rows if r.run_id.startswith(prefix)]
        if len(hit) != 1:
            raise ValueError(f"ledger run {prefix!r} for {cand}: {len(hit)} rows match")
        out[cand] = set(hit[0].detail.get("high_wer_cues") or [])
    return out


def _numbers_html(ledger: Path | None, sets: Sequence[SetSpec]) -> str | None:
    from fukikae.eval.ledger import read_rows

    if not any(s.ledger_runs for s in sets):
        return None
    rows, _ = read_rows(ledger)
    cols = [
        ("speaker_similarity_heldout", "similarity held-out"),
        ("speaker_similarity_ceiling", "ceiling"),
        ("wer", "WER pooled"),
        ("wer_median", "WER median"),
        ("wer_floor", "WER floor"),
        ("fit_rate", "fit"),
        ("cps_median", "cps"),
        ("emotion_agreement", "emotion agr."),
        ("suspect_cues", "suspect"),
        ("high_wer_cues", "WER > 0.5"),
        ("ref_duration_ratio", "duration vs English"),
    ]
    e = html.escape
    parts = [
        "<h1>eval tts numbers</h1><p>The ledger rows behind the candidates (8 cues per "
        "character, seed 0, deterministic ASR). Held-out similarity rests on 3 characters per "
        "set, so small gaps are noise.</p>"
    ]
    for s in sets:
        if not s.ledger_runs:
            continue
        head = "".join(f"<th>{e(t)}</th>" for _, t in cols)
        body = []
        for cand in s.candidates:
            prefix = s.ledger_runs.get(cand.name)
            row = next((r for r in rows if prefix and r.run_id.startswith(prefix)), None)
            cells = "".join(
                f"<td>{e(str(row.metrics.get(k, '')))}</td>" if row else "<td></td>"
                for k, _ in cols
            )
            body.append(f"<tr><td>{e(cand.name)}</td><td>{e(prefix or '')}</td>{cells}</tr>")
        parts.append(
            f"<h2>{e(s.name)}</h2><table><tr><th>candidate</th><th>run</th>{head}</tr>"
            + "".join(body)
            + "</table>"
        )
    return "\n".join(parts)


class _Renderer:
    """One LazySynth per (candidate, backend), closed at the end, each GPU job under the
    stage's own lock (ruling CD15)."""

    def __init__(self) -> None:
        self.synths: dict[tuple[str, str], Any] = {}
        self.plans: dict[Path, dict[str, Any]] = {}

    def _stage_render(self, job: RenderJob, work: Path, cache: Path) -> Rendered | None:
        """The tts stage's own render of this cue (B4), when its plan in ``work`` holds the
        same card and a key derived from this job's render key (at the card's rate, a fitted
        rate, or a guard retry) and that wav is still in the cache."""
        from fukikae.artifacts import read_json
        from fukikae.models import TtsPlan
        from fukikae.stages.tts import fitted_key, retry_key

        if work not in self.plans:
            path = work / "07-tts" / "plan.json"
            try:
                self.plans[work] = (
                    {c.cue_id: c for c in read_json(path, TtsPlan).cues} if path.exists() else {}
                )
            except (OSError, ValueError):
                self.plans[work] = {}
        pc = self.plans[work].get(job.cue.id)
        if pc is None or pc.card_id != job.card_id or pc.key is None or pc.seconds is None:
            return None
        firsts = {job.key, fitted_key(job.key, pc.rate_factor)}
        allowed = set(firsts)
        if pc.retry_seed is not None:
            retries = {retry_key(k, pc.retry_seed) for k in firsts}
            allowed |= retries | {fitted_key(k, pc.rate_factor) for k in retries}
        wav = cache / f"{pc.key}.wav"
        if pc.key not in allowed or not wav.exists():
            return None
        from fukikae.eval.tts import _voice_name

        return Rendered(
            wav, pc.seconds, pc.rate_factor, pc.key, job.card_id, _voice_name(job.card),
            pc.retry_seed, pc.suspect, 0,
        )  # fmt: skip

    def render(
        self,
        cand: Candidate,
        job: RenderJob,
        cfg: Config,
        ws: Workspace,
        store: CardStore,
        *,
        work: Path | None = None,
    ) -> Rendered:
        """The cue rendered for its slot; with ``work`` (the episode's work dir), the tts
        stage's own render is reused when it is the same render (B4)."""
        from fukikae.stages.tts import LazySynth, max_speedup, render_for_slot, synth_device

        if work is not None:
            reused = self._stage_render(job, work, ws.root / ".fukikae" / "tts-cache")
            if reused is not None:
                return reused
        backend = job.card.voice.backend
        k = (cand.name, backend)
        if k not in self.synths:
            # one model at a time: close the others before a new one loads
            self.close()
            self.synths[k] = LazySynth(job.backend_cfg, synth_device(cfg, job.cls), cfg)
        cache = ws.root / ".fukikae" / "tts-cache"
        cache.mkdir(parents=True, exist_ok=True)
        ref_dir = ws.root / ".fukikae" / "eval" / "tts-ab" / "ref"
        key, seconds, _first, factor, calls, g = render_for_slot(
            job, self.synths[k], cache, store, max_speedup(cfg), ref_dir
        )
        from fukikae.eval.tts import _voice_name

        return Rendered(
            cache / f"{key}.wav", seconds, factor, key, job.card_id, _voice_name(job.card),
            g.retry_seed, g.suspect, calls,
        )  # fmt: skip

    def close(self) -> None:
        for s in self.synths.values():
            s.close()
        self.synths.clear()


def pick_scene(
    cues: Sequence[Any],
    covered: set[int],
    blocked: Sequence[tuple[float, float]],
    seconds: float,
    total: float,
) -> tuple[float, float]:
    """The ``seconds`` window with the most distinct ``covered`` characters speaking, then the
    most spoken cues of theirs, that no ``blocked`` span (songs, singing) touches, from a
    cue's start minus 0.5 s; the earliest such window on a tie."""
    best: tuple[tuple[int, int], float] | None = None
    spoken = [c for c in cues if c.text and c.skip is None and c.error is None]
    for c in spoken:
        a = max(0.0, c.start - 0.5)
        b = a + seconds
        if b > total:
            continue
        if any(x < b and a < y for x, y in blocked):
            continue
        inside = [x for x in spoken if x.start >= a and x.end <= b]
        mine = [x for x in inside if x.character_id in covered]
        score = (len({x.character_id for x in mine}), len(mine))
        if best is None or score > best[0]:
            best = (score, a)
    if best is None:
        raise ValueError(f"no {seconds:g} s window clear of songs")
    return best[1], best[1] + seconds


def gather_set(
    spec: SetSpec, *, seed: int, ledger: Path | None, tmp: Path, log: Any = print
) -> tuple[list[CueMaterial], list[SceneMaterial], list[str]]:
    """The cue and scene materials of one set (see the module docstring)."""
    from fukikae.backends import loader
    from fukikae.cards.store import CardStore
    from fukikae.eval import tts as ev
    from fukikae.registry.store import Registry
    from fukikae.stages.tts import check_card, prepare_cue, tag_events
    from fukikae.timefit import apply_fit, fit_settings, plan_fit
    from fukikae.workspace import Workspace

    ws = Workspace(spec.workspace)
    cfg = ws.config
    target = cfg.target_language
    e = spec.episode
    docs = ev._cues(ws, [e], target)
    doc = docs[e]
    store = CardStore(ws.root)
    picked = ev.sample_cues(docs, SAMPLE_PER_CHARACTER, SAMPLE_SEED)
    chars = sorted({ev._cid(c) for _, c in picked})
    with Registry(ws.registry_path) as reg:
        names = {c.id: c.name for c in reg.characters() if c.name}
        assigned: dict[int, str] = {}
        for cid in chars:
            a = reg.card_for(cid, target)
            if a is not None and a.card_id is not None:
                assigned[cid] = a.card_id
    exclusions: dict[str, list[tuple[float, float]]] = {}
    cands = {cid: ev.character_candidates(ws, [e], cid, exclusions) for cid in chars}
    refs = {cid: ev.character_references(ws, [e], cid, candidates=cands[cid]) for cid in chars}
    notes: list[str] = []
    cards: dict[str, dict[int, str]] = {}
    clone_backends: set[str] = set()
    for cand in spec.candidates:
        ccfg = _with_mode(cfg, cand)
        cls = loader.backend_class("tts", cand.backend)
        sources = getattr(cls, "sources", frozenset())
        if "clone" in sources and "stock" not in sources:
            clone_backends.add(cand.name)
            cards[cand.name] = ev._clone_cards(ws, ccfg, store, cls, refs, doc.source_language)
        else:
            cards[cand.name] = ev._stock_cards(ws, ccfg, store, cand.backend, picked, assigned)
    covered = set(chars)
    for name in clone_backends:
        covered &= set(cards[name])
    left = sorted(set(chars) - covered)
    if left:
        notes.append(
            f"{spec.name}: characters without a clone card on every candidate left out: "
            + ", ".join(f"{c} ({names.get(c, '?')})" for c in left)
        )
    picked = [(ep, c) for ep, c in picked if ev._cid(c) in covered]
    high = _high_wer(ledger, spec.ledger_runs)
    events = tag_events(ws.work_dir(e))
    vocals = ws.work_dir(e) / "01-separate" / "vocals.wav"

    def who(cid: int) -> str:
        return f"character {cid}" + (f" ({names[cid]})" if cid in names else "")

    renderer = _Renderer()
    rendered: dict[tuple[str, str], Rendered] = {}
    cfgs = {c.name: _with_mode(cfg, c) for c in spec.candidates}
    try:
        for cand in spec.candidates:
            ccfg = cfgs[cand.name]
            checked = {
                cid: check_card(store, card, who(cid), target)
                for cid, card in cards[cand.name].items()
                if cid in covered
            }
            made = 0
            for _ep, c in picked:
                cid = ev._cid(c)
                job = prepare_cue(
                    c, cards[cand.name][cid], ccfg, store, who=who(cid), checked=checked[cid],
                    vocals=vocals, events=events,
                )  # fmt: skip
                r = renderer.render(cand, job, ccfg, ws, store, work=ws.work_dir(_ep))
                made += r.renders_made
                rendered[(cand.name, c.id)] = r
            renderer.close()
            log(f"{spec.name} {cand.name}: {len(picked)} cues, {made} render(s) made now")
    finally:
        renderer.close()

    infos = []
    for ep, c in picked:
        suspect: dict[str, str] = {}
        for cand in spec.candidates:
            r = rendered[(cand.name, c.id)]
            reasons = []
            if r.suspect:
                reasons.append(f"render guard: {r.suspect}")
            if f"{ep}:{c.id}" in high.get(cand.name, set()):
                reasons.append(f"eval tts WER > 0.5 (run {spec.ledger_runs[cand.name]})")
            if reasons:
                suspect[cand.name] = "; ".join(reasons)
        infos.append(CueInfo(ep, c.id, ev._cid(c), c.delivery.emotion, suspect))
    chosen = select_cues(infos, count=spec.cues, characters=spec.characters, seed=seed)
    for b in sorted({b for i in infos for b in i.suspect} - {b for i in chosen for b in i.suspect}):
        notes.append(f"{spec.name}: no suspect line of {b} could be kept")
    for cand in spec.candidates:
        if not any(cand.name in i.suspect for i in infos):
            notes.append(f"{spec.name}: {cand.name} has no suspect render in this sample")

    ordered = sorted(doc.cues, key=lambda c: (c.start, c.end, c.id))
    by_id = {c.id: c for c in ordered}
    original = ws.work_dir(e) / "00-extract" / "audio.wav"
    english: Path | None = None
    if spec.reference_workspace is not None:
        p = Workspace(spec.reference_workspace).work_dir(e) / "00-extract" / "audio.wav"
        if p.exists():
            english = p
        else:
            notes.append(f"{spec.name}: no English track at {p}")
    out: list[CueMaterial] = []
    for info in chosen:
        c = by_id[info.cue_id]
        n = ordered.index(c)
        prev = [x.end for x in ordered[:n]]
        prev_end = max(prev) if prev else None
        next_start = next((x.start for x in ordered[n + 1 :] if x.start >= c.start), None)
        renders: dict[str, Clip] = {}
        detail: dict[str, dict[str, Any]] = {}
        for cand in spec.candidates:
            r = rendered[(cand.name, c.id)]
            fp = plan_fit(
                r.seconds, (c.start, c.end), prev_end, next_start, r.rate_factor,
                fit_settings(cfgs[cand.name]),
            )  # fmt: skip
            dest = tmp / spec.name / cand.name.replace(":", "-") / f"{c.id}.wav"
            apply_fit(r.path, fp, dest)
            renders[cand.name] = Clip(dest, fp.start)
            detail[cand.name] = {
                "card": r.card, "voice": r.voice, "key": r.key, "seconds": r.seconds,
                "rate_factor": r.rate_factor, "retry_seed": r.retry_seed,
                "stretch": fp.stretch, "start": fp.start, "over_by": fp.over_by,
            }  # fmt: skip
        refs_clip = {"original": Clip(original, 0.0)}
        if english is not None:
            refs_clip["english"] = Clip(english, 0.0)
        out.append(
            CueMaterial(
                set=spec.name, episode=e, cue_id=c.id, character_id=info.character_id,
                character=names.get(info.character_id, f"character {info.character_id}"),
                source_text=c.source_text, text=cast(str, c.text), emotion=c.delivery.emotion,
                slot=(c.start, c.end), references=refs_clip, renders=renders,
                suspect=dict(info.suspect), info=detail,
            )
        )  # fmt: skip

    scenes: list[SceneMaterial] = []
    if spec.scene is not None:
        scenes.append(
            _scene(spec, ws, cfg, cfgs, store, cards, covered, names, assigned, tmp, notes, log)
        )
    return out, scenes, notes


def _scene(
    spec: SetSpec,
    ws: Workspace,
    cfg: Config,
    cfgs: Mapping[str, Config],
    store: CardStore,
    cards: Mapping[str, Mapping[int, str]],
    covered: set[int],
    names: Mapping[int, str],
    assigned: Mapping[int, str],
    tmp: Path,
    notes: list[str],
    log: Any,
) -> SceneMaterial:
    """One window through tts, timefit and mix per candidate (the stages' own functions over
    the window's cues)."""
    from scipy.io import wavfile

    from fukikae.artifacts import read_json
    from fukikae.eval import tts as ev
    from fukikae.mix import build_dub, loudness_stats, normalize, removal_windows, subtract, union
    from fukikae.models import MixReport, Tags
    from fukikae.stages.mix import mix_settings
    from fukikae.stages.tts import TtsStage, check_card, prepare_cue, stage_settings, tag_events
    from fukikae.timefit import apply_fit, fit_settings, plan_fit

    assert spec.scene is not None
    e = spec.episode
    work = ws.work_dir(e)
    doc = ev._cues(ws, [e], cfg.target_language)[e]
    ordered = sorted(doc.cues, key=lambda c: (c.start, c.end, c.id))
    tags_path = work / "04b-tag" / "tags.json"
    tags = read_json(tags_path, Tags) if tags_path.exists() else None
    blocked: list[tuple[float, float]] = [(c.start, c.end) for c in ordered if c.skip == "song"]
    blocked += [(t.start, t.end) for t in (tags.events if tags else []) if t.label == "singing"]
    report = work / "09-mix" / "report.json"
    if report.exists():
        blocked += [(p.start, p.end) for p in read_json(report, MixReport).songs]
    rate, bg = wavfile.read(work / "01-separate" / "background.wav", mmap=True)
    total = len(bg) / rate
    if spec.scene.start is not None:
        window = (spec.scene.start, spec.scene.start + spec.scene.seconds)
    else:
        window = pick_scene(ordered, covered, blocked, spec.scene.seconds, total)
    w0, w1 = window
    inside = [c for c in ordered if c.start >= w0 and c.end <= w1]
    spoken = [c for c in inside if TtsStage._skipped(c) is None]
    default_backend = stage_settings(cfg)[0].backend
    extra_cards = TtsStage._extra_cards(doc.cues, cfg, store, default_backend)
    chars = sorted({c.character_id for c in spoken if c.character_id is not None})
    stock = ev._stock_cards(
        ws, cfg, store, default_backend, [(e, c) for c in spoken if c.character_id is not None],
        dict(assigned),
    )  # fmt: skip
    vocals = work / "01-separate" / "vocals.wav"
    events = tag_events(work)
    s = mix_settings(cfg)
    sdir = tmp / spec.name / "scene"
    sdir.mkdir(parents=True, exist_ok=True)
    # the window's bed and vocals, cut once
    bed_path, vo_path = sdir / "background.wav", sdir / "vocals.wav"
    vrate, vo = wavfile.read(vocals, mmap=True)
    wavfile.write(bed_path, rate, np.asarray(bg[round(w0 * rate) : round(w1 * rate)]))
    wavfile.write(vo_path, vrate, np.asarray(vo[round(w0 * vrate) : round(w1 * vrate)]))
    original = work / "00-extract" / "audio.wav"
    orig_cut = sdir / "original.wav"
    orate, odata = wavfile.read(original, mmap=True)
    wavfile.write(orig_cut, orate, np.asarray(odata[round(w0 * orate) : round(w1 * orate)]))
    orig_level = loudness_stats(orig_cut)["i"]
    shifted = (
        tags.model_copy(
            update={
                "events": [
                    t.model_copy(update={"start": t.start - w0, "end": t.end - w0})
                    for t in tags.events
                ]
            }
        )
        if tags
        else None
    )
    before = [c for c in ordered if c.start < w0]
    renders: dict[str, Clip] = {}
    info: dict[str, dict[str, Any]] = {}
    fallback = sorted(c for c in chars if c not in covered)
    renderer = _Renderer()
    try:
        for cand in spec.candidates:
            ccfg = cfgs[cand.name]
            mine = cards[cand.name]

            def card_for(c: Any, mine: Mapping[int, str] = mine) -> str:
                if c.character_id is None:
                    return extra_cards[c.extra or "unknown"]
                return mine.get(c.character_id) or stock[c.character_id]

            jobs = []
            for c in spoken:
                card = card_for(c)
                who = c.speaker
                job = prepare_cue(
                    c, card, ccfg, store, who=who,
                    checked=check_card(store, card, who, cfg.target_language),
                    vocals=vocals, events=events,
                )  # fmt: skip
                jobs.append(job)
            jobs.sort(key=lambda j: (j.card.voice.backend, j.cue.start))
            done: dict[str, Rendered] = {}
            for job in jobs:
                done[job.cue.id] = renderer.render(cand, job, ccfg, ws, store, work=work)
            renderer.close()
            fs = fit_settings(ccfg)
            cdir = sdir / cand.name.replace(":", "-")
            placed: list[tuple[Path, float]] = []
            played: list[tuple[float, float]] = []
            slots: list[tuple[float, float]] = []
            over = 0
            prev_end: float | None = max((c.end for c in before), default=None)
            for c in inside:
                next_start = next(
                    (x.start for x in ordered[ordered.index(c) + 1 :] if x.start >= c.start), None
                )
                r = done.get(c.id)
                if r is None:
                    prev_end = c.end if prev_end is None else max(prev_end, c.end)
                    continue
                fp = plan_fit(r.seconds, (c.start, c.end), prev_end, next_start, r.rate_factor, fs)
                out_wav = cdir / f"{c.id}.wav"
                secs = apply_fit(r.path, fp, out_wav)
                placed.append((out_wav, fp.start - w0))
                played.append((fp.start - w0, fp.start - w0 + secs))
                slots.append((c.start - w0, c.end - w0))
                over += fp.over_by > 1e-3
                end = fp.start + secs
                prev_end = end if prev_end is None else max(prev_end, end)
            dubbed = {c.id for c in inside if c.id in done}
            undubbed = [(c.start - w0, c.end - w0) for c in inside if c.id not in dubbed]
            dub_spans = union([*played, *subtract(slots, undubbed)])
            if s.keep_original_vocals:
                windows = removal_windows(dub_spans, shifted, protect_bgm=s.protect_bgm)
            else:
                windows = removal_windows([(0.0, w1 - w0)], shifted, protect_bgm=s.protect_bgm)
            premix = cdir / "premix.wav"
            stats = build_dub(bed_path, vo_path, placed, windows, s, premix)
            mixed = cdir / "scene.wav"
            normalize(premix, orig_level, mixed, s.true_peak_db)
            renders[cand.name] = Clip(mixed, w0)
            info[cand.name] = {
                "cues": len(done),
                "over": over,
                "dub_gain_db": None if stats.dub_gain_db is None else round(stats.dub_gain_db, 2),
                "stock_fallback": [f"{c} ({names.get(c, '?')})" for c in fallback]
                if cand.name in {x.name for x in spec.candidates if x.backend not in STOCK_BACKENDS}
                else [],
                "rendered_now": sum(r.renders_made for r in done.values()),
            }
            log(f"{spec.name} scene {cand.name}: {len(done)} cues, "
                f"{info[cand.name]['rendered_now']} rendered now")  # fmt: skip
    finally:
        renderer.close()
    if fallback:
        notes.append(
            f"{spec.name} scene: stock voice for characters without enough clean speech: "
            + ", ".join(f"{c} ({names.get(c, '?')})" for c in fallback)
        )
    video: Path | None = None
    try:
        video = ws.find_episode(e).source
    except FileNotFoundError as err:
        notes.append(f"{spec.name} scene: no video ({err}); audio only")
    return SceneMaterial(spec.name, e, window, Clip(original, 0.0), renders, info, video)


def build_pack(spec: PackSpec, out: Path, *, log: Any = print) -> tuple[Path, list[str]]:
    """Gather every set and write the pack to ``out``; returns it and the notes."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; give a new folder")
    cues: list[CueMaterial] = []
    scenes: list[SceneMaterial] = []
    notes: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fukikae-ab-") as tmpdir:
        tmp = Path(tmpdir)
        for s in spec.sets:
            c, sc, n = gather_set(s, seed=spec.seed, ledger=spec.ledger, tmp=tmp, log=log)
            cues += c
            scenes += sc
            notes += n
        write_pack(
            out, seed=spec.seed, cues=cues, scenes=scenes,
            numbers=_numbers_html(spec.ledger, spec.sets), notes=notes,
        )  # fmt: skip
    return out, notes


__all__ = [
    "Candidate",
    "Clip",
    "CueInfo",
    "CueMaterial",
    "PackSpec",
    "SceneMaterial",
    "SetSpec",
    "Verdict",
    "blind_order",
    "build_pack",
    "load_spec",
    "read_verdict",
    "select_cues",
    "write_pack",
]
