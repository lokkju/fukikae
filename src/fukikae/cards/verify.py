"""Card fingerprints: does this card still render the voice it rendered when recorded?

``verify_card`` renders ``TEST_SCRIPT[card.target_language]`` (three fixed sentences) with the
card's own backend and seed, through the tts stage's ``render_cue`` (ruling CD12: the same
trim and gain as a dub), embeds the three renders end to end with the fingerprint's embedder
(only an allowlisted embedder, ``FINGERPRINT_EMBEDDERS``, loaded from the local cache and never
downloaded, ruling CD23(a); today the pinned WeSpeaker ResNet34, the identity embedder, so one
embedding space for the project) and compares that with the stored vector: ``match`` when the
cosine reaches the fingerprint's threshold, else ``mismatch``. ``record`` stores the new vector
as a ``vector`` sidecar and returns the card with its ``[voice.fingerprint]``; the fingerprint
is outside the card id (spec), so recording changes no id, assignment or cache key.

It reports, never raises, for what stops a verification: ``no_fingerprint`` (nothing recorded
and not recording; nothing rendered), ``missing_sidecar`` (a clip, latent or the fingerprint
vector absent or corrupted; nothing rendered), ``unavailable`` (backend not installed or its
environment not set up, a different model revision installed, no test script for the
language, the embedder not loadable). A fingerprint says a voice is the same voice, not that it
is a good one.

Rendering and embedding run one after the other: the synth is closed before the embedder
loads (one model at a time). The caller's ``make_synth`` decides the device and holds the GPU
lock for a cloning backend (the CLI uses the tts stage's ``LazySynth``: cards on GPU 0 under
``gpu_lock``, Kokoro on the CPU); the CLI's embedder runs on the CPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import numpy as np

from fukikae.backends import loader
from fukikae.cards.card import Card, CardError, Fingerprint, card_id, validate_for_backend
from fukikae.cards.store import CardStore
from fukikae.config import ConfigError

if TYPE_CHECKING:
    from fukikae.backends.base import Embedder, Synthesizer

# Three sentences per target language: varied vowels and consonants, about 3 s each.
TEST_SCRIPT: dict[str, list[str]] = {
    "en": [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Would you mind telling me where the old library used to be?",
        "We have seven hours left, so pack your things and meet me outside.",
    ],
    "es": [
        "El veloz murciélago hindú comía feliz cardillo y kiwi.",
        "¿Podrías decirme dónde estaba la vieja biblioteca?",
        "Nos quedan siete horas, así que recoge tus cosas y espérame fuera.",
    ],
    "de": [
        "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.",
        "Könntest du mir sagen, wo die alte Bibliothek früher war?",
        "Wir haben noch sieben Stunden, also pack deine Sachen und warte draußen.",
    ],
    "fr": [
        "Portez ce vieux whisky au juge blond qui fume.",
        "Pourriez-vous me dire où se trouvait l'ancienne bibliothèque ?",
        "Il nous reste sept heures, alors prends tes affaires et attends-moi dehors.",
    ],
}
FINGERPRINT_EMBEDDER = "wespeaker:pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_THRESHOLD = 0.85
GAP_S = 0.25  # silence between the test sentences in the embedded file
EMBED_RATE = 16000

VerifyStatus = Literal[
    "match", "mismatch", "no_fingerprint", "missing_sidecar", "unavailable", "recorded"
]


@dataclass(frozen=True)
class VerifyResult:
    """``recorded`` (with ``record``): a new fingerprint was made; ``cosine`` is then the
    comparison with the previous one, when there was one."""

    status: VerifyStatus
    cosine: float | None
    detail: str


class SynthSource(Protocol):
    """A synthesizer created and loaded on first ``get``, released by ``close``."""

    def get(self) -> Synthesizer: ...

    def close(self) -> None: ...


class EmbedderUnavailable(Exception):  # noqa: N818 - named like loader.BackendUnavailable
    """A fingerprint embedder that is not on the allowlist, or not in the local cache."""


@dataclass(frozen=True)
class PinnedEmbedder:
    """A fingerprint embedder fukikae trusts: an embed backend and the one checkpoint file,
    at a pinned revision, it loads from the local Hugging Face cache."""

    backend: str
    repo: str
    revision: str
    filename: str


# Ruling CD23(a): the only embedders verify loads. A card's embedder string is untrusted (a
# pack carries it), so it selects from this table and never names a model to fetch.
FINGERPRINT_EMBEDDERS: dict[str, PinnedEmbedder] = {
    FINGERPRINT_EMBEDDER: PinnedEmbedder(
        backend="wespeaker",
        repo="pyannote/wespeaker-voxceleb-resnet34-LM",
        revision="837717ddb9ff5507820346191109dc79c958d614",
        filename="pytorch_model.bin",
    ),
}


def try_to_load_from_cache(repo: str, filename: str, revision: str | None = None) -> Any:
    """``huggingface_hub.try_to_load_from_cache``: a lookup in the local cache, never the
    network."""
    from huggingface_hub import try_to_load_from_cache as lookup

    return lookup(repo, filename, revision=revision)


def local_checkpoint(name: str) -> tuple[PinnedEmbedder, Path]:
    """The allowlisted embedder ``name`` and its checkpoint file in the local cache; raises
    ``EmbedderUnavailable`` for any other name or a checkpoint that isn't there. Nothing is
    downloaded."""
    pinned = FINGERPRINT_EMBEDDERS.get(name)
    if pinned is None:
        raise EmbedderUnavailable(_not_allowed(name))
    found = try_to_load_from_cache(pinned.repo, pinned.filename, revision=pinned.revision)
    if not isinstance(found, str) or not Path(found).is_file():
        raise EmbedderUnavailable(
            f"{pinned.repo}@{pinned.revision[:12]} ({pinned.filename}) is not in the local "
            "Hugging Face cache, and verify never downloads; run any WeSpeaker stage once "
            "(identity, eval) to fetch it"
        )
    return pinned, Path(found)


def _not_allowed(name: str) -> str:
    return (
        f"fingerprint embedder {name!r} is not on the allowlist "
        f"({', '.join(sorted(FINGERPRINT_EMBEDDERS))}); nothing was loaded"
    )


def _needed_sidecars(card: Card) -> list[str]:
    refs = [r.clip for r in card.voice.references]
    if card.voice.latent:
        refs.append(card.voice.latent)
    return refs


def _sidecar_problem(store: CardStore, refs: list[str]) -> str | None:
    problems = []
    for ref in refs:
        try:
            store.sidecar_path(ref)
        except CardError as e:
            problems.append(str(e))
    return "; ".join(problems) or None


def _backend_problem(card: Card, store: CardStore) -> str | None:
    """Why the card's backend can't render it here, or None."""
    v = card.voice
    try:
        cls = cast("type[Synthesizer]", loader.backend_class("tts", v.backend))
    except (ConfigError, loader.BackendUnavailable) as e:
        return f"backend {v.backend!r} is not installed: {e}"
    rev = getattr(cls, "model_revision", None)
    if rev and v.model_revision != rev:
        return (
            f"card was made for {v.backend} model revision {v.model_revision!r}, and this "
            f"fukikae renders with {rev!r}"
        )
    try:
        validate_for_backend(card, cls, lookup=store.get)
    except CardError as e:
        return f"{v.backend} can't render this card: {e}"
    if not loader.supports(cls, card.target_language):
        return f"{v.backend} does not speak {card.target_language!r}"
    return None


def _concat(paths: list[Path], out: Path) -> float:
    """The wavs end to end with ``GAP_S`` of silence between, as 16 kHz mono 16-bit pcm (what
    the embedders read); returns seconds."""
    from math import gcd

    from scipy.io import wavfile
    from scipy.signal import resample_poly

    parts: list[np.ndarray] = []
    for p in paths:
        rate, data = wavfile.read(p)
        x = data.astype(np.float64)
        if np.issubdtype(data.dtype, np.integer):
            x /= float(np.iinfo(data.dtype).max) + 1.0
        if x.ndim > 1:
            x = x.mean(axis=1)
        if rate != EMBED_RATE:
            g = gcd(EMBED_RATE, rate)
            x = resample_poly(x, EMBED_RATE // g, rate // g)
        if parts:
            parts.append(np.zeros(int(GAP_S * EMBED_RATE)))
        parts.append(x)
    y = np.concatenate(parts)
    wavfile.write(out, EMBED_RATE, (np.clip(y, -1.0, 1.0) * 32767).astype("<i2"))
    return len(y) / EMBED_RATE


def _render(card: Card, store: CardStore, synth: SynthSource, work: Path) -> list[Path]:
    from fukikae.config import BackendConfig
    from fukikae.models import Delivery, DubCue
    from fukikae.stages.tts import RenderJob, render_cue

    cid = card_id(card)
    cls = cast("type[Synthesizer]", loader.backend_class("tts", card.voice.backend))
    out: list[Path] = []
    for i, text in enumerate(TEST_SCRIPT[card.target_language]):
        cue = DubCue(
            id=f"verify{i}", start=0.0, end=30.0, character_id=None, extra=None,
            speaker="verify", source_text=text, text=text, budget=len(text),
            delivery=Delivery(reference=(0.0, 0.0)),
        )  # fmt: skip
        # neutral: no emotion audio, vector, instruction or target length
        bc = BackendConfig(card.voice.backend, {})
        job = RenderJob(cue, cid, card, cls, bc, key="verify", slot=30.0, passed={})
        dest = work / f"verify{i}.wav"
        render_cue(job, synth, dest, store)
        out.append(dest)
    return out


def _load_vector(path: Path) -> np.ndarray:
    v = np.load(path, allow_pickle=False)
    if not isinstance(v, np.ndarray) or v.ndim != 1:
        raise CardError(f"fingerprint vector {path.name} is not a one-dimensional array")
    return v.astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def verify_card(
    card: Card,
    store: CardStore,
    *,
    make_synth: Callable[[Card], SynthSource],
    make_embedder: Callable[[str], contextlib.AbstractContextManager[Embedder]],
    record: bool,
    work: Path,
) -> tuple[VerifyResult, Card]:
    """Verify (or with ``record``, fingerprint) ``card``; returns the result and the card,
    which carries the new fingerprint after a recording and is ``card`` otherwise. The
    caller stores the returned card (its id is unchanged)."""
    fp = card.voice.fingerprint
    if fp is None and not record:
        return VerifyResult(
            "no_fingerprint", None,
            "the card has no fingerprint; record one with `fukikae voices verify --record`",
        ), card  # fmt: skip
    problem = _sidecar_problem(store, _needed_sidecars(card))
    if problem:
        return VerifyResult("missing_sidecar", None, problem), card
    stored: np.ndarray | None = None
    if fp is not None:
        vec_problem = _sidecar_problem(store, [fp.vector])
        if vec_problem and not record:
            return VerifyResult("missing_sidecar", None, f"fingerprint vector: {vec_problem}"), card
        if not vec_problem:
            try:
                stored = _load_vector(store.sidecar_path(fp.vector))
            except (CardError, ValueError, OSError) as e:
                if not record:
                    return VerifyResult("missing_sidecar", None, f"fingerprint vector: {e}"), card
    # the embedder: always the pinned one to record; the card's own to verify, which must be
    # on the allowlist (ruling CD23(a)), checked before anything renders or loads
    name = FINGERPRINT_EMBEDDER if record else (fp.embedder if fp is not None else "")
    if name not in FINGERPRINT_EMBEDDERS:
        return VerifyResult("unavailable", None, _not_allowed(name)), card
    if fp is not None and fp.embedder != name:
        stored = None  # another embedder's vector: nothing to compare the new one with
    if card.target_language not in TEST_SCRIPT:
        return VerifyResult(
            "unavailable", None,
            f"no test script for target language {card.target_language!r} "
            f"(there are: {', '.join(sorted(TEST_SCRIPT))})",
        ), card  # fmt: skip
    problem = _backend_problem(card, store)
    if problem:
        return VerifyResult("unavailable", None, problem), card

    work.mkdir(parents=True, exist_ok=True)
    try:
        synth = make_synth(card)
    except (ConfigError, CardError, loader.BackendUnavailable) as e:
        return VerifyResult("unavailable", None, f"{card.voice.backend}: {e}"), card
    try:
        renders = _render(card, store, synth, work)
    except loader.BackendUnavailable as e:
        return VerifyResult("unavailable", None, f"{card.voice.backend}: {e}"), card
    finally:
        synth.close()
    test = work / "verify.wav"
    seconds = _concat(renders, test)

    try:
        with make_embedder(name) as emb:
            vec = np.asarray(emb.embed(test, [(0.0, seconds)])[0], dtype=np.float32)
    except (
        EmbedderUnavailable, ConfigError, loader.BackendUnavailable, ImportError, RuntimeError,
        OSError,
    ) as e:  # fmt: skip
        return VerifyResult("unavailable", None, f"embedder {name!r}: {e}"), card

    cos = None
    if stored is not None:
        if stored.shape != vec.shape:
            if not record:
                return VerifyResult(
                    "unavailable", None,
                    f"the stored fingerprint has {stored.size} dimensions and {name} gives "
                    f"{vec.size}; record it again",
                ), card  # fmt: skip
        else:
            cos = _cosine(stored, vec)
    if record:
        path = work / "fingerprint.npy"
        np.save(path, vec)
        ref = store.add_sidecar(path, "vector")
        threshold = fp.threshold if fp is not None and fp.embedder == name else DEFAULT_THRESHOLD
        new = replace(
            card, voice=replace(card.voice, fingerprint=Fingerprint(name, ref, threshold))
        )
        before = f"; the previous fingerprint gave cosine {cos:.3f}" if cos is not None else ""
        return VerifyResult(
            "recorded", cos, f"fingerprint recorded with {name} (threshold {threshold:g}){before}"
        ), new  # fmt: skip
    assert fp is not None and cos is not None
    status: VerifyStatus = "match" if cos >= fp.threshold else "mismatch"
    return VerifyResult(
        status, cos, f"cosine {cos:.3f} against threshold {fp.threshold:g} ({fp.embedder})"
    ), card  # fmt: skip
