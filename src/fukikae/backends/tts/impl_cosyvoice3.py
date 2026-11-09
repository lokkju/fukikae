"""CosyVoice 3 inside the TTS worker (``fukikae.backends.tts.worker``).

Runs under ``<checkout>/.venvs/cosyvoice3`` (Python 3.10, ``workers/cosyvoice3``), so the module
level imports only the standard library; ``fukikae.backends.tts.cosyvoice3`` imports this module
in fukikae's own environment to read the pins and to use the pure helpers below.

Pins:

- the model snapshot ``FunAudioLLM/Fun-CosyVoice3-0.5B-2512`` at ``REVISION``, only the files
  inference reads (``MODEL_FILES``: the base LLM ``llm.pt``, not ``llm.rl.pt``; no TensorRT
  or batch ONNX files);
- the code: CosyVoice and its Matcha-TTS submodule ship no package, so ``fetch`` downloads each
  repository's archive at its pinned commit (``CODE``), keeps only the package directory it
  needs, and refuses it unless the extracted files hash to the pinned ``digest``
  (``tree_digest``). ``load`` hashes them again before importing anything from them, so an
  edited or swapped checkout never runs.

``fetch`` (``fukikae backends setup cosyvoice3``) is the only step that downloads; ``load``
reads the Hugging Face cache with ``local_files_only`` (the worker runs with
``HF_HUB_OFFLINE=1``) and names the setup command when something is missing.

What it honours (the class in ``fukikae.backends.tts.cosyvoice3`` declares the same):

- the speaker: the card's reference clips joined in card order, with 0.2 s of silence, into
  one prompt of whole clips up to ``PROMPT_MAX_S`` (CosyVoice's speech tokenizer refuses over
  30 s), with their transcripts joined the same way. The prompt's features are computed once
  per card and reused (CosyVoice's ``add_zero_shot_spk``);
- the mode (option ``mode``): ``zero_shot`` gives the LLM the prompt's transcript and speech
  tokens, so it continues the speaker's prosody, and needs the transcript to match the clips;
  ``cross_lingual`` gives the LLM neither (only the flow model sees the prompt), CosyVoice's
  recommendation when the prompt's language differs from the text's. ``auto`` (the default)
  picks ``cross_lingual`` when any reference clip's language differs from the target;
- no text normalizer: wetext (CosyVoice's default) downloads its grammars from ModelScope,
  unpinned, at every start, so it is not installed and ``load`` refuses one that appears;
  CosyVoice still spells out English numbers with inflect;
- ``rate`` as CosyVoice's ``speed`` (the mel spectrogram resampled in time before the vocoder,
  so pitch is kept); ``seed`` (``random``, numpy and torch seeded before each line; 0 when the
  card sets none). Sampling (top-k 25, top-p 0.8, repetition-aware) is the model config's.

Every line is neutral: no instruction (CosyVoice's instruct mode was withdrawn under ruling
CD20: its prompt wording for a free-form direction was unverified, and it replaces the LLM's
speaker prompt), no emotion audio or vector, no per-line duration target. A request carrying
an instruction is refused. CosyVoice's own control tokens (``CONTROL_MARKUP``: ``[laughter]``,
``[breath]``, ``<strong>``, the phoneme tags and the rest of its tokenizer's added tokens) are
stripped from the text and transcripts by the worker before they reach the model, as the
``<|...|>`` syntax is for every backend.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import wave
from typing import Any

REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
MODEL_FILES = [
    "cosyvoice3.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "CosyVoice-BlankEN/config.json",
    "CosyVoice-BlankEN/generation_config.json",
    "CosyVoice-BlankEN/merges.txt",
    "CosyVoice-BlankEN/model.safetensors",
    "CosyVoice-BlankEN/tokenizer_config.json",
    "CosyVoice-BlankEN/vocab.json",
]
# name -> (GitHub owner/repo, commit, package directory kept, sha256 tree digest of it)
CODE: dict[str, tuple[str, str, str, str]] = {
    "CosyVoice": (
        "FunAudioLLM/CosyVoice",
        "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
        "cosyvoice",
        "c2332488251e2562a7054a4f3e4cc5f8bc8be2c888e99d0c3a6057b47394a6b4",
    ),
    "Matcha-TTS": (
        "shivammehta25/Matcha-TTS",
        "dd9105b34bf2be2230f4aa1e4769fb586a3c824e",
        "matcha",
        "b83b9cf5e88ec34082daa0a80dc9083891a108b46ae985a991528cb456ec1c4c",
    ),
}
ARCHIVE_URL = "https://codeload.github.com/{repo}/tar.gz/{commit}"
MAX_ARCHIVE = 256 << 20
SAMPLE_RATE = 24000
PROMPT_MAX_S = 25.0
PROMPT_GAP_S = 0.2
# Fun-CosyVoice3's languages, without Japanese: it needs katakana input, which fukikae doesn't
# produce.
LANGUAGES = ("en", "zh", "ko", "de", "es", "fr", "it", "ru")
MODES = ("auto", "zero_shot", "cross_lingual")
SYSTEM = "You are a helpful assistant."
END = "<|endofprompt|>"
# CosyVoice3Tokenizer's added special tokens at the pinned commit outside the <|...|> form the
# worker strips for every backend (ruling CD20): the event and emphasis tags and the CMU and
# pinyin phoneme tags, matched exactly (a bracketed word that isn't a token stays).
_EVENT_TAGS = (
    "breath", "noise", "laughter", "cough", "clucking", "accent", "quick_breath", "hissing",
    "sigh", "vocalized-noise", "lipsmack", "mn",
)  # fmt: skip
_CMU = (
    "AA", "AA0", "AA1", "AA2", "AE", "AE0", "AE1", "AE2", "AH", "AH0", "AH1", "AH2", "AO", "AO0",
    "AO1", "AO2", "AW", "AW0", "AW1", "AW2", "AY", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
    "EH", "EH0", "EH1", "EH2", "ER", "ER0", "ER1", "ER2", "EY", "EY0", "EY1", "EY2", "F", "G",
    "HH", "IH", "IH0", "IH1", "IH2", "IY", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG",
    "OW", "OW0", "OW1", "OW2", "OY", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH", "UH",
    "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH",
)  # fmt: skip
_PINYIN = (
    "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng", "f", "g", "h", "i",
    "ian", "in", "ing", "iu", "ià", "iàn", "iàng", "iào", "iá", "ián", "iáng", "iáo", "iè", "ié",
    "iòng", "ióng", "iù", "iú", "iā", "iān", "iāng", "iāo", "iē", "iě", "iōng", "iū", "iǎ", "iǎn",
    "iǎng", "iǎo", "iǒng", "iǔ", "j", "k", "l", "m", "n", "o", "ong", "ou", "p", "q", "r", "s",
    "sh", "t", "u", "uang", "ue", "un", "uo", "uà", "uài", "uàn", "uàng", "uá", "uái", "uán",
    "uáng", "uè", "ué", "uì", "uí", "uò", "uó", "uā", "uāi", "uān", "uāng", "uē", "uě", "uī",
    "uō", "uǎ", "uǎi", "uǎn", "uǎng", "uǐ", "uǒ", "vè", "w", "x", "y", "z", "zh", "à", "ài", "àn",
    "àng", "ào", "á", "ái", "án", "áng", "áo", "è", "èi", "èn", "èng", "èr", "é", "éi", "én",
    "éng", "ér", "ì", "ìn", "ìng", "í", "ín", "íng", "ò", "òng", "òu", "ó", "óng", "óu", "ù",
    "ùn", "ú", "ún", "ā", "āi", "ān", "āng", "āo", "ē", "ēi", "ēn", "ēng", "ě", "ěi", "ěn", "ěng",
    "ěr", "ī", "īn", "īng", "ō", "ōng", "ōu", "ū", "ūn", "ǎ", "ǎi", "ǎn", "ǎng", "ǎo", "ǐ", "ǐn",
    "ǐng", "ǒ", "ǒng", "ǒu", "ǔ", "ǔn", "ǘ", "ǚ", "ǜ",
)  # fmt: skip
CONTROL_MARKUP: tuple[str, ...] = (
    r"</?(?:strong|laughter)>",
    r"\[(?:"
    + "|".join(re.escape(t) for t in sorted({*_EVENT_TAGS, *_CMU, *_PINYIN}, key=len, reverse=True))
    + r")\]",
)

SETUP = "fukikae backends setup cosyvoice3"


# ---- pure helpers (also used in fukikae's environment, by the tests) -----------------------


def tree_digest(root: str) -> str:
    """sha256 over every file under ``root`` (relative path and content hash, sorted),
    bytecode included; a symbolic link anywhere is refused."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirnames)
        for d in dirnames:
            if os.path.islink(os.path.join(dirpath, d)):
                raise RuntimeError(f"{os.path.join(dirpath, d)} is a symbolic link")
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                raise RuntimeError(f"{path} is a symbolic link")
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            with open(path, "rb") as f:
                h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(f.read()).digest())
    return h.hexdigest()


def extract_package(archive: bytes, package: str, dest: str) -> None:
    """Write ``<top>/<package>/...`` of a GitHub archive (gzip tar) under ``dest/<package>``:
    regular files and directories only; links, devices, absolute paths and ``..`` refused."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError("empty archive")
        top = members[0].name.split("/", 1)[0]
        prefix = f"{top}/{package}/"
        for m in members:
            name = m.name
            if name.startswith("/") or ".." in name.split("/"):
                raise RuntimeError(f"archive member {name!r} escapes its directory")
            if name != f"{top}/{package}" and not name.startswith(prefix):
                continue
            if not (m.isfile() or m.isdir()):
                raise RuntimeError(f"archive member {name!r} is not a regular file or directory")
            rel = name[len(top) + 1 :]
            target = os.path.join(dest, *rel.split("/"))
            if m.isdir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            src = tar.extractfile(m)
            assert src is not None
            with open(target, "wb") as out:
                out.write(src.read())


def code_root(name: str, prefix: str | None = None) -> str:
    """Where ``fetch`` puts a pinned repository's package: inside the worker environment."""
    _, commit, _, _ = CODE[name]
    return os.path.join(prefix or sys.prefix, "fukikae-code", f"{name}-{commit[:12]}")


def card_mode(card_mode: str | None, stage_mode: str | None) -> str:
    """The mode to resolve: the card's own (``cosyvoice3_mode``, ruling CD21) unless it is at its
    default ``cross_lingual``, then the stage's ``[tts.cosyvoice3] mode`` when it sets one."""
    card_mode = card_mode or "cross_lingual"
    if card_mode not in MODES:
        raise ValueError(f"cosyvoice3 mode must be one of {', '.join(MODES)}, not {card_mode!r}")
    if card_mode == "cross_lingual" and stage_mode:
        return stage_mode
    return card_mode


def resolve_mode(mode: str, reference_languages: list[str], target: str) -> str:
    """``zero_shot`` or ``cross_lingual``: ``auto`` is cross-lingual when any reference clip's
    language (when known) differs from the target language."""
    if mode not in MODES:
        raise ValueError(f"cosyvoice3 mode must be one of {', '.join(MODES)}, not {mode!r}")
    if mode != "auto":
        return mode
    return (
        "cross_lingual"
        if any(lang and lang != target for lang in reference_languages)
        else ("zero_shot")
    )


def prompt_plan(refs: list[tuple[str, float, str]]) -> tuple[list[str], str]:
    """(clips, joined transcript) of the whole clips, in card order, that fit ``PROMPT_MAX_S``
    with ``PROMPT_GAP_S`` between them; the first clip is always taken (cut to fit later).
    ``refs`` is (clip, seconds, transcript)."""
    clips: list[str] = []
    texts: list[str] = []
    total = 0.0
    for clip, seconds, text in refs:
        need = seconds + (PROMPT_GAP_S if clips else 0.0)
        if clips and total + need > PROMPT_MAX_S:
            break
        clips.append(clip)
        texts.append(text.strip())
        total += need
    return clips, " ".join(t for t in texts if t)


def drop_bytecode(root: str) -> None:
    """Remove every ``__pycache__`` under ``root``: only checked source may be imported."""
    for dirpath, dirnames, _ in os.walk(root):
        for d in [d for d in dirnames if d == "__pycache__"]:
            shutil.rmtree(os.path.join(dirpath, d))
            dirnames.remove(d)


# ---- the worker side -------------------------------------------------------------------------


def _snapshot() -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(
            REPO, revision=REVISION, allow_patterns=MODEL_FILES, local_files_only=True
        )
    except LocalEntryNotFoundError as e:
        raise RuntimeError(
            f"{REPO}@{REVISION[:12]} is not in the Hugging Face cache; download it with `{SETUP}`"
        ) from e


def fetch() -> None:
    """The setup step: the pinned code into the environment, then the model snapshot."""
    import urllib.request

    from huggingface_hub import snapshot_download

    for name, (repo, commit, package, digest) in CODE.items():
        root = code_root(name)
        if os.path.isdir(root):
            drop_bytecode(root)
        if os.path.isdir(root) and _digest_or_none(os.path.join(root, package)) == digest:
            print(f"{name}@{commit[:12]} already in place", file=sys.stderr, flush=True)
            continue
        url = ARCHIVE_URL.format(repo=repo, commit=commit)
        print(f"fetching {url}", file=sys.stderr, flush=True)
        with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 - a fixed https URL
            data = r.read(MAX_ARCHIVE + 1)
        if len(data) > MAX_ARCHIVE:
            raise RuntimeError(f"{url} is larger than {MAX_ARCHIVE} bytes")
        parent = os.path.dirname(root)
        os.makedirs(parent, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix=f".{name}-", dir=parent)
        try:
            extract_package(data, package, tmp)
            drop_bytecode(tmp)
            found = tree_digest(os.path.join(tmp, package))
            if found != digest:
                raise RuntimeError(
                    f"{repo}@{commit[:12]}'s {package}/ hashes to {found}, not the pinned {digest}"
                )
            shutil.rmtree(root, ignore_errors=True)
            os.replace(tmp, root)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"fetching {REPO}@{REVISION[:12]}", file=sys.stderr, flush=True)
    snapshot_download(REPO, revision=REVISION, allow_patterns=MODEL_FILES)


def _digest_or_none(path: str) -> str | None:
    try:
        return tree_digest(path) if os.path.isdir(path) else None
    except (OSError, RuntimeError):
        return None


def check_code() -> list[str]:
    """The pinned code's roots for ``sys.path``, after checking each one's digest."""
    roots = []
    for name, (repo, commit, package, digest) in CODE.items():
        root = code_root(name)
        found = _digest_or_none(os.path.join(root, package))
        if found != digest:
            state = "is missing" if found is None else f"hashes to {found}"
            raise RuntimeError(
                f"{repo}@{commit[:12]} in {root} {state}, not the pinned {digest}; set the "
                f"worker up again with `{SETUP}`"
            )
        roots.append(root)
    return roots


class Impl:
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.cv: Any = None
        self.mode: str | None = None
        self._speakers: dict[tuple[str, ...], str] = {}
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def load(self, options: dict[str, Any], device: str) -> None:
        self.mode = str(options["mode"]) if options.get("mode") else None  # the fallback
        if self.mode is not None:
            resolve_mode(self.mode, [], "en")  # rejects an unknown mode before the model loads
        roots = check_code()
        model_dir = _snapshot()
        sys.dont_write_bytecode = True  # keep the checked code tree as it was fetched
        for root in reversed(roots):
            if root not in sys.path:
                sys.path.insert(0, root)
        from cosyvoice.cli.cosyvoice import CosyVoice3  # pyright: ignore[reportMissingImports]

        self.cv = CosyVoice3(
            model_dir,
            load_trt=False,
            load_vllm=False,
            fp16=bool(options.get("fp16", False)) and device != "cpu",
        )
        if self.cv.frontend.text_frontend != "":
            # wetext and ttsfrd are not in the worker's lock; one that appears would change
            # what is spoken (and wetext downloads unpinned grammars)
            raise RuntimeError(
                f"CosyVoice 3 found the text frontend {self.cv.frontend.text_frontend!r}, which "
                f"this worker doesn't pin; set it up again with `{SETUP}`"
            )
        if int(self.cv.sample_rate) != SAMPLE_RATE:
            raise RuntimeError(f"CosyVoice 3 runs at {self.cv.sample_rate} Hz, not {SAMPLE_RATE}")
        self._tmp = tempfile.TemporaryDirectory(prefix="fukikae-cosyvoice3-")

    def unload(self) -> None:
        self.cv = None
        self._speakers.clear()
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _speaker(self, refs: list[dict[str, Any]]) -> str:
        """The cached prompt id of the card's reference clips (computed once per clip list)."""
        key = tuple(f"{r['clip']}\0{r.get('transcript') or ''}" for r in refs)
        if key in self._speakers:
            return self._speakers[key]
        import librosa
        import numpy as np

        timed = []
        for r in refs:
            seconds = float(librosa.get_duration(path=r["clip"]))
            timed.append((r["clip"], seconds, r.get("transcript") or ""))
        clips, transcript = prompt_plan(timed)
        parts: list[Any] = []
        gap = np.zeros(int(PROMPT_GAP_S * SAMPLE_RATE), np.float32)
        for clip in clips:
            y, _ = librosa.load(clip, sr=SAMPLE_RATE, mono=True)
            if parts:
                parts.append(gap)
            parts.append(y.astype(np.float32))
        audio = np.concatenate(parts)[: int(PROMPT_MAX_S * SAMPLE_RATE)]
        assert self._tmp is not None
        spk = f"card{len(self._speakers)}"
        path = os.path.join(self._tmp.name, f"{spk}.wav")
        _write_wav(path, audio, SAMPLE_RATE)
        if not self.cv.add_zero_shot_spk(f"{SYSTEM}{END}{transcript}", path, spk):
            raise RuntimeError("CosyVoice 3 refused the reference prompt")
        self._speakers[key] = spk
        return spk

    def synthesize(self, req: dict[str, Any]) -> float:
        if self.cv is None:
            raise RuntimeError("cosyvoice3: load() before synthesize()")
        language = req["language"]
        if language not in LANGUAGES:
            raise ValueError(f"CosyVoice 3 speaks {', '.join(LANGUAGES)}, not {language!r}")
        import numpy as np
        import torch

        voice = req["voice"]
        refs = list(voice.get("references", []))
        if not refs:
            raise ValueError("CosyVoice 3 needs a clone card with reference clips")
        if req.get("instruction") or req.get("emotion_audio") or req.get("emotion_vector"):
            raise ValueError("CosyVoice 3 takes no instruction, emotion audio or emotion vector")
        mode = resolve_mode(
            card_mode(voice.get("cosyvoice3_mode"), self.mode),
            [r.get("language") or "" for r in refs],
            language,
        )
        if mode == "zero_shot" and not all((r.get("transcript") or "").strip() for r in refs):
            raise ValueError("zero_shot mode needs a transcript for every reference clip")
        spk = self._speaker(refs)
        speed = float(voice.get("rate") or 1.0)
        seed = int(voice.get("seed") or 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        fe = self.cv.frontend
        chunks = []
        for seg in fe.text_normalize(req["text"], split=True, text_frontend=True):
            if mode == "cross_lingual":
                mi = fe.frontend_cross_lingual(f"{SYSTEM}{END}{seg}", "", SAMPLE_RATE, spk)
            else:
                mi = fe.frontend_zero_shot(seg, "", "", SAMPLE_RATE, spk)
            for out in self.cv.model.tts(**mi, stream=False, speed=speed):
                chunks.append(out["tts_speech"].reshape(-1).float().cpu().numpy())
        if not chunks:
            raise RuntimeError("CosyVoice 3 returned no audio")
        audio = np.concatenate(chunks)
        _write_wav(req["out"], audio, SAMPLE_RATE)
        return len(audio) / SAMPLE_RATE


def _write_wav(path: str, audio: Any, rate: int) -> None:
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
