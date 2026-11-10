"""IndexTTS2 through upstream's TensorRT backend, inside the TTS worker (D28 lever 2, a trial).

Runs in the ``indextts2_trt`` container (``workers/indextts2_trt/Dockerfile``: Python 3.12,
TensorRT-LLM 0.21, OpenMPI, upstream index-tts at ``CODE_REVISION`` on PYTHONPATH), so the
module level imports only the standard library and ``impl_indextts2`` (stdlib-only too), whose
pins it shares: the same IndexTTS-2 snapshot and auxiliary models, hence the same voice cards.

Upstream's ``backends/trt`` (index-tts PR #733, from MuyangDu/index-tts, arXiv 2607.21042) runs
every network on TensorRT: the GPT on TensorRT-LLM (beam search, 3 beams, with upstream's
sampler settings), and w2v-BERT, the semantic codec, CAMPPlus, the two perceiver conditioners,
the latent projector, the length regulator, the DiT of the diffusion decoder and BigVGAN as
plain TensorRT fp16 engines. ``build`` (the setup step, run in the container on the GPU) exports
the ONNX graphs, converts the GPT checkpoint and builds the engines for the GPU it runs on;
``Impl`` loads them.

What it honours is what ``impl_indextts2`` honours, the same way: the card's reference clips
joined into one prompt of at most 15 s (``impl_indextts2.Impl._prompt``), the source line's span
as the emotion prompt with ``emotion_strength`` as ``emo_alpha``, ``seed``, ``temperature`` and
``top_p``. The seed seeds torch (the diffusion decoder's noise) and TensorRT-LLM's sampler
(``random_seed``, which upstream's runtime leaves unset). The engines are fp16 by construction,
so the ``fp16`` and ``tf32`` options of the PyTorch worker don't apply.

Files: the engines directory is mounted at ``$FUKIKAE_TRT_ENGINES`` (read-only at run time),
and ``$HF_HUB_CACHE`` points at a shim cache inside it whose ``refs/main`` name the pinned
revisions, because upstream's exporters and its feature extractor ask the hub for ``main``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

from fukikae.backends.tts import impl_indextts2 as _base

REPO = _base.REPO
REVISION = _base.REVISION
CODE_REVISION = _base.CODE_REVISION
SAMPLE_RATE = _base.SAMPLE_RATE
LANGUAGES = _base.LANGUAGES
INDEX_TTS = "/opt/index-tts"
MANIFEST = "manifest.json"
MANIFEST_VERSION = 1
PRECISION = "fp16"
# Batch 1 is what upstream verified; the DiT runs conditional and unconditional together.
MAX_BATCH = 1
CONDITIONING_TOKENS = 34
SETUP = "fukikae backends setup indextts2_trt"

# (export script, output) and (build script, onnx, engine), as upstream's scripts list them.
EXPORTS = [
    "bigvgan", "dit", "speech_semantic_encoder", "semantic_codec",
    "speaker_perceiver_conditioner", "emotion_perceiver_conditioner", "latent_projector",
    "campplus", "length_regulator",
]  # fmt: skip


def fetch() -> None:
    """The setup step's download: the same pinned snapshots as the PyTorch worker."""
    _base.fetch()


def _hub_root() -> Path:
    """The real Hugging Face hub cache (mounted at its host path)."""
    explicit = os.environ.get("FUKIKAE_HUB_REAL")
    if explicit:
        return Path(explicit)
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def make_hub_shim(real: Path, shim: Path) -> None:
    """A hub cache at ``shim`` whose repos resolve ``main`` to the pinned revisions: for each
    pinned repo, ``refs/main`` holds the revision and ``snapshots/<rev>`` links (absolute) to
    the real snapshot, whose own links into ``blobs`` resolve from the real directory."""
    pins = [(_base.REPO, _base.REVISION), *((r, rev) for r, rev, _ in _base.AUX.values())]
    for repo, rev in pins:
        folder = "models--" + repo.replace("/", "--")
        target = real / folder / "snapshots" / rev
        if not target.is_dir():
            raise RuntimeError(f"{repo}@{rev[:12]} is not in {real}; run `{SETUP}`")
        root = shim / folder
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "snapshots").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text(rev)
        link = root / "snapshots" / rev
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def engine_paths(engines: Path) -> dict[str, str]:
    """Upstream's ``resolve_engine_paths`` layout under ``engines``."""
    trt = engines / f"trt_engines_{PRECISION}"
    out = {
        "gpt_engine_dir": str(engines / f"tllm_engines_{PRECISION}"),
        "speed_emb_path": str(engines / "onnx_models" / "speed_emb.pt"),
    }
    for name in EXPORTS:
        out[f"{name}_engine"] = str(trt / f"{name}.engine")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(args: list[str], cwd: Path, env: dict[str, str]) -> float:
    print("+ " + " ".join(args), file=sys.stderr, flush=True)
    started = time.monotonic()
    subprocess.run(args, cwd=cwd, env=env, check=True, stdout=sys.stderr)  # noqa: S603
    return time.monotonic() - started


def build(engines: Path) -> dict[str, Any]:
    """Build every engine into a fresh ``<engines>.tmp-<id>`` beside ``engines`` and move it
    into place whole once its ``manifest.json`` is written: a failed or killed build never
    leaves a manifest beside partial engines, and the engines already there stay as they were
    until the new set is complete. The old set is renamed aside, the new one renamed in and
    the old one deleted (two renames, so for an instant there are no engines at all, which
    ``environment_problem`` reports as not set up). A killed build's ``.tmp-``/``.old-``
    directories are removed first: builds run under the GPU lock, one at a time."""
    import secrets
    import shutil

    engines = Path(engines)
    engines.parent.mkdir(parents=True, exist_ok=True)
    for stale in [*engines.parent.glob(f"{engines.name}.tmp-*"),
                  *engines.parent.glob(f"{engines.name}.old-*")]:  # fmt: skip
        shutil.rmtree(stale, ignore_errors=True)
    tag = secrets.token_hex(4)
    tmp = engines.parent / f"{engines.name}.tmp-{tag}"
    tmp.mkdir()
    try:
        manifest = _build_into(tmp)
        (tmp / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        old = None
        if engines.exists() or engines.is_symlink():
            old = engines.parent / f"{engines.name}.old-{tag}"
            os.rename(engines, old)
        os.rename(tmp, engines)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)
    return manifest


def _build_into(engines: Path) -> dict[str, Any]:
    """Export, convert and build every engine into ``engines`` (empty) on this process's GPU
    and return the manifest (timings, sizes, sha256 of each engine and the build id, the first
    12 hex of the sha256 over them). Intermediate ONNX graphs, the converted GPT checkpoint
    and the work directory are deleted once the engines exist; their sizes are recorded."""
    import shutil

    engines.mkdir(parents=True, exist_ok=True)
    work = engines / "work"
    work.mkdir(exist_ok=True)
    shim = engines / "hub"
    make_hub_shim(_hub_root(), shim)
    model_dir = shim / ("models--" + REPO.replace("/", "--")) / "snapshots" / REVISION
    ckpt = work / "checkpoints"
    if not ckpt.is_symlink():
        ckpt.symlink_to(model_dir)
    env = dict(os.environ, HF_HUB_CACHE=str(shim), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (INDEX_TTS, env.get("PYTHONPATH")) if p)
    trt_dir = Path(INDEX_TTS) / "backends" / "trt"
    onnx = engines / "onnx_models"
    onnx.mkdir(exist_ok=True)
    trt_out = engines / f"trt_engines_{PRECISION}"
    trt_out.mkdir(exist_ok=True)
    tllm_ckpt = work / f"tllm_checkpoint_{PRECISION}"
    tllm_out = engines / f"tllm_engines_{PRECISION}"
    py = sys.executable
    times: dict[str, float] = {}

    for name in EXPORTS:
        dest = onnx / f"{name}.onnx"
        if not dest.exists():
            script = trt_dir / "export" / f"export_{name}_onnx.py"
            times[f"export {name}"] = _run([py, str(script), "--output", str(dest)], work, env)
    speed = onnx / "speed_emb.pt"
    if not speed.exists():
        script = trt_dir / "export" / "export_speed_emb.py"
        times["export speed_emb"] = _run(
            [py, str(script), "--model_dir", "checkpoints", "--output", str(speed)], work, env
        )
    if not (tllm_ckpt / "config.json").exists():
        times["convert gpt"] = _run(
            [py, "-m", "backends.trt.trtllm.convert_gpt_checkpoint", "--model_dir",
             "checkpoints", "--output_dir", str(tllm_ckpt), "--dtype", "float16"],
            work, env,
        )  # fmt: skip
    for name in EXPORTS:
        dest = trt_out / f"{name}.engine"
        if dest.exists():
            continue
        batch = MAX_BATCH * 2 if name == "dit" else MAX_BATCH
        script = trt_dir / "build_engines" / f"build_{name}_trt.py"
        times[f"build {name}"] = _run(
            [py, str(script), "--onnx", str(onnx / f"{name}.onnx"), "--output", str(dest),
             "--max-batch", str(batch)],
            work, env,
        )  # fmt: skip
    if not list(tllm_out.glob("*.engine")):
        times["build gpt"] = _run(
            [py, "-m", "backends.trt.trtllm.build_gpt_engine", "--checkpoint_dir",
             str(tllm_ckpt), "--max_batch_size", str(MAX_BATCH), "--max_beam_width", "3",
             "--output_dir", str(tllm_out), "--gather_all_token_logits",
             "--max_prompt_embedding_table_size", str(MAX_BATCH * CONDITIONING_TOKENS)],
            work, env,
        )  # fmt: skip

    intermediate = {
        "onnx_bytes": sum(p.stat().st_size for p in onnx.rglob("*") if p.is_file())
        - speed.stat().st_size,
        "tllm_checkpoint_bytes": sum(p.stat().st_size for p in tllm_ckpt.rglob("*") if p.is_file()),
    }
    files = sorted([*trt_out.glob("*.engine"), *tllm_out.glob("*"), speed], key=lambda p: str(p))
    digests = {str(p.relative_to(engines)): _sha256(p) for p in files if p.is_file()}
    sizes = {str(p.relative_to(engines)): p.stat().st_size for p in files if p.is_file()}
    build_id = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()[:12]
    import torch

    props = torch.cuda.get_device_properties(0)
    import tensorrt  # pyright: ignore[reportMissingImports]
    import tensorrt_llm  # pyright: ignore[reportMissingImports]

    manifest = {
        "version": MANIFEST_VERSION,
        "build_id": build_id,
        "precision": PRECISION,
        "max_batch": MAX_BATCH,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "tensorrt": tensorrt.__version__,
        "tensorrt_llm": tensorrt_llm.__version__,
        "model": f"{REPO}@{REVISION}",
        "code": CODE_REVISION,
        "times_s": times,
        "sizes": sizes,
        "intermediate": intermediate,
        "sha256": digests,
    }
    for p in onnx.glob("*.onnx*"):
        p.unlink()
    for p in onnx.glob("*.data"):
        p.unlink()
    shutil.rmtree(tllm_ckpt, ignore_errors=True)
    # its checkpoints link names this directory's path, which changes when it moves into place
    shutil.rmtree(work, ignore_errors=True)
    return manifest


def read_manifest(engines: Path) -> dict[str, Any]:
    path = engines / MANIFEST
    if not path.is_file():
        raise RuntimeError(f"no TensorRT engines at {engines}; build them with `{SETUP}`")
    return json.loads(path.read_text())


class Impl(_base.Impl):
    """The PyTorch impl's prompt handling, with upstream's TensorRT pipeline underneath."""

    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        super().__init__()
        self._speakers: dict[str, Any] = {}
        self._seed = 0

    def load(self, options: dict[str, Any], device: str) -> None:
        if options:
            raise ValueError(f"indextts2_trt takes no options, got {sorted(options)}")
        if device == "cpu":
            raise RuntimeError("indextts2_trt runs only on a GPU")
        engines = Path(os.environ["FUKIKAE_TRT_ENGINES"])
        manifest = read_manifest(engines)
        if manifest.get("code") != CODE_REVISION or manifest.get("model") != f"{REPO}@{REVISION}":
            raise RuntimeError(f"the engines at {engines} are for another pin; run `{SETUP}`")
        import tempfile

        import tensorrt_llm.runtime as trt_runtime  # pyright: ignore[reportMissingImports]

        # Upstream's runtime builds its SamplingConfig without a seed; give it the card's.
        if not getattr(trt_runtime.SamplingConfig, "_fukikae_seeded", False):
            original = trt_runtime.SamplingConfig
            impl = self

            def seeded(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("random_seed", impl._seed)
                return original(*args, **kwargs)

            seeded._fukikae_seeded = True  # type: ignore[attr-defined]
            trt_runtime.SamplingConfig = seeded  # type: ignore[misc]

        from backends.trt.pipeline import FasterIndexTTS2  # pyright: ignore[reportMissingImports]

        model_dir = _base._snapshot(REPO, REVISION, _base.CHECKPOINT_FILES)
        self.tts = FasterIndexTTS2(
            config_path=os.path.join(model_dir, "config.yaml"),
            model_dir=model_dir,
            device=device,
            **engine_paths(engines),
        )
        self._tmp = tempfile.TemporaryDirectory(prefix="fukikae-indextts2-trt-")

    def unload(self) -> None:
        self._speakers.clear()
        super().unload()

    def _speaker(self, path: str) -> Any:
        if path not in self._speakers:
            self._speakers[path] = self.tts.preload_speaker(path)
        return self._speakers[path]

    def synthesize(self, req: dict[str, Any]) -> float:
        if self.tts is None:
            raise RuntimeError("indextts2_trt: load() before synthesize()")
        if req["language"] not in LANGUAGES:
            raise ValueError(f"IndexTTS2 speaks {', '.join(LANGUAGES)}, not {req['language']!r}")
        import random

        import numpy as np
        import torch

        voice = req["voice"]
        clips = [r["clip"] for r in voice.get("references", [])]
        if not clips:
            raise ValueError("IndexTTS2 needs a clone card with reference clips")
        speaker = self._speaker(self._prompt(clips))
        seed = int(voice.get("seed") or 0)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        gen: dict[str, Any] = {}
        if voice.get("temperature") is not None:
            gen["temperature"] = float(voice["temperature"])
        if voice.get("top_p") is not None:
            gen["top_p"] = float(voice["top_p"])
        emo = req.get("emotion_audio")
        rate, data = self.tts.generate(
            text=req["text"],
            speaker=speaker,
            emo_speaker=self.tts.preload_speaker(emo) if emo else None,
            emo_alpha=float(req.get("emotion_strength", 1.0)) if emo else 1.0,
            **gen,
        )
        if int(rate) != SAMPLE_RATE:
            raise RuntimeError(f"IndexTTS2 TRT returned {rate} Hz, expected {SAMPLE_RATE}")
        pcm = np.asarray(data).reshape(-1).astype("<i2")
        with wave.open(req["out"], "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return len(pcm) / SAMPLE_RATE


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "build":
        manifest = build(Path(argv[1]))
        print(json.dumps({k: manifest[k] for k in ("build_id", "gpu", "times_s", "sizes")}))
        return 0
    print("usage: python -m fukikae.backends.tts.impl_indextts2_trt build <engines-dir>",
          file=sys.stderr)  # fmt: skip
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
