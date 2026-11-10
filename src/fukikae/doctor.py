"""Environment checks for `fukikae doctor`."""

import shutil
import subprocess


def _llama_server() -> tuple[str, bool, str]:
    """The llama-server on PATH (mise pins it) and its build, for the translate stage."""
    found = shutil.which("llama-server")
    if found is None:
        return ("llama-server", False, "not found on PATH (`mise install` in the checkout)")
    try:
        r = subprocess.run(  # noqa: S603 - the binary found on PATH, fixed argument
            [found, "--version"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return ("llama-server", False, f"{found}: --version failed ({type(e).__name__})")
    text = (r.stdout + r.stderr).splitlines()
    version = next((ln.strip() for ln in text if ln.startswith("version:")), "version unknown")
    return ("llama-server", True, f"{found} ({version})")


def _tts_workers() -> list[tuple[str, bool, str]]:
    """Each out-of-process TTS backend's worker environment (``.venvs/<extra>``)."""
    from fukikae.backends.loader import available
    from fukikae.backends.tts.remote import WorkerSynthesizer

    rows: list[tuple[str, bool, str]] = []
    for name, cls in sorted(available("tts").items()):
        if not (isinstance(cls, type) and issubclass(cls, WorkerSynthesizer)):
            continue
        inst = cls({}, "cpu")
        if inst.environment_location() != inst.venv():
            # a worker outside a venv (indextts2_trt's container) reports its own problem
            problem = inst.environment_problem()
            detail = f"{problem}: `{inst.setup_hint()}`" if problem else inst.environment_location()
            rows.append((f"tts worker: {name}", problem is None, str(detail)))
        elif not inst.interpreter().exists():
            detail = f"{inst.venv()} not set up: `{inst.setup_hint()}`"
            rows.append((f"tts worker: {name}", False, detail))
        elif problem := inst.lock_problem():
            rows.append((f"tts worker: {name}", False, f"{problem} (lock): `{inst.setup_hint()}`"))
        else:
            rows.append((f"tts worker: {name}", True, str(inst.venv())))
    return rows


def doctor() -> list[tuple[str, bool, str]]:
    """Return (label, ok, detail) lines. Never prints or returns the HF token."""
    from fukikae.backends.base import KINDS
    from fukikae.backends.loader import names

    rows: list[tuple[str, bool, str]] = []
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        rows.append((tool, found is not None, found or "not found on PATH"))

    try:
        from fukikae.gpu import free_mb

        rows.append(("nvml", True, f"GPU 0: {free_mb(0)} MiB free"))
    except Exception as e:
        rows.append(("nvml", False, f"no NVIDIA device visible ({type(e).__name__})"))

    try:
        from huggingface_hub import get_token

        has = get_token() is not None
    except ImportError:
        has = False
    rows.append(("huggingface token", has, "set" if has else "not set"))
    rows.append(_llama_server())
    rows += _tts_workers()

    for kind in KINDS:
        found = names(kind)
        rows.append((f"backends: {kind}", bool(found), ", ".join(found) if found else "(none)"))
    return rows
