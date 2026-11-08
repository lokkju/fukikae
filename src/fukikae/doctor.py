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

    for kind in KINDS:
        found = names(kind)
        rows.append((f"backends: {kind}", bool(found), ", ".join(found) if found else "(none)"))
    return rows
