"""Environment checks for `fukikae doctor`."""

import shutil


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

    for kind in KINDS:
        found = names(kind)
        rows.append((f"backends: {kind}", bool(found), ", ".join(found) if found else "(none)"))
    return rows
