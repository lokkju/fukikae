"""Pinned model files from the local Hugging Face cache, never the network (ruling CD24).

At run time a backend loads its pinned snapshot with ``local_snapshot`` or ``local_file``:
``local_files_only``, so a cache miss raises ``BackendUnavailable`` naming
``fukikae backends setup <backend>`` instead of downloading. ``fetch_snapshot`` and
``fetch_file`` are the download, used only by a backend's ``fetch`` from that setup command.
"""

from __future__ import annotations

from pathlib import Path

from fukikae.backends.loader import BackendUnavailable


def _miss(backend: str, what: str) -> BackendUnavailable:
    return BackendUnavailable(
        f"{backend}: {what} is not in the local Hugging Face cache; run "
        f"`fukikae backends setup {backend}` to fetch it (fukikae downloads models only there)"
    )


def local_snapshot(
    backend: str, repo: str, revision: str, patterns: list[str] | None = None
) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(
            snapshot_download(
                repo, revision=revision, allow_patterns=patterns, local_files_only=True
            )
        )
    except LocalEntryNotFoundError as e:
        raise _miss(backend, f"{repo}@{revision[:12]}") from e


def local_file(backend: str, repo: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(hf_hub_download(repo, filename, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError as e:
        raise _miss(backend, f"{repo}/{filename}@{(revision or 'main')[:12]}") from e


def fetch_snapshot(repo: str, revision: str, patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def fetch_file(repo: str, filename: str, revision: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, revision=revision))
