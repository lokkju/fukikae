"""Ruling CD24: in-process backends load pinned models from the local cache only; a cache
miss names `fukikae backends setup`, and only that step downloads."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fukikae.backends.loader import BackendUnavailable

runner = CliRunner()


class Hub:
    """snapshot_download and hf_hub_download over an empty cache: local lookups miss,
    anything else is a download, recorded."""

    def __init__(self) -> None:
        self.downloads: list[tuple] = []
        self.local: list[tuple] = []

    def snapshot_download(self, repo, revision=None, local_files_only=False, **_kw):
        from huggingface_hub.errors import LocalEntryNotFoundError

        if local_files_only:
            self.local.append((repo, revision))
            raise LocalEntryNotFoundError("not cached")
        self.downloads.append((repo, revision))
        return f"/cache/{repo}/{revision}"

    def hf_hub_download(self, repo, filename, revision=None, local_files_only=False, **_kw):
        from huggingface_hub.errors import LocalEntryNotFoundError

        if local_files_only:
            self.local.append((repo, filename, revision))
            raise LocalEntryNotFoundError("not cached")
        self.downloads.append((repo, filename, revision))
        return f"/cache/{repo}/{filename}"


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> Hub:
    import huggingface_hub

    h = Hub()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", h.snapshot_download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", h.hf_hub_download)
    fake = types.ModuleType("funasr")
    fake.AutoModel = lambda **kw: kw  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "funasr", fake)
    return h


def _backends() -> list[tuple[str, str, dict]]:
    return [
        ("tts", "kokoro", {}),
        ("tag", "sensevoice", {}),
        ("tag", "emotion2vec", {}),
        (
            "translate",
            "openai_compat",
            {
                "mode": "managed",
                "binary": "/bin/false",
                "model": "Qwen/Qwen3-8B-GGUF",
                "file": "q.gguf",
                "revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
            },
        ),
    ]


@pytest.mark.parametrize(("kind", "name", "options"), _backends())
def test_a_cache_miss_never_downloads_and_names_setup(hub: Hub, kind, name, options):
    from fukikae.backends import loader
    from fukikae.config import BackendConfig

    if name == "kokoro":
        pytest.importorskip("kokoro")
    inst = loader.create(kind, BackendConfig(name, options), "cpu")
    with pytest.raises(BackendUnavailable, match=f"fukikae backends setup {name}"):
        inst.load()
    assert hub.downloads == [] and hub.local


@pytest.mark.parametrize(("kind", "name", "options"), _backends())
def test_setup_fetches_the_pinned_files(hub: Hub, kind, name, options):
    from fukikae.backends import loader

    if name == "kokoro":
        pytest.importorskip("kokoro")
    cls = loader.backend_class(kind, name)
    cls.fetch(options)  # type: ignore[attr-defined]
    assert hub.downloads and all(d[-1] for d in hub.downloads)  # a pinned revision each


def test_backends_setup_runs_an_in_process_backends_fetch(monkeypatch: pytest.MonkeyPatch):
    from fukikae.backends import loader
    from fukikae.cli import app

    called: list[str] = []

    class Fetchy:
        name = "fetchy"

        @classmethod
        def fetch(cls, options: dict | None = None) -> list[Path]:  # noqa: ARG003
            called.append("fetch")
            return [Path("/cache/x")]

    monkeypatch.setattr(loader, "names", lambda kind: ["fetchy"] if kind == "tag" else [])
    monkeypatch.setattr(loader, "backend_class", lambda _kind, _name: Fetchy)
    r = runner.invoke(app, ["backends", "setup", "fetchy"])
    assert r.exit_code == 0, r.output
    assert called == ["fetch"] and "/cache/x" in r.output
