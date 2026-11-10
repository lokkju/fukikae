import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fukikae.backends.translate.llama_server import LlamaServer, LlamaServerError

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "fake_llama_server.py"


@pytest.fixture
def fake(tmp_path: Path) -> str:
    """An executable that runs the fake server with this interpreter."""
    wrapper = tmp_path / "llama-server"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FIXTURE}" "$@"\n')
    wrapper.chmod(0o755)
    return str(wrapper)


@pytest.fixture
def gguf(tmp_path: Path) -> Path:
    p = tmp_path / "model.gguf"
    p.write_bytes(b"GGUF")
    return p


def _alive(pid: int | None) -> bool:
    assert pid is not None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_server_starts_and_stops(fake, gguf):
    with LlamaServer(fake, gguf, 0) as s:
        pid = s.pid
        assert s.base_url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(s.base_url + "/health", timeout=5) as r:  # noqa: S310
            assert r.status == 200
    assert not _alive(pid)  # terminated and reaped, so not even a zombie remains


def test_server_killed_when_body_raises(fake, gguf):
    pid: int | None = None
    with pytest.raises(RuntimeError, match="boom"), LlamaServer(fake, gguf, 0) as s:
        pid = s.pid
        raise RuntimeError("boom")
    assert not _alive(pid)


def test_server_that_never_gets_healthy_times_out_and_is_killed(fake, gguf, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "hang")
    server = LlamaServer(fake, gguf, 0, startup_timeout_s=2.0)
    with pytest.raises(LlamaServerError, match="not healthy after 2"), server:
        pass
    assert server.pid is not None and not _alive(server.pid)


def test_server_that_exits_reports_its_stderr(fake, gguf, monkeypatch):
    monkeypatch.setenv("FAKE_LLAMA_MODE", "exit")
    with pytest.raises(LlamaServerError, match="failed to load model"), LlamaServer(fake, gguf, 0):
        pass


def test_gpu_index_reaches_cuda_visible_devices(fake, gguf, tmp_path, monkeypatch):
    out = tmp_path / "env.json"
    monkeypatch.setenv("FAKE_LLAMA_ENV_OUT", str(out))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with LlamaServer(fake, gguf, 1, ctx=4096, port=None) as s:
        port = s.base_url.rsplit(":", 1)[1]
    seen = json.loads(out.read_text())
    assert seen["CUDA_VISIBLE_DEVICES"] == "1"
    assert seen["argv"][:-2] == [
        "-m", str(gguf), "--host", "127.0.0.1", "--port", port,
        "-c", "4096", "-ngl", "999", "--jinja",
    ]  # fmt: skip
    assert seen["argv"][-2] == "--api-key-file"  # a per-run key; never on the command line


def test_missing_binary_is_named(gguf):
    with (
        pytest.raises(LlamaServerError, match="no-such-llama"),
        LlamaServer("no-such-llama", gguf, 0),
    ):
        pass


def test_a_server_that_is_not_ours_is_refused(fake, gguf, monkeypatch):
    # /health answers, but /props names another model: not the process fukikae started
    monkeypatch.setenv("FAKE_LLAMA_PROPS_MODEL", "/elsewhere/other.gguf")
    server = LlamaServer(fake, gguf, 0)
    with pytest.raises(LlamaServerError, match="other.gguf"), server:
        pass
    assert not _alive(server.pid)


def test_api_key_is_required_by_our_server(fake, gguf):
    with LlamaServer(fake, gguf, 0) as s:
        assert s.api_key and len(s.api_key) >= 32
        req = urllib.request.Request(s.base_url + "/props")  # noqa: S310
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)  # noqa: S310
        assert e.value.code == 401


def test_port_taken_before_bind_is_retried_once(fake, gguf, tmp_path, monkeypatch):
    marker = tmp_path / "bound-once"
    monkeypatch.setenv("FAKE_LLAMA_MODE", "bindfail-once")
    monkeypatch.setenv("FAKE_LLAMA_MARKER", str(marker))
    with LlamaServer(fake, gguf, 0) as s:
        assert s.base_url.startswith("http://127.0.0.1:")
    assert marker.exists()
