"""Ruling CD24: third-party telemetry is off process-wide, and model loads stay local."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fukikae import privacy

TELEMETRY = {
    "PYANNOTE_METRICS_ENABLED": "0",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "ORT_DISABLE_TELEMETRY": "1",
}


def test_the_telemetry_switches():
    assert privacy.TELEMETRY_OFF == TELEMETRY


def test_importing_fukikae_turns_telemetry_off_before_anything_else():
    env = {k: v for k, v in os.environ.items() if k not in TELEMETRY}
    env.update({k: "1" if v == "0" else "0" for k, v in TELEMETRY.items()})  # opposite values
    keys = sorted(TELEMETRY)
    code = (
        "import json, os, sys; import fukikae; "
        f"print(json.dumps({{'env': {{k: os.environ.get(k) for k in {keys!r}}}, "
        "'early': [m for m in ('pyannote', 'onnxruntime', 'huggingface_hub') "
        "if m in sys.modules]}))"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True,
                         text=True, check=True).stdout  # fmt: skip
    got = json.loads(out)
    assert got["env"] == TELEMETRY and got["early"] == []


def test_the_tts_worker_env_carries_the_switches(monkeypatch: pytest.MonkeyPatch):
    from tests.backends.test_tts_worker import FakeWorker

    for k in TELEMETRY:
        monkeypatch.delenv(k, raising=False)
    w = FakeWorker({}, "cpu")
    for online in (False, True):
        env = w.worker_env(online=online)
        assert {k: env.get(k) for k in TELEMETRY} == TELEMETRY


_LOADS = r"""
import socket, sys
def hook(event, args):
    if event == "socket.connect":
        fam = getattr(args[0], "family", None)
        if fam in (socket.AF_INET, socket.AF_INET6):
            raise SystemExit(f"python-level connect {args[1:]}")
    if event == "socket.getaddrinfo":
        raise SystemExit(f"python-level lookup {args[:2]}")
sys.addaudithook(hook)
from fukikae.backends import loader
from fukikae.config import BackendConfig
from fukikae.cards.verify import FINGERPRINT_EMBEDDER, local_checkpoint
pinned, ckpt = local_checkpoint(FINGERPRINT_EMBEDDER)
emb = loader.create("embed", BackendConfig(pinned.backend, {"model": str(ckpt)}), "cpu")
emb.load()
tts = loader.create("tts", BackendConfig("kokoro", {}), "cpu")
tts.load()
import time
time.sleep(3)  # native beacons run on their own threads after the load
print("loaded")
"""


def _external_connects(trace: str) -> list[str]:
    return [
        ln for ln in trace.splitlines()
        if "connect(" in ln and re.search(r"AF_INET6?\b", ln)
    ]  # fmt: skip


@pytest.mark.slow
def test_no_outbound_connection_loading_the_embedder_and_a_stage_model(tmp_path: Path):
    """(a) the fingerprint embedder and (b) Kokoro, both offline from the local cache: no
    Python-level socket (audit hook) and, under strace, no native connect either (the
    onnxruntime beacon pyannote's import pulls in), not even a DNS lookup."""
    pytest.importorskip("pyannote.audio")
    pytest.importorskip("kokoro")
    script = tmp_path / "loads.py"
    script.write_text(_LOADS)
    cmd = [sys.executable, "-I", str(script)]
    strace = shutil.which("strace")
    if strace and subprocess.run([strace, "-qq", "-o", os.devnull, "true"], check=False,
                                 capture_output=True).returncode != 0:  # fmt: skip
        strace = None  # ptrace not allowed here: the audit hook alone
    trace = tmp_path / "trace"
    if strace:
        cmd = [strace, "-f", "-qq", "-e", "trace=connect", "-o", str(trace), *cmd]
    env = {k: v for k, v in os.environ.items() if k not in TELEMETRY and k != "HF_HUB_OFFLINE"}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600, check=False)
    assert r.returncode == 0 and "loaded" in r.stdout, r.stdout + r.stderr[-3000:]
    if strace:
        assert _external_connects(trace.read_text()) == []
