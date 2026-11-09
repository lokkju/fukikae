"""Third-party telemetry off, process-wide (ruling CD24).

``fukikae/__init__`` calls ``disable_telemetry`` before any submodule loads, so the switches
are in the environment before pyannote, huggingface_hub or onnxruntime is imported (several
read them once, at import). The TTS workers get the same variables (``backends.tts.remote``).

- ``PYANNOTE_METRICS_ENABLED=0``: pyannote.audio 4 sends an OpenTelemetry span to
  otel.pyannote.ai from ``track_model_init`` on every model load otherwise.
- ``HF_HUB_DISABLE_TELEMETRY=1`` and ``DO_NOT_TRACK=1``: huggingface_hub's (and others')
  usage pings.
- ``ORT_DISABLE_TELEMETRY=1``: onnxruntime 1.31 on Linux carries Microsoft's 1DS client and
  opens a native TLS connection to mobile.events.data.microsoft.com (Azure addresses) when
  pyannote's import pulls it in. ``onnxruntime.disable_telemetry_events()`` does not stop that
  connection (measured with strace on 1.31.1: one connection with the call, none with this
  variable); it is still called once onnxruntime is loaded, for builds where it does.
"""

from __future__ import annotations

import os
import sys

TELEMETRY_OFF: dict[str, str] = {
    "PYANNOTE_METRICS_ENABLED": "0",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "ORT_DISABLE_TELEMETRY": "1",
}


def disable_telemetry() -> None:
    """Set the switches (overriding any value: fukikae sends nothing anywhere)."""
    os.environ.update(TELEMETRY_OFF)
    quiet_onnxruntime()


def quiet_onnxruntime() -> None:
    """``onnxruntime.disable_telemetry_events()`` when onnxruntime is already imported (never
    imports it)."""
    ort = sys.modules.get("onnxruntime")
    off = getattr(ort, "disable_telemetry_events", None)
    if callable(off):
        off()
