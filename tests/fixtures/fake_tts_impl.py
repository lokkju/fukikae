"""A worker impl for tests (FUKIKAE_TTS_FAKE_IMPL=tests.fixtures.fake_tts_impl).

Stdlib only, like a real impl may be: it runs under the worker's interpreter. It prints to
stdout on load (as model libraries do), so the worker must keep stdout for the protocol.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time
import wave

RATE = 16000


class Impl:
    sample_rate = RATE

    def __init__(self) -> None:
        self.loaded = False
        self.options: dict = {}

    def load(self, options: dict, device: str) -> None:
        print("library chatter on stdout during load")  # must not reach the protocol
        sys.stdout.flush()
        self.options = dict(options, device=device)
        self.loaded = True

    def synthesize(self, req: dict) -> float:
        text = req["text"]
        if text == "boom":
            print("Traceback detail: model exploded", file=sys.stderr)
            raise RuntimeError("boom from impl")
        if text == "die":
            print("dying on purpose", file=sys.stderr, flush=True)
            os._exit(3)
        if text == "hang":
            time.sleep(60)
        if text == "wrong-rate":
            return _tone(req["out"], 0.5, 8000)
        if text == "lie":
            _tone(req["out"], 0.5, RATE)
            return 9.0
        if text == "echo":
            # the request as the impl saw it, for the test to read back
            import json

            keys = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE")  # fmt: skip
            env: dict = {k: os.environ.get(k) for k in keys}
            env["names"] = sorted(os.environ)
            with open(req["out"] + ".json", "w", encoding="utf-8") as f:
                json.dump({**req, "env": env, "device": self.options["device"]}, f)
        print("progress bar on stdout")
        return _tone(req["out"], 0.05 * len(text), RATE)

    def unload(self) -> None:
        self.loaded = False


def _tone(path: str, seconds: float, rate: int) -> float:
    n = max(1, int(seconds * rate))
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(frames)
    return n / rate


def fetch() -> None:
    """The setup step's download, for tests: records that it ran and under what env."""
    import json

    dest = os.path.join(os.environ["TMPDIR"], "fake-fetch.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"offline": os.environ.get("HF_HUB_OFFLINE")}, f)
