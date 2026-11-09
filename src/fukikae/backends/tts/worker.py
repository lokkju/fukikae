"""The TTS worker: one cloning backend in its own environment, behind a JSON-lines protocol.

Run as ``python -m fukikae.backends.tts.worker <name>`` by ``fukikae.backends.tts.remote`` with
the interpreter of ``<checkout>/.venvs/<extra>`` and ``PYTHONPATH=<checkout>/src``. This module
and the ``impl_<name>`` it loads run under that interpreter, which has none of fukikae's own
dependencies (it may even be another Python version), so they import only the standard library
and the model's own packages, and nothing else from fukikae.

Protocol (version ``PROTOCOL``): each request is one JSON object on one line of stdin, at most
``MAX_LINE`` bytes; each answer is one JSON object on one line of stdout, ``{"ok": true, ...}``
or ``{"ok": false, "error": "..."}``. Requests:

- ``{"op": "hello"}`` -> ``{"ok": true, "protocol": 1, "impl": <name>}``
- ``{"op": "load", "options": {...}, "device": "cuda:0" | "cpu"}`` -> ``{"ok": true}``
- ``{"op": "synthesize", "text", "voice", "language", "out", "emotion_audio",
  "emotion_vector", "instruction", "emotion_strength", "target_seconds"}`` ->
  ``{"ok": true, "seconds": <float>}``; the impl writes a mono wav to ``out``
- ``{"op": "unload"}`` -> ``{"ok": true}``

The worker exits when stdin closes. Every path in a request must be absolute and name an
existing file (``out``: an existing directory); a malformed request gets an error answer and the
worker keeps going. Library output on stdout (progress bars, prints) would corrupt the
protocol, so file descriptor 1 is pointed at stderr before anything is imported, and answers go
to a private duplicate of the original stdout.

``FUKIKAE_TTS_FAKE_IMPL`` (tests only) names a module to use instead of ``impl_<name>``.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import sys
import traceback
from typing import Any

PROTOCOL = 1
MAX_LINE = 1 << 20
MAX_TEXT = 4000
MAX_ERROR = 4000
_NAME = re.compile(r"[a-z][a-z0-9_]*")
_SYNTH_KEYS = frozenset(
    {
        "op",
        "text",
        "voice",
        "language",
        "out",
        "emotion_audio",
        "emotion_vector",
        "instruction",
        "emotion_strength",
        "target_seconds",
    }
)


class RequestError(ValueError):
    """A request that breaks the protocol; answered with an error, never acted on."""


def _file(value: Any, key: str) -> str:
    if not isinstance(value, str) or not os.path.isabs(value):
        raise RequestError(f"{key} must be an absolute path")
    if not os.path.isfile(value):
        raise RequestError(f"{key}: no such file {value}")
    return value


def _number(value: Any, key: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{key} must be a number")
    x = float(value)
    if not math.isfinite(x) or not lo <= x <= hi:
        raise RequestError(f"{key} must be within [{lo:g}, {hi:g}], got {value!r}")
    return x


def _voice(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise RequestError("voice must be an object")
    refs = raw.get("references", [])
    if not isinstance(refs, list):
        raise RequestError("voice.references must be a list")
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            raise RequestError(f"voice.references[{i}] must be an object")
        _file(r.get("clip"), f"voice.references[{i}].clip")
    if raw.get("latent") is not None:
        _file(raw["latent"], "voice.latent")
    return raw


def validate_request(req: Any) -> dict[str, Any]:
    """The request when it is well formed, else ``RequestError``."""
    if not isinstance(req, dict):
        raise RequestError("a request must be a JSON object")
    op = req.get("op")
    if op == "hello" or op == "unload":
        return req
    if op == "load":
        if not isinstance(req.get("options", {}), dict):
            raise RequestError("load: options must be an object")
        device = req.get("device")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(:\d+)?", device):
            raise RequestError("load: device must be cpu or cuda[:n]")
        return req
    if op != "synthesize":
        raise RequestError(f"unknown op {op!r}")
    unknown = set(req) - _SYNTH_KEYS
    if unknown:
        raise RequestError(f"synthesize: unknown keys {sorted(unknown)}")
    text = req.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise RequestError(f"synthesize: text must be 1 to {MAX_TEXT} characters")
    if not isinstance(req.get("language"), str):
        raise RequestError("synthesize: language must be a string")
    out = req.get("out")
    if (
        not isinstance(out, str)
        or not os.path.isabs(out)
        or not os.path.isdir(os.path.dirname(out))
    ):
        raise RequestError("synthesize: out must be an absolute path in an existing directory")
    _voice(req.get("voice"))
    if req.get("emotion_audio") is not None:
        _file(req["emotion_audio"], "emotion_audio")
    ev = req.get("emotion_vector")
    if ev is not None:
        if not isinstance(ev, dict) or not all(isinstance(k, str) for k in ev):
            raise RequestError("emotion_vector must map labels to numbers")
        for k, v in ev.items():
            _number(v, f"emotion_vector[{k}]", 0.0, 1.0)
    if req.get("instruction") is not None and (
        not isinstance(req["instruction"], str) or len(req["instruction"]) > MAX_TEXT
    ):
        raise RequestError("instruction must be a string")
    _number(req.get("emotion_strength", 1.0), "emotion_strength", 0.0, 1.0)
    if req.get("target_seconds") is not None:
        _number(req["target_seconds"], "target_seconds", 0.01, 600.0)
    return req


def _impl_module(name: str) -> str:
    fake = os.environ.get("FUKIKAE_TTS_FAKE_IMPL")
    return fake if fake else f"fukikae.backends.tts.impl_{name}"


def main(name: str) -> int:
    if not _NAME.fullmatch(name):
        print(f"worker: not a backend name: {name!r}", file=sys.stderr)
        return 2
    # Keep the real stdout for answers; everything else written to fd 1 goes to stderr.
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def answer(obj: dict[str, Any]) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        proto.flush()

    impl: Any = None
    module = importlib.import_module(_impl_module(name))
    stdin = sys.stdin.buffer
    while True:
        raw = stdin.readline(MAX_LINE + 1)
        if not raw:
            break
        if len(raw) > MAX_LINE:
            answer({"ok": False, "error": f"request longer than {MAX_LINE} bytes"})
            # drop the rest of that line
            while raw and not raw.endswith(b"\n"):
                raw = stdin.readline(MAX_LINE + 1)
            continue
        if not raw.strip():
            continue
        try:
            req = validate_request(json.loads(raw))
            op = req["op"]
            if op == "hello":
                answer({"ok": True, "protocol": PROTOCOL, "impl": name})
            elif op == "load":
                impl = module.Impl()
                impl.load(dict(req.get("options", {})), req["device"])
                answer({"ok": True})
            elif op == "unload":
                if impl is not None:
                    impl.unload()
                    impl = None
                answer({"ok": True})
            else:
                if impl is None:
                    raise RequestError("synthesize before load")
                seconds = float(impl.synthesize(req))
                answer({"ok": True, "seconds": seconds})
        except (RequestError, json.JSONDecodeError, UnicodeDecodeError) as e:
            answer({"ok": False, "error": str(e)[:MAX_ERROR]})
        except Exception as e:  # noqa: BLE001 - reported to the client, which raises
            traceback.print_exc(file=sys.stderr)
            answer({"ok": False, "error": f"{type(e).__name__}: {e}"[:MAX_ERROR]})
    if impl is not None:
        impl.unload()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m fukikae.backends.tts.worker <name>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
