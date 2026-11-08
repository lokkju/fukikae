"""A stand-in for llama-server: stdlib HTTP on /health and /v1/chat/completions.

Takes llama-server's arguments (only ``--host`` and ``--port`` matter). Behaviour comes from
the environment, since LlamaServer passes a fixed argument list:

- ``FAKE_LLAMA_MODE``: ``ok`` (default), ``hang`` (runs but /health never answers 200) or
  ``exit`` (writes to stderr and exits 3 at once).
- ``FAKE_LLAMA_ENV_OUT``: write ``CUDA_VISIBLE_DEVICES`` and the argv there as JSON.
- ``FAKE_LLAMA_RECORD``: append each chat request (headers and body) there as a JSON line.
- ``FAKE_LLAMA_REPLY``: a file whose text is the assistant message content to return; by
  default a valid rendering for every cue id in the request's schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODE = os.environ.get("FAKE_LLAMA_MODE", "ok")


def _default_content(body: dict) -> str:
    schema = body.get("response_format", {}).get("json_schema", {}).get("schema", {})
    out = {}
    for cid, sub in schema.get("properties", {}).items():
        out[cid] = {"text": f"line {cid}"}
        if "literal" in sub.get("properties", {}):
            out[cid]["literal"] = f"literal {cid}"
        if "direction" in sub.get("properties", {}):
            out[cid]["direction"] = None
    return json.dumps(out)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler's signature
        pass

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path == "/health" and MODE != "hang":
            self._send(200, {"status": "ok"})
        else:
            self._send(503, {"status": "loading model"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        record = os.environ.get("FAKE_LLAMA_RECORD")
        if record:
            with open(record, "a", encoding="utf-8") as f:
                f.write(json.dumps({"headers": dict(self.headers), "body": body}) + "\n")
        reply = os.environ.get("FAKE_LLAMA_REPLY")
        if reply:
            with open(reply, encoding="utf-8") as f:
                content = f.read()
        else:
            content = _default_content(body)
        self._send(
            200,
            {
                "choices": [
                    {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                ]
            },
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-m")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("-c")
    p.add_argument("-ngl")
    p.add_argument("--jinja", action="store_true")
    args = p.parse_args()
    env_out = os.environ.get("FAKE_LLAMA_ENV_OUT")
    if env_out:
        with open(env_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "argv": sys.argv[1:],
                },
                f,
            )
    if MODE == "exit":
        print("error: failed to load model 'nope.gguf'", file=sys.stderr, flush=True)
        sys.exit(3)
    print(f"fake llama-server pid {os.getpid()}", file=sys.stderr, flush=True)
    time.sleep(0.2)  # a real server takes a while before /health answers
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
