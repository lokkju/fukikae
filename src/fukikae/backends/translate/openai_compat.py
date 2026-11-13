"""Translation through any OpenAI-compatible chat endpoint.

``mode = "managed"`` (the default) downloads a GGUF from the Hugging Face hub and runs
llama-server on the configured GPU for as long as the backend is loaded. ``mode =
"external"`` talks to ``base_url`` (a gateway, or a server someone else runs); the API key,
when one is needed, is read from the environment variable ``api_key_env`` names, and only its
name ever appears in errors or logs.

Requests ask for JSON by schema (``response_format`` ``json_schema``, which llama-server
turns into a grammar) and turn Qwen3's thinking off. Sampling is greedy (``temperature`` 0)
with a fixed ``seed`` by default, so the same request gets the same answer wherever the server
honours them (#41); a strict retry (#38) is greedy whatever ``temperature`` says. A managed
llama-server is also asked not to reuse its prompt cache, which changes batch sizes and with
them the logits a greedy pick reads.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from fukikae.backends.local import fetch_file, local_file
from fukikae.backends.translate.llama_server import LlamaServer
from fukikae.config import ConfigError
from fukikae.translation import (
    PromptTooLargeError,
    Rendering,
    TranslateBackendError,
    TranslateFormatError,
    TranslateRequest,
    build_messages,
    estimate_tokens,
    parse_response,
    response_schema,
)

log = logging.getLogger(__name__)

OPTIONS = frozenset(
    {
        "mode", "model", "file", "revision", "binary", "ctx", "base_url", "api_key_env",
        "temperature", "seed", "timeout_s", "vram_mb", "gpu_index", "max_tokens",
    }
)  # fmt: skip
DEFAULT_VRAM_MB = 10000


class OpenAICompatTranslator:
    name: ClassVar[str] = "openai_compat"
    kind: ClassVar[str] = "translate"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cuda", "cpu"})
    vram_mb: int = DEFAULT_VRAM_MB

    def __init__(self, options: dict[str, Any], device: str) -> None:
        unknown = sorted(set(options) - OPTIONS)
        if unknown:
            raise ConfigError(
                f"[translate] openai_compat: unknown option(s) {', '.join(unknown)}; "
                f"allowed: {', '.join(sorted(OPTIONS))}"
            )
        self.mode = str(options.get("mode", "managed"))
        if self.mode not in ("managed", "external"):
            raise ConfigError(
                f'[translate] mode must be "managed" or "external", not {self.mode!r}'
            )
        self.model: str | None = options.get("model")
        self.file: str | None = options.get("file")
        self.revision: str | None = options.get("revision")
        self.binary = str(options.get("binary", "llama-server"))
        self.ctx = int(options.get("ctx", 8192))
        self.base_url_option: str | None = options.get("base_url")
        self.api_key_env: str | None = options.get("api_key_env")
        self.temperature = float(options.get("temperature", 0.0))
        self.seed = int(options.get("seed", 0))
        self.timeout_s = float(options.get("timeout_s", 300))
        self.max_tokens = int(options.get("max_tokens", 4096))
        self.vram_mb = int(options.get("vram_mb", DEFAULT_VRAM_MB))
        self.gpu_index = int(options.get("gpu_index", 0))
        self.device = device
        self._server: LlamaServer | None = None
        self._base_url: str | None = None
        self._key: str | None = None

    def load(self) -> None:
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ConfigError(
                    f"[translate] api_key_env names {self.api_key_env}, which is not set"
                )
            self._key = key
        if self.mode == "external":
            if not self.base_url_option:
                raise ConfigError('[translate] mode = "external" needs base_url')
            self._base_url = self.base_url_option.rstrip("/")
            return
        if not self.model or not self.file:
            raise ConfigError('[translate] mode = "managed" needs model (a hub repo) and file')
        # the pinned GGUF from the local cache only (ruling CD24)
        gguf = local_file("openai_compat", self.model, self.file, self.revision)
        server = LlamaServer(self.binary, gguf, self.gpu_index, ctx=self.ctx)
        server.__enter__()
        self._server = server
        self._base_url = server.base_url + "/v1"
        self._key = server.api_key

    @classmethod
    def fetch(cls, options: dict[str, Any] | None = None) -> list[Path]:
        """Download the managed mode's GGUF (``fukikae backends setup openai_compat``)."""
        inst = cls(options or {}, "cpu")
        if inst.mode != "managed" or not inst.model or not inst.file:
            return []
        return [fetch_file(inst.model, inst.file, inst.revision)]

    def unload(self) -> None:
        server, self._server = self._server, None
        self._base_url = None
        self._key = None
        if server is not None:
            server.__exit__(None, None, None)

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        assert self._base_url is not None, "load() first"
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(  # noqa: S310 - base_url is operator config
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:  # noqa: S310
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:500].decode("utf-8", errors="replace")
            raise TranslateBackendError(
                f"{self._base_url}: HTTP {e.code} from chat/completions: {detail}"
            ) from None
        except (OSError, ValueError) as e:  # URLError, a timeout, a reset, a bad body
            raise TranslateBackendError(f"{self._base_url}: {type(e).__name__}: {e}") from None

    def translate(self, req: TranslateRequest) -> dict[str, Rendering]:
        messages = build_messages(req)
        room = self.ctx - self.max_tokens
        need = estimate_tokens(messages)
        if need > room:
            raise PromptTooLargeError(
                f"prompt of about {need} tokens exceeds the {room} the context leaves "
                f"(ctx {self.ctx} - max_tokens {self.max_tokens})"
            )
        body: dict[str, Any] = {
            "model": self.model or "default",
            "messages": messages,
            "temperature": 0.0 if req.strict else self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "cues", "schema": response_schema(req)},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.mode == "managed":
            body["cache_prompt"] = False  # llama.cpp only; other servers may refuse the field
        log.debug("translate request: %d cue(s), mode %s", len(req.cues), req.mode)
        reply = self._post(body)
        try:
            choice = reply["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise TranslateFormatError(f"reply has no message content ({e!r})") from None
        if choice.get("finish_reason") == "length":
            raise TranslateFormatError("reply was cut off at max_tokens")
        if not isinstance(content, str):
            raise TranslateFormatError("reply content is not text")
        return parse_response(req, content)
