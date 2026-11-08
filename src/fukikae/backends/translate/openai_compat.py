"""Translation through any OpenAI-compatible chat endpoint.

``mode = "managed"`` (the default) downloads a GGUF from the Hugging Face hub and runs
llama-server on the configured GPU for as long as the backend is loaded. ``mode =
"external"`` talks to ``base_url`` (a gateway, or a server someone else runs); the API key,
when one is needed, is read from the environment variable ``api_key_env`` names, and only its
name ever appears in errors or logs.

Requests ask for JSON by schema (``response_format`` ``json_schema``, which llama-server
turns into a grammar) and turn Qwen3's thinking off.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from huggingface_hub import hf_hub_download

from fukikae.backends.translate.llama_server import LlamaServer
from fukikae.config import ConfigError
from fukikae.translation import (
    Rendering,
    TranslateFormatError,
    TranslateRequest,
    build_messages,
    parse_response,
    response_schema,
)

log = logging.getLogger(__name__)

OPTIONS = frozenset(
    {
        "mode", "model", "file", "revision", "binary", "ctx", "base_url", "api_key_env",
        "temperature", "timeout_s", "vram_mb", "gpu_index", "max_tokens",
    }
)  # fmt: skip
DEFAULT_VRAM_MB = 10000


class TranslateBackendError(RuntimeError):
    """The endpoint could not be reached or answered with an HTTP error."""


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
        self.temperature = float(options.get("temperature", 0.2))
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
        gguf = Path(hf_hub_download(self.model, self.file, revision=self.revision))
        server = LlamaServer(self.binary, gguf, self.gpu_index, ctx=self.ctx)
        server.__enter__()
        self._server = server
        self._base_url = server.base_url + "/v1"

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
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            raise TranslateBackendError(f"{self._base_url}: {type(e).__name__}: {e}") from None

    def translate(self, req: TranslateRequest) -> dict[str, Rendering]:
        body = {
            "model": self.model or "default",
            "messages": build_messages(req),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "cues", "schema": response_schema(req)},
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
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
