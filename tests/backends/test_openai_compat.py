import json
import logging
import sys
from pathlib import Path

import pytest

from fukikae.backends.translate import openai_compat
from fukikae.backends.translate.llama_server import LlamaServer
from fukikae.backends.translate.openai_compat import OpenAICompatTranslator
from fukikae.config import ConfigError
from fukikae.translation import RequestCue, TranslateRequest

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "fake_llama_server.py"
SECRET = "sk-test-very-secret-value-123"  # noqa: S105 - a test value


@pytest.fixture
def fake(tmp_path: Path) -> str:
    wrapper = tmp_path / "llama-server"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FIXTURE}" "$@"\n')
    wrapper.chmod(0o755)
    return str(wrapper)


def _req() -> TranslateRequest:
    return TranslateRequest(
        source_language="ja",
        target_language="en",
        mode="translate",
        cues=[
            RequestCue(id="c0001", speaker="Natsu", text="行くぞ", budget=20, marked=True),
            RequestCue(id="c0002", speaker="Lucy", text="うん", budget=8, marked=False),
        ],
        previous=[],
        styles={},
        glossary={},
    )


def test_options_are_checked():
    with pytest.raises(ConfigError, match="nonsense"):
        OpenAICompatTranslator({"nonsense": 1}, "cpu")
    with pytest.raises(ConfigError, match="mode"):
        OpenAICompatTranslator({"mode": "sideways"}, "cpu")
    with pytest.raises(ConfigError, match="base_url"):
        OpenAICompatTranslator({"mode": "external"}, "cpu").load()
    t = OpenAICompatTranslator({"model": "Qwen/Qwen3-8B-GGUF", "vram_mb": 7000}, "cuda:0")
    assert t.vram_mb == 7000 and t.model == "Qwen/Qwen3-8B-GGUF"


def test_external_mode_reads_key_from_env_and_never_logs_it(caplog, monkeypatch, fake, tmp_path):
    record = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(record))
    monkeypatch.setenv("MY_LLM_KEY", SECRET)
    caplog.set_level(logging.DEBUG)
    with LlamaServer(fake, tmp_path / "m.gguf", 0) as server:
        t = OpenAICompatTranslator(
            {"mode": "external", "base_url": server.base_url + "/v1", "api_key_env": "MY_LLM_KEY"},
            "cpu",
        )
        t.load()
        out = t.translate(_req())
        t.unload()
    assert out["c0001"].text == "line c0001"
    (line,) = record.read_text().splitlines()
    headers = json.loads(line)["headers"]
    assert headers["Authorization"] == f"Bearer {SECRET}"
    assert SECRET not in caplog.text


def test_missing_key_names_the_variable_not_a_value(monkeypatch):
    monkeypatch.delenv("MY_LLM_KEY", raising=False)
    t = OpenAICompatTranslator(
        {"mode": "external", "base_url": "http://127.0.0.1:9/v1", "api_key_env": "MY_LLM_KEY"},
        "cpu",
    )
    with pytest.raises(ConfigError, match="MY_LLM_KEY"):
        t.load()


def test_json_schema_sent_and_thinking_off(monkeypatch, fake, tmp_path):
    record = tmp_path / "requests.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(record))
    with LlamaServer(fake, tmp_path / "m.gguf", 0) as server:
        t = OpenAICompatTranslator(
            {"mode": "external", "base_url": server.base_url + "/v1", "temperature": 0.1}, "cpu"
        )
        t.load()
        t.translate(_req())
    body = json.loads(record.read_text().splitlines()[0])["body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]["schema"]
    assert set(schema["required"]) == {"c0001", "c0002"}
    assert body["temperature"] == 0.1
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "Authorization" not in json.loads(record.read_text().splitlines()[0])["headers"]


def test_managed_mode_downloads_and_runs_the_server(monkeypatch, fake, tmp_path):
    gguf = tmp_path / "Qwen3-8B-Q6_K.gguf"
    gguf.write_bytes(b"GGUF")
    calls = []

    def fake_download(repo, filename, revision=None):
        calls.append((repo, filename, revision))
        return str(gguf)

    monkeypatch.setattr(openai_compat, "hf_hub_download", fake_download)
    t = OpenAICompatTranslator(
        {
            "mode": "managed",
            "model": "Qwen/Qwen3-8B-GGUF",
            "file": "Qwen3-8B-Q6_K.gguf",
            "revision": "abc123",
            "binary": fake,
            "gpu_index": 0,
        },
        "cuda:0",
    )
    t.load()
    try:
        server = t._server
        assert server is not None
        pid = server.pid
        assert pid is not None
        assert t.translate(_req())["c0002"].text == "line c0002"
    finally:
        t.unload()
    assert calls == [("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q6_K.gguf", "abc123")]
    assert t._server is None
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
