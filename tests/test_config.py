import pytest

from doppio.config import BackendConfig, ConfigError, load_config


def test_language_override_wins(tmp_path):
    (tmp_path / "doppio.toml").write_text(
        '[asr]\nbackend="faster_whisper"\nmodel="large-v3"\n[asr.lang.ko]\nbackend="qwen3_asr"\nmodel="Qwen/Qwen3-ASR-1.7B"\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.resolve("asr", "ko") == BackendConfig("qwen3_asr", {"model": "Qwen/Qwen3-ASR-1.7B"})
    assert c.resolve("asr", "fr") == BackendConfig("faster_whisper", {"model": "large-v3"})


def test_override_inherits_backend_and_options(tmp_path):
    (tmp_path / "doppio.toml").write_text(
        '[translate]\nbackend="openai_compat"\nmodel="a"\nmax_speed=1.25\n'
        '[translate.pair.ko-en]\nmodel="b"\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.resolve("translate", pair="ko-en") == BackendConfig(
        "openai_compat", {"model": "b", "max_speed": 1.25}
    )


def test_pair_beats_language(tmp_path):
    (tmp_path / "doppio.toml").write_text(
        '[translate]\nbackend="x"\nmodel="a"\n'
        '[translate.lang.ko]\nmodel="l"\n[translate.pair.ko-en]\nmodel="p"\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.resolve("translate", "ko", "ko-en").options["model"] == "p"
    assert c.resolve("translate", "ko", "ko-fr").options["model"] == "l"


def test_workspace_overrides_user_config(tmp_path):
    user = tmp_path / "user.toml"
    user.write_text(
        'gpu_device=1\ntarget_language="fr"\n[asr]\nbackend="faster_whisper"\nmodel="small"\n'
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "doppio.toml").write_text('target_language="en"\n[asr]\nmodel="large-v3"\n')
    c = load_config(ws, user_config=user)
    assert c.target_language == "en"
    assert c.gpu_device == 1
    assert c.resolve("asr") == BackendConfig("faster_whisper", {"model": "large-v3"})


def test_missing_backend_raises(tmp_path):
    (tmp_path / "doppio.toml").write_text('[asr]\nbackend="x"\n')
    c = load_config(tmp_path, user_config=None)
    with pytest.raises(ConfigError, match="diarize"):
        c.resolve("diarize", "ko")


def test_defaults_and_source(tmp_path):
    c = load_config(tmp_path, user_config=None)
    assert c.target_language == "en"
    assert c.source_language is None
    assert c.source is None
    assert c.gpu_device == 0
    assert c.registry.min_new_seconds == 20.0
    (tmp_path / "doppio.toml").write_text(
        'source="media/Show"\nsource_language="ko"\n[registry]\nmatch_threshold=0.7\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.source == tmp_path / "media/Show"
    assert c.source_language == "ko"
    assert c.registry.match_threshold == 0.7
    assert c.registry.min_new_seconds == 20.0


def test_default_user_config_missing_is_fine(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    assert load_config(tmp_path).target_language == "en"


def test_override_with_other_backend_drops_base_options(tmp_path):
    (tmp_path / "doppio.toml").write_text(
        '[asr]\nbackend="faster_whisper"\nmodel="large-v3"\ncompute_type="int8"\n'
        '[asr.lang.ko]\nbackend="qwen3_asr"\nmodel="Qwen/Qwen3-ASR-1.7B"\n'
        '[asr.lang.fr]\nmodel="large-v2"\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.resolve("asr", "ko") == BackendConfig("qwen3_asr", {"model": "Qwen/Qwen3-ASR-1.7B"})
    assert c.resolve("asr", "fr") == BackendConfig(
        "faster_whisper", {"model": "large-v2", "compute_type": "int8"}
    )
