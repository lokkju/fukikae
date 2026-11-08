import pytest

from fukikae.config import BackendConfig, ConfigError, load_config


def test_language_override_wins(tmp_path):
    (tmp_path / "fukikae.toml").write_text(
        '[asr]\nbackend="faster_whisper"\nmodel="large-v3"\n[asr.lang.ko]\nbackend="qwen3_asr"\nmodel="Qwen/Qwen3-ASR-1.7B"\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.resolve("asr", "ko") == BackendConfig("qwen3_asr", {"model": "Qwen/Qwen3-ASR-1.7B"})
    assert c.resolve("asr", "fr") == BackendConfig("faster_whisper", {"model": "large-v3"})


def test_override_inherits_backend_and_options(tmp_path):
    (tmp_path / "fukikae.toml").write_text(
        '[translate]\nbackend="x"\nmodel="a"\nmax_speed=1.25\n[translate.pair.ko-en]\nmodel="b"\n'
    )
    c = load_config(tmp_path, user_config=None)
    # backend "x", not the default profile's openai_compat, so no profile option joins in
    assert c.resolve("translate", pair="ko-en") == BackendConfig(
        "x", {"model": "b", "max_speed": 1.25}
    )


def test_pair_beats_language(tmp_path):
    (tmp_path / "fukikae.toml").write_text(
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
    (ws / "fukikae.toml").write_text('target_language="en"\n[asr]\nmodel="large-v3"\n')
    c = load_config(ws, user_config=user)
    assert c.target_language == "en"
    assert c.gpu_device == 1
    assert c.resolve("asr") == BackendConfig("faster_whisper", {"model": "large-v3"})


def test_missing_backend_raises(tmp_path):
    (tmp_path / "fukikae.toml").write_text('[asr]\nbackend="x"\n')
    c = load_config(tmp_path, user_config=None)
    # a kind no shipped profile configures (the profiles name [translate] since Task 4)
    with pytest.raises(ConfigError, match="lid"):
        c.resolve("lid", "ko")


def test_defaults_and_source(tmp_path):
    c = load_config(tmp_path, user_config=None)
    assert c.target_language == "en"
    assert c.source_language is None
    assert c.source is None
    assert c.gpu_device == 0
    assert c.registry.min_new_seconds == 20.0
    (tmp_path / "fukikae.toml").write_text(
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
    (tmp_path / "fukikae.toml").write_text(
        '[asr]\nbackend="faster_whisper"\nmodel="large-v3"\ncompute_type="int8"\n'
        '[asr.lang.ko]\nbackend="qwen3_asr"\nmodel="Qwen/Qwen3-ASR-1.7B"\n'
        '[asr.lang.fr]\nmodel="large-v2"\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.resolve("asr", "ko") == BackendConfig("qwen3_asr", {"model": "Qwen/Qwen3-ASR-1.7B"})
    assert c.resolve("asr", "fr") == BackendConfig(
        "faster_whisper", {"model": "large-v2", "compute_type": "int8"}
    )


def test_series_defaults_to_dir_name_and_profile(tmp_path):
    ws = tmp_path / "bbt-de"
    ws.mkdir()
    (ws / "fukikae.toml").write_text('profile = "live-action"\n')
    s = load_config(ws, user_config=None).series
    assert (s.title, s.genre, s.ids) == ("bbt-de", "live-action", {})


def test_series_default_without_profile(tmp_path):
    ws = tmp_path / "grid"
    ws.mkdir()
    s = load_config(ws, user_config=None).series
    assert (s.title, s.genre) == ("grid", "default")


def test_explicit_genre_wins_over_profile(tmp_path):
    (tmp_path / "fukikae.toml").write_text(
        '[series]\ntitle = "The Big Bang Theory"\ngenre = "live-action"\n'
    )
    c = load_config(tmp_path, user_config=None)
    assert c.profile == "default"
    assert (c.series.title, c.series.genre) == ("The Big Bang Theory", "live-action")


def test_series_accepts_external_ids_and_rejects_unknown_keys(tmp_path):
    (tmp_path / "fukikae.toml").write_text('[series]\ntmdb = 1418\nanidb = "x1"\n')
    assert load_config(tmp_path, user_config=None).series.ids == {"tmdb": 1418, "anidb": "x1"}
    (tmp_path / "fukikae.toml").write_text('[series]\nimdb = "x"\n')
    with pytest.raises(ConfigError, match="imdb"):
        load_config(tmp_path, user_config=None)


def test_series_genre_must_be_lowercase_slug(tmp_path):
    (tmp_path / "fukikae.toml").write_text('[series]\ngenre = "Live Action"\n')
    with pytest.raises(ConfigError, match="genre"):
        load_config(tmp_path, user_config=None)


@pytest.mark.parametrize(
    "body",
    [
        "series = 3\n",
        "[series]\ntitle = 3\n",
        "[series]\ntmdb = true\n",
        "[series]\ntvdb = 1.5\n",
    ],
)
def test_series_rejects_wrong_types(tmp_path, body):
    (tmp_path / "fukikae.toml").write_text(body)
    with pytest.raises(ConfigError, match="series"):
        load_config(tmp_path, user_config=None)


def test_config_default_series():
    from fukikae.config import Config, SeriesInfo

    assert Config().series == SeriesInfo("", "default", {})


def test_older_top_level_series_string_is_the_title(tmp_path):
    (tmp_path / "fukikae.toml").write_text('profile = "anime"\nseries = "Fairy Tail"\n')
    s = load_config(tmp_path, user_config=None).series
    assert (s.title, s.genre, s.ids) == ("Fairy Tail", "anime", {})
