import pytest

from fukikae import config as cfgmod
from fukikae.config import BackendConfig, ConfigError, load_config


def _ws(tmp_path, text):
    (tmp_path / "fukikae.toml").write_text(text)
    return load_config(tmp_path, user_config=None)


def _with_profile(monkeypatch, tables):
    real = cfgmod.load_profile

    def fake(name):
        return tables if name == "anime" else real(name)

    monkeypatch.setattr(cfgmod, "load_profile", fake)
    monkeypatch.setattr(cfgmod, "profile_names", lambda: ["anime", "default"])


def test_profile_layers_under_language_and_workspace(tmp_path, monkeypatch):
    _with_profile(monkeypatch, {"embed": {"backend": "x", "opt": 1}})
    c = _ws(tmp_path, 'profile="anime"\n[embed.lang.ko]\nbackend="y"\n')
    assert c.resolve("embed", "ja") == BackendConfig("x", {"opt": 1})
    assert c.resolve("embed", "ko") == BackendConfig("y", {})  # R4: x's options not inherited


def test_workspace_base_beats_profile(tmp_path, monkeypatch):
    tables = {"diarize": {"backend": "x", "opt": 1, "lang": {"ja": {"opt": 2}}}}
    _with_profile(monkeypatch, tables)
    c = _ws(tmp_path, 'profile="anime"\n[diarize]\nbackend="z"\n')
    assert c.resolve("diarize", "ja") == BackendConfig("z", {})
    c = _ws(tmp_path, 'profile="anime"\n[diarize]\nopt=9\n')
    assert c.resolve("diarize", "ja") == BackendConfig("x", {"opt": 9})


def test_unknown_profile_lists_available(tmp_path):
    with pytest.raises(ConfigError, match=r"nope.*anime.*default.*live-action"):
        _ws(tmp_path, 'profile="nope"\n')


def test_registry_threshold_from_profile_unless_workspace_sets_it(tmp_path, monkeypatch):
    _with_profile(monkeypatch, {"registry": {"match_threshold": 0.4, "min_new_seconds": 5}})
    c = _ws(tmp_path, 'profile="anime"\n')
    assert (c.registry.match_threshold, c.registry.min_new_seconds) == (0.4, 5.0)
    c = _ws(tmp_path, 'profile="anime"\n[registry]\nmatch_threshold=0.7\n')
    assert (c.registry.match_threshold, c.registry.min_new_seconds) == (0.7, 5.0)


def test_shipped_profiles_load():
    for name in ("default", "anime", "live-action"):
        assert isinstance(cfgmod.load_profile(name), dict)


def _old_resolve(raw, kind, language, pair):
    table = raw.get(kind)
    if not isinstance(table, dict):
        table = {}
    merged = {k: v for k, v in table.items() if k not in ("lang", "pair")}
    for group, key in (("lang", language), ("pair", pair)):
        if key is None:
            continue
        override = table.get(group, {}).get(key)
        if not isinstance(override, dict):
            continue
        if "backend" in override and override["backend"] != merged.get("backend"):
            merged = dict(override)
        else:
            merged.update(override)
    backend = merged.pop("backend", None)
    if not backend:
        return None
    options = {k: v for k, v in merged.items() if k not in ("lang", "pair")}
    return BackendConfig(str(backend), options)


TYPICAL = (
    '[asr]\nbackend="faster_whisper"\nmodel="large-v3"\n[asr.lang.ko]\nbackend="qwen3_asr"\n'
    '[separation]\nmodel="htdemucs"\n[translate]\nbackend="o"\nm=1\n[translate.pair.ko-en]\nm=2\n'
    "[registry]\nmatch_threshold=0.5\n"
)


@pytest.mark.parametrize("text", ["", TYPICAL])
def test_no_profile_resolves_as_before(tmp_path, text):
    c = _ws(tmp_path, text)
    # The shipped default profile now names embed, diarize and separation (Task 9), and
    # translate (character-cards Task 4) and tts (Task 6).
    kinds = ["asr", "align", "lid"]
    for kind in kinds:
        for lang, pair in ((None, None), ("ko", None), ("ja", "ja-en"), ("ko", "ko-en")):
            want = _old_resolve(c.raw, kind, lang, pair)
            if want is None:
                with pytest.raises(ConfigError):
                    c.resolve(kind, lang, pair)
            else:
                assert c.resolve(kind, lang, pair) == want
    assert c.profile == "default"


def test_extract_config_key_records_audio_title(tmp_path, monkeypatch):
    from fukikae import media
    from fukikae.stages.extract import ExtractStage

    probe = media.ProbeResult(
        audio=[
            media.AudioTrack(1, "aac", 2, "spa", "Spanish (Latin America)", False),
            media.AudioTrack(2, "aac", 2, "spa", "Spanish (Castilian)", False),
        ],
        subtitles=[],
    )
    monkeypatch.setattr(media, "probe", lambda _p: probe)
    ep = type("E", (), {"source": tmp_path / "x.mkv"})()
    base = _ws(tmp_path, 'source_language="es"\n')
    titled = _ws(tmp_path, 'source_language="es"\n[extract]\naudio_title="Castilian"\n')
    stage = ExtractStage()
    assert stage.config_key(base, ep)["track"] == 1  # type: ignore[arg-type]
    key = stage.config_key(titled, ep)  # type: ignore[arg-type]
    assert key["track"] == 2 and key["audio_title"] == "Castilian"


SHIPPED = ("default", "anime", "live-action")


@pytest.mark.parametrize("name", SHIPPED)
def test_every_profile_table_names_its_backend(name):
    for kind, table in cfgmod.load_profile(name).items():
        # [registry] holds plain numbers; [resolve] selects its matcher by `matcher`, as
        # [characters] does (Ruling R17), so neither names a `backend`.
        if kind == "registry":
            continue
        key = "matcher" if kind in ("characters", "resolve") else "backend"
        assert isinstance(table, dict) and table.get(key), f"{name}: [{kind}] lacks {key}"


@pytest.mark.parametrize("name", SHIPPED)
def test_shipped_profiles_resolve_to_the_tested_backends(tmp_path, name):
    c = _ws(tmp_path, f'profile="{name}"\n')
    assert c.resolve("embed").backend == "wespeaker"
    assert c.resolve("diarize").backend == "pyannote"
    assert c.resolve("separation").backend == "demucs"  # Ruling R15
    assert c.resolve("characters", key="matcher").backend == "centroid"
    assert c.registry.match_threshold == 0.6


def test_profile_workspace_override_still_wins(tmp_path):
    c = _ws(tmp_path, 'profile="anime"\n[embed]\nbackend="ecapa"\n')
    assert c.resolve("embed").backend == "ecapa"


def test_live_action_profile_is_flagged_provisional():
    text = (
        cfgmod.resources.files("fukikae.profiles").joinpath("live-action.toml").read_text("utf-8")
    )
    assert "PROVISIONAL" in text


def test_no_profile_workspace_with_separation_options_keeps_demucs(tmp_path):
    c = _ws(tmp_path, '[separation]\nmodel="htdemucs"\n')
    assert c.resolve("separation") == BackendConfig("demucs", {"model": "htdemucs"})


def _settings(tmp_path, text, **cli):
    from fukikae.registry.cli import resolve_settings

    c = _ws(tmp_path, text)
    return resolve_settings(
        c, cli.get("matcher"), cli.get("threshold"), cli.get("options"), cli.get("language")
    )


def test_resolve_settings_come_from_the_profile(tmp_path):
    assert _settings(tmp_path, 'profile="anime"\n') == ("joint", 0.6, {"episode_cannot_link": True})
    assert _settings(tmp_path, 'profile="live-action"\n') == (
        "joint",
        0.6,
        {"episode_cannot_link": False},
    )


def test_resolve_cli_flags_override_the_profile(tmp_path):
    got = _settings(
        tmp_path, 'profile="anime"\n', threshold=0.7, options={"episode_cannot_link": False}
    )
    assert got == ("joint", 0.7, {"episode_cannot_link": False})


def test_resolve_without_a_profile_keeps_todays_defaults(tmp_path):
    c = _ws(tmp_path, "")
    assert _settings(tmp_path, "") == ("joint", c.registry.match_threshold, {})
    assert c.registry.match_threshold == 0.6


def test_resolve_table_without_matcher_still_applies(tmp_path):
    """A workspace [resolve] with no matcher on the default profile: joint, and its own
    threshold and options, never dropped for the built-ins."""
    text = "[resolve]\nthreshold = 0.75\nepisode_cannot_link = false\n"
    assert _settings(tmp_path, text) == ("joint", 0.75, {"episode_cannot_link": False})


def test_profile_matcher_and_workspace_threshold_merge(tmp_path):
    text = 'profile="anime"\n[resolve]\nthreshold = 0.7\n'
    assert _settings(tmp_path, text) == ("joint", 0.7, {"episode_cannot_link": True})


def test_resolve_language_override_follows_the_given_language(tmp_path):
    text = 'profile="anime"\n[resolve.lang.ja]\nthreshold = 0.65\n'
    assert _settings(tmp_path, text, language="ja")[1] == 0.65
    assert _settings(tmp_path, text, language="en")[1] == 0.6


def test_workspace_embed_options_without_backend_fail_naming_the_backend(tmp_path):
    """`[embed] model = "speechbrain/..."` used to mean ECAPA; under the WeSpeaker default
    it must fail at resolution, not at model load."""
    c = _ws(tmp_path, '[embed]\nmodel = "speechbrain/spkrec-ecapa-voxceleb"\n')
    with pytest.raises(ConfigError, match=r"model.*default profile.*wespeaker.*Set `backend`"):
        c.resolve("embed")
    c = _ws(tmp_path, '[embed.lang.ja]\nmodel = "speechbrain/spkrec-ecapa-voxceleb"\n')
    assert c.resolve("embed", "en").backend == "wespeaker"
    with pytest.raises(ConfigError, match="backend"):
        c.resolve("embed", "ja")


def test_workspace_embed_options_with_backend_resolve(tmp_path):
    text = '[embed]\nbackend = "ecapa"\nmodel = "speechbrain/spkrec-ecapa-voxceleb"\n'
    c = _ws(tmp_path, text)
    assert c.resolve("embed") == BackendConfig(
        "ecapa", {"model": "speechbrain/spkrec-ecapa-voxceleb"}
    )
    c = _ws(tmp_path, '[embed]\nbackend = "ecapa"\n[embed.lang.ja]\nmodel = "m"\n')
    assert c.resolve("embed", "ja") == BackendConfig("ecapa", {"model": "m"})
