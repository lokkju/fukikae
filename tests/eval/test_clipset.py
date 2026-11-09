import json
import wave
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.cli import app
from fukikae.config import BackendConfig
from fukikae.eval import clipset as cs
from fukikae.eval.clipset import (
    Clip,
    clipset_accuracy,
    embed_clips,
    prepare_audio,
    scan_clipset,
    split_clips,
)
from fukikae.eval.identity.metrics import OVERALL
from fukikae.eval.ledger import read_rows


def write_wav(path: Path, seconds: float = 0.5, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (np.sin(np.linspace(0, 200, int(sr * seconds))) * 8000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def clip(character: str, name: str, episode: int | None) -> Clip:
    return Clip(character, Path(f"/x/{character}/{name}.wav"), episode)


def test_scan_takes_top_level_dirs_as_characters_recursively(tmp_path):
    write_wav(tmp_path / "renmin" / "丁义珍" / "丁义珍_人民的名义 01_vocals_260.wav")
    write_wav(tmp_path / "renmin" / "丁义珍" / "丁义珍_人民的名义 12_vocals_9.wav")
    write_wav(tmp_path / "renmin" / ".fukikae-cache" / "16k" / "junk.wav")
    write_wav(tmp_path / "renmin" / "李达康" / "sub" / "李达康_人民的名义 03_vocals_1.wav")
    (tmp_path / "renmin" / "丁义珍" / "notes.txt").write_text("x")
    clips = scan_clipset(tmp_path / "renmin")
    assert sorted({c.character for c in clips}) == ["丁义珍", "李达康"]
    assert sorted(c.episode or 0 for c in clips if c.character == "丁义珍") == [1, 12]
    assert len(clips) == 3

    write_wav(tmp_path / "shendiao" / "spk01" / "happy" / "x.wav")
    write_wav(tmp_path / "shendiao" / "spk02" / "sad" / "y.wav")
    sd = scan_clipset(tmp_path / "shendiao", exclude={"spk02"})
    assert [(c.character, c.episode) for c in sd] == [("spk01", None)]


def test_episode_split_keeps_episodes_on_one_side():
    clips = [clip("a", f"a{i}", ep) for i, ep in enumerate([1, 1, 2, 3, 3, 3])]
    clips += [clip("b", f"b{i}", ep) for i, ep in enumerate([5, 6, 6, 7])]
    train, test, rule, skipped = split_clips(clips)
    assert rule == "episode"
    assert skipped == []
    assert {c.episode for c in train if c.character == "a"} == {1, 2}
    assert {c.episode for c in test if c.character == "a"} == {3}
    # b: 4 clips, half = 2; ep 5 (1) < 2, ep 5+6 (3) >= 2 -> train {5, 6}, test {7}
    assert {c.episode for c in train if c.character == "b"} == {5, 6}
    assert {c.episode for c in test if c.character == "b"} == {7}


def test_half_split_when_any_clip_lacks_an_episode():
    clips = [clip("a", f"{i}", None) for i in (3, 1, 4, 2)]
    clips += [clip("b", "p", 4), clip("b", "q", 5)]
    train, test, rule, skipped = split_clips(clips)
    assert rule == "half"
    assert [c.path.stem for c in train if c.character == "a"] == ["1", "2"]
    assert [c.path.stem for c in test if c.character == "a"] == ["3", "4"]
    assert skipped == []


def test_single_episode_character_is_not_scored():
    clips = [clip("a", "1", 1), clip("a", "2", 2), clip("solo", "1", 4), clip("solo", "2", 4)]
    train, test, rule, skipped = split_clips(clips)
    assert rule == "episode"
    assert skipped == ["solo"]
    assert {c.character for c in train + test} == {"a"}


def fake_embed_factory(calls: list, char_of: dict[Path, str]):
    centres = {"a": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0]), "c": np.array([0, 0, 1.0])}

    def embed(_cfg, audio, spans):
        calls.append(audio)
        rng = np.random.default_rng(len(calls))
        return np.stack([centres[char_of[audio]] + rng.normal(0, 0.05, 3) for _ in spans])

    return embed


def chars(clips, audio):
    return {audio[c.path][0]: c.character for c in clips}


@pytest.mark.ffmpeg
def test_accuracy_is_one_for_separable_fake_embeddings(tmp_path):
    clips = []
    for ch in "abc":
        for i in range(4):
            write_wav(tmp_path / ch / f"{ch}_{i}.wav")
    clips = scan_clipset(tmp_path)
    train, test, _, _ = split_clips(clips)
    cache = tmp_path / ".fukikae-cache"
    audio = prepare_audio(clips, cache)
    cfg = BackendConfig("fake", {})
    embed = fake_embed_factory([], chars(clips, audio))
    tr = embed_clips(train, audio, cfg, embed, cache)
    te = embed_clips(test, audio, cfg, embed, cache)
    acc = clipset_accuracy(train, test, tr, te, audio=audio)
    assert acc[OVERALL] == 1.0
    assert set(acc) == {"a", "b", "c", OVERALL}


@pytest.mark.ffmpeg
def test_embedding_cache_hit_skips_the_embedder(tmp_path):
    for i in range(3):
        write_wav(tmp_path / "a" / f"a_{i}.wav")
    clips = scan_clipset(tmp_path)
    cache = tmp_path / ".fukikae-cache"
    audio = prepare_audio(clips, cache)
    assert all(d == pytest.approx(0.5, abs=0.01) for _, d in audio.values())
    calls: list = []
    cfg = BackendConfig("fake", {})
    mk = lambda: fake_embed_factory(calls, chars(clips, audio))  # noqa: E731
    first = embed_clips(clips, audio, cfg, mk(), cache)
    n = len(calls)
    assert n == 3 and first.shape == (3, 3)
    second = embed_clips(clips, audio, cfg, mk(), cache)
    assert len(calls) == n
    np.testing.assert_allclose(first, second)
    embed_clips(clips, audio, BackendConfig("fake", {"x": 1}), mk(), cache)
    assert len(calls) == 2 * n


@pytest.mark.ffmpeg
def test_cli_writes_one_row_per_embedder(tmp_path, monkeypatch):
    root = tmp_path / "set1"
    for ch in "ab":
        for ep in (1, 2, 3, 4):
            write_wav(root / ch / f"{ch}_show {ep:02d}_v.wav")
    centres = {"a": np.array([1.0, 0.1]), "b": np.array([0.1, 1.0])}

    class FakeResident:
        def __init__(self, _cfg):
            self.seen = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def __call__(self, emb_cfg, audio, spans):
            self.seen.append(emb_cfg.backend)
            # the 16 kHz cache copy is named by hash; classify by sample content is not
            # possible, so the fake reads the character from the cache map below
            return np.stack([centres[char_of[audio]] for _ in spans])

    import fukikae.eval.clipset as cs

    char_of: dict[Path, str] = {}
    real_prepare = cs.prepare_audio

    def spying_prepare(clips, cache):
        out = real_prepare(clips, cache)
        for c in clips:
            char_of[out[c.path][0]] = c.character
        return out

    monkeypatch.setattr(cs, "prepare_audio", spying_prepare)
    monkeypatch.setattr("fukikae.eval.identity.cache.ResidentEmbedder", FakeResident)
    monkeypatch.setattr("fukikae.backends.loader.names", lambda _kind: ["ecapa", "wespeaker"])
    ledger = tmp_path / "l.jsonl"
    res = CliRunner().invoke(
        app,
        [
            "eval",
            "embedders",
            str(root),
            "--embedder",
            "ecapa",
            "--embedder",
            "wespeaker",
            "--language",
            "zh",
            "--set",
            "renmin",
            "--tag",
            "t6",
            "--ledger",
            str(ledger),
        ],
    )
    assert res.exit_code == 0, res.output
    rows, warnings = read_rows(ledger)
    assert warnings == []
    assert [r.backend for r in rows] == ["ecapa", "wespeaker"]
    r = rows[0]
    assert (r.stage, r.split, r.test_set, r.language, r.genre, r.tag) == (
        "embedder",
        "tune",
        "renmin",
        "zh",
        "live-action",
        "t6",
    )
    assert r.episodes == []
    assert r.metrics["nc_overall"] == 1.0
    assert r.metrics["characters"] == 2
    assert r.options["split_rule"] == "episode"
    assert set(r.detail["per_character"]) == {"a", "b"}
    assert "ecapa" in res.output and "1.000" in res.output
    json.dumps(r.detail)


def test_cli_rejects_unknown_embedder(tmp_path, monkeypatch):
    write_wav(tmp_path / "s" / "a" / "a_1.wav")
    monkeypatch.setattr("fukikae.backends.loader.names", lambda _kind: ["ecapa"])
    res = CliRunner().invoke(
        app,
        [
            "eval",
            "embedders",
            str(tmp_path / "s"),
            "--embedder",
            "nope",
            "--language",
            "zh",
            "--no-ledger",
        ],
    )
    assert res.exit_code != 0


def test_folder_split_keeps_whole_folders_on_one_side():
    clips = [
        Clip("a", Path(f"/x/a/{emo}/{i}.wav"), None)
        for emo, n in (("angry", 2), ("happy", 3), ("sad", 3))
        for i in range(n)
    ]
    train, test, rule, skipped = split_clips(clips)
    assert rule == "folder"
    assert skipped == []
    assert {c.path.parent.name for c in train} == {"angry", "happy"}
    assert {c.path.parent.name for c in test} == {"sad"}


def test_sherpa_ledger_row_records_the_onnx_file():
    from fukikae.eval.ledger import model_identity

    model, _ = model_identity("embed", BackendConfig("campplus", {}))
    assert model == "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    model, _ = model_identity("embed", BackendConfig("titanet", {"path": "/m/custom.onnx"}))
    assert model == "custom.onnx"


@pytest.mark.ffmpeg
def test_embedding_cache_misses_when_the_model_identity_changes(tmp_path):
    write_wav(tmp_path / "a" / "a_0.wav")
    clips = scan_clipset(tmp_path)
    cache = tmp_path / ".fukikae-cache"
    audio = prepare_audio(clips, cache)
    calls: list = []
    embed = fake_embed_factory(calls, chars(clips, audio))
    cfg = BackendConfig("fake", {})
    embed_clips(clips, audio, cfg, embed, cache, "old.onnx|None")
    embed_clips(clips, audio, cfg, embed, cache, "old.onnx|None")
    assert len(calls) == 1
    embed_clips(clips, audio, cfg, embed, cache, "new.onnx|None")
    assert len(calls) == 2


def test_cache_identity_never_degrades_to_none(monkeypatch):
    """GT6's fallback rule: when model resolution fails, the clip-set cache key takes the
    backend's declared default (or an `unresolved:` marker), never `None|None`."""
    import fukikae.backends.identity as ident
    from fukikae.backends.embed.wespeaker import DEFAULT_MODEL

    monkeypatch.setattr(ident, "model_identity", lambda _kind, _cfg: (None, None))
    key = cs.cache_identity(BackendConfig("wespeaker", {}))
    assert key != "None|None"
    assert key.split("|")[0] == DEFAULT_MODEL


def test_cache_identity_keeps_the_resolved_model_format():
    from fukikae.eval.ledger import model_identity

    cfg = BackendConfig("campplus", {})
    assert cs.cache_identity(cfg) == "|".join(str(x) for x in model_identity("embed", cfg))
