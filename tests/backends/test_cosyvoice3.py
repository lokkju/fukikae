"""CosyVoice 3: the declarations fukikae reads, the worker impl's pure helpers, delivery through
the stage, and (gpu) a real render through the worker."""

from __future__ import annotations

import io
import os
import tarfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from fukikae.backends import loader
from fukikae.backends.tts import impl_cosyvoice3 as impl
from fukikae.cards.card import Card, Reference, Voice
from fukikae.config import ConfigError
from fukikae.models import Delivery, DubCue
from fukikae.stages import tts


def _cls():
    from fukikae.backends.tts.cosyvoice3 import CosyVoice3Synthesizer

    return CosyVoice3Synthesizer


def test_cosyvoice3_declares_what_the_released_model_supports():
    cls = _cls()
    assert loader.backend_class("tts", "cosyvoice3") is cls
    assert (cls.name, cls.extra) == ("cosyvoice3", "cosyvoice3")
    assert cls.sources == frozenset({"clone"})
    # ruling CD20: no delivery capability without A/B evidence; instruct mode is withdrawn
    assert cls.capabilities == frozenset()
    assert cls.capabilities_for("clone") == frozenset()
    # rate is CosyVoice's speed
    assert cls.voice_keys == frozenset({"seed", "rate"})
    assert cls.sample_rate == 24000
    assert "en" in cls.languages and "ja" not in cls.languages
    assert cls.default_device == "cuda"
    assert cls.option_keys == frozenset({"fp16", "mode"})
    assert cls.model == "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    # the model snapshot and both code pins: any of them can change a voice
    assert impl.REVISION in cls.model_revision
    for _repo, commit, _pkg, _digest in impl.CODE.values():
        assert commit[:12] in cls.model_revision
    assert set(cls.speech_cps) == {"en"} and 8.0 < cls.speech_cps["en"] < 25.0
    assert cls.vram_mb > 0


def test_an_unknown_mode_is_refused_at_construction():
    cls = _cls()
    assert cls({"mode": "cross_lingual"}, "cpu").options == {"mode": "cross_lingual"}
    with pytest.raises(ConfigError, match="mode"):
        cls({"mode": "sideways"}, "cpu")
    with pytest.raises(ConfigError, match="no option"):
        cls({"temperature": 0.5}, "cpu")


def test_auto_mode_is_cross_lingual_when_a_reference_differs_from_the_target():
    assert impl.resolve_mode("auto", ["ja", "ja"], "en") == "cross_lingual"
    assert impl.resolve_mode("auto", ["en"], "en") == "zero_shot"
    assert impl.resolve_mode("auto", ["", "en"], "en") == "zero_shot"  # unknown is no evidence
    assert impl.resolve_mode("zero_shot", ["ja"], "en") == "zero_shot"
    with pytest.raises(ValueError, match="mode"):
        impl.resolve_mode("both", [], "en")


def test_the_prompt_takes_whole_clips_up_to_the_limit():
    refs = [("a.wav", 12.0, "uno"), ("b.wav", 8.0, "dos"), ("c.wav", 6.0, "tres")]
    clips, text = impl.prompt_plan(refs)
    assert clips == ["a.wav", "b.wav"] and text == "uno dos"  # c would pass 25 s
    assert impl.prompt_plan([("long.wav", 40.0, "x")]) == (["long.wav"], "x")


def test_cosyvoice_bracket_tags_are_stripped_by_the_worker():
    """Ruling CD20: CosyVoice's own control tokens ([laughter], [breath], <strong>, the
    phoneme tags) never reach the model from the text or a transcript."""
    from fukikae.backends.tts import worker

    markup = impl.CONTROL_MARKUP
    text = "Well [laughter] that's <strong>it</strong>[breath] [AA1]ok [ǜ] <laughter>ha</laughter>"
    assert worker.strip_control_markup(text, markup) == "Well that's it ok ha"
    # bracketed words that are not tokens stay
    assert worker.strip_control_markup("[sic] and [Natsu]", markup) == "[sic] and [Natsu]"
    req = {
        "op": "synthesize", "text": "Hi [cough] there", "instruction": None,
        "voice": {"references": [{"clip": "/x.wav", "transcript": "やあ[sigh]"}]},
    }  # fmt: skip
    clean = worker.sanitize_request(req, markup)
    assert clean["text"] == "Hi there"
    assert clean["voice"]["references"][0]["transcript"] == "やあ"


def test_control_markup_covers_every_special_token_of_the_pinned_tokenizer():
    """Each additional special token of the pinned CosyVoice3Tokenizer that isn't <|...|>
    syntax is in CONTROL_MARKUP (needs the fetched code in .venvs/cosyvoice3)."""
    import ast

    from fukikae.backends.tts import worker
    from fukikae.backends.tts.cosyvoice3 import CosyVoice3Synthesizer

    venv = CosyVoice3Synthesizer({}, "cpu").venv()
    src = Path(impl.code_root("CosyVoice", str(venv))) / "cosyvoice" / "tokenizer" / "tokenizer.py"
    if not src.is_file():
        pytest.skip("the pinned CosyVoice code is not fetched here")
    tree = ast.parse(src.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CosyVoice3Tokenizer"
    )
    tokens = [
        c.value
        for node in ast.walk(cls)
        if isinstance(node, ast.List)
        for c in node.elts
        if isinstance(c, ast.Constant) and isinstance(c.value, str)
    ]
    assert len(tokens) > 100
    for tok in tokens:
        assert worker.strip_control_markup(f"a{tok}b", impl.CONTROL_MARKUP) in ("a b", "ab"), tok


def test_fetch_drops_bytecode_and_the_digest_counts_it(tmp_path):
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text("a = 1\n")
    (tmp_path / "pkg" / "__pycache__" / "a.cpython-310.pyc").write_bytes(b"cache")
    with_cache = impl.tree_digest(str(tmp_path / "pkg"))
    impl.drop_bytecode(str(tmp_path / "pkg"))
    assert not (tmp_path / "pkg" / "__pycache__").exists()
    # unchecked bytecode changes the digest, so load refuses a tree that holds any
    assert impl.tree_digest(str(tmp_path / "pkg")) != with_cache


def _archive(members: Sequence[tuple[str, bytes | None, str]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "link":
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar.addfile(info)
            else:
                assert data is not None
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_extract_keeps_only_the_package_and_refuses_escapes(tmp_path):
    good = _archive([
        ("Repo-abc/", None, "dir"),
        ("Repo-abc/README.md", b"readme", "file"),
        ("Repo-abc/pkg/", None, "dir"),
        ("Repo-abc/pkg/__init__.py", b"x = 1\n", "file"),
        ("Repo-abc/pkg/sub/m.py", b"y = 2\n", "file"),
    ])  # fmt: skip
    impl.extract_package(good, "pkg", str(tmp_path / "a"))
    kept = sorted(p.relative_to(tmp_path / "a").as_posix() for p in (tmp_path / "a").rglob("*.py"))
    assert kept == ["pkg/__init__.py", "pkg/sub/m.py"]
    assert not (tmp_path / "a" / "README.md").exists()
    for bad in (
        [("Repo-abc/", None, "dir"), ("Repo-abc/pkg/../../evil.py", b"!", "file")],
        [("Repo-abc/", None, "dir"), ("Repo-abc/pkg/l", None, "link")],
    ):
        with pytest.raises(RuntimeError):
            impl.extract_package(_archive(bad), "pkg", str(tmp_path / "b"))
    assert not (tmp_path / "evil.py").exists()


def test_tree_digest_tracks_content_and_refuses_links(tmp_path):
    (tmp_path / "pkg").mkdir(parents=True)
    (tmp_path / "pkg" / "a.py").write_text("a = 1\n")
    first = impl.tree_digest(str(tmp_path / "pkg"))
    (tmp_path / "pkg" / "a.py").write_text("a = 2\n")
    assert impl.tree_digest(str(tmp_path / "pkg")) != first
    os.symlink("/etc/passwd", tmp_path / "pkg" / "b.py")
    with pytest.raises(RuntimeError, match="symbolic link"):
        impl.tree_digest(str(tmp_path / "pkg"))


def test_load_refuses_code_that_does_not_match_its_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(impl.sys, "prefix", str(tmp_path))
    with pytest.raises(RuntimeError, match=r"missing.*fukikae backends setup cosyvoice3"):
        impl.check_code()
    for name, (_repo, _commit, package, _digest) in impl.CODE.items():
        root = Path(impl.code_root(name))
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("edited = True\n")
    with pytest.raises(RuntimeError, match="hashes to"):
        impl.check_code()
    pinned = {
        n: (r, c, p, impl.tree_digest(os.path.join(impl.code_root(n), p)))
        for n, (r, c, p, _) in impl.CODE.items()
    }
    monkeypatch.setattr(impl, "CODE", pinned)
    assert impl.check_code() == [impl.code_root(n) for n in pinned]


def _clone_card(emotion_strength: float = 0.8) -> Card:
    cls = _cls()
    voice = Voice(
        backend="cosyvoice3",
        model_revision=cls.model_revision,
        source="clone",
        seed=1234,
        emotion_strength=emotion_strength,
        references=(Reference(clip="sha256:" + "a" * 64, transcript="やあ", language="ja"),),
    )
    return Card(voice=voice, target_language="en")


def _cue(direction: str | None) -> DubCue:
    return DubCue(
        id="c0001", start=3.0, end=5.0, character_id=1, extra=None, speaker="x", source_text="src",
        text="Hello there.", budget=40,
        delivery=Delivery(reference=(3.0, 5.0), direction=direction, emotion="angry",
                          emotion_confidence=0.9),
    )  # fmt: skip


def test_delivery_passes_nothing():
    """No direction, emotion audio or vector reaches CosyVoice 3: every line is neutral."""
    cls = _cls()
    passed = tts.delivery_passed(_cue("sarcastic"), _clone_card(), cls, 2.0)
    assert passed == {} and tts.neutral_only(passed)


@pytest.mark.gpu
def test_real_cosyvoice3_renders_a_clone_line(tmp_path, monkeypatch):
    """Needs `.venvs/cosyvoice3` (README, "Cloning backends") and GPU 0."""
    import numpy as np
    from scipy.io import wavfile

    monkeypatch.delenv("FUKIKAE_WORKER_CHECKOUT", raising=False)
    cls = _cls()
    ref = tmp_path / "ref.wav"
    t = np.arange(24000 * 4) / 24000
    wavfile.write(ref, 24000, (0.2 * np.sin(2 * np.pi * 180 * t) * 32767).astype("<i2"))
    synth = cls({}, "cuda:0")
    synth.load()
    try:
        card = _clone_card()
        out = tmp_path / "o.wav"
        seconds = synth.synthesize(
            "Hello there, this is a test.", card.voice, "en", out, resolve=lambda _r: ref
        )
        assert 0.5 < seconds < 10.0
    finally:
        synth.unload()
