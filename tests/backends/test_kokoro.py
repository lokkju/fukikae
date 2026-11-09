import wave
from pathlib import Path

import pytest

from fukikae.backends.loader import EXTRAS, backend_class
from fukikae.backends.tts.kokoro import (
    LANG_CODES,
    REPO,
    REVISION,
    STOCK_VOICES,
    KokoroSynthesizer,
)
from fukikae.cards.card import CardError, parse_card, validate_for_backend
from fukikae.cards.stock import STOCK_POOLS, stock_card
from fukikae.config import ConfigError


def test_kokoro_declares_stock_and_blend():
    k = KokoroSynthesizer
    assert backend_class("tts", "kokoro") is k
    assert k.name == "kokoro" and k.kind == "tts"
    assert k.sources == {"stock", "blend"} and k.voice_keys == {"rate"}
    assert k.capabilities == frozenset() and k.sample_rate == 24000
    assert k.languages == {"en", "es", "fr", "ja", "zh", "hi", "it", "pt"}
    assert k.devices == {"cuda", "cpu"} and k.vram_mb == 1500
    assert REPO == "hexgrad/Kokoro-82M" and len(REVISION) == 40
    assert k.model_revision == REVISION
    assert LANG_CODES["en"] == "a"
    assert EXTRAS["kokoro"] == "kokoro"


def test_every_pool_voice_is_a_kokoro_voice():
    for pool in STOCK_POOLS["kokoro"]["en"].values():
        assert set(pool) <= STOCK_VOICES


def test_identity_is_the_pinned_model():
    from fukikae.backends.identity import identity_key
    from fukikae.config import BackendConfig

    assert identity_key("tts", BackendConfig("kokoro", {})) == {"model": REPO, "revision": REVISION}


def test_kokoro_refuses_options_and_unsupported_keys():
    with pytest.raises(CardError, match="steps"):
        KokoroSynthesizer.validate_options({"steps": 4})
    with pytest.raises(ConfigError, match="speed"):
        KokoroSynthesizer({"speed": 2}, "cpu")
    card = parse_card(
        'target_language = "en"\n[voice]\nbackend = "kokoro"\n'
        f'model_revision = "{REVISION}"\nsource = "stock"\nvoice = "af_heart"\nseed = 3\n'
    )
    with pytest.raises(CardError, match="seed"):
        validate_for_backend(card, KokoroSynthesizer)
    validate_for_backend(stock_card("kokoro", "af_heart", "en", REVISION), KokoroSynthesizer)


def test_blend_averages_voice_tensors_by_weight():
    torch = pytest.importorskip("torch")
    from fukikae.backends.tts.kokoro import blend_packs

    a = torch.zeros(3, 1, 4)
    b = torch.ones(3, 1, 4)
    out = blend_packs([a, b], [1.0, 3.0])
    assert torch.allclose(out, torch.full((3, 1, 4), 0.75))
    with pytest.raises(CardError, match="weight"):
        blend_packs([a, b], [0.0, 0.0])


def _render(tmp_path: Path, device: str) -> None:
    k = KokoroSynthesizer({}, device)
    k.load()
    try:
        out = tmp_path / "hello.wav"
        card = stock_card("kokoro", "af_heart", "en", REVISION)
        seconds = k.synthesize(
            "Hello there, Lucy.", card.voice, "en", out, resolve=lambda r: Path(r)
        )
    finally:
        k.unload()
    with wave.open(str(out)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1
        assert w.getnframes() / 24000 == pytest.approx(seconds)
    assert 0.5 < seconds < 5.0


@pytest.mark.gpu
def test_kokoro_renders_a_stock_line(tmp_path: Path):
    _render(tmp_path, "cuda:0")


@pytest.mark.slow
def test_kokoro_renders_a_stock_line_on_cpu(tmp_path: Path):
    _render(tmp_path, "cpu")
