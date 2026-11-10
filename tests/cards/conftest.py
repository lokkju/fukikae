import pytest


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch: pytest.MonkeyPatch):
    """The test cards' backend "fake" resolves to the test Synthesizer, so assign and edit can
    run the tts stage's card check (A8) on them; every other name resolves as installed."""
    from fukikae.backends import loader
    from tests.cards.test_card import Fake

    real = loader.backend_class
    monkeypatch.setattr(
        loader, "backend_class", lambda kind, name: Fake if name == "fake" else real(kind, name)
    )
