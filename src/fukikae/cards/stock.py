"""Stock voices: the card a stock voice becomes, read for cards made before D41 and for
``eval tts`` baselines.

fukikae never renders a stock voice it chose (D41): no pipeline path makes or assigns one,
and the pools ``eval tts`` picks its stock baselines from live in ``fukikae.eval.stock``. A
stock card holds nothing but the backend, its pinned model revision, ``source = "stock"`` and
the voice id, so the same voice on the same model revision is always the same card id (CD1).
"""

from __future__ import annotations

from fukikae.cards.card import Card, CardError, Voice


def stock_card(backend: str, voice_id: str, target_language: str, revision: str) -> Card:
    return Card(
        target_language=target_language,
        voice=Voice(backend=backend, model_revision=revision, source="stock", voice=voice_id),
        meta={"title": f"{backend} stock voice {voice_id}"},
    )


def parse_stock_ref(ref: str) -> tuple[str, str]:
    """``stock:<backend>:<voice>`` as (backend, voice)."""
    parts = ref.split(":")
    if len(parts) != 3 or parts[0] != "stock" or not parts[1] or not parts[2]:
        raise CardError(f"{ref!r} is not a stock voice; write it as stock:<backend>:<voice>")
    return parts[1], parts[2]


def revision(backend: str) -> str:
    """The model revision a TTS backend renders with, which its stock cards record."""
    from fukikae.backends import loader

    rev = getattr(loader.backend_class("tts", backend), "model_revision", None)
    if not isinstance(rev, str) or not rev:
        raise CardError(f"TTS backend {backend!r} declares no model_revision")
    return rev


def check_stock_voice(backend: str, voice_id: str) -> None:
    """Refuse a voice id the backend doesn't ship (when it lists its voices)."""
    from fukikae.backends import loader

    known = getattr(loader.backend_class("tts", backend), "stock_voices", None)
    if known is not None and voice_id not in known:
        raise CardError(
            f"{backend} has no stock voice {voice_id!r}; it has {', '.join(sorted(known))}"
        )
