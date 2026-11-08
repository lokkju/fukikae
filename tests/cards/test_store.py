import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from fukikae.cards.card import CardError, Fingerprint, card_id, parse_card
from fukikae.cards.store import CardStore
from tests.cards.test_card import CLIP, SPEC_CARD, STOCK_CARD, VECTOR


def test_put_get_is_content_addressed(tmp_path: Path):
    store = CardStore(tmp_path)
    card = parse_card(SPEC_CARD)
    cid = store.put(card)
    assert cid == card_id(card)
    assert store.put(card) == cid
    assert sorted(p.name for p in store.root.glob("*.toml")) == [f"{cid}.toml"]
    assert store.root == tmp_path / ".fukikae" / "cards"
    assert store.card_path(cid) == store.root / f"{cid}.toml"
    assert store.has(cid)
    assert store.get(cid) == card


def test_get_missing_card_names_id_and_dir(tmp_path: Path):
    store = CardStore(tmp_path)
    assert not store.has("0123456789abcdef")
    with pytest.raises(CardError, match=r"card 0123456789abcdef not found in .*cards"):
        store.get("0123456789abcdef")


def test_get_refuses_a_file_whose_content_is_another_card(tmp_path: Path):
    store = CardStore(tmp_path)
    cid = store.put(parse_card(STOCK_CARD))
    store.card_path(cid).write_text(STOCK_CARD.replace("af_heart", "am_adam"), encoding="utf-8")
    with pytest.raises(CardError, match=cid):
        store.get(cid)


def test_card_path_rejects_a_malformed_id(tmp_path: Path):
    with pytest.raises(CardError, match="card id"):
        CardStore(tmp_path).card_path("../../etc")


def test_sidecar_copied_by_hash(tmp_path: Path):
    store = CardStore(tmp_path)
    wav = tmp_path / "ref.WAV"
    wav.write_bytes(b"RIFF fake wav bytes")
    digest = hashlib.sha256(b"RIFF fake wav bytes").hexdigest()
    ref = store.add_sidecar(wav, "clip")
    assert ref == f"sha256:{digest}"
    assert (store.root / "clips" / f"{digest}.wav").read_bytes() == b"RIFF fake wav bytes"
    assert store.add_sidecar(wav, "clip") == ref
    assert store.sidecar_path(ref) == store.root / "clips" / f"{digest}.wav"
    npy = tmp_path / "v.npy"
    npy.write_bytes(b"vector")
    vref = store.add_sidecar(npy, "vector")
    assert store.sidecar_path(vref).parent == store.root / "vectors"
    lref = store.add_sidecar(npy, "latent")
    assert lref == vref
    assert (store.root / "latents" / f"{vref.removeprefix('sha256:')}.npy").exists()


def test_add_sidecar_rejects_unknown_kind(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(CardError, match="kind"):
        CardStore(tmp_path).add_sidecar(wav, "audio")  # type: ignore[arg-type]


def test_missing_sidecar_is_named(tmp_path: Path):
    store = CardStore(tmp_path)
    card = parse_card(SPEC_CARD)
    assert store.missing_sidecars(card) == [CLIP, VECTOR]
    with pytest.raises(CardError, match="9e1f"):
        store.sidecar_path(CLIP)
    with pytest.raises(CardError, match="sidecar"):
        store.sidecar_path("sha256:9e1f...")


def test_recording_fingerprint_keeps_id_and_overwrites(tmp_path: Path):
    store = CardStore(tmp_path)
    card = parse_card(SPEC_CARD)
    bare = replace(card, voice=replace(card.voice, fingerprint=None))
    cid = store.put(bare)
    assert store.get(cid).voice.fingerprint is None
    recorded = replace(
        bare,
        voice=replace(
            bare.voice,
            fingerprint=Fingerprint(embedder="wespeaker:resnet34", vector=VECTOR, threshold=0.9),
        ),
    )
    assert store.put(recorded) == cid
    got = store.get(cid).voice.fingerprint
    assert got is not None and got.threshold == 0.9
    assert list(store.root.glob("*.tmp")) == []


def test_corrupted_sidecar_is_refused_on_use(tmp_path: Path):
    store = CardStore(tmp_path)
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFF" + b"x" * 100)
    ref = store.add_sidecar(wav, "clip")
    stored = store.root / "clips" / f"{ref.removeprefix('sha256:')}.wav"
    stored.write_bytes(stored.read_bytes()[:50])
    with pytest.raises(CardError, match=r"corrupted"):
        store.sidecar_path(ref)


def test_files_are_group_readable(tmp_path: Path):
    store = CardStore(tmp_path)
    cid = store.put(parse_card(STOCK_CARD))
    assert store.card_path(cid).stat().st_mode & 0o777 == 0o644
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    ref = store.add_sidecar(wav, "clip")
    assert store.sidecar_path(ref).stat().st_mode & 0o777 == 0o644


def test_put_refuses_an_invalid_card_built_in_code(tmp_path: Path):
    card = parse_card(STOCK_CARD)
    bad = replace(card, voice=replace(card.voice, rate=-1.0))
    with pytest.raises(CardError, match="rate"):
        CardStore(tmp_path).put(bad)
