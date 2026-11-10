"""Card fingerprints: record, verify, and every way verification can't run."""

from __future__ import annotations

import contextlib
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from typer.testing import CliRunner

from fukikae.backends import loader
from fukikae.backends.base import Synthesizer
from fukikae.cards import cli as cards_cli
from fukikae.cards import verify
from fukikae.cards.card import Card, Reference, Voice, card_id
from fukikae.cards.store import CardStore
from fukikae.cards.verify import TEST_SCRIPT, VerifyResult, verify_card
from fukikae.cli import app
from fukikae.registry.store import Registry

runner = CliRunner()


class FakeSynth(Synthesizer):
    """A sine whose pitch is set by the seed (and the stock voice), 0.05 s per character."""

    name: ClassVar[str] = "fake"
    kind: ClassVar[str] = "tts"
    languages: ClassVar = "*"
    devices: ClassVar = frozenset({"cpu"})
    vram_mb: ClassVar[int] = 0
    sources: ClassVar = frozenset({"stock", "clone"})
    voice_keys: ClassVar = frozenset({"seed", "rate"})
    capabilities: ClassVar = frozenset()
    sample_rate: ClassVar[int] = 8000
    speech_cps: ClassVar = {"en": 20.0}
    model_revision = "1.0"
    RENDERS: ClassVar[list[str]] = []

    def __init__(self, options: dict[str, Any], device: str) -> None:
        self.options, self.device = options, device

    @classmethod
    def validate_options(cls, options: dict[str, Any], source: str | None = None) -> None:
        pass

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def synthesize(self, text: str, voice: Voice, language: str, out: Path, **kw: Any) -> float:  # noqa: ARG002
        self.RENDERS.append(text)
        if voice.source == "clone":
            kw["resolve"](voice.references[0].clip)  # a clone reads its clip
        freq = 150.0 + 40.0 * (voice.seed or 0) + (7.0 if voice.voice == "am_adam" else 0.0)
        rate = self.sample_rate
        t = np.arange(int(len(text) * 0.05 * rate)) / rate
        x = 0.3 * np.sin(2 * np.pi * freq * t)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((x * 32767).astype("<i2").tobytes())
        return len(t) / rate


class OldRevision(FakeSynth):
    name: ClassVar[str] = "old"
    model_revision = "2.0"


class FftEmbedder:
    """Magnitudes of 64 FFT bins of the whole file, L2-normalized."""

    name = "fft"
    dim = 64

    def embed(self, audio: Path, spans: list[tuple[float, float]]) -> np.ndarray:
        from scipy.io import wavfile

        _, data = wavfile.read(audio)
        x = data.astype(np.float64)
        mag = np.abs(np.fft.rfft(x, n=4096))[:2048].reshape(self.dim, -1).sum(axis=1)
        v = (mag / np.linalg.norm(mag)).astype(np.float32)
        return np.stack([v for _ in spans])


EMBEDDERS: list[str] = []


@contextlib.contextmanager
def fft_embedder(name: str) -> Iterator[Any]:
    EMBEDDERS.append(name)
    yield FftEmbedder()


class Source:
    def __init__(self, card: Card) -> None:
        self.inst = loader.backend_class("tts", card.voice.backend)({}, "cpu")
        self.closed = False

    def get(self) -> Any:
        return self.inst

    def close(self) -> None:
        self.closed = True


CLASSES: dict[str, type] = {"fake": FakeSynth, "old": OldRevision}


def _backend_class(kind: str, name: str) -> type:
    if name not in CLASSES:
        raise loader.BackendUnavailable(f"{kind} backend {name!r} cannot be imported (test)")
    return CLASSES[name]


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch):
    FakeSynth.RENDERS = []
    EMBEDDERS.clear()
    monkeypatch.setattr(loader, "backend_class", _backend_class)
    monkeypatch.setattr(loader, "create", lambda _k, bc, dev: CLASSES[bc.backend](bc.options, dev))


def card(seed: int = 1, backend: str = "fake", revision: str = "1.0", lang: str = "en") -> Card:
    return Card(lang, Voice(backend=backend, model_revision=revision, source="stock",
                            voice="af_heart", seed=seed))  # fmt: skip


def run(c: Card, store: CardStore, tmp: Path, *, record: bool = False) -> tuple[VerifyResult, Card]:
    return verify_card(c, store, make_synth=Source, make_embedder=fft_embedder, record=record,
                       work=tmp / "work")  # fmt: skip


def test_test_script_has_three_english_sentences():
    assert len(TEST_SCRIPT["en"]) == 3 and all(TEST_SCRIPT["en"])


def test_record_then_verify_matches(tmp_path: Path):
    store = CardStore(tmp_path)
    res, recorded = run(card(), store, tmp_path, record=True)
    assert res.status == "recorded", res
    fp = recorded.voice.fingerprint
    assert fp is not None and fp.threshold == 0.85 and fp.embedder == verify.FINGERPRINT_EMBEDDER
    assert EMBEDDERS == [verify.FINGERPRINT_EMBEDDER]
    assert TEST_SCRIPT["en"] == FakeSynth.RENDERS
    store.put(recorded)
    res, same = run(recorded, store, tmp_path)
    assert res.status == "match" and res.cosine == pytest.approx(1.0, abs=1e-5)
    assert same == recorded


def test_record_keeps_card_id(tmp_path: Path):
    store = CardStore(tmp_path)
    c = card()
    _, recorded = run(c, store, tmp_path, record=True)
    assert recorded.voice.fingerprint is not None
    assert card_id(recorded) == card_id(c)


def test_changed_seed_mismatches(tmp_path: Path):
    store = CardStore(tmp_path)
    _, recorded = run(card(seed=1), store, tmp_path, record=True)
    fp = recorded.voice.fingerprint
    other = card(seed=9)
    other = Card(other.target_language, Voice(**{**other.voice.__dict__, "fingerprint": fp}))
    res, _ = run(other, store, tmp_path)
    assert res.status == "mismatch" and res.cosine is not None and res.cosine < 0.85
    assert "0.85" in res.detail


def test_rerecord_reports_the_old_comparison(tmp_path: Path):
    store = CardStore(tmp_path)
    _, recorded = run(card(seed=1), store, tmp_path, record=True)
    other = Card("en", Voice(**{**card(seed=9).voice.__dict__,
                                "fingerprint": recorded.voice.fingerprint}))  # fmt: skip
    res, again = run(other, store, tmp_path, record=True)
    assert res.status == "recorded" and res.cosine is not None and res.cosine < 0.85
    assert again.voice.fingerprint != recorded.voice.fingerprint


def test_no_fingerprint_without_record_renders_nothing(tmp_path: Path):
    res, same = run(card(), CardStore(tmp_path), tmp_path)
    assert res.status == "no_fingerprint" and "--record" in res.detail
    assert FakeSynth.RENDERS == [] and same == card()


def test_missing_clip_reports_missing_sidecar(tmp_path: Path):
    store = CardStore(tmp_path)
    wav = tmp_path / "c.wav"
    FakeSynth({}, "cpu").synthesize("hello there", card().voice, "en", wav)
    ref = store.add_sidecar(wav, "clip")
    c = Card("en", Voice(backend="fake", model_revision="1.0", source="clone", seed=1,
                         references=(Reference(ref, "hi", "ja"),)))  # fmt: skip
    _, recorded = run(c, store, tmp_path, record=True)
    store.sidecar_path(ref).unlink()
    FakeSynth.RENDERS = []
    res, _ = run(recorded, store, tmp_path)
    assert res.status == "missing_sidecar" and ref in res.detail
    assert FakeSynth.RENDERS == []


def test_missing_fingerprint_vector_reports_missing_sidecar(tmp_path: Path):
    store = CardStore(tmp_path)
    _, recorded = run(card(), store, tmp_path, record=True)
    fp = recorded.voice.fingerprint
    assert fp is not None
    store.sidecar_path(fp.vector).unlink()
    res, _ = run(recorded, store, tmp_path)
    assert res.status == "missing_sidecar" and fp.vector in res.detail


def test_uninstalled_backend_is_unavailable_not_an_exception(tmp_path: Path):
    res, _ = run(card(backend="missing"), CardStore(tmp_path), tmp_path, record=True)
    assert res.status == "unavailable" and "missing" in res.detail


def test_other_model_revision_is_unavailable(tmp_path: Path):
    res, _ = run(card(backend="old", revision="1.0"), CardStore(tmp_path), tmp_path, record=True)
    assert res.status == "unavailable"
    assert "'1.0'" in res.detail and "'2.0'" in res.detail
    assert FakeSynth.RENDERS == []


def test_backend_unavailable_at_load_is_unavailable(tmp_path: Path):
    def broken(_card: Card) -> Any:
        raise loader.BackendUnavailable("its environment is not set up")

    res, _ = verify_card(card(), CardStore(tmp_path), make_synth=broken,
                         make_embedder=fft_embedder, record=True, work=tmp_path)  # fmt: skip
    assert res.status == "unavailable" and "environment" in res.detail


def test_language_without_a_test_script_is_unavailable(tmp_path: Path):
    res, _ = run(card(lang="tlh"), CardStore(tmp_path), tmp_path, record=True)
    assert res.status == "unavailable" and "tlh" in res.detail


def test_synth_is_closed_after_rendering(tmp_path: Path):
    made: list[Source] = []

    def make(c: Card) -> Source:
        made.append(Source(c))
        return made[-1]

    verify_card(card(), CardStore(tmp_path), make_synth=make, make_embedder=fft_embedder,
                record=True, work=tmp_path)  # fmt: skip
    assert made and made[0].closed


# -- CLI ------------------------------------------------------------------------------------


@pytest.fixture
def cli_fakes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cards_cli, "_synth_factory", lambda _cfg: Source)
    monkeypatch.setattr(cards_cli, "_embedder_factory", lambda: fft_embedder)


def _ws(tmp_path: Path, name: str = "ws") -> tuple[Path, int, CardStore]:
    root = tmp_path / name
    root.mkdir()
    (root / "fukikae.toml").write_text('[series]\ntitle = "Fairy Tail"\n')
    with Registry(root / ".fukikae" / "registry" / "registry.sqlite") as reg:
        _, cid = reg.create_actor_with_character(np.array([1.0, 0.0], dtype=np.float32))
        reg.rename_character(cid, "Natsu")
    return root, cid, CardStore(root)


def test_cli_verify_record_then_verify(tmp_path: Path, cli_fakes: None):  # noqa: ARG001
    root, cid, store = _ws(tmp_path)
    c = store.put(card())
    with Registry(root / ".fukikae" / "registry" / "registry.sqlite") as reg:
        reg.assign_card(cid, "en", c, by="assign", locked=True)
    r = runner.invoke(app, ["voices", "verify", str(root), "Natsu"])
    assert r.exit_code == 1 and "no fingerprint" in r.output
    r = runner.invoke(app, ["voices", "verify", str(root), "Natsu", "--record"])
    assert r.exit_code == 0, r.output
    assert "recorded" in r.output and c in r.output
    assert store.get(c).voice.fingerprint is not None  # same id, fingerprint in the file
    with Registry(root / ".fukikae" / "registry" / "registry.sqlite") as reg:
        assert len(reg.card_history(cid, "en")) == 1  # nothing reassigned
    r = runner.invoke(app, ["voices", "verify", str(root), c])  # by card id
    assert r.exit_code == 0 and "match" in r.output


def test_cli_assign_reports_mismatch_without_blocking(tmp_path: Path, cli_fakes: None):  # noqa: ARG001
    root, cid, store = _ws(tmp_path)
    res, recorded = run(card(seed=1), store, tmp_path, record=True)
    fp = recorded.voice.fingerprint
    drifted = Card("en", Voice(**{**card(seed=7).voice.__dict__, "fingerprint": fp}))
    did = store.put(drifted)
    r = runner.invoke(app, ["voices", "assign", str(root), "Natsu", did])
    assert r.exit_code == 0, r.output
    assert "mismatch" in r.output
    with Registry(root / ".fukikae" / "registry" / "registry.sqlite") as reg:
        cur = reg.card_for(cid, "en")
        assert cur is not None and cur.card_id == did


def test_cli_import_verifies_fingerprinted_cards(tmp_path: Path, cli_fakes: None):  # noqa: ARG001
    src, src_cid, store = _ws(tmp_path, "src")
    _, recorded = run(card(seed=1), store, tmp_path, record=True)
    fp = recorded.voice.fingerprint
    drifted = store.put(Card("en", Voice(**{**card(seed=7).voice.__dict__, "fingerprint": fp})))
    with Registry(src / ".fukikae" / "registry" / "registry.sqlite") as reg:
        reg.assign_card(src_cid, "en", drifted, by="assign", locked=True)
    pack = tmp_path / "p.tar.zst"
    assert runner.invoke(app, ["voices", "export", str(src), "-o", str(pack)]).exit_code == 0
    dst, dst_cid, _ = _ws(tmp_path, "dst")
    r = runner.invoke(app, ["voices", "import", str(dst), str(pack)])
    assert r.exit_code == 0, r.output
    assert "mismatch" in r.output
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        cur = reg.card_for(dst_cid, "en")
        assert cur is not None and cur.card_id == drifted


# -- one real verify --------------------------------------------------------------------------


@pytest.mark.slow
def test_real_kokoro_record_and_verify_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("kokoro")
    pytest.importorskip("pyannote.audio")
    monkeypatch.undo()  # the real loader
    from fukikae.cards.stock import revision, stock_card
    from fukikae.config import load_config

    (tmp_path / "fukikae.toml").write_text('[tts]\nbackend = "kokoro"\n')
    cfg = load_config(tmp_path, None)
    store = CardStore(tmp_path)
    c = stock_card("kokoro", "af_heart", "en", revision("kokoro"))
    make_synth = cards_cli._synth_factory(cfg)
    make_embedder = cards_cli._embedder_factory()
    res, recorded = verify_card(c, store, make_synth=make_synth, make_embedder=make_embedder,
                                record=True, work=tmp_path / "w")  # fmt: skip
    assert res.status == "recorded", res
    res, _ = verify_card(recorded, store, make_synth=make_synth, make_embedder=make_embedder,
                         record=False, work=tmp_path / "w2")  # fmt: skip
    assert res.status == "match" and res.cosine is not None and res.cosine > 0.99, res


# -- ruling CD23(a): the fingerprint embedder is an allowlist, loaded from local files only ----


@pytest.fixture
def no_network_no_models(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every way a model could be fetched or loaded fails the test if it is called."""
    import huggingface_hub

    calls: list[str] = []

    def trap(what: str):
        def f(*a: Any, **kw: Any) -> Any:
            calls.append(what)
            raise AssertionError(f"{what} called with {a} {kw}")

        return f

    for fn in ("hf_hub_download", "snapshot_download", "try_to_load_from_cache"):
        monkeypatch.setattr(huggingface_hub, fn, trap(fn))
        monkeypatch.setattr(verify, fn, trap(fn), raising=False)
    real_create = loader.create

    def create(kind: str, bc: Any, dev: str) -> Any:
        if kind == "embed":
            trap("loader.create(embed)")()
        return real_create(kind, bc, dev)

    monkeypatch.setattr(loader, "create", create)
    return calls


def _with_embedder(store: CardStore, tmp: Path, embedder: str) -> Card:
    path = tmp / "v.npy"
    np.save(path, np.ones(64, dtype=np.float32))
    ref = store.add_sidecar(path, "vector")
    from fukikae.cards.card import Fingerprint

    return Card(
        "en", Voice(**{**card().voice.__dict__, "fingerprint": Fingerprint(embedder, ref, 0.85)})
    )


@pytest.mark.parametrize("embedder", ["wespeaker:evil/model", "ecapa:anything", "wespeaker"])
def test_unknown_embedder_loads_nothing_and_is_unavailable(
    tmp_path: Path, no_network_no_models: list[str], embedder: str
):
    store = CardStore(tmp_path)
    c = _with_embedder(store, tmp_path, embedder)
    made: list[str] = []

    @contextlib.contextmanager
    def never(name: str) -> Iterator[Any]:
        made.append(name)
        if name != verify.FINGERPRINT_EMBEDDER:
            raise AssertionError(f"the embedder factory was asked for {name}")
        yield FftEmbedder()

    for record in (False, True):
        res, _ = verify_card(c, store, make_synth=Source, make_embedder=never, record=record,
                             work=tmp_path / "w")  # fmt: skip
        if record:  # recording ignores the old embedder and uses the pinned one: still no load
            continue
        assert res.status == "unavailable" and embedder in res.detail and "allow" in res.detail
    assert made == [verify.FINGERPRINT_EMBEDDER] and no_network_no_models == []
    assert TEST_SCRIPT["en"] == FakeSynth.RENDERS  # only the recording rendered


def test_cli_embedder_factory_refuses_unknown_names(no_network_no_models: list[str]):
    make = cards_cli._embedder_factory()
    for name in ("wespeaker:evil/model", "ecapa:speechbrain/whatever"):
        with pytest.raises(verify.EmbedderUnavailable), make(name):
            pass
    assert no_network_no_models == []


def test_missing_local_model_is_unavailable_and_never_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_network_no_models: list[str]
):
    looked: list[tuple] = []

    def cache_miss(repo: str, filename: str, revision: str | None = None, **kw: Any) -> None:
        looked.append((repo, filename, revision, kw))
        return None

    monkeypatch.setattr(verify, "try_to_load_from_cache", cache_miss)
    res, _ = verify_card(card(), CardStore(tmp_path), make_synth=Source,
                         make_embedder=cards_cli._embedder_factory(), record=True,
                         work=tmp_path / "w")  # fmt: skip
    assert res.status == "unavailable" and "not in the local" in res.detail
    pinned = verify.FINGERPRINT_EMBEDDERS[verify.FINGERPRINT_EMBEDDER]
    assert looked == [(pinned.repo, pinned.filename, pinned.revision, {})]
    assert no_network_no_models == []


def test_cli_import_with_a_hostile_embedder_loads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_network_no_models: list[str]
):
    monkeypatch.setattr(cards_cli, "_synth_factory", lambda _cfg: Source)
    src, src_cid, store = _ws(tmp_path, "src")
    hostile = store.put(_with_embedder(store, tmp_path, "ecapa:attacker/hyperparams"))
    with Registry(src / ".fukikae" / "registry" / "registry.sqlite") as reg:
        reg.assign_card(src_cid, "en", hostile, by="assign", locked=True)
    pack = tmp_path / "p.tar.zst"
    assert runner.invoke(app, ["voices", "export", str(src), "-o", str(pack)]).exit_code == 0
    dst, dst_cid, _ = _ws(tmp_path, "dst")
    r = runner.invoke(app, ["voices", "import", str(dst), str(pack)])
    assert r.exit_code == 0, r.output
    assert "unavailable" in r.output and "allow" in r.output
    assert no_network_no_models == [] and FakeSynth.RENDERS == []
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        cur = reg.card_for(dst_cid, "en")
        assert cur is not None and cur.card_id == hostile
