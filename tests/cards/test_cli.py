import sys
import tomllib
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from fukikae.cards.card import parse_card
from fukikae.cards.store import CardStore
from fukikae.cli import app
from fukikae.registry.store import CardAssignment, Registry
from fukikae.workspace import Workspace
from tests.cards.test_card import STOCK_CARD

runner = CliRunner()

STYLED = (
    STOCK_CARD
    + """
[style]
register = "casual and rough"
catchphrases = [ { source = "燃えてきたぞ", target = "I'm all fired up!" } ]
"""
)


def setup(tmp_path: Path, target: str | None = None) -> tuple[Workspace, int, CardStore]:
    root = tmp_path / "ws"
    root.mkdir()
    if target is not None:
        (root / "fukikae.toml").write_text(f'target_language = "{target}"\n')
    ws = Workspace(root)
    with Registry(ws.registry_path) as reg:
        _, cid = reg.create_actor_with_character(np.array([1.0, 0.0], dtype=np.float32))
        reg.rename_character(cid, "Natsu")
    return ws, cid, CardStore(root)


def _cur(reg: Registry, cid: int, lang: str) -> CardAssignment:
    a = reg.card_for(cid, lang)
    assert a is not None
    return a


def invoke(*args: str):
    return runner.invoke(app, ["voices", *args])


def test_assign_locks_by_default_and_list_shows_it(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    card_id = store.put(parse_card(STOCK_CARD))
    r = invoke("assign", str(ws.root), "natsu", card_id)
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        cur = reg.card_for(cid, "en")
        assert cur is not None and cur.card_id == card_id and cur.locked and cur.by == "assign"
    r = invoke("list", str(ws.root))
    assert r.exit_code == 0, r.output
    assert "Natsu" in r.output and card_id in r.output and "locked" in r.output
    r = invoke("show", str(ws.root), "Natsu")
    assert r.exit_code == 0, r.output
    assert card_id in r.output and "af_heart" in r.output


def test_assign_no_lock(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    card_id = store.put(parse_card(STOCK_CARD))
    assert invoke("assign", str(ws.root), str(cid), card_id, "--no-lock").exit_code == 0
    with Registry(ws.registry_path) as reg:
        assert not _cur(reg, cid, "en").locked


def test_assign_refuses_other_language_card(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    card_id = store.put(parse_card(STOCK_CARD))  # an "en" card
    r = invoke("assign", str(ws.root), "Natsu", card_id, "--lang", "es")
    assert r.exit_code == 1
    assert "'en'" in r.output and "'es'" in r.output
    with Registry(ws.registry_path) as reg:
        assert reg.card_for(cid, "es") is None


def test_lang_defaults_to_workspace_target_language(tmp_path: Path):
    ws, cid, store = setup(tmp_path, target="es")
    card_id = store.put(parse_card(STOCK_CARD.replace('"en"', '"es"')))
    assert invoke("assign", str(ws.root), "Natsu", card_id).exit_code == 0
    with Registry(ws.registry_path) as reg:
        assert _cur(reg, cid, "es").card_id == card_id
        assert reg.card_for(cid, "en") is None


def test_assign_refuses_unknown_card_and_character(tmp_path: Path):
    ws, _, store = setup(tmp_path)
    r = invoke("assign", str(ws.root), "Natsu", "0123456789abcdef")
    assert r.exit_code == 1 and "0123456789abcdef" in r.output
    r = invoke("assign", str(ws.root), "Gray", store.put(parse_card(STOCK_CARD)))
    assert r.exit_code == 1 and "Gray" in r.output


def test_assign_stock_voice_makes_and_locks_a_stock_card(tmp_path: Path):
    from fukikae.backends.tts.kokoro import REVISION

    ws, cid, store = setup(tmp_path)
    r = invoke("assign", str(ws.root), "Natsu", "stock:kokoro:af_heart")
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        cur = _cur(reg, cid, "en")
    assert cur.by == "assign" and cur.locked and cur.card_id is not None
    card = store.get(cur.card_id)
    assert (card.voice.backend, card.voice.source, card.voice.voice) == (
        "kokoro",
        "stock",
        "af_heart",
    )
    assert card.voice.model_revision == REVISION and card.target_language == "en"
    assert cur.card_id in r.output


def test_assign_stock_refuses_unknown_voice_or_backend(tmp_path: Path):
    ws, cid, _ = setup(tmp_path)
    r = invoke("assign", str(ws.root), "Natsu", "stock:kokoro:zz_nobody")
    assert r.exit_code == 1 and "zz_nobody" in r.output
    r = invoke("assign", str(ws.root), "Natsu", "stock:nosuch:af_heart")
    assert r.exit_code == 1 and "nosuch" in r.output
    r = invoke("assign", str(ws.root), "Natsu", "stock:kokoro")
    assert r.exit_code == 1 and "stock:<backend>:<voice>" in r.output
    with Registry(ws.registry_path) as reg:
        assert reg.card_for(cid, "en") is None


def test_history_lists_numbered_rows(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    a = store.put(parse_card(STOCK_CARD))
    b = store.put(parse_card(STOCK_CARD.replace("af_heart", "am_adam")))
    assert invoke("assign", str(ws.root), "Natsu", a).exit_code == 0
    assert invoke("assign", str(ws.root), "Natsu", b).exit_code == 0
    r = invoke("revert", str(ws.root), "Natsu")
    assert r.exit_code == 0, r.output
    assert a in r.output
    r = invoke("history", str(ws.root), "Natsu")
    assert r.exit_code == 0, r.output
    lines = [ln for ln in r.output.splitlines() if ln[:1].isdigit()]
    assert [ln.split()[0] for ln in lines] == ["1", "2", "3"]
    assert "assign" in lines[0] and a in lines[0] and b in lines[1] and "revert" in lines[2]
    r = invoke("revert", str(ws.root), "Natsu", "--to", "2")
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        assert _cur(reg, cid, "en").card_id == b
    r = invoke("revert", str(ws.root), "Natsu", "--to", "9")
    assert r.exit_code == 1 and "9" in r.output


def test_lock_and_unlock(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    a = store.put(parse_card(STOCK_CARD))
    assert invoke("assign", str(ws.root), "Natsu", a, "--no-lock").exit_code == 0
    assert invoke("lock", str(ws.root), "Natsu").exit_code == 0
    with Registry(ws.registry_path) as reg:
        assert _cur(reg, cid, "en").locked
    assert invoke("unlock", str(ws.root), "Natsu").exit_code == 0
    with Registry(ws.registry_path) as reg:
        assert not _cur(reg, cid, "en").locked
    r = invoke("lock", str(ws.root), "Natsu", "--lang", "de")
    assert r.exit_code == 1 and "no card" in r.output


EDIT_RATE = """\
import sys, tomllib, tomli_w
p = sys.argv[1]
with open(p, "rb") as f:
    d = tomllib.load(f)
d["voice"]["rate"] = 1.2
with open(p, "wb") as f:
    tomli_w.dump(d, f)
"""


def test_edit_creates_derived_card_and_locks(tmp_path: Path, monkeypatch):
    ws, cid, store = setup(tmp_path)
    a = store.put(parse_card(STOCK_CARD))
    assert invoke("assign", str(ws.root), "Natsu", a, "--no-lock").exit_code == 0
    script = tmp_path / "editor.py"
    script.write_text(EDIT_RATE)
    monkeypatch.setenv("EDITOR", f"{sys.executable} {script}")
    r = invoke("edit", str(ws.root), "Natsu")
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        cur = reg.card_for(cid, "en")
    assert cur is not None and cur.card_id is not None
    assert cur.card_id != a and cur.by == "edit" and cur.locked
    new = store.get(cur.card_id)
    assert new.voice.rate == 1.2 and new.meta["derived_from"] == a
    assert store.get(a).voice.rate == 1.0  # the original card is untouched


def test_edit_without_change_assigns_nothing(tmp_path: Path, monkeypatch):
    ws, cid, store = setup(tmp_path)
    a = store.put(parse_card(STOCK_CARD))
    assert invoke("assign", str(ws.root), "Natsu", a).exit_code == 0
    monkeypatch.setenv("EDITOR", "true")
    r = invoke("edit", str(ws.root), "Natsu")
    assert r.exit_code == 0, r.output
    assert "unchanged" in r.output
    with Registry(ws.registry_path) as reg:
        assert len(reg.card_history(cid, "en")) == 1


def test_edit_refuses_an_invalid_card(tmp_path: Path, monkeypatch):
    ws, cid, store = setup(tmp_path)
    a = store.put(parse_card(STOCK_CARD))
    assert invoke("assign", str(ws.root), "Natsu", a).exit_code == 0
    script = tmp_path / "bad.py"
    script.write_text("import sys\nopen(sys.argv[1], 'w').write('target_language = 3\\n')\n")
    monkeypatch.setenv("EDITOR", f"{sys.executable} {script}")
    r = invoke("edit", str(ws.root), "Natsu")
    assert r.exit_code == 1 and "target_language" in r.output
    with Registry(ws.registry_path) as reg:
        assert _cur(reg, cid, "en").card_id == a


def test_derive_empties_catchphrase_targets(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    a = store.put(parse_card(STYLED))
    assert invoke("assign", str(ws.root), "Natsu", a).exit_code == 0
    r = invoke("derive", str(ws.root), "Natsu", "--to", "es")
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        es = reg.card_for(cid, "es")
        assert _cur(reg, cid, "en").card_id == a  # languages never replace each other
    assert es is not None and es.card_id is not None and es.by == "derive" and not es.locked
    card = store.get(es.card_id)
    original = store.get(a)
    assert card.target_language == "es" and card.meta["derived_from"] == a
    assert card.voice == original.voice
    assert card.style is not None
    assert card.style["catchphrases"] == [{"source": "燃えてきたぞ", "target": ""}]
    assert card.style["register"] == "casual and rough"
    # a second derive would replace the es card: refused without --replace
    r = invoke("derive", str(ws.root), "Natsu", "--to", "es")
    assert r.exit_code == 1 and "--replace" in r.output
    assert invoke("derive", str(ws.root), "Natsu", "--to", "es", "--replace").exit_code == 0
    r = invoke("derive", str(ws.root), "Natsu", "--to", "en")
    assert r.exit_code == 1


def test_registry_rename_sets_external_id(tmp_path: Path):
    ws, cid, _ = setup(tmp_path)
    r = runner.invoke(
        app, ["registry", "rename", str(ws.root), str(cid), "Natsu", "--external-id", "anidb:1"]
    )
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        assert reg.characters()[0].external_id == "anidb:1"
    r = runner.invoke(app, ["registry", "show", str(ws.root), "1"])
    assert "anidb:1" in r.output


def test_tomllib_reads_show_output_card(tmp_path: Path):
    ws, _, store = setup(tmp_path)
    a = store.put(parse_card(STOCK_CARD))
    assert invoke("assign", str(ws.root), "Natsu", a).exit_code == 0
    r = invoke("show", str(ws.root), "Natsu")
    toml_part = r.output.split("\n\n", 1)[1]
    assert tomllib.loads(toml_part)["voice"]["voice"] == "af_heart"


def test_revert_refuses_a_card_missing_from_the_store(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    a = store.put(parse_card(STOCK_CARD))
    b = store.put(parse_card(STOCK_CARD.replace("af_heart", "am_adam")))
    assert invoke("assign", str(ws.root), "Natsu", a).exit_code == 0
    assert invoke("assign", str(ws.root), "Natsu", b).exit_code == 0
    store.card_path(a).unlink()
    for args in ([], ["--to", "1"]):
        r = invoke("revert", str(ws.root), "Natsu", *args)
        assert r.exit_code == 1 and a in r.output and "not in the card store" in r.output
    with Registry(ws.registry_path) as reg:
        assert _cur(reg, cid, "en").card_id == b and len(reg.card_history(cid, "en")) == 2
    r = invoke("assign", str(ws.root), "Natsu", a)
    assert r.exit_code == 1 and "not in the card store" in r.output


def test_registry_merge_reports_the_card_that_lost(tmp_path: Path):
    ws, c1, store = setup(tmp_path)
    with Registry(ws.registry_path) as reg:
        a2, c2 = reg.create_actor_with_character(np.array([0.0, 1.0], dtype=np.float32))
        reg.assign_card(c1, "en", "a" * 16, by="assign", locked=True)
        reg.assign_card(c2, "en", "b" * 16, by="assign", locked=True)
    r = runner.invoke(app, ["registry", "merge", str(ws.root), "1", str(a2)])
    assert r.exit_code == 0, r.output
    assert "a" * 16 in r.output and "b" * 16 in r.output and "stays in history" in r.output


def _auto_stock(ws: Workspace, store: CardStore, cid: int, voice: str, rev: str) -> str:
    from fukikae.cards.stock import stock_card

    card = store.put(stock_card("kokoro", voice, "en", rev))
    with Registry(ws.registry_path) as reg:
        reg.assign_card(cid, "en", card, by="auto", locked=False, note="gender estimate f")
    return card


def test_show_prints_the_assignment_note(tmp_path: Path):
    ws, cid, store = setup(tmp_path)
    _auto_stock(ws, store, cid, "af_heart", "old")
    r = invoke("show", str(ws.root), "Natsu")
    assert r.exit_code == 0 and "gender estimate f" in r.output


def test_restock_repins_only_uncurated_auto_stock_cards(tmp_path: Path):
    from fukikae.backends.tts.kokoro import REVISION

    ws, natsu, store = setup(tmp_path)
    with Registry(ws.registry_path) as reg:
        lucy = reg.create_actor_with_character(np.array([0.0, 1.0], dtype=np.float32))[1]
        gray = reg.create_actor_with_character(np.array([1.0, 1.0], dtype=np.float32))[1]
    old = _auto_stock(ws, store, natsu, "am_adam", "old")
    locked = _auto_stock(ws, store, lucy, "af_heart", "old")
    chosen = _auto_stock(ws, store, gray, "am_puck", "old")
    with Registry(ws.registry_path) as reg:
        reg.lock_card(lucy, "en", True)
        reg.assign_card(gray, "en", chosen, by="assign", locked=False)

    r = invoke("restock", str(ws.root), "--dry-run")
    assert r.exit_code == 0, r.output
    assert "am_adam" in r.output and "af_heart" not in r.output and "am_puck" not in r.output
    with Registry(ws.registry_path) as reg:
        assert _cur(reg, natsu, "en").card_id == old

    r = invoke("restock", str(ws.root))
    assert r.exit_code == 0, r.output
    with Registry(ws.registry_path) as reg:
        cur = _cur(reg, natsu, "en")
        assert cur.card_id != old and cur.by == "restock" and not cur.locked
        assert store.get(cur.card_id).voice.model_revision == REVISION  # type: ignore[arg-type]
        assert store.get(cur.card_id).voice.voice == "am_adam"  # type: ignore[arg-type]
        assert _cur(reg, lucy, "en").card_id == locked
        assert _cur(reg, gray, "en").card_id == chosen
    r = invoke("restock", str(ws.root))
    assert r.exit_code == 0 and "nothing to restock" in r.output


def test_restock_follows_two_successive_revision_bumps(tmp_path: Path, monkeypatch):
    """Ruling CD9: a restocked card is restocked again at the next revision bump."""
    from fukikae.backends.tts.kokoro import KokoroSynthesizer

    ws, natsu, store = setup(tmp_path)
    _auto_stock(ws, store, natsu, "am_adam", "rev0")
    for rev in ("rev1", "rev2"):
        monkeypatch.setattr(KokoroSynthesizer, "model_revision", rev)
        r = invoke("restock", str(ws.root))
        assert r.exit_code == 0 and "restocked" in r.output, r.output
        with Registry(ws.registry_path) as reg:
            cur = _cur(reg, natsu, "en")
        assert cur.by == "restock" and store.get(cur.card_id).voice.model_revision == rev  # type: ignore[arg-type]


def test_restock_never_touches_a_locked_or_operator_assigned_card(tmp_path: Path, monkeypatch):
    from fukikae.backends.tts.kokoro import KokoroSynthesizer

    ws, natsu, store = setup(tmp_path)
    with Registry(ws.registry_path) as reg:
        lucy = reg.create_actor_with_character(np.array([0.0, 1.0], dtype=np.float32))[1]
    restocked = _auto_stock(ws, store, natsu, "am_adam", "rev0")
    mine = _auto_stock(ws, store, lucy, "af_heart", "rev0")
    with Registry(ws.registry_path) as reg:
        reg.assign_card(natsu, "en", restocked, by="restock", locked=False)
        reg.lock_card(natsu, "en", True)
        reg.assign_card(lucy, "en", mine, by="assign", locked=False)
    monkeypatch.setattr(KokoroSynthesizer, "model_revision", "rev1")
    r = invoke("restock", str(ws.root))
    assert r.exit_code == 0 and "nothing to restock" in r.output
    with Registry(ws.registry_path) as reg:
        assert _cur(reg, natsu, "en").card_id == restocked
        assert _cur(reg, lucy, "en").card_id == mine


def test_show_warns_when_references_may_carry_emotion(tmp_path: Path):
    """CD5: the voices output repeats the clone card's reference-emotion warning."""
    ws, cid, store = setup(tmp_path)
    card = parse_card(STOCK_CARD)
    from dataclasses import replace

    card_id = store.put(replace(card, meta={"reference_emotions": ["angry", "sad"]}))
    with Registry(ws.registry_path) as reg:
        reg.assign_card(cid, "en", card_id, by="assign", locked=True)
    r = invoke("show", str(ws.root), "Natsu")
    assert r.exit_code == 0, r.output
    assert "references may carry emotion: angry, sad" in r.output
