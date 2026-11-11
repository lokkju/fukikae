"""Voice packs: export, read (untrusted input), series check, import planning and applying."""

from __future__ import annotations

import io
import os
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zstandard
from scipy.io import wavfile
from typer.testing import CliRunner

from fukikae.cards import packs
from fukikae.cards.card import Card, Fingerprint, Reference, Voice, card_id, dump_card, parse_card
from fukikae.cards.packs import (
    CharacterKey,
    PackError,
    PackLimits,
    apply_import,
    export_pack,
    normalize_name,
    plan_import,
    read_pack,
    series_check,
)
from fukikae.cards.store import CardStore
from fukikae.cli import app
from fukikae.config import SeriesInfo
from fukikae.registry.store import Registry

runner = CliRunner()
SERIES = SeriesInfo("Fairy Tail", "anime", {"anidb": 6662})


# -- helpers --------------------------------------------------------------------------------


def _wav(path: Path, seconds: float = 1.0, freq: float = 220.0, rate: int = 8000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    wavfile.write(path, rate, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32))
    return path


def designed(voice: str = "af_heart", lang: str = "en") -> Card:
    """A designed voice, distinct per ``voice`` (no stock voice: imports refuse those, D41)."""
    return Card(lang, Voice(backend="fake", model_revision="1.0", source="design",
                            description=f"a voice like {voice}"))  # fmt: skip


def clone(store: CardStore, tmp: Path, freq: float = 220.0, seconds: float = 1.0) -> Card:
    clip = store.add_sidecar(_wav(tmp / f"clip{freq}.wav", seconds, freq), "clip")
    ref = Reference(clip=clip, transcript="なんだよ", language="ja")
    return Card(
        "en", Voice(backend="fake", model_revision="1.0", source="clone", references=(ref,))
    )


def with_fingerprint(store: CardStore, tmp: Path, card: Card) -> Card:
    path = tmp / "fp.npy"
    np.save(path, np.ones(4, dtype=np.float32) / 2)
    ref = store.add_sidecar(path, "vector")
    return Card(
        card.target_language,
        Voice(**{**card.voice.__dict__, "fingerprint": Fingerprint("fake:1", ref, 0.85)}),
        card.style,
        card.meta,
    )


FT_TOML = '[series]\ntitle = "Fairy Tail"\nanidb = 6662\n'


def workspace(tmp_path: Path, name: str, series: str = FT_TOML) -> tuple[Path, Registry, CardStore]:
    root = tmp_path / name
    root.mkdir()
    (root / "fukikae.toml").write_text(series)
    reg = Registry(root / ".fukikae" / "registry" / "registry.sqlite")
    return root, reg, CardStore(root)


def character(reg: Registry, name: str | None, external_id: str | None = None) -> int:
    _, cid = reg.create_actor_with_character(np.array([1.0, 0.0], dtype=np.float32))
    if name is not None:
        reg.rename_character(cid, name)
    if external_id is not None:
        reg.set_character_external_id(cid, external_id)
    return cid


def snapshot(root: Path) -> dict[str, bytes]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and "registry" not in p.parts:
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def unchanged_but_orphans(before: dict[str, bytes], after: dict[str, bytes]) -> bool:
    """Every file in ``before`` is byte-identical in ``after``, and the only new files are
    content-addressed card-store files a rolled-back import left (A6: harmless orphans)."""
    new = set(after) - set(before)
    return all(after.get(k) == v for k, v in before.items()) and all(
        k.startswith(".fukikae/cards/") for k in new
    )


def history(reg: Registry) -> list[tuple]:
    return [
        (c.id, lang, h.card_id, h.by, h.locked)
        for c in reg.characters()
        for lang in ("en", "es")
        for h in reg.card_history(c.id, lang)
    ]


def tar_zst(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
    path.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    return path


def reg_member(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def good_pack_members(tmp_path: Path) -> list[tuple[tarfile.TarInfo, bytes | None]]:
    """The members of a valid exported pack, read back from a real export."""
    root, reg, store = workspace(tmp_path, "src-members")
    cid = character(reg, "Natsu")
    reg.assign_card(cid, "en", store.put(designed()), by="auto", locked=False)
    out = tmp_path / "good-dir"
    export_pack(reg, store, SERIES, out, characters=None, languages=None, no_clips=False,
                title="FT voices", author="lokkju")  # fmt: skip
    reg.close()
    return [
        reg_member(str(p.relative_to(out)), p.read_bytes())
        for p in sorted(out.rglob("*"))
        if p.is_file()
    ]


# -- round trip -----------------------------------------------------------------------------


def test_export_import_round_trip_between_registries(tmp_path: Path):
    src, reg_a, store_a = workspace(tmp_path, "a")
    natsu = character(reg_a, "Natsu")
    lucy = character(reg_a, "Lucy", external_id="1234")
    unnamed = character(reg_a, None)  # unnamed: left out, listed
    s = store_a.put(designed())
    reg_a.assign_card(unnamed, "en", s, by="auto", locked=False)
    c = store_a.put(with_fingerprint(store_a, tmp_path, clone(store_a, tmp_path)))
    reg_a.assign_card(natsu, "en", s, by="assign", locked=True)
    reg_a.assign_card(lucy, "en", c, by="auto", locked=False)
    pack_path = tmp_path / "ft.tar.zst"
    report = export_pack(reg_a, store_a, SERIES, pack_path, characters=None, languages=None,
                         no_clips=False, title="FT voices", author="lokkju")  # fmt: skip
    assert report.written == 2
    assert any("unnamed" in why for _, why in report.left_out)

    dst, reg_b, store_b = workspace(tmp_path, "b")
    b_natsu = character(reg_b, "ＮＡＴＳＵ")  # width and case differ
    b_lucy = character(reg_b, "Lucy Heartfilia", external_id="1234")  # name differs, id matches
    with read_pack(pack_path) as pack:
        assert pack.meta.title == "FT voices" and pack.meta.ids == {"anidb": "6662"}
        plan = plan_import(pack, reg_b, replace=False)
        assert {(m.character_id, m.card_id, m.matched_by) for m in plan.assign} == {
            (b_natsu, s, "name"),
            (b_lucy, c, "external id"),
        }
        assert not plan.unmatched and not plan.skipped
        apply_import(plan, pack, reg_b, store_b)
    for cid, card in ((b_natsu, s), (b_lucy, c)):
        cur = reg_b.card_for(cid, "en")
        assert cur is not None and cur.card_id == card
        assert cur.by == "import:FT voices" and not cur.locked
        assert store_b.get(card) == store_a.get(card)
        assert not store_b.missing_sidecars(store_b.get(card))


def test_external_id_beats_name(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "Gray", external_id="77"), "en", store_a.put(designed()),
                      by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, _ = workspace(tmp_path, "b")
    character(reg_b, "Gray")  # same name, no external id
    by_id = character(reg_b, "Gray Fullbuster", external_id="77")
    with read_pack(tmp_path / "p") as pack:
        plan = plan_import(pack, reg_b, replace=False)
    assert [(m.character_id, m.matched_by) for m in plan.assign] == [(by_id, "external id")]


def test_name_does_not_match_a_character_with_another_external_id(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "Gray", external_id="77"), "en", store_a.put(designed()),
                      by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, _ = workspace(tmp_path, "b")
    character(reg_b, "Gray", external_id="99")
    with read_pack(tmp_path / "p") as pack:
        plan = plan_import(pack, reg_b, replace=False)
    assert not plan.assign and [u.key for u in plan.unmatched] == [CharacterKey("Gray", "77")]


def test_width_and_case_insensitive_names():
    assert normalize_name("ＬＵＣＹ") == normalize_name("lucy")
    assert normalize_name("  Lucy \t Heartfilia ") == "lucy heartfilia"
    assert normalize_name("Lu​cy") == "lucy"  # format characters don't count


def test_locked_character_skipped_without_replace(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "Natsu"), "en", store_a.put(designed("am_adam")),
                      by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, store_b = workspace(tmp_path, "b")
    locked = character(reg_b, "Natsu")
    mine = store_b.put(designed("af_heart"))
    reg_b.assign_card(locked, "en", mine, by="auto", locked=True)
    with read_pack(tmp_path / "p") as pack:
        plan = plan_import(pack, reg_b, replace=False)
        assert not plan.assign and [(s.character_id) for s in plan.skipped] == [locked]
        assert "--replace" in plan.skipped[0].reason
        apply_import(plan, pack, reg_b, store_b)
        cur = reg_b.card_for(locked, "en")
        assert cur is not None and cur.card_id == mine
        plan = plan_import(pack, reg_b, replace=True)
        assert [m.character_id for m in plan.assign] == [locked]
        apply_import(plan, pack, reg_b, store_b)
    cur = reg_b.card_for(locked, "en")
    assert cur is not None and cur.card_id != mine and cur.by == "import:t"


def test_operator_chosen_skipped_but_auto_and_earlier_import_replaced(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    for name in ("A", "B", "C"):
        reg_a.assign_card(character(reg_a, name), "en", store_a.put(designed("am_adam")),
                          by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, store_b = workspace(tmp_path, "b")
    mine = store_b.put(designed("af_heart"))
    a, b, c = (character(reg_b, n) for n in ("A", "B", "C"))
    reg_b.assign_card(a, "en", mine, by="assign", locked=False)  # operator-chosen, unlocked
    reg_b.assign_card(b, "en", mine, by="auto", locked=False)
    reg_b.assign_card(c, "en", mine, by="import:older", locked=False)  # CD3: replaceable
    with read_pack(tmp_path / "p") as pack:
        plan = plan_import(pack, reg_b, replace=False)
    assert {m.character_id for m in plan.assign} == {b, c}
    assert [s.character_id for s in plan.skipped] == [a]


def test_same_card_already_assigned_is_unchanged(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    s = store_a.put(designed())
    reg_a.assign_card(character(reg_a, "A"), "en", s, by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, store_b = workspace(tmp_path, "b")
    a = character(reg_b, "A")
    reg_b.assign_card(a, "en", store_b.put(designed()), by="assign", locked=True)
    with read_pack(tmp_path / "p") as pack:
        plan = plan_import(pack, reg_b, replace=False)
    assert not plan.assign and not plan.skipped and [u.character_id for u in plan.unchanged] == [a]


def test_conflicting_series_ids_refused_unless_forced(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "A"), "en", store_a.put(designed()), by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    with read_pack(tmp_path / "p") as pack:
        ok, why = series_check(pack, SeriesInfo("Fairy Tail", "anime", {"anidb": 9999}))
        assert not ok and "anidb" in why and "6662" in why and "9999" in why
        ok, why = series_check(pack, SeriesInfo("Something else", "anime", {"anidb": "6662"}))
        assert ok  # ids agree: the title doesn't matter

    # CLI: refused, then --force-series imports
    dst, reg_b, _ = workspace(tmp_path, "b", '[series]\ntitle = "Fairy Tail"\nanidb = 1\n')
    b = character(reg_b, "A")
    reg_b.close()
    r = runner.invoke(app, ["voices", "import", str(dst), str(tmp_path / "p")])
    assert r.exit_code == 1 and "--force-series" in r.output
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        assert reg.card_for(b, "en") is None
    r = runner.invoke(app, ["voices", "import", str(dst), str(tmp_path / "p"), "--force-series"])
    assert r.exit_code == 0, r.output
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        cur = reg.card_for(b, "en")
        assert cur is not None and cur.by == "import:t"


def test_title_match_warns_when_ids_absent(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "A"), "en", store_a.put(designed()), by="auto", locked=False)
    export_pack(reg_a, store_a, SeriesInfo("Fairy Tail", "anime", {}), tmp_path / "p",
                characters=None, languages=None, no_clips=False, title="t", author="")  # fmt: skip
    with read_pack(tmp_path / "p") as pack:
        ok, why = series_check(pack, SeriesInfo("ＦＡＩＲＹ  tail", "anime", {"anidb": 6662}))
        assert ok and "title" in why
        ok, why = series_check(pack, SeriesInfo("Bleach", "anime", {}))
        assert not ok and "Bleach" in why


def test_no_clips_lists_clone_cards_left_out(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "A"), "en", store_a.put(designed()), by="auto", locked=False)
    c = store_a.put(clone(store_a, tmp_path))
    reg_a.assign_card(character(reg_a, "B"), "en", c, by="auto", locked=False)
    report = export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                         no_clips=True, title="t", author="")  # fmt: skip
    assert report.written == 1
    assert [what for what, _ in report.left_out] == [f"B en card {c}"]
    assert "clips" in report.left_out[0][1]
    assert not (tmp_path / "p" / "clips").exists()
    assert not (tmp_path / "p" / "cards" / f"{c}.toml").exists()


def test_export_refuses_a_clip_over_30_s(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    c = store_a.put(clone(store_a, tmp_path, seconds=31.0))
    reg_a.assign_card(character(reg_a, "B"), "en", c, by="auto", locked=False)
    with pytest.raises(PackError, match="30"):
        export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                    no_clips=False, title="t", author="")  # fmt: skip
    assert not (tmp_path / "p").exists()


def test_export_filters_characters_and_languages(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    a, b = character(reg_a, "A"), character(reg_a, "B")
    for cid in (a, b):
        reg_a.assign_card(cid, "en", store_a.put(designed()), by="auto", locked=False)
        reg_a.assign_card(cid, "es", store_a.put(designed(lang="es")), by="auto", locked=False)
    report = export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=[a],
                         languages=["es"], no_clips=False, title="t", author="")  # fmt: skip
    assert report.written == 1
    with read_pack(tmp_path / "p") as pack:
        assert [(e.key.name, e.target_language) for e in pack.cast] == [("A", "es")]


def test_pack_never_contains_registry_ids(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    for _ in range(5):
        character(reg_a, None)  # push the ids up so they'd be visible
    cid = character(reg_a, "Natsu", external_id="555")
    reg_a.assign_card(cid, "en", store_a.put(designed()), by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    text = "\n".join(p.read_text() for p in (tmp_path / "p").rglob("*.toml"))
    assert "character_id" not in text and "history" not in text
    cast = (tmp_path / "p" / "cast.toml").read_text()
    assert "Natsu" in cast and "555" in cast and str(cid) not in cast.replace("555", "")


def test_tar_zst_round_trip_is_reproducible(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "A"), "en", store_a.put(clone(store_a, tmp_path)),
                      by="auto", locked=False)  # fmt: skip
    for name in ("one.tar.zst", "two.tar.zst"):
        export_pack(reg_a, store_a, SERIES, tmp_path / name, characters=None, languages=None,
                    no_clips=False, title="t", author="")  # fmt: skip
    assert (tmp_path / "one.tar.zst").read_bytes() == (tmp_path / "two.tar.zst").read_bytes()
    with read_pack(tmp_path / "one.tar.zst") as pack:
        assert len(pack.cards) == 1 and len(pack.sidecars) == 1
        staging = pack.root
    assert not staging.exists()  # the staging copy is removed on close


def test_blended_card_travels_with_its_blend(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    base = store_a.put(designed())
    voice = Voice(backend="fake", model_revision="1.0", source="design",
                  description="a voice like am_adam",
                  blend={"with": f"card:{base}", "weight": 0.3})  # fmt: skip
    blended = Card("en", voice)
    bid = store_a.put(blended)
    reg_a.assign_card(character(reg_a, "A"), "en", bid, by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "A")
    with read_pack(tmp_path / "p") as pack:
        apply_import(plan_import(pack, reg_b, replace=False), pack, reg_b, store_b)
    assert store_b.has(bid) and store_b.has(base)


# -- untrusted packs: refused, nothing written ----------------------------------------------


def _import_refused(tmp_path: Path, pack_path: Path, match: str) -> None:
    """The CLI refuses ``pack_path`` naming ``match``, and the workspace is byte-identical."""
    dst = tmp_path / "victim"
    if not dst.exists():
        dst.mkdir()
        (dst / "fukikae.toml").write_text("")
        with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
            character(reg, "Natsu")
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        before_hist = history(reg)
    before = snapshot(dst)
    r = runner.invoke(app, ["voices", "import", str(dst), str(pack_path)])
    assert r.exit_code == 1, r.output
    assert match in r.output, r.output
    assert snapshot(dst) == before
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        assert history(reg) == before_hist


def test_tar_zst_rejects_path_traversal(tmp_path: Path):
    members = good_pack_members(tmp_path)
    evil = tmp_path / "evil.tar.zst"
    tar_zst(evil, [*members, reg_member("../escaped.txt", b"pwned")])
    _import_refused(tmp_path, evil, "..")
    assert not (tmp_path / "escaped.txt").exists()
    tar_zst(evil, [*members, reg_member("cards/../../escaped.txt", b"pwned")])
    _import_refused(tmp_path, evil, "..")


def test_tar_zst_rejects_absolute_paths(tmp_path: Path):
    target = tmp_path / "abs-target.txt"
    tar_zst(
        tmp_path / "evil.tar.zst", [*good_pack_members(tmp_path), reg_member(str(target), b"x")]
    )
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "absolute")
    assert not target.exists()


def test_tar_zst_rejects_links_and_devices(tmp_path: Path):
    members = good_pack_members(tmp_path)
    for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE):
        info = tarfile.TarInfo("clips/" + "a" * 64 + ".wav")
        info.type = kind
        info.linkname = "/etc/passwd"
        tar_zst(tmp_path / "evil.tar.zst", [*members, (info, None)])
        _import_refused(tmp_path, tmp_path / "evil.tar.zst", "not a regular file")


def test_directory_pack_rejects_symlinks(tmp_path: Path):
    members = good_pack_members(tmp_path)
    d = tmp_path / "evil-dir"
    for info, data in members:
        (d / info.name).parent.mkdir(parents=True, exist_ok=True)
        (d / info.name).write_bytes(data or b"")
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"secret")
    (d / "clips").mkdir(exist_ok=True)
    os.symlink(secret, d / "clips" / ("b" * 64 + ".wav"))
    _import_refused(tmp_path, d, "symlink")
    (d / "clips" / ("b" * 64 + ".wav")).unlink()
    cards = d / "cards"
    real = tmp_path / "real-cards"
    cards.rename(real)
    os.symlink(real, cards)  # a symlinked directory too
    _import_refused(tmp_path, d, "symlink")


def test_sidecar_hash_mismatch_refused(tmp_path: Path):
    src, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "Natsu"), "en", store_a.put(clone(store_a, tmp_path)),
                      by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    clip = next((tmp_path / "p" / "clips").iterdir())
    data = bytearray(clip.read_bytes())
    data[-1] ^= 0xFF
    clip.write_bytes(bytes(data))
    _import_refused(tmp_path, tmp_path / "p", "hash")


def test_card_id_mismatch_refused(tmp_path: Path):
    members = good_pack_members(tmp_path)
    tampered = []
    for info, data in members:
        if info.name.startswith("cards/"):
            text = (data or b"").decode().replace("af_heart", "am_adam")
            tampered.append(reg_member(info.name, text.encode()))
        else:
            tampered.append((info, data))
    tar_zst(tmp_path / "evil.tar.zst", tampered)
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "holds card")


def test_bad_card_refused(tmp_path: Path):
    members = good_pack_members(tmp_path)
    bad = []
    for info, data in members:
        if info.name.startswith("cards/"):
            text = (data or b"").decode() + '\n[style]\nregister = "' + "x" * 400 + '"\n'
            bad.append(reg_member(info.name, text.encode()))
        else:
            bad.append((info, data))
    tar_zst(tmp_path / "evil.tar.zst", bad)
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "300")


def test_unknown_card_key_refused(tmp_path: Path):
    members = good_pack_members(tmp_path)
    bad = []
    for info, data in members:
        if info.name.startswith("cards/"):
            text = (data or b"").decode().replace("[voice]\n", '[voice]\nexec = "rm -rf /"\n')
            bad.append(reg_member(info.name, text.encode()))
        else:
            bad.append((info, data))
    tar_zst(tmp_path / "evil.tar.zst", bad)
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "unknown key")


def test_oversized_member_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(packs, "DEFAULT_LIMITS", PackLimits(max_member=4096))
    members = good_pack_members(tmp_path)
    big = reg_member("clips/" + "c" * 64 + ".wav", b"\0" * 1_000_000)  # compresses to nothing
    tar_zst(tmp_path / "evil.tar.zst", [*members, big])
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "too large")


def test_total_size_limit(tmp_path: Path):
    members = good_pack_members(tmp_path)
    tar_zst(tmp_path / "p.tar.zst", members)
    total = sum(len(d or b"") for _, d in members)
    with pytest.raises(PackError, match="total"):
        read_pack(tmp_path / "p.tar.zst", limits=PackLimits(max_total=total - 1))


def test_too_many_files_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(packs, "DEFAULT_LIMITS", PackLimits(max_files=3))
    members = good_pack_members(tmp_path)
    assert len(members) == 3
    extra = reg_member("cards/" + "0" * 16 + ".toml", b"")
    tar_zst(tmp_path / "evil.tar.zst", [*members, extra])
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "files")


def test_unexpected_and_duplicate_members_refused(tmp_path: Path):
    members = good_pack_members(tmp_path)
    tar_zst(tmp_path / "evil.tar.zst", [*members, reg_member("run.sh", b"#!/bin/sh\n")])
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "unexpected")
    cast = next(m for m in members if m[0].name == "cast.toml")
    tar_zst(tmp_path / "evil.tar.zst", [*members, reg_member("cast.toml", cast[1] or b"")])
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "twice")


def test_newer_format_refused(tmp_path: Path):
    members = [
        reg_member(i.name, (d or b"").replace(b"format = 1", b"format = 2"))
        if i.name == "pack.toml" else (i, d)
        for i, d in good_pack_members(tmp_path)
    ]  # fmt: skip
    tar_zst(tmp_path / "evil.tar.zst", members)
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "newer")


def test_control_characters_in_names_never_reach_the_terminal(tmp_path: Path):
    members = []
    for info, data in good_pack_members(tmp_path):
        if info.name == "cast.toml":
            data = (data or b"").replace(b'name = "Natsu"', b'name = "Nobody\\u001b[2J\\u0007"')
            members.append(reg_member("cast.toml", data))
        else:
            members.append((info, data))
    tar_zst(tmp_path / "odd.tar.zst", members)
    dst, reg_b, _ = workspace(tmp_path, "b")
    reg_b.close()
    r = runner.invoke(app, ["voices", "import", str(dst), str(tmp_path / "odd.tar.zst"),
                            "--dry-run"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "\x1b" not in r.output and "\x07" not in r.output and "Nobody[2J" in r.output


def test_failed_apply_leaves_store_and_registry_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, reg_a, store_a = workspace(tmp_path, "a")
    for name in ("A", "B"):
        reg_a.assign_card(character(reg_a, name), "en",
                          store_a.put(clone(store_a, tmp_path, freq=200 + len(name) * ord(name))),
                          by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    dst, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "A")
    character(reg_b, "B")
    before = snapshot(dst)
    calls = []
    real = Registry.assign_card

    def failing(self, *a, **kw):
        calls.append(a)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real(self, *a, **kw)

    monkeypatch.setattr(Registry, "assign_card", failing)
    with read_pack(tmp_path / "p") as pack:
        plan = plan_import(pack, reg_b, replace=False)
        with pytest.raises(RuntimeError):
            apply_import(plan, pack, reg_b, store_b)
    # A6: the registry is unchanged; content-addressed cards and sidecars the failed import
    # copied stay as harmless orphans (another import may already be using them)
    assert unchanged_but_orphans(before, snapshot(dst))
    assert history(reg_b) == []


def test_cli_import_prints_mapping_and_unmatched(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    for name in ("Natsu", "Happy"):
        reg_a.assign_card(character(reg_a, name), "en", store_a.put(designed()), by="auto",
                          locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p.tar.zst", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    dst, reg_b, _ = workspace(tmp_path, "b")
    natsu = character(reg_b, "natsu")
    reg_b.close()
    r = runner.invoke(app, ["voices", "import", str(dst), str(tmp_path / "p.tar.zst"),
                            "--dry-run"])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "Natsu" in r.output and f"character {natsu}" in r.output
    assert "unmatched" in r.output and "Happy" in r.output and "dry run" in r.output
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        assert reg.card_for(natsu, "en") is None


def test_cli_export_then_assign_from_pack(tmp_path: Path):
    src, reg_a, store_a = workspace(tmp_path, "a")
    s = store_a.put(designed("am_adam"))
    reg_a.assign_card(character(reg_a, "Natsu"), "en", s, by="auto", locked=False)
    reg_a.close()
    pack = tmp_path / "p.tar.zst"
    r = runner.invoke(app, ["voices", "export", str(src), "-o", str(pack), "--title", "FT"])
    assert r.exit_code == 0, r.output
    assert "1 card" in r.output
    dst, reg_b, store_b = workspace(tmp_path, "b")
    other = character(reg_b, "Someone else")
    reg_b.close()
    r = runner.invoke(app, ["voices", "assign", str(dst), "Someone else", f"{pack}:natsu"])
    assert r.exit_code == 0, r.output
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        cur = reg.card_for(other, "en")
        assert cur is not None and cur.card_id == s and cur.by == "assign" and cur.locked


def test_dump_card_is_what_the_pack_carries(tmp_path: Path):
    # the exported card file is the store's canonical dump, so its id checks on import
    _, reg_a, store_a = workspace(tmp_path, "a")
    s = store_a.put(designed())
    reg_a.assign_card(character(reg_a, "A"), "en", s, by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    text = (tmp_path / "p" / "cards" / f"{s}.toml").read_text()
    assert text == dump_card(store_a.get(s)) and card_id(parse_card(text)) == s


def test_deeply_nested_toml_refused(tmp_path: Path):
    members = [
        reg_member("cast.toml", b"cast = " + b"[" * 100_000) if i.name == "cast.toml" else (i, d)
        for i, d in good_pack_members(tmp_path)
    ]
    tar_zst(tmp_path / "evil.tar.zst", members)
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "cast.toml")


def _fingerprinted_dir_pack(tmp_path: Path) -> tuple[Path, str]:
    """A directory pack of one fingerprinted stock card; returns it and the card file name."""
    _, reg_a, store_a = workspace(tmp_path, "fp-src")
    cid = store_a.put(with_fingerprint(store_a, tmp_path, designed()))
    reg_a.assign_card(character(reg_a, "Natsu"), "en", cid, by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "fp", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    reg_a.close()
    return tmp_path / "fp", f"{cid}.toml"


def _swap_vector(pack: Path, card_file: str, payload: Path) -> None:
    """Replace the card's fingerprint vector with ``payload``: the fingerprint is outside the
    card id, so the card still checks; only the sidecar checks stand between it and a load."""
    import hashlib

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    old = next((pack / "vectors").iterdir())
    old.unlink()
    (pack / "vectors" / f"{digest}.npy").write_bytes(payload.read_bytes())
    card = pack / "cards" / card_file
    card.write_text(card.read_text().replace(old.stem, digest))


def test_pickled_npy_refused(tmp_path: Path):
    pack, card_file = _fingerprinted_dir_pack(tmp_path)
    evil = tmp_path / "evil.npy"
    np.save(evil, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    _swap_vector(pack, card_file, evil)
    _import_refused(tmp_path, pack, "plain .npy")


def test_npy_header_larger_than_file_refused(tmp_path: Path):
    pack, card_file = _fingerprinted_dir_pack(tmp_path)
    evil = tmp_path / "bomb.npy"
    np.save(evil, np.zeros(4, dtype=np.float32))
    raw = evil.read_bytes()  # magic (6), version (2), header length (2), header, data
    header_len = int.from_bytes(raw[8:10], "little")
    head = raw[10 : 10 + header_len].decode("latin1").replace("(4,)", "(4000000000,)")
    head = head.rstrip(" \n")
    pad = header_len - len(head) - 1
    evil.write_bytes(raw[:10] + (head + " " * pad + "\n").encode("latin1") + raw[10 + header_len :])
    _swap_vector(pack, card_file, evil)
    _import_refused(tmp_path, pack, "plain .npy")


# -- fix round 1: CD22, CD23(d), CD23(e) and the minors -------------------------------------


def _fp(store: CardStore, tmp: Path, card: Card, value: float, name: str = "fake:1") -> Card:
    path = tmp / f"fp{value}.npy"
    np.save(path, np.full(4, value, dtype=np.float32))
    ref = store.add_sidecar(path, "vector")
    voice = Voice(**{**card.voice.__dict__, "fingerprint": Fingerprint(name, ref, 0.85)})
    return Card(card.target_language, voice, card.style, card.meta)


def test_cd22_pack_fingerprint_merges_into_a_stored_card_without_one(tmp_path: Path):
    pack, card_file = _fingerprinted_dir_pack(tmp_path)
    cid = card_file.removesuffix(".toml")
    _, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "Natsu")
    store_b.put(designed())  # the same card, no fingerprint
    with read_pack(pack) as p:
        report = apply_import(plan_import(p, reg_b, replace=False), p, reg_b, store_b)
        pack_fp = p.cards[cid].voice.fingerprint
    assert report.merged == [cid] and report.conflicts == []
    got = store_b.get(cid)  # same id, fingerprint now in the file
    assert got.voice.fingerprint == pack_fp and pack_fp is not None
    assert not store_b.missing_sidecars(got)


def test_cd22_rollback_restores_the_old_card_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pack, card_file = _fingerprinted_dir_pack(tmp_path)
    dst, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "Natsu")
    store_b.put(designed())
    before = snapshot(dst)

    def boom(self, *a, **kw):  # noqa: ARG001
        raise RuntimeError("disk full")

    monkeypatch.setattr(Registry, "assign_card", boom)
    with read_pack(pack) as p, pytest.raises(RuntimeError):
        apply_import(plan_import(p, reg_b, replace=False), p, reg_b, store_b)
    assert unchanged_but_orphans(before, snapshot(dst))


def test_cd22_differing_fingerprints_keep_the_local_one_and_report_both(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    theirs = _fp(store_a, tmp_path, designed(), 0.5)
    cid = store_a.put(theirs)
    reg_a.assign_card(character(reg_a, "Natsu"), "en", cid, by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "Natsu")
    mine = _fp(store_b, tmp_path, designed(), 0.25)
    assert store_b.put(mine) == cid
    with read_pack(tmp_path / "p") as p:
        report = apply_import(plan_import(p, reg_b, replace=False), p, reg_b, store_b)
    assert store_b.get(cid).voice.fingerprint == mine.voice.fingerprint
    assert report.merged == [] and len(report.conflicts) == 1
    line = report.conflicts[0]
    assert cid in line and mine.voice.fingerprint.vector in line  # type: ignore[union-attr]
    assert theirs.voice.fingerprint.vector in line  # type: ignore[union-attr]


def test_cli_import_merges_and_verifies_the_merged_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from fukikae.cards import cli as cards_cli

    verified: list[str] = []
    monkeypatch.setattr(
        cards_cli, "_verify_after",
        lambda _ws, store, cards: verified.extend(
            c for _, c in cards if store.get(c).voice.fingerprint is not None
        ),
    )  # fmt: skip
    pack, card_file = _fingerprinted_dir_pack(tmp_path)
    dst, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "Natsu")
    store_b.put(designed())
    reg_b.close()
    r = runner.invoke(app, ["voices", "import", str(dst), str(pack)])
    assert r.exit_code == 0, r.output
    assert "fingerprint" in r.output and "merged" in r.output
    assert verified == [card_file.removesuffix(".toml")]


def test_protection_has_one_predicate(tmp_path: Path):
    # CD23(d): plan_import asks the registry, the same rule assign_card enforces
    from fukikae.registry.store import ProtectedAssignmentError, replace_refusal

    _, reg, store = workspace(tmp_path, "r")
    cid = character(reg, "A")
    s1, s2 = store.put(designed("af_heart")), store.put(designed("am_adam"))
    for by, locked in (("assign", False), ("auto", True), ("import:x", False), ("auto", False)):
        reg.assign_card(cid, "en", s1, by=by, locked=locked, override=True)
        cur = reg.card_for(cid, "en")
        why = replace_refusal(cur, "import:t")
        try:
            reg.assign_card(cid, "en", s2, by="import:t", locked=False)
            refused = False
        except ProtectedAssignmentError:
            refused = True
        assert refused == (why is not None), (by, locked)
        assert not hasattr(packs, "_protected")


def test_cli_import_reports_a_registry_refusal_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "Natsu"), "en", store_a.put(clone(store_a, tmp_path)),
                      by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    dst, reg_b, store_b = workspace(tmp_path, "b")
    natsu = character(reg_b, "Natsu")
    reg_b.assign_card(natsu, "en", store_b.put(designed()), by="auto", locked=False)
    reg_b.close()
    real = packs.plan_import

    def racing(pack, reg, *, replace):  # the operator locks Natsu between plan and apply
        plan = real(pack, reg, replace=replace)
        reg.lock_card(natsu, "en", True)
        return plan

    monkeypatch.setattr(packs, "plan_import", racing)
    before = snapshot(dst)
    r = runner.invoke(app, ["voices", "import", str(dst), str(tmp_path / "p")])
    assert r.exit_code == 1 and "nothing was imported" in r.output and "Traceback" not in r.output
    assert unchanged_but_orphans(before, snapshot(dst))


def test_escape_in_a_card_key_never_reaches_stdout(tmp_path: Path):
    # CD23(e): CardError messages quote the key; the CLI prints them through printable
    members = []
    for info, data in good_pack_members(tmp_path):
        if info.name.startswith("cards/"):
            text = (data or b"").decode().replace("[voice]\n", '[voice]\n"\\u001b[2J\\u0007" = 1\n')
            members.append(reg_member(info.name, text.encode()))
        else:
            members.append((info, data))
    tar_zst(tmp_path / "evil.tar.zst", members)
    _import_refused(tmp_path, tmp_path / "evil.tar.zst", "unknown key")
    r = runner.invoke(app, ["voices", "import", str(tmp_path / "victim"),
                            str(tmp_path / "evil.tar.zst")])  # fmt: skip
    assert "\x1b" not in r.output and "\x07" not in r.output and "[2J" in r.output


def test_escape_in_a_stored_card_never_reaches_stdout(tmp_path: Path):
    root, reg, store = workspace(tmp_path, "w")
    cid = character(reg, "Natsu")
    sid = store.put(designed())
    reg.assign_card(cid, "en", sid, by="assign", locked=True)
    reg.close()
    path = store.card_path(sid)
    path.write_text(path.read_text() + '\n[style]\nregister = "x\\u001b[31mred"\n')
    for args in (["show", str(root), "Natsu"], ["verify", str(root), "Natsu"]):
        r = runner.invoke(app, ["voices", *args])
        assert "\x1b" not in r.output, (args, r.output)


def test_assign_from_pack_with_a_colon_in_the_name(tmp_path: Path):
    src, reg_a, store_a = workspace(tmp_path, "a")
    s = store_a.put(designed("am_adam"))
    reg_a.assign_card(character(reg_a, "Dr. Who: Origins"), "en", s, by="auto", locked=False)
    reg_a.close()
    pack = tmp_path / "p.tar.zst"
    assert runner.invoke(app, ["voices", "export", str(src), "-o", str(pack)]).exit_code == 0
    dst, reg_b, _ = workspace(tmp_path, "b")
    other = character(reg_b, "Someone")
    reg_b.close()
    r = runner.invoke(app, ["voices", "assign", str(dst), "Someone", f"{pack}:dr. who: origins"])
    assert r.exit_code == 0, r.output
    with Registry(dst / ".fukikae" / "registry" / "registry.sqlite") as reg:
        cur = reg.card_for(other, "en")
        assert cur is not None and cur.card_id == s


def test_assign_from_pack_merges_the_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fukikae.cards import cli as cards_cli

    monkeypatch.setattr(cards_cli, "_verify_after", lambda *_a: None)
    pack, card_file = _fingerprinted_dir_pack(tmp_path)
    dst, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "Someone")
    store_b.put(designed())
    reg_b.close()
    r = runner.invoke(
        app, ["voices", "assign", str(dst), "Someone", f"{pack}:Natsu", "--no-verify"]
    )
    assert r.exit_code == 0, r.output
    assert store_b.get(card_file.removesuffix(".toml")).voice.fingerprint is not None


def test_import_notes_another_target_language(tmp_path: Path):
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "A"), "es", store_a.put(designed(lang="es")), by="auto",
                      locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    dst, reg_b, _ = workspace(tmp_path, "b")
    character(reg_b, "A")
    reg_b.close()
    r = runner.invoke(app, ["voices", "import", str(dst), str(tmp_path / "p"), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "'es'" in r.output and "target language" in r.output and "'en'" in r.output


def test_bomb_refused_before_anything_is_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    members = good_pack_members(tmp_path)
    big = reg_member("clips/" + "c" * 64 + ".wav", b"\0" * 1_000_000)
    tar_zst(tmp_path / "bomb.tar.zst", [*members, big])  # the oversized member comes last
    copied: list[str] = []
    real = packs._copy_bounded

    def spy(src, dest, size, name):
        copied.append(name)
        return real(src, dest, size, name)

    monkeypatch.setattr(packs, "_copy_bounded", spy)
    with pytest.raises(PackError, match="total"):
        read_pack(tmp_path / "bomb.tar.zst", limits=PackLimits(max_total=100_000))
    assert copied == []


def test_directory_entries_count_toward_the_file_limit(tmp_path: Path):
    members = good_pack_members(tmp_path)  # 3 files
    dirs = []
    for d in ("clips", "latents", "vectors"):
        info = tarfile.TarInfo(d)
        info.type = tarfile.DIRTYPE
        dirs.append((info, None))
    tar_zst(tmp_path / "p.tar.zst", [*dirs, *members])
    with pytest.raises(PackError, match="files"):
        read_pack(tmp_path / "p.tar.zst", limits=PackLimits(max_files=4))
    d = tmp_path / "dirpack"
    for info, data in members:
        (d / info.name).parent.mkdir(parents=True, exist_ok=True)
        (d / info.name).write_bytes(data or b"")
    for sub in ("clips", "latents"):
        (d / sub).mkdir()
    with pytest.raises(PackError, match="files"):
        read_pack(d, limits=PackLimits(max_files=4))


@pytest.mark.parametrize("fmt", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_tar_zst_refuses_pax_and_gnu_extended_headers(tmp_path: Path, fmt: int):
    """A3: export writes ustar only; pax (x/g) and GNU (L/K) headers are refused before
    tarfile reads their payload into memory."""
    members = good_pack_members(tmp_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
        long = tarfile.TarInfo("clips/" + "a" * 200 + ".wav")  # needs an extended header
        tf.addfile(long, io.BytesIO(b""))
    evil = tmp_path / "evil.tar.zst"
    evil.write_bytes(zstandard.ZstdCompressor().compress(buf.getvalue()))
    _import_refused(tmp_path, evil, "extended header")


def test_tar_zst_refuses_a_huge_pax_header_without_buffering_it(tmp_path: Path):
    info = tarfile.TarInfo("PaxHeader")
    info.type = tarfile.XHDTYPE
    info.size = 2 * 1024**3  # claims 2 GiB
    raw = info.tobuf(tarfile.USTAR_FORMAT, "utf-8", "surrogateescape") + b"\0" * 4096
    evil = tmp_path / "evil.tar.zst"
    evil.write_bytes(zstandard.ZstdCompressor().compress(raw))
    _import_refused(tmp_path, evil, "extended header")


def _stock_in(lang: str) -> Card:
    return parse_card(
        dump_card(designed()).replace('target_language = "en"', f'target_language = "{lang}"')
    )


def test_export_leaves_out_rows_import_would_refuse(tmp_path: Path):
    """A4: export runs import's validators: per row it leaves out, so the pack it writes
    always imports."""
    _, reg, store = workspace(tmp_path, "a")
    ok = character(reg, "Natsu")
    long_name = character(reg, "N" * 250)
    long_ext = character(reg, "Lucy")
    reg._con.execute("UPDATE characters SET external_id = ? WHERE id = ?", ("9" * 150, long_ext))
    bad_lang = character(reg, "Gray")
    s = store.put(designed())
    for cid in (ok, long_name, long_ext):
        reg.assign_card(cid, "en", s, by="auto", locked=False)
    reg.assign_card(bad_lang, "pt_BR", store.put(_stock_in("pt_BR")), by="auto", locked=False)
    out = tmp_path / "pack"
    report = export_pack(reg, store, SERIES, out, characters=None, languages=None,
                         no_clips=False, title="FT", author="lokkju")  # fmt: skip
    assert report.written == 1
    reasons = " | ".join(why for _, why in report.left_out)
    assert "250 characters" in reasons and "external" in reasons and "pt_BR" in reasons
    with read_pack(out) as pack:
        assert [e.key.name for e in pack.cast] == ["Natsu"]


@pytest.mark.parametrize(
    ("series", "author", "notes", "match"),
    [
        (SeriesInfo("T" * 400, "anime", {}), "a", "", "series_title"),
        (SERIES, "a" * 300, "", "author"),
        (SERIES, "a", "n" * 3000, "notes"),
        (SeriesInfo("Fairy Tail", "anime", {"anidb": "x" * 200}), "a", "", "anidb"),
    ],
)
def test_export_refuses_pack_fields_import_would_refuse(
    tmp_path: Path, series: SeriesInfo, author: str, notes: str, match: str
):
    _, reg, store = workspace(tmp_path, "a")
    reg.assign_card(character(reg, "Natsu"), "en", store.put(designed()), by="auto", locked=False)
    out = tmp_path / "pack"
    with pytest.raises(PackError, match=match):
        export_pack(reg, store, series, out, characters=None, languages=None,
                    no_clips=False, title="FT", author=author, notes=notes)  # fmt: skip
    assert not out.exists()


def test_rollback_never_deletes_content_addressed_files(tmp_path: Path):
    """A6: two imports racing: one copies card X, the other finds X already stored and
    assigns it; the first fails and rolls back. X must still be there."""
    _, reg_a, store_a = workspace(tmp_path, "a")
    reg_a.assign_card(character(reg_a, "A"), "en", store_a.put(clone(store_a, tmp_path)),
                      by="auto", locked=False)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, _, store_b = workspace(tmp_path, "b")
    with read_pack(tmp_path / "p") as pack:
        ids = list(pack.cards)
        report = packs.copy_cards(pack, store_b, ids)
        assert report.created
        report.rollback()
    for cid in ids:
        assert store_b.has(cid)
        assert not store_b.missing_sidecars(store_b.get(cid))


def test_import_warns_on_a_fingerprint_threshold_below_one_half(tmp_path: Path):
    """A11: a low threshold reports almost any render as a match; import says so."""
    src, reg_a, store_a = workspace(tmp_path, "a")
    card = with_fingerprint(store_a, tmp_path, designed())
    assert card.voice.fingerprint is not None
    low = replace(card.voice.fingerprint, threshold=0.3)
    cid = store_a.put(Card(card.target_language, replace(card.voice, fingerprint=low)))
    reg_a.assign_card(character(reg_a, "Natsu"), "en", cid, by="auto", locked=False)
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    dst, reg_b, _ = workspace(tmp_path, "b")
    character(reg_b, "Natsu")
    reg_b.close()
    r = runner.invoke(app, ["voices", "import", str(dst), str(tmp_path / "p"), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert f"card {cid}" in r.output and "threshold 0.3" in r.output and "0.5" in r.output


def test_pack_import_refuses_a_stock_card(tmp_path: Path):
    """D41: a pack may not bring a stock voice into a workspace."""
    from fukikae.cards.card import Card, Voice

    _, reg_a, store_a = workspace(tmp_path, "a")
    stock_voice = Card("en", Voice(backend="fake", model_revision="1.0", source="stock",
                                   voice="af_heart"))  # fmt: skip
    reg_a.assign_card(character(reg_a, "A"), "en", store_a.put(stock_voice), by="assign",
                      locked=True)  # fmt: skip
    export_pack(reg_a, store_a, SERIES, tmp_path / "p", characters=None, languages=None,
                no_clips=False, title="t", author="")  # fmt: skip
    _, reg_b, store_b = workspace(tmp_path, "b")
    character(reg_b, "A")
    with read_pack(tmp_path / "p") as pack, pytest.raises(PackError, match=r"D41"):
        apply_import(plan_import(pack, reg_b, replace=False), pack, reg_b, store_b)
    assert not list(store_b.root.glob("*.toml"))
