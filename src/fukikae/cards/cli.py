"""`fukikae voices` commands: which card each character speaks with, per target language."""

from __future__ import annotations

import datetime as dt
import os
import shlex
import sqlite3
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from fukikae.cards.card import (
    MODE_FIELDS,
    Card,
    CardError,
    Voice,
    card_id,
    dump_card,
    is_card_id,
    parse_card,
)
from fukikae.cards.store import CardStore, StaleCardError, StrippedCardError
from fukikae.workspace import Workspace

if TYPE_CHECKING:
    from fukikae.backends.base import Embedder, Synthesizer
    from fukikae.cards.packs import CastEntry, CopyReport
    from fukikae.cards.verify import SynthSource, VerifyResult
    from fukikae.config import Config
    from fukikae.registry.store import CardAssignment, Character, Registry

_DEFAULTS = Voice("", "", "")

voices_app = typer.Typer(
    help="Character cards: assign, swap, edit and derive each character's voice.",
    no_args_is_help=True,
)

_WS = typer.Argument(..., help="Workspace directory.")
_CHAR = typer.Argument(..., metavar="CHARACTER", help="Character id or name.")
_LANG = typer.Option(
    None, "--lang", help="Target language (default: the workspace's target_language)."
)


def _terminal_safe(text: str) -> str:
    """``text`` without control or format characters (tabs and line breaks kept): card and
    pack text, error messages that quote it included, reaches the terminal only through this
    (ruling CD23(e))."""
    return "\n".join(
        "".join(ch for ch in line if ch == "\t" or unicodedata.category(ch)[0] != "C")
        for line in text.split("\n")
    )


def _echo(message: str = "", nl: bool = True, err: bool = False) -> None:
    typer.echo(_terminal_safe(message), nl=nl, err=err)


def _fail(msg: str) -> typer.Exit:
    _echo(f"error: {msg}", err=True)
    return typer.Exit(1)


@contextmanager
def _session(workspace: Path, lang: str | None) -> Iterator[tuple[Registry, CardStore, str]]:
    """The workspace's registry, card store and target language. Input and card errors, a
    known backend or render failure (``user_errors``: a TTS worker that crashed, a busy GPU),
    a registry sqlite can't use or a newer fukikae wrote, and OS errors (a full disk, a
    permission) exit 1 with one line (A10). click's own exits (typer.Exit, typer.Abort) pass
    through, and any other RuntimeError is a bug and keeps its traceback."""
    import sqlite3

    import click

    from fukikae.config import check_language
    from fukikae.errors import user_errors
    from fukikae.registry.store import Registry, RegistrySchemaError

    ws = Workspace(workspace)
    try:
        language = check_language(lang, "--lang") if lang else ws.config.target_language
    except user_errors() as e:
        raise _fail(str(e)) from e
    try:
        with Registry(ws.registry_path) as reg:
            yield reg, CardStore(ws.root), language
    except (click.exceptions.Exit, click.exceptions.Abort):  # both are RuntimeErrors
        raise
    except KeyError as e:
        raise _fail(f"no character {e.args[0]}") from e
    except sqlite3.DatabaseError as e:
        raise _fail(f"registry {ws.registry_path}: {e}") from e
    except (*user_errors(), RegistrySchemaError, OSError) as e:
        raise _fail(str(e) or type(e).__name__) from e


def _label(c: Character) -> str:
    return f"{c.id} ({c.name})" if c.name else str(c.id)


def _state(a: CardAssignment) -> str:
    return "locked" if a.locked else "unlocked"


def _current(reg: Registry, char: Character, lang: str) -> CardAssignment:
    cur = reg.card_for(char.id, lang)
    if cur is None or cur.card_id is None:
        raise _fail(
            f"character {_label(char)} has no card for {lang!r}; "
            "assign one with `fukikae voices assign`"
        )
    return cur


@voices_app.command("list")
def list_(workspace: Path = _WS, lang: str | None = _LANG) -> None:
    """List every character with its card for the target language, then each flagged card,
    one line per card with every reason (``cards.auto.card_status_lines``): cloned from turns
    the subtitle names now give someone else (D29 #4), under the clone floor (2 clips, 10 s;
    D32) or a thin-reference clone (A2), references the reference gates refuse (A1); each
    says whether the next tts run re-picks an automatic card or the card stays (a curated,
    imported or locked one stays until `voices clone`)."""
    from fukikae.cards import auto

    with _session(workspace, lang) as (reg, store, language):
        _echo(f"character\tname\t{language} card\tby\tstate")
        chars = reg.characters()
        for c in chars:
            a = reg.card_for(c.id, language)
            if a is None:
                _echo(f"{c.id}\t{c.name or '-'}\t-\t-\t-")
            else:
                _echo(f"{c.id}\t{c.name or '-'}\t{a.card_id}\t{a.by}\t{_state(a)}")
        notes: list[str] = []
        ws = Workspace(workspace)
        ids = [c.id for c in chars]
        for line in auto.card_status_lines(ws, reg, store, ids, language, notes=notes):
            _echo(line)
        for note in notes:
            _echo(note)


def _modes(card: Card) -> list[tuple[str, str, tuple[str, ...], bool]]:
    """(field, value, allowed values, at its default) of the card's own backend's clone mode."""
    return [
        (key, getattr(card.voice, key), modes, getattr(card.voice, key) == getattr(_DEFAULTS, key))
        for key, modes in MODE_FIELDS.items()
        if key.startswith(card.voice.backend + "_")
    ]


@voices_app.command("show")
def show(workspace: Path = _WS, character: str = _CHAR, lang: str | None = _LANG) -> None:
    """Show a character's current card for the target language, then the card itself."""
    with _session(workspace, lang) as (reg, store, language):
        char = reg.character_by_ref(character)
        others = [x for x in reg.card_languages(char.id) if x != language]
        _echo(
            f"character {_label(char)}"
            + (f", external id {char.external_id}" if char.external_id else "")
            + (f"; cards also for {', '.join(others)}" if others else "")
        )
        a = reg.card_for(char.id, language)
        if a is None or a.card_id is None:
            _echo(f"{language}: no card")
            return
        _echo(f"{language}: card {a.card_id} ({a.by}, {_state(a)}, history row {a.n})")
        if a.note:
            _echo(f"note: {a.note}")
        card = store.get(a.card_id)
        emotions = card.meta.get("reference_emotions")
        if emotions:  # ruling CD5
            _echo(f"references may carry emotion: {', '.join(map(str, emotions))}")
        for _key, mode, modes, is_default in _modes(card):  # ruling CD21
            _echo(f"clone mode: {mode}" + (f" (default; {', '.join(modes)})" if is_default else ""))
        _echo("")
        _echo(dump_card(card), nl=False)


@voices_app.command("assign")
def assign(
    workspace: Path = _WS,
    character: str = _CHAR,
    ref: str = typer.Argument(
        ...,
        metavar="CARD",
        help="A card id in the workspace's store, or one character's card from a pack "
        "(<pack>:<character name or external id>). Stock voices are retired (D41).",
    ),
    no_lock: bool = typer.Option(False, "--no-lock", help="Leave the assignment unlocked."),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Don't verify a fingerprinted card after assigning it."
    ),
    lang: str | None = _LANG,
) -> None:
    """Give a character a card for the target language (locked unless --no-lock). A card with
    a fingerprint is then verified; a mismatch is reported and the card stays assigned."""
    from fukikae.cards.packs import PackError

    with _session(workspace, lang) as (reg, store, language):
        char = reg.character_by_ref(character)
        copied: CopyReport | None = None
        try:
            if ref.startswith("stock:"):
                raise _fail(STOCK_RETIRED)
            if not is_card_id(ref) and ":" in ref:
                ref, copied = _from_pack(workspace, store, ref, language)
            if not store.has(ref):
                raise _fail(
                    f"card {ref} is not in the card store ({store.root}); nothing was assigned"
                )
            try:
                card = store.get(ref)
            except StaleCardError as e:
                # stored before the id rules changed: store it under today's id and assign that
                ref = store.put(e.card)
                _echo(f"card {e.card_id} was stored under the old id rules; it is card {ref} now")
                card = store.get(ref)
            _refuse_stock(card, f"card {ref}")
            if card.target_language != language:
                raise _fail(
                    f"card {ref} is for target language {card.target_language!r}, not "
                    f"{language!r}; derive a {language!r} card with `fukikae voices derive "
                    f"<workspace> <character> --lang {card.target_language} --to {language}`"
                )
            try:
                _check_assignable(store, card, ref, _label(char), language)
            except CardError as e:
                raise _fail(f"{e}; nothing was assigned") from e
            a = reg.assign_card(char.id, language, ref, by="assign", locked=not no_lock)
        except BaseException as e:
            if copied is not None:
                copied.rollback()
            if isinstance(e, PackError):
                raise _fail(str(e)) from e
            raise
        _echo(f"{_label(char)} {language}: card {ref} ({_state(a)}, history row {a.n})")
        if copied is not None:
            _copy_lines(copied)
        if not no_verify:
            _verify_after(workspace, store, [(_label(char), ref)])


CLIP_SECONDS = (3.0, 30.0)
STOCK_RETIRED = (
    "stock voices are retired (D41); `fukikae voices clone` or `fukikae studio` makes a clone"
)


def _refuse_stock(card: Card, what: str) -> None:
    """Stop with `STOCK_RETIRED` when ``card`` is a stock voice: no operator path assigns
    one (D41)."""
    if card.voice.source == "stock":
        raise _fail(f"{what} is a stock voice; nothing changed. {STOCK_RETIRED}")


def _check_clone_card(store: CardStore, card_id: str, who: str, language: str) -> None:
    """The tts stage's own card check (backend, model revision, options, sidecars), so a card
    the stage would refuse is never assigned locked; raises CardError."""
    from fukikae.stages.tts import check_card

    check_card(store, card_id, who, language)


def _check_assignable(store: CardStore, card: Card, card_id: str, who: str, language: str) -> None:
    """What assign and edit run before assigning (A8): the tts stage's card check, and every
    sidecar the card names (a fingerprint vector included) present in the store; raises
    CardError."""
    from fukikae.stages.tts import check_loaded_card

    check_loaded_card(store, card, card_id, who, language)
    missing = store.missing_sidecars(card)
    if missing:
        raise CardError(
            f"{who}: card {card_id} names sidecar(s) missing from {store.root}: "
            f"{', '.join(missing)}; import the pack with them or re-record the fingerprint"
        )


@voices_app.command("clone")
def clone(
    workspace: Path = _WS,
    character: str = _CHAR,
    clip: Path = typer.Option(
        ..., "--clip", help="A wav of the character alone, 3 to 30 s (any rate or channels)."
    ),
    transcript: str = typer.Option(..., "--transcript", help="What the clip says."),
    backend: str | None = typer.Option(
        None, "--backend", help="The cloning backend; default [tts] clone_backend."
    ),
    clip_language: str | None = typer.Option(
        None, "--clip-language", help="The clip's language; default the workspace's source."
    ),
    lang: str | None = _LANG,
) -> None:
    """Clone a character's voice from a clip by hand, for one the automatic references got
    wrong (a merged cluster, say): the clip is copied into the card store as a 24 kHz mono
    reference, and the clone card is assigned locked."""
    from fukikae.backends import loader
    from fukikae.cards.auto import clone_backend
    from fukikae.cards.card import Card, Reference, Voice
    from fukikae.cards.select import CLONE_SEED, cut_clip
    from fukikae.config import ConfigError
    from fukikae.errors import user_errors

    try:
        cfg = Workspace(workspace).config
        name = backend or clone_backend(cfg)
    except user_errors() as e:
        raise _fail(str(e)) from e
    if not name:
        raise _fail("no [tts] clone_backend in the workspace or its profile; pass --backend")
    if not transcript.strip():
        raise _fail("--transcript is empty: give what the clip says")
    problem = loader.engine_problem("tts", name, "--backend" if backend else "[tts] clone_backend")
    if problem:
        raise _fail(problem)
    try:
        cls = loader.backend_class("tts", name)
    except (ConfigError, loader.BackendUnavailable) as e:
        raise _fail(str(e)) from e
    if "clone" not in getattr(cls, "sources", ()):
        raise _fail(f"{name} makes no clone voices; pass a cloning --backend")
    source = clip_language or cfg.source_language
    if not source:
        raise _fail("the workspace sets no source_language; pass --clip-language")
    if not clip.is_file():
        raise _fail(f"{clip}: no such file")
    from scipy.io import wavfile

    try:
        rate, data = wavfile.read(clip, mmap=True)
    except (ValueError, OSError) as e:
        raise _fail(f"{clip}: not a readable wav ({e})") from e
    if int(rate) <= 0:
        raise _fail(f"{clip}: its header gives a sample rate of {rate}")
    seconds = len(data) / float(rate)
    if not CLIP_SECONDS[0] <= seconds <= CLIP_SECONDS[1]:
        raise _fail(
            f"{clip} is {seconds:.1f} s; a reference clip is {CLIP_SECONDS[0]:.0f} to "
            f"{CLIP_SECONDS[1]:.0f} s"
        )
    with _session(workspace, lang) as (reg, store, language):
        char = reg.character_by_ref(character)
        with tempfile.TemporaryDirectory(prefix="fukikae-clip-") as tmp:
            cut = cut_clip(clip, 0.0, seconds, Path(tmp) / "clip.wav")
            ref = Reference(
                clip=store.add_sidecar(cut, "clip"),
                transcript=transcript.strip(),
                language=source,
                origin={"source": "voices clone"},
            )
        card = Card(
            target_language=language,
            voice=Voice(
                backend=cls.name,
                model_revision=getattr(cls, "model_revision", None) or "",
                source="clone",
                seed=CLONE_SEED,
                references=(ref,),
            ),
        )
        existed = store.has(card_id(card))
        card_ref = store.put(card)
        try:
            _check_clone_card(store, card_ref, _label(char), language)
        except CardError as e:
            if not existed:  # a card nothing refers to stays out of the store
                store.card_path(card_ref).unlink(missing_ok=True)
            why = str(e).split("; assign another card", 1)[0]
            raise _fail(
                f"the clone card from {clip.name} is refused ({why}); nothing was assigned"
            ) from e
        a = reg.assign_card(char.id, language, card_ref, by="assign", locked=True)
        _echo(
            f"{_label(char)} {language}: clone card {card_ref} of {cls.name} from {clip.name} "
            f"({seconds:.1f} s; {_state(a)}, history row {a.n})"
        )


def _split_pack_ref(ref: str) -> tuple[Path, str]:
    """``<pack>:<character>`` split at the first colon whose left side is a pack (a directory
    or a ``.tar.zst``), so a character name may hold colons."""
    for i, ch in enumerate(ref):
        if ch != ":":
            continue
        path = Path(ref[:i])
        if path.is_dir() or (path.is_file() and path.name.endswith(".tar.zst")):
            return path, ref[i + 1 :]
    from fukikae.cards.packs import PackError

    raise PackError(f"{ref!r} is not a card id or <pack>:<character>")


def _from_pack(
    workspace: Path, store: CardStore, ref: str, language: str
) -> tuple[str, CopyReport]:
    """``<pack>:<character>``: that character's card for ``language`` copied (checked) from
    the pack into the store, its fingerprint merged into a stored copy that has none (CD22);
    returns its id and what was copied (for rollback). The series check only warns here: the
    operator named the card."""
    from fukikae.cards.packs import (
        copy_cards,
        low_thresholds,
        printable,
        read_pack,
        series_check,
    )

    path, who = _split_pack_ref(ref)
    with read_pack(path) as pack:
        entry = pack.entry(who, language)
        ok, why = series_check(pack, Workspace(workspace).config.series)
        if not ok:
            _echo(f"warning: {why}")
        _echo(
            f"pack {printable(pack.meta.title)!r}: {printable(entry.key.name)} {language} "
            f"is card {entry.card}"
        )
        for line in low_thresholds(pack, [entry.card]):
            _echo(f"warning: {line}")
        return entry.card, copy_cards(pack, store, [entry.card])


def _copy_lines(copied: CopyReport) -> None:
    for cid in copied.merged:
        _echo(f"card {cid}: fingerprint merged from the pack (the store's copy had none)")
    for line in copied.conflicts:
        _echo(f"fingerprint conflict: {line}")


@voices_app.command("restock", hidden=True)
def restock(
    workspace: Path = _WS,
    dry_run: bool = typer.Option(False, "--dry-run", hidden=True),
    lang: str | None = _LANG,
) -> None:
    """Retired with stock voices (D41, D43 #5): it re-pinned automatic stock cards, and the
    next tts run now re-picks each of them as a clone. Kept only to say so."""
    del workspace, dry_run, lang
    raise _fail(
        "`fukikae voices restock` is retired: stock voices are retired (D41), and the next tts "
        "run re-picks every automatic stock card as a clone"
    )


@voices_app.command("history")
def history(workspace: Path = _WS, character: str = _CHAR, lang: str | None = _LANG) -> None:
    """List a character's card assignments for the target language, numbered from 1."""
    with _session(workspace, lang) as (reg, _, language):
        char = reg.character_by_ref(character)
        rows = reg.card_history(char.id, language, by_time=True)
        if not rows:
            _echo(f"{_label(char)} {language}: no history")
            return
        cur = reg.card_for(char.id, language)
        _echo("n\tassigned\tby\tcard\tstate\tnote")
        for h in rows:
            mark = " *" if cur is not None and h.n == cur.n else ""
            _echo(
                f"{h.n}{mark}\t{h.assigned}\t{h.by}\t{h.card_id or '-'}\t{_state(h)}\t"
                f"{h.note or ''}".rstrip()
            )


@voices_app.command("revert")
def revert(
    workspace: Path = _WS,
    character: str = _CHAR,
    to: int | None = typer.Option(
        None, "--to", help="History row to go back to (default: the previous card)."
    ),
    lang: str | None = _LANG,
) -> None:
    """Assign an earlier card again (locked), as a new history row."""
    with _session(workspace, lang) as (reg, store, language):
        char = reg.character_by_ref(character)
        with reg.transaction():  # a card missing from the store rolls the revert back
            a = reg.revert_card(char.id, language, to)
            assert a.card_id is not None
            if not store.has(a.card_id):
                raise ValueError(
                    f"card {a.card_id} is not in the card store ({store.root}); nothing was "
                    "reverted. Restore the card file or pick another history row with --to"
                )
            _refuse_stock(store.get(a.card_id), f"card {a.card_id}")  # rolls the revert back
        _echo(f"{_label(char)} {language}: card {a.card_id} (history row {a.n})")


def _lock(workspace: Path, character: str, lang: str | None, locked: bool) -> None:
    with _session(workspace, lang) as (reg, _, language):
        char = reg.character_by_ref(character)
        a = reg.lock_card(char.id, language, locked)
        _echo(f"{_label(char)} {language}: card {a.card_id} {_state(a)}")


@voices_app.command("lock")
def lock(workspace: Path = _WS, character: str = _CHAR, lang: str | None = _LANG) -> None:
    """Lock a character's card: imports and automatic assignment leave it alone."""
    _lock(workspace, character, lang, True)


@voices_app.command("unlock")
def unlock(workspace: Path = _WS, character: str = _CHAR, lang: str | None = _LANG) -> None:
    """Unlock a character's card."""
    _lock(workspace, character, lang, False)


def _editor() -> list[str]:
    cmd = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not cmd:
        raise _fail("set EDITOR (or VISUAL) to the editor to open the card in")
    return shlex.split(cmd)


@voices_app.command("edit")
def edit(workspace: Path = _WS, character: str = _CHAR, lang: str | None = _LANG) -> None:
    """Edit a copy of a character's card in $EDITOR; the copy becomes a new card (with
    derived_from set), assigned and locked. The original card is never changed."""
    with _session(workspace, lang) as (reg, store, language):
        char = reg.character_by_ref(character)
        cur = _current(reg, char, language)
        assert cur.card_id is not None
        try:
            old = store.get(cur.card_id)
        except StrippedCardError as e:  # CD23(c): edit the stripped text, save a new card
            old = e.card
            _echo(f"card {cur.card_id} held special-token syntax in [style]; it is removed in "
                  "the copy being edited")  # fmt: skip
        editor = _editor()
        with tempfile.TemporaryDirectory(prefix="fukikae-card-") as tmp:
            path = Path(tmp) / f"{cur.card_id}.toml"
            hint = "".join(  # a default is not in the file; say what can be set (ruling CD21)
                f'# {key} = "{mode}"  ({", ".join(modes)})\n'
                for key, mode, modes, is_default in _modes(old)
                if is_default
            )
            path.write_text(hint + dump_card(old), encoding="utf-8")
            done = subprocess.run([*editor, str(path)], check=False)  # noqa: S603
            if done.returncode != 0:
                raise _fail(f"the editor exited with {done.returncode}; nothing changed")
            text = path.read_text(encoding="utf-8")
        try:
            new = parse_card(text)
        except CardError as e:
            raise _fail(f"the edited card is invalid, nothing changed: {e}") from e
        _refuse_stock(new, "the edited card")
        if new.target_language != language:
            raise _fail(
                f"the edited card is for target language {new.target_language!r}, not "
                f"{language!r}; use `fukikae voices derive` for another language"
            )
        if card_id(new) == cur.card_id:
            _echo(
                f"{_label(char)} {language}: card {cur.card_id} unchanged ([meta] and the "
                "fingerprint are not part of a card's identity)"
            )
            return
        if (
            new.voice.fingerprint is not None
            and new.voice.fingerprint == old.voice.fingerprint
            and replace(new.voice, fingerprint=None) != replace(old.voice, fingerprint=None)
        ):
            # a fingerprint describes how the old voice rendered, not the edited one
            new = replace(new, voice=replace(new.voice, fingerprint=None))
            _echo("the voice changed, so its fingerprint was dropped")
        new = replace(
            new,
            meta={**new.meta, "derived_from": cur.card_id, "created": dt.date.today()},
        )
        try:
            _check_assignable(store, new, card_id(new), _label(char), language)
        except CardError as e:
            raise _fail(f"the edited card can't be used, nothing changed: {e}") from e
        new_id = store.put(new)
        a = reg.assign_card(char.id, language, new_id, by="edit", locked=True)
        _echo(
            f"{_label(char)} {language}: card {new_id}, derived from {cur.card_id} "
            f"({_state(a)}, history row {a.n})"
        )


@voices_app.command("derive")
def derive(
    workspace: Path = _WS,
    character: str = _CHAR,
    to: str = typer.Option(..., "--to", help="Target language of the new card."),
    replace_existing: bool = typer.Option(
        False, "--replace", help="Replace the character's existing card for --to."
    ),
    lang: str | None = _LANG,
) -> None:
    """Copy a character's card to another target language: same voice section, derived_from
    set, catchphrase renderings emptied; assigned unlocked for that language."""
    with _session(workspace, lang) as (reg, store, language):
        char = reg.character_by_ref(character)
        if not to or to == language:
            raise _fail(f"--to must name another target language than {language!r}")
        cur = _current(reg, char, language)
        assert cur.card_id is not None
        existing = reg.card_for(char.id, to)
        if existing is not None and not replace_existing:
            raise _fail(
                f"character {_label(char)} already has card {existing.card_id} for {to!r} "
                f"({existing.by}, {_state(existing)}); pass --replace to replace it"
            )
        old = store.get(cur.card_id)
        _refuse_stock(old, f"card {cur.card_id}")
        style = None
        if old.style:
            style = dict(old.style)
            if "catchphrases" in style:
                style["catchphrases"] = [
                    {**p, "target": ""} for p in style["catchphrases"]
                ]  # renderings are per target language and start empty
        new = Card(
            target_language=to,
            # a fingerprint describes renders of the old card, not this one (A7)
            voice=replace(old.voice, fingerprint=None),
            style=style,
            meta={**old.meta, "derived_from": cur.card_id, "created": dt.date.today()},
        )
        new_id = store.put(new)
        a = reg.assign_card(char.id, to, new_id, by="derive", locked=False)
        _echo(
            f"{_label(char)} {to}: card {new_id}, derived from {language} card {cur.card_id} "
            f"({_state(a)}, history row {a.n})"
        )


# -- packs and verification -------------------------------------------------------------------


def _synth_factory(cfg: Config) -> Callable[[Card], SynthSource]:
    """A card's backend as the tts stage would run it: its ``[tts.<backend>]`` config and
    device (ruling CD14: Kokoro on the CPU, cloning backends on ``gpu_device`` under the GPU
    lock, ruling CD15)."""
    from fukikae.backends import loader
    from fukikae.stages.tts import LazySynth, backend_config, synth_device

    def make(card: Card) -> SynthSource:
        cls = cast("type[Synthesizer]", loader.backend_class("tts", card.voice.backend))
        device = synth_device(cfg, cls)
        return LazySynth(backend_config(cfg, card.voice.backend), device, cfg)

    return make


def _embedder_factory() -> Callable[[str], AbstractContextManager[Embedder]]:
    """The fingerprint embedder, on the CPU (three short sentences; the GPU stays free)."""
    from fukikae.backends import loader
    from fukikae.cards.verify import local_checkpoint
    from fukikae.config import BackendConfig

    @contextmanager
    def make(name: str) -> Iterator[Embedder]:
        # ruling CD23(a): an allowlisted name and its pinned checkpoint from the local cache,
        # handed to the backend by path, so the load makes no hub call at all
        pinned, ckpt = local_checkpoint(name)
        bc = BackendConfig(pinned.backend, {"model": str(ckpt)})
        inst = cast("Embedder", loader.create("embed", bc, "cpu"))
        inst.load()
        try:
            yield inst
        finally:
            inst.unload()

    return make


def _verify(cfg: Config, store: CardStore, card: Card, record: bool) -> tuple[VerifyResult, Card]:
    from fukikae.cards.verify import verify_card

    with tempfile.TemporaryDirectory(prefix="fukikae-verify-") as tmp:
        return verify_card(
            card, store, make_synth=_synth_factory(cfg), make_embedder=_embedder_factory(),
            record=record, work=Path(tmp),
        )  # fmt: skip


def _verify_after(workspace: Path, store: CardStore, cards: list[tuple[str, str]]) -> None:
    """Verify each fingerprinted card among ``(label, card id)`` and print the outcome; never
    raises (import and assign report a mismatch without blocking)."""
    from fukikae.cards.packs import printable

    cfg = None
    done: set[str] = set()
    for label, cid in cards:
        try:
            card = store.get(cid)
            if card.voice.fingerprint is None or cid in done:
                continue
            done.add(cid)
            cfg = cfg or Workspace(workspace).config
            res, _ = _verify(cfg, store, card, False)
            line = f"verify {printable(label)} card {cid}: {res.status} ({res.detail})"
        except Exception as e:  # report, never block
            line = f"verify {printable(label)} card {cid}: could not run ({type(e).__name__}: {e})"
        _echo(line)


@voices_app.command("verify")
def verify_cmd(
    workspace: Path = _WS,
    target: str = typer.Argument(..., metavar="CARD|CHARACTER", help="A card id, or a character."),
    record: bool = typer.Option(
        False, "--record", help="Record the card's fingerprint (its id doesn't change)."
    ),
    lang: str | None = _LANG,
) -> None:
    """Render the fixed test script with the card and compare it with its fingerprint; exits
    1 unless it matches (or, with --record, was recorded)."""
    with _session(workspace, lang) as (reg, store, language):
        if is_card_id(target) and store.has(target):
            cid = target
        else:
            char = reg.character_by_ref(target)
            cur = _current(reg, char, language)
            assert cur.card_id is not None
            cid = cur.card_id
        card = store.get(cid)
        res, new = _verify(Workspace(workspace).config, store, card, record)
        # the fingerprint is outside the id: putting the recorded card rewrites the same file
        if res.status == "recorded" and store.put(new) != cid:  # pragma: no cover
            raise _fail(f"recording changed card {cid}'s id; nothing was stored")
        _echo(f"card {cid}: {res.status.replace('_', ' ')} ({res.detail})")
        if res.status not in ("match", "recorded"):
            raise typer.Exit(1)


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


@voices_app.command("export")
def export(
    workspace: Path = _WS,
    out: Path = typer.Option(..., "-o", "--out", help="New pack: a directory or a .tar.zst."),
    characters: str | None = typer.Option(
        None, "--characters", help="Comma-separated characters (ids or names); default all."
    ),
    langs: str | None = typer.Option(
        None, "--lang", help="Comma-separated target languages; default every one with a card."
    ),
    no_clips: bool = typer.Option(
        False, "--no-clips", help="Leave out cards that need audio clips (clones), listed."
    ),
    title: str | None = typer.Option(None, "--title", help="Pack title (default: series title)."),
    author: str = typer.Option("", "--author", help="Pack author."),
    require_fingerprints: bool = typer.Option(
        False,
        "--require-fingerprints",
        help="Refuse to export when a card has no fingerprint and none can be recorded here.",
    ),
    no_record: bool = typer.Option(
        False, "--no-record", help="Don't render to record missing fingerprints; list them."
    ),
) -> None:
    """Write the characters' current cards to a pack. Packs carry names and external ids,
    never registry ids; unnamed characters are left out and listed, and when that leaves
    nothing to export, nothing is written and it exits 1. A card without a
    fingerprint gets one first (ruling CD26: one render of the test script on its backend);
    when its backend can't run here the card is exported without one and listed, or, with
    --require-fingerprints, nothing is written."""
    from fukikae.cards.packs import PackError

    with _session(workspace, None) as (reg, store, _):
        cfg = Workspace(workspace).config
        series = cfg.series
        ids = None
        if characters is not None:
            ids = [reg.character_by_ref(c).id for c in _csv(characters) or []]
        from fukikae.cards.packs import export_selection

        cast_rows, _, left_out = export_selection(
            reg, store, characters=ids, languages=_csv(langs), no_clips=no_clips
        )
        if not cast_rows:
            for what, why in left_out:
                _echo(f"left out: {what}: {why}")
            raise _fail(
                "no character has a card that can be exported; nothing was written (packs "
                "match characters by name: name them with `fukikae registry rename`)"
            )
        missing = _fingerprint_for_export(cfg, store, cast_rows, record=not no_record)
        for label, card_id, why in missing:
            _echo(f"without a fingerprint: {label} card {card_id} ({why})")
        if missing and require_fingerprints:
            raise _fail(
                f"{len(missing)} card(s) have no fingerprint and none can be recorded here; "
                "nothing was written (drop --require-fingerprints to export them without)"
            )
        try:
            report = _export_pack(
                reg, store, series, out, characters=ids, languages=_csv(langs),
                no_clips=no_clips, title=title or f"{series.title} voices", author=author,
            )  # fmt: skip
        except PackError as e:
            raise _fail(str(e)) from e
        for what, why in report.left_out:
            _echo(f"left out: {what}: {why}")
        _echo(f"wrote {out}: {report.written} card assignment(s)")


def _export_pack(*args: Any, **kwargs: Any) -> Any:
    from fukikae.cards.packs import export_pack

    return export_pack(*args, **kwargs)


def _fingerprint_for_export(
    cfg: Config, store: CardStore, rows: list[CastEntry], *, record: bool
) -> list[tuple[str, str, str]]:
    """Record a fingerprint for every card among the export's own ``rows`` that has none
    (CD26), each on its backend under the GPU lock; returns (character, card id, why) for
    those left without one: ``record`` off, a backend that can't run here
    (``clone_problem``), or a render that failed."""
    from fukikae.backends import loader
    from fukikae.cards import auto
    from fukikae.cards.packs import printable
    from fukikae.config import ConfigError

    missing: list[tuple[str, str, str]] = []
    done: set[str] = set()
    for row in rows:
        if row.card in done or not store.has(row.card):
            continue
        done.add(row.card)
        card = store.get(row.card)
        if card.voice.fingerprint is not None:
            continue
        label = f"{printable(row.key.name)} {row.target_language}"
        if not record:
            missing.append((label, row.card, "--no-record"))
            continue
        try:
            cls = loader.backend_class("tts", card.voice.backend)
            why = auto.clone_problem(cast("type[Synthesizer]", cls), cfg)
        except (ConfigError, loader.BackendUnavailable) as e:
            why = str(e)
        if why is not None:
            missing.append((label, row.card, why))
            continue
        try:
            res, new = _verify(cfg, store, card, True)
        except Exception as e:  # noqa: BLE001 - a crash, timeout or OOM leaves one card unprinted
            missing.append((label, row.card, f"could not run ({type(e).__name__}: {e})"))
            continue
        if res.status != "recorded":
            missing.append((label, row.card, f"{res.status}: {res.detail}"))
            continue
        if store.put(new) != row.card:  # pragma: no cover - the fingerprint is outside the id
            raise _fail(f"recording changed card {row.card}'s id; nothing was stored")
        _echo(f"{label}: recorded a fingerprint for card {row.card}")
    return missing


@voices_app.command("import")
def import_(
    workspace: Path = _WS,
    pack_path: Path = typer.Argument(..., metavar="PACK", help="A pack directory or .tar.zst."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the mapping; change nothing."),
    replace_cards: bool = typer.Option(
        False, "--replace", help="Also replace locked and operator-chosen cards."
    ),
    force_series: bool = typer.Option(
        False, "--force-series", help="Import a pack whose series doesn't match this one."
    ),
    no_verify: bool = typer.Option(
        False, "--no-verify", help="Don't verify fingerprinted cards after importing."
    ),
) -> None:
    """Assign a pack's cards to this workspace's characters, matched by external character id,
    then by name (case- and width-insensitive), as `import:<pack title>`, unlocked. Locked and
    operator-chosen cards are skipped unless --replace."""
    from fukikae.cards.packs import (
        CharacterKey,
        PackError,
        apply_import,
        low_thresholds,
        plan_import,
        printable,
        read_pack,
        series_check,
    )

    with _session(workspace, None) as (reg, store, _):
        try:
            pack = read_pack(pack_path)
        except PackError as e:
            raise _fail(f"refused {pack_path}: {e}; nothing was imported") from e
        with pack:
            ok, why = series_check(pack, Workspace(workspace).config.series)
            meta = pack.meta
            _echo(
                f"pack {printable(meta.title)!r}"
                + (f" by {printable(meta.author)}" if meta.author else "")
                + f": {len(pack.cast)} cast entr{'y' if len(pack.cast) == 1 else 'ies'}"
            )
            if not ok:
                if not force_series:
                    raise _fail(f"{why}; pass --force-series to import it anyway")
                _echo(f"forced: {why}")
            else:
                _echo(why)
            plan = plan_import(pack, reg, replace=replace_cards)
            for line in low_thresholds(pack, [m.card_id for m in [*plan.assign, *plan.unchanged]]):
                _echo(f"warning: {line}")
            names = {c.id: c for c in reg.characters()}
            target = Workspace(workspace).config.target_language
            for other in sorted({m.target_language for m in plan.assign} - {target}):
                _echo(
                    f"note: the pack assigns cards for {other!r}, not this workspace's target "
                    f"language {target!r}; they are used only when dubbing into {other!r}"
                )

            def who(key: CharacterKey, cid: int) -> str:
                return f"{printable(key.name)} -> character {_label(names[cid])}"

            for m in plan.assign:
                _echo(
                    f"{who(m.key, m.character_id)} {m.target_language} by {m.matched_by}: "
                    f"card {m.card_id}"
                )
            for m in plan.unchanged:
                _echo(f"unchanged: {who(m.key, m.character_id)} {m.target_language}: "
                           f"already card {m.card_id}")  # fmt: skip
            for s in plan.skipped:
                _echo(f"skipped: {who(s.key, s.character_id)}: {s.reason}")
            for u in plan.unmatched:
                ext = f" (external id {u.key.external_id})" if u.key.external_id else ""
                _echo(f"unmatched: {printable(u.key.name)}{ext} {u.target_language}: {u.reason}")
            if dry_run:
                _echo("dry run: nothing changed")
                return
            try:
                copied = apply_import(plan, pack, reg, store)  # rolls back on any failure
            except (PackError, CardError, ValueError, sqlite3.Error) as e:
                # ProtectedAssignmentError (a ValueError) included: a card locked meanwhile
                raise _fail(f"{e}; nothing was imported") from e
            _echo(f"imported {len(plan.assign)} card assignment(s) as import:"
                       f"{printable(meta.title)}")  # fmt: skip
            _copy_lines(copied)
            if not no_verify:
                label = {m.card_id: names[m.character_id].name or str(m.character_id)
                         for m in [*plan.unchanged, *plan.assign]}  # fmt: skip
                todo = [m.card_id for m in plan.assign] + copied.merged
                _verify_after(workspace, store, [(label.get(c, c), c) for c in dict.fromkeys(todo)])
