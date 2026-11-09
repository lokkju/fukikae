"""`fukikae voices` commands: which card each character speaks with, per target language."""

from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from fukikae.cards.card import (
    MODE_FIELDS,
    Card,
    CardError,
    Voice,
    card_id,
    dump_card,
    parse_card,
)
from fukikae.cards.store import CardStore
from fukikae.workspace import Workspace

if TYPE_CHECKING:
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


def _fail(msg: str) -> typer.Exit:
    typer.echo(f"error: {msg}", err=True)
    return typer.Exit(1)


@contextmanager
def _session(workspace: Path, lang: str | None) -> Iterator[tuple[Registry, CardStore, str]]:
    """The workspace's registry, card store and target language; input errors exit 1."""
    from fukikae.errors import user_errors
    from fukikae.registry.store import Registry

    ws = Workspace(workspace)
    try:
        language = lang or ws.config.target_language
    except user_errors() as e:
        raise _fail(str(e)) from e
    with Registry(ws.registry_path) as reg:
        try:
            yield reg, CardStore(ws.root), language
        except (CardError, ValueError) as e:
            raise _fail(str(e)) from e
        except KeyError as e:
            raise _fail(f"no character {e.args[0]}") from e


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
    """List every character with its card for the target language."""
    with _session(workspace, lang) as (reg, _, language):
        typer.echo(f"character\tname\t{language} card\tby\tstate")
        for c in reg.characters():
            a = reg.card_for(c.id, language)
            if a is None:
                typer.echo(f"{c.id}\t{c.name or '-'}\t-\t-\t-")
            else:
                typer.echo(f"{c.id}\t{c.name or '-'}\t{a.card_id}\t{a.by}\t{_state(a)}")


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
        typer.echo(
            f"character {_label(char)}"
            + (f", external id {char.external_id}" if char.external_id else "")
            + (f"; cards also for {', '.join(others)}" if others else "")
        )
        a = reg.card_for(char.id, language)
        if a is None or a.card_id is None:
            typer.echo(f"{language}: no card")
            return
        typer.echo(f"{language}: card {a.card_id} ({a.by}, {_state(a)}, history row {a.n})")
        if a.note:
            typer.echo(f"note: {a.note}")
        card = store.get(a.card_id)
        emotions = card.meta.get("reference_emotions")
        if emotions:  # ruling CD5
            typer.echo(f"references may carry emotion: {', '.join(map(str, emotions))}")
        for _key, mode, modes, is_default in _modes(card):  # ruling CD21
            typer.echo(
                f"clone mode: {mode}" + (f" (default; {', '.join(modes)})" if is_default else "")
            )
        typer.echo("")
        typer.echo(dump_card(card), nl=False)


@voices_app.command("assign")
def assign(
    workspace: Path = _WS,
    character: str = _CHAR,
    ref: str = typer.Argument(
        ...,
        metavar="CARD",
        help="A card id in the workspace's store, or a stock voice: stock:<backend>:<voice>.",
    ),
    no_lock: bool = typer.Option(False, "--no-lock", help="Leave the assignment unlocked."),
    lang: str | None = _LANG,
) -> None:
    """Give a character a card for the target language (locked unless --no-lock)."""
    with _session(workspace, lang) as (reg, store, language):
        char = reg.character_by_ref(character)
        if ref.startswith("stock:"):
            ref = _stock(store, ref, language)
        if not store.has(ref):
            raise _fail(f"card {ref} is not in the card store ({store.root}); nothing was assigned")
        card = store.get(ref)
        if card.target_language != language:
            raise _fail(
                f"card {ref} is for target language {card.target_language!r}, not "
                f"{language!r}; derive a {language!r} card with `fukikae voices derive "
                f"<workspace> <character> --lang {card.target_language} --to {language}`"
            )
        a = reg.assign_card(char.id, language, ref, by="assign", locked=not no_lock)
        typer.echo(f"{_label(char)} {language}: card {ref} ({_state(a)}, history row {a.n})")


def _stock(store: CardStore, ref: str, language: str) -> str:
    """Put the card for a ``stock:<backend>:<voice>`` ref in the store; return its id."""
    from fukikae.backends.loader import BackendUnavailable
    from fukikae.cards.stock import check_stock_voice, parse_stock_ref, revision, stock_card
    from fukikae.config import ConfigError

    backend, voice = parse_stock_ref(ref)
    try:
        check_stock_voice(backend, voice)
        return store.put(stock_card(backend, voice, language, revision(backend)))
    except (ConfigError, BackendUnavailable) as e:
        raise CardError(f"{ref}: {e}") from e


@voices_app.command("restock")
def restock(
    workspace: Path = _WS,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List what would change; change nothing."
    ),
    lang: str | None = _LANG,
) -> None:
    """Re-pin automatic (or earlier restocked) stock cards to the revision their backend renders
    with now: same voice, a new card, assigned by `restock` (unlocked, still uncurated, so the
    next revision bump restocks it again). A curated card (locked, or chosen by an operator or
    an import) is never touched."""
    from fukikae.backends.loader import BackendUnavailable
    from fukikae.cards.stock import revision, stock_card
    from fukikae.config import ConfigError

    with _session(workspace, lang) as (reg, store, language):
        changed = 0
        for char in reg.characters():
            a = reg.card_for(char.id, language)
            # CD9: automatic and restocked cards only; never a locked or curated one
            if a is None or a.card_id is None or a.locked or a.chosen_by not in ("auto", "restock"):
                continue
            if not store.has(a.card_id):
                continue
            v = store.get(a.card_id).voice
            if v.source != "stock" or not v.voice:
                continue
            try:
                rev = revision(v.backend)
            except (CardError, ConfigError, BackendUnavailable) as e:
                typer.echo(f"{_label(char)}: card {a.card_id} left alone ({e})")
                continue
            if v.model_revision == rev:
                continue
            new = stock_card(v.backend, v.voice, language, rev)
            what = (
                f"{_label(char)} {language}: {v.backend}:{v.voice} from revision "
                f"{v.model_revision!r} to {rev!r}"
            )
            changed += 1
            if dry_run:
                typer.echo(f"would restock {what}")
                continue
            new_id = store.put(new)
            n = reg.assign_card(
                char.id, language, new_id, by="restock", locked=False,
                note=f"restocked from card {a.card_id}",
            ).n  # fmt: skip
            typer.echo(f"restocked {what}: card {new_id} (history row {n})")
        if not changed:
            typer.echo("nothing to restock")


@voices_app.command("history")
def history(workspace: Path = _WS, character: str = _CHAR, lang: str | None = _LANG) -> None:
    """List a character's card assignments for the target language, numbered from 1."""
    with _session(workspace, lang) as (reg, _, language):
        char = reg.character_by_ref(character)
        rows = reg.card_history(char.id, language)
        if not rows:
            typer.echo(f"{_label(char)} {language}: no history")
            return
        cur = reg.card_for(char.id, language)
        typer.echo("n\tassigned\tby\tcard\tstate\tnote")
        for h in rows:
            mark = " *" if cur is not None and h.n == cur.n else ""
            typer.echo(
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
        typer.echo(f"{_label(char)} {language}: card {a.card_id} (history row {a.n})")


def _lock(workspace: Path, character: str, lang: str | None, locked: bool) -> None:
    with _session(workspace, lang) as (reg, _, language):
        char = reg.character_by_ref(character)
        a = reg.lock_card(char.id, language, locked)
        typer.echo(f"{_label(char)} {language}: card {a.card_id} {_state(a)}")


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
        old = store.get(cur.card_id)
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
        if new.target_language != language:
            raise _fail(
                f"the edited card is for target language {new.target_language!r}, not "
                f"{language!r}; use `fukikae voices derive` for another language"
            )
        if card_id(new) == cur.card_id:
            typer.echo(
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
            typer.echo("the voice changed, so its fingerprint was dropped")
        new = replace(
            new,
            meta={**new.meta, "derived_from": cur.card_id, "created": dt.date.today()},
        )
        new_id = store.put(new)
        a = reg.assign_card(char.id, language, new_id, by="edit", locked=True)
        typer.echo(
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
        style = None
        if old.style:
            style = dict(old.style)
            if "catchphrases" in style:
                style["catchphrases"] = [
                    {**p, "target": ""} for p in style["catchphrases"]
                ]  # renderings are per target language and start empty
        new = Card(
            target_language=to,
            voice=old.voice,
            style=style,
            meta={**old.meta, "derived_from": cur.card_id, "created": dt.date.today()},
        )
        new_id = store.put(new)
        a = reg.assign_card(char.id, to, new_id, by="derive", locked=False)
        typer.echo(
            f"{_label(char)} {to}: card {new_id}, derived from {language} card {cur.card_id} "
            f"({_state(a)}, history row {a.n})"
        )
