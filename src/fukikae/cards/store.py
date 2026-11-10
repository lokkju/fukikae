"""The card store: ``<workspace>/.fukikae/cards/<id>.toml`` plus sidecar directories."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from fukikae.cards.card import (
    POST_DEFAULTS,
    Card,
    CardError,
    dump_card,
    is_card_id,
    is_sidecar_ref,
    parse_card,
    sidecars,
)
from fukikae.cards.card import card_id as compute_card_id
from fukikae.config import normalize_language

SidecarKind = Literal["clip", "latent", "vector"]
SIDECAR_DIRS: dict[str, str] = {"clip": "clips", "latent": "latents", "vector": "vectors"}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(dest: Path, data: bytes | Path) -> None:
    """Write through a temp file in ``dest``'s directory, then rename over ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "wb") as f:
            if isinstance(data, bytes):
                f.write(data)
            else:
                with data.open("rb") as src:
                    shutil.copyfileobj(src, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class StrippedCardError(CardError):
    """A hand-written card whose ``[style]`` holds special-token syntax: its file name is the
    id of the text as written, and the loaded (stripped) card has another id. ``card`` is the
    stripped card, which ``voices edit`` saves as a new card."""

    def __init__(self, card_id: str, path: Path, card: Card) -> None:
        self.card_id, self.path, self.card = card_id, path, card
        super().__init__(
            f"card {card_id} ({path}): [style] holds special-token syntax (such as <|...|>), "
            "which is stripped at load (CD23(c)); remove it from the file and re-save it as a "
            "new card. To do that: `fukikae voices edit <workspace> <character>` opens the "
            "card with the markup already removed; save it, and the clean copy becomes a new "
            "card (derived_from the old id), assigned to the character"
        )


class StaleCardError(CardError):
    """A card stored before the card id rules changed (A5: language case, ``[voice.post]``
    defaults and empty ``[style]`` fields no longer count): its file name is its id under the
    old rules, and today it hashes to ``new_id``. ``card`` is the loaded card, which
    ``voices assign`` stores under ``new_id``."""

    def __init__(self, card_id: str, path: Path, card: Card, new_id: str) -> None:
        self.card_id, self.path, self.card, self.new_id = card_id, path, card, new_id
        super().__init__(
            f"card {card_id} ({path}) was stored before the card id rules changed (language "
            "case, [voice.post] defaults and empty [style] fields no longer count); today it "
            f"is card {new_id}. Re-assign it with `fukikae voices assign <workspace> "
            f"<character> {card_id}`, which stores it under {new_id}"
        )


def _old_id_rules_apply(text: str) -> bool:
    """Whether the card text holds something the A5 id rules drop or rewrite: a language tag
    not in conventional case, a ``[voice.post]`` key at its default, or an empty ``[style]``
    field. A card stored before A5 under such text has another id today."""
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return False

    def table(parent: dict[str, Any], key: str) -> dict[str, Any]:
        value = parent.get(key)
        return value if isinstance(value, dict) else {}

    voice = table(raw, "voice")
    refs = voice.get("references")
    refs = refs if isinstance(refs, list) else []
    langs = [raw.get("target_language")] + [r.get("language") for r in refs if isinstance(r, dict)]
    if any(isinstance(lang, str) and normalize_language(lang) != lang for lang in langs):
        return True
    post = table(voice, "post")
    if any(k in POST_DEFAULTS and v == POST_DEFAULTS[k] for k, v in post.items()):
        return True
    style = table(raw, "style")
    return any(v in ("", []) for v in style.values())


class CardStore:
    def __init__(self, workspace_root: Path) -> None:
        self.root = Path(workspace_root) / ".fukikae" / "cards"

    def card_path(self, card_id: str) -> Path:
        if not is_card_id(card_id):
            raise CardError(f"not a card id (16 lowercase hex digits): {card_id!r}")
        return self.root / f"{card_id}.toml"

    def put(self, card: Card) -> str:
        """Write the card under its id. Idempotent. When the store already holds this id, the
        stored fingerprint and ``[meta]`` (both outside the id) are kept unless ``card``
        carries its own fingerprint, which then replaces the stored one (recording); stored
        ``[meta]`` is kept whenever ``card`` has none. A re-put of the same voice therefore
        never drops a recorded fingerprint (CD22, CD26)."""
        cid = compute_card_id(card)
        stored = self._stored(cid)
        if stored is not None:
            fingerprint = card.voice.fingerprint or stored.voice.fingerprint
            meta = card.meta or stored.meta
            card = replace(card, meta=meta, voice=replace(card.voice, fingerprint=fingerprint))
        text = dump_card(card)
        parse_card(text)  # a card built in code gets the same checks as one read from disk
        _atomic_write(self.card_path(cid), text.encode("utf-8"))
        return cid

    def _stored(self, card_id: str) -> Card | None:
        """The stored card ``card_id`` when its file parses as that card, else None (a
        missing or corrupted file, which ``put`` rewrites)."""
        path = self.card_path(card_id)
        try:
            stored = parse_card(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, CardError):
            return None
        return stored if compute_card_id(stored) == card_id else None

    def has(self, card_id: str) -> bool:
        return self.card_path(card_id).is_file()

    def get(self, card_id: str) -> Card:
        path = self.card_path(card_id)
        if not path.is_file():
            raise CardError(f"card {card_id} not found in {self.root}")
        text = path.read_text(encoding="utf-8")
        card = parse_card(text)
        actual = compute_card_id(card)
        if actual != card_id and compute_card_id(parse_card(text, strip_tokens=False)) == card_id:
            raise StrippedCardError(card_id, path, card)
        if actual != card_id and _old_id_rules_apply(text):
            raise StaleCardError(card_id, path, card, actual)
        if actual != card_id:
            raise CardError(
                f"{path} holds card {actual}, not card {card_id}; a card file is never edited "
                "in place (edit a copy, which gets its own id)"
            )
        return card

    def add_sidecar(self, path: Path, kind: SidecarKind) -> str:
        """Copy ``path`` into the store as ``<sha256>.<ext>``; return ``sha256:<hex>``."""
        if kind not in SIDECAR_DIRS:
            raise CardError(f"unknown sidecar kind {kind!r}; use one of {', '.join(SIDECAR_DIRS)}")
        digest = _sha256_file(path)
        dest = self.root / SIDECAR_DIRS[kind] / f"{digest}{path.suffix.lower()}"
        if not dest.is_file():
            _atomic_write(dest, path)
        return f"sha256:{digest}"

    def adopt_sidecar(self, path: Path, ref: str, kind: SidecarKind) -> Path | None:
        """Copy ``path`` in as sidecar ``ref`` of ``kind``, after checking its bytes hash to
        ``ref``; return the new file, or None when the store already has ``ref`` (left as it
        is)."""
        if kind not in SIDECAR_DIRS or not is_sidecar_ref(ref):
            raise CardError(f"not a sidecar of a known kind: {kind!r} {ref!r}")
        if self._find(ref) is not None:
            return None
        digest = ref.removeprefix("sha256:")
        if _sha256_file(path) != digest:
            raise CardError(f"{path} does not hash to {ref}; not added to the card store")
        dest = self.root / SIDECAR_DIRS[kind] / f"{digest}{path.suffix.lower()}"
        _atomic_write(dest, path)
        return dest

    def _find(self, ref: str) -> Path | None:
        if not is_sidecar_ref(ref):
            return None
        digest = ref.removeprefix("sha256:")
        for sub in SIDECAR_DIRS.values():
            d = self.root / sub
            if d.is_dir():
                for p in sorted(d.glob(f"{digest}.*")) + sorted(d.glob(digest)):
                    if p.is_file():
                        return p
        return None

    def sidecar_path(self, ref: str) -> Path:
        """The sidecar's path, after re-checking that its bytes still hash to ``ref``."""
        found = self._find(ref)
        if found is None:
            raise CardError(
                f"sidecar {ref} not found in {self.root} "
                f"({', '.join(d + '/' for d in SIDECAR_DIRS.values())})"
            )
        actual = _sha256_file(found)
        if f"sha256:{actual}" != ref:
            raise CardError(
                f"sidecar {ref} at {found} is corrupted: its bytes hash to sha256:{actual}; "
                "restore the file (re-import the pack or re-add the clip)"
            )
        return found

    def missing_sidecars(self, card: Card) -> list[str]:
        return [ref for ref in sidecars(card) if self._find(ref) is None]
