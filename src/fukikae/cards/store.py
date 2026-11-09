"""The card store: ``<workspace>/.fukikae/cards/<id>.toml`` plus sidecar directories."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from fukikae.cards.card import (
    Card,
    CardError,
    dump_card,
    is_card_id,
    is_sidecar_ref,
    parse_card,
    sidecars,
)
from fukikae.cards.card import card_id as compute_card_id

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


class CardStore:
    def __init__(self, workspace_root: Path) -> None:
        self.root = Path(workspace_root) / ".fukikae" / "cards"

    def card_path(self, card_id: str) -> Path:
        if not is_card_id(card_id):
            raise CardError(f"not a card id (16 lowercase hex digits): {card_id!r}")
        return self.root / f"{card_id}.toml"

    def put(self, card: Card) -> str:
        """Write the card under its id. Idempotent; overwrites, so a recorded fingerprint
        (which doesn't change the id) lands in the stored file."""
        cid = compute_card_id(card)
        text = dump_card(card)
        parse_card(text)  # a card built in code gets the same checks as one read from disk
        _atomic_write(self.card_path(cid), text.encode("utf-8"))
        return cid

    def has(self, card_id: str) -> bool:
        return self.card_path(card_id).is_file()

    def get(self, card_id: str) -> Card:
        path = self.card_path(card_id)
        if not path.is_file():
            raise CardError(f"card {card_id} not found in {self.root}")
        card = parse_card(path.read_text(encoding="utf-8"))
        actual = compute_card_id(card)
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
