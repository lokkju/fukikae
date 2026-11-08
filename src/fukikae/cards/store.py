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
        with os.fdopen(fd, "wb") as f:
            if isinstance(data, bytes):
                f.write(data)
            else:
                with data.open("rb") as src:
                    shutil.copyfileobj(src, f)
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
        _atomic_write(self.card_path(cid), dump_card(card).encode("utf-8"))
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
        found = self._find(ref)
        if found is None:
            raise CardError(
                f"sidecar {ref} not found in {self.root} "
                f"({', '.join(d + '/' for d in SIDECAR_DIRS.values())})"
            )
        return found

    def missing_sidecars(self, card: Card) -> list[str]:
        return [ref for ref in sidecars(card) if self._find(ref) is None]
