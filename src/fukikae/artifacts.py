"""Atomic artifact IO: nothing partial is ever visible at the final path."""

import contextlib
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel


class ArtifactVersionError(Exception):
    """The artifact was written by a newer schema than this code understands."""


@contextlib.contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temp path beside `path`; rename onto it on clean exit, delete on error."""
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json(path: Path, model: BaseModel) -> None:
    with atomic_path(path) as tmp:
        tmp.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def read_json[T: BaseModel](path: Path, cls: type[T]) -> T:
    raw = Path(path).read_text(encoding="utf-8")
    obj = cls.model_validate_json(raw)
    field = cls.model_fields.get("schema_version")
    if field is not None and isinstance(field.default, int):
        found = getattr(obj, "schema_version")  # noqa: B009
        if found > field.default:
            raise ArtifactVersionError(
                f"{path}: schema_version {found} is newer than supported {field.default}"
            )
    return obj
