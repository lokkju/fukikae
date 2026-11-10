"""The names stage's records. `NamedLine` is a pydantic dataclass: frozen and `replace`-able
like the alignment check expects, and serialisable into `names.json`. `Piece` is a diarize turn,
or the part of one between cuts, with the key it takes; `LabelInfo` describes one refined
cluster label."""

from typing import Literal

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from fukikae.names.normalize import NameClass


@dataclass(frozen=True)
class NamedLine:
    """One dialogue line of a subtitle track with its parsed Name."""

    id: int  # position among the track's dialogue lines, in start order
    start: float
    end: float
    raw: str  # the Name field as written
    key: str | None  # the folded canonical name; None for multi and ignored
    display: str | None
    cls: NameClass
    inner: bool  # an inner-voice or off-screen marker was stripped
    text: str  # the spoken text


class Piece(BaseModel):
    """A diarize turn, or one side of a cut in it, with its refined label."""

    start: float
    end: float
    cluster: str  # the diarize cluster the turn came from
    label: str  # `<cluster>~<slug>` when named, else the cluster itself
    key: str | None  # the name key it takes; None for the unnamed remainder
    line_ids: list[int]  # the lines attributed to the turn that overlap this piece


class LabelInfo(BaseModel):
    """One refined cluster label: its parent diarize cluster and the key it stands for."""

    parent: str
    key: str | None
    display: str | None
    cls: Literal["named", "role", "remainder"]
