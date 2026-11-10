"""The names stage's records. `NamedLine` is a pydantic dataclass: frozen and `replace`-able
like the alignment check expects, and serialisable into `names.json`. `Piece` is a diarize turn,
or the part of one between cuts, with the key it takes; `LabelInfo` describes one refined
cluster label; `NamesResult` is `04c-names/names.json`."""

from typing import Any, Literal

from pydantic import BaseModel, Field
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
    line_ids: list[int]  # the visible lines attributed to the turn that overlap this piece
    turn: int  # index of the diarize turn it comes from


class LabelInfo(BaseModel):
    """One refined cluster label: its parent diarize cluster and the key it stands for."""

    parent: str
    key: str | None
    display: str | None
    cls: Literal["named", "role", "remainder"]
    slug_clash: bool = False  # the key's slug collides with another key's in this cluster, so
    # the label carries the hashed form (`n` + 8 hex of sha256(key))


Verdict = Literal["off", "none", "holdout", "partial", "full"]


class NamesResult(BaseModel):
    """`04c-names/names.json`, written on every run of the names stage.

    `lines` are every dialogue line of the chosen track (named or not), already moved back by
    `offset_s`, hidden ones included with their keys (the eval grades on them). Consumers read
    names through `visible_lines()`: a hidden line never names, cuts, labels, excludes or
    suggests anything (Review Focus 3). Empty for `off`, `holdout` and `none`."""

    schema_version: int = 1
    episode: str
    used: bool  # turns.json and clusters.npz beside this file are the identity inputs
    verdict: Verdict
    reason: str | None  # why names weren't used (the failing test for `none`)
    source: dict[str, Any] | None  # the chosen track: how, language, track, codec, file, title
    tried: list[dict[str, Any]] = Field(default_factory=list)  # every candidate and its verdict
    offset_s: float  # lines = audio + offset; the lines here are already moved back
    coverage: dict[str, float | int]
    lines: list[NamedLine]
    hidden: list[int]  # ids of lines withheld from the stage (`[names] hide_fraction`)
    pieces: list[Piece]
    labels: dict[str, LabelInfo]
    unvoiced: list[int]
    excluded: list[tuple[float, float]]  # inner-voice and multi spans: never references
    dropped_row_seconds: float
    suggestions: list[tuple[str, str]]  # (rare, common) keys one edit apart

    def visible_lines(self) -> list[NamedLine]:
        hidden = set(self.hidden)
        return [ln for ln in self.lines if ln.id not in hidden]
