"""The names stage's records. `NamedLine` is a pydantic dataclass: frozen and `replace`-able
like the alignment check expects, and serialisable into `names.json`."""

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
