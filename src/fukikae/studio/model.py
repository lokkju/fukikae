"""The studio's data as pydantic models: decisions first (the one thing the studio writes).

Every identity edit and every accept is a `Decision`, discriminated on ``op`` and applied by
`fukikae.studio.apply.apply_decision`. Each decision names its characters by registry id,
and carries the character's ``key`` (folded name) and ``external_id`` beside the id for
sharing later, and the ``digest`` of the registry rows it was made against
(`fukikae.studio.apply.row_digest`); a decision whose rows changed since is refused.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from fukikae.pitch import FEMALE_ABOVE, MALE_BELOW


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RefSpan(_Strict):
    """A stretch of an episode's speech used as a reference clip."""

    episode: str
    start: float
    end: float

    def label(self) -> str:
        return f"{self.episode} {self.start:.2f}-{self.end:.2f}"


class _Base(_Strict):
    # the character's alias key and external id when the decision was made (sharing later);
    # for a merge, the kept character's
    key: str | None = None
    external_id: str | None = None
    # sha256 of the registry rows the decision was made against; None: nothing to check
    # against, which only --force applies
    digest: str | None = None


class NameDecision(_Base):
    op: Literal["name"]
    character: int
    name: str = Field(min_length=1)


class AliasDecision(_Base):
    op: Literal["alias"]
    character: int
    name: str = Field(min_length=1)  # a subtitle Name as written; keyed as names keys it


class GenderDecision(_Base):
    op: Literal["gender"]
    character: int
    gender: Literal["f", "m"] | None


class MergeDecision(_Base):
    op: Literal["merge"]
    keep: int  # character ids; their actors are merged
    drop: int


class SplitDecision(_Base):
    op: Literal["split"]
    character: int  # the character whose cluster it is (the digest's rows)
    episode: str
    cluster: str
    by: Literal["pitch", "cluster"]
    # pitch only: the bands each turn's median is classified by
    male_below: float = MALE_BELOW
    female_above: float = FEMALE_ABOVE


class KindDecision(_Base):
    op: Literal["kind"]
    character: int
    kind: Literal["character", "narrator", "ignored"]


class LockDecision(_Base):
    op: Literal["lock"]
    character: int


class ReviewedDecision(_Base):
    op: Literal["reviewed"]
    character: int


class AcceptDecision(_Base):
    op: Literal["accept"]
    character: int
    language: str = Field(min_length=1)
    card_id: str
    draft: str  # the draft's name, <character>.<language>
    version: int
    references: list[RefSpan] = []
    overrides: list[RefSpan] = []  # gated references the operator chose anyway
    auditions: int = 0  # how many auditions led to it, for the card's note


Decision = Annotated[
    NameDecision
    | AliasDecision
    | GenderDecision
    | MergeDecision
    | SplitDecision
    | KindDecision
    | LockDecision
    | ReviewedDecision
    | AcceptDecision,
    Field(discriminator="op"),
]

IDENTITY_OPS = frozenset({"name", "alias", "gender", "merge", "split", "kind", "lock", "reviewed"})

_ADAPTER: TypeAdapter[Decision] = TypeAdapter(Decision)


def parse_decision(obj: Any) -> Decision:
    """One decision from its JSON form; ValueError (pydantic's) when it isn't one."""
    return _ADAPTER.validate_python(obj)


class Decisions(_Strict):
    """A file of decisions, applied in order by `fukikae studio apply`. ``workspace`` (the
    absolute root the ids belong to) is checked when set: registry ids are workspace-local."""

    schema_version: Literal[1] = 1
    workspace: str | None = None
    decisions: list[Decision]


__all__ = [
    "AcceptDecision",
    "AliasDecision",
    "Decision",
    "Decisions",
    "GenderDecision",
    "IDENTITY_OPS",
    "KindDecision",
    "LockDecision",
    "MergeDecision",
    "NameDecision",
    "RefSpan",
    "ReviewedDecision",
    "SplitDecision",
    "ValidationError",
    "parse_decision",
]
