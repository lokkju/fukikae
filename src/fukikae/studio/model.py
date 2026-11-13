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

from fukikae.cards.select import Tier
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
    version: int = Field(ge=1)
    references: list[RefSpan] = []
    overrides: list[RefSpan] = []  # gated references the operator chose anyway
    auditions: int = Field(default=0, ge=0)  # how many auditions led to it, for the note


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


# -- the season index (`fukikae.studio.season`, ``<ws>/studio/season.json``) -----------------
#
# Derived from the workspace and rebuilt when its inputs change; never edited. The page and
# later tools read it (`GET /api/season`); ``schema_version`` versions it.


class Sample(_Strict):
    """A clip of a character's (or an extras cluster's) speech: ``clip`` is the file name
    under ``<ws>/studio/audio/`` (`fukikae.studio.season.cut_sample`)."""

    episode: str
    start: float
    end: float
    transcript: str
    clip: str


class GenderInfo(_Strict):
    """A0's estimate over the season (class, confidence, median), seconds of turns per class
    (``m``, ``u``, ``f``), the class per episode, and the gender the operator stored."""

    estimate: Literal["f", "m", "u"]
    confidence: float
    median: float | None
    seconds: dict[str, float]
    per_episode: dict[str, str]
    stored: str | None
    stored_by: str | None


class CardInfo(_Strict):
    """A character's current card for one target language. ``source`` is the card's voice
    source (``clone``; ``stock`` only for a card D41 retired); ``flags`` names each reason
    the card is flagged (``thin``, ``floor``, ``impure``, ``stock``, ``stale``, ``gender``).
    ``digest`` is `fukikae.studio.apply.row_digest` for the language: what an accept made
    against this assignment carries."""

    language: str
    card_id: str | None
    source: str | None
    chosen_by: str
    locked: bool
    tier: Tier | None
    flags: list[str]
    digest: str | None = None


class Suggestion(_Strict):
    """A merge (``other``: the other character, ``cosine`` of their registry centroids; or
    None from a names hint, with its ``episode``) or a pitch split of one episode's cluster
    (``parts``: per class its seconds and sample spans)."""

    kind: Literal["merge", "split"]
    other: int | None = None
    cosine: float | None = None
    episode: str | None = None
    cluster: str | None = None
    parts: dict[str, Any] | None = None


class Flag(_Strict):
    """Something for the operator to look at: ``kind`` one of ``thin``, ``stock``,
    ``impure``, ``bimodal``, ``gated``, ``names``, ``floor``, ``stale``, ``gender``,
    ``dropped``, ``split`` (an operator pitch split the names stage skipped because the
    cluster changed since), ``identity`` (a pitch split's new character, unreviewed);
    ``episode`` when it belongs to one."""

    kind: str
    text: str
    episode: str | None = None


class EpisodeSpeech(_Strict):
    """A character's speech in one episode: its clusters, seconds of turns, the clusters'
    clean seconds, turns and translated cues."""

    clusters: list[str]
    seconds: float
    clean_seconds: float
    turns: int
    cues: int


class CharacterEntry(_Strict):
    id: int
    actor_id: int
    name: str | None
    key: str | None
    external_id: str | None
    kind: str
    locked: bool
    reviewed: str | None
    digest: str  # row_digest of the character: what an identity decision carries
    episodes: dict[str, EpisodeSpeech]
    seconds: float
    clean_seconds: float
    turns: int
    cues: int
    samples: list[Sample]
    gender: GenderInfo
    cards: list[CardInfo]
    suggestions: list[Suggestion]
    names: list[str]  # names evidence (D27): key decisions, named lines, disagreements
    flags: list[Flag]


class ExtraEntry(_Strict):
    """An extras cluster of one episode: its seconds of turns, and its episode-local clone
    card and tier from the episode's last tts run (None before one)."""

    episode: str
    label: str
    seconds: float
    tier: Tier | None
    card_id: str | None
    samples: list[Sample]


class EpisodeEntry(_Strict):
    """An episode with a work dir: the last stage its manifest records done, whether the
    studio can use it (its characters stage has written assignments), the command that
    brings it in when not, and a ``note`` on what the studio can't show for it (no vocals
    stem: no samples or pitch)."""

    id: str
    last_stage: str | None
    ready: bool
    command: str | None
    note: str | None = None


class SeasonDoc(_Strict):
    schema_version: Literal[1] = 1
    workspace: str
    built: str  # ISO 8601, UTC
    inputs_digest: str
    episodes: list[EpisodeEntry]
    characters: list[CharacterEntry]
    extras: list[ExtraEntry]
    queue: list[int]  # character ids, the review order (`fukikae.studio.season.order_queue`)


# -- voice drafts (`fukikae.studio.drafts`, ``<ws>/studio/drafts/<character>.<language>.json``)


class Candidate(_Strict):
    """A stretch of the character's own speech offered as a reference: an eligible candidate
    (``rank``, its place in the CD5 order: name-confirmed turns first, then by emotion and
    SNR) or a refused turn (``rank`` None). ``verdicts`` has every check that ran (True:
    passed): ``length``, ``overlap``, ``excluded``, ``foreign``, ``no_text``, ``sparse``
    (`fukikae.cards.select.REASONS`), the A1 gates ``off_register``, ``unvoiced`` and
    ``song_like``, and ``scene``, False when a better-ranked eligible candidate of the episode
    starts within ``SCENE_GAP_S`` (the automatic fill takes one clip per scene first; it
    refuses nothing, so it is never a reason). ``reasons`` names each failed check (``repeat``:
    a turn passing alone whose audio a joined candidate already holds; ``split``: the part of a
    joined acoustic candidate left outside the name-confirmed turns it overlapped, judged only
    as part of that whole; ``excluded`` on a cue span: a theme song or tagged singing).
    ``gender`` is the band of ``median``. ``clip`` is the file name under
    ``<ws>/studio/audio/``, cut when first fetched. ``source`` is ``cue`` for a character with
    no turn anywhere: its own spoken cues' source spans, what the cue-thin tier clones it
    from."""

    episode: str
    start: float
    end: float
    seconds: float
    transcript: str
    snr: float | None
    median: float | None
    gender: Literal["f", "m", "u"] | None
    rank: int | None
    verdicts: dict[str, bool]
    reasons: list[str]
    clip: str
    emotion: str | None = None
    confirmed: bool = False  # a name-confirmed turn of a names-used episode
    source: Literal["turn", "cue"] = "turn"

    def span(self) -> RefSpan:
        return RefSpan(episode=self.episode, start=self.start, end=self.end)


class Audition(_Strict):
    """One audition of a draft version (Task 19 renders them): the line, and what the render
    gave."""

    line: dict[str, Any]
    cue_key: str | None = None
    seconds: float | None = None
    suspect: bool | None = None
    passed: bool | None = None
    wav: str | None = None


class DraftVersion(_Strict):
    """One version of a voice draft: the clone card built from ``references`` with
    ``backend`` and ``params`` (the backend's voice keys), the references the A1 gates or the
    other checks refused that the operator chose anyway (``overrides``), the digest of the
    language's card assignment when it was made (`fukikae.studio.apply.row_digest`: what an
    accept of it carries), its tier (`fukikae.studio.drafts.floor_meter`) and its auditions.
    ``unlisted`` counts the card's references with no origin span: a version 1 opened from a
    card keeps the card's, and each later version carries the previous version's after its
    own unless the operator removes them; they are in its card, and listed nowhere else.
    ``thin_origin`` is the pipeline thin card whose references are exactly this version's
    (`fukikae.studio.apply.thin_origin`): the one selection exempt from the clean rule, at
    draft time and at accept. Version 1 (opened from a card or the automatic proposal) is
    checked at open by accept's own rules but never refused: ``acceptable`` says whether
    accept would take it and ``accept_problems`` gives accept's refusal reasons
    (`fukikae.studio.drafts.accept_problems`); later versions are refused at draft time, so
    they are always acceptable when made."""

    n: int = Field(ge=1)
    backend: str
    references: list[RefSpan]
    params: dict[str, Any]
    card_id: str
    overrides: list[RefSpan] = []
    created: str
    auditions: list[Audition] = []
    digest: str | None = None
    tier: Tier
    unlisted: int = Field(default=0, ge=0)
    thin_origin: str | None = None
    acceptable: bool = True
    accept_problems: list[str] = []


class Draft(_Strict):
    """A character's voice-tuning session for one target language: where version 1 came
    from (``card``: its current card; ``derived``: an accepted card of ``derived_from``;
    ``proposed``: the automatic rules, assigned nowhere), and every version since."""

    schema_version: Literal[1] = 1
    character_id: int
    language: str
    base: Literal["card", "derived", "proposed"]
    derived_from: str | None = None
    versions: list[DraftVersion]

    @property
    def name(self) -> str:
        return f"{self.character_id}.{self.language}"


__all__ = [
    "AcceptDecision",
    "AliasDecision",
    "Audition",
    "Candidate",
    "Draft",
    "DraftVersion",
    "CardInfo",
    "CharacterEntry",
    "Decision",
    "Decisions",
    "EpisodeEntry",
    "EpisodeSpeech",
    "ExtraEntry",
    "Flag",
    "GenderDecision",
    "GenderInfo",
    "IDENTITY_OPS",
    "KindDecision",
    "LockDecision",
    "MergeDecision",
    "NameDecision",
    "RefSpan",
    "ReviewedDecision",
    "Sample",
    "SeasonDoc",
    "SplitDecision",
    "Suggestion",
    "ValidationError",
    "parse_decision",
]
