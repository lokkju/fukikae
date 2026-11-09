"""`fukikae eval report`: the results ledger as a matrix.

A matrix row is one configuration on one test set and split: (stage, series, test set,
split, backend, model, options hash, tag). Two series that share a test-set name stay apart,
and their labels then name the series (``set [series]``). A column is ``language/genre``. A
cell is the headline metric (or ``--metric``) of the latest row with that key in that column.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from fukikae.eval.ledger import STAGES, LedgerRow

HEADLINE = {
    "asr": "cer",
    "identity": "fraction",
    "embedder": "nc_overall",
    "tts": "speaker_similarity",
}
MISSING = "-"
# Shares in [0, 1] shown to two places, as `eval identity` prints them; others to four
# significant digits (a CER of 0.1234 keeps its digits).
_SHARES = frozenset({"fraction", "strict_fraction", "weighted_share", "coverage", "nc_overall"})


@dataclass(frozen=True)
class Matrix:
    columns: list[str]
    rows: list[tuple[str, list[str]]]


def filter_rows(
    rows: Iterable[LedgerRow],
    *,
    stages: Sequence[str] = (),
    backends: Sequence[str] = (),
    sets: Sequence[str] = (),
    splits: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> list[LedgerRow]:
    """Rows matching every non-empty filter (any value within one filter)."""
    return [
        r
        for r in rows
        if (not stages or r.stage in stages)
        and (not backends or r.backend in backends)
        and (not sets or r.test_set in sets)
        and (not splits or r.split in splits)
        and (not tags or r.tag in tags)
    ]


def _key(r: LedgerRow) -> tuple[str, ...]:
    return (
        r.stage,
        r.series,
        r.test_set,
        r.split,
        r.backend,
        r.model or "",
        r.options_hash,
        r.tag or "",
    )


def _label(r: LedgerRow, with_series: bool = False) -> str:
    who = f"{r.backend} {r.model}" if r.model else r.backend
    test_set = f"{r.test_set} [{r.series}]" if with_series else r.test_set
    return " | ".join(
        (r.stage, test_set, r.split, who, f"opts:{r.options_hash[:8]}", r.tag or MISSING)
    )


def _fmt(metric: str, value: float | int | str | None) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool | int | str):
        return str(value)
    return f"{value:.2f}" if metric in _SHARES else f"{value:.4g}"


def _cell(r: LedgerRow, metric: str | None) -> str:
    name = metric or HEADLINE.get(r.stage, "")
    text = _fmt(name, r.metrics.get(name))
    if r.stage == "identity" and name == "fraction" and text != MISSING:
        clean, merges = r.metrics.get("clean"), r.metrics.get("merges")
        if clean is not None and merges is not None:
            text += f" c{clean} m{merges}"
    return text


def _stage_order(stage: str) -> int:
    return STAGES.index(stage) if stage in STAGES else len(STAGES)


def build_matrix(rows: Sequence[LedgerRow], metric: str | None = None) -> Matrix:
    """Group rows by key and ``language/genre``; the latest row (by date, then by ledger
    order) fills each cell."""
    latest: dict[tuple[tuple[str, ...], str], tuple[str, int, LedgerRow]] = {}
    first: dict[tuple[str, ...], LedgerRow] = {}
    columns: set[str] = set()
    for i, r in enumerate(rows):
        key, col = _key(r), f"{r.language}/{r.genre}"
        columns.add(col)
        first.setdefault(key, r)
        prev = latest.get((key, col))
        if prev is None or (r.date, i) >= (prev[0], prev[1]):
            latest[(key, col)] = (r.date, i, r)
    # A test-set name used by more than one series gets the series in its labels.
    series_of_set: dict[str, set[str]] = {}
    for r in first.values():
        series_of_set.setdefault(r.test_set, set()).add(r.series)
    labels = {key: _label(r, len(series_of_set[r.test_set]) > 1) for key, r in first.items()}
    cols = sorted(columns)
    keys = sorted(labels, key=lambda k: (_stage_order(k[0]), labels[k]))
    out = []
    for key in keys:
        cells = []
        for col in cols:
            hit = latest.get((key, col))
            cells.append(MISSING if hit is None else _cell(hit[2], metric))
        out.append((labels[key], cells))
    return Matrix(cols, out)


def render(m: Matrix, fmt: Literal["text", "md"] = "text") -> str:
    header = ["row", *m.columns]
    body = [[label, *cells] for label, cells in m.rows]
    if fmt == "md":

        def line(cells: list[str]) -> str:
            return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

        return "\n".join([line(header), "|" + "---|" * len(header), *(line(r) for r in body)])
    widths = [max(len(r[i]) for r in (header, *body)) for i in range(len(header))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in (header, *body)
    )
