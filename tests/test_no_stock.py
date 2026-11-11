"""D41: fukikae never renders a stock voice; no pipeline code path makes a stock card."""

import re
from pathlib import Path

import fukikae

SRC = Path(fukikae.__file__).parent
# eval baselines may make stock cards (eval is not the pipeline, R1); cards/stock.py defines
# the stock card for reading and for eval
ALLOWED = {SRC / "cards" / "stock.py"}
PATTERN = re.compile(r"\bstock_card\(|\bpick_stock\b|\bSTOCK_POOLS\b|\bextras_voices\b")


def test_pipeline_never_creates_stock_cards():
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        if path in ALLOWED or (SRC / "eval") in path.parents:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{path.relative_to(SRC)}:{n}: {line.strip()}")
    assert hits == []


def test_no_pipeline_code_writes_the_auto_fallback_stand_in():
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        if (SRC / "eval") in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"by\s*=\s*FALLBACK_BY|by=\"auto-fallback\"", text):
            hits.append(str(path.relative_to(SRC)))
    assert hits == []
