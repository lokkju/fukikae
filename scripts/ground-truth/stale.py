"""Which stages up to diarize would re-run in each t8 copy (tuning episodes only)."""

import os
from pathlib import Path

from fukikae import job
from fukikae.stages.asr import AsrStage
from fukikae.stages.diarize import DiarizeStage
from fukikae.stages.extract import ExtractStage
from fukikae.stages.language import LanguageStage
from fukikae.stages.separate import SeparateStage
from fukikae.workspace import Workspace

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
TUNE = {
    "bbt": ["S01E01", "S01E02"],
    "grid": ["S01E02", "S01E04"],
    "aib": ["S01E01", "S01E02"],
    "bazinga": ["S01E01", "S01E02", "S01E03"],
    "kaze": ["S01E01", "S01E02", "S01E03"],
}
for d in sorted((WS / "t8" / "ws").iterdir()):
    eps = TUNE[d.name.split("-")[0]]
    ws = Workspace(d)
    for e in eps:
        ep = ws.find_episode(e)
        man = job.load_manifest(ep)
        stale = []
        for st in [ExtractStage(), SeparateStage(), LanguageStage(), AsrStage(), DiarizeStage()]:
            rec = man.stages.get(st.name)
            if (
                rec is None
                or rec.status != "done"
                or rec.key != job.cache_key(st, ep, ws.config)
                or any(not p.exists() for p in st.outputs(ep))
            ):
                stale.append(st.name)
        print(d.name, e, ",".join(stale) or "fresh", flush=True)
