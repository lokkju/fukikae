"""Which stages up to characters would re-run in each <set>-pick copy (tuning + held-out)."""

import os
from pathlib import Path

from fukikae import job
from fukikae.stages.asr import AsrStage
from fukikae.stages.characters import CharactersStage
from fukikae.stages.diarize import DiarizeStage
from fukikae.stages.extract import ExtractStage
from fukikae.stages.language import LanguageStage
from fukikae.stages.separate import SeparateStage
from fukikae.workspace import Workspace

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
EPS = {
    "bbt": ["S01E01", "S01E02", "S01E03"],
    "grid": ["S01E02", "S01E04", "S01E05"],
    "bazinga": ["S01E01", "S01E02", "S01E03", "S01E05"],
}
for d in sorted((WS / "t8" / "ws").glob("*-pick")):
    ws = Workspace(d)
    for e in EPS[d.name.split("-")[0]]:
        ep = ws.find_episode(e)
        man = job.load_manifest(ep)
        stale = []
        for st in [
            ExtractStage(),
            SeparateStage(),
            LanguageStage(),
            AsrStage(),
            DiarizeStage(),
            CharactersStage(),
        ]:
            rec = man.stages.get(st.name)
            if (
                rec is None
                or rec.status != "done"
                or rec.key != job.cache_key(st, ep, ws.config)
                or any(not p.exists() for p in st.outputs(ep))
            ):
                stale.append(st.name)
        print(d.name, e, ",".join(stale) or "fresh", flush=True)
