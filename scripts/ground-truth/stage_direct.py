"""Run named stages directly on one copy's episode (grid only: its separate and asr manifest
keys predate GT6, so `fukikae run` would redo Demucs and ASR). Each stage records its manifest
entry with the current key; other records are left as they are.
usage: stage_direct.py <workspace dir> <episode> <stage>...   (stages: diarize, characters)
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from fukikae import job
from fukikae.stages.characters import CharactersStage
from fukikae.stages.diarize import DiarizeStage
from fukikae.workspace import Workspace

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
STAGES = {"diarize": DiarizeStage, "characters": CharactersStage}
ws = Workspace(Path(sys.argv[1]).expanduser())
ep = ws.find_episode(sys.argv[2])
for name in sys.argv[3:]:
    st = STAGES[name]()
    key = job.cache_key(st, ep, ws.config)
    start = datetime.now(UTC)
    st.run(ep, ws.config)
    man = job.load_manifest(ep)
    man.stages[st.name] = job.StageRecord(
        key=key, status="done", started=start, finished=datetime.now(UTC)
    )
    job.save_manifest(ep, man)
    print(
        ws.root.name,
        sys.argv[2],
        name,
        "done",
        (datetime.now(UTC) - start).seconds,
        "s",
        flush=True,
    )
