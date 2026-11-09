"""Run only the diarize stage on one copy's episode, on its existing vocals_16k.wav.

Grid's separate and asr manifest keys predate GT6, so `fukikae run --to diarize` would redo
Demucs and ASR (and change the vocals the T9 sweep used). This runs DiarizeStage itself and
records its manifest entry with the current key; separate/asr records are left as they are.
usage: diar_direct.py <copy> <episode>
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from fukikae import job
from fukikae.stages.diarize import DiarizeStage
from fukikae.workspace import Workspace

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
ws = Workspace(WS / "t8" / "ws" / sys.argv[1])
ep = ws.find_episode(sys.argv[2])
st = DiarizeStage()
key = job.cache_key(st, ep, ws.config)
start = datetime.now(UTC)
st.run(ep, ws.config)
man = job.load_manifest(ep)
man.stages[st.name] = job.StageRecord(
    key=key, status="done", started=start, finished=datetime.now(UTC)
)
job.save_manifest(ep, man)
print(sys.argv[1], sys.argv[2], "diarized", (datetime.now(UTC) - start).seconds, "s")
