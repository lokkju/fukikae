"""Task 8 Step 4: held-out workspaces on the committed live-action profile.

Each <set>-pick is a hard-linked copy of the ct0.5 sweep copy (same diarization of the tuning
episodes) whose fukikae.toml drops the [diarize] override, so every setting comes from the
committed profile. registry/ and 05-characters/ are removed (the sweep copies have none).
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
T = WS / "t8" / "ws"
SETS = [
    "bbt-en",
    "bbt-de",
    "bbt-es",
    "bazinga-tbbt-pal",
    "bazinga-tbbt-film",
    "bazinga-office-pal",
    "bazinga-office-film",
    "grid",
    "kaze",
]
for s in SETS:
    src, dst = T / f"{s}-ct0.5", T / f"{s}-pick"
    if dst.exists():
        continue
    subprocess.run(["cp", "-al", str(src), str(dst)], check=True)
    reg = dst / ".fukikae" / "registry"
    if reg.exists():
        shutil.rmtree(reg)
    for ch in (dst / ".fukikae" / "episodes").glob("*/05-characters"):
        shutil.rmtree(ch)
    toml = dst / "fukikae.toml"
    text = toml.read_text()
    toml.unlink()
    new = re.sub(r'\n\[diarize\]\nbackend = "pyannote"\nclustering_threshold = 0\.5\n', "\n", text)
    assert new != text, s
    toml.write_text(new)
    print(dst)
