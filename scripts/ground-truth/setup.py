"""Task 8 Step 1: hard-linked workspace copies per pyannote clustering threshold."""

import os
import shutil
import subprocess
from pathlib import Path

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
OUT = WS / "t8" / "ws"
SETS = [
    "bbt-en",
    "bbt-de",
    "bbt-es",
    "grid",
    "aib",
    "bazinga-tbbt-pal",
    "bazinga-tbbt-film",
    "bazinga-office-pal",
    "bazinga-office-film",
    "kaze",
]
CTS = [None, 0.5, 0.4]  # None = model default (community-1 ships 0.6)
GRID_EXTRA = 'profile = "live-action"\n\n[series]\ntitle = "Grid"\ngenre = "live-action"\n'
for s in SETS:
    src = WS / s
    for ct in CTS:
        dst = OUT / f"{s}-ct{'0.6' if ct is None else ct}"
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
        toml.unlink()  # break the hard link before writing
        if s == "grid":
            text = text.replace(
                "gpu_device = 0\n", "gpu_device = 0\n" + GRID_EXTRA.split("\n\n")[0] + "\n", 1
            )
            text += "\n" + GRID_EXTRA.split("\n\n", 1)[1]
        if ct is not None:
            text += f'\n[diarize]\nbackend = "pyannote"\nclustering_threshold = {ct}\n'
        toml.write_text(text)
        print(dst)
