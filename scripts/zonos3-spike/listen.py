"""Build the spike's listening page under /nvme/fukikae/tmp/zonos3/listen/ (index.html, key.toml,
README.txt, one folder per row with original.wav and blinded A.wav, B.wav, ...).

    python3 listen.py   (stdlib + numpy + soundfile; run in the worktree env)

Sections: likeness (eval tts renders of the same cues), emotion, thin and shouted-only, pace.
The candidates in a row are shuffled with a seed kept only in key.toml.
"""

from __future__ import annotations

import html
import json
import random
import secrets
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/nvme/fukikae/tmp/zonos3")
OUT = ROOT / "listen"
TARGET_RMS_DB = -23.0
LABELS = "ABCDEFG"


def norm_copy(src: Path, dest: Path) -> None:
    y, sr = sf.read(str(src), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    rms = np.sqrt(np.mean(y**2)) + 1e-9
    g = 10 ** (TARGET_RMS_DB / 20) / rms
    peak = np.max(np.abs(y)) * g
    if peak > 0.98:
        g *= 0.98 / peak
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), y * g, sr, subtype="PCM_16")


def load(path: Path) -> list[dict]:
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    seed = secrets.randbits(64)
    sections: list[tuple[str, str, list[dict]]] = []
    spec = json.loads((ROOT / "listen-spec.json").read_text())
    key_lines = [f"seed = {seed}\n"]
    body = []
    for sec in spec["sections"]:
        rows_html = []
        for i, row in enumerate(sec["rows"], 1):
            rid = f"{sec['id']}-{i:02d}"
            folder = OUT / rid
            if row.get("original"):
                norm_copy(Path(row["original"]), folder / "original.wav")
            cands = list(row["candidates"].items())
            if sec.get("blind", True):
                random.Random(f"{seed}:{rid}").shuffle(cands)
            key_lines.append(f'\n["{rid}"]\n')
            for k, v in row.get("meta", {}).items():
                key_lines.append(f"{k} = {json.dumps(v, ensure_ascii=False)}\n")
            cells = []
            for lab, (name, path) in zip(LABELS, cands):
                norm_copy(Path(path), folder / f"{lab}.wav")
                key_lines.append(f'{lab} = "{name}"\n')
                title = "" if sec.get("blind", True) else f"<br><small>{html.escape(name)}</small>"
                cells.append(
                    f'<td><audio controls preload="none" src="{rid}/{lab}.wav"></audio>{title}</td>'
                )
            orig = (
                f'<td><audio controls preload="none" src="{rid}/original.wav"></audio></td>'
                if row.get("original")
                else "<td></td>"
            )
            rows_html.append(
                f"<tr><td>{rid}</td><td>{html.escape(row.get('label', ''))}</td>"
                f"<td lang=ja>{html.escape(row.get('source_text', ''))}</td>{orig}"
                f"<td>{html.escape(row.get('text', ''))}</td>{''.join(cells)}"
                f"<td>{html.escape(sec.get('ask', ''))}</td></tr>"
            )
        n = max(len(r["candidates"]) for r in sec["rows"])
        heads = "".join(f"<th>{LABELS[j]}</th>" for j in range(n))
        body.append(
            f"<h2>{html.escape(sec['title'])}</h2><p>{html.escape(sec['intro'])}</p>"
            f"<table><tr><th>row</th><th>cue</th><th>original line (Japanese)</th><th>original audio</th>"
            f"<th>English line</th>{heads}<th>verdict</th></tr>{''.join(rows_html)}</table>"
        )
    page = (
        "<!doctype html><meta charset=utf-8><title>ZONOS3-Mini spike</title>"
        "<style>body{font-family:sans-serif;margin:16px}td{padding:4px 8px;border-bottom:1px solid #ccc;"
        "vertical-align:top}small{color:#555}</style>"
        "<h1>ZONOS3-Mini against IndexTTS2: listening checks (lokkju/fukikae#43)</h1>"
        "<p>See README.txt. Blind sections shuffle the candidates per row; key.toml names them. "
        "Note a verdict per row before opening the key.</p>" + "".join(body)
    )
    (OUT / "index.html").write_text(page)
    (OUT / "key.toml").write_text("".join(key_lines))
    (OUT / "README.txt").write_text(spec["readme"])
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
