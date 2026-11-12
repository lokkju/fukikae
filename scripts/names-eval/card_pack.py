"""Blind A/B listening pack for one character's old card against its new card (D32).

    uv run --directory <repo> python scripts/names-eval/card_pack.py \
        <old renders dir> <workspace> <episode> <out dir> <cue> [<cue> ...] \
        --slug young-makarov --title "Young Makarov" [--seed 0] [--readme README.txt]

``<old renders dir>`` holds the old renders (``<cue>.wav``) and the ``plan.json`` they came
from, saved before the tts stage re-rendered the cues; the new renders and their plan are read
from the workspace's episode. Each row gets the original Japanese line as audio (the tts
stage's emotion reference, ``07-tts/ref/<cue>.wav``) and as text (the ASR words inside the
cue), the English line, and the two renders under a shuffled A and B. The key (which letter
is the old card, balanced over the rows) is in key.toml, apart from index.html and
README.txt, in the same layout as ``listening_pack.py``'s packs.
"""

import argparse
import html
import json
import random
import shutil
from pathlib import Path

from fukikae.artifacts import read_json
from fukikae.cards.select import turn_text
from fukikae.models import Transcript
from fukikae.transcripts import transcript_path


def plan(path: Path) -> dict[str, dict]:
    return {p["cue_id"]: p for p in json.loads(path.read_text())["cues"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("episode")
    ap.add_argument("out", type=Path)
    ap.add_argument("cues", nargs="+")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readme", type=Path, help="README text; {rows} is the row count")
    args = ap.parse_args()
    rng = random.Random(args.seed)  # noqa: S311 (a reproducible shuffle, not a secret)
    work = args.workspace / ".fukikae" / "episodes" / args.episode
    cues = {
        c["id"]: c for c in json.loads((work / "06-translate" / "cues.en.json").read_text())["cues"]
    }
    transcript = read_json(transcript_path(work), Transcript)
    old_plan, new_plan = plan(args.old / "plan.json"), plan(work / "07-tts" / "plan.json")
    args.out.mkdir(parents=True, exist_ok=True)
    key_lines, rows = [], []
    # balanced: as many rows with the old card as A as with it as B, in a shuffled order
    orders = [["old", "new"], ["new", "old"]] * ((len(args.cues) + 1) // 2)
    orders = orders[: len(args.cues)]
    rng.shuffle(orders)
    for i, cue in enumerate(args.cues, 1):
        c, o, n = cues[cue], old_plan[cue], new_plan[cue]
        if o["card_id"] == n["card_id"]:
            raise SystemExit(f"{cue}: the old and new renders share card {o['card_id']}")
        tag = f"{args.slug}-{i:02d}"
        d = args.out / tag
        d.mkdir(exist_ok=True)
        order = orders[i - 1]
        src = {"old": args.old / f"{cue}.wav", "new": work / "07-tts" / f"{cue}.wav"}
        for letter, which in zip("AB", order, strict=True):
            shutil.copyfile(src[which], d / f"{letter}.wav")
        ref = work / "07-tts" / "ref" / f"{cue}.wav"
        if ref.exists():
            shutil.copyfile(ref, d / "original.wav")
        ja = turn_text(transcript, c["start"], c["end"]) or "-"
        key_lines.append(
            f'[{json.dumps(tag)}]\nA = "{order[0]}"\nB = "{order[1]}"\n'
            f'episode = "{args.episode}"\ncue = "{cue}"\n'
            f'old = "character {o["character_id"]}, card {o["card_id"]}"\n'
            f'new = "character {n["character_id"]}, card {n["card_id"]}"\n'
        )
        orig = (
            f'<audio controls preload="none" src="{tag}/original.wav"></audio>'
            if ref.exists()
            else "-"
        )
        rows.append(
            f"<tr><td>{tag}</td><td>{args.episode} {cue}</td><td lang=ja>{html.escape(ja)}</td>"
            f"<td>{orig}</td><td>{html.escape(c['text'])}</td>"
            f'<td><audio controls preload="none" src="{tag}/A.wav"></audio></td>'
            f'<td><audio controls preload="none" src="{tag}/B.wav"></audio></td>'
            "<td>A / B / same</td></tr>"
        )
    title = html.escape(args.title)
    (args.out / "key.toml").write_text("\n".join(key_lines), encoding="utf-8")
    (args.out / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>{title} card check</title>"
        "<style>body{font-family:sans-serif;margin:16px}td{padding:4px 8px;"
        "border-bottom:1px solid #ccc;vertical-align:top}</style>"
        f"<h1>{title}: card listening check (blind)</h1><p>See README.txt. key.toml holds "
        "which of A and B is the old card; open it only after noting a verdict per row.</p>"
        "<table><tr><th>row</th><th>cue</th><th>original line (Japanese)</th>"
        "<th>original audio</th><th>English line</th><th>A</th><th>B</th>"
        "<th>which sounds better (A / B / same)</th></tr>" + "".join(rows) + "</table>",
        encoding="utf-8",
    )
    if args.readme:
        text = args.readme.read_text(encoding="utf-8").format(rows=len(rows))
        (args.out / "README.txt").write_text(text, encoding="utf-8")
    print(f"{len(rows)} rows in {args.out}")


if __name__ == "__main__":
    main()
