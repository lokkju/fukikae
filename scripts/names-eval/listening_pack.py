"""Blind before/after listening pack for the names arm's re-rendered cues (D29 #9).

    uv run --directory <repo> python scripts/names-eval/listening_pack.py \
        <before ws> <after ws> <out dir> NATSU="YOUNG MAKAROV" ... [--per 10] [--seed 0]

For each ground-truth name (the subtitle Name of the cue), picks up to --per cues of S01E12 to
S01E14 whose text is the same in both workspaces and whose card changed, longest first spread
over the episodes. Each cue gets the original line (the tts stage's emotion reference, the
source-language audio of that cue), and the two renders under a shuffled A and B. The key is
in key.toml, apart from index.html and README.txt.
"""

import argparse
import html
import json
import random
import shutil
from pathlib import Path

EPS = ("S01E12", "S01E13", "S01E14")


def ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def ass_lines(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        out.append((ts(parts[1]), ts(parts[2]), parts[4].strip().upper()))
    return out


def gt_name(lines: list[tuple[float, float, str]], a: float, b: float) -> str | None:
    best, cover = None, 0.0
    for s, e, n in lines:
        ov = min(b, e) - max(a, s)
        if ov > cover:
            best, cover = n, ov
    return best if cover >= 0.5 * (b - a) else None


def load(ws: Path, ep: str) -> tuple[dict, dict, Path]:
    work = ws / ".fukikae" / "episodes" / ep
    cues = {
        c["id"]: c for c in json.loads((work / "06-translate" / "cues.en.json").read_text())["cues"]
    }
    plan = {p["cue_id"]: p for p in json.loads((work / "07-tts" / "plan.json").read_text())["cues"]}
    return cues, plan, work


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--per", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)  # noqa: S311 (a reproducible shuffle, not a secret)
    args.out.mkdir(parents=True, exist_ok=True)
    picks: dict[str, list[dict]] = {n.upper(): [] for n in args.names}
    for ep in EPS:
        bc, bp, bwork = load(args.before, ep)
        ac, ap_, awork = load(args.after, ep)
        lines = ass_lines(awork / "06-translate" / "subs.en.ass")
        for cid, c in ac.items():
            n = gt_name(lines, c["start"], c["end"])
            if n not in picks or cid not in bc or cid not in bp or cid not in ap_:
                continue
            b, a = bp[cid], ap_[cid]
            if bc[cid]["text"] != c["text"] or b["card_id"] == a["card_id"]:
                continue
            bw, aw = bwork / "07-tts" / f"{cid}.wav", awork / "07-tts" / f"{cid}.wav"
            ref = awork / "07-tts" / "ref" / f"{cid}.wav"
            if not (bw.exists() and aw.exists()):
                continue
            picks[n].append(
                {
                    "episode": ep,
                    "cue": cid,
                    "text": c["text"],
                    "seconds": c["end"] - c["start"],
                    "before_character": b["character_id"],
                    "before_card": b["card_id"],
                    "after_character": a["character_id"],
                    "after_card": a["card_id"],
                    "before": bw,
                    "after": aw,
                    "ref": ref if ref.exists() else None,
                }
            )
    key_lines, rows, readme_counts = [], [], []
    for name, cands in picks.items():
        by_ep = {
            ep: sorted((c for c in cands if c["episode"] == ep), key=lambda c: -c["seconds"])
            for ep in EPS
        }
        chosen: list[dict] = []
        while len(chosen) < args.per and any(by_ep.values()):
            for ep in EPS:
                if by_ep[ep] and len(chosen) < args.per:
                    chosen.append(by_ep[ep].pop(0))
        readme_counts.append(f"{name}: {len(chosen)} of {len(cands)} changed cues")
        slug = name.lower().replace(" ", "-")
        for i, c in enumerate(chosen, 1):
            tag = f"{slug}-{i:02d}"
            d = args.out / tag
            d.mkdir(exist_ok=True)
            order = ["before", "after"]
            rng.shuffle(order)
            for letter, which in zip("AB", order, strict=True):
                shutil.copyfile(c[which], d / f"{letter}.wav")
            if c["ref"]:
                shutil.copyfile(c["ref"], d / "original.wav")
            key_lines.append(
                f'[{json.dumps(tag)}]\nA = "{order[0]}"\nB = "{order[1]}"\n'
                f'episode = "{c["episode"]}"\ncue = "{c["cue"]}"\n'
                f'before = "character {c["before_character"]}, card {c["before_card"]}"\n'
                f'after = "character {c["after_character"]}, card {c["after_card"]}"\n'
            )
            orig = (
                f'<audio controls preload="none" src="{tag}/original.wav"></audio>'
                if c["ref"]
                else "-"
            )
            rows.append(
                f"<tr><td>{tag}</td><td>{html.escape(name)}</td><td>{c['episode']} {c['cue']}</td>"
                f"<td>{html.escape(c['text'])}</td><td>{orig}</td>"
                f'<td><audio controls preload="none" src="{tag}/A.wav"></audio></td>'
                f'<td><audio controls preload="none" src="{tag}/B.wav"></audio></td>'
                f"<td>A / B / same</td></tr>"
            )
    (args.out / "key.toml").write_text("\n".join(key_lines), encoding="utf-8")
    (args.out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>names listening check</title>"
        "<style>body{font-family:sans-serif;margin:16px}td{padding:4px 8px;"
        "border-bottom:1px solid #ccc;vertical-align:top}</style>"
        "<h1>Names listening check (blind)</h1><p>See README.txt. key.toml holds which of A and "
        "B is the render before names; open it only after noting a verdict per row.</p>"
        "<table><tr><th>row</th><th>subtitle name</th><th>cue</th><th>text</th>"
        "<th>original line</th><th>A</th><th>B</th><th>which matches the original</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )
    print("\n".join(readme_counts))


if __name__ == "__main__":
    main()
