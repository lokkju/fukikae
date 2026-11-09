"""Tables for the Task 8 Step 1 report, from the per-run JSON files."""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
J = WS / "t8" / "json"
pat = re.compile(
    r"^(?P<set>.+)-ct(?P<ct>[0-9.]+)-(?P<seg>diarization|subtitles)-(?P<cond>centroid|joint-eclfalse|joint-ecltrue)-(?P<thr>[0-9.]+)\.json$"
)
rows = []
for p in sorted(J.glob("*.json")):
    m = pat.match(p.name)
    if not m:
        continue
    s = json.loads(p.read_text())["tune"]["score"]
    merged = {n for mm in s["merges"] for n in mm[:-1]}
    clean = sum(1 for c in s["characters"] if c["passing"] and c["name"] not in merged)
    rows.append(
        {
            **m.groupdict(),
            "clean": clean,
            "merges": len(s["merges"]),
            "frac": s["fraction"],
            "rec": s["recurring"],
            "pass": s["passing"],
            "cov": s["coverage"],
            "mlist": ["+".join(mm[:-1]) for mm in s["merges"]],
        }
    )
seg = sys.argv[1] if len(sys.argv) > 1 else "diarization"
SETS = [
    "bbt-en",
    "bbt-de",
    "bbt-es",
    "grid",
    "bazinga-tbbt-pal",
    "bazinga-tbbt-film",
    "bazinga-office-pal",
    "bazinga-office-film",
    "kaze",
    "aib",
]
CONDS = ["centroid", "joint-eclfalse", "joint-ecltrue"]
by = defaultdict(list)
for r in rows:
    if r["seg"] == seg:
        by[(r["set"], r["ct"])].append(r)
key = lambda r: (-r["clean"], r["merges"], -r["frac"])
if seg == "diarization":
    print("## Full grid: clean/merges/fraction per cell\n")
    for s in SETS:
        print(
            f"### {s}\n\n| ct | condition | "
            + " | ".join(f"{t:.2f}" for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
            + " |"
        )
        print("|---|---|" + "---|" * 6)
        for ct in ["0.6", "0.5", "0.4"]:
            for c in CONDS:
                cells = sorted((r for r in by[(s, ct)] if r["cond"] == c), key=lambda r: r["thr"])
                print(
                    f"| {ct} | {c} | "
                    + " | ".join(f"{r['clean']}/{r['merges']}/{r['frac']:.2f}" for r in cells)
                    + " |"
                )
        rec = {r["rec"] for ct in ["0.6", "0.5", "0.4"] for r in by[(s, ct)]}
        print(f"\nrecurring: {sorted(rec)}\n")
    print(
        "## Best cell per set and clustering threshold (pick rule: clean, then merges, then fraction)\n"
    )
    print("| set | ct | best clean/merges/fraction | cells tied at best | merges in best |")
    print("|---|---|---|---|---|")
    for s in SETS:
        for ct in ["0.6", "0.5", "0.4"]:
            rs = by[(s, ct)]
            if not rs:
                continue
            b = min(rs, key=key)
            ties = [r for r in rs if key(r) == key(b)]
            tie = ", ".join(sorted({f"{r['cond']}@{r['thr']}" for r in ties}))
            if len(ties) == 18:
                tie = "all 18"
            print(
                f"| {s} | {ct} | {b['clean']}/{b['merges']}/{b['frac']:.2f} | {tie} | {', '.join(b['mlist']) or '-'} |"
            )
    print("\n## Gate sums per (ct, condition, threshold), PAL bazinga only, aib excluded\n")
    GATE = ["bbt-en", "bbt-de", "bbt-es", "grid", "bazinga-tbbt-pal", "bazinga-office-pal", "kaze"]
    agg = defaultdict(lambda: [0, 0, 0.0, 0])
    for r in rows:
        if r["seg"] == "diarization" and r["set"] in GATE:
            a = agg[(r["ct"], r["cond"], r["thr"])]
            a[0] += r["clean"]
            a[1] += r["merges"]
            a[2] += r["frac"]
            a[3] += 1
    print("| ct | condition | thr | clean | merges | mean fraction | sets |")
    print("|---|---|---|---|---|---|---|")
    for k, a in sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[1][1], -kv[1][2])):
        print(f"| {k[0]} | {k[1]} | {k[2]} | {a[0]} | {a[1]} | {a[2] / a[3]:.3f} | {a[3]} |")
else:
    print(f"## Segmentation {seg} (diagnostic)\n")
    for s in ["bbt-en", "bbt-de", "bbt-es"]:
        print(
            f"### {s}\n\n| condition | "
            + " | ".join(f"{t:.2f}" for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
            + " |"
        )
        print("|---|" + "---|" * 6)
        for c in CONDS:
            cells = sorted((r for r in by[(s, "0.6")] if r["cond"] == c), key=lambda r: r["thr"])
            print(
                f"| {c} | "
                + " | ".join(
                    f"{r['clean']}/{r['merges']}/{r['frac']:.2f} ({'+'.join(sorted({x for m in r['mlist'] for x in m.split('+')})) or '-'})"
                    for r in cells
                )
                + " |"
            )
        print()
