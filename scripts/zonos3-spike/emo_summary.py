import json
import statistics as st
import sys
from collections import defaultdict

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
src = {(r["set"], r["cue"]): r for r in rows if r["arm"] == "source"}
by = defaultdict(list)
for r in rows:
    if r["arm"] == "source":
        continue
    s = src[(r["set"], r["cue"])]
    by[r["arm"]].append((r, s))
print(
    "arm, n, label agrees with tag, render label non-neutral, sim to refs, dRMS dB vs plain, dF0 st vs plain"
)
plain = {(r["set"], r["cue"]): r for r in rows if r["arm"] == "zonos3-plain"}
import math

for arm, items in by.items():
    agree = sum(1 for r, s in items if r.get("render_emotion") == s["emotion"])
    nonneu = sum(1 for r, s in items if r.get("render_emotion") not in (None, "neutral"))
    sims = [r["sim_refs"] for r, s in items if "sim_refs" in r]
    drms = [r["rms_db"] - plain[(r["set"], r["cue"])]["rms_db"] for r, s in items]
    df0 = [
        12 * math.log2(r["f0_median"] / plain[(r["set"], r["cue"])]["f0_median"])
        for r, s in items
        if r.get("f0_median") and plain[(r["set"], r["cue"])].get("f0_median")
    ]
    print(
        arm,
        len(items),
        agree,
        nonneu,
        round(st.mean(sims), 3),
        round(st.mean(drms), 2),
        round(st.mean(df0), 2) if df0 else None,
    )
print("source labels:", sorted(defaultdict(int, {}).items()), [s["emotion"] for s in src.values()])
print(
    "source emotion2vec on the clip itself agrees with tag:",
    sum(1 for s in src.values() if s.get("render_emotion") == s["emotion"]),
    "of",
    len(src),
)
