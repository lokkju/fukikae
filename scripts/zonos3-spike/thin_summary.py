import json
import statistics as st
import sys
from collections import defaultdict

for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]
    by = defaultdict(dict)
    for r in rows:
        by[(r["card"], r["cue"])][r["arm"]] = r
    cards = defaultdict(lambda: defaultdict(list))
    for (card, cue), arms in by.items():
        for a, r in arms.items():
            cards[card][a].append(r)
    print(f)
    tot = defaultdict(list)
    for card, arms in cards.items():
        line = [card, str(next(iter(arms.values()))[0]["character"])]
        for a in ("indextts2", "zonos3-reference"):
            rs = arms.get(a, [])
            sr = [r.get("sim_refs") for r in rs if r.get("sim_refs") is not None]
            sh = [r.get("sim_heldout") for r in rs if r.get("sim_heldout") is not None]
            tot[a + " refs"] += sr
            tot[a + " held"] += sh
            line.append(
                f"{a}: refs {st.mean(sr):.3f}"
                + (f" held {st.mean(sh):.3f}" if sh else "")
                + f" secs {[round(r['seconds'], 2) for r in rs]}"
            )
        print("  ", " | ".join(line))
    for k, v in tot.items():
        if v:
            print("   mean", k, round(st.mean(v), 3), len(v))
