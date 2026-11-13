import json
import math
import statistics as st
import sys

for set_name in sys.argv[1:]:
    pace = [json.loads(l) for l in open(f"/nvme/fukikae/tmp/zonos3/results/pace-{set_name}.jsonl")]
    slot = {r["cue"]: r["slot"] for r in pace}
    arms = {}
    for r in pace:
        arms.setdefault(r["arm"], {})[r["cue"]] = r["seconds"]
    for l in open(f"/nvme/fukikae/tmp/zonos3/results/speed-{set_name}.jsonl"):
        s = json.loads(l)
        if s["arm"].startswith("indextts2"):
            arms["indextts2"] = {r["cue"]: r["seconds"] for r in s["rows"]}
    common = set.intersection(*(set(v) for v in arms.values()))
    print(set_name, "cues", len(common))
    for a, d in arms.items():
        rs = [d[c] / slot[c] for c in common]
        print(
            f"  {a:14s} median {st.median(rs):.2f}  over slot {sum(x > 1 for x in rs):2d}  over 1.25x {sum(x > 1.25 for x in rs):2d}  under 0.7 {sum(x < 0.7 for x in rs):2d}  mean|log| {st.mean(abs(math.log(x)) for x in rs):.3f}  max {max(rs):.2f}"
        )
