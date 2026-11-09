"""Speaker agreement between our transcript-aligned BBT labels and Bazinga's gold labels retimed
onto the same timeline (Task 7c's aligner check, 98.17% over 1958 s on S01E01-E03).

usage: agree.py [<ours dir> <bazinga dir> <aliases yaml>] [S01E01 ...]
Without the three paths it reads $FUKIKAE_WS (default ~/fukikae-ws): subs/bbt/labels,
subs/bazinga/tbbt-on-bbt-en and bbt-en/labels-aliases.yaml; without episodes, S01E01-E03.
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fukikae.eval.groundtruth import read_ground_truth
from fukikae.eval.groundtruth.base import load_alias_file

NAME_MAP = {
    "leonard_hofstadter": "Leonard",
    "sheldon_cooper": "Sheldon",
    "penny": "Penny",
    "howard_wolowitz": "Howard",
    "raj_koothrappali": "Raj",
    "leslie_winkle": "Lesley",
    "althea": "Receptionist",
    "kurt": "Enormous man",
    "instructor": "Class instructor",
    "doug": "Doug",
    "waitress": "Waitress",
}
H = 0.01
WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
args = sys.argv[1:]
if args and not args[0].startswith("S01E"):
    ours_dir, baz_dir, alias_file, eps = Path(args[0]), Path(args[1]), Path(args[2]), args[3:]
else:
    ours_dir, baz_dir = WS / "subs/bbt/labels", WS / "subs/bazinga/tbbt-on-bbt-en"
    alias_file, eps = WS / "bbt-en/labels-aliases.yaml", args
eps = eps or ["S01E01", "S01E02", "S01E03"]
aliases = load_alias_file(alias_file)


def frames(lines, n, mapper):
    owner = [set() for _ in range(n)]
    for ln in lines:
        name = mapper(ln.speaker)
        for i in range(int(round(ln.start / H)), min(int(round(ln.end / H)), n)):
            owner[i].add(name)
    return [next(iter(s)) if len(s) == 1 and "?" not in s else None for s in owner]


tot = Counter()
conf = defaultdict(Counter)
lines_tot = Counter()
bad_lines = []
for ep in eps:
    ours = read_ground_truth(ours_dir / f"{ep}.ass", "ass", aliases)
    baz = read_ground_truth(baz_dir / f"{ep}.ass", "ass", None)
    n = int(max(x.end for x in ours + baz) / H) + 2
    fo = frames(ours, n, lambda s: s)
    fb = frames(baz, n, lambda s: NAME_MAP.get(s, s))
    both = [(a, b) for a, b in zip(fo, fb) if a and b]
    agree = sum(a == b for a, b in both)
    ours_any = sum(1 for a in fo if a)
    baz_any = sum(1 for b in fb if b)
    for a, b in both:
        conf[b][a] += 1
    # speaker-agnostic mask lag between the two label sets (residual timing check)
    ma = np.array([a is not None for a in fo])
    mb = np.array([b is not None for b in fb])
    lags = {
        L: (ma[max(L, 0) : n + min(L, 0)] & mb[max(-L, 0) : n - max(L, 0)]).sum()
        for L in range(-100, 101, 2)
    }
    best = max(lags, key=lags.get)
    # line level: each of our lines against the gold speaker holding most of its time
    lag = agree_l = 0
    nl = 0
    for ln in ours:
        if ln.speaker == "?":
            continue
        span = [
            fb[i] for i in range(int(round(ln.start / H)), min(int(round(ln.end / H)), n)) if fb[i]
        ]
        if len(span) < 0.3 * max(1, (ln.end - ln.start) / H):
            continue
        top, cnt = Counter(span).most_common(1)[0]
        nl += 1
        if top == ln.speaker:
            agree_l += 1
        else:
            bad_lines.append((ep, ln.start, ln.end, ln.speaker, top, cnt / len(span)))
    tot["both"] += len(both)
    tot["agree"] += agree
    tot["lines"] += nl
    tot["lines_ok"] += agree_l
    print(
        f"{ep}: both-named {len(both) * H:.0f} s (ours {ours_any * H:.0f} s, Bazinga {baz_any * H:.0f} s); "
        f"time agreement {agree / len(both):.1%}; line agreement {agree_l}/{nl} ({agree_l / nl:.1%}); "
        f"mask lag ours->Bazinga {best * H:+.2f} s"
    )
print(
    f"ALL: time agreement {tot['agree'] / tot['both']:.2%} over {tot['both'] * H:.0f} s; "
    f"line agreement {tot['lines_ok']}/{tot['lines']} ({tot['lines_ok'] / tot['lines']:.1%})"
)
print("confusion (rows Bazinga, seconds by our label):")
for b in sorted(conf, key=lambda k: -sum(conf[k].values())):
    row = conf[b]
    t = sum(row.values())
    print(
        f"  {b:<16} {t * H:7.1f} s  agree {row[b] / t:6.1%}  "
        + ", ".join(f"{a} {c * H:.1f}" for a, c in row.most_common() if a != b)
    )
print("lines whose gold majority differs (ep, start, end, ours, gold, gold share):")
for r in bad_lines:
    print(f"  {r[0]} {r[1]:8.2f} {r[2]:8.2f}  {r[3]:<14} -> {r[4]:<14} {r[5]:.0%}")
