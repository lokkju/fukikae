"""Caption onsets against speech onsets for Kaze, Kaoru (Task 7b's check: the retimed captions
led speech by a median 0.24-0.31 s on E01-E03).

usage: kaze_onset.py <workspace> <episode> <caption file>...
Caption files are relative to $FUKIKAE_WS/subs/kaze-kaoru (default ~/fukikae-ws); the episode
needs its separate and diarize stages done in <workspace>. For each file it prints the median
lag (and IQR) from each gap-led cue's onset to the nearest rising edge of speech within 1.5 s,
by pyannote's turns and by an energy VAD on the vocal stem; positive means the caption leads.
"""

import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from fukikae.eval.autolabel import speech_turns
from fukikae.eval.groundtruth import subalign as sa
from fukikae.eval.groundtruth.base import read_cues
from fukikae.workspace import Workspace

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))
S = WS / "subs/kaze-kaoru"
ws = Workspace(Path(sys.argv[1]).expanduser())
ep = sys.argv[2]
files = sys.argv[3:]
work = ws.work_dir(ep)
turns = sorted(speech_turns(work), key=lambda t: t.start)


def rising(iv, bridge=0.3):
    out = []
    ce = -1e9
    for a, b in iv:
        if a > ce + bridge:
            out.append(a)
        ce = max(ce, b)
    return np.array(out)


pya = rising([(t.start, t.end) for t in turns])
# energy VAD on the vocal stem: 20 ms frames, threshold at -40 dBFS relative to the 95th pct
w, sr = sf.read(work / "01-separate/vocals_16k.wav", dtype="float32")
if w.ndim > 1:
    w = w.mean(1)
hop = int(0.02 * sr)
n = len(w) // hop
e = 10 * np.log10((w[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
thr = np.percentile(e, 95) - 30
on = e > thr
iv = []
i = 0
while i < n:
    if on[i]:
        j = i
        while j < n and on[j]:
            j += 1
        if (j - i) * 0.02 >= 0.1:
            iv.append((i * 0.02, j * 0.02))
        i = j
    else:
        i += 1
vad = rising(iv, 0.25)
print(ep, "pyannote onsets", len(pya), "energy-VAD onsets", len(vad))
for f in files:
    cues = [c for c in sa._merge_events(read_cues(S / f)) if c.end > c.start]
    # collapse the web .srt's split pieces: same text, contiguous
    m = []
    for c in cues:
        if m and c.text == m[-1][2] and abs(c.start - m[-1][1]) < 0.05:
            m[-1][1] = c.end
        else:
            m.append([c.start, c.end, c.text])
    st = [a for k, (a, b, t) in enumerate(m) if k == 0 or a - m[k - 1][1] >= 1.0]
    st = np.array(st)
    for name, edges in (("pyannote", pya), ("energy", vad)):
        lag = []
        for s in st:
            d = edges - s
            d = d[np.abs(d) <= 1.5]
            if len(d):
                lag.append(d[np.argmin(np.abs(d))])
        lag = np.array(lag)
        q = np.percentile(lag, [25, 50, 75])
        print(
            f"  {f[:40]:40} {name:8} gap-led cues {len(st)} matched {len(lag)} median {q[1]:+.2f} IQR {q[0]:+.2f}..{q[2]:+.2f} |lag|<=0.2: {np.mean(np.abs(lag) <= 0.2):.0%}"
        )
