"""VAD onsets against label turns: lag and drift (Task 7c's film-speed check).

usage: vadcheck.py <audio> <labels.ass>, e.g. with WS=${FUKIKAE_WS:-~/fukikae-ws}:
  uv run --with webrtcvad --with 'setuptools<81' python vadcheck.py \
    "$WS/media/bazinga-tbbt-pal/TheBigBangTheory - S01E01.mkv" "$WS/subs/bazinga/tbbt-pal/S01E01.ass"
ffmpeg comes from mise (run under `mise -C <repo> exec --`).
"""

import subprocess
import sys

import numpy as np
import webrtcvad

audio, ass = sys.argv[1], sys.argv[2]
raw = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", audio, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
    check=True,
    capture_output=True,
).stdout
x = np.frombuffer(raw, np.int16)
sr = 16000
H = 0.01
n = len(x) // 160
vad = webrtcvad.Vad(3)
v = np.zeros(n, bool)
for i in range(0, len(x) - 480, 160):
    if vad.is_speech(x[i : i + 480].tobytes(), sr):
        v[i // 160 : i // 160 + 3] = True
on = np.flatnonzero(v[1:] & ~v[:-1]) * H


def sec(s):
    h, m, r = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(r)


T = []
for ln in open(ass, encoding="utf-8"):
    if ln.startswith("Dialogue:"):
        p = ln.split(",", 9)
        T.append((sec(p[1]), sec(p[2])))
starts = np.array([t[0] for t in T])
d = np.array([on[np.argmin(np.abs(on - s))] - s for s in starts])
r = np.random.default_rng(0).uniform(0, n * H, 2000)
dn = np.array([np.min(np.abs(on - t)) for t in r])
print(f"audio {n * H:.1f}s turns {len(T)} last end {max(t[1] for t in T):.1f}s")
print(
    f"turn starts: |onset-start| median {np.median(abs(d)):.3f}s, within 0.2s {100 * np.mean(abs(d) < 0.2):.0f}% "
    f"(random times {100 * np.mean(dn < 0.2):.0f}%), signed median {np.median(d):+.3f}s"
)
a = np.zeros(n, bool)
for s, e in T:
    a[int(s / H) : min(int(e / H) + 1, n)] = True


def best_lag(lo, hi, maxlag):
    best = (-1e9, 0)
    for L in range(-maxlag, maxlag + 1):
        i0, i1 = max(lo, lo + L, 0), min(hi, hi + L, n)
        aa = a[i0 - L : i1 - L]
        b = v[i0:i1]
        s = np.logical_and(aa, b).sum() - 0.5 * np.logical_and(aa, ~b).sum()
        if s > best[0]:
            best = (s, L)
    return best[1] * H


print(
    f"label frames VAD-speech {np.logical_and(a, v).sum() / a.sum():.3f} (VAD speech share {v.mean():.3f})"
)
print(f"global lag {best_lag(0, n, 300):+.2f}s")
W = int(120 / H)
mids, lags = [], []
for lo in range(0, n - W, W // 2):
    if a[lo : lo + W].sum() < 300:
        continue
    mids.append((lo + W / 2) * H)
    lags.append(best_lag(lo, lo + W, 200))
print("  lags", " ".join(f"{m:.0f}:{l:+.2f}" for m, l in zip(mids, lags)))
mids, lags = np.array(mids), np.array(lags)
ok = np.abs(lags - np.median(lags)) < 0.3
k, c = np.polyfit(mids[ok], lags[ok], 1)
print(
    f"robust fit over {ok.sum()} of {len(ok)} windows (|lag - median| < 0.3 s), median lag {np.median(lags):+.2f}s"
)
print(
    f"windows {len(mids)}: lag ~ {k:+.6f}*t {c:+.3f}s; window lags {min(lags):+.2f}..{max(lags):+.2f}s"
)
