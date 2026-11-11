"""Score the theme-song detector on episodes' separation stems, with the committed settings.

Run: uv run --directory <checkout> python scripts/song_scores.py <episode work dir>...
(each dir holds 01-separate/background.wav, vocals.wav and vocals_16k.wav).

For each episode it prints the head and tail candidates (energy score, voicing share,
accepted or why not; ``tail2``, ``tail3``, ... are the repeated tail passes), the merged spans
the mix keeps, then two anime negatives run through the same window logic:
- ``cold open``: the best window before the accepted opening, when there is room for one;
- ``middle``: the best window between ``head_s`` and the last ``tail_s`` seconds (what the
  detector would do if the position limit didn't keep it out).
A negative ``would pass`` only when it clears both the score and the voicing bar.

The Task 7 measurements (docs/superpowers/ledgers/2026-11-08-character-cards.md) came from
~/fukikae-ws/{fairytail,kaze,bbt-en,bbt-de,grid}/.fukikae/episodes/*.
"""

import sys
from pathlib import Path

import numpy as np

from fukikae.songs import SongSettings, _gap, _spans, find_candidates, frame_features, voiced_share


def _best(
    good: np.ndarray, lo: int, hi: int, s: SongSettings, v16: Path
) -> tuple[int, float, float] | None:
    w = s.window_s
    if hi - lo < w:
        return None
    score = np.convolve(good[lo:hi], np.ones(w), "valid") / w
    i = int(np.argmax(score))
    start = lo + i
    return start, float(score[i]), voiced_share(v16, start, start + w, s.voiced_strength)


def _verdict(score: float, voiced: float, s: SongSettings) -> str:
    if score < s.min_score:
        return "below score"
    if voiced < s.voiced_share:
        return "refused by voicing"
    return "WOULD PASS"


def main(dirs: list[str]) -> int:
    s = SongSettings()
    print(f"bars: score >= {s.min_score}, voiced >= {s.voiced_share}, window {s.window_s} s")
    for d in dirs:
        sep = Path(d).expanduser() / "01-separate"
        bg, vo, act = frame_features(sep / "background.wav", sep / "vocals.wav", s.active_db)
        v16 = sep / "vocals_16k.wav"
        name = "/".join(Path(d).expanduser().parts[-4::3])
        cands = find_candidates(
            bg, vo, act, s, voiced=lambda a, b, v16=v16: voiced_share(v16, a, b, s.voiced_strength)
        )
        for c in cands:
            voiced = "-" if c.voiced is None else f"{c.voiced:.2f}"
            print(f"{name} {c.region}: {c.start:.0f}-{c.end:.0f} s score {c.score:.2f} "
                  f"voiced {voiced} {c.reason}")  # fmt: skip
        spans = ", ".join(f"{sp.start:.0f}-{sp.end:.0f}" for sp in _spans(cands, _gap(s)))
        print(f"{name} spans: {spans or 'none'}")
        good = ((act >= s.active_share) & (bg >= vo - s.music_margin_db)).astype(np.float64)
        dur = len(good)
        head = next((c for c in cands if c.region == "head" and c.accepted), None)
        negatives = {"middle": (s.head_s, dur - s.tail_s)}
        if head is not None:
            negatives["cold open"] = (0, int(head.start))
        for label, (lo, hi) in negatives.items():
            best = _best(good, lo, hi, s, v16)
            if best is None:
                print(f"{name} {label}: no room for a window")
                continue
            start, score, voiced = best
            print(f"{name} {label}: {start}-{start + s.window_s} s score {score:.2f} "
                  f"voiced {voiced:.2f} {_verdict(score, voiced, s)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
