"""Task 8 Step 1 matcher sweep: in-process `fukikae eval identity` calls (ledger rows, JSON).

usage: sweep.py <copy-name>... [--seg subtitles]
"""

import json
import os
import sys
import time
from pathlib import Path

import typer

from fukikae.cli import app

WS = Path(os.environ.get("FUKIKAE_WS", Path.home() / "fukikae-ws"))

T8 = WS / "t8"
THR = ["0.50", "0.55", "0.60", "0.65", "0.70", "0.75"]
CONDS = [("centroid", None), ("joint", "false"), ("joint", "true")]


def flags(copy: str) -> tuple[list[str], str, str]:
    base, ct = copy.rsplit("-ct", 1)
    if base.startswith("bbt-"):
        return (
            [
                "--tune",
                "S01E01,S01E02",
                "--gt",
                "ass",
                "--gt-file",
                str(WS / "subs/bbt/labels"),
                "--set",
                "bbt",
            ],
            f"t8-sweep-ct{ct}",
            ct,
        )
    if base == "grid":
        return (
            ["--tune", "S01E02,S01E04", "--gt", "sdh", "--subtitle-track", "2", "--set", "grid"],
            f"t8-sweep-ct{ct}",
            ct,
        )
    if base == "kaze":
        return (
            [
                "--tune",
                "S01E01,S01E02,S01E03",
                "--gt",
                "colour",
                "--gt-file",
                str(WS / "subs/kaze-kaoru/aligned"),
                "--force",
                "--set",
                "kaze",
            ],
            f"t8-sweep-ct{ct}",
            ct,
        )
    if base == "aib":
        return (
            ["--tune", "S01E01,S01E02", "--gt", "sdh", "--force", "--set", "aib"],
            f"t8-diag-aib-ct{ct}",
            ct,
        )
    if base.startswith("bazinga-"):
        _, show, cond = base.split("-")
        return (
            [
                "--tune",
                "S01E01,S01E02,S01E03",
                "--gt",
                "ass",
                "--gt-file",
                str(WS / f"subs/bazinga/{show}-{cond}"),
                "--set",
                f"bazinga-{show}",
            ],
            f"t8-sweep-ct{ct}-{cond}",
            ct,
        )
    raise SystemExit(f"unknown copy {copy}")


def main(argv: list[str]) -> None:
    seg = "diarization"
    if "--seg" in argv:
        i = argv.index("--seg")
        seg = argv[i + 1]
        argv = argv[:i] + argv[i + 2 :]
    cmd = typer.main.get_command(app)
    for copy in argv:
        fl, tag, ct = flags(copy)
        if seg != "diarization":
            tag = f"t8-diag-{seg}"
        for matcher, ecl in CONDS:
            for thr in THR:
                name = f"{copy}-{seg}-{matcher}{'' if ecl is None else '-ecl' + ecl}-{thr}"
                out = T8 / "json" / f"{name}.json"
                args = [
                    "eval",
                    "identity",
                    str(T8 / "ws" / copy),
                    *fl,
                    "--segmentation",
                    seg,
                    "--embedder",
                    "wespeaker",
                    "--matcher",
                    matcher,
                    "--threshold",
                    thr,
                    "--tag",
                    tag,
                    "--json",
                    str(out),
                ]
                if ecl is not None:
                    args += ["--matcher-option", f"episode_cannot_link={ecl}"]
                t0 = time.time()
                try:
                    import contextlib
                    import io

                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        cmd.main(args, standalone_mode=False)
                except SystemExit as e:
                    if e.code:
                        print(name, "EXIT", e.code, flush=True)
                        continue
                d = json.loads(out.read_text())
                s = d["tune"]["score"]
                merged = {n for m in s["merges"] for n in m[:-1]}
                clean = sum(1 for c in s["characters"] if c["passing"] and c["name"] not in merged)
                print(
                    name,
                    f"{time.time() - t0:.0f}s",
                    f"rec={s['recurring']} pass={s['passing']}",
                    f"clean={clean} merges={len(s['merges'])} frac={s['fraction']:.2f}",
                    f"cov={s['coverage']:.2f}",
                    s["merges"],
                    flush=True,
                )


if __name__ == "__main__":
    main(sys.argv[1:])
