"""Markdown tables for the subtitle speaker names results doc, from the arms' --json outputs.

    uv run --directory <repo> python scripts/names-eval/tables.py <out dir> <arms root> ja [en]

Reads <out dir>/<lang>-<arm>.{identity,names}.json and each arm's 04c-names/names.json.
"""

import json
import sys
from pathlib import Path

ARMS_A = ("baseline", "names", "ceiling")
ARMS_B = ("hidden-base", "hidden")
EPS = ("S01E11", "S01E12", "S01E13", "S01E14")


def load(out: Path, lang: str, arm: str, kind: str) -> dict | None:
    p = out / f"{lang}-{arm}.{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def identity_row(arm: str, split: str, s: dict) -> str:
    merges = s.get("merges") or []
    pairs = sorted({" + ".join(m[:-1]) for m in merges})
    return (
        f"| {arm} | {split} | {s['recurring']} | {s['passing']} | {s['fraction']:.2f} | "
        f"{len(pairs)} | {pct(s.get('weighted_share'))} | {s.get('strict_fraction', 0):.2f} |"
    )


def identity_table(out: Path, lang: str, arms: tuple[str, ...]) -> list[str]:
    lines = [
        "| arm | split | recurring | passing | fraction | merges | modal share | strict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        d = load(out, lang, arm, "identity")
        if d is None:
            continue
        for split in ("tune", "holdout", "ceiling"):
            part = d.get(split)
            if not part:
                continue
            parts = part if isinstance(part, list) else [part]
            for p in parts:
                label = split if split == "tune" else f"{split} {p.get('episode', '')}".strip()
                lines.append(identity_row(arm, label, p["score"]))
    return lines


def merges_line(out: Path, lang: str, arm: str, split: str) -> str:
    d = load(out, lang, arm, "identity")
    if d is None or not d.get(split):
        return ""
    part = d[split]
    p = part[0] if isinstance(part, list) else part
    ms = p["score"].get("merges") or []
    return "; ".join(f"{' + '.join(m[:-1])} (id {m[-1]})" for m in ms)


def per_character(out: Path, lang: str, arms: tuple[str, ...], split: str) -> list[str]:
    rows: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for arm in arms:
        d = load(out, lang, arm, "identity")
        if d is None:
            continue
        sp = "ceiling" if arm == "ceiling" and split == "holdout" else split
        part = d.get(sp)
        if not part:
            continue
        p = part[0] if isinstance(part, list) else part
        for c in p["score"].get("characters", []):
            if not c.get("recurring"):
                continue
            name = c["name"]
            if name not in rows:
                rows[name] = {}
                order.append(name)
            rows[name][arm] = f"{pct(c.get('share'))} (modal {c.get('modal_id', '-')})" + (
                "" if c.get("passing") else " F"
            )
    head = "| character | " + " | ".join(arms) + " |"
    lines = [head, "|---|" + "---|" * len(arms)]
    for n in order:
        lines.append(f"| {n} | " + " | ".join(rows[n].get(a, "-") for a in arms) + " |")
    return lines


def names_table(out: Path, lang: str, arms: tuple[str, ...]) -> list[str]:
    lines = [
        "| arm | split | cues | cue accuracy | hidden accuracy | purity | pure s | foreign s "
        "| unverifiable s (share) | max chars/name |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        d = load(out, lang, arm, "names")
        if d is None:
            continue
        cpn = d.get("characters_per_name") or {}
        over = ", ".join(f"{k} {v}" for k, v in sorted(cpn.items()) if v > 1) or "1"
        for split in ("tune", "holdout", "ceiling", "hidden"):
            s = d.get(split)
            if not s:
                continue
            lines.append(
                f"| {arm} | {split} | {s['cues']} | {pct(s['cue_accuracy'])} | "
                f"{pct(s.get('hidden_accuracy'))} | {pct(s['reference_purity'])} | "
                f"{s['pure_seconds']:.1f} | {s['foreign_seconds']:.1f} | "
                f"{s['unverifiable_seconds']:.1f} ({pct(s['unverifiable_share'])}) | {over} |"
            )
    return lines


def purity_per_character(out: Path, lang: str, arms: tuple[str, ...], split: str) -> list[str]:
    rows: dict[str, dict[str, str]] = {}
    for arm in arms:
        d = load(out, lang, arm, "names")
        sp = "ceiling" if arm == "ceiling" and split == "holdout" else split
        if d is None or not d.get(sp):
            continue
        for name, v in d[sp]["purity_seconds_per_character"].items():
            tot = v["pure_seconds"] + v["foreign_seconds"]
            share = pct(v["pure_seconds"] / tot) if tot else "-"
            val = f"{share} ({v['foreign_seconds']:.1f} s foreign)"
            rows.setdefault(name, {})[arm] = val
    lines = ["| character | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for n in sorted(rows):
        lines.append(f"| {n} | " + " | ".join(rows[n].get(a, "-") for a in arms) + " |")
    return lines


def coverage(root: Path, lang: str, arm: str) -> list[str]:
    lines = [
        "| episode | verdict | offset s | named share | sub-clusters | cut turns | dropped row s "
        "| unvoiced | hidden | suggestions |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in EPS:
        p = root / f"{lang}-{arm}" / ".fukikae" / "episodes" / e / "04c-names" / "names.json"
        if not p.exists():
            continue
        n = json.loads(p.read_text())
        cov = n.get("coverage") or {}
        sugg = n.get("suggestions") or []
        lines.append(
            f"| {e} | {n.get('verdict')} | {n.get('offset_s')} | {pct(cov.get('share'))} | "
            f"{cov.get('sub_clusters', '-')} | {cov.get('cut_turns', '-')} | "
            f"{n.get('dropped_row_seconds', 0):.1f} | {len(n.get('unvoiced') or [])} | "
            f"{len(n.get('hidden') or [])} | {', '.join(str(x) for x in sugg) or '-'} |"
        )
    return lines


def main() -> None:
    out, root = Path(sys.argv[1]), Path(sys.argv[2])
    for lang in sys.argv[3:]:
        print(f"\n## {lang}\n\n### Protocol A identity\n")
        print("\n".join(identity_table(out, lang, ARMS_A)))
        for arm in ARMS_A:
            for sp in ("holdout", "ceiling"):
                m = merges_line(out, lang, arm, sp)
                if m:
                    print(f"\n{arm} {sp} merges: {m}")
        print("\n### Held-out E14 per character\n")
        print("\n".join(per_character(out, lang, ARMS_A, "holdout")))
        print("\n### eval names, protocol A\n")
        print("\n".join(names_table(out, lang, ARMS_A)))
        print("\n### Purity per character, tune\n")
        print("\n".join(purity_per_character(out, lang, ARMS_A, "tune")))
        print("\n### Purity per character, E14\n")
        print("\n".join(purity_per_character(out, lang, ARMS_A, "holdout")))
        print("\n### Protocol B identity\n")
        print("\n".join(identity_table(out, lang, ARMS_B)))
        print("\n### eval names, protocol B\n")
        print("\n".join(names_table(out, lang, ARMS_B)))
        for arm in ("names", "ceiling", "hidden"):
            print(f"\n### Coverage, {arm}\n")
            print("\n".join(coverage(root, lang, arm)))


if __name__ == "__main__":
    main()
