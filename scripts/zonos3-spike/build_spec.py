"""Write /nvme/fukikae/tmp/zonos3/listen-spec.json from the eval renders and the spike's results.

uv run --directory <wt> python build_spec.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fukikae.cards.select import cut_clip
from fukikae.eval import tts as ev
from fukikae.workspace import Workspace

ROOT = Path("/nvme/fukikae/tmp/zonos3")
SETS = [("sb", "S00E01", "sol-bianca"), ("ft", "S01E12", "fairytail-e12")]
PER_SET = 10


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def source_clip(ws: Workspace, e: str, c) -> str:
    dest = ROOT / "renders" / "source" / ws.root.name / f"{e}-{c.id}.wav"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        cut_clip(ws.work_dir(e) / "01-separate" / "vocals.wav", *c.delivery.reference, dest)
    return str(dest)


def likeness_rows() -> list[dict]:
    rows = []
    for short, ep, set_name in SETS:
        ws = Workspace(ROOT / "ws" / short)
        base = ws.root / ".fukikae" / "eval" / "tts"
        idx = next(base.glob("indextts2-*"), None)
        zon = next(base.glob("zonos3-*"), None)
        if idx is None or zon is None:
            continue
        docs = ev._cues(ws, [ep], ws.config.target_language)
        picked = ev.sample_cues(docs, 8, 0, ev.MIN_SLOT_S)
        seen: set[int] = set()
        for e, c in picked:
            cid = ev._cid(c)
            if cid in seen:
                continue
            i = sorted(p for p in idx.glob(f"{e}-{c.id}.*.wav") if not p.name.endswith(".16k.wav"))
            # one file per cue and mode; the reference run came first, so the older is its
            zs = sorted(
                (p for p in zon.glob(f"{e}-{c.id}.*.wav") if not p.name.endswith(".16k.wav")),
                key=lambda p: p.stat().st_mtime,
            )
            if not i or len(zs) != 2:
                continue
            zr, ze = [zs[0]], [zs[1]]
            seen.add(cid)
            rows.append(
                {
                    "label": f"{set_name} {e} {c.id} (character {cid})",
                    "source_text": c.source_text,
                    "text": c.text or "",
                    "original": source_clip(ws, e, c),
                    "candidates": {
                        "indextts2": str(i[0]),
                        "zonos3-reference": str(zr[0]),
                        "zonos3-embedding": str(ze[0]),
                    },
                    "meta": {"set": set_name, "episode": e, "cue": c.id, "character": cid},
                }
            )
            if len([r for r in rows if r["meta"]["set"] == set_name]) >= PER_SET:
                break
    return rows


def grouped(arm_file: str, arms: list[str]) -> list[dict]:
    rows = []
    for short, ep, set_name in SETS:
        data = load(ROOT / "results" / f"{arm_file}-{set_name}.jsonl")
        by_cue: dict[str, dict] = {}
        for r in data:
            by_cue.setdefault(r["cue"], {})[r["arm"]] = r
        ws = Workspace(ROOT / "ws" / short)
        docs = ev._cues(ws, [ep], ws.config.target_language) if data else {}
        cues = {c.id: c for c in docs[ep].cues} if data else {}
        for cue, arms_got in by_cue.items():
            if not all(a in arms_got for a in arms):
                continue
            c = cues[cue]
            any_row = next(iter(arms_got.values()))
            label = f"{set_name} {ep} {cue} (character {c.character_id})"
            if any_row.get("emotion"):
                label += f", tagged {any_row['emotion']}"
            if arm_file == "pace":
                label += f", slot {c.end - c.start:.2f} s"
            rows.append(
                {
                    "label": label,
                    "source_text": c.source_text,
                    "text": c.text or "",
                    "original": arms_got["source"]["path"]
                    if "source" in arms_got
                    else source_clip(ws, ep, c),
                    "candidates": {a: arms_got[a]["path"] for a in arms},
                    "meta": {
                        "set": set_name,
                        "cue": cue,
                        "character": c.character_id,
                        **({"emotion": any_row.get("emotion")} if any_row.get("emotion") else {}),
                    },
                }
            )
    return rows


def main() -> None:
    sections = [
        {
            "id": "like",
            "title": "1. Likeness",
            "blind": True,
            "intro": "The same English line in each character's clone, one cue per character from eval tts's sample. ZONOS3 in embedding mode is included to hear why it fails (it drifts into Japanese-like speech).",
            "ask": "Which is most like the original speaker? Any unusable?",
            "rows": likeness_rows(),
        },
        {
            "id": "emo",
            "title": "2. Emotion transfer",
            "blind": True,
            "intro": "Lines the tag stage marked emotional. Candidates: IndexTTS2 with the line as emotion audio; "
            "ZONOS3 plain; ZONOS3 steered by the tag's emotion; ZONOS3 with the line itself added "
            "to the references.",
            "ask": "Which carries the original's delivery (shouting, crying, whispering)? Which keeps the voice?",
            "rows": grouped(
                "emotion", ["indextts2", "zonos3-plain", "zonos3-steer", "zonos3-lineref"]
            ),
        },
        {
            "id": "thin",
            "title": "3. Thin references and the shouted-only case",
            "blind": True,
            "intro": "Pipeline cards with under 10 s of references or in the thin tier, rendered from the "
            "same references by each engine.",
            "ask": "Which sounds like the speaker, and which is strained or broken?",
            "rows": grouped("thin", ["indextts2", "zonos3-reference"]),
        },
        {
            "id": "pace",
            "title": "4. Pace steering",
            "blind": False,
            "intro": "ZONOS3 without and with the speaking-rate row set from the slot (UTF-8 bytes over the "
            "slot length). Not blind: the label shows the slot.",
            "ask": "Does the paced take fit the slot without sounding rushed?",
            "rows": grouped("pace", ["zonos3-plain", "zonos3-pace"])[:12],
        },
    ]
    sections = [s for s in sections if s["rows"]]
    readme = (
        "ZONOS3-Mini spike listening page (lokkju/fukikae#43), Sol Bianca S00E01 and Fairy Tail E12, ja to en.\n\n"
        "Each row is one English dub cue. original.wav is the Japanese line (separated vocals). The "
        "candidates are every engine's render of the same line from the character's own references "
        "(D41: no stock voice anywhere). Blind sections shuffle the candidates per row; key.toml names "
        "them. Note a verdict per row, then open key.toml.\n\n"
        "Every clip is RMS-normalized to -23 dBFS (peak-limited) so loudness doesn't decide.\n"
        "Open index.html in a browser (the players read the wavs beside it).\n"
    )
    (ROOT / "listen-spec.json").write_text(
        json.dumps({"sections": sections, "readme": readme}, ensure_ascii=False, indent=1)
    )
    for s in sections:
        print(s["id"], len(s["rows"]))


if __name__ == "__main__":
    main()
