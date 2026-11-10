"""Would the listening pack's cues get another character or card in a rebuilt names arm?

    uv run --directory <repo> python scripts/names-eval/pack_check.py \
        <key.toml> <rendered arm> <rebuilt arm>

The rendered arm is the one the pack was built from (its ``06-translate`` cues and ``07-tts``
renders); the rebuilt arm went through names, characters and resolve again on newer code but
not through translate or tts. Each pack cue is attributed in both arms the subtitle path's way
(`fukikae.eval.names.attributed_cues` over the track translate read, cached by the eval under
``.fukikae/cache/subtitles``); the rendered arm's attribution must agree with its own
``cues.en.json``, or the check refuses. A cue whose character is the same in both arms is
compared on that character's reference picks (`character_references` over E11 to E14), the
clips an automatic card would be cloned from.
"""

import json
import sys
import tomllib
from pathlib import Path

from fukikae.artifacts import read_json
from fukikae.cards.auto import character_references
from fukikae.eval.names import _Episode, attributed_cues
from fukikae.models import Assignments
from fukikae.names.inputs import names_path
from fukikae.names.models import NamesResult
from fukikae.workspace import Workspace

EPS = ["S01E11", "S01E12", "S01E13", "S01E14"]


def attribute(ws: Workspace, eid: str) -> dict[str, int | None]:
    work = ws.work_dir(eid)
    asg = read_json(work / "05-characters" / "assignments.json", Assignments)
    npath = names_path(work)
    names = read_json(npath, NamesResult) if npath.exists() else None
    track = ws.root / ".fukikae" / "cache" / "subtitles" / eid / "eval" / "subtitles.ass"
    e = _Episode(eid, work, asg, [], track, names, [])
    return {c.id: c.character_id for c in attributed_cues(ws, ws.config, e)}


_REFS: dict[tuple[str, int], object] = {}


def refs(ws: Workspace, cid: int | None) -> object:
    if cid is None:
        return None
    k = (str(ws.root), cid)
    if k not in _REFS:
        _REFS[k] = _refs(ws, cid)
    return _REFS[k]


def _refs(ws: Workspace, cid: int) -> object:
    got = character_references(ws, EPS, cid)
    if got is None:
        return None
    ep, picks, confirmed = got
    return ep, [(round(p.start, 2), round(p.end, 2)) for p in picks], confirmed


def main() -> int:
    key = tomllib.loads(Path(sys.argv[1]).read_text())
    old, new = Workspace(Path(sys.argv[2])), Workspace(Path(sys.argv[3]))
    by_ep: dict[str, list[tuple[str, str]]] = {}
    for row, v in key.items():
        by_ep.setdefault(v["episode"], []).append((row, v["cue"]))
    changed = 0
    for eid, rows in sorted(by_ep.items()):
        rendered = json.loads((old.work_dir(eid) / "06-translate" / "cues.en.json").read_text())[
            "cues"
        ]
        want = {c["id"]: c["character_id"] for c in rendered}
        before, after = attribute(old, eid), attribute(new, eid)
        bad = [c for c in want if before.get(c) != want[c]]
        if bad:
            print(f"{eid}: the rendered arm's attribution disagrees with cues.en.json on {bad[:5]}")
            return 1
        for row, cue in rows:
            b, a = before[cue], after[cue]
            same_refs = refs(old, b) == refs(new, a) if a == b else None
            flag = "same" if a == b and same_refs else "CHANGED"
            changed += flag != "same"
            print(
                f"{row} {eid} {cue}: character {b} -> {a}; reference picks "
                f"{'same' if same_refs else 'differ' if same_refs is False else '-'} ({flag})"
            )
    print(f"{changed} of {sum(len(r) for r in by_ep.values())} pack cues would change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
