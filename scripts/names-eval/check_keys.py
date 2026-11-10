"""Print, per episode, which stages before `names` a `fukikae run --from names` would rerun.

A copied workspace whose manifests don't match the current code's cache keys reruns
extract, separate, asr or diarize (GPU work, and new diarize turns). Run this on every arm
copy before the first `fukikae run`; it writes only the workspace's hash cache.

    uv run --directory <repo> python scripts/names-eval/check_keys.py <ws> S01E11 S01E12 ...
"""

import sys
from pathlib import Path

from fukikae import job
from fukikae.cli import pipeline_stages
from fukikae.workspace import Workspace

UPSTREAM = ("extract", "separate", "language", "asr", "diarize")


def main() -> int:
    ws = Workspace(Path(sys.argv[1]))
    stale = 0
    for code in sys.argv[2:]:
        ep = ws.find_episode(code)
        manifest = job.load_manifest(ep)
        cache = job._HashCache(ep)
        out = []
        for stage in pipeline_stages(ws.config):
            if stage.name not in UPSTREAM:
                continue
            key = job._cache_key(stage, ep, ws.config, cache)
            rec = manifest.stages.get(stage.name)
            ok = rec is not None and rec.status == "done" and rec.key == key
            ok = ok and all(p.exists() for p in stage.outputs(ep))
            out.append(f"{stage.name}={'ok' if ok else 'STALE'}")
            stale += not ok
        cache.save()
        print(ep.id, " ".join(out))
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
