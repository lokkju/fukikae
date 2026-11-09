#!/usr/bin/env bash
# usage: diar.sh copy:episode ...   (one fukikae run at a time, GPU 0)
F=$(realpath "$(dirname "$0")/../.."); T=${FUKIKAE_WS:-$HOME/fukikae-ws}/t8/ws
for spec in "$@"; do c=${spec%%:*}; e=${spec##*:}; s=$(date +%s)
  r=$(mise -C $F exec -- uv run --directory $F fukikae run $T/$c $e --to diarize 2>&1 | tail -1)
  echo "$c $e: $r ($(( $(date +%s)-s ))s)" | tee -a ${FUKIKAE_WS:-$HOME/fukikae-ws}/t8/diar.log
done
