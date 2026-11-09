#!/usr/bin/env bash
# Task 8 Step 4: build each -pick copy's registry on its tuning episodes in order, then the
# held-out episode, with the committed live-action profile (centroid pick: `fukikae run --to
# characters`). One fukikae process at a time on GPU 0.
# usage: holdout_run.sh <copy>...   (copy = bbt-en-pick, bazinga-tbbt-pal-pick, grid-pick, ...)
F=$(realpath "$(dirname "$0")/../.."); T=${FUKIKAE_WS:-$HOME/fukikae-ws}/t8/ws; LOG=${FUKIKAE_WS:-$HOME/fukikae-ws}/t8/holdout.log
for c in "$@"; do
  case $c in
    bbt-*) eps="S01E01 S01E02 S01E03";;
    bazinga-*) eps="S01E01 S01E02 S01E03 S01E05";;
    grid-*) eps="S01E02 S01E04 S01E05";;
    kaze-*) eps="${KAZE_EPS:-S01E01 S01E02 S01E03 S01E11}";;
  esac
  for e in $eps; do s=$(date +%s)
    if [[ $c == grid-* ]]; then
      stages=characters; [[ $e == S01E05 ]] && stages="diarize characters"
      r=$(mise -C $F exec -- uv run --directory $F python "$(dirname "$0")/stage_direct.py" $T/$c $e $stages 2>&1 | tail -1)
    else
      r=$(mise -C $F exec -- uv run --directory $F fukikae run $T/$c $e --to characters 2>&1 | tail -1)
    fi
    echo "$c $e: $r ($(( $(date +%s)-s ))s)" | tee -a $LOG
  done
done
