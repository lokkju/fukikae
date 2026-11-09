#!/usr/bin/env bash
# Task 8 Step 4: score each held-out episode once from the pipeline's assignments.
# usage: holdout_score.sh <workspace dir>...  (a -pick copy under t8/ws or a -default workspace)
F=$(realpath "$(dirname "$0")/../.."); W=${FUKIKAE_WS:-$HOME/fukikae-ws}; OUT=$W/t8/holdout
for d in "$@"; do n=$(basename "$d"); base=${n%-pick}; base=${base%-default}
  case $base in
    bbt-*) fl=(--tune S01E01,S01E02 --holdout S01E03 --gt ass --gt-file $W/subs/bbt/labels --set bbt); tag=t8-holdout;;
    grid) fl=(--tune S01E02,S01E04 --holdout S01E05 --gt sdh --subtitle-track 2 --set grid); tag=t8-holdout;;
    kaze) fl=(--tune S01E01,S01E02,S01E03 --holdout S01E11 --gt colour --gt-file $W/subs/kaze-kaoru/aligned --force --set kaze); tag=t8-holdout;;
    bazinga-*) IFS=- read -r _ show cond <<<"$base"
      fl=(--tune S01E01,S01E02,S01E03 --holdout S01E05 --gt ass --gt-file $W/subs/bazinga/$show-$cond --set bazinga-$show); tag=t8-holdout-$cond;;
  esac
  mise -C $F exec -- uv run --directory $F fukikae eval identity "$d" --from-assignments "${fl[@]}" \
    --tag $tag --json $OUT/$n.json > $OUT/$n.txt 2>&1; echo "$n exit $?"
done
