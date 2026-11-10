#!/usr/bin/env bash
# Task 11 of docs/superpowers/plans/2026-11-10-subtitle-speaker-names.md: the commands behind
# docs/results/2026-11-10-subtitle-speaker-names.md, as run on 2026-11-10.
#
#   scripts/names-eval/run_arms.sh <repo>
#
# ja arms copy ~/fukikae-scratch/t12/fairytail (full manifests: nothing before `names` reruns).
# en arms copy en-base, itself a copy of ~/fukikae-ws/t9hold/fairytail-en whose E11 went
# through extract to diarize on GPU 0 (D29 #7) and whose E12 to E14 reran separate, asr and
# diarize (their separate keys predate the resolved-model-identity key; same Demucs model).
set -euo pipefail
REPO=${1:?repo}
D=~/fukikae-scratch/names-t11
OUT=$D/out
F() { mise -C "$REPO" exec -- uv run --directory "$REPO" fukikae "$@"; }
MK=$REPO/scripts/names-eval/make_arm.sh
CHECK() { uv run --directory "$REPO" python "$REPO/scripts/names-eval/check_keys.py" "$@"; }
EPS=(S01E11 S01E12 S01E13 S01E14)
mkdir -p "$OUT"

declare -A FRAG=(
  [baseline]=$'[names]\nuse = "off"'
  [names]=$'[names]\nuse = "auto"\nholdout_episodes = ["S01E14"]'
  [ceiling]=$'[names]\nuse = "auto"'
  [hidden]=$'[names]\nuse = "auto"\nhide_fraction = 0.2\nhide_seed = 0'
  [hidden-base]=$'[names]\nuse = "off"\nhide_fraction = 0.2\nhide_seed = 0'
)
ARMS=(baseline names ceiling hidden hidden-base)

# en base: the dub workspace, upstream rerun on GPU 0
"$MK" ~/fukikae-ws/t9hold/fairytail-en "$D/en-base" ''
F run "$D/en-base" "${EPS[@]}" --to diarize

for L in ja en; do
  if [ "$L" = ja ]; then SRC=~/fukikae-scratch/t12/fairytail SET=fairytail; else SRC=$D/en-base SET=fairytail-en; fi
  for a in "${ARMS[@]}"; do
    "$MK" "$SRC" "$D/$L-$a" "${FRAG[$a]}"
    CHECK "$D/$L-$a" "${EPS[@]}"   # every upstream stage must print ok
    F run "$D/$L-$a" "${EPS[@]}" --from names --to characters
  done
  # protocol A: tune episodes resolved together, then the held-out episode alone (R13)
  for a in baseline names ceiling; do
    F registry resolve "$D/$L-$a" --episodes S01E11,S01E12,S01E13
    F registry resolve "$D/$L-$a" --episodes S01E14
  done
  # protocol B tunes on all four episodes (Task 10 review B5)
  for a in hidden hidden-base; do
    F registry resolve "$D/$L-$a" --episodes S01E11,S01E12,S01E13,S01E14
  done
  id() { a=$1; shift; F eval identity "$D/$L-$a" --from-assignments --gt ass --test-set $SET \
           --tag "t11-names-$L-$a" --json "$OUT/$L-$a.identity.json" "$@" > "$OUT/$L-$a.identity.txt"; }
  nm() { a=$1; shift; F eval names "$D/$L-$a" --gt ass --test-set $SET \
           --tag "t11-names-$L-$a" --json "$OUT/$L-$a.names.json" "$@" > "$OUT/$L-$a.names.txt"; }
  for k in id nm; do
    $k baseline --tune S01E11,S01E12,S01E13 --holdout S01E14
    $k names --tune S01E11,S01E12,S01E13 --holdout S01E14
    $k ceiling --tune S01E11,S01E12,S01E13 --ceiling S01E14
    $k hidden --tune S01E11,S01E12,S01E13,S01E14
    $k hidden-base --tune S01E11,S01E12,S01E13,S01E14
  done
done
uv run --directory "$REPO" python "$REPO/scripts/names-eval/tables.py" "$OUT" "$D" ja en > "$OUT/tables.md"

# Listening check (D29 #9): the ja names arm through tts on GPU 0. The worker environment is
# the main checkout's (.venvs/indextts2; workers/ is the same on this branch).
FUKIKAE_WORKER_CHECKOUT=~/projects/fukikae F run "$D/ja-names" S01E12 S01E13 S01E14 --to tts
uv run --directory "$REPO" python "$REPO/scripts/names-eval/listening_pack.py" \
  ~/fukikae-scratch/t12/fairytail "$D/ja-names" ~/fukikae-review/names-natsu NATSU "YOUNG MAKAROV"

# Ledger export
T=(); for L in ja en; do for a in "${ARMS[@]}"; do T+=(--tag "t11-names-$L-$a"); done; done
uv run --directory "$REPO" fukikae eval report "${T[@]}" \
  --export "$REPO/docs/results/2026-11-10-subtitle-speaker-names.jsonl"
