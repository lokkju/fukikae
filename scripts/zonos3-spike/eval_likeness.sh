#!/usr/bin/env bash
# eval_likeness.sh <ws> <episode> <set>: eval tts --backend for indextts2, zonos3 reference, zonos3 embedding
set -uo pipefail
W=/home/lokkju/projects/fukikae/.claude/worktrees/zonos3
WS=$1 EP=$2 SET=$3
L=/nvme/fukikae/tmp/zonos3/ledger.jsonl
OUT=/nvme/fukikae/tmp/zonos3/eval
mkdir -p "$OUT"
F() { HF_HOME_ARG="$1"; shift; mise -C "$W" exec -- env $HF_HOME_ARG uv run --directory "$W" fukikae "$@"; }
toml="$WS/fukikae.toml"
cp "$toml" "$toml.orig"
F "X=1" eval tts "$WS" --episodes "$EP" --backend indextts2 --set "$SET" --tag zonos3-spike-indextts2 --ledger "$L" > "$OUT/$SET-indextts2.txt" 2>&1; echo "indextts2 $?"
for mode in reference embedding; do
  cp "$toml.orig" "$toml"; printf '\n[tts.zonos3]\nmode = "%s"\n' "$mode" >> "$toml"
  F "X=1" eval tts "$WS" --episodes "$EP" --backend zonos3 --set "$SET" --tag "zonos3-spike-zonos3-$mode" --ledger "$L" > "$OUT/$SET-zonos3-$mode.txt" 2>&1; echo "zonos3 $mode $?"
done
cp "$toml.orig" "$toml"
