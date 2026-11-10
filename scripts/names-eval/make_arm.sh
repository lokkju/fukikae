#!/usr/bin/env bash
# make_arm.sh <source workspace> <arm workspace> <names toml fragment>
# A fresh arm copy: hard links for the immutable stage outputs (audio, stems, ASR, diarize,
# mix), real copies of everything a stage or the registry writes, and a [names] table appended
# to the copy's fukikae.toml. Refuses when a copied file still has more than one link.
set -euo pipefail
src=$1 dst=$2 frag=$3
rm -rf "$dst"
mkdir -p "$(dirname "$dst")"
cp -al "$src" "$dst"
real() { if [ -e "$src/$1" ]; then rm -rf "${dst:?}/$1"; cp -a "$src/$1" "$dst/$1"; fi; }
for f in fukikae.toml labels-aliases.yaml .fukikae/registry .fukikae/hashcache.json \
         .fukikae/cards .fukikae/cache .fukikae/eval .fukikae/tts-cache; do real "$f"; done
for ep in "$src"/.fukikae/episodes/*/; do
  e=.fukikae/episodes/$(basename "$ep")
  for f in manifest.json 04c-names 05-characters 04b-tag 06-translate 07-tts 08-timefit eval; do
    real "$e/$f"
  done
done
printf '\n%s\n' "$frag" >> "$dst/fukikae.toml"
checks=()
for p in fukikae.toml .fukikae/registry .fukikae/hashcache.json .fukikae/cards \
         .fukikae/episodes/*/manifest.json .fukikae/episodes/*/05-characters; do
  for q in "$dst"/$p; do [ -e "$q" ] && checks+=("$q"); done
done
bad=$(find "${checks[@]}" -type f -links +1 | head -1)
if [ -n "$bad" ]; then echo "still hard-linked: $bad" >&2; exit 1; fi
echo "arm $dst ready"
