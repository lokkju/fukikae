# Backlog

Open follow-ups from the character-cards, ground-truth, speaker-names (D27) and speed (D28/D30)
work, written as tickets so they can move to an issue tracker unchanged. Each names where it came
from; the rulings ledgers in `docs/superpowers/ledgers/` and the results docs in `docs/results/`
hold the evidence. Remove an entry when it ships or is dropped, and say which in the commit.

## Identity and speaker names

### B1. Remainders with their own rows still go to the acoustic matcher in name-split clusters
When names split a diarize cluster, a remainder that has embedding rows of its own is matched
acoustically and often becomes a new character. This is most of the remaining protocol B gap
(Natsu's hidden-line accuracy 32% against 66% with names off). Changing the rule again on the
Fairy Tail numbers would tune on test data, so any fix needs a fresh evaluation set or a rule
argued from first principles, then a held-out score.
Done when: a rule for remainders with rows is specified before any run and scored on a set it
wasn't designed on. Source: `docs/results/2026-11-10-subtitle-speaker-names.md`, "Limits".

### B2. SDH speaker prefixes as a second names source
Grid's CC and Alice in Borderland's （名前） prefixes name some speakers. D29 #8 put this in a
later plan. Done when: the names stage reads SDH prefixes as a partial names source, with
detection thresholds that tolerate low coverage. Source: D29 #8.

### B3. Face and active-speaker detection
The next identity lever after subtitle names, for releases that name nobody and for the
held-out limit (names can't split a nameless episode's merged clusters). Source: `docs/ideas.md`,
D27.

### B4. Live-action held-out failures: merged leads
The live-action profile fails held-out on BBT German (leads merge in WeSpeaker space), Bazinga
TBBT, Grid without merges and Kaze, Kaoru E11 (直美 merges into りん, りん 0.76). Done when: a
change lifts at least one of these over the bar without tuning on its held-out episode. Source:
`docs/results/2026-11-09-live-action-ground-truth.md`.

### B5. `registry adapt --gt colour` should use the colour map
It labels from colour captions without the series-wide colour map that `eval colour-map` builds.
Source: ground-truth ledger, follow-ups.

### B6. Resolve: reuse an orphan's id when a new group is exactly its former clusters
Resolve's documented design mints a new id from the high-water mark even when the new group
holds exactly an orphan's old clusters (character 3 became 39 on Fairy Tail), which loses
continuity. Source: D27 Task 7 review.

### B7. A pin with no embedding rows keeps an actor alive
Pre-existing behaviour that names made common: characters with no exemplars survive on pinned
rowless remainders (Fairy Tail's characters 2 and 10). Source: D27 Task 6 review.

## Cards and voices

### C1. Shouted-only characters (parked, D34)
A character whose only speech is shouting gets a strained clone; the D32 floor didn't help
Young Makarov (old one-clip card preferred 7/10, bad emphasis on both versions). Parked until a
recurring character hits it. Lever to try first: IndexTTS2 emotion strength (emo_alpha) or a
calmer emotion reference when every reference clip scores high-arousal, then a blind listen.
Source: A21, D34.

### C2. `voices clone` needs an ASR fallback for `--transcript`
Cloning from a hand-picked clip requires a transcript typed by hand. Done when: `voices clone`
transcribes the clip with the pipeline's ASR when `--transcript` is omitted, and shows it for
confirmation. Source: Task 12 fix round.

### C3. Floor search limits and caching (D32)
The floor's combination search looks at the 64 best-ranked turns per episode, and the
candidate cache lasts one process, so a flagged character re-runs its reference search on every
single-episode run. Done when: the cache persists on disk keyed by inputs, or the cost is shown
not to matter. Source: D32 fix report.

## Translation

### T1. Per-card translation budgets (CD8), including German
Budgets are per backend speech rate (now per cue from the speaker's card backend, C4 of the D27
review), but not per card: a slow-speaking clone still gets the backend's average budget. Needs
a budget design for German, whose lines run long. Source: CD8, character-cards ledger.

### T2. A sparse untitled signs track can pass the subtitle floor (CD25)
An unflagged, untitled signs track at 3 to 6 cues a minute passes the 3-a-minute floor and could
be taken as the dialogue track. Rare. Source: Task 12 fix round 2 concerns.

## TTS and timing

### S1. Rate refit overestimates the speedup for lines whose pauses were capped (D33)
`render_for_slot` scales the capped duration as if it all shrinks with rate, but a capped pause
is constant, so the fitted line can still miss its slot. Only Kokoro (the rate-capable backend)
is affected, and it rarely has pauses over the cap. Source: D33 review, finding 1.

### S2. `_link` race on the nothing-to-cut capped file (D33)
`unlink` then `link` leaves a window where a concurrent reader sees no file. Only two
concurrent tts runs on one workspace can hit it; the pipeline doesn't do that. Fix: link to a
temp name and `os.replace`. Source: D33 review, finding 2.

### S3. `indextts2.py`: the "render-weighted mean" comment is wrong
A stale comment from the Task 9 era. Source: character-cards follow-ups.

## Speed (D28/D30)

### P1. TensorRT likeness: test the GPT path
fp32 conditioners didn't close TensorRT's likeness gap (D30); every TRT build is about 6% shorter
and its renders are either near-identical or entirely different per line, which points at
TensorRT-LLM's fp16 GPT and its beam search. Next measurement: TensorRT-LLM with sampling
matched to PyTorch's, or the PyTorch GPT in front of the TRT conditioners and decoder. Then the
eval gate and a listening checkpoint. Source: `docs/results/2026-11-09-character-cards.md`, D30.

### P2. IndexTTS2 accel sampler patch, as a measured experiment
D28 ruled the upstream accel engine's sampler patch a later measured experiment (its sampler
still uses temperature alone; upstream issue index-tts/index-tts#771 is open). Source: D28.

### P3. `vram_mb` for the fp32 TensorRT variant
The fp32-conditioner engines use 18.6 GB against the backend's 16.5 GB `vram_mb`. Moot while
`fp32` stays unset (the default). Source: D30.

## Operations

### O1. Pin and take offline the ten older backends
ASR, alignment, pyannote, dialogue separation and the wespeaker, ecapa, wavlm and sherpa
embedders still resolve models online at load. Done when: each pins a revision, resolves
offline at runtime and downloads only in `fukikae backends setup` (CD24), like the TTS workers.
Source: character-cards follow-ups.
