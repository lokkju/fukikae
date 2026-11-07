# Dialogue separation versus Demucs

_Run 2026-11-07 on ml-01 (RTX 3090, index 0), branch feat/speaker-identity. Tuning pairs only: no `--holdout` was passed._

## Backend and model

Ruling R14: audio-separator 0.48.0 lists no dialogue/music/effects model (vocal, karaoke, denoise, crowd, dereverb, male-female and drum Roformers only), so the `dialogue` backend runs **DBS-TIGER-DnR** (`JusperLee/DBS-TIGER-DnR`, Apache-2.0, 1.0M parameters, trained on Divide and Remaster v2; SI-SDR on the DnR v2 test set: speech 15.90 dB, music 7.78, effects 6.70). It is the only public cinematic dialogue/music/effects checkpoint found that loads from a small packaged library. Code: the `dbs` package from github.com/JusperLee/Dynamic-Band-Splitting, pinned to commit ff526463a58ee0278e7bd05c4b8bce112b743e34 in the `dialogue` extra (not on PyPI; its dependencies are torch, numpy, einops, huggingface_hub>=0.23, safetensors, rotary-embedding-torch, soundfile, and `uv sync --all-extras` resolves without moving existing pins).

The backend mixes the 44.1 kHz stereo input to mono, runs 6 s chunks with 3 s hop, and writes the speech stem as `vocals.wav` (copied to both channels). `background.wav` is the mix minus speech (Ruling R5 residual; option `background = "stems"` gives music plus effects instead). Peak VRAM about 6.5 GB; an hour of audio took about 7 minutes of separation, CPU-bound.

## Method

Copies of the tuning workspaces (`~/fukikae-ws/<set>-dialogue/`) with `[separation] backend = "dialogue"` and `[embed] backend = "wespeaker"` re-ran separate then diarize (shipped pyannote, which regenerates `clusters.npz`; `SeparateStage` and `DiarizeStage` called directly, ASR was not re-run since diarization does not read it). Demucs is the original workspace scored with the same WeSpeaker embedder through the harness. Driver and raw results: `~/fukikae-ws/separation/{run.py,score.py,results.json}`. Anime: joint (episode cannot-link on, 0.6) and centroid 0.6; live action: centroid 0.6 and joint with episode cannot-link off, 0.6. Clean means passing and in no merge.

## Results

| Set | Separation | Matcher | Fraction | Pass/rec | Clean | Merges | Share | Strict | Coverage | NC |
|---|---|---|---|---|---|---|---|---|---|---|
| fairytail (ja) | Demucs | joint | 0.56 | 5/9 | 5 | 1 | 81% | 0.25 | 41% | 70% |
| fairytail (ja) | dialogue | joint | 0.44 | 4/9 | 4 | 1 | 73% | 0.09 | 34% | 72% |
| fairytail (ja) | Demucs | centroid | 0.44 | 4/9 | 4 | 1 | 72% | 0.25 | 36% | 70% |
| fairytail (ja) | dialogue | centroid | 0.44 | 4/9 | 2 | 3 | 71% | 0.09 | 31% | 72% |
| fairytail-en | Demucs | joint | 0.80 | 8/10 | 7 | 1 | 81% | 0.08 | 50% | 71% |
| fairytail-en | dialogue | joint | 0.50 | 5/10 | 4 | 1 | 84% | 0.25 | 46% | 76% |
| fairytail-en | Demucs | centroid | 0.70 | 7/10 | 6 | 1 | 72% | 0.08 | 43% | 71% |
| fairytail-en | dialogue | centroid | 0.50 | 5/10 | 4 | 1 | 81% | 0.25 | 45% | 76% |
| grid (ko) | Demucs | centroid | 0.50 | 1/2 | 1 | 0 | 73% | 0.17 | 12% | 96% |
| grid (ko) | dialogue | centroid | 1.00 | 1/1 | 1 | 0 | 79% | 0.40 | 12% | 100% |
| grid (ko) | Demucs | joint (cannot-link off) | 1.00 | 2/2 | 2 | 0 | 81% | 0.17 | 14% | 96% |
| grid (ko) | dialogue | joint (cannot-link off) | 1.00 | 1/1 | 1 | 0 | 91% | 0.40 | 13% | 100% |

## Reading the table

- Nearest-centroid accuracy rises on every set (70 to 72, 71 to 76, 96 to 100): the dialogue stem gives slightly more separable embeddings.
- Clean counts fall on both anime sets (ja joint 5 to 4, centroid 4 to 2; dub joint 7 to 4, centroid 6 to 4). Dialogue separation does not fix the Japanese female-voice merge, and on centroid it adds two merged pairs.
- Grid: the recurring set changed, because a character that had at least 5 s of labeled speech in both episodes under Demucs falls under that bar under dialogue (the stem or the diarization removed enough of that character's speech). Dialogue passes its one remaining recurring character at higher share (91% against 81%) but the comparison is 1 against 2 recurring characters, so it counts as no gain. Coverage is unchanged at 12 to 14%.
- Strict per-clip is higher on grid and the dub (0.25 to 0.40), lower on ja (0.25 to 0.09).

## Recommendation (ruling candidate for the ledger)

- **Anime (ja and dub)**: drop. Fewer clean characters than Demucs in every cell.
- **Live action (ko)**: drop for now. Higher share and NC on one character, but a recurring character is lost; one 2-episode set with two characters can't support a switch.
- Keep the backend registered (opt-in, no profile or default changed): its stem improves NC everywhere and could help a different segmentation, but nothing here justifies it as a profile default.

Caveat: a single dubbed or dialogue-heavy tuning pair per genre, one chunking setting (6 s / 3 s), and the shipped pyannote clustering only.
