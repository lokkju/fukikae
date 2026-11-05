# fukikae: speaker identity rework

_Status: draft for review, 2026-11-05. Follows `docs/results/2026-11-05-registry-fairytail.md` (verdict rework). Builds on the merged foundation (`docs/superpowers/specs/2026-11-04-doppio-design.md`)._

## Intent

Make a recurring character resolve to one identity across episodes reliably enough to pass the spec's bar (≥ 90% of recurring characters on one id with no merges), before translation and TTS build on it. The foundation's first measurement, Fairy Tail: 100 Years Quest S01E12-E13, put 70% of recurring speech on each character's modal id, and showed three failures: similar voices collapse into one id (Happy/Charle/Wendy, Lucy/Cana/Juvia), some characters split across episodes (Erza, Cana, Kiria), and pyannote over-splits and mixes speakers within an episode.

Subtitles that name every speaker (Crunchyroll's ASS) now give automatic ground truth, so this plan is measurement-led: build the harness first, then try candidate fixes in order of expected payoff, keep what moves the number.

## Goals

1. An identity evaluation harness that scores any combination of segmentation, embedder and matching against subtitle ground truth, on several series and languages, in minutes and without the GPU stages re-running when only matching changes.
2. Metrics that separate failure modes: segmentation purity and coverage, embedding separability (with segmentation held to ground truth), cross-episode identity (strict spec metric and a speech-weighted majority metric), and merges.
3. Tested default **profiles** rather than one winning configuration (operator decision, 2026-11-05): the best embedder, segmentation, matcher and threshold may differ by genre (anime vs live action) and by language. Each profile is chosen on its own test sets and passes the bar on its own held-out episode: at minimum an anime profile (Fairy Tail), a live-action profile (The Cyberfarm, Grid), and a general default used when nothing more specific applies. Where data allows, language-specific variants (Japanese, Korean, Russian, Spanish, Chinese) layer on top; a variant ships only when it measurably beats the profile beneath it.
4. Profiles are configuration, not code: a workspace selects one with `profile = "anime"` (or `"live-action"`), the profile supplies defaults for every identity stage, and the existing per-language tables (`[embed.lang.ko]`, ...) override it, which override nothing the workspace sets explicitly. Resolution order: workspace explicit > per-language override > profile > built-in default.

**The bar (my call, refining the foundation spec's 90% wording for a measurable gate):** a *recurring character* has at least 20 s of labeled speech in each episode compared. Pass: at least 90% of recurring characters keep at least 80% of their speech on a single character id, and no two recurring characters share their modal id. The strict per-clip metric (every clip on one id) is still reported. Spans labeled `?` and overlapped speech are excluded from every metric; each run also reports coverage (share of ground-truth speech that received a usable segment and id), so no method wins by labeling less.
5. Every new model is a backend behind the existing protocols (embed, diarize) or a new kind where none fits (segment).

## Non-goals

- Translation, TTS, time fitting, mixing (the next plan once this passes).
- Face or active-speaker detection. Logged in `docs/ideas.md`; it's the strongest lever for similar voices but a separate project. This plan leaves the hooks for it (identity evidence is combinable).
- Training speaker embedders from scratch.

## Anti-goals

- Tuning on the test episodes until the number passes. Each series gets a held-out episode pair that isn't used for choosing thresholds or models.
- Needing named subtitles at inference time. Subtitles are ground truth for evaluation, and an optional extra signal when present; the pipeline must still work on a release with no subtitles.

## Options considered

Brainstormed widely; grouped by the failure they attack. Expected payoff is a guess until the harness measures it.

**Better embeddings (attacks similar-voice collapse; highest expected payoff):**
- Stronger pretrained speaker embedders: CAM++ and ERes2NetV2 (3D-Speaker, trained on large Mandarin+English corpora), WavLM-based speaker verification models, NVIDIA TitaNet-L, ReDimNet, pyannote's WeSpeaker ResNet34. Compare on ground-truth segments.
- Score fusion of two embedders (concatenate normalized embeddings, or average cosine scores).
- Per-series adaptation: learn an LDA/PLDA projection (or a small linear head) from the labeled spans of named-subtitle episodes, apply it to the rest of the series. Cheap, and directly targets "these voices are close in a generic space but separable for this cast".
- Exemplar matching instead of single centroids: k-nearest exemplars or several centroids per actor (anime characters shout, whisper and narrate; one mean blurs that).

**Better segmentation (attacks over-splitting and mixed clusters):**
- Subtitle-line segmentation: when a release has timed subtitles (named or not), each line is a candidate single-speaker segment; split lines at dash turn markers. Replaces pyannote's turns where available.
- ASR-guided segmentation: Qwen3-ASR/aligner word timing plus a speaker-change detector, segments bounded by sentence ends.
- Pyannote tuning: clustering threshold, minimum cluster size, overlapped-speech removal before embedding.
- Season-level joint clustering: embed every segment of every episode and cluster once, with cannot-link constraints for overlapping turns, instead of diarizing per episode and then matching (removes one stage of compounding error).

**Better audio (attacks contaminated embeddings):**
- Dialogue-specific separation (cinematic dialogue/music/effects models, e.g. a Mel-Band Roformer dialogue model or a DnR-trained model) in place of music-oriented Demucs vocals, which keep effects and shouts.

**Other evidence (attacks similar voices without better acoustics):**
- Text cues: vocatives and names in the transcript ("Lucy!"), turn-taking; an LLM attributes lines to characters given the cast list and context, fused with acoustic scores. Useful later; needs the translate plan's LLM plumbing.
- Visual: face/active-speaker (deferred, see Non-goals).
- Operator in the loop: confirm only `review` clusters, from the video clips; a five-minute labeled seed per series feeds the per-series adaptation above.

**Metric fixes:**
- Keep the strict spec metric, add the speech-weighted majority metric and character-id purity (as in the results doc), plus segmentation purity/coverage, so a change is credited to the stage that earned it.

## Plan shape (decided here)

1. **Harness** (first, everything else depends on it). Ground-truth readers beyond the existing ASS Name field: SDH/CC speaker prefixes in SRT/ASS text (`NAME:`, `(NAME)`, `[NAME]`, Cyrillic, Hangul and CJK names, with sound-effect filtering), and colour-coded WebVTT/EBU-TT (per-episode colour classes, mapped to characters from co-occurring name tags or an operator map). Then `fukikae eval identity <workspace> --episodes ...` computes all metrics from existing artifacts plus subtitle ground truth; `--segmentation subtitles|diarization`, `--embedder <name>`, `--matcher centroid|knn|...` select variants; embeddings are cached per (segmentation, embedder) so matching experiments are cheap.
2. **Embedder comparison** on ground-truth (subtitle-line) segments: ECAPA vs at least three of the candidates above, plus one fusion. Metric: cross-episode nearest-centroid accuracy per character and the majority metric.
3. **Segmentation comparison** with the best embedder: pyannote (default and tuned) vs subtitle-line vs ASR-guided.
4. **Matching**: exemplar/kNN and per-series adaptation (LDA/PLDA from labeled episodes), evaluated on the held-out episode. Adaptation is an optional stage: it uses subtitle names when a release has them, or an operator-labeled seed, and the registry stores the per-series projection; the default pipeline still runs without it. Season-level joint clustering is evaluated as an alternative matcher behind config; the existing per-episode diarize-then-match flow stays the default unless it measures better.
5. **Separation**: one dialogue-separation model vs Demucs, with the best of 2-4.
6. **Threshold sweep and profiles**: per profile, sweep on its tuning episodes and confirm on its held-out episode; write results; ship the profiles (`anime`, `live-action`, the general default, and any language variant that earned its place) as built-in config presets.

Steps 2-5 each end with a results note and a keep/drop decision recorded as a ruling.

## Test sets

- Fairy Tail: 100 Years Quest S01E11-E14 (Japanese; CR ASS fully named). Tuning: E12 builds the registry, E13 resolves against it. Held out: E14 resolves against the registry built from E12-E13, and nothing is chosen by looking at E14. Per-series adaptation trains only on E12-E13 labels. E11 (a different release) is spare.
- The same episodes' English dub track (E12-E14 have `eng` audio): same ground truth, a second language for free. Each audio language is its own workspace and registry (a dub has different voice actors); the subtitle names label both. Linking one character across languages is the actor model's job and out of scope here.
- Spanish: DAN DA DAN S01E01-E03, Crunchyroll WEB-DL, multi-audio (13 audio tracks verified from MediaInfo, including es-ES and es-419 Spanish dubs, plus Japanese and English; CR English ASS). Each Spanish dub is its own registry, labeled by the English ASS names (dubs keep the original timing).
- Chinese: Link Click S03 (Bridon Arc) E04-E06, Crunchyroll WEB-DL, dual audio (Chinese original plus English dub, verified; CR English ASS).
- Russian: not yet found. The 2026-11-05 survey searched only Crunchyroll releases, which carry no Russian audio (Crunchyroll left Russia in 2022). Russian sources (dub and fansub groups such as AniLibria, Anidub, SHIZA Project, whose ASS scripts often name speakers; Russian streaming originals; Russian subtitle archives) were then searched: the best lead is Crunchyroll's multi-dub releases (Mob Psycho 100 S03, Chainsaw Man S01, SPY x FAMILY S01), which carry an official Russian dub track beside CR subtitles, so the CR English ASS names label the Russian audio the same way as the Spanish set (to be confirmed on the files). Russian fansub archives and streaming services were blocked or behind accounts in that search.
- Name-field coverage on the Spanish and Chinese sets is unverified until the files are in the library; checked first when they arrive.
- Live action (required, not optional): the product is for foreign TV generally, and anime voices are stylized in ways live-action actors aren't, so a config tuned on anime alone proves little. At least one live-action set with speaker ground truth must be in the evaluation before the final config is chosen. Ground-truth sources being surveyed: SDH/CC tracks with speaker tags on foreign originals, colour-coded broadcast subtitles (a consistent per-character colour scores consistency without names), published scripts aligned through ASR, and partially tagged series already in the library (Grid, Parallels, Mr. Sunshine, Nocturne). Partial labels are usable: every metric scores only labeled spans and reports coverage. Best found so far (2026-11-05): RTVE's El Ministerio del Tiempo (Spanish) and ARD fiction (German) publish colour-coded subtitles through public endpoints (about 45% of RTVE cues coloured, 5 colours, plus occasional name tags). Colour marks the speaker within an episode but is reassigned between episodes, so it scores within-episode segmentation purity directly, and cross-episode identity once each episode's colours are mapped to characters (from co-occurring name tags, or a short manual map). A full probe of the library's subtitle tracks (2026-11-05) then found partial speaker tags worth using:
  - The Cyberfarm S02E01-E04 (Russian live action, consecutive): Russian SDH track "для слабослышащих" with speaker prefixes on 29/18/16/28% of lines; the leads Костя and Коля are tagged in every episode. The first live-action set, and Russian.
  - Grid S01E02 (Korean live action): Korean CC track, 49% of lines tagged (새하 45, 새벽 36, 부국장 24, 어진 20); its two sibling episode files with the same track still to be probed for a consecutive pair.
  - Parallels S01E01 (French, 32%) and Alice in Borderland S02E01 (Japanese live action, English SDH 22%): single episodes, for within-episode checks.
  Grid, Parallels, Mr. Sunshine and Nocturne's English SDH tracks alone are too sparse.
- Chimera (Korean) has no speaker ground truth yet; until a source is found it's used only to check the chosen config doesn't break on live action (cluster counts, review rate).

## Constraints

- Same as the foundation: uv, backends behind entry points, optional extras, the GPU host's free 3090 (index 0), artifacts on disk, resumable stages.
- Compute: the GPU host's 3090 index 0 is free (index 1 is vLLM). Re-running separation per candidate (hours, in the background) is acceptable; embeddings are cached per (segmentation, embedder) so matching experiments don't touch the GPU.
- New model families go into their own extras so `uv sync --all-extras` stays resolvable (transformers is capped < 5 by qwen-asr).
- Test media stays local and uncommitted; new episodes are copied from the NAS media share through a read-only Docker NFS volume.

## Left to the builder

- Which three-plus embedders, after checking which are installable on this stack.
- The harness's exact CLI and cache layout.
- How subtitle-line segmentation handles multi-speaker lines beyond dash markers.
