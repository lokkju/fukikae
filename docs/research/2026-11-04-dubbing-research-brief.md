# Automatic English Dubbing for Foreign-Language TV: Research Brief

_Researched 2026-11-04. Star counts and maintenance status are approximate as of that date; re-check before committing to a base._

## Goal

Take a foreign-language TV episode and produce an English-dubbed version automatically: diarize speakers, produce speaker-attributed transcripts/subtitles, translate, assign each speaker a distinct (ideally consistent) English voice, synthesize, time-fit, and remix over the original background audio.

## Existing projects

**pyVideoTrans** (github.com/jianchang512/pyvideotrans, GPL-v3, ~18k stars). Best current fit. Full pipeline with GUI, CLI, and packaged Windows build. When the ASR channel is set to WhisperX or Parakeet, it assigns a different dubbing voice per detected speaker. Default segmentation uses faster-whisper VAD; LLM re-segmentation is available and improves sentence splitting. Primary candidate to fork or extend.

**VideoLingo** (github.com/Huanshere/VideoLingo, Apache-2.0, ~17-18k stars). Strongest subtitle quality (better segmentation, literal-then-free translation pass) but its README states it cannot dub multiple characters separately. Wrong base for multi-speaker dubbing; worth mining for its segmentation and two-pass translation approach.

**SoniTranslate** (github.com/R3gm/SoniTranslate, ~1.3k stars). pyannote diarization with per-speaker TTS assignment (verify current speaker limit and TTS backends). Known unresolved pacing issue: it can prevent sped-up speech but not over-slowed speech when time-fitting.

**WeeaBlind** (github.com/FlorianEagox/WeeaBlind, ~280 stars). Built for exactly this use case (anime/foreign TV). Stack: ffmpeg + pydub, Coqui TTS, speechbrain language ID, pyannote.audio diarization, vocal isolation, speaking rate and volume matching, option to dub only foreign-language segments. Subtitle-driven (takes existing subtitle files as input). Stale: last push August 2024. Good reference for the subtitle-first path.

**yt-dbl** (github.com/brolnickij/yt-dbl, PyPI `yt-dbl`). Most modern stack: ASR with diarization, Qwen3-ForcedAligner for word-level timestamps, translation via the Claude API, per-speaker voice cloning with Qwen3-TTS, background ducking, max speed-factor cap, resumable step-based jobs. Apple Silicon only (MLX). On a CUDA box, use it as a reference architecture, not a dependency. Its step/resume job model is worth copying.

**Video-Translator** (github.com/Praj-17/Video-Translator). Small project: diarized transcription, translation, one Coqui XTTS-v2 clone per detected speaker, cue-level timing fit, ffmpeg remix. Emits `speakers.json` and per-speaker reference WAVs. Useful as a minimal example of the per-speaker clone wiring.

**Commercial bar:** ElevenLabs Dubbing Studio does the full pipeline with an editable per-speaker track. Use it as the quality benchmark.

## Recommendation

Use pyVideoTrans as the base (or as the reference implementation if a cleaner standalone pipeline is preferred). Do not use VideoLingo as the base. Borrow yt-dbl's job model and WeeaBlind's subtitle-first input path.

## Gap to build: cross-episode speaker identity

Every project above diarizes per file. SPEAKER_00 in episode 1 and SPEAKER_00 in episode 2 are unrelated, so the same character gets a different voice each episode. For a series this is the first thing to build:

1. Run diarization per episode (WhisperX / pyannote).
2. For each diarized cluster, extract speaker embeddings (ECAPA-TDNN via speechbrain, or pyannote's embedding model) from clean, non-overlapping segments, ideally after vocal isolation.
3. Cluster embeddings across the whole season (agglomerative with a cosine threshold, or HDBSCAN) to get season-level character IDs.
4. Persist a character registry: character ID, centroid embedding, assigned English voice (preset or clone reference), optional human-readable name.
5. On new episodes, match clusters against the registry centroids first; only create new characters below the match threshold. Allow manual merge/split/rename.

## Subtitle-first path

When official subtitles exist (common for TV), skip ASR + machine translation for content and use the human English subs:

1. Diarize the source audio.
2. Align English subtitle cues to the diarization timeline (overlap-based assignment; forced alignment of source-language ASR as a tiebreaker if needed).
3. Synthesize per cue with the character's voice.

This removes the two noisiest stages. Still run source-language ASR if you need word-level timing for tighter fit.

## Known hard problems

- **Source separation:** need clean vocals for diarization/cloning and a clean background bed for remix. Demucs or UVR-class models.
- **Overlapping speech and music:** TV audio is far worse than YouTube talking heads; expect diarization errors in crosstalk and scored scenes. Budget for manual correction tooling.
- **Time fitting:** English is often longer than the source. Strategies, in order of preference: translation length constraint in the prompt (ask the LLM for a condensed rendering when the slot is short), TTS rate control, bounded time-stretch (cap both speed-up and slow-down; SoniTranslate's failure mode is unbounded slow-down), borrowing adjacent silence.
- **Voice choice:** cloning the original actor's voice into English vs. assigning distinct stock English voices per character. Cloning preserves character but carries accent artifacts; stock voices are cleaner. Make it a per-character registry setting.

## Proposed pipeline stages

1. Extract audio (ffmpeg).
2. Source separation into vocals and background.
3. ASR with word timestamps (WhisperX) or ingest existing subtitles.
4. Diarization, then cross-episode character resolution against the registry.
5. Translation (LLM, length-aware, with character/scene context) unless using human subs.
6. Per-cue TTS with the character's assigned voice.
7. Time-fit each cue into its slot.
8. Remix over the background with ducking; mux into MKV with an added English audio track and speaker-labeled subtitle track (keep original audio as a secondary track).

Make each stage idempotent and resumable with artifacts on disk per episode (yt-dbl style), so any stage can be re-run after manual correction.

## First tasks for Claude Code

1. Clone pyVideoTrans; locate the diarization-to-voice assignment code path and confirm how speaker labels flow into TTS.
2. Prototype the season-level character registry as a standalone module consuming WhisperX diarization output.
3. Test on two episodes of one show: measure whether recurring characters resolve to the same registry ID.
