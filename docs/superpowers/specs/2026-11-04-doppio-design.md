# fukikae: automatic English dubbing for foreign-language TV

Renamed from "doppio" to "fukikae" (吹き替え, Japanese for dubbing) on 2026-11-05, an operator decision.

_Status: draft for review, 2026-11-04. Research behind it: `docs/research/2026-11-04-dubbing-research-brief.md`._

## Intent

Take a foreign-language TV episode and produce an MKV with an English dub track, where each recurring character keeps the same English voice across every episode of the series. The original audio stays on the file as a secondary track, along with a speaker-labeled English subtitle track.

The cross-episode character identity is the reason this exists rather than just running pyVideoTrans: every existing project diarizes per file, so a character's voice changes from one episode to the next.

## Goals

1. A CLI that runs the whole pipeline on one episode, or on a season directory, unattended.
2. Every model stage is a pluggable backend chosen in config, and the choice can differ per language: the STT model for Japanese can differ from the one for Spanish, the translator can differ per source/target pair, and the TTS backend and voice are chosen per character.
3. Translation defaults to a lightweight model, not a frontier API. Any OpenAI-compatible endpoint works, so a self-hosted LLM gateway, a local vLLM or llama.cpp server, or a hosted API are all one backend with different settings. The gateway today serves only the resident 27B, so the default (my call) is a `managed` mode: fukikae starts a llama.cpp server with a small GGUF translation model on its own card for the translate stage and stops it afterwards, like any other stage's model. Pointing `base_url` at the gateway or anywhere else skips that and needs no infra change.
4. A season-level character registry that survives across episodes and resolves new diarization clusters to existing characters.
5. A subtitle-first path that uses human English subtitles, when the release has them, instead of ASR plus machine translation.
6. Every stage is resumable and idempotent, with its artifacts on disk, so a hand-corrected artifact can be fed back in and only downstream stages re-run.

## Non-goals (v1)

- A GUI or web editor. Manual correction in v1 means editing the per-stage JSON artifacts and using registry CLI commands (merge, split, rename, assign voice).
- Real-time or streaming dubbing.
- Lip sync or any video modification.
- Target languages other than English. The interfaces take a target language parameter so this isn't foreclosed, but only English gets tested.
- Windows or macOS support. Target is Linux with CUDA. Backends declare the devices they support (CUDA, CPU, and later OpenVINO or ROCm), so other hardware is a backend addition, not a redesign.
- Depending on Kubernetes, a database server, or any other service an ordinary user wouldn't have.

## Anti-goals

- Hardcoding a model anywhere outside its backend module. If swapping Whisper for Parakeet on one language means touching pipeline code, the design has failed.
- Unbounded time-stretch. SoniTranslate's failure mode (stretching short English lines into slow-motion speech) is exactly what the speed bounds prevent.
- A character's voice changing between episodes without anyone asking for it to change.
- Re-running expensive stages (separation, ASR, diarization) because a cheap downstream stage changed.

## Constraints

- Runs on a GPU host with two RTX 3090s (24 GB each). One is resident vLLM serving the LLM gateway; the other is shared with other GPU jobs (a llama.cpp server, CI). Stages load one model at a time and release VRAM before the next stage, so the whole pipeline fits on a single 24 GB card.
- Where it runs (my call): v1 runs on the GPU host under uv, against the shared card, selected by `CUDA_VISIBLE_DEVICES` in config. Before a GPU stage loads, the runner checks free VRAM on that card against the backend's declared need and waits (or fails with `--no-wait`) rather than colliding with llama.cpp or CI. fukikae never depends on Kubernetes: the CLI on one machine with a GPU (or CPU, slowly) is the product. A k8s Job launcher requesting `nvidia.com/gpu` may come later as an optional extra; the job model already keeps all state in the series directory, so nothing changes but the launcher.
- Python, managed with uv (`uv sync`, `uv run`), ruff, pyright, pre-commit, per the user's python-project conventions. License PolyForm Shield 1.0.0.
- pyannote's gated models need a Hugging Face token with the model terms accepted.
- ffmpeg and ffprobe are required external binaries, not assumed on the host (the GPU host has neither); the repo pins them through mise, and `fukikae doctor` checks for them.
- Test material: Chimera (2021) S01E01-E02, Korean audio with Korean and English srt tracks, is the first pair; Death Note S01E01-E02, Japanese with an English ASS fansub, is the second. Copied once from the NAS media share (no longer mounted on the GPU host since 2026-11-04) into `~/fukikae-ws/media/<Show>/Season 1/` on ml-01, owner-only permissions. fukikae treats that directory as read-only source, the same as it would a media share.
- The GitHub repo, when it exists, comes from the operator's infrastructure-as-code config, not `gh repo create`. Until then the project is a local git repo.

## Architecture

### Package layout (builder may adjust)

```
src/fukikae/
  cli.py              # typer app: run, stage, registry, backends
  config.py           # load + validate TOML, resolve per-language overrides
  job.py              # episode work dir, manifest, stage runner, cache keys
  stages/             # one module per pipeline stage; orchestration only, no model code
  backends/
    base.py           # Protocols for every backend kind
    separation/ asr/ align/ diarize/ embed/ translate/ tts/
  registry/           # character registry: storage, matching, CLI ops
  timefit.py          # pure-DSP time fitting
  mix.py              # ducking, remix, mux (ffmpeg)
```

### Backend contract

Each backend kind is a `typing.Protocol` in `backends/base.py`. The kinds:

| Kind | Input | Output | First implementation (my call) | Next |
|---|---|---|---|---|
| `separation` | mixed audio | vocals, background | Demucs htdemucs (via `audio-separator` or `demucs`) | UVR MDX/Roformer models |
| `asr` | vocals, language | segments with word timestamps | Qwen3-ASR-1.7B (zh/yue/ja/ko) and faster-whisper large-v3 (fallback, CPU path) | Parakeet-TDT-0.6B-v3 (European languages); OpenVINO and whisper.cpp runtimes for the Whisper backend |
| `align` | audio, text, language | word timestamps | Qwen3-ForcedAligner-0.6B (its 11 languages) and WhisperX wav2vec2 (the rest) | |
| `tag` | vocals or mix | time spans of events (BGM, laughter, applause, cry) and emotion | SenseVoice-Small (optional, CPU) | |
| `diarize` | vocals | speaker turns | pyannote community-1 | |
| `embed` | audio span | speaker embedding vector | speechbrain ECAPA-TDNN | pyannote embedding |
| `translate` | cues + context + slot durations | English cues | `openai_compat` (chat completions) | `ctranslate2` NLLB/MADLAD for a no-LLM path |
| `tts` | text, voice spec, optional rate | audio | Kokoro (stock voices) and one cloning backend (XTTS-v2 or Qwen3-TTS; builder picks after a listening test) | F5-TTS, CosyVoice |

Backends register through Python entry points (group `fukikae.backends.<kind>`), so a backend can live in a separate package without touching fukikae. Built-ins register the same way. `fukikae backends list` shows what is installed.

Each backend declares the languages it supports (or `*`) and its VRAM need, and exposes `load()` / `unload()` so the stage runner can free the card between stages.

### Per-language configuration

One TOML file per series (`fukikae.toml` in the series directory), layered over a user-level default. Every backend kind has a default and optional per-language overrides:

```toml
target_language = "en"

[asr]
backend = "faster_whisper"   # fallback for any language
model = "large-v3"

[asr.lang.ko]
backend = "qwen3_asr"
model = "Qwen/Qwen3-ASR-1.7B"

[asr.lang.es]
backend = "parakeet"
model = "nvidia/parakeet-tdt-0.6b-v3"

[align]
backend = "qwen3_forced_aligner"  # falls back to wav2vec2 outside its 11 languages

[translate]
backend = "openai_compat"
mode = "managed"          # fukikae runs llama-server for the stage; "external" uses base_url
model = "qwen3-8b-q6_k.gguf"
# mode = "external"
# base_url = "https://llm-gateway.example/v1"
# api_key_env = "FUKIKAE_LLM_KEY"
max_speed = 1.25          # used to compute the length budget per cue

[translate.pair.ko-en]
model = "hunyuan-mt-7b"

[tts]
backend = "kokoro"        # default for characters with no voice assigned

[timefit]
max_speedup = 1.25
max_slowdown = 0.9
borrow_silence_ms = 400
```

Resolution order for a stage: per-language (or per-pair) override, then the series default, then the user default. The model names above are illustrative; the builder picks real defaults after testing.

Why these ASR defaults: `docs/research/2026-11-04-asr-backends.md` compares thirteen candidates. A backend whose model emits no timestamps (Cohere Transcribe, Omnilingual) is still usable, because the `align` stage runs on any backend's text; the ASR interface reports whether its own timing is word-level so the runner can skip alignment.

Source language is detected per episode (Whisper language ID on the first minutes of vocals) unless the config pins it. Per-segment language detection, for shows that mix languages (WeeaBlind's dub-only-foreign-segments option), is a later feature; the segment model leaves room for a per-segment `language` field.

### Job model

Borrowed from yt-dbl. Each episode gets a work directory:

```
<series>/.fukikae/<episode-stem>/
  manifest.json          # per stage: status, input hashes, backend + config hash, timings
  00-extract/audio.wav
  01-separate/vocals.wav  background.wav
  02-asr/segments.json   (or 02-subs/cues.json on the subtitle-first path)
  03-diarize/turns.json
  04-characters/assignments.json
  05-translate/cues.en.json
  06-tts/<cue-id>.wav
  07-timefit/<cue-id>.wav  report.json
  08-mix/dub.en.wav
  out/<episode>.fukikae.mkv
```

A stage's cache key is the hash of its input artifacts plus its resolved backend config. A stage re-runs when its key changes; downstream stages invalidate through the input hashes. A hand-edited artifact therefore re-runs everything after it and nothing before it. `fukikae run <ep> --from translate` forces a re-run from a named stage.

Artifacts are JSON with a version field, documented in the repo, so hand correction is a supported workflow and not an accident.

### Character registry

Lives at `<series>/.fukikae/registry/`, one registry per series, not per season (my call): characters recur across seasons, and `registry cluster` takes the series directory and walks `Season N/` subdirectories (or any episode files under it) (SQLite for records, `.npy` for embeddings; builder may choose). Per character: stable ID, optional name, centroid embedding, per-episode exemplar embeddings, voice spec (`{backend, voice}` for a stock voice or `{backend, reference_wav, reference_text}` for a clone), and a lock flag.

Flow per episode:

1. Diarize. For each cluster, take embeddings from clean, non-overlapping turns longer than a minimum duration (from separated vocals).
2. Score each cluster against registry centroids by cosine similarity. Above the match threshold: assign. Below: create a new character. Two clusters matching the same character in one episode are flagged for review rather than silently merged.
3. Update centroids with the new exemplars (locked characters keep their centroid).
4. Write `assignments.json` with scores, so the operator can see marginal matches.

Assignments are pinned once written (my call, from the anti-goal on voices changing): re-running an episode's later stages reuses its `assignments.json`, and re-running diarization maps the new clusters onto the old assignments by turn overlap. Centroid drift from later episodes never reassigns an earlier one. Only `fukikae registry rematch <episode>` or a registry op re-matches, and it reports every change.

Extras (my call): a cluster becomes a registry character only if it has at least 20 s of clean speech in the episode, or matches an existing character. Clusters below that bar are episode-local extras: they get voices from a separate small pool of stock voices, reused across extras, and kept out of the registry. The threshold is config. Stock voices for registry characters may also be reused once the pool runs out, preferring characters who have never appeared in the same episode.

`fukikae registry cluster <season-dir>` does the batch version from the brief: agglomerative clustering over every episode's cluster embeddings at once, to seed the registry for a season that's already on disk.

### Actors

A speaker embedding identifies a voice, and a voice belongs to an actor, not a character. One source-language actor often plays several characters (routine in anime), and the same actor turns up across series. So the registry has two entities:

- **Actor**: owns the embeddings (centroid and exemplars). Optional name and an external ID (TMDB or AniDB person ID) for later metadata lookup.
- **Character**: what gets an English voice. Belongs to a series and links to one actor (sometimes more, for recasts).

v1 behavior (my call): matching runs against actors; a new actor gets one new character, so v1 behaves as one character per voice. Splitting one actor into several characters is a manual op in v1 (`registry split --by-character`, with the operator assigning turns or episodes). The schema has both tables from day one, so later work can add: cast lists from TMDB/AniDB to name characters and actors, a cross-series actor library (an actor recognized in a new show starts with a known identity), and disambiguating characters of the same actor from subtitle speaker names or scene context. Assigning a consistent English voice per source actor, rather than per character, becomes one config switch.

CLI ops: `list`, `show`, `rename`, `merge`, `split` (by episode cluster, or by character), `voice` (assign), `lock`, `actor` (link, name, set external ID). Any op that changes an assignment invalidates the affected episodes' downstream stages through the manifest.

Voice assignment default: each new character gets the next unused stock voice from a gender-matched pool (gender estimate from the embedding or pitch; the operator can override). Cloning is opt-in per character, with the reference clip picked automatically from the cleanest exemplar turns and replaceable by hand.

### Subtitle-first path

If the source has an English subtitle stream (embedded or a sidecar `.srt`/`.ass`), or one is passed explicitly, stages 02 and 05 change: cues come from the subtitles, diarization turns are assigned to cues by time overlap, and translation is skipped. Source-language ASR still runs when the config asks for word-level timing, and serves as the tiebreaker when a cue overlaps two speakers. Signs, songs and SDH annotations (`[door slams]`, `♪`) are filtered out of synthesis by rule; the filter is configurable. Human subtitles are written to be read, so a cue over its spoken-length budget goes through the translator in condense mode (same backend, English to shorter English), and only that cue; within-budget cues are spoken verbatim. On by default (my call), `subs_condense = false` turns it off.

### Translation

The `translate` stage sends cues in scene-sized windows (cues grouped by gaps), not one at a time, with: the previous window's translations, character names from the registry, a per-series glossary file, and each cue's character budget derived from slot duration and `max_speed`. The prompt asks for the literal meaning first and then a speakable rendering within budget (VideoLingo's two-pass idea), and the response is structured JSON keyed by cue ID. A cue that comes back over budget gets one retry with a tighter budget before falling through to time fitting.

### Time fitting

Order, per the brief: length-constrained translation (above), TTS rate control where the backend supports it, bounded time-stretch (pitch-preserving, `max_speedup` and `max_slowdown` both enforced), then borrowing silence before and after the slot up to `borrow_silence_ms` when the next cue allows it. Anything still over is left long and reported in `07-timefit/report.json`, never silently truncated. Short audio is placed at the cue start and padded, never slowed past `max_slowdown`.

### Mix and mux

Background bed from separation, dub cues placed at their times, loudness-normalized (EBU R128) to match the original. Original vocals (my call, configurable): removed wherever a dubbed cue plays, and kept elsewhere, so laughs, screams, songs and lines the SDH filter dropped still sound like the original. When the optional `tag` stage ran, its BGM and singing spans keep the original vocals untouched (songs stay in the source language), and its laughter spans are never covered by a dub cue's removal window. An optional duck level keeps a quiet bleed of the original voice under the dub instead of removing it. Mux with ffmpeg: video copied, English dub as the default audio track, original audio kept, an ASS subtitle track with speaker names, and the original subtitle tracks kept.

## Milestones

1. **Recon** (parallel with 2): read pyVideoTrans's diarization-to-voice path and note anything worth borrowing; read yt-dbl's job runner.
2. **Skeleton**: config loader with per-language resolution, backend Protocols and entry-point registry, job manifest and cache keys, `extract` and `separate` stages.
3. **ASR and diarization**: Qwen3-ASR with Qwen3-ForcedAligner, faster-whisper with WhisperX alignment, pyannote, ECAPA embeddings. Score both ASR backends on Chimera E01 by character error rate against its Korean subtitle track (timing-insensitive text comparison after normalization), to confirm the Korean default with a number.
4. **Registry test**: the registry, then two episodes of one show. Success is measured, not eyeballed: hand-label the speakers in a sample of turns from both episodes and report what fraction of recurring characters resolve to the same ID. This is the first result worth looking at. Bar (my call): a recurring character is one with clean speech in both episodes; pass is 90% of them resolving to one ID with no two characters merged into one. Under 75% sends the work back to embeddings and separation before milestone 5; between the two, tune the threshold and exemplar selection, then proceed.
5. **Translate, TTS, time fit, mix**: an end-to-end dubbed episode.
6. **Subtitle-first path.**
7. **Second backends**: Parakeet ASR, the SenseVoice tagger, a cloning TTS, ctranslate2 MT, to prove the plugin contract with more than one implementation of each kind.

## Left to the builder

- Exact package layout, artifact schemas, and test strategy (unit tests on config resolution, cache invalidation, registry matching and time fitting; GPU stages get a small fixture clip and are marked slow).
- Real default models per language, picked by testing.
- Registry storage format and the default match threshold (tune it on milestone 4's labeled sample).
- Which cloning TTS goes first.
