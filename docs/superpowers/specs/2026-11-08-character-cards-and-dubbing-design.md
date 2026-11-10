# fukikae: character cards, translation and speech synthesis

_Status: approved 2026-11-08 (D17, with cards made per target language). Builds on `2026-11-04-doppio-design.md` (the foundation spec) and amends its voice assignment and TTS sections; where this document is silent, the foundation spec stands. Operator rulings: D11 (voice cards ahead of face detection), D12 (clone by default), D13 (speech style on the same card, simple for now), D15 (the operator picks the cloning backend by listening at a checkpoint; stock is the interim default if no candidate passes)._

## Intent

Every character gets a card per target language: a small file that reproduces that character's voice in that language on anyone's machine, and tells the translator how the character talks. Cards are what people improve, swap and share, so a better voice for a character, or a whole season's cast in one language, can travel as a file, and anyone can try a different voice on a character without touching the rest of the dub. How a given line is said (angry, whispered, laughing) is not on the card: it comes from that line of the original, so one voice can shout in one scene and mumble in the next.

This plan also builds the stages the cards feed: the optional `tag` stage (needed for neutral reference clips, per-line emotion and the mix's song and laughter protection), translate (including the foundation spec's subtitle-first path with condense mode), tts, timefit and mix, which together produce the first complete dubbed episode. ASR plus translation is built first, being the general case; the subtitle-first path follows in the same plan (my call).

## Goals

1. A character card is for one target language (fukikae is meant to go from any language to any language; this plan builds and tests English). It holds a **voice** section and an optional **speech style** section. The voice section names a TTS backend, a pinned model revision, a voice source and generation parameters, and renders the same voice (perceptually; bit-identical output isn't promised across GPUs) wherever it runs.
2. A new character gets a cross-lingual clone of its original actor by default (D12), with the reference clip chosen automatically from the cleanest speech the registry holds. Stock voices, blends and text-designed voices are opt-in per character.
3. Cards are shareable: exported and imported as packs keyed by the series' external ids and the character's name, so they map onto anyone's registry.
4. Swapping a character's card re-renders only that character's lines, then the mix. Every assignment is kept in a history and can be reverted.
5. Delivery is per line: the original line's audio, an optional emotion label and an optional translator tag reach the TTS backend as far as it supports them, scaled by the card's emotion strength.
6. The translator receives each speaking character's speech style, so politeness level, tics and catchphrases survive translation consistently (D13; free text for now).
7. A card can be verified: a fixed test render compared against the card's stored fingerprint says whether this machine reproduces the voice.
8. TTS quality is measured, not guessed: an `eval tts` command scores speaker similarity, intelligibility and duration fit, and writes to the results ledger by language and genre (the ledger itself comes from the ground-truth plan; see Dependencies).

## Non-goals (this plan)

- A hosted pack registry, search or ratings. A pack is a file; where people put it is theirs.
- A GUI. Cards are TOML, edited by hand or through `fukikae voices` commands.
- A linguistically grounded speech style model (registers, honorific systems, academic input). The style section is free text now and will become structured later (D13).
- Lip sync, singing, and dubbing songs (songs stay in the source language, per the foundation spec).
- Face or active-speaker detection (next plan, after this one).

## Anti-goals

- A character's voice changing between episodes without an operator action. Assignments are pinned; a card changes only by `voices assign`, `voices revert` or an import the operator confirms.
- A card that silently stops reproducing. Verification exists so a model change or a missing clip is reported, not discovered by ear three episodes later.
- Emotion baked into timbre: a reference clip of the character shouting makes every line sound like shouting. Reference clips are picked for clean, non-overlapping speech, and for neutral delivery only as far as a scored emotion exists to say so (an unscored or unlabeled turn is not known to be neutral); delivery comes from each line.
- An import overwriting curated voices. Locked or hand-assigned cards are never replaced without `--replace`.
- Truncating a line to fit its slot (inherited from the foundation spec's time fitting).

## Constraints

- One 24 GB card, one model loaded at a time (foundation spec). TTS backends declare VRAM and load/unload like every other backend.
- Offline once models are downloaded. Cloning reads only the operator's own media and registry.
- Model revisions are pinned by Hugging Face commit (or package version for non-HF backends) on the card itself.
- Cards and packs are plain files: TOML plus audio sidecars, no database needed to read them.

## The character card

One TOML file per card, stored content-addressed at `<series>/.fukikae/cards/<id>.toml`, where the id is the first 16 hex digits of the SHA-256 of the card's canonical form (TOML re-serialized with sorted keys, excluding `[meta]` and `[voice.fingerprint]`, plus the hashes of its sidecar files). The fingerprint describes how the voice renders, not what it is, so recording one never changes the card id, its assignments or any cache key (my call). Sidecar audio lives at `<series>/.fukikae/cards/clips/<sha256>.wav`, and latents at `.../latents/<sha256>.npy`.

```toml
[meta]                       # descriptive; not part of the card id
title = "Natsu, cloned from the original, a bit brighter"
author = "lokkju"
created = 2026-11-08
derived_from = "3f9a0c1d2e4b5a69"   # the card this one was edited from, if any

target_language = "en"              # part of the card id; a character has at most one assigned card per target language

[voice]
backend = "indextts2"
model_revision = "a1b2c3d"          # pinned
source = "clone"                    # stock | blend | clone | latent | design
seed = 1234
temperature = 0.7
top_p = 0.9
rate = 1.0                          # base speaking rate
pitch_semitones = 0.0
emotion_strength = 0.8              # 0 = ignore per-line emotion, 1 = full

[[voice.references]]                # source = "clone"
clip = "sha256:9e1f..."             # sidecar file
transcript = "なんだよ、それ"
language = "ja"
origin = { episode = "S01E11", start = 512.40, end = 516.10 }   # provenance, informational

[voice.blend]                       # optional, any source with a speaker embedding
with = "card:77c1e0aa12f3b4c5"      # or a stock voice id
weight = 0.3

[voice.options]                     # backend-specific; validated by the backend
# e.g. diffusion steps, a vocoder choice

[voice.post]
gain_db = 0.0
eq = "none"                         # a named preset

[voice.fingerprint]                 # written by `voices verify --record`
embedder = "wespeaker:resnet34"
vector = "sha256:5d2c..."           # .npy sidecar: embedding of the canonical test render
threshold = 0.85

[style]                             # optional (D13): fed to the translator
register = "casual and rough; first person 'ore'; never polite speech"
address = "calls Lucy by name, Gray 'ice-brain'"
tics = "ends boasts with 'got it?'"
catchphrases = [ { source = "燃えてきたぞ", target = "I'm all fired up!" } ]   # target: the rendering in this card's target language
notes = ""
```

The voice sources are:

- **stock**: `voice = "<id>"` from the backend's built-in voices (Kokoro).
- **blend**: `voices = [{ id, weight }, ...]` of stock voices, for backends whose voices are vectors (Kokoro).
- **clone**: one or more `references` with transcripts.
- **latent**: a speaker latent or embedding sidecar exported by a backend that supports it, which shares a clone without its audio.
- **design**: `description = "..."` for backends that build a voice from text (Qwen3-TTS VoiceDesign, Parler-TTS). The seed is essential here.

Each TTS backend declares which sources and which `[voice]` keys it supports, and validates `[voice.options]`; a card naming a key or source its backend can't honor fails at load, naming the key, as the config system already does for profiles.

## Series metadata

`fukikae.toml` gains a `[series]` table: `title`, optional `tmdb`, `tvdb`, `anidb` ids, and `genre` (defaulting to the profile name: anime, live-action, default). The operator sets these by hand; nothing is looked up online, keeping the offline constraint. `characters` gains an optional `external_id` (an AniDB character id, set with `registry rename --external-id`). Pack import refuses a conflicting id (both sides have, say, an AniDB id and they differ) and passes on a shared one; when the two sides share no id it compares normalized titles: equal passes with a warning that matching was by title, different refuses. `--force-series` overrides either refusal. The results ledger reads `genre` from here.

## Registry changes

- `characters.voice` (free text today, unused) is replaced by a `card_assignments` table keyed by character id and target language, plus a `card_history` table: character id, target language, card id, assigned when, by what (`auto`, `assign`, `import:<pack>`, `revert`), and whether the operator locked it. The migration moves any non-empty `voice` value into history as a note.
- Cards are referenced by id, so editing a card makes a new card. `voices edit <character>` opens a copy, writes the new card with `derived_from`, and assigns it.

## Target languages

The series' `target_language` (foundation spec config) picks which assignment the tts stage reads. A card for another language is usually derived from an existing one: `voices derive <character> --to es` copies the voice section (same references, so the same timbre for a clone) into a new card for the new target language, with `derived_from` set, for the operator to adjust; style catchphrase renderings are per target language and start empty. Cards for different target languages never replace each other.

## Default assignment (D12)

When the characters stage creates a character, or when a character first needs TTS, fukikae builds a clone card:

- References come from the character's exemplars on the separated vocals: turns with no overlap, a high speech-to-noise ratio, between 3 and 12 s each, and, as a ranking rule, not a non-neutral emotion: a turn that a scored emotion source (emotion2vec, in the anime and live-action profiles) labels non-neutral with confidence at least 0.5 ranks after every other eligible turn, so it is used only when needed to reach the minimum total; an unlabeled, unscored or low-confidence turn is ranked by signal-to-noise and overlap alone (SenseVoice's labels carry no score, so they demote nothing). Whenever a chosen reference is scored non-neutral, the card report and the `voices` output say "references may carry emotion: <labels>", up to the backend's preferred total (my call: 3 references totalling 10-20 s, the builder tunes this). Their transcripts come from the ASR artifact.
- Fragments and the clone floor (operator ruling D32, after the A19 listening test found a one-clip, 6.3 s shouted clone strained). Before the 3 s check, a character's adjacent turns at most 0.6 s apart (`JOIN_GAP_S`) with no other speaker's turn in the gap are joined, up to 12 s; a joined turn that fails a check (an excluded span or 10% of someone else's subtitle names over its gaps, a sparse transcript) falls back to its pieces. The preferred total's lower end is a floor: at least 2 clips and 10 s, and scored non-neutral turns are taken until it is reached rather than only up to 6 s. Episodes whose picks reach the floor are preferred over ones with more seconds that don't; the name-confirmed tier still comes before the acoustic one. A character whose best picks fall short keeps a clone from them, flagged for review (tts `run.json` flags, `voices list`, `names show`) with the shortfall in clips and seconds and `voices clone` as the fix; never a stock voice or another character's card. An automatic clone card under the floor is re-picked like a stale one (a new card id, the old kept in history, CD1) when the current rules give picks that beat it (reaching the floor, then clips up to 2, then seconds); curated, imported and locked cards are never changed.
- A character with too little clean speech for that (my call: under 6 s total) gets a stock voice from a gender-matched pool instead, as the foundation spec described, and the report says so. The operator can still clone from a clip by hand (`voices clone <character> --clip <file>`).
- Auto cards are unlocked; `voices assign` and `voices lock` lock them.

## Delivery

Each cue carries a `delivery` record, built in the translate stage and stored with the cue in `05-translate/cues.en.json`:

- `reference`: the cue's source span on the separated vocals (start, end), the emotional reference audio;
- `emotion`: a label and confidence from the optional `tag` stage (SenseVoice, or emotion2vec as a second backend), when it ran;
- `direction`: a short free-text tag from the translator ("whispering", "sarcastic", "shouting over noise"), which the prompt asks for only when the delivery is marked.

TTS backends declare capability flags: `emotion_audio` (takes an emotion reference separate from the speaker reference), `emotion_vector`, `instruction` (takes text direction). The tts stage passes what the backend supports, scaled by the card's `emotion_strength`, and records what it passed in `06-tts/plan.json`. A backend with none of these renders neutral speech; that is reported, not hidden.

## Speech style and the translator (D13)

The translate stage (foundation spec, "Translation") adds to each window's prompt the `[style]` section of every character speaking in it, and the catchphrase table as fixed renderings. Glossary and previous-window context stay as the foundation spec describes. Style text is untrusted input to the LLM in the same sense as subtitle text: it shapes wording, never the response format. The section stays free text in this plan.

## Sharing: packs

A pack is a directory (or a `.tar.zst` of one):

```
pack.toml        # format version, series title, external ids (tmdb, tvdb, anidb), pack title, author, notes
cards/<id>.toml  # the cards, with sidecars
clips/ latents/
cast.toml        # (character key, target language) -> card id; key = name plus optional external character id (AniDB character id)
```

- `fukikae voices export <series> [--characters ...] [--no-clips] -o <pack>` writes one. `--no-clips` exports only cards that need no audio (stock, blend, design, latent) and lists the ones left out.
- `fukikae voices import <series> <pack> [--dry-run] [--replace]` matches characters by external character id, then by name (case- and width-insensitive), shows the mapping and the unmatched characters, and assigns the cards with history `import:<pack title>`. Locked or hand-assigned characters are skipped unless `--replace`. A pack whose series ids conflict with the series' own, or that shares no id with it and has a different title, is refused unless `--force-series`.
- Registry ids never appear in a pack; only the character key does.

Sharing therefore works between registries whose characters are named. Naming is the operator's (registry `rename`, or a cast-list import later).

## Swapping and history

`voices assign <character> <card|pack-card|stock id>`, `voices history <character>`, `voices revert <character> [--to <n>]`, `voices lock`. Any of these invalidates `06-tts` for that character's cues, then `07-timefit` and `08-mix`. Per-cue TTS cache keys make this cheap: a cue's key is the hash of its card id, text, delivery record and resolved backend config, so the other characters' audio is reused.

## Verification

`fukikae voices verify <card> [--record]` renders a fixed English test script, embeds the result with the card's fingerprint embedder, and compares it with the stored fingerprint (cosine against `threshold`). `--record` writes the fingerprint; import and assign run verify when a fingerprint exists and report a mismatch (missing clip, different model revision, backend not installed) without blocking. Note: a fingerprint says the voice is the same voice, not that it is a good one.

## Stages

These follow the foundation spec, with what this plan adds:

- **05-translate**: as specified, plus `[style]` in the prompt and the delivery record per cue. Default backend `openai_compat` in `managed` mode (a small GGUF on llama.cpp), per the foundation spec.
- **04b-tag** (optional, foundation milestone 7): SenseVoice-Small on CPU over the vocals for event spans (BGM, singing, laughter, crying); per-turn emotion labels with a confidence from emotion2vec+ large (`emotion_backend`, CPU) in the anime and live-action profiles, because SenseVoice labels about 9% of turns and scores none. Default on in the anime and live-action profiles once built (my call), since reference selection, delivery and the mix use it.
- **06-tts**: one wav per cue from the character's card. Backends: Kokoro (stock, blend) and one cloning backend with emotion separate from timbre. IndexTTS2 is the leading candidate (timbre and emotion references are separate inputs, and it offers duration control); CosyVoice 3 and Qwen3-TTS are the alternates. The operator picks (D15): the plan builds the Kokoro path end to end first, then the candidate backends with `eval tts`, then stops at a checkpoint where the operator listens to a short A/B per candidate with the numbers alongside. If no candidate clones ja to en acceptably, stock voices become the interim default for new characters (D12 resumes once one passes), and the choice is recorded in the ledger. The evaluation runs on Fairy Tail E11-E14 (ja to en) and The Big Bang Theory S01's German and Spanish dub tracks (de and es to en) once they land. BBT's original English track is a free reference there: `eval tts` compares the dubbed English line's duration and delivery with what the real English line did (my call).
- **07-timefit**: as specified. When the backend offers duration control, the target duration goes to the TTS call first, ahead of time-stretch in the foundation spec's order.
- **08-mix**: as specified (background bed, original vocals kept outside dub cues, laughter and songs untouched, EBU R128, mux with an ASS track naming speakers).

## TTS evaluation

`fukikae eval tts <workspace> [--episodes ...] [--backend ...]` renders a sample of cues per character and reports, per character and overall:

- speaker similarity of the render to the card's references (WeSpeaker cosine, the same embedder as identity work);
- intelligibility: word error rate of an English ASR pass over the render against the cue text;
- duration fit: how often the render fits its slot before time-stretch, and the stretch factor needed;
- emotion carry-over, when the tag stage ran: agreement between the source line's label and the render's.

Rows go to the results ledger keyed by backend, model revision, language pair and genre. The listening test stays the deciding input for the backend choice; the numbers catch regressions and compare configurations.

## Dependencies

- The **ground-truth plan** (D14, separate and smaller, runs first and owns the ledger's row schema and `eval report`; this plan's `eval tts` writes rows in that schema): the BBT transcript aligner, the Japanese （名前） SDH parser for Alice in Borderland, the per-character drama clip set for embedder checks, and the results ledger with `fukikae eval report`. `eval tts` writes to that ledger.
- The identity work's registry (merged). Cards attach to its characters.

## Calls I made (for the operator to check)

- Card ids are content hashes, so an edited card is a new card, and history points at immutable cards.
- Reference selection defaults: 3 clips, 10-20 s total, 3-12 s each, scored non-neutral turns rank last and are reported when used; under 6 s of clean speech falls back to stock. Amended by D32: fragments 0.6 s apart are joined first, and 2 clips and 10 s is a floor that non-neutral turns fill and below which a clone is flagged.
- Packs carry no registry ids; matching is by external character id, then name.
- Fingerprints use WeSpeaker, the identity embedder, for one embedding space across the project.
- Verification reports but never blocks.
- The translator's delivery `direction` is requested only for marked delivery, to keep prompts short on a small model.
- The fingerprint is excluded from the card id, so recording one never re-renders anything.
- This plan builds the `tag` stage (on by default in the anime and live-action profiles) and the subtitle-first path, ASR plus translation first.
- Series ids and genre are set by hand in `[series]`; pack import refuses on conflicting ids, and with no shared id matches by title (with a warning) or refuses a different title; `--force-series` overrides.
- BBT's German and Spanish dubs are the second TTS evaluation set, with the original English track as the reference for duration and delivery.

## Left to the builder

- Exact artifact schemas, the canonical serialization for card ids, the test sentences for verification, and the gender-matched stock pools.
- The listening-test materials: a handful of cues per character rendered by each candidate, laid out for the operator's A/B (a static page or a folder with an index), and how the verdict is recorded.
- Whether `blend` across different backends is refused or approximated (my lean: refused).
- Pack compression and size limits for clips.
