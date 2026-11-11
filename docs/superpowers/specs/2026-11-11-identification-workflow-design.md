# Character identification workflow: design

_Draft, 2026-11-11, for lokkju/fukikae#33 (operator ruling D40: yes). Builds on the character cards spec (`2026-11-08-character-cards-and-dubbing-design.md`, rulings CD1 to CD26) and the subtitle speaker names spec (`2026-11-10-subtitle-speaker-names-design.md`, D27 and D29), with D32, D33 and D34 standing. Evidence: the Sol Bianca S00E01 diagnostic (`~/fukikae-scratch/sb-diagnose/report.md` on the GPU host, summarised in the issue)._

## Why

The Sol Bianca dub sounds wrong for four measured reasons, none of them bandwidth or flat prosody: 35 of 402 cues with a known gender on both sides (8.7%, strict bands) are rendered in the other gender, 68 of 483 (14.1%) at a single 180 Hz cut. Extras disagree with their source line on 21 of 29 known-gender cues because nothing reads their pitch. Four clone cards (chars 1, 2, 3 and 10, 217 of 570 cues) are built from references that are narration in another register, a mixed-gender cluster, an unvoiced processed voice or an English insert song. One "character" (char 11, 12 cues) is Whisper's 「ご視聴ありがとうございました」 over the ending song. The 51 stock-voiced cues have a render-vs-source speaker cosine of 0.06 against 0.29 for clones: they are voices that belong to nobody on screen.

Some of this is fixable without a person: pitch is measurable and the bad references fail tests anyone would write down. Some of it is not: SPEAKER_02 and SPEAKER_16 (cosine 0.76) are probably one speaker, SPEAKER_13 holds a male butler and a female speaker, and the girl who narrates the opening is the same character as char 1 in another register. Those need an operator, and today the operator's only tools are `registry merge`, `registry split`, `registry rename`, `voices clone` and reading `names show` output. Part A is the automatic fixes; Part B is a review checkpoint that puts the evidence on one page per episode and writes the operator's answers through the registry API.

## Goals

- Clone references that sound like the character's ordinary speech: in its pitch register, voiced, outside songs, and not all from one scene.
- Stock voices, extras included, that match the source speaker's gender.
- No line whose source is the other gender drags a clone toward that gender through `emotion_audio`.
- No ASR hallucination over the credits becomes a character.
- A per-episode review page with sample clips, estimated gender, card and references, suggested merges and splits, names evidence and flags, from which the operator exports decisions that `fukikae review --apply` writes into the registry.
- Gender as a stored character attribute, operator-set over estimated.
- An opt-in `[review] required = true` that stops `fukikae run` until an episode's new characters are reviewed, carried across a season through the registry.

## Non-goals

- Automatic merges or automatic cluster splits. Both are suggestions on the review page; the operator decides (the issue's own call on SPEAKER_02 and SPEAKER_16).
- Room or EQ matching of renders in the mix (the diagnostic found `match_dialogue` is level only; that is a mix issue, not identification).
- Face or active-speaker evidence (#3) and fixes for merged live-action leads (#4). The decisions file leaves room for another evidence source; it adds none.
- The full character web UI (#25). The review page is its first slice: decisions are data, the HTML is one client.
- Fetching cast lists from TMDB or AniDB at run time (CD19, CD24). Cast priors are a later phase, below.

## Anti-goals

- An automatic change to a curated, imported or locked card (CD2, CD3, CD9). Every new gate re-picks automatic cards only and flags the rest, the way D29 #4 and D32 do.
- A character's voice changing between episodes without an operator action or a rule this spec names (the cards spec's anti-goal). Each re-pick reason gets its own history note.
- Tuning a threshold on a held-out episode. Fairy Tail E14 stays held out; Sol Bianca S00E01 is the evaluation episode and is reported, never tuned on.

## Terms

"Gender" in this spec means the register of the voice as heard, the thing a stock voice or a reference must match, never the character's gender in the story. Happy in Fairy Tail is a male character with a high voice; for fukikae he is `f` unless the operator says otherwise. The review page labels it "voice: female / male / unknown" for that reason.

"Review" collides with `ClusterAssignment.status = "review"`, which already means "this cluster lost a tie and adds no exemplars". The new feature is "the identification review" or "the checkpoint" in prose, `fukikae review` on the command line and `review/` on disk; the status value keeps its meaning and is never reused.

## Part A: automatic fixes

### Ranking

| Change | Strict mismatches fixed | At the 180 Hz cut | Cues whose voice changes |
|---|---|---|---|
| A1 reference gates | 3 | 23 | 217 |
| A2 extras gender, pools, anime threshold | 19 | 21 | 39 |
| A3 per-line gender gate | 10 (estimate for rerouting; see below) | 11 | 24 flagged |
| A4 song tail and hallucinations | 0 | 5 | 12 removed |

A2 fixes the most strict mismatches and is the cheapest; A1 changes the most cues. The plan builds the shared pitch module first, then A2, A1, A3 and A4, each shippable alone.

### A0: one pitch estimator

Everything in Part A reads pitch, so it gets one module, `fukikae.pitch`, CPU only, numpy and scipy only.

The pipeline estimator today (`cards.select.speaker_f0`) returns the 20th percentile of an autocorrelation F0 between 70 and 400 Hz. Both choices hurt here: the cap sits under the median of three Sol Bianca clusters (SPEAKER_16 406 Hz, SPEAKER_02 377 Hz with a p90 of 622 Hz), so high frames fold to subharmonics, and the 20th percentile reads 172 to 237 Hz for clusters whose pYIN medians are 236 to 406 Hz, which is why the anime profile had to push `female_above` to 300 and then called every woman `u`.

I keep the autocorrelation estimator and fix it rather than adding librosa's pYIN as a dependency: librosa pulls numba and llvmlite into every install for one function, and the estimator's problems are its range and its statistic, not its method. The changes:

- range 60 to 800 Hz (`F0_MIN`, `F0_MAX`), with the existing shortest-lag-within-90% rule against subharmonics;
- a frame track rather than one number: `f0_track(vocals, spans) -> F0Track` holds per-frame F0 (NaN when unvoiced) and the span each frame came from;
- `F0Summary` from a track: `median`, `p10`, `p20`, `p90`, `voiced_s`, `seconds`, `voiced_fraction`;
- classification on the **median** of voiced frames: `m` under `male_below`, `f` over `female_above`, `u` between or under a minimum of voiced speech (1.0 s for a speaker, 0.4 s for a line);
- a confidence: the distance in semitones from the nearest band edge, divided by 4 and capped at 1, and 0 for `u`. The review page shows it; A3 reads it.

`speaker_f0` stays as a thin wrapper (it returns the new median) so `eval tts` and the stock-card note keep working; the note says "median" instead of "20th percentile".

Calibration is a plan task, done before any profile value changes: the estimator's class per speaker and per line against pYIN (run through `uv run --with librosa` in a script, never a dependency) and against a hand-written voice-gender truth file on the tuning episodes of three sets that already have workspaces on the GPU host: Fairy Tail E12 and E13 (ja, the anime profile's tuning episodes), The Big Bang Theory (en, live-action tuning episodes) and Kaze (ja). Sol Bianca S00E01 is reported beside them and not tuned on.

The proposed defaults, before calibration, are `male_below = 165` and `female_above = 200` on the median. On Sol Bianca's pYIN medians they separate every cluster that isn't mixed: the women sit at 236 to 406 Hz, the men at 88 to 145 Hz, and the two ambiguous clusters (SPEAKER_13 at 205 Hz, which is the mixed one, and SPEAKER_15 at 193 Hz, the hallucination) fall in or near the unknown band where they belong. Adult speech norms (roughly 85 to 155 Hz for men, 165 to 255 Hz for women) put the same edges in the same place. The calibration's acceptance rule: no main character (30 s or more of speech) on any of the three sets lands in the other class; the unknown band holds at most 10% of main characters' seconds; per-line agreement with pYIN at least 90% where both are known. If one set needs other edges, that set's profile gets them and the results doc says why.

`[tts.gender]` keeps its keys (`male_below`, `female_above`) and they now apply to the median. A workspace that sets them was tuned on the 20th percentile, so `gender_thresholds` logs a warning naming the change once per run when the workspace (not a profile) sets either key. The anime profile drops its `female_above = 300` override.

### A1: reference gates

`cards.select.eligible_references` gains four gates, applied to each candidate after the existing ones (overlap, length, song exclusions, foreign names, sparse text). Each is a filter, not a change to the CD5 ranking: SNR order among the survivors stays as it is.

1. **Pitch register.** The character's speech median is computed over the majority-gender turns of the character's speech in that episode (each turn classified by its own median, the majority by seconds), so a mixed cluster like SPEAKER_13 gets the register of its larger part and not a meaningless average. A candidate whose median is more than `REF_PITCH_ST` (4 semitones) from it is dropped. Char 1's narration (171 to 183 Hz against 240 Hz, about 5.3 semitones) fails; char 10's male clips (127 and 147 Hz against the female part) fail; char 2's 192 Hz line (11 semitones under 377 Hz) fails.
2. **Voicing.** A candidate under `REF_VOICED_MIN` (50%) voiced frames is dropped: char 2's processed voice had none.
3. **Song-like audio.** A candidate with `REF_SONG_SHARE` (half) or more of its seconds song-like by the theme detector's per-second test (music bed within 3 dB of the voice, voice active) and a voiced share at or over the detector's `voiced_share` (0.7) is dropped. This is the theme detector's evidence applied per turn anywhere in the episode, so it catches char 3's English insert song, which no theme span and no singing tag covered. A false positive costs one candidate, so the per-turn test can be looser than the theme detector, which decides what the mix leaves untouched.
4. **One scene.** Candidates whose starts are within `SCENE_GAP_S` (60 s) of an already-taken pick count as the same scene; the greedy fill takes at most one clip per scene until it would fall short of the D32 floor, and only then takes a second from a scene already used. Narrator characters (Part B, `kind = "narrator"`) are exempt: narration is often one long scene, and that is fine for a narrator.

A gated candidate is logged at debug with the gate that refused it, and `reference_report` adds one line when a gate removed a candidate that would otherwise have ranked in the top three ("gated: 2 candidates off-register, 1 song-like"), so `voices show` can explain a surprising pick.

**Re-picking cards the gates now refuse.** A new `impure_cards(ws, reg, store, language)` in `cards.auto`, beside `stale_cards` and `below_floor`, measures every automatic clone card's references on their origin spans (the episode's vocals, `origin = {episode, start, end}`) and lists the cards with a reference that fails gate 1, 2 or 3 (gate 4 is about the set, and the floor already judges sets). `ensure_assignments` treats them exactly as D29 #4 treats stale cards and D32 treats cards under the floor:

- an automatic card (`replace_refusal(a, "auto")` is None) is re-picked when the current rules give picks that pass every gate, a new card with a new `auto` row noted `re-picked: references fail the reference gates (<which>)`;
- with no passing pick, it is kept and flagged (never made stock, as D32 rules for the floor);
- a curated, imported or locked card is flagged only, naming the failing reference and pointing at `fukikae voices clone`;
- a card with no usable origin (`voices clone` cards say `{"source": "voices clone"}`) is never judged.

Precedence when a card is stale, below the floor and impure at once: one re-pick, with all the reasons in the note.

### A2: extras gender, pools, the anime threshold

- `extras_voices(cues, cfg, genders)` takes each extra label's gender instead of passing `u`. For an extra cluster, the gender is the cluster's `F0Summary` class over its turns in `assignments.json`. For the cluster-less `unknown` label it is per line: the label is split into `unknown~f`, `unknown~m` and `unknown~u`, each cue keyed by its own source span's class, so a woman's "嗯" and a man's "Huh?" no longer share `am_echo`.
- A `u` extra (too little voiced speech, or a median in the unknown band) takes the pool its median is nearer to by the band's geometric midpoint (about 182 Hz), and alternates only when there is no median at all.
- The extras pools widen from two voices each to five: `extras_f` becomes `af_kore`, `af_nova`, `af_aoede`, `bf_lily`, `af_river`; `extras_m` becomes `am_echo`, `bm_daniel`, `am_eric`, `am_liam`, `am_onyx`. `af_alloy` leaves the female pool: its renders of c0013 and c0014 read 147 and 149 Hz. `bf_alice` leaves it too, to be re-admitted only if the calibration below passes it. None of these is in a character pool, so an extra still never sounds like a character.
- Each candidate pool voice is checked once, in a plan task, by rendering a fixed set of ten English sentences on Kokoro (CPU) and requiring the render medians to sit in the pool's own class by the A0 bands. A voice that fails is left out and the results doc says so.
- Character stock voices (`ensure_assignments`'s stock path) use the stored gender when the operator set one (Part B), else the A0 estimate.

The extras' voice choice stays fixed per label by first appearance, so a label keeps its voice when a translation changes. It is not a registry assignment, and the tts stage key already holds the extras map.

### A3: per-line gender gate

The tts stage computes each spoken cue's source class from its `delivery.reference` span (A0, line minimum 0.4 s voiced) and compares it with the speaking character's established gender: the operator's when set, else the class of the character's card references (measured on the card's clips, so it is the voice that will actually render), else the A1 majority class of the character's speech.

When the line is strictly the other class (`m` against `f` or the reverse; `u` on either side never triggers), there were three ways to route it: flag it for review, send it to a better-matching character, or render it in its own character's clone without `emotion_audio`. I chose the third, with a flag:

- **The line renders in its character's card with no `emotion_audio`**, recorded in `plan.json` as `emotion_dropped: "gender"` (the same field CD18 uses for laughter and crying), and `TtsPlanCue` gains `source_gender` and `gender_gate: true`.
- **The line is flagged** in `run.json` and on the review page, with the cue, its source median and the character's established gender.
- **Wrong-gender source audio is never passed as `emotion_audio`**, to any card, on any path: the rule is enforced in `delivery_passed` itself, so it holds for extras and for `eval tts` (CD12, one render path).

Why not the other two. Sending a line to a better-matching character needs a line-level speaker embedding the tts stage doesn't have (diarize embeds clean spans per cluster, not per cue), and on pitch alone it would move lines between characters on a measurement that is weakest exactly where these lines live: short interjections, shouts and processed voices (c0288's 「嗯」 reads 648 Hz). Moving lines is the D27 kind of identity decision, and names already win over acoustics; a pitch rule shouldn't quietly outrank both. Flagging alone leaves the measured harm in place: IndexTTS2's emotion prompt is what pulls renders to the line's pitch (log-F0 correlation source-to-render 0.64 to 0.86 for most clones; 44 of char 1's 81 renders pulled out of the card's register). Dropping `emotion_audio` removes the active cause while keeping the character's voice, and the flag sends the real problem, a line diarization gave to the wrong speaker, to the review page, where a split or a merge fixes it for every line at once.

The cost, stated plainly: a line that truly belongs to someone of the other gender still renders in the character's voice until the operator fixes the cluster. The diagnostic's "10 / 11 fixed" assumed a gender-matched stock voice; this choice will fix fewer of the counted mismatches. The evaluation reports both numbers (mismatches, and wrong-gender emotion prompts passed, which must be 0). Rerouting to a gender-matched extras voice is open decision 3.

Narrator characters are exempt from the flag (narration often sits in another register by design) but not from the `emotion_audio` rule.

### A4: song tail and ASR hallucinations

**Ending songs.** The theme detector looks for one best 80 s window in the last 240 s and grows it up to 100 s. Sol Bianca's ending has two song stretches and the detector caught 3658 to 3740 s only, missing 3771 to 3817 s. The detector gains a second pass over the tail region: after a span is accepted, the rest of the tail (outside accepted spans) is searched again with a shorter window (`tail_window_s`, 30 s) under the same score and voicing tests, repeated until nothing passes; spans within `grow_s` of each other merge. `[mix.songs]` gets `tail_window_s = 30` and `tail_repeat = true`; `VERSION` goes to 3, so the mix stage reruns where it was keyed on version 2. The existing measurement set has to hold: Fairy Tail E11 to E14 openings and endings, Kaze's opening on E01 to E03, and no span on The Big Bang Theory (en, de) or Grid (`scripts/song_scores.py` reproduces it).

**Hallucinations.** A new CPU-only stage, `asrclean`, after `asr`, reads `03-asr/transcript.json` and the separation stems and writes `03b-asrclean/transcript.json` plus `03b-asrclean/dropped.json`. Every consumer of the transcript (characters, the cues builder in translate, reference selection through `episode_artifacts`) reads it through one helper, `transcript_path(ep)`, the way identity inputs go through `names.inputs.identity_inputs`. A segment is dropped when both hold:

- its text, NFKC-folded with spaces and punctuation stripped, equals a phrase in `HALLUCINATION_PHRASES` for the transcript's language or is that phrase repeated (ja 「ご視聴ありがとうございました」, 「ご覧いただきありがとうございました」, 「チャンネル登録よろしくお願いします」; en "thank you for watching", "thanks for watching", "subtitles by the amara.org community", "please subscribe"; zh, ko, de and es equivalents, the list in the module with a comment per entry saying where it was seen);
- and its span is music or no speech: inside a theme span (A4's detector), or at least 60% of its seconds song-like by the detector's per-second test, or the vocals stem active on under 30% of its 100 ms frames.

The phrase list alone would delete a character who really says 「ご視聴ありがとうございました」 to camera at the end of an episode; the music-or-silence condition keeps that line. Dropped segments are listed with the reason in `dropped.json` and on the review page.

The separate stage is deliberate: putting the filter inside `asr` would change the asr key and rerun Whisper or Qwen3-ASR on the GPU for every existing episode. `asrclean` costs a second; its output is byte-identical to its input when nothing is dropped. Translate reruns once on existing workspaces because its input path changes, and that is the price of the cleaner seam.

With A4 in place char 11 never forms (its only turns were the hallucination), and char 3's insert-song reference goes through A1's gate 3.

### Part A evaluation

`fukikae eval gender <ws> <episode>` ports the diagnostic's measurements into the eval package (R1: the pipeline never imports eval), on the A0 estimator, CPU only:

- cue gender agreement, source line against render, strict bands and the single 180 Hz cut, per character and for extras;
- extras: the stock voice's class against the source class per cue;
- reference purity per clone card: references within 4 semitones of the character's majority median, voiced at least 50%, outside song-like audio;
- gated lines (A3) and `emotion_audio` passed against a wrong-gender source (must be 0);
- cues from dropped hallucinations and cues inside song spans;
- stock-voiced cue count and the render-vs-source speaker cosine (WeSpeaker on the CPU, optional with `--similarity`).

It writes `docs/results/`-ready JSON and a Markdown table. Before and after on Sol Bianca S00E01 run on a hard-link copy (`cp -al`) of the workspace; the "after" needs the tts stage to re-render the changed cues, which uses GPU 0 under the GPU lock, so that step waits for the operator's go-ahead (open decision 7). Everything before the render (references, extras voices, gate counts, hallucination drops) is measurable on the CPU.

Targets on Sol Bianca, from the diagnostic's own estimates: strict mismatches from 35 down to 15 or fewer; extras disagreement from 21 of 29 to 3 or fewer; reference purity 100% on chars 1, 2, 3 and 10 (or the card flagged with no passing pick); char 11 gone; wrong-gender emotion prompts 0.

No regression on Fairy Tail, the D27 names arm (E11 to E14, E14 held out as before): reference purity (the names ledger's metric) and cue accuracy (`eval names`) must not drop on any episode, and no card on E12 or E13 may change except through a gate this spec names, each such change listed in the results doc.

## Part B: the review checkpoint

### Data first

The checkpoint is two files per episode and a command that applies one of them:

- `review/<episode>/review.json`: what fukikae knows, written by `fukikae review`. It is the data model #25 will read.
- `review/<episode>/index.html` and `review/<episode>/audio/*.wav`: a client of `review.json`, one page per episode in the same style as the `eval tts-ab` listening packs (`eval/ab.py`: inline CSS, `<audio preload="none">` with relative paths, owner-only directory). The page embeds `review.json` in a `<script type="application/json">` block, because browsers refuse `fetch` from `file://`.
- `review/<episode>/decisions.json` (or any path): what the operator decided, exported by the page and applied by `fukikae review --apply <file>`.

Nothing in the page writes the registry; only `--apply` does, through the registry API, under one `Registry.transaction()`.

### `fukikae review <ws> [episodes...]`

Without episodes, it builds pages for every episode that needs review (below). Options: `--out <dir>` (default `<ws>/review`), `--serve [--port 8765]` to serve the directory on 127.0.0.1 and accept the page's "save decisions" as a POST to `review/<episode>/decisions.json` (the page falls back to a browser download when opened from the filesystem), and `--clips N` (default 4, range 3 to 5).

`review.json` per episode, one entry per diarize cluster (sub-clusters included) and one per character:

- **samples**: N clips per cluster, cut from `01-separate/vocals.wav` at 24 kHz mono, chosen as the longest clean turns spread over the episode (one per fifth of the cluster's turns, so the page shows the range, not N copies of the loudest scene), each with its transcript and time;
- **gender**: class, confidence and median from A0, the per-band seconds (m, u, f), and the stored gender with who set it;
- **seconds**: total and clean seconds, turns, cues;
- **card**: the current card for the target language (id, source, `chosen_by`, locked) and its reference clips (copied from the card store's sidecars); for a character with no card yet, the references the current rules would pick (A1 included), marked "proposed", computed without assigning (as `eval tts` already does);
- **merge suggestions**: the three nearest other clusters or characters by centroid cosine (from `clusters.npz` and the registry centroids), shown when at least `MERGE_SUGGEST_COS` (0.5); on Sol Bianca only SPEAKER_02 to SPEAKER_16 (0.76) qualifies;
- **split suggestions**: a cluster is bimodal when its minor gender band holds at least 5 s and 15% of its classified seconds (the diagnostic's test: SPEAKER_13 is the only one on Sol Bianca); the page shows both parts' samples and the split they would make;
- **names evidence** (D27): the `NamesReport` key decisions for the episode, named lines over the cluster's turns, and the disagreement and stale-decision lines `names show` computes today (moved into a shared function so both use one implementation);
- **flags**: stale (D29 #4), below the floor (D32), impure references (A1), gated lines (A3), dropped hallucinations (A4), names disagreements;
- **snapshot**: each character's id, actor id, name, `name_by`, kind, gender, locked, and a digest of the registry rows the page was built from.

### What the operator decides

Per character or cluster, the page offers:

| Decision | Applied through | Invalidates |
|---|---|---|
| name | `rename_character` (`name_by = "operator"`, operator alias) | characters stage of every episode naming the character (as `registry rename` does) |
| alias | `add_alias(by="operator")` | same as name |
| gender (f, m) | new `set_character_gender(cid, gender, by="operator")` | tts of episodes where the character speaks |
| merge (keep, drop) | `merge_actors` on the two characters' actors, then `_remap_assignments` | characters of both actors' episodes |
| split by pitch | operator split file read by the names stage (below) | names, characters, and everything after, for that episode |
| split cluster out (undo a match) | `split_cluster(actor_id, episode, cluster)` | characters of that episode |
| this is narration | new `set_character_kind(cid, "narrator")` | characters of episodes where it speaks |
| not a character | new `set_character_kind(cid, "ignored")` | characters of episodes where it speaks |
| lock | `lock_character(cid, True)` | nothing downstream |
| reviewed, no change | new `mark_reviewed(cid)` | nothing |

"Invalidates" means `job.mark_stale(ep, stage)` on the named stage; the file-driven cascade (assignments.json is a translate input; cards and cues are tts inputs) reruns the rest. Every applied decision also marks its characters reviewed.

`--apply` checks the file's snapshot against the registry first and refuses, naming each difference, when a character the file decides on changed since the page was built (renamed, merged, locked by someone else); `--force` applies anyway. It prints one line per change, and `--dry-run` prints them without writing. A decision that the registry refuses (merging into a locked actor, naming a character whose name an operator already set) fails the whole file; nothing is half-applied.

### Split by pitch

`registry split` moves a whole episode cluster to a new actor; it cannot cut one cluster in two. The names stage already can: it relabels turns into `<cluster>~<slug>` sub-clusters and moves the `clusters.npz` rows without re-embedding. A pitch split reuses that machinery. `--apply` writes `review/<episode>/splits.json` (`{cluster: {"by": "pitch", "male_below": Hz, "female_above": Hz}}`); the names stage reads it as an operator input after subtitle names, classifies each turn of the cluster by its own median, and labels the parts `<cluster>~voice-f` and `<cluster>~voice-m` (the `voice-` prefix can't collide with a name slug, which comes from a subtitle name). `u` turns stay with the larger part. The split file is part of the names stage's key, so the episode reruns from names on.

### Narration and "not a character"

The registry gains a character `kind`: `character` (default), `narrator` or `ignored`.

- A **narrator** is a character with its own card. A1's one-scene gate and A3's flag skip it, and merge suggestions never pair it with a dialogue character. The Sol Bianca case is a pitch split of char 1 followed by "this is narration" on the low part: the narration gets a clone from narration, the scene speech a clone from scene speech.
- An **ignored** character's clusters get the new `ClusterAssignment.status = "ignored"` from the characters stage; the cues builder marks their cues `skip = "ignored"`, so they are never translated or rendered and the mix keeps the original audio under them. It is for hallucinations A4 misses, crowd walla, a TV in the background.

### Gender as a stored attribute

`characters.gender` (`f`, `m` or NULL) and `characters.gender_by` (`operator`, later `cast`, or NULL) in registry schema 5. The estimate is never stored in the registry: it changes with every episode's speech and is cheap to recompute, so it lives in a pitch cache under `.fukikae/pitch-cache/` keyed by the vocals' fingerprint and the spans' digest. Who reads the stored gender: A2's stock pick for characters, A1's register gate (the operator's class picks which mode is the character's when a cluster is mixed), A3's established gender, and the review page.

An operator gender that disagrees with an automatic card is a fourth automatic re-pick reason, beside stale, below the floor and the `auto-fallback` upgrade: an automatic stock card of the other gender is re-picked from the right pool, and an automatic clone card whose references are the other class is re-picked under the gates with the operator's class as the register. Curated, imported and locked cards are flagged, as everywhere else.

### Where the checkpoint sits in `fukikae run`

`[review]` is a settings table (exempt from the R3 `backend` rule like `[resolve]` and `[names]`), read from profile and workspace:

```toml
[review]
required = false        # true: stop before translate until the episode is reviewed
min_seconds = 20.0      # a character with less speech never holds an episode up
```

With `required = true`, `job.run_stages` checks a gate before the `translate` stage. An episode needs review when it has a character (kind `character` or `narrator`, not `ignored`, not an extra) with at least `min_seconds` of speech in the episode that has never been reviewed, or a flag raised since its last review (a new bimodal split suggestion, impure references, a names disagreement). Such an episode stops with exit code 3 and a line naming the command (`fukikae review <ws> <episode>`); in a multi-episode run the other episodes go on. Without `required`, nothing stops, and `fukikae review` works the same on demand.

I put the gate before translate, not before tts as the issue suggested, because `05-characters/assignments.json` is a translate input: any decision that changes identity (merge, split, ignore) reruns translate, and translate is the GPU LLM step. Stopping before it costs nothing (the page needs turns, clusters, pitch and proposed references, all available after `characters`) and saves a translate run per reviewed episode. The tradeoff: the page can't show translated lines, and with the gate before tts translate could overlap with the operator's review. Open decision 4.

### Across a season

The registry is per workspace, which in practice is a series or a season, and decisions go into it, so they carry: a name, gender, kind or lock set on episode 1 holds when episode 2's clusters match that character. "Reviewed" is a timestamp on the character (`characters.reviewed`), so episode 2 stops only for characters that are new or newly flagged. A season review is `fukikae review <ws>` with no episodes, which builds pages for every episode that needs one; an operator can work through E01 and then find E02 to E12 asking about two new characters between them.

### Fit with #25 and #26

`review.json` and `decisions.json` are versioned plain files (`schema_version = 1`). #25's web UI reads `review.json` (or the same builder as an API) and posts `decisions.json` to the same apply function; the HTML here is the first client, not the model. For #26's shareable season details, decisions carry each character's key (folded name) and external id beside the workspace-local registry id, and each applied change is a registry row with `by = "operator"`, so a later pack export can carry names, aliases, genders, kinds and locks keyed the way `cast.toml` already keys cards.

### Cast priors (later phase)

A hand-written `<ws>/cast.toml` (`[[cast]] name, gender, external_id, aliases`) is cheap and is the optional last task: the page offers its names in the name field, shows the cast count against the character count ("cast lists 9 speaking roles; this episode has 14 characters"), and offers each cast gender as a one-click gender decision. Nothing from the cast list is applied without the operator's click, and an applied cast gender is `gender_by = "operator"`, not `cast`. Fetching the list from TMDB or AniDB (`fukikae cast fetch`, an explicit network command like `backends setup`, never at run time) and using cast genders as priors without a click (`gender_by = "cast"`) are later work.

## Calls I made (for the operator to check)

1. Fix the autocorrelation estimator (range to 800 Hz, median) instead of adding pYIN; pYIN is the calibration reference only.
2. Default bands 165 and 200 Hz on the median, subject to calibration; the anime override goes.
3. A1's gates filter, never re-rank, so CD5 stands.
4. A3 renders the line in its own clone without `emotion_audio` and flags it, instead of rerouting.
5. A4's filter is a separate `asrclean` stage, so no ASR reruns.
6. The checkpoint gate sits before translate.
7. Pitch splits ride on the names stage's sub-cluster relabelling, with `~voice-f` and `~voice-m` labels.
8. Gender is voice register, stored only when the operator sets it.

## Open decisions (for the operator)

1. **Estimator:** fix the in-house autocorrelation estimator (range 60 to 800 Hz, median), or add librosa and use pYIN in the pipeline? Recommended: fix the estimator; pYIN stays a calibration reference run with `uv run --with librosa`.
2. **Bands:** `male_below = 165`, `female_above = 200` on the median for every profile (anime override removed), subject to the calibration task's acceptance rule? Yes or no. Recommended: yes.
3. **A3 routing:** render a wrong-gender line in its own character's clone without `emotion_audio` and flag it (a), or send it to a gender-matched extras stock voice (b)? Recommended: (a); (b) fixes more of the counted mismatches but puts a voice that belongs to nobody (cosine 0.06) in the middle of a character's scene, and the review page's split fixes the cause for every line.
4. **Gate position:** stop before translate (a) or before tts (b)? Recommended: (a), because identity decisions rerun translate.
5. **Extras pools:** replace `extras_f` with `af_kore, af_nova, af_aoede, bf_lily, af_river` and `extras_m` with `am_echo, bm_daniel, am_eric, am_liam, am_onyx` (dropping `af_alloy` and `bf_alice`), each voice admitted only if its CPU renders fall in its class? Yes or no. Recommended: yes.
6. **Hallucination drop rule:** phrase match and music-or-silence (a), or phrase match alone (b)? Recommended: (a).
7. **GPU for the after-measurement:** may the Sol Bianca and Fairy Tail re-renders for the evaluation run on GPU 0 under the GPU lock (a few hundred IndexTTS2 lines)? Yes or no. Recommended: yes, scheduled when no other GPU job is queued.
8. **Cast priors:** ship the hand-written `cast.toml` suggestions in this plan (a) or leave all of cast priors to a later issue (b)? Recommended: (a); it is one small task and gives the page its name and gender suggestions.

## Left to the builder

- The page's layout and keyboard shortcuts, within the listening packs' style.
- The exact sample-clip spread and cut lengths.
- The `HALLUCINATION_PHRASES` entries beyond the ones named, each with a source comment.
