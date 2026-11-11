# Character identification workflow: design

_Draft, 2026-11-11, for lokkju/fukikae#33 (operator ruling D40: yes), revised the same day for operator ruling D41 (below). Builds on the character cards spec (`2026-11-08-character-cards-and-dubbing-design.md`, rulings CD1 to CD26) and the subtitle speaker names spec (`2026-11-10-subtitle-speaker-names-design.md`, D27 and D29), with D32, D33 and D34 standing. Evidence: the Sol Bianca S00E01 diagnostic (`~/fukikae-scratch/sb-diagnose/report.md` on the GPU host, summarised in the issue)._

## Why

The Sol Bianca dub sounds wrong for four measured reasons, none of them bandwidth or flat prosody: 35 of 402 cues with a known gender on both sides (8.7%, strict bands) are rendered in the other gender, 68 of 483 (14.1%) at a single 180 Hz cut. Extras disagree with their source line on 21 of 29 known-gender cues because nothing reads their pitch. Four clone cards (chars 1, 2, 3 and 10, 217 of 570 cues) are built from references that are narration in another register, a mixed-gender cluster, an unvoiced processed voice or an English insert song. One "character" (char 11, 12 cues) is Whisper's 「ご視聴ありがとうございました」 over the ending song. The 51 stock-voiced cues have a render-vs-source speaker cosine of 0.06 against 0.29 for clones: they are voices that belong to nobody on screen.

Some of this is fixable without a person: pitch is measurable, the bad references fail tests anyone would write down, and a stock voice can be replaced by a clone of the speaker's own speech, however little of it there is. Some of it is not: SPEAKER_02 and SPEAKER_16 (cosine 0.76) are probably one speaker, SPEAKER_13 holds a male butler and a female speaker, and the girl who narrates the opening is the same character as char 1 in another register. Those need an operator, and so does the last step automatic selection can't take, which is listening to a clone and deciding it sounds like the character. Today the operator's only tools are `registry merge`, `registry split`, `registry rename`, `voices clone` and reading `names show` output. Part A is the automatic fixes. Part B is the studio: a local web app for a season of a show, where the operator fixes identity across the season and tunes each character's cloned voice per target language until it is right, and the result persists for every later episode and language.

## Operator ruling D41 (2026-11-11)

1. Part A's automatic fixes are approved, except that **fukikae never uses a stock voice, anywhere**. Every speaker, extras clusters included, gets a clone from its own speech; a speaker with thin audio is cloned from whatever there is and flagged to the workflow, never given a stock voice. The stock fallbacks go: the under-6 s stock card rule in `cards.auto`, the `auto-fallback` stand-in when cloning can't run, and stock cards through `voices assign` (kept only as an explicit operator choice, or removed; this spec removes it, A2 below). Existing workspaces' automatic stock cards are re-picked as clones at the next tts run; curated, imported and locked cards follow the existing rules. The gender bands, the pitch module, A1, A3 and A4 stand as specified; A3's routing (the line renders in its own clone without `emotion_audio`, flagged) is consistent with the rule. The evaluation follows.
2. Part B is redesigned: the per-episode review checkpoint is replaced by a season, show and character workflow: detect the characters across a season with sample clips; per character and per target language, a voice-tuning loop (choose or swap reference clips from the eligible turns with the floor and gates shown, adjust render parameters, audition typed and real lines beside the original, A/B against earlier versions, accept, which writes a curated card); identity editing in the same place (name, gender as register, merge, split, narrator, not a character, lock); season state in the registry and the card store, reused by later episodes and other languages and shareable later. It is the first version of #25.
3. The rulings are recorded here, and the open decisions list drops what D41 settled (the extras stock pools; A3's routing; the per-episode gate position) and gains the new ones.

## Goals

- Clone references that sound like the character's ordinary speech: in its pitch register, voiced, outside songs, and not all from one scene.
- A clone from its own speech for every speaker fukikae renders, characters and extras alike, with no stock voice on any path.
- No line whose source is the other gender drags a clone toward that gender through `emotion_audio`.
- No ASR hallucination over the credits becomes a character.
- A season studio, served by `fukikae` on localhost, where the operator works through a show's characters: evidence and sample clips across the season, identity decisions written through the registry API, and a voice-tuning loop per character and target language that ends in a curated card.
- Gender as a stored character attribute, operator-set over estimated.
- Opt-in gates in `fukikae run` that stop an episode until its main characters are identity-reviewed (before translate) or voiced with an accepted card (before tts), carried across a season through the registry.

## Non-goals

- Automatic merges or automatic cluster splits. Both are suggestions in the studio; the operator decides (the issue's own call on SPEAKER_02 and SPEAKER_16).
- Room or EQ matching of renders in the mix (the diagnostic found `match_dialogue` is level only; that is a mix issue, not identification).
- Face or active-speaker evidence (#3) and fixes for merged live-action leads (#4). The season index leaves room for another evidence source; it adds none.
- The parts of #25 this first version leaves out: editing style notes and catchphrases, reverting through the UI (the CLI's card history and revert stay), per-character eval numbers, the blind listening packs as a built-in feature beyond the A/B the tuning loop has, and access from anywhere but the host itself.
- Promoting an extras cluster to a character from the studio. An extra gets its own clone (A2); making it a registry character stays `fukikae registry` work for now.
- Fetching cast lists from TMDB or AniDB at run time (CD19, CD24). Cast priors are a later phase, below.

## Anti-goals

- An automatic change to a curated, imported or locked card (CD2, CD3, CD9). Every new gate and re-pick reason re-picks automatic cards only and flags the rest, the way D29 #4 and D32 do.
- A character's voice changing between episodes without an operator action or a rule this spec names (the cards spec's anti-goal). Each re-pick reason gets its own history note.
- A stock voice in a dub. Not as a fallback, not for extras, not for a speaker with two seconds of audio (D41).
- Tuning a threshold on a held-out episode. Fairy Tail E14 stays held out; Sol Bianca S00E01 is the evaluation episode and is reported, never tuned on.

## Terms

"Gender" in this spec means the register of the voice as heard, the thing a reference must match, never the character's gender in the story. Happy in Fairy Tail is a male character with a high voice; for fukikae he is `f` unless the operator says otherwise. The studio labels it "voice: female / male / unknown" for that reason.

"Review" collides with `ClusterAssignment.status = "review"`, which already means "this cluster lost a tie and adds no exemplars". The new feature is "the studio" in prose, `fukikae studio` on the command line and `studio/` on disk; "reviewed" is only the registry timestamp of an identity review. The status value keeps its meaning and is never reused.

"Speaker" means anything fukikae renders a voice for: a registry character, an extras cluster (a diarize cluster the characters stage gave no character, `review` clusters included), or a cluster-less `unknown` line.

## Part A: automatic fixes

### Ranking

| Change | Strict mismatches fixed | At the 180 Hz cut | Cues whose voice changes |
|---|---|---|---|
| A1 reference gates | 3 | 23 | 217 |
| A2 a clone for every speaker | up to 19 (the diagnostic's estimate for gender-matched extras; measured in the evaluation) | 21 | 39 extras cues (char 11's 12 stock cues go with A4) |
| A3 per-line gender gate | 10 (estimate for rerouting; see below) | 11 | 24 flagged |
| A4 song tail and hallucinations | 0 | 5 | 12 removed |

A2 fixes the most strict mismatches; A1 changes the most cues. The plan builds the shared pitch module and the bands first, then A1, A2, A3 and A4, each shippable alone.

### A0: one pitch estimator

Everything in Part A reads pitch, so it gets one module, `fukikae.pitch`, CPU only, numpy and scipy only.

The pipeline estimator today (`cards.select.speaker_f0`) returns the 20th percentile of an autocorrelation F0 between 70 and 400 Hz. Both choices hurt here: the cap sits under the median of three Sol Bianca clusters (SPEAKER_16 406 Hz, SPEAKER_02 377 Hz with a p90 of 622 Hz), so high frames fold to subharmonics, and the 20th percentile reads 172 to 237 Hz for clusters whose pYIN medians are 236 to 406 Hz, which is why the anime profile had to push `female_above` to 300 and then called every woman `u`.

I keep the autocorrelation estimator and fix it instead of adding librosa's pYIN as a dependency: librosa pulls numba and llvmlite into every install for one function, and the estimator's problems are its range and its statistic, not its method. The changes:

- range 60 to 800 Hz (`F0_MIN`, `F0_MAX`), with the existing shortest-lag-within-90% rule against subharmonics;
- a frame track instead of one number: `f0_track(vocals, spans) -> F0Track` holds per-frame F0 (NaN when unvoiced) and the span each frame came from;
- `F0Summary` from a track: `median`, `p10`, `p20`, `p90`, `voiced_s`, `seconds`, `voiced_fraction`;
- classification on the **median** of voiced frames: `m` under `male_below`, `f` over `female_above`, `u` between or under a minimum of voiced speech (1.0 s for a speaker, 0.4 s for a line);
- a confidence: the distance in semitones from the nearest band edge, divided by 4 and capped at 1, and 0 for `u`. The studio shows it; A3 reads it.

`speaker_f0` stays as a thin wrapper (it returns the new median) so `eval tts` keeps working.

Calibration is a plan task, done before any profile value changes: the estimator's class per speaker and per line against pYIN (run through `uv run --with librosa` in a script, never a dependency) and against a hand-written voice-gender truth file on the tuning episodes of three sets that already have workspaces on the GPU host: Fairy Tail E12 and E13 (ja, the anime profile's tuning episodes), The Big Bang Theory (en, live-action tuning episodes) and Kaze (ja). Sol Bianca S00E01 is reported beside them and not tuned on.

The proposed defaults, before calibration, are `male_below = 165` and `female_above = 200` on the median. On Sol Bianca's pYIN medians they separate every cluster that isn't mixed: the women sit at 236 to 406 Hz, the men at 88 to 145 Hz, and the two ambiguous clusters (SPEAKER_13 at 205 Hz, which is the mixed one, and SPEAKER_15 at 193 Hz, the hallucination) fall in or near the unknown band where they belong. Adult speech norms (roughly 85 to 155 Hz for men, 165 to 255 Hz for women) put the same edges in the same place. The calibration's acceptance rule: no main character (30 s or more of speech) on any of the three sets lands in the other class; the unknown band holds at most 10% of main characters' seconds; per-line agreement with pYIN at least 90% where both are known. If one set needs other edges, that set's profile gets them and the results doc says why.

`[tts.gender]` keeps its keys (`male_below`, `female_above`) and they now apply to the median. A workspace that sets them was tuned on the 20th percentile, so `gender_thresholds` logs a warning naming the change once per run when the workspace (not a profile) sets either key. The anime profile drops its `female_above = 300` override.

### A1: reference gates

`cards.select.eligible_references` gains four gates, applied to each candidate after the existing ones (overlap, length, song exclusions, foreign names, sparse text). Each is a filter, not a change to the CD5 ranking: SNR order among the survivors stays as it is.

1. **Pitch register.** The speaker's speech median is computed over the majority-gender turns of its speech in that episode (each turn classified by its own median, the majority by seconds), so a mixed cluster like SPEAKER_13 gets the register of its larger part and not a meaningless average. A candidate whose median is more than `REF_PITCH_ST` (4 semitones) from it is dropped. Char 1's narration (171 to 183 Hz against 240 Hz, about 5.3 semitones) fails; char 10's male clips (127 and 147 Hz against the female part) fail; char 2's 192 Hz line (11 semitones under 377 Hz) fails.
2. **Voicing.** A candidate under `REF_VOICED_MIN` (50%) voiced frames is dropped: char 2's processed voice had none.
3. **Song-like audio.** A candidate with `REF_SONG_SHARE` (half) or more of its seconds song-like by the theme detector's per-second test (music bed within 3 dB of the voice, voice active) and a voiced share at or over the detector's `voiced_share` (0.7) is dropped. This is the theme detector's evidence applied per turn anywhere in the episode, which catches char 3's English insert song, covered by no theme span and no singing tag. A false positive costs one candidate, and the per-turn test can be looser than the theme detector, which decides what the mix leaves untouched.
4. **One scene.** Candidates whose starts are within `SCENE_GAP_S` (60 s) of an already-taken pick count as the same scene; the greedy fill takes at most one clip per scene until it would fall short of the D32 floor, and only then takes a second from a scene already used. Narrator characters (Part B, `kind = "narrator"`) are exempt: narration is often one long scene, and that is fine for a narrator.

A gated candidate is logged at debug with the gate that refused it, and `reference_report` adds one line when a gate removed a candidate that would otherwise have ranked in the top three ("gated: 2 candidates off-register, 1 song-like"), so `voices show` can explain a surprising pick. The studio shows the same verdicts per candidate.

**Re-picking cards the gates now refuse.** A new `impure_cards(ws, reg, store, language)` in `cards.auto`, beside `stale_cards` and `below_floor`, measures every automatic clone card's references on their origin spans (the episode's vocals, `origin = {episode, start, end}`) and lists the cards with a reference that fails gate 1, 2 or 3 (gate 4 is about the set, and the floor already judges sets). `ensure_assignments` treats them exactly as D29 #4 treats stale cards and D32 treats cards under the floor:

- an automatic card (`replace_refusal(a, "auto")` is None) is re-picked when the current rules give picks that pass every gate, a new card with a new `auto` row noted `re-picked: references fail the reference gates (<which>)`;
- with no passing pick, it is kept and flagged (never made stock, as D32 rules for the floor and D41 for everything);
- a curated, imported or locked card is flagged only, naming the failing reference and pointing at the studio and `fukikae voices clone`;
- a card with no usable origin (`voices clone` cards say `{"source": "voices clone"}`) is never judged.

Precedence when a card is stale, below the floor, impure and stock at once: one re-pick, with all the reasons in the note.

### A2: a clone for every speaker, and no stock voice anywhere

D41 replaces the earlier A2 (extras gender, wider stock pools). Every voice fukikae renders is now a clone of the speaker's own speech, made by `[tts] clone_backend` (IndexTTS2 in every profile, D24).

**Reference tiers.** Any speaker's references come from one rule with three tiers, applied to a character over the workspace's episodes (as today) and to an extras cluster over its own episode:

1. **Floor.** The gated eligible turns reach the D32 floor (2 clips, 10 s): the card is made from them, unflagged. Unchanged.
2. **Short.** The gated eligible turns, joined across the speaker's turns by D32's fragment join (gaps up to 0.6 s, no other speaker between), reach `CLONE_MIN_S` (3.0 s) but not the floor: the card is made from them and flagged below the floor, with D32's wording. Today's 6 s minimum (`MIN_CLONE_S`) goes; its only job was choosing between a clone and a stock voice.
3. **Thin.** Under 3.0 s of gated eligible speech: the card is made anyway, from the speaker's longest turns up to IndexTTS2's 15 s prompt, with the A1 gates and the existing length, SNR and sparse-text checks set aside. Two exclusions still hold, because breaking either clones a voice that is not the speaker's: a span inside a theme song span, and a span with `STALE_OVERLAP` or more under speech the subtitle names give someone else. If those leave nothing, the speaker's own turns are used regardless. The card is flagged `thin reference: <seconds> s from <n> turns`, in `run.json`, `voices list`, `names show` and at the top of the studio's queue.

No tier ends in a stock voice. Both flags say the same thing in two severities: listen to this one in the studio. I put the hard minimum at the top of D41's 2 to 3 s range because the tier boundary decides only which flag a card gets and where it sits in the queue, and a 3 s line keeps the one-short-clip clones, the kind A19 found strained (Young Makarov's single 6.3 s shout), out of the milder flag. Open decision 3.

**Extras.** An extras cluster gets an episode-local clone card from its turns in that episode (`assignments.json`), under the same tiers, the A1 register gate reading the cluster's own majority register. It is not a registry assignment, because an extras cluster is an episode's cluster and not a character: the map from extras label to card id lives in the tts stage key, where the stock map lived, and the card goes in the store like any other. Card ids are content hashes, so the same turns give the same card on every run, and a label keeps its voice when a translation changes.

**Cluster-less lines.** A cue with no cluster (the `unknown` label) has no other speech known to be the same speaker's, and joining all of them would put several speakers into one voice. Each such cue is cloned from its own source span (`delivery.reference`): a one-line clone, always in the thin tier, flagged once per episode with the count ("23 unknown lines cloned from themselves"). A3 never triggers on these, because the speaker prompt and the source line are the same audio. Open decision 4.

**What is retired.**

- The under-6 s stock card rule in `cards.auto.ensure_assignments` (the stock branch, `pick_stock` for characters, the stock note, `MIN_CLONE_S`), `extras_voices`, and the extras and character pools in `STOCK_POOLS`.
- The `auto-fallback` stand-in. When the clone backend can't run here (its worker environment isn't set up, a model file is missing, the device is wrong) or no `[tts] clone_backend` is set, the tts stage refuses before any model loads, naming the backend and `fukikae backends setup <backend>` (or the missing setting). I chose refusing over rendering nothing for the affected cues: a missing clone backend affects every speaker of the episode at once, and "nothing, flagged" would produce a dub with no dubbed voice in it that exits 0 and looks finished to `fukikae run` and to the mix. The tts stage already stops for a broken card before any model loads; this is the same failure with the same treatment. `auto-fallback` stays readable in history rows and is never written again.
- `[tts] backend` as the choice of automatic and extras cards. The profiles drop `backend = "kokoro"`; a workspace that sets the key gets a warning once per run that it no longer chooses any card (D41).
- `fukikae voices assign <character> stock:<backend>:<voice>`, removed, with the error naming D41 and pointing at the studio and `fukikae voices clone`; importing a stock card from a pack is refused the same way; `fukikae voices restock` goes, because it re-pinned automatic stock cards and none remain. I removed the explicit operator path instead of keeping it behind a flag: D41 says never, anywhere, and an escape hatch is the one door through which a voice that resembles nobody (cosine 0.06) gets back into a dub. Every case where an operator might want one has a better answer in the studio: a thin speaker gets its clips chosen by ear, and a bad clone gets its references swapped. The `stock` card source and the Kokoro backend stay in the code for `eval tts` baselines (eval is not the pipeline, R1) and for reading old cards. Open decision 5.

**Existing workspaces.** A new re-pick reason, `STOCK_NOTE = "re-picked: stock voices are retired (D41)"`: an automatic card (`replace_refusal(a, "auto")` is None: `auto`, `auto-fallback` or `restock`, unlocked) whose card source is `stock` is re-picked as a clone at the next tts run, through the tiers above. A curated, imported or locked stock card is kept (CD2, CD3, CD9) and flagged on every run: "character X speaks with a stock card (assign, locked); stock voices are retired (D41): accept a clone in the studio or use `fukikae voices clone`". Extras need nothing: their cards are rebuilt from the stage key on the next run.

**Cost.** Extras move from Kokoro on the CPU to IndexTTS2 on GPU 0: on Sol Bianca, 39 more GPU lines at about 3.4 s each (D26's tf32 figure), a little over two minutes an episode.

**Gender after D41.** No stock pick reads gender any more. The stored or estimated gender is read by A1's register gate, A3's line gate, the studio, and the operator-gender re-pick of automatic clone cards (Part B).

### A3: per-line gender gate

The tts stage computes each spoken cue's source class from its `delivery.reference` span (A0, line minimum 0.4 s voiced) and compares it with the speaking character's established gender: the operator's when set, else the class of the character's card references (measured on the card's clips, which is the voice that will actually render), else the A1 majority class of the character's speech.

When the line is strictly the other class (`m` against `f` or the reverse; `u` on either side never triggers), there were three ways to route it: flag it, send it to a better-matching speaker, or render it in its own character's clone without `emotion_audio`. I chose the third, with a flag, and D41 confirmed it:

- **The line renders in its character's card with no `emotion_audio`**, recorded in `plan.json` as `emotion_dropped: "gender"` (the same field CD18 uses for laughter and crying), and `TtsPlanCue` gains `source_gender` and `gender_gate: true`.
- **The line is flagged** in `run.json` and in the studio, with the cue, its source median and the character's established gender.
- **Wrong-gender source audio is never passed as `emotion_audio`**, to any card, on any path: the rule is enforced in `delivery_passed` itself, so it holds for extras and for `eval tts` (CD12, one render path).

Sending a line to a better-matching character needs a line-level speaker embedding the tts stage doesn't have (diarize embeds clean spans per cluster, not per cue), and on pitch alone it would move lines between characters on a measurement that is weakest exactly where these lines live: short interjections, shouts and processed voices (c0288's 「嗯」 reads 648 Hz). Moving lines is the D27 kind of identity decision, and names already win over acoustics; a pitch rule shouldn't quietly outrank both. The other target the issue named, a gender-matched stock voice, is gone with D41. Flagging alone leaves the measured harm in place: IndexTTS2's emotion prompt is what pulls renders to the line's pitch (log-F0 correlation source-to-render 0.64 to 0.86 for most clones; 44 of char 1's 81 renders pulled out of the card's register). Dropping `emotion_audio` removes the active cause while keeping the character's voice, and the flag sends the real problem, a line diarization gave to the wrong speaker, to the studio, where a split or a merge fixes it for every line at once.

The cost, stated plainly: a line that truly belongs to someone of the other gender still renders in the character's voice until the operator fixes the cluster. The diagnostic's "10 / 11 fixed" assumed a gender-matched stock voice; this routing will fix fewer of the counted mismatches. The evaluation reports both numbers (mismatches, and wrong-gender emotion prompts passed, which must be 0).

Narrator characters are exempt from the flag (narration often sits in another register by design) but not from the `emotion_audio` rule.

### A4: song tail and ASR hallucinations

**Ending songs.** The theme detector looks for one best 80 s window in the last 240 s and grows it up to 100 s. Sol Bianca's ending has two song stretches and the detector caught 3658 to 3740 s only, missing 3771 to 3817 s. The detector gains a second pass over the tail region: after a span is accepted, the rest of the tail (outside accepted spans) is searched again with a shorter window (`tail_window_s`, 30 s) under the same score and voicing tests, repeated until nothing passes; spans within `grow_s` of each other merge. `[mix.songs]` gets `tail_window_s = 30` and `tail_repeat = true`; `VERSION` goes to 3, and the mix stage reruns where it was keyed on version 2. The existing measurement set has to hold: Fairy Tail E11 to E14 openings and endings, Kaze's opening on E01 to E03, and no span on The Big Bang Theory (en, de) or Grid (`scripts/song_scores.py` reproduces it).

**Hallucinations.** A new CPU-only stage, `asrclean`, after `asr`, reads `03-asr/transcript.json` and the separation stems and writes `03b-asrclean/transcript.json` plus `03b-asrclean/dropped.json`. Every consumer of the transcript (characters, the cues builder in translate, reference selection through `episode_artifacts`) reads it through one helper, `transcript_path(ep)`, the way identity inputs go through `names.inputs.identity_inputs`. A segment is dropped when both hold:

- its text, NFKC-folded with spaces and punctuation stripped, equals a phrase in `HALLUCINATION_PHRASES` for the transcript's language or is that phrase repeated (ja 「ご視聴ありがとうございました」, 「ご覧いただきありがとうございました」, 「チャンネル登録よろしくお願いします」; en "thank you for watching", "thanks for watching", "subtitles by the amara.org community", "please subscribe"; zh, ko, de and es equivalents, the list in the module with a comment per entry saying where it was seen);
- and its span is music or no speech: inside a theme span (A4's detector), or at least 60% of its seconds song-like by the detector's per-second test, or the vocals stem active on under 30% of its 100 ms frames.

The phrase list alone would delete a character who really says 「ご視聴ありがとうございました」 to camera at the end of an episode; the music-or-silence condition keeps that line. Dropped segments are listed with the reason in `dropped.json` and in the studio.

The separate stage is deliberate: putting the filter inside `asr` would change the asr key and rerun Whisper or Qwen3-ASR on the GPU for every existing episode. `asrclean` costs a second; its output is byte-identical to its input when nothing is dropped. Translate reruns once on existing workspaces because its input path changes, and that is the price of the cleaner seam.

With A4 in place char 11 never forms (its only turns were the hallucination), and char 3's insert-song reference goes through A1's gate 3.

### Part A evaluation

`fukikae eval gender <ws> <episode>` ports the diagnostic's measurements into the eval package (R1: the pipeline never imports eval), on the A0 estimator, CPU only:

- cue gender agreement, source line against render, strict bands and the single 180 Hz cut, per character and for extras;
- extras: the render's class against the source class per cue (extras now render from their own clones);
- reference purity per clone card: references within 4 semitones of the speaker's majority median, voiced at least 50%, outside song-like audio;
- gated lines (A3) and `emotion_audio` passed against a wrong-gender source (must be 0);
- stock-voiced cues (must be 0), and speakers in the short and thin tiers with their seconds;
- cues from dropped hallucinations and cues inside song spans;
- the render-vs-source speaker cosine (WeSpeaker on the CPU, optional with `--similarity`), reported for the cues that were stock-voiced before beside the clone cues.

It writes `docs/results/`-ready JSON and a Markdown table. Before and after on Sol Bianca S00E01 run on a hard-link copy (`cp -al`) of the workspace; the "after" needs the tts stage to re-render the changed cues, which uses GPU 0 under the GPU lock, and that step waits for the operator's go-ahead (open decision 8). Everything before the render (references, tiers, extras cards, gate counts, hallucination drops) is measurable on the CPU.

Targets on Sol Bianca, from the diagnostic's own estimates: strict mismatches from 35 down to 15 or fewer; extras disagreement from 21 of 29 to 3 or fewer; stock-voiced cues from 51 to 0; reference purity 100% on chars 1, 2, 3 and 10 (or the card flagged with no passing pick); char 11 gone; wrong-gender emotion prompts 0. The former stock cues' cosine is reported against the 0.06 before and the clones' 0.29, with no target: it is the first measurement of extras clones.

No regression on Fairy Tail, the D27 names arm (E11 to E14, E14 held out as before): reference purity (the names ledger's metric) and cue accuracy (`eval names`) must not drop on any episode, and no card on E12 or E13 may change except through a gate or re-pick reason this spec names, each such change listed in the results doc.

## Part B: the studio

### Shape

`fukikae studio <ws>` serves a local web app for one workspace, which in practice is a show or a season of one. It is a Python process with three parts:

- a JSON API over stdlib `http.server` (`ThreadingHTTPServer`), bound to 127.0.0.1 only, with a random per-run token printed in the URL and required on every request (another local user, or a web page in the operator's browser, can't drive it);
- one page, shipped as package data, that is the API's first client: Preact and htm vendored into the package as ES modules with their versions and hashes recorded, no build step and no network at any point (CD24's no-fetch rule covers the browser too: no CDN, no fonts, no external URL in the page). Open decision 6;
- an audition worker: the draft card's clone backend through the same `WorkerSynthesizer` the tts stage uses, loaded on the first audition and holding `gpu_lock` for its card (CD15) while loaded, unloaded after `AUDITION_IDLE_S` (300 s) without an audition or when the operator presses "release GPU".

The operator works over ssh on the GPU host, so the server prints the tunnel line beside the URL (`ssh -L 8765:127.0.0.1:8765 <host>`). Binding to the tailnet is later work. Open decision 7.

A static page can't do this job: auditions need the GPU worker, and every accepted voice and identity decision is a registry write. The studio is a server for those two reasons and no other.

### The two constraints it lives with

**The GPU lock.** One fukikae job per card at a time. While a `fukikae run` holds GPU 0, auditions queue, and the page shows who holds the card (the lock file names the holder's pid and job). While the studio's worker holds it, a run waits like it would for any other job; the idle unload and the release button keep that wait short. Auditions run one at a time, in order.

**The one-committer registry.** The registry and the card store are written only by `fukikae.studio.apply`, the same function `fukikae studio apply <file>` calls, each decision in one `Registry.transaction()`. Every decision carries the digest of the registry rows it was made against (the character, its actor, its card assignment for the language); a decision whose rows changed since the page loaded them (a run assigned a card, another studio tab renamed the character) is refused, naming the difference, and the page reloads that character. Nothing in the page writes a file. A `fukikae run` may run beside the studio: the registry's own refusals (`ProtectedAssignmentError` and friends) and the digest check cover the overlap, and the invalidations a decision makes (`job.mark_stale`) are picked up by the next run.

### Data model

Season state lives where it already lives, so later episodes, other languages and other tools see it without the studio:

- **Registry (schema 5).** Identity: names, aliases, merges, splits and locks as today, plus `characters.gender`, `gender_by`, `kind` and `reviewed` (below). Voice: each accepted card is a card assignment with `chosen_by = "studio"` (a new operator source in `OPERATOR_BY`), locked, its note naming the draft version and how many auditions led to it. Card history and revert work as for any assignment.
- **Card store.** Accepted cards, and the cards each audition was rendered with (content-hashed, so a draft that returns to an earlier setting finds its card again).
- **`<ws>/studio/season.json`.** The season index, derived, rebuilt when an episode's stage outputs change (keyed by their digests), never edited: episodes and how far each has run; per character its episodes, seconds and clusters per episode, sample clips spread across the season, estimated and stored gender, kind, cards per target language, flags and merge or split suggestions; per episode its extras clusters with their seconds, tiers and samples. `schema_version = 1`. It is the data model #25's later versions read, served at `GET /api/season`.
- **`<ws>/studio/drafts/<character>.<language>.json`.** One voice-tuning session: versions, each with its reference spans, backend, render parameters and card id, and each version's auditions (line, cue key, wav). Workspace-local and disposable: deleting it loses only the history of what was tried.
- **`<ws>/studio/decisions.jsonl`.** Every applied decision, with its digest, time and result lines. It is an audit log, and `fukikae studio apply` can replay a hand-written or exported file in the same format.
- **`<ws>/studio/splits/<episode>.json`.** Operator pitch splits, read by the names stage (below).
- **`<ws>/studio/audio/`.** Clips cut on demand from the episodes' vocals at 24 kHz mono, owner-only, a cache.

The decisions-as-data idea from the first draft stays: every identity edit and every accept is a `Decision` (discriminated on `op`), validated by one pydantic model, applied by one function. The page posts decisions; a script, a later UI or `fukikae studio apply` can do the same.

### Detecting the season's characters

The pipeline already finds characters across episodes: each episode's characters stage matches its clusters to the registry, and `registry resolve` reconciles. The studio reads that result; it detects nothing on its own. An episode the studio can use has run through `characters`; the episode list shows the others with the command that would bring them in (`fukikae run <ws> <episode> --to characters`, or `--to translate` to have real lines to audition).

Per character, `season.json` holds:

- **samples**: `--clips` (default 6) clips across the season, chosen as the longest clean turns spread over episodes and over each episode's turns, each with its transcript and time, so the page shows the character's range and not six copies of the loudest scene;
- **gender**: class, confidence and median from A0 over the season (and per episode), per-band seconds, and the stored gender with who set it;
- **seconds**: per episode and in total, clean seconds, turns, cues;
- **cards**: per target language the current card (id, source, `chosen_by`, locked, tier and flags);
- **merge suggestions**: the three nearest other characters by registry centroid cosine, shown at or over `MERGE_SUGGEST_COS` (0.5), narrators never paired with dialogue characters (on Sol Bianca only SPEAKER_02 to SPEAKER_16, 0.76, qualifies);
- **split suggestions**: a cluster whose minor gender band holds at least 5 s and 15% of its classified seconds is bimodal (the diagnostic's test: SPEAKER_13 is the only one on Sol Bianca); the page shows both parts' samples;
- **names evidence** (D27): key decisions, named lines over the character's turns, and the disagreement and stale-decision lines `names show` computes today (moved into a shared function, `names.flags`, so both use one implementation);
- **flags**: thin reference and below the floor (A2, D32), impure references (A1), stale (D29 #4), stock card kept (A2), gated lines (A3), dropped hallucinations (A4), names disagreements, bimodal.

The studio's home page is a queue of characters, ordered by flags first (thin, stock kept, impure, bimodal, gated lines, names disagreements, below the floor), then by unreviewed, then by seconds of speech. Extras clusters are listed per episode with their tier and a play button; their voices are automatic and episode-local, and the studio shows them so the operator can hear them, not tune them.

### Identity editing

Per character, the identity panel offers:

| Decision | Applied through | Invalidates |
|---|---|---|
| name | `rename_character` (`name_by = "operator"`, operator alias) | characters stage of every episode naming the character (as `registry rename` does) |
| alias | `add_alias(by="operator")` | same as name |
| gender (f, m, clear) | new `set_character_gender(cid, gender, by="operator")` | tts of episodes where the character speaks |
| merge (keep, drop) | `merge_actors` on the two characters' actors, then `_remap_assignments` | characters of both actors' episodes |
| split by pitch | operator split file read by the names stage (below) | names, characters, and everything after, for that episode |
| split cluster out (undo a match) | `split_cluster(actor_id, episode, cluster)` | characters of that episode |
| narrator | new `set_character_kind(cid, "narrator")` | characters of episodes where it speaks |
| not a character | new `set_character_kind(cid, "ignored")` | characters of episodes where it speaks |
| lock | `lock_character(cid, True)` | nothing downstream |
| reviewed, no change | new `mark_reviewed(cid)` | nothing |

"Invalidates" means `job.mark_stale(ep, stage)` on the named stage; the file-driven cascade (assignments.json is a translate input; cards and cues are tts inputs) reruns the rest. Every applied identity decision also marks its characters reviewed. The page shows, before the click, which episodes and stages a decision invalidates ("reruns translate on E02, E05 and E07"). A decision the registry refuses (merging into a locked actor, naming a character whose name an operator already set) fails with the registry's message and changes nothing.

### The voice-tuning loop

Per character and target language, the voice panel is a loop the operator stays in until the voice is right:

1. **Start.** A draft opens from the character's current card for the language (its references and parameters as version 1). With no card for this language and an accepted card for another, it starts from that card's references and parameters, the way `voices derive` does. With neither, it starts from what the automatic rules would pick, computed without assigning.
2. **References.** The candidate list is the character's eligible turns across the season, ranked as CD5 ranks them, each with its episode, time, transcript, seconds, SNR, median and class, and its verdict from every check (overlap, length, song, foreign names, sparse text, A1's four gates). The operator plays any of them and adds, removes or reorders clips; a meter shows the selection against the D32 floor (2 clips, 10 s) and IndexTTS2's 15 s prompt. Gated turns are hidden behind a "show gated" toggle and can be chosen there; a chosen gated turn is named in the card's note as an operator override, so the gates stay visible in the record.
3. **Parameters.** The controls are the draft backend's own voice keys, read from the backend class so a new key appears without a UI change: for IndexTTS2 `emotion_strength` (its `emo_alpha`), `seed`, `temperature` and `top_p`; for CosyVoice3 `seed`, `rate` and its mode; for Qwen3-TTS `seed`, `temperature`, `top_p` and its mode. The backend itself is a choice among the installed clone backends (CD20's capabilities apply as declared). IndexTTS2 has no rate or duration control, so the page shows no rate slider for it; the dub's timing comes from the tts stage's rate fit and time stretch, which the audition applies the same way.
4. **Audition.** Two kinds of test line: typed text in the target language (rendered without `emotion_audio`, or with a source line the operator picks as its emotion prompt), and real lines: the character's cues from `06-translate/cues.<language>.json` in any episode that has run translate, each played beside its source span. Real lines render through the tts stage's own path (`prepare_cue`, `delivery_passed` with A3's gate, `render_guarded`, the pause cap, CD12), so an audition sounds the way the dub will, and are cached in the workspace's tts cache under the same `cue_key`: when the operator accepts, the next tts run finds those lines already rendered.
5. **Compare.** Every change of references, parameters or backend that is auditioned becomes a new version. The page plays any two versions on the same line side by side, or blind (the `eval tts-ab` pack's A/B with a hidden key, revealed on demand).
6. **Accept.** Writes the version's card to the store and assigns it: `chosen_by = "studio"`, locked, the note naming the draft version, the reference spans, any gated override and the audition count. It is a curated card from then on (CD2): no automatic rule replaces it, and every later episode where the character speaks uses it.

The loop is per language, and that is the only per-language part of the studio. Identity decisions are language-free.

### Split by pitch

`registry split` moves a whole episode cluster to a new actor; it cannot cut one cluster in two. The names stage already can: it relabels turns into `<cluster>~<slug>` sub-clusters and moves the `clusters.npz` rows without re-embedding. A pitch split reuses that machinery. Applying a split decision writes `studio/splits/<episode>.json` (`{cluster: {"by": "pitch", "male_below": Hz, "female_above": Hz}}`); the names stage reads it as an operator input after subtitle names, classifies each turn of the cluster by its own median, and labels the parts `<cluster>~voice-f` and `<cluster>~voice-m` (the `voice-` prefix can't collide with a name slug, which comes from a subtitle name). `u` turns stay with the larger part. The split file is part of the names stage's key, and the episode reruns from names on.

### Narration and "not a character"

The registry gains a character `kind`: `character` (default), `narrator` or `ignored`.

- A **narrator** is a character with its own card. A1's one-scene gate and A3's flag skip it, and merge suggestions never pair it with a dialogue character. The Sol Bianca case is a pitch split of char 1 followed by "narrator" on the low part: the narration gets a clone from narration, the scene speech a clone from scene speech.
- An **ignored** character's clusters get the new `ClusterAssignment.status = "ignored"` from the characters stage; the cues builder marks their cues `skip = "ignored"`, so they are never translated or rendered and the mix keeps the original audio under them. It is for hallucinations A4 misses, crowd walla, a TV in the background. It is the one way a speaker ends up with no clone, and only by the operator's decision.

### Gender as a stored attribute

`characters.gender` (`f`, `m` or NULL) and `characters.gender_by` (`operator`, later `cast`, or NULL) in registry schema 5. The estimate is never stored in the registry: it changes with every episode's speech and is cheap to recompute, so it lives in a pitch cache under `.fukikae/pitch-cache/` keyed by the vocals' fingerprint and the spans' digest. Who reads the stored gender: A1's register gate (the operator's class picks which mode is the character's when a cluster is mixed), A3's established gender, and the studio.

An operator gender that disagrees with an automatic clone card is a re-pick reason, beside stale, below the floor, impure and stock: an automatic clone card whose references are the other class is re-picked under the gates with the operator's class as the register (`re-picked: operator gender`). Curated, imported and locked cards are flagged, as everywhere else.

### Gates in `fukikae run`

`[review]` is a settings table (exempt from the R3 `backend` rule like `[resolve]` and `[names]`), read from profile and workspace:

```toml
[review]
identity = false     # true: stop before translate until the episode's main characters are identity-reviewed
voices = false       # true: stop before tts until each main character has a curated card in the target language
min_seconds = 20.0   # a character with less speech in the episode never holds it up
```

A main character here is one of kind `character` or `narrator` (never `ignored`, never an extra) with at least `min_seconds` of speech in the episode.

- **`identity`** checks before `translate`: the episode stops when a main character has never been reviewed, or has a flag raised since its last review (a new bimodal split suggestion, impure references, a names disagreement). Identity decisions change `assignments.json`, a translate input, and stopping before translate saves the GPU LLM run that a later merge or split would throw away.
- **`voices`** checks before `tts`: the episode stops when a main character has no curated, imported or locked card in the target language. Auditioning real lines needs translated cues, which is why this gate sits after translate.

A stopped episode exits with code 3 and a line naming the command (`fukikae studio <ws>`, and the characters to look at); in a multi-episode run the other episodes go on. Both gates are season-wide by construction: "reviewed" is a timestamp on the character and an accepted card is a registry assignment, so episode 2 stops only for characters that are new, newly flagged or still unvoiced. The first draft's single gate before translate (its open decision 4) is replaced by these two. Open decision 2.

### Across a season and across languages

The registry is per workspace, and both kinds of decision go into it, so they carry: a name, gender, kind, lock or accepted card set while working on episode 1 holds when episode 2's clusters match that character. The intended order for a new season: `fukikae run <ws> --to characters` over every episode, identity in the studio, `--to translate`, voices in the studio, then the rest. With the gates on, `fukikae run <ws>` on the whole season stops each episode at the right point and names what is missing.

A second target language reuses everything identity-related and starts each voice draft from the first language's accepted card (the loop's step 1), so the operator re-auditions instead of re-choosing.

### Fit with #25 and #26

The studio is the first version of #25. From #25's seeds it covers testing (type a line in a character's voice through the configured backend, A/B two versions, play rendered lines beside the original), editing (reference clips from the eligible turns with the floor and gate rules visible; rename, alias, merge, split, lock), and review flows (the flags as a queue). What it leaves for later versions is in the non-goals. The API and `season.json` are versioned (`schema_version = 1`), and the page is one client of them; a later UI replaces the page, not the model.

For sharing (packs, and later #26's per-series state), decisions carry each character's key (folded name) and external id beside the workspace-local registry id, and each applied change is a registry row with an operator `by`. A later pack export can carry names, aliases, genders, kinds, locks and the accepted cards keyed the way `cast.toml` already keys cards.

### Cast priors (later phase)

A hand-written `<ws>/cast.toml` (`[[cast]] name, gender, external_id, aliases`) is cheap and is the optional last task: the studio offers its names in the name field, shows the cast count against the character count ("cast lists 9 speaking roles; this season has 14 characters"), and offers each cast gender as a one-click gender decision. Nothing from the cast list is applied without the operator's click, and an applied cast gender is `gender_by = "operator"`, not `cast`. Fetching the list from TMDB or AniDB (`fukikae cast fetch`, an explicit network command like `backends setup`, never at run time) and using cast genders as priors without a click (`gender_by = "cast"`) are later work.

## Calls I made (for the operator to check)

1. Fix the autocorrelation estimator (range to 800 Hz, median) instead of adding pYIN; pYIN is the calibration reference only.
2. Default bands 165 and 200 Hz on the median, subject to calibration; the anime override goes.
3. A1's gates filter, never re-rank, and CD5 stands.
4. A3 renders the line in its own clone without `emotion_audio` and flags it (D41 confirmed).
5. A4's filter is a separate `asrclean` stage, and no ASR reruns.
6. Pitch splits ride on the names stage's sub-cluster relabelling, with `~voice-f` and `~voice-m` labels.
7. Gender is voice register, stored only when the operator sets it.
8. Thin speakers are cloned with the A1 gates set aside but with song spans and other characters' named speech still excluded.
9. Extras cards are episode-local and live in the tts stage key, not the registry.
10. A missing clone backend stops the tts stage before any model loads.
11. Accepted studio cards are `chosen_by = "studio"` and locked, as `voices assign` locks by default.
12. Auditions of real lines share the tts stage's render path and cache, and an accepted voice's lines are already rendered when the next run starts.
13. The registry and card store are written only by `fukikae.studio.apply`, one transaction per decision, guarded by a row digest.

## Open decisions (for the operator)

D41 settled the first draft's decisions 3 (A3 routing: (a), confirmed), 4 (the single gate before translate: replaced by decision 2 below) and 5 (extras stock pools: no stock voices at all).

1. **Estimator:** fix the in-house autocorrelation estimator (range 60 to 800 Hz, median), or add librosa and use pYIN in the pipeline? Recommended: fix the estimator; pYIN stays a calibration reference run with `uv run --with librosa`.
2. **Run gates:** two opt-in gates, `[review] identity` before translate and `[review] voices` before tts, both off in every profile (a); one gate before tts only (b); no gates, the studio on demand only (c)? Recommended: (a); identity decisions rerun translate and want the earlier stop, voice tuning needs translated lines and wants the later one, and off by default keeps unattended runs as they are.
3. **Thin-audio minimum:** `CLONE_MIN_S = 3.0` s of gated speech between the short and thin tiers (a), or 2.0 s (b)? Either way nothing is stock and both tiers are flagged. Recommended: (a), the top of D41's range, so one-short-clip clones carry the stronger flag.
4. **Cluster-less lines:** clone each `unknown` cue from its own source span (a), or one clone per episode from all `unknown` lines joined (b)? Recommended: (a); (b) blends unrelated speakers into one voice.
5. **Stock through the CLI:** remove `voices assign stock:...`, refuse stock cards from packs and retire `voices restock` (a), or keep `voices assign stock:...` behind an explicit `--stock` flag as an operator-only choice (b)? Recommended: (a), per D41's "never, anywhere"; existing curated stock cards are kept and flagged either way.
6. **Studio front end:** Preact and htm vendored as ES modules in the package, no build step (a), a Preact and Vite build with a node toolchain in the repo (b), or plain JavaScript (c)? Recommended: (a); it keeps the Preact default without adding node to a Python package, and the page stays one client of the API that a built UI can replace later.
7. **Studio access:** loopback only, reached over `ssh -L` (a), or also bind to the tailnet address (b)? Recommended: (a) for this version; the per-run token already guards the API, and (b) needs an access decision of its own.
8. **GPU for the after-measurement:** may the Sol Bianca and Fairy Tail re-renders for the evaluation run on GPU 0 under the GPU lock (a few hundred IndexTTS2 lines, extras included now)? Yes or no. Recommended: yes, scheduled when no other GPU job is queued.
9. **Hallucination drop rule:** phrase match and music-or-silence (a), or phrase match alone (b)? Recommended: (a).
10. **Cast priors:** ship the hand-written `cast.toml` suggestions in this plan (a) or leave all of cast priors to a later issue (b)? Recommended: (a); it is one small task and gives the studio its name and gender suggestions.
11. **Bands:** `male_below = 165`, `female_above = 200` on the median for every profile (anime override removed), subject to the calibration task's acceptance rule? Yes or no. Recommended: yes.

## Left to the builder

- The studio page's layout and keyboard shortcuts.
- The exact sample-clip spread and cut lengths, and trimming a reference clip's edges in the picker.
- The `HALLUCINATION_PHRASES` entries beyond the ones named, each with a source comment.
