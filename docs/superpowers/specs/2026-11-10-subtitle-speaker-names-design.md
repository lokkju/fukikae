# fukikae: subtitle speaker names for identity

_Status: draft for operator review, 2026-11-10. Implements operator ruling D27: when the subtitles name speakers, use those names to assign lines to characters and split merged clusters before cloning; `registry split` plus `voices clone` stay the manual fallback; face and active-speaker detection stay a later lever (`docs/ideas.md`). Builds on `2026-11-05-speaker-identity-design.md` (the registry, matchers, `registry resolve`) and `2026-11-08-character-cards-and-dubbing-design.md` (cards, reference selection, the subtitle path); where this document is silent, those stand. Rulings R1 to R20, GT1 to GT15 and CD1 to CD26 stand._

## Why

Fairy Tail 100 Years Quest E12's character 2 is diarize cluster SPEAKER_12, which holds Charle, Wendy and Happy (all voiced by women) and two Natsu spans; its clone card was built from Wendy and Charle, so Natsu's "I'd even settle for Jellal!" came out in a girl's voice (`docs/results/2026-11-09-character-cards.md`). The merge happens upstream of cloning: pyannote puts several similar voices in one cluster, and no embedder or matcher tested separates young and female anime voices (WeSpeaker with the joint matcher and episode cannot-link is the best, 0.56 on ja; `docs/results/2026-11-07-identity-profiles.md`). Reference selection can't fix it, and `registry split` works on whole diarize clusters, so it can't either.

Crunchyroll's ASS track names the speaker on every Dialogue line. I measured what that gives on the t12 workspace (`~/fukikae-scratch/t12/fairytail`, anime profile, WeSpeaker, E11 to E14) against the release's own ASS files:

| Episode | Diarize turns | Turns one name covers at 60% | Turns holding 2+ names | Named lines | Lines with no turn under them | Clusters | Clusters holding 2+ names (3 s each) |
|---|---|---|---|---|---|---|---|
| E11 | 667 | 497 (75%) | 4 | 341 | 6 | 12 | 6 |
| E12 | 697 | 550 (79%) | 5 | 354 | 6 | 14 | 8 |
| E13 | 635 | 418 (66%) | 6 | 251 | 4 | 9 | 6 |
| E14 | 572 | 426 (74%) | 14 | 323 | 7 | 10 | 6 |

Two thirds of the clusters in every episode hold two or more named voices (E14's SPEAKER_08: Wendy 129 s, Happy 36 s, Charle 32 s, Juvia 6 s), while a turn almost never does. So the names can relabel speech turn by turn, and only a handful of turns per episode need cutting.

## What the names look like

Read-only survey of the four Crunchyroll English ASS files (E11 to E14, 254 to 357 Dialogue events each):

- **Every event has a Name.** No blank names. Upper case, given name only, consistent across episodes: `NATSU` (163 lines over four episodes), `LUCY`, `WENDY`; never "Natsu Dragneel".
- **Signs** are Name `SIGN` (title cards, "Next Time", the next episode's title), three per episode, with styles `White-Standard`, `Next Time`, `Next Ep Title`.
- **Inner voice** is a `/I` suffix: `WRAITH/I`, `ERZA/I`, `WENDY/I` (17 lines in E14). Same speaker, often processed audio.
- **Possession** is written three ways for one state: `MAKAROV (WRAITH)`, `MAKAROV(WRAITH)`, `MAKAROV/WRAITH`; also `HAPPY (WRAITH)` and `HAPPY/WRAITH`. The voice is a story question, not a spelling one.
- **Two speakers** appear as `NATSU/ZEREF` and `JUVIA/TOUKA`, and once as two events with identical timing and different names (`CHARLE` and `TOUKA`, E14 17:28). `/` therefore means either two speakers or possession.
- **Role labels**: `TOWNSPEOPLE A` to `D`, `WIZARD A`, `WIZARD B`, `BANDIT A`. They name a slot in one scene, not a person across the season.
- **Age variants** are separate names: `YOUNG MAKAROV` (16 lines) beside `MAKAROV` (2) and the possession forms. Different voice, different character.
- **A typo**: `NABARU` (3 lines, E12 only) beside `NEBARU` (51 lines over E11 to E14). The ground-truth workspace already aliases it in `labels-aliases.yaml`.
- No dialogue dashes (`- line\N- line`) and no two different names overlapping in time, apart from the one simultaneous pair above.

Licensor subtitles (Link Click) and Bilibili releases carry no Name values, and SRT, WebVTT and TTML have no Name field at all; those releases keep today's acoustic path.

## Goals

1. When an episode has a subtitle track whose names are usable, every diarize turn that one name covers is attributed to that name, turns holding two names are cut between them, and each diarize cluster is split into one sub-cluster per name plus an unnamed remainder, before the characters stage runs.
2. Names map onto registry characters by a stable key, with aliases across spellings, episodes and languages. A name with no character gets one (or renames an unnamed one that is plainly the same person), and the registry's operator protections hold: a locked actor keeps its speech, an operator's name is never replaced, and curated cards are never re-picked.
3. Clone references come only from name-confirmed turns whenever the character has any, and an automatic card whose references the names now contradict is re-picked.
4. On the subtitle path, a cue whose own Name resolves to a character is spoken by that character, whatever the diarization says.
5. Whether a track's names are trustworthy is decided per episode, reported, and overridable.
6. The gain is measured against the identity harness with a held-out protocol that never grades on the labels the run used, and recorded in the results ledger.

## Non-goals

- Face or active-speaker detection (`docs/ideas.md`, the next lever).
- SDH speaker prefixes (`LUCY:`, `（アリス）`) and colour-coded captions as name sources. The readers exist in `eval/groundtruth`; making them pipeline sources is a later plan (open decision 8).
- Matching names across languages automatically (a Japanese track's `ナツ` to an English track's `NATSU`). The operator aliases them; a co-timing heuristic can come later.
- Re-embedding cut turn pieces (open decision 6).
- Using names to fix diarization boundaries beyond cutting a turn between two named lines.

## Anti-goals

- A name overriding a locked actor, an operator's name, or an operator's alias.
- An automatic card re-pick replacing a card someone chose, locked or imported (CD2, CD3, CD9).
- A track whose Name column holds something other than speakers (one value on every line, style names, the typesetter) silently driving identity.
- Thresholds tuned on the held-out episode.
- Pipeline code importing `fukikae.eval` (R1): the name canonicalisation and alignment check move into the pipeline and eval imports them.

## Source of truth

From strongest to weakest:

1. **Operator state.** A locked actor (`registry lock`) or locked character; a name the operator set (`registry rename`); an alias the operator set (`registry alias`); `[names] use = "off"`.
2. **Subtitle names**, from a track the episode's detection accepted, on the turns they cover.
3. **Acoustics**: pins to the previous run, the stage matcher (centroid or lda) and `registry resolve`.

When names and acoustics disagree, names win for the speech they cover, and the disagreement is reported, never acted on: `fukikae names show` lists each named sub-cluster's cosine to its character's centroid built from other episodes, lowest first, so a mis-named line or a typo shows up as an outlier. Acoustics decide only what names don't cover: the unnamed remainder of each cluster, episodes without usable names, and `registry resolve`'s grouping of unnamed clusters.

When names and operator state disagree, operator state wins and the conflict is reported: a sub-cluster whose parent cluster is pinned or matched to a locked actor stays with that actor's character, and `names show` says how many seconds of other names it kept and how to release them (`registry lock --unlock`, then rerun from `names`).

## Names, keys and aliases

A Name value becomes a **key** in the pipeline module `fukikae.names.normalize` (moved out of `eval/groundtruth/base.py`; eval re-exports it):

1. Strip a trailing inner-voice or off-screen marker (`/I`, `/OS`, `/OC`, `/VO`, `(V.O.)`, `(O.S.)`) and remember it (`inner`).
2. Collapse whitespace and normalise spacing around parentheses, so `MAKAROV(WRAITH)` and `MAKAROV (WRAITH)` are one name.
3. Fold: NFKC, then casefold (the registry's existing `_fold`). `NATSU`, `Natsu` and `ＮＡＴＳＵ` share the key `natsu`.

Then each line is classified:

- **sign or song**: a Name in `subtitles.SIGN_NAMES` or `SONG_NAMES`, or a line `filter_cue` filters. Not dialogue; ignored. (`ed` is a speaker here, as in `subtitles.py`; eval's `IGNORED_NAMES` lists it, and the move keeps eval's list for eval only.)
- **multi**: the folded name contains `/`, `&`, `,` or the word `and` after step 1, or is a group word (`all`, `everyone`, `both`, `crowd`, `together`), or two events with different names cover the same span. Two speakers or possession; it names nobody, and turns it covers are unnamed and never references.
- **role**: the name ends in a single letter or a number of up to two digits (`TOWNSPEOPLE A`, `WIZARD B`, `GUARD 2`). It names a speaker within this episode only.
- **named**: anything else. Possession in parentheses (`makarov (wraith)`) is a name of its own; the operator aliases it to whichever character's voice it uses.

The **display** form is the name as written, title-cased per word when the track writes it all upper case (`YOUNG MAKAROV` becomes "Young Makarov").

**Registry keys.** A character answers to the fold of its own name and to its aliases. A new table holds the aliases (schema 4):

```
character_aliases(alias TEXT PRIMARY KEY,           -- the folded key
                  character_id INTEGER NOT NULL REFERENCES characters(id),
                  by TEXT NOT NULL,                  -- 'subs' | 'operator'
                  language TEXT,                     -- the track's language, informational
                  created TEXT NOT NULL)
```

and `characters` gains `name_by TEXT` (`'operator'` or `'subs'`; the migration marks every existing name `'operator'`, since `registry rename` was the only way to set one). An alias key maps to one character. A key resolves by rank: an operator alias, then an operator's name, then a `subs` alias, then a name subtitles gave; two names of the deciding rank make the key ambiguous. `registry rename` sets `name_by = 'operator'` and adds the folded new name as an operator alias, moving a `subs` alias of that key from another character (which keeps its name and exemplars); it is skipped, with a warning, when another character holds the key by an operator alias or an operator name. `registry merge` moves the dropped character's aliases to the kept one; pruning deletes a pruned character's aliases. `registry alias <ws> <character> <NAME> [--remove]` sets or removes an operator alias; it replaces a `subs` alias of the same key and refuses another operator's without `--replace`.

Names are never matched fuzzily. A rare key one edit away from a common one (`nabaru`, 3 lines, against `nebaru`, 51) is listed by `names show` and `names scan` as a suggestion with the `registry alias` command that applies it.

Across languages: the names come from target-language subtitles, so a series dubbed into English and Spanish from English and Spanish tracks may see `NATSU` and `Natsu` (one key) or genuinely different spellings (two keys). A second spelling meets no character the first time it appears; with an unnamed or new character in hand, the operator aliases it once (`registry alias`) and every later episode resolves. External ids (`registry rename --external-id`) are untouched; packs keep matching by external id, then name (CD spec), and an alias is not a pack key.

## Mapping names onto characters

The characters stage, for an episode whose names were used, takes each key in turn (most named speech first):

1. **Known key.** A character answers to it (a key two characters answer to at the same rank is reported and its sub-clusters take the remainder's acoustic path, named nobody): every sub-cluster of that key goes to that character (status `matched`, `source = "name"`, score the cosine when the character has a centroid). One character may take several clusters of an episode this way; the "one cluster per character" rule (`review`) applies only to acoustic matches.
2. **Role label**, or a key with under `[names] min_new_seconds` (5 s, Ruling R6's recurring bar) of named speech in the episode and no character: its sub-clusters are `extra` with `name` set. The extras voice pool speaks them and nothing enters the registry.
3. **Rename an unnamed character.** The key's sub-clusters come mostly from one parent cluster (at least 60% of the key's named seconds), that parent is pinned or acoustically matched to a character with no name and nothing curated (no locked or operator card, no import, no external id, actor unlocked), the key holds at least 60% of that parent's named seconds, and no other key took that character this run: the character takes the display name (`name_by = 'subs'`) and a `subs` alias, and keeps its id, card history and earlier exemplars. Episodes run in order, so the first episode that names a key decides it: on Fairy Tail (names-t6), E11 named WENDY on 10.9 s, too little to hold 60% of any unnamed character, so E11 created a new Wendy (character 22), and from E12 on `wendy` was a known key that took every Wendy sub-cluster, including those of E12 to E14's merged clusters on character 2. Character 2 therefore keeps no name and holds what acoustics give it; folding it into Wendy is the operator's call (`registry merge`), which `names show` suggests.
4. **Create.** Otherwise a new actor and character, named by the display form with a `subs` alias, centroid from the sub-clusters' embedding rows (left empty when the cut left none; acoustic matching skips an empty centroid until a later episode adds rows).

A character named by subtitles is never renamed by subtitles again: a new spelling is a new key, never a rename. An operator rename later takes over the name and keeps the old key as an alias, so later episodes still resolve.

**Locked parents.** Before step 1, a sub-cluster whose parent cluster pins or acoustically matches a locked actor (or a locked character) goes to that actor's character with `source = "lock"`, whatever its key says, and is reported. A named sub-cluster's acoustic evidence, here and for the 60/60 rename and the disagreement report, is its parent's: the pin of the parent's remainder label, else its own pin when that came from an acoustic assignment, else the parent cluster's match. Its own earlier name or lock decision is never evidence, so a lock set after a names run takes effect on the rerun, and a rerun reports the same disagreements as the first run.

**The unnamed remainder** of each cluster keeps the parent's label and goes through today's flow: pinned when the previous run pinned it, else matched by the stage matcher. A character claimed by names does not block a remainder's acoustic match to it. A remainder left with no embedding rows of its own and no pin goes by its parent's named sub-clusters, in the stage and in `registry resolve` (Task 11 controller ruling, amending R1): it takes the character holding at least 60% of the parent's named seconds (`source = "voice"`, reported as following its named siblings), is `extra` when the named speech splits with no character at 60%, and takes the parent cluster's acoustic match only when the parent has no named sub-cluster (a role label is not one).

Each episode's exemplars are rebuilt from scratch as today, from the matched, new and pinned (sub-)clusters' rows, so a rerun with names moves that episode's evidence to the right characters. Episodes processed before names were on keep their mixed exemplars until they are rerun (see Existing workspaces).

`registry resolve` (the anime profile's joint batch, R12) reads the same refined clusters. Named sub-clusters don't take part in the grouping: their rows seed the anchor of the character their key maps to, keys are must-link across episodes and cannot-link to each other (anchors never share a group, R13), and only unnamed clusters are grouped against them.

## Line-to-turn alignment

The `names` stage, after `diarize`:

1. **Lines.** The chosen track's cues through `merge_layers`, then classified as above. A line with no spoken text is dropped. `[names] holdout_episodes` and `hide_fraction` (Evaluation) hide lines here.
2. **Offset.** The alignment check moved from `eval/groundtruth/base.py` (`alignment`, `check_alignment`): share of named-line time over diarize speech, best global offset within plus or minus 10 s, applied when it gains 0.1 and reaches 0.6. A track below 0.6 at its best offset fails detection.
3. **Coverage per turn.** For each diarize turn, the seconds of each key's lines (named and role; inner lines count for their key) overlapping it. A turn with one key covering at least `[names] turn_share` (0.6, the auto-label rule's share) of its length takes that key.
4. **Cuts.** A turn where two or more keys each cover at least 0.5 s and together reach `turn_share` is cut between consecutive lines of different keys: at the midpoint of the gap between them, or at the boundary where they touch. Each piece takes its key; a piece under 0.3 s joins its neighbour. Multi lines are never a side of a cut; they leave their stretch unnamed.
5. **Overlapping turns.** When two diarize turns overlap and one line covers both, the line names only the turn it covers more. Overlapping turns are never clean spans, so this affects attribution, not evidence.
6. **Unvoiced lines.** A named line with under 0.2 s of turn under it is listed as unvoiced (4 to 7 per Fairy Tail episode). It creates nothing; on the subtitle path its cue still gets its speaker from its own Name (Goal 4).
7. **Sub-clusters.** Each (cluster, key) pair becomes a label `<cluster>~<slug>` (an ASCII slug of the key, or `n` plus 8 hex digits of its sha256 when the key isn't ASCII); unnamed turns keep the cluster's label. The refined turns go to `04c-names/turns.json`.
8. **Evidence.** `clusters.npz` rows are clean spans of whole turns. A row whose span lies at least 90% inside one piece moves to that piece's label; a row across a cut is dropped (reported as seconds). `clean_seconds` per label is recomputed with `stages.diarize.clean_spans` over the refined turns. Nothing is embedded, so the stage is CPU-only and takes seconds. The refined file goes to `04c-names/clusters.npz` in the diarize stage's format.

Inner-voice and multi lines' spans are recorded so reference selection can exclude them.

## Detecting usable names

**Which track.** In order: `[names] file` (`{episode}` replaced, `~` expanded, relative to the source's folder, as `[subs] file`), `[names] track` (a stream index), then the dialogue track CD25 picks for the target language (the same `media.find_target_subtitles` call with the same `[subs]` settings, so a release's forced-flagged full track, Fairy Tail E11's, is read exactly as translate reads it), then the dialogue tracks of other languages in probe order (`dialogue_track` per language), taking the first that passes. Only ASS carries names; other formats are verdict `none` at once.

**The test**, on dialogue lines (after `filter_cue` and `merge_layers`, ruby and empty lines dropped):

- named share: lines with a named or role key over all dialogue lines, at least `[names] min_coverage` (0.5);
- at least two distinct keys, and the most common key on at most 80% of named lines (a Name column filled with one value is not speakers);
- named keys equal to a style name of the same file on under half the named lines (some typesetters put the style in Name);
- the alignment check above passes.

The verdict is `full` (named share at least 0.9), `partial` (at least `min_coverage`), or `none` with the failing test named. `use = "auto"` (the default) uses `full` and `partial`; `use = "on"` uses any track with at least two keys and skips the share and dominance tests (alignment still applies unless `[names] align = false`); `use = "off"` reads nothing. A partial track works the same way as a full one: only named lines act, and an unnamed line is evidence of nothing.

`fukikae names scan <ws> [episodes]` runs only this test, read-only, and prints per episode the track chosen, the share, the distinct keys, the verdict and the suggestions.

## Reference picking

`cards.auto.character_candidates` changes in two ways when names were used:

- In an episode whose names were used, a character's candidate turns are only its sub-clusters with `source = "name"` (or `lock`), never its unnamed remainders, and never a span overlapping an inner-voice or multi line (added to `exclude`).
- `character_references` picks among names-used episodes whenever any of them yields picks; acoustic-only episodes are used only when none does, and the card report then says "references are not name-confirmed".

**Stale automatic cards.** A clone card's references carry `origin = {episode, start, end}`. When a character's current card was chosen by `auto`, `auto-fallback` or `restock` and any reference origin lies in a names-used episode outside that character's name-confirmed turns (90% containment), `ensure_assignments` re-picks it (a new `by = "auto"` row with note "re-picked: references not name-confirmed"). Curated and imported cards are never re-picked; `voices list` and `names show` flag them ("character 7's card (assign) was cloned from turns now named CHARLE; `voices clone` replaces it").

## The subtitle path

`assign_subtitle_speakers` takes a key-to-speaker map from the episode's assignments (each `ClusterAssignment.name` to its character, or to its extra label for `extra`). A cue whose own Name gives a key in the map, and which the names stage did not hide, is spoken by that character; any other cue goes by turn overlap as today. When names come from a different track than translate reads (another language's track), cues carry no usable Name and go by turn overlap, which the refined turns already improve. Translate reads `04c-names/names.json` for the name map, the hidden lines and the track offset, and puts them in its config key (`cue_names`) rather than taking the file as an input, so a names stage that read no names (off, none, held out) never reruns translate (final review B, reversing the earlier input ruling).

## Configuration and CLI

```toml
[names]                        # a settings table (no backend), exempt from R3 like [resolve]
use = "auto"                   # auto | on | off
# file = "names/{episode}.ass" # an operator-supplied names file (the manual route below)
# track = 3
min_coverage = 0.5
turn_share = 0.6
min_new_seconds = 5.0
align = true
# evaluation only:
# holdout_episodes = ["S01E14"]
# hide_fraction = 0.2
# hide_seed = 0
```

Profiles don't set `[names]` in this plan; the code defaults apply to every genre.

- `fukikae names scan <ws> [episodes]`: the detection test, read-only.
- `fukikae names show <ws> <episode>`: verdict, track, offset, coverage, per-key seconds and sub-clusters, cuts, unvoiced lines, rows dropped, the character each key went to and how (known, renamed, created, extra, kept by lock), voice disagreements lowest first, typo suggestions, and stale or flagged cards.
- `fukikae registry alias <ws> <character> <NAME> [--remove] [--replace]`; `registry show` lists a character's aliases and `name_by`.
- `fukikae run <ws> <episodes...> --from names` reruns from the stage.

## The manual fallback

For releases without names (licensor, Bilibili, SRT), as D27 rules, and for any name the operator disputes:

1. `registry show` and `names show` (or the voices report) point at a merged character.
2. `registry split <ws> --episode E --cluster C` moves a whole diarize cluster (or, on a names-used episode, a sub-cluster `C~slug`) to a new character; `registry rename` names it; `registry lock` keeps it there against later names.
3. `voices clone <ws> <character> --clip <file>` gives a character a hand-picked voice.

`registry split` cannot separate two voices inside one diarize cluster, which is exactly the anime failure. The second manual route covers that: `[names] file` points at an ASS file the operator writes or borrows (another release's Crunchyroll track, or a few named lines typed over the episode's own subtitles), and the episode goes through the same names path with the same protections. A partial file is fine; only named lines act.

## Existing workspaces

- **Stages.** `names` is a new stage between `diarize` and `characters`. The characters stage's inputs become `04c-names/names.json` plus either `04c-names/turns.json` and `clusters.npz` (names used) or `04-diarize/turns.json` and `clusters.npz` (not used), so every episode's characters stage reruns once after upgrading. Without usable names it reproduces its assignments exactly (pins), and the manifest's content hashes stop the rerun there. With names, assignments change, and translate, tts, timefit and mix rerun for that episode; tts re-renders only cues whose character or card changed (per-cue cache keys).
- **Registry.** Schema 4 migrates on open (`name_by`, `character_aliases`). The characters stage rebuilds only the episode it runs, so earlier episodes' mixed exemplars stay on their characters until rerun. To migrate a series: `fukikae run <ws> <every episode, in order> --from names`, then for the anime profile `fukikae registry resolve <ws> --episodes ...`. `names show` warns when a names-used character still holds exemplars from episodes that ran without names.
- **Cards.** The next tts run re-picks stale automatic cards (above); curated cards are reported.
- **Ground-truth workspaces** keep `labels-aliases.yaml` for eval; the pipeline reads registry aliases only.

## Evaluation

The names are both the input and the ground truth on Fairy Tail E11 to E14, so no number is reported that grades the run on lines it used, except as a labelled ceiling.

**Sets.** `fairytail` (ja) from a hard-link copy of `~/fukikae-scratch/t12/fairytail` (E11 to E14 through tag on the anime profile, WeSpeaker), and `fairytail-en` (English dub track with the same ASS). The en track has E12 to E14 diarized (`~/fukikae-ws/t9hold/fairytail-en`); E11 needs extract, separate, ASR and diarize on GPU 0 first (open decision 7), else the en rows use tune E12 and E13.

**Protocol A, held-out episode (primary).** Three arms, same diarize outputs:
- *baseline*: `[names] use = "off"`; characters for E11 to E14 in order; `registry resolve --episodes S01E11,S01E12,S01E13`; `registry resolve --episodes S01E14`;
- *names*: `use = "auto"`, `holdout_episodes = ["S01E14"]`; same commands. E14's names are never read by the pipeline;
- *ceiling*: `use = "auto"` everywhere (E14 named), labelled as graded on its own input.

Scored with the existing harness: `fukikae eval identity <ws> --from-assignments --tune S01E11,S01E12,S01E13 --holdout S01E14 --gt ass` (the harness learns to segment from `assignments.turns`, so sub-cluster labels score), and with `fukikae eval names` (below). The headline is the *names* arm's held-out row against *baseline*'s: does a registry built with names attribute an unnamed episode better (clean characters, merges, share)?

**Protocol B, hidden lines.** `use = "auto"`, `hide_fraction = 0.2`, `hide_seed = 0`: in every episode, 20% of each key's lines (keys with at least 5 lines) are withheld from the stage, which records them in `names.json`. Scored on the hidden lines only: does a split cluster carry speech that names didn't cover to the right character? Baseline arm: the same hidden lines under `use = "off"`.

**`fukikae eval names <ws> --tune ... [--holdout ...] --gt ass`** (new, CPU only):
- *cue accuracy*: spoken dialogue cues with one usable ground-truth key, attributed by `assign_subtitle_speakers` from the episode's assignments (honouring hidden and held-out lines), speech-weighted; correct when the cue's character's modal ground-truth name over the tuning episodes (the harness's convention) equals the cue's key. Reported per split (tune, held-out, hidden) and per character.
- *reference purity*: per character, the share of its auto-picked reference seconds (`character_references` over the workspace's episodes, computed, not rendered) whose ground-truth label (60% cover) is the character's modal name. This is the measure of the actual failure: Natsu cloned from Wendy is a purity failure.
- *coverage*: named share, sub-clusters per episode, cut turns, dropped row seconds, unvoiced lines.

Rows go to the results ledger with `stage = "names"` (GT2-additive), tags naming the arm, and the identity harness's own rows beside them. Results in `docs/results/2026-11-10-subtitle-speaker-names.md`. Every threshold here (0.5, 0.6, 0.5 s, 0.3 s, 5 s, 90%, 80%) is fixed by this spec before any run; a defect found on E14 is a ruling with its numbers, never a retune on E14.

## Calls I made (for the operator to check)

- Names win over acoustics for the speech they cover; locked actors, operator names and operator aliases win over names; disagreements are reported, never acted on.
- Turn-level relabelling, cuts only between two named lines, and no re-embedding: the stage stays CPU-only and evidence is the existing rows.
- An unnamed, uncurated character whose cluster is 60% one name is renamed, not orphaned, so its id and card history survive.
- Role labels and keys under 5 s of speech are extras; they never create registry characters.
- `/` and `&` names, group words and simultaneous different names name nobody; possession in parentheses is its own key for the operator to alias.
- Inner-voice lines attribute but are never references.
- Automatic clone cards whose references the names contradict are re-picked; curated and imported ones are only flagged.
- Auto detection accepts named share 0.5 and up, with dominance, style and alignment tests; another language's named track is used when the target language's has none.
- No fuzzy name matching; edit-distance neighbours are suggestions.
- `[names]` is the same in every profile and off nowhere by default.

## Open decisions (for the operator)

1. `[names] use = "auto"` as the default in every profile, so any release with usable names uses them without configuration? Recommended: yes.
2. Names win over acoustics for the speech they cover, with locked actors, operator names and operator aliases as the only overrides? Recommended: yes.
3. When one name holds 60% of an unnamed, uncurated character's named speech (and the reverse), rename that character, or always create new characters and leave the old one with whatever acoustics give it? Recommended: rename (keeps ids and card history). On Fairy Tail it never fires for character 2: E11 creates Wendy first, and the operator merges the two (`registry merge`).
4. Re-pick automatic clone cards (`auto`, `auto-fallback`, `restock`) whose references the names contradict, at the next tts run? Recommended: yes; curated and imported cards are only flagged.
5. Role labels (`TOWNSPEOPLE A`) and names under 5 s of speech in an episode become extras and never registry characters? Recommended: yes.
6. Re-embed the pieces of cut turns (one embedder pass on GPU or CPU per episode), or drop rows that cross a cut? Recommended: drop; 4 to 14 turns per Fairy Tail episode are cut, and the stage stays CPU-only.
7. Run Fairy Tail E11 on the English dub track through diarize on GPU 0 so the en set gets the same E11 to E13 tuning as ja? Recommended: yes; otherwise the en rows tune on E12 and E13 only.
8. SDH speaker prefixes (Grid's Korean CC, Alice in Borderland's （名前）) as a second names source in this plan, or a later one? Recommended: later; they name 12 to 30% of lines and need their own coverage rules.
9. After the evaluation, re-render the affected E12 to E14 cues with IndexTTS2 on GPU 0 for a listening check of the Natsu and Young Makarov lines? Recommended: yes.
10. When the target language's dialogue track has no names, use another language's named track (an English Crunchyroll ASS for a German target)? Recommended: yes.
11. The named-share threshold for `use = "auto"`: 0.5, 0.7 or 0.9? Recommended: 0.5 (only named lines act, so a partial track can only add information; the dominance, style and alignment tests catch junk Name columns).
12. Possession: `MAKAROV (WRAITH)` as its own name for the operator to alias, and `MAKAROV/WRAITH` as naming nobody? Recommended: yes; which voice possession uses is a per-show call.

## Left to the builder

- Exact `names.json` schema beyond the fields named in the plan, and the slug rule's edge cases.
- How `names show` lays out its tables.
- Whether `names scan` caches extracted tracks under `.fukikae/cache/subtitles/` (the identity harness's location) or extracts each time.
