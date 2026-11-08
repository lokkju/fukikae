# Ideas

Logged, not scheduled.

## Face detection and active-speaker detection (2026-11-05)

What: detect and track faces in the video, then run an active-speaker model (TalkNet, LoCoNet or Light-ASD class) to tie each diarized turn to an on-screen face track.

Why: labeling voices without visuals is hard (raised by the operator). A face track gives a second identity signal across episodes: face embeddings alongside voice embeddings, which helps when voices are similar or overlap. It also separates on-screen from off-screen speech (narration, phone calls, characters speaking out of frame) and gives the labeler a face to look at.

Rough shape: a new `visual` stage after extract; backends for face detection and tracking and for active-speaker scoring, registered like the other kinds. Output is face tracks with per-turn speaker scores, stored per episode. The registry grows face embeddings next to voice centroids, and matching combines the two.

## Video labeling clips (2026-11-05)

What: `eval sample` exports short mp4 clips (picture plus vocals, or a contact sheet of frames) instead of bare wav.

Why: the labeler can see who is speaking, which wav alone never shows.

Rough shape: cut each sampled turn with ffmpeg from the source video, muxing the separated vocals as the audio track; the contact-sheet variant tiles a few frames across the turn with the transcript line. A flag on `eval sample` picks wav, mp4 or sheet.

## Subtitle-derived speaker labels (2026-11-05)

What: `eval speakers --from-subtitles` turns ASS Name/Actor fields, or SDH/CC speaker prefixes (`JOHN:`, `[John]`), into speaker labels automatically when a release carries them.

Why: it removes hand labeling for those releases and gives ground truth for scoring diarization. Fairy Tail: 100 Years Quest S01E11 and Quality Assurance in Another World S01E01 in the library have fully named ASS lines.

Rough shape: parse the subtitle track, map each cue's speaker to the diarized turn with the greatest time overlap, and write the result in the same format hand labels use.

## Ground truth for face and active-speaker work (found 2026-11-08)

- `wangyueqian/friends_mmc` (Hugging Face, open): Friends episodes with the speaker and named face boxes per turn, frame-numbered relative to the episode, raw video plus face-track audio. A ready test set for the face/active-speaker plan.
- NHK asadora broadcast captions (kitsunekko mirror, `drama_tv`, `[字]` .ass files) colour the main cast per line, which also labels on-screen speakers for face work once aligned.
