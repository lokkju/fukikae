# Korean ASR on Chimera (2021) S01E01

Character error rate against the episode's Korean subtitle track (stream 2, `subrip`, `kor`), whole episode (59:33), scored on one normalized reference string and one normalized hypothesis string per backend (markup, SDH notes, punctuation and all whitespace removed; timing ignored). The ASR input is the Demucs vocals track; runtime is the wall-clock of the ASR stage including alignment.

| backend | model | CER | runtime |
|---|---|---|---|
| qwen3_asr | Qwen/Qwen3-ASR-1.7B (qwen-asr 0.0.6) | 0.1251 | 121 s |
| faster_whisper | large-v3 (faster-whisper 1.2.1, ctranslate2 4.8.5) | 0.1287 | 140 s |

Result: a tie within reference noise. The 0.0036 CER gap (about 3% relative) comes from one episode and one run, and does not separate the backends. With Latin and digit runs removed from both sides the two are 0.1177 and 0.1173 (faster_whisper marginally ahead), see below.

Run order was qwen3_asr first, then faster_whisper, in one process, one run each. The model weights were already on disk in the Hugging Face cache (downloaded earlier that day by the backend tests); neither model had been loaded in that process before, so each runtime includes its own load and CUDA warm-up, and qwen3_asr also paid the process's first CUDA initialization. The 19 s runtime gap is a single measurement under those conditions and is not evidence of a speed difference.

`qwen3_asr` is the provisional Korean default. Neither CER nor runtime separates the two (see above), and segment count is no basis either: qwen3_asr produced 152 segments against 610 for faster-whisper (whose VAD removed 32:08 of the audio before decoding), but a segment is a decoding chunk, not a dubbing cue, and cues will come from word timing and diarization turns. The choice stands until a second episode, or a later stage that depends on the backend, gives a reason to change it.

## Effect of Latin text in the reference

The subtitle keeps some English as Latin text (for example `ANFO`, `IED`, `social psychology`, `spontaneous human combustion`). After normalization the reference has 7100 characters, 262 of them (41 tokens) Latin letters or digits. A Hangul transcription of those words can never match them, so every one a backend writes in Hangul is an error; together they are about 3.7% of the reference. Both backends wrote some of them in Latin letters, which match only when spelled as the subtitle spells them.

| scoring | qwen3_asr | faster_whisper |
|---|---|---|
| as reported | 0.1251 | 0.1287 |
| Latin/digit runs removed from reference and hypothesis | 0.1177 | 0.1173 |

(Computed from the saved transcripts and the srt, no model re-run.) Qwen3-ASR emitted 125 Latin characters and faster-whisper 144; where they differ from the reference spelling they cost extra, which is the 0.0036 gap that disappears once those runs are removed. The Korean reference text itself is the same for both.

## Command

```
uv run fukikae eval asr ~/fukikae-ws/chimera S01E01 --subtitle-track 2 --backend qwen3_asr --backend faster_whisper
```

Workspace `~/fukikae-ws/chimera/fukikae.toml` (outside the repo), with the winner set after the run:

```toml
source = "~/fukikae-ws/media/Chimera (2021)"
source_language = "ko"
gpu_device = 0

[asr.lang.ko]
backend = "qwen3_asr"
```

GPU 0 (RTX 3090) was free; GPU 1 runs vLLM.

## Where each backend erred

Reference first, hypothesis second (normalized text, no spaces).

qwen3_asr:

- English loanwords written in Hangul phonetically where the subtitle keeps Latin text: `spontaneoushumancombustion` against `스펀티네이스휴먼컴버스전`; `socialpsychology` against `소셜사이클러즈` (mishears the word).
- Dropped a repeated shout the subtitle keeps: `야야야아어떻게` against `어어떻게` (the subtitle does keep repeats, e.g. `없어 없어 없어 없어`).
- Overlapping speech: `돈이어디서나서요` against `돈이어디서나왔어음아요`.

faster_whisper:

- Mishears names and words: `그헤더웨이면은` against `그해서외면은`.
- Drops whole phrases: `위법행위그런거안해요` against `그런거아네요`; `닥터닥터랑기자라` missing before `손한기기자`.
- Gets `socialpsychology` right phonetically (`소셜사이콜로지`), which Qwen did not.
