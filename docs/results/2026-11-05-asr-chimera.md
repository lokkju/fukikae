# Korean ASR on Chimera (2021) S01E01

Character error rate against the episode's Korean subtitle track (stream 2, `subrip`, `kor`), whole episode (59:33), scored on one normalized reference string and one normalized hypothesis string per backend (markup, SDH notes, punctuation and all whitespace removed; timing ignored). The ASR input is the Demucs vocals track; runtime is the wall-clock of the ASR stage including alignment.

| backend | model | CER | runtime |
|---|---|---|---|
| qwen3_asr | Qwen/Qwen3-ASR-1.7B (qwen-asr 0.0.6) | 0.1251 | 121 s |
| faster_whisper | large-v3 (faster-whisper 1.2.1, ctranslate2 4.8.5) | 0.1287 | 140 s |

Winner: `qwen3_asr`, by 0.0036 CER (about 3% relative) and 19 s. That is a narrow margin on one episode; the subtitle track is not a verbatim transcript (it drops repeated words and some interjections), so both numbers carry a floor of reference mismatch. Qwen3-ASR produced 152 segments and faster-whisper 610 (VAD removed 32:08 of the audio before decoding), which has no effect on the CER but matters for downstream segmentation.

## Command

```
uv run doppio eval asr ~/doppio-ws/chimera S01E01 --subtitle-track 2 --backend qwen3_asr --backend faster_whisper
```

Workspace `~/doppio-ws/chimera/doppio.toml` (outside the repo), with the winner set after the run:

```toml
source = "~/doppio-ws/media/Chimera (2021)"
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
- Dropped a repeated shout the subtitle keeps: `야야야아어떻게` against `어어떻게`.
- Overlapping speech: `돈이어디서나서요` against `돈이어디서나왔어음아요`.

faster_whisper:

- Mishears names and words: `그헤더웨이면은` against `그해서외면은`.
- Drops whole phrases: `위법행위그런거안해요` against `그런거아네요`; `닥터닥터랑기자라` missing before `손한기기자`.
- Gets `socialpsychology` right phonetically (`소셜사이콜로지`), which Qwen did not.
