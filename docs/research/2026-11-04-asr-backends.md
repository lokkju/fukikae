# ASR backend comparison

_Checked against primary sources 2026-11-04. Prompted by the question of why the first spec draft started with Whisper._

| Backend | Languages | Timestamps | Built-in LID | Extras | Runtime / hardware | Long-form, robustness | Released |
|---|---|---|---|---|---|---|---|
| faster-whisper large-v3 + WhisperX | 99; good zh/ja/ko/es/European | Segment from Whisper; word via wav2vec2 alignment (WhisperX has ja/zh/ko/es aligners) | Yes | none | CTranslate2: CUDA, CPU int8 | Needs VAD chunking; hallucinates on silence and music | 2023, mature |
| Qwen3-ASR 1.7B / 0.6B | 30 + 22 Chinese dialects | Word/char via separate Qwen3-ForcedAligner-0.6B (zh, en, yue, fr, de, it, ja, ko, pt, ru, es; up to 5 min per call) | Yes (97.9% vs Whisper-v3 94.1%) | Robust to singing and BGM (~14 WER on full songs) | Transformers, vLLM; NVIDIA; CPU unverified | "Long audio" claimed, no stated limit | 2026-01-29, [repo](https://github.com/QwenLM/Qwen3-ASR) |
| SenseVoice-Small | zh, yue, en, ja, ko | Word (CTC-derived, precision unverified) | Yes | Emotion (7) and events: BGM, Laughter, Applause, Cry, Cough, Sneeze, Breath | FunASR, ONNX, sherpa-onnx, GGUF; CPU and mobile | 30 s per call, needs VAD; ~15x faster than Whisper-Large | 2024, [repo](https://github.com/FunAudioLLM/SenseVoice) |
| Fun-ASR-MLT-Nano (800M) | 31 | None documented for MLT | No | none | FunASR, vLLM, GGUF | Lyrics WER 30.85 | 2025-12, [repo](https://github.com/FunAudioLLM/Fun-ASR) |
| Paraformer | zh/en | Yes, granularity not stated | No | VAD + punctuation pipeline | FunASR: CPU, CUDA | Mature, zh-centric | 2023, [repo](https://github.com/modelscope/FunASR) |
| Dolphin | 40 Eastern languages + 22 Chinese dialects | Word (added 2026-05) | Yes | none | PyTorch: CUDA, MPS, Ascend | Only base (0.1B) and small (0.4B) public; WER 33/25 | 2025-03, [repo](https://github.com/DataoceanAI/Dolphin) |
| Omnilingual ASR | 1,600+ | None documented | Optional conditioning | none | fairseq2/PyTorch; CTC 2-15 GiB, LLM 5-17 GiB at batch 1 | CTC/LLM under 40 s; "Unlimited" variants for long audio | 2025-11, [repo](https://github.com/facebookresearch/omnilingual-asr) |
| Cohere Transcribe 03-2026 (2B) | 14 incl. zh/ja/ko/es | None | No (language required) | none | Transformers, vLLM, MLX, GGUF; local weights plus API | Auto-chunking; transcribes noise unless VAD-gated | 2026-03-26, [card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) |
| OpenVINO Whisper | Whisper's 99 | Word (alignment heads) | Yes | none | Intel CPU, iGPU/Arc, NPU | A runtime for Whisper, not a separate model | [docs](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.WhisperPipeline.html) |
| Parakeet-TDT-0.6B-v3 | 25 European (no CJK) | Word, char, segment, native | Yes | none | NeMo, GGUF C++; NVIDIA | 24 min in one pass; 11.7% WER at 0 dB SNR; RTFx ~3,300 | 2025-08, [card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Canary-1B-v2 | Same 25 | Segment via NeMo Forced Aligner | not checked | Speech translation | NeMo | | 2025-08, [paper](https://arxiv.org/abs/2509.14128) |
| Nemotron-3.5-ASR-Streaming (0.6B) | 40 locales incl. zh/ja/ko/es | Not documented | Yes | none | NeMo, Transformers; NVIDIA | Streaming-first | 2026-06-04, [card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |
| Voxtral Mini Transcribe V2 | 13 | Word | not checked | Diarization | API only (Realtime variant has open weights) | | 2026-02, [post](https://mistral.ai/news/voxtral-transcribe-2) |

VRAM figures are mostly unpublished; Omnilingual's are the only measured ones. No third-party ja/ko comparisons found.

## Conclusions used in the spec

- Qwen3-ASR plus Qwen3-ForcedAligner is the default for zh, yue, ja, ko. The aligner is model-agnostic and can align any backend's text in its 11 languages.
- Parakeet-TDT-0.6B-v3 is the default for the 25 European languages it covers: native word timing and very fast.
- faster-whisper plus WhisperX alignment is the fallback for everything else and the CPU path. OpenVINO and whisper.cpp are later runtime adapters for the same backend.
- SenseVoice-Small is an event and emotion tagger (BGM, laughter, applause), not a transcriber.
- Skipped for now: Cohere Transcribe (no timestamps, no LID), Paraformer (Qwen3-ASR covers zh), Omnilingual (rare languages only), Dolphin (until larger weights ship), Voxtral (API only).
