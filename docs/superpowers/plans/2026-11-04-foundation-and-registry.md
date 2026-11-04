# doppio Foundation and Character Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A resumable CLI pipeline that takes an episode from extraction through ASR, diarization and season-level character resolution, plus the evaluation tools to measure whether recurring characters keep one ID across two episodes of Chimera (2021).

**Architecture:** A `uv` Python package. Pipeline stages are thin orchestration over pluggable backends found through entry points; each stage writes versioned JSON/WAV artifacts into a per-episode work directory and records a cache key in a manifest, so a changed input re-runs only what depends on it. Characters and actors live in a per-series SQLite registry.

**Tech Stack:** Python 3.12, uv, typer, pydantic v2, numpy, scipy, sqlite3 (stdlib), pytest, ruff, pyright, pre-commit; ffmpeg/ffprobe via mise (`conda:ffmpeg`); backends: demucs, faster-whisper, torchaudio (wav2vec2 forced alignment), qwen-asr (Qwen3-ASR and Qwen3-ForcedAligner), pyannote.audio, speechbrain, nvidia-ml-py, jiwer.

**Spec:** `docs/superpowers/specs/2026-11-04-doppio-design.md` (ASR choices: `docs/research/2026-11-04-asr-backends.md`). This plan covers spec milestones 2-4. Translation, TTS, time fitting, mix and the subtitle-first path are the next plan.

## Global Constraints

- License PolyForm Shield 1.0.0 (`LICENSE`, `license = "LicenseRef-PolyForm-Shield-1.0.0"` in pyproject).
- uv only: `uv sync`, `uv run`; never pip. Python `>=3.12`.
- Never depend on Kubernetes or any service an ordinary user wouldn't have.
- Model code lives only in `src/doppio/backends/<kind>/`; stages never import a model library.
- ffmpeg and ffprobe come from `.mise/config.toml` (`ffmpeg = "latest"` resolves to `conda:ffmpeg`); never assume them on the host. Invoke them with argument lists, never a shell string.
- Source media is read-only: nothing is ever written beside a source episode. Work directories live in the workspace.
- Heavy backends are optional extras (`doppio[demucs]`, `[whisper]`, `[qwen]`, `[pyannote]`); the core package imports none of them at module load.
- Test media is never committed. GPU and media tests are marked `@pytest.mark.gpu` / `@pytest.mark.media` and read `DOPPIO_TEST_MEDIA` (a directory made by `scripts/make-fixtures`); they skip when it is unset.
- New-character bar: 20 s of clean speech in the episode (`registry.min_new_seconds = 20.0`).
- Milestone 4 bar: pass at >= 90% of recurring characters resolving to one ID with no two characters merged; < 75% means rework embeddings/separation.
- Commit messages: Conventional Commits, no AI attribution. Commit after every task.

## Review Focus

1. **Paths with spaces, parentheses and apostrophes** (`Chimera (2021)/Season 1/Chimera - S01E01 - A Strange Shadow WEBDL-1080p.mkv`): every subprocess call works on them. Pinned in Task 7.
2. **Multi-track sources**: an MKV with an English dub plus the original track (Alice in Borderland) must extract the original-language track, not the default one; 5.1 audio is downmixed. Pinned in Task 7.
3. **Interrupted stage**: Ctrl-C mid-write leaves no artifact that a later run mistakes for complete. Atomic writes, manifest updated last. Pinned in Task 5.
4. **Episodes with little or no clean speech**: zero qualifying clusters produce an empty `assignments.json`, not a crash or a junk character. Pinned in Task 14.
5. **Read-only source share**: a run never attempts a write under the source directory, even for temp files. Pinned in Task 7.

---

## File Structure

```
pyproject.toml  LICENSE  README.md  CLAUDE.md  .pre-commit-config.yaml
.mise/config.toml                      # python, uv, ffmpeg
scripts/make-fixtures                  # cuts test clips from the library episodes into $DOPPIO_TEST_MEDIA
src/doppio/
  __init__.py  cli.py                  # typer root; subcommands registered from the modules below
  models.py                            # Word, Segment, Transcript, Turn, Separation, Assignment (pydantic)
  artifacts.py                         # atomic JSON/WAV writes, schema_version check
  config.py                            # Config, BackendConfig, load_config, resolve
  workspace.py                         # Workspace, Episode (paths, SxxEyy parse, source track choice)
  job.py                               # Manifest, Stage protocol, cache keys, run_stages
  gpu.py                               # wait_for_vram
  media.py                             # ffprobe/ffmpeg wrappers
  doctor.py
  backends/
    base.py  loader.py
    separation/demucs.py
    asr/faster_whisper.py  asr/qwen3.py
    align/wav2vec2.py  align/qwen3.py
    diarize/pyannote.py
    embed/ecapa.py
  stages/
    extract.py  separate.py  language.py  asr.py  diarize.py  characters.py
  registry/
    store.py                           # SQLite schema and CRUD
    matching.py                        # pure matching and pinning logic
    cluster.py                         # batch agglomerative seeding
    cli.py
  eval/
    asr.py  speakers.py  cli.py
tests/ (mirrors src/doppio)
```

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `README.md`, `CLAUDE.md`, `.pre-commit-config.yaml`, `.mise/config.toml`, `.gitignore`, `src/doppio/__init__.py`, `src/doppio/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `doppio.cli.app: typer.Typer`; console script `doppio`; `doppio.__version__: str`.

Use the `lokkju:python-project` skill's conventions for pyproject, ruff, pyright and pre-commit. `.mise/config.toml`: `[tools] python = "3.12"`, `uv = "latest"`, `ffmpeg = "latest"`. pytest markers `gpu`, `media`, `slow` registered in pyproject; default `addopts = "-m 'not gpu and not media'"`. Optional extras as in Global Constraints (empty lists are fine until their task). `.gitignore` includes `.doppio/`.

- [ ] **Step 1: Write the failing test**

```python
from typer.testing import CliRunner
from doppio.cli import app
from doppio import __version__

def test_version():
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.stdout
```

- [ ] **Step 2:** `uv run pytest tests/test_cli.py -v` — Expected: FAIL (import error).
- [ ] **Step 3:** Create the scaffold and the typer app with a `--version` callback.
- [ ] **Step 4:** `uv run pytest -v && uv run ruff check && uv run pyright && mise exec -- ffprobe -version` — Expected: all pass, ffprobe prints a version.
- [ ] **Step 5:** Commit `chore: scaffold doppio package`.

### Task 2: Shared models and atomic artifacts

**Files:**
- Create: `src/doppio/models.py`, `src/doppio/artifacts.py`
- Test: `tests/test_models.py`, `tests/test_artifacts.py`

**Interfaces:**
- Produces (all pydantic `BaseModel`, times in float seconds):
  - `Word(start, end, text, score: float | None = None)`
  - `Segment(start, end, text, words: list[Word] | None = None)`
  - `Transcript(schema_version: int = 1, language: str, backend: str, word_timing: bool, segments: list[Segment])`
  - `Turn(start, end, speaker: str)`; `Diarization(schema_version: int = 1, backend: str, turns: list[Turn])`
  - `Separation(vocals: Path, background: Path, vocals_16k: Path)`
  - `ClusterAssignment(cluster: str, actor_id: int | None, character_id: int | None, score: float | None, clean_seconds: float, status: Literal["matched", "new", "extra", "pinned", "review"])`; `Assignments(schema_version: int = 1, episode: str, threshold: float, clusters: list[ClusterAssignment])`
  - `artifacts.write_json(path: Path, model: BaseModel) -> None`, `artifacts.read_json(path: Path, cls: type[T]) -> T`, `artifacts.atomic_path(path: Path) -> ContextManager[Path]` (yields a temp path in the same directory; renames onto `path` on clean exit, deletes on exception).

- [ ] **Step 1: Write the failing tests**

```python
def test_transcript_roundtrip(tmp_path):
    t = Transcript(language="ko", backend="x", word_timing=True,
                   segments=[Segment(start=0.0, end=1.5, text="안녕", words=[Word(start=0.0, end=1.5, text="안녕")])])
    write_json(tmp_path / "t.json", t)
    assert read_json(tmp_path / "t.json", Transcript) == t

def test_read_rejects_newer_schema(tmp_path):
    (tmp_path / "t.json").write_text('{"schema_version": 99, "language": "ko", "backend": "x", "word_timing": false, "segments": []}')
    with pytest.raises(ArtifactVersionError):
        read_json(tmp_path / "t.json", Transcript)

def test_atomic_path_leaves_nothing_on_error(tmp_path):
    target = tmp_path / "a.wav"
    with pytest.raises(RuntimeError):
        with atomic_path(target) as p:
            p.write_bytes(b"partial")
            raise RuntimeError
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement. `ArtifactVersionError` lives in `artifacts.py`; reading compares `schema_version` against the model field default.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5:** Commit `feat: shared models and atomic artifact io`.

### Task 3: Config loading and per-language resolution

**Files:**
- Create: `src/doppio/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `BackendConfig(backend: str, options: dict[str, Any])` (frozen dataclass); `options` is every key in the table except `backend`, `lang`, `pair`.
  - `Config` with `target_language: str = "en"`, `source_language: str | None`, `source: Path | None` (the series directory on the media share), `gpu_device: int = 0`, `registry: RegistrySettings(min_new_seconds=20.0, match_threshold: float)`, `raw: dict`.
  - `load_config(workspace: Path, user_config: Path | None = None) -> Config`: user config default `~/.config/doppio/config.toml`; workspace file `doppio.toml`. Deep-merge user then workspace.
  - `Config.resolve(kind: str, language: str | None = None, pair: str | None = None) -> BackendConfig`: per-pair table (`[kind.pair.ko-en]`) over per-language (`[kind.lang.ko]`) over the kind's base table. An override table that omits `backend` inherits it. Raises `ConfigError` naming the kind when no backend resolves.

- [ ] **Step 1: Write the failing tests**

```python
def test_language_override_wins(tmp_path):
    (tmp_path / "doppio.toml").write_text('[asr]\nbackend="faster_whisper"\nmodel="large-v3"\n[asr.lang.ko]\nbackend="qwen3_asr"\nmodel="Qwen/Qwen3-ASR-1.7B"\n')
    c = load_config(tmp_path, user_config=None)
    assert c.resolve("asr", "ko") == BackendConfig("qwen3_asr", {"model": "Qwen/Qwen3-ASR-1.7B"})
    assert c.resolve("asr", "fr") == BackendConfig("faster_whisper", {"model": "large-v3"})

def test_override_inherits_backend_and_options(tmp_path): ...
    # [translate] backend="openai_compat" model="a" max_speed=1.25 ; [translate.pair.ko-en] model="b"
    # resolve("translate", pair="ko-en") == BackendConfig("openai_compat", {"model": "b", "max_speed": 1.25})

def test_workspace_overrides_user_config(tmp_path): ...
def test_missing_backend_raises(tmp_path): ...  # ConfigError mentions "diarize"
```

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement with `tomllib`. `match_threshold` default 0.6 (placeholder until Task 17 tunes it; leave a comment saying so).
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5:** Commit `feat: config with per-language and per-pair resolution`.

### Task 4: Backend protocols and entry-point loader

**Files:**
- Create: `src/doppio/backends/base.py`, `src/doppio/backends/loader.py`
- Modify: `src/doppio/cli.py` (add `backends list`)
- Test: `tests/backends/test_loader.py`

**Interfaces:**
- Produces in `base.py`:
  - `class Backend(Protocol)`: class attrs `name: str`, `kind: str`, `languages: frozenset[str] | Literal["*"]`, `devices: frozenset[str]` (subset of `{"cuda", "cpu"}`), `vram_mb: int`; `__init__(self, options: dict[str, Any], device: str)`; `load() -> None`; `unload() -> None` (must free GPU memory: drop references, `gc.collect()`, `torch.cuda.empty_cache()` when torch is imported).
  - `Separator.separate(audio: Path, out_dir: Path) -> Separation`
  - `Transcriber.transcribe(audio: Path, language: str | None) -> Transcript`; `Transcriber.detect_language(audio: Path) -> tuple[str, float]` (optional; `supports_lid: bool` class attr)
  - `Aligner.align(audio: Path, transcript: Transcript) -> Transcript` (returns `word_timing=True`)
  - `Diarizer.diarize(audio: Path) -> Diarization`
  - `Embedder.embed(audio: Path, spans: list[tuple[float, float]]) -> np.ndarray` (shape `(len(spans), dim)`, L2-normalized rows); `dim: int`
- Produces in `loader.py`: `available(kind: str) -> dict[str, type[Backend]]` (entry point group `doppio.backends.<kind>`); `create(kind: str, cfg: BackendConfig, device: str) -> Backend` (raises `ConfigError` listing installed names on an unknown backend; raises `BackendUnavailable` with the extra to install when the entry point's import fails); `supports(cls, language: str) -> bool`.

- [ ] **Step 1: Write the failing tests** using a fake entry point (monkeypatch `importlib.metadata.entry_points` to return an `EntryPoint` for `tests.backends.fakes:FakeTranscriber`):

```python
def test_create_unknown_lists_installed(fake_eps):
    with pytest.raises(ConfigError, match="fake"):
        create("asr", BackendConfig("nope", {}), "cpu")

def test_create_passes_options(fake_eps):
    b = create("asr", BackendConfig("fake", {"model": "m"}), "cpu")
    assert b.options == {"model": "m"}

def test_import_failure_names_extra(broken_ep): ...  # BackendUnavailable mentions "doppio[whisper]"
def test_backends_list_cli(fake_eps): ...  # output contains "asr" and "fake"
```

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement. Each entry point name is the backend name; the extra name comes from a class attr `extra: str`, read without importing by the loader keeping a small static map `{"faster_whisper": "whisper", ...}` populated in each backend task.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5:** Commit `feat: backend protocols and entry-point loader`.

### Task 5: Manifest, cache keys and the stage runner

**Files:**
- Create: `src/doppio/job.py`
- Test: `tests/test_job.py`

**Interfaces:**
- Consumes: `Config`, `artifacts.atomic_path`.
- Produces:
  - `class Stage(Protocol)`: `name: str`; `inputs(ep: Episode) -> list[Path]`; `outputs(ep: Episode) -> list[Path]`; `config_key(cfg: Config, ep: Episode) -> dict` (the resolved backend configs this stage uses, JSON-serializable); `run(ep: Episode, cfg: Config) -> None`.
  - `Manifest` (pydantic, `manifest.json` in the episode work dir): `stages: dict[str, StageRecord(key: str, status: Literal["done", "failed"], started: datetime, finished: datetime | None, error: str | None)]`.
  - `cache_key(stage, ep, cfg) -> str`: sha256 over the stage name, `config_key` JSON (sorted keys) and each input's fingerprint. Fingerprint: for paths inside the workspace, sha256 of contents (cache by `(path, size, mtime_ns)` in `.doppio/hashcache.json` to avoid rehashing); for paths outside it (source media), `(resolved path, size, mtime_ns)`.
  - `run_stages(ep: Episode, cfg: Config, stages: Sequence[Stage], from_stage: str | None = None, to_stage: str | None = None) -> list[str]` (returns the names it ran). A stage runs when it is at or after `from_stage`, or its key differs from the manifest, or any output is missing. The manifest record is written only after `run` returns; on exception the record is `failed` with the error and the exception propagates.

- [ ] **Step 1: Write the failing tests** with two fake stages (`a` writes `a.txt` from a config value; `b` reads `a.txt`, writes `b.txt`) and a call counter:

```python
def test_second_run_skips(ep, cfg): run_stages(...); assert run_stages(...) == []
def test_config_change_reruns_downstream(ep, cfg): ...  # change a's option -> ["a", "b"]
def test_same_output_content_stops_propagation(ep, cfg): ...  # a's key changes but writes identical a.txt -> ["a"]
def test_hand_edit_reruns_only_downstream(ep, cfg): ...  # edit a.txt -> ["b"]
def test_from_stage_forces(ep, cfg): ...  # from_stage="b" -> ["b"]
def test_failure_records_and_reraises(ep, cfg): ...  # b raises; manifest b.status == "failed"; a still "done"
def test_interrupted_stage_reruns(ep, cfg): ...  # b raises KeyboardInterrupt after writing via atomic_path; b.txt absent; next run runs b
```

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run, expect PASS.
- [ ] **Step 5:** Commit `feat: manifest, cache keys and stage runner`.

### Task 6: VRAM gate

**Files:**
- Create: `src/doppio/gpu.py`
- Test: `tests/test_gpu.py`

**Interfaces:**
- Produces: `free_mb(device: int) -> int` (nvidia-ml-py); `wait_for_vram(device: int, need_mb: int, timeout_s: float | None, poll_s: float = 15.0, query: Callable[[int], int] = free_mb, sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = ...) -> None` (raises `VramTimeout`; `timeout_s=0` means fail at once, used by `--no-wait`); `device_string(cfg: Config) -> str` (`"cuda:<gpu_device>"` when NVML finds the device, else `"cpu"`).

- [ ] **Step 1: Write the failing tests** with an injected `query` returning a sequence and a fake `sleep`: returns when free >= need; raises `VramTimeout` after timeout; `timeout_s=0` raises without sleeping; logs once per wait with need and free.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: vram gate before gpu stages`.

### Task 7: Workspace, episodes, media wrappers, extract stage, doctor

**Files:**
- Create: `src/doppio/workspace.py`, `src/doppio/media.py`, `src/doppio/stages/extract.py`, `src/doppio/doctor.py`, `scripts/make-fixtures`
- Modify: `src/doppio/cli.py` (add `run`, `doctor`)
- Test: `tests/test_workspace.py`, `tests/test_media.py`, `tests/stages/test_extract.py`

**Interfaces:**
- Produces:
  - `Workspace(root: Path)`: `.config` (`load_config(root)`), `.work_dir(episode_id) -> Path` (`root/.doppio/episodes/<episode_id>`), `.registry_path -> Path` (`root/.doppio/registry/registry.sqlite`), `.find_episode(ref: str) -> Episode` (`ref` is a path or an `SxxEyy` searched under `config.source`).
  - `Episode(source: Path, id: str, work: Path)`; `id` is `S01E01` parsed from the filename (`[Ss](\d+)[Ee](\d+)`), else the stem slugified.
  - `media.probe(path) -> ProbeResult(audio: list[AudioTrack(index, codec, channels, language: str | None, title: str | None, default: bool)], subtitles: list[SubTrack(index, codec, language, title, forced: bool)])`.
  - `media.choose_audio_track(probe, source_language: str | None, override: int | None) -> AudioTrack`: override wins; else the first track whose language equals `source_language`; else the first non-English track; else the first track.
  - `media.run_ffmpeg(args: list[str]) -> None` (raises `MediaError` with stderr tail).
  - `ExtractStage` (`name="extract"`): writes `00-extract/audio.wav` (44.1 kHz, stereo downmix, pcm_s16le) from the chosen track; `config_key` includes the chosen track index.
  - `doctor()` checks ffmpeg/ffprobe on PATH, NVML visibility, HF token presence (`huggingface_hub.get_token()` is not None; never print it), installed backends per kind.
  - `doppio run <workspace> <episode-ref> [--from STAGE] [--to STAGE] [--no-wait]`.
  - `scripts/make-fixtures`: given `DOPPIO_TEST_MEDIA`, cuts 90 s clips (`S01E01-clip.mkv`, `S01E02-clip.mkv`, from 00:05:00) of the two Chimera episodes with stream copy, plus a full-episode symlink for each. Episode paths are in the script (from the spec's test material).

- [ ] **Step 1: Write the failing tests**

```python
def test_episode_id_from_awkward_path(tmp_path):
    p = tmp_path / "Chimera (2021)" / "Season 1" / "Chimera - S01E01 - A Strange Shadow WEBDL-1080p.mkv"
    ...; assert ws.find_episode(str(p)).id == "S01E01"

def test_choose_track_prefers_source_language():
    probe = ProbeResult(audio=[AudioTrack(1, "aac", 2, "eng", None, True), AudioTrack(2, "aac", 6, "jpn", None, False)], subtitles=[])
    assert choose_audio_track(probe, "ja", None).index == 2
    assert choose_audio_track(probe, None, None).index == 2   # first non-English

@pytest.mark.media
def test_extract_on_readonly_source_with_spaces(tmp_path, readonly_copy):
    # readonly_copy: a fixture copying a generated 5 s 5.1 test tone mkv (made with ffmpeg lavfi) into
    # "dir (x)/it's a file.mkv" and chmod -R a-w on its directory
    ...; run_stages(ep, cfg, [ExtractStage()])
    info = soundfile.info(ep.work / "00-extract" / "audio.wav")
    assert (info.samplerate, info.channels) == (44100, 2)
    assert sorted(readonly_copy.parent.iterdir()) == [readonly_copy]  # nothing written beside the source
```

(The 5.1 tone fixture uses `ffmpeg -f lavfi`, so this test needs ffmpeg but no library media; mark it `media` only because it shells out. Keep it running in CI-less local runs via `-m media`.)

- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `uv run pytest -m "not gpu" -v` — Expected: PASS. `uv run doppio doctor` lists ffmpeg found.
- [ ] **Step 5:** Commit `feat: workspace, episode resolution, extract stage and doctor`.

### Task 8: Demucs separation backend and separate stage

**Files:**
- Create: `src/doppio/backends/separation/demucs.py`, `src/doppio/stages/separate.py`
- Modify: `pyproject.toml` (extra `demucs`, entry point `doppio.backends.separation: demucs`)
- Test: `tests/stages/test_separate.py`, `tests/backends/test_demucs.py`

**Interfaces:**
- Consumes: `Separator` protocol, `wait_for_vram`, `ExtractStage` output.
- Produces: `DemucsSeparator` (`name="demucs"`, option `model` default `htdemucs_ft`, `devices={"cuda","cpu"}`, `vram_mb=6000`); `SeparateStage` (`name="separate"`) writes `01-separate/vocals.wav`, `background.wav` (44.1 kHz stereo; background = sum of non-vocal stems) and `vocals_16k.wav` (16 kHz mono, for every downstream speech model).
- Stage contract used by every GPU stage from here on: resolve backend, `wait_for_vram(cfg.gpu_device, backend.vram_mb, ...)` when the device is CUDA, `load()`, work, `unload()` in a `finally`.

- [ ] **Step 1: Write the failing tests**: stage test with a fake `Separator` (writes sine files) asserting the three outputs exist, `vocals_16k.wav` is 16 kHz mono, and `unload` was called even when `separate` raises. `@pytest.mark.gpu` test on the 90 s fixture: vocals RMS > background RMS in a known dialogue window (60-70 s of the clip), and vocals + background ≈ input within -30 dB residual.
- [ ] **Step 2–4:** fail, implement, pass (`uv run pytest -m "gpu or not gpu" tests/stages/test_separate.py tests/backends/test_demucs.py`).
- [ ] **Step 5:** Commit `feat: demucs separation stage`.

### Task 9: Language stage, faster-whisper ASR, wav2vec2 aligner, ASR stage

**Files:**
- Create: `src/doppio/stages/language.py`, `src/doppio/backends/asr/faster_whisper.py`, `src/doppio/backends/align/wav2vec2.py`, `src/doppio/stages/asr.py`
- Modify: `pyproject.toml` (extra `whisper`: faster-whisper, torchaudio; entry points)
- Test: `tests/stages/test_language.py`, `tests/stages/test_asr.py`, `tests/backends/test_faster_whisper.py`, `tests/backends/test_wav2vec2.py`

**Interfaces:**
- Produces:
  - `LanguageStage` (`name="language"`): writes `02-language/language.json` `{"language": str, "confidence": float | null, "source": "config" | "detected"}`. Pinned `source_language` skips detection. Otherwise uses the `lid` kind (`cfg.resolve("lid")`, default faster_whisper), which must be a `Transcriber` with `supports_lid`.
  - `FasterWhisperTranscriber` (`name="faster_whisper"`, options `model="large-v3"`, `compute_type` default `float16` on cuda / `int8` on cpu, `vad_filter=True`, `languages="*"`, `supports_lid=True`, `vram_mb=5000`). `detect_language` uses `language_detection_segments=4` over the vocals. `word_timing=False` in its output (Whisper's own word timestamps are not trusted; the aligner owns timing).
  - `Wav2Vec2Aligner` (`name="wav2vec2"`): torchaudio `forced_align` with a per-language CTC model map (`ko: "kresnik/wav2vec2-large-xlsr-korean"`, `ja: "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"`, `zh: "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"`, `en: torchaudio WAV2VEC2_ASR_BASE_960H`; option `model` overrides). Characters it can't align keep their segment's span; words from unaligned segments get `score=None`.
  - `AsrStage` (`name="asr"`): reads language, `resolve("asr", lang)`, transcribes `vocals_16k.wav`; if the result has `word_timing=False`, runs `resolve("align", lang)` (falling back to the base `align` config when the resolved aligner doesn't support the language). Writes `03-asr/transcript.json`.

- [ ] **Step 1: Write the failing tests**: language stage honors pinned config without loading a backend; ASR stage skips the aligner when `word_timing=True`, calls it otherwise, and falls back from an aligner that doesn't support the language (fakes). `@pytest.mark.gpu`: faster-whisper detects `ko` on the Chimera clip; wav2vec2 alignment output has monotonic word times inside their segments.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: language detection, faster-whisper asr and wav2vec2 alignment`.

### Task 10: Qwen3-ASR and Qwen3-ForcedAligner backends

**Files:**
- Create: `src/doppio/backends/asr/qwen3.py`, `src/doppio/backends/align/qwen3.py`
- Modify: `pyproject.toml` (extra `qwen`: `qwen-asr`; entry points `qwen3_asr`, `qwen3_forced_aligner`)
- Test: `tests/backends/test_qwen3.py`

**Interfaces:**
- Produces: `Qwen3AsrTranscriber` (`name="qwen3_asr"`, option `model="Qwen/Qwen3-ASR-1.7B"`, languages: the 30 from the model card, `supports_lid=True`, `word_timing=False`, `vram_mb` measured in this task and recorded); `Qwen3ForcedAligner` (`name="qwen3_forced_aligner"`, option `model="Qwen/Qwen3-ForcedAligner-0.6B"`, languages `{zh, en, yue, fr, de, it, ja, ko, pt, ru, es}`). The aligner's 5-minute limit per call: align per segment group, chunks cut at segment boundaries no longer than 300 s.
- Check `qwen-asr` 0.0.6's actual API against the repo README (https://github.com/QwenLM/Qwen3-ASR) before writing; if the package can't co-install with the `whisper` extra's torch, record the conflict in the task report and pin compatible versions rather than dropping either.

- [ ] **Step 1: Write the failing tests**: unit test for the 300 s chunker (segments 0-200, 200-350, 350-400 → chunks [0-200], [200-400]; a single 400 s segment becomes its own chunk and logs a warning). `@pytest.mark.gpu`: transcribe the Chimera clip, language `ko`, non-empty Hangul text; aligner word times monotonic and within the clip.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: qwen3-asr and qwen3 forced aligner backends`.

### Task 11: ASR evaluation against a subtitle track

**Files:**
- Create: `src/doppio/eval/asr.py`, `src/doppio/eval/cli.py`
- Modify: `src/doppio/cli.py` (add `eval asr`), `pyproject.toml` (jiwer)
- Test: `tests/eval/test_asr.py`

**Interfaces:**
- Produces: `normalize(text: str, language: str) -> str` (strip ASS/HTML tags, SDH brackets `[...]`/`(...)`, music symbols, punctuation; collapse whitespace; for zh/ja/ko remove all whitespace); `cer(reference: str, hypothesis: str) -> float` (jiwer); `doppio eval asr <workspace> <episode> --subtitle-track N [--backend NAME ...]`: extracts the subtitle track text with ffmpeg (`-map 0:N -f srt`), runs the ASR stage once per listed backend into `eval/asr-<backend>/transcript.json` (outside the normal stage dirs so the pipeline's own artifact isn't replaced), and prints a table of backend, CER, runtime.

- [ ] **Step 1: Write the failing tests**: `normalize("<i>[문 닫히는 소리]</i> 안녕 하세요!", "ko") == "안녕하세요"`; `cer("안녕하세요", "안녕하세요") == 0.0`; `cer("안녕하세요", "안녕하새요") == pytest.approx(0.2)`.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: asr evaluation against subtitle tracks`.
- [ ] **Step 6: Run the spec's milestone-3 comparison** on full Chimera S01E01 (Korean srt is subtitle stream index from `media.probe`, language `kor`): `uv run doppio eval asr <ws> S01E01 --subtitle-track <n> --backend qwen3_asr --backend faster_whisper`. Record the table in `docs/results/2026-xx-xx-asr-chimera.md` and set the workspace's `[asr.lang.ko]` default to the winner. Commit `docs: chimera korean asr comparison`.

### Task 12: Diarization and embedding backends, diarize stage

**Files:**
- Create: `src/doppio/backends/diarize/pyannote.py`, `src/doppio/backends/embed/ecapa.py`, `src/doppio/stages/diarize.py`
- Modify: `pyproject.toml` (extra `pyannote`: pyannote.audio, speechbrain; entry points)
- Test: `tests/stages/test_diarize.py`, `tests/backends/test_diarize_embed.py`

**Interfaces:**
- Produces:
  - `PyannoteDiarizer` (`name="pyannote"`, option `model="pyannote/speaker-diarization-community-1"`, reads the HF token via `huggingface_hub.get_token()`, raises a clear error naming the model page when it's missing).
  - `EcapaEmbedder` (`name="ecapa"`, option `model="speechbrain/spkrec-ecapa-voxceleb"`, `dim=192`).
  - `clean_spans(turns: list[Turn], speaker: str, min_len: float = 1.5, pad: float = 0.1) -> list[tuple[float, float]]`: the speaker's turns that overlap no other speaker's turn, shrunk by `pad` at each end, at least `min_len` long after shrinking.
  - `DiarizeStage` (`name="diarize"`): writes `04-diarize/turns.json` (`Diarization`) and `04-diarize/clusters.npz` with, per cluster label, `embeddings (n, dim)`, `spans (n, 2)` and `clean_seconds` (sum of span lengths). Clusters with no clean span get zero rows.

- [ ] **Step 1: Write the failing tests**:

```python
def test_clean_spans_drop_overlap_and_short():
    turns = [Turn(start=0, end=5, speaker="A"), Turn(start=4, end=8, speaker="B"), Turn(start=10, end=11.5, speaker="A"), Turn(start=12, end=20, speaker="A")]
    assert clean_spans(turns, "A") == [(12.1, 19.9)]      # 0-5 overlaps B; 10-11.5 too short after padding
```

  Stage test with fakes: `clusters.npz` holds a key per cluster including one with zero rows. `@pytest.mark.gpu`: on the Chimera clip, at least 2 clusters, embeddings unit-norm.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: pyannote diarization, ecapa embeddings, diarize stage`.

### Task 13: Registry store

**Files:**
- Create: `src/doppio/registry/store.py`
- Test: `tests/registry/test_store.py`

**Interfaces:**
- Produces: `Registry(path: Path)` (opens/creates SQLite, `PRAGMA foreign_keys=ON`, `schema_version` table). Tables:
  - `actors(id INTEGER PK, name TEXT, external_id TEXT, centroid BLOB, locked INTEGER DEFAULT 0)`
  - `characters(id INTEGER PK, actor_id INTEGER REFERENCES actors, name TEXT, voice TEXT /* JSON, null until the TTS plan */, locked INTEGER DEFAULT 0)`
  - `exemplars(id INTEGER PK, actor_id INTEGER REFERENCES actors, episode TEXT, cluster TEXT, embedding BLOB, seconds REAL)`
- Methods: `actors() -> list[Actor]`, `characters() -> list[Character]`, `create_actor_with_character(centroid: np.ndarray) -> tuple[int, int]`, `add_exemplars(actor_id, episode, cluster, embeddings: np.ndarray, seconds: float)`, `recompute_centroid(actor_id)` (mean of exemplar embeddings, normalized; no-op when locked), `rename_character(id, name)`, `set_actor(id, name=None, external_id=None)`, `merge_actors(keep: int, drop: int)` (moves exemplars and characters, recomputes), `lock_actor(id, locked: bool)`, `delete_episode(episode)` (removes its exemplars, recomputes affected centroids). Embeddings stored as float32 bytes; `Actor.centroid` returned as `np.ndarray`.

- [ ] **Step 1: Write the failing tests**: create two actors, add exemplars, merge, assert exemplar count moved, centroid equals normalized mean, dropped actor gone, its character re-pointed; locked actor centroid unchanged by `add_exemplars` + `recompute_centroid`; `delete_episode` recomputes.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: sqlite character and actor registry`.

### Task 14: Matching, pinning and the characters stage

**Files:**
- Create: `src/doppio/registry/matching.py`, `src/doppio/stages/characters.py`
- Test: `tests/registry/test_matching.py`, `tests/stages/test_characters.py`

**Interfaces:**
- Consumes: `Registry`, `clusters.npz`, `Assignments`.
- Produces:
  - `ClusterEvidence(cluster: str, embedding: np.ndarray | None, clean_seconds: float)` (embedding = normalized mean of the cluster's rows, `None` with zero rows).
  - `match(evidence: list[ClusterEvidence], actors: list[Actor], threshold: float, min_new_seconds: float) -> list[ClusterAssignment]`: cosine score against each actor centroid. Best score >= threshold → `matched` to that actor (and its first character). Below threshold with `clean_seconds >= min_new_seconds` → `new`. Otherwise → `extra` (actor and character `None`). When two clusters' best match is the same actor, the higher-scoring one is `matched` and the other `review` (with its score and the actor id, not merged).
  - `pin(previous: Assignments, previous_turns: Diarization, current_turns: Diarization) -> dict[str, ClusterAssignment]`: maps each current cluster to the previous cluster it overlaps most in time; when that overlap is > 50% of the current cluster's total turn time, it inherits the previous assignment with status `pinned`.
  - `CharactersStage` (`name="characters"`): if `05-characters/assignments.json` exists from an earlier run, pins first and matches only the unpinned clusters; creates actors/characters for `new`; adds exemplars for `matched` and `new`; writes `05-characters/assignments.json`. The previous turns are kept as `05-characters/turns.prev.json` for pinning.
  - CLI `doppio registry rematch <workspace> <episode>`: deletes that episode's exemplars and assignments, reruns `characters`, prints each cluster whose character changed.

- [ ] **Step 1: Write the failing tests**:

```python
def test_match_new_extra_and_review():
    a = Actor(id=1, centroid=unit([1, 0, 0]), ...)
    ev = [ClusterEvidence("S0", unit([1, 0.1, 0]), 60.0),     # matches actor 1
          ClusterEvidence("S1", unit([0.95, 0.2, 0]), 40.0),  # also best-matches actor 1, lower score -> review
          ClusterEvidence("S2", unit([0, 1, 0]), 25.0),       # new
          ClusterEvidence("S3", unit([0, 0, 1]), 5.0),        # extra (under 20 s)
          ClusterEvidence("S4", None, 0.0)]                    # extra (no clean speech)
    got = {c.cluster: c.status for c in match(ev, [a], threshold=0.8, min_new_seconds=20.0)}
    assert got == {"S0": "matched", "S1": "review", "S2": "new", "S3": "extra", "S4": "extra"}

def test_empty_evidence_gives_empty_assignments(): ...   # Review Focus 4: stage writes clusters == []
def test_pin_inherits_by_overlap(): ...
def test_rerun_after_registry_drift_keeps_assignment(): ...
    # run stage on ep1, add a far-away exemplar to the actor so its centroid moves below threshold for ep1's cluster,
    # re-run diarize+characters on ep1 -> status "pinned", same character id
```

- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: cluster matching, pinning and characters stage`.

### Task 15: Batch seeding with agglomerative clustering

**Files:**
- Create: `src/doppio/registry/cluster.py`, `src/doppio/registry/cli.py`
- Modify: `src/doppio/cli.py` (mount `registry`)
- Test: `tests/registry/test_cluster.py`

**Interfaces:**
- Produces: `seed_clusters(evidence: dict[str, list[ClusterEvidence]], distance_threshold: float, min_new_seconds: float) -> list[list[tuple[str, str]]]` (groups of `(episode, cluster)`; scipy `linkage(method="average", metric="cosine")` + `fcluster(criterion="distance")`; groups whose total clean seconds < `min_new_seconds` are dropped). CLI `doppio registry cluster <workspace> [--episodes S01E01,S01E02] [--threshold T]`: requires every listed episode's `diarize` stage done and an empty registry (refuses otherwise, telling the user to use per-episode matching), creates one actor+character per group, writes each episode's `assignments.json` with status `matched`. Also the CLI ops `list`, `show`, `rename`, `merge`, `split`, `lock`, `actor` over Task 13's methods (`split <workspace> --episode E --cluster C` moves that episode cluster's exemplars to a new actor with a new character, needing a `Registry.split_cluster(actor_id, episode, cluster) -> tuple[int, int]` added to the store in this task; splitting one actor into several characters without new embeddings, and the `voice` op, belong to the TTS plan); `merge` and `rename` print which episodes' downstream stages are now stale (it marks their `characters` record stale in each manifest by deleting it).

- [ ] **Step 1: Write the failing tests**: three episodes of synthetic evidence with two true speakers (noisy unit vectors around two centers) plus one 5-second outlier → two groups, outlier dropped; CLI refuses on a non-empty registry.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: batch registry seeding and registry cli`.

### Task 16: Speaker evaluation tools

**Files:**
- Create: `src/doppio/eval/speakers.py`
- Modify: `src/doppio/eval/cli.py`
- Test: `tests/eval/test_speakers.py`

**Interfaces:**
- Produces:
  - `doppio eval sample <workspace> --episodes S01E01,S01E02 --per-cluster 4 --out DIR`: for each episode cluster with clean speech, exports up to 4 clean-span clips (`DIR/<episode>/<cluster>-<k>.wav`, 16 kHz mono, from `vocals_16k.wav`) and writes `DIR/labels.yaml` with one entry per clip: `{episode, cluster, clip, start, end, label: ""}`. Clips are named by episode cluster, never by registry ID, so labeling is blind to the system's answer.
  - `score(labels: list[LabeledClip], assignments: dict[str, Assignments]) -> SpeakerScore(recurring: int, consistent: int, merged_pairs: list[tuple[str, str]], fraction: float)`: a label is recurring when it appears in >= 2 episodes; it is consistent when all its clips' clusters map to one character id; a merged pair is two distinct labels whose clusters map to the same character id. Clips with label `""`, `"?"` or `"extra"` are ignored.
  - `doppio eval speakers <workspace> --labels DIR/labels.yaml`: prints the score and the verdict against the spec bar: `pass` (>= 0.90 and no merged pairs), `tune` (0.75-0.90 or merges), `rework` (< 0.75).

- [ ] **Step 1: Write the failing tests**: label "detective" in two episodes mapped to character 1 both times → consistent; "profiler" mapped to 2 and 3 → inconsistent; "surgeon" and "profiler" sharing character 2 → merged pair; fraction 0.5; verdict `rework`.
- [ ] **Step 2–4:** fail, implement, pass.
- [ ] **Step 5:** Commit `feat: speaker consistency evaluation`.

### Task 17: Milestone 4 run on Chimera S01E01-E02

**Files:**
- Create: `docs/results/<date>-registry-chimera.md`, the workspace's `doppio.toml` (outside the repo, e.g. `~/doppio-ws/chimera/`; record its contents in the results doc)

- [ ] **Step 1:** Create the workspace with `source = "<media-export>/TV/Chimera (2021)"`, `source_language = "ko"`, the ASR default from Task 11, `gpu_device` set to the 3090 not running vLLM (check `nvidia-smi` for the card with the vLLM process).
- [ ] **Step 2:** `uv run doppio run ~/doppio-ws/chimera S01E01 --to characters`, then `S01E02`. Expected: both complete; `S01E02/05-characters/assignments.json` has `matched` entries.
- [ ] **Step 3:** `uv run doppio eval sample ~/doppio-ws/chimera --episodes S01E01,S01E02 --per-cluster 4 --out ~/doppio-ws/chimera/labels`. Labeling needs a person listening to the clips: hand `labels.yaml` to the operator as a queued action and wait.
- [ ] **Step 4:** `uv run doppio eval speakers ~/doppio-ws/chimera --labels ~/doppio-ws/chimera/labels/labels.yaml`. If the verdict is `tune`, sweep `registry.match_threshold` (0.5-0.8 in 0.05 steps, `registry rematch` on S01E02 for each) and report the curve; set the best value as the default in `config.py`.
- [ ] **Step 5:** Write the results doc (scores, threshold curve, verdict, failure examples by cluster and timestamp) and commit `docs: chimera registry milestone results`.
