"""The audition worker (Task 19): a draft version's voice on test lines, rendered the way the
dub will be.

`AuditionWorker` is one thread with a FIFO queue of jobs, each a draft version (its card and
backend, `fukikae.studio.drafts`) and a line: a real line (one of the character's own cues in
an episode's ``06-translate/cues.<language>.json``) or a typed one (text in the target
language, with no emotion prompt unless ``emotion_from`` names a cue whose source span is
then the prompt). Every line goes through the tts stage's own path (CD12):
`fukikae.stages.tts.prepare_cue` (the card checked with ``[tts] engines``, so the class that
renders is the one the stage would use; `delivery_passed` with A3's line gate, the speaker's
operator gender beating the estimate) and `fukikae.stages.tts.render_line` (the render guards,
the rate estimate and fit, the pause cap) into the workspace's tts cache. A real line is keyed
exactly as the stage keys it: its gate is measured on the character's spoken cues of that
episode against the character's turns there, as the stage measures it, so once the version is
accepted the next tts run finds the line rendered (the rate fit aside: an audition starts the
card's speech rate estimate fresh, which is the stage's estimate for a card's first lines, so
on a backend that takes a rate a later line can key differently; IndexTTS2 takes none). A
typed line's emotion prompt is gated against the character's register in that cue's episode,
or across the season when it doesn't speak there. Wrong-gender source audio is never an
emotion prompt here either: the rule lives in `delivery_passed`.

An audition renders only with its version's own backend and only a clone card (D41, a
hand-edited draft naming a stock card included). A backend that isn't installed is refused
when the job is submitted (`AuditionRefused`). Whether it can run here
(`fukikae.cards.auto.clone_problem`, which can take 30 s for a Docker engine) is checked on
the worker, never on the server's request thread: a job it refuses fails with ``refused`` set
and a message naming ``fukikae backends setup``, and for `CLONE_CHECK_TTL_S` after that a
submit for the backend is refused at once. Nothing falls back to another backend or to a stock
voice.

The GPU (CD15). The model loads only when a render isn't in the cache. On a GPU the worker
first takes ``fukikae.gpulock`` for the workspace's card (``gpu_device``), polling every
``poll_s`` without blocking anything but its own queue: while another fukikae job holds the
card the job is ``waiting`` and `status` names the holder from the lock file (the way
``fukikae run`` reports a wait, also logged every `fukikae.gpulock.WAIT_LOG_S`), and a
waiting job can be cancelled. A busy lock whose file names nobody yet is tried again; what is
left of a job that exited fails the job (`fukikae.gpulock.GpuLeftover`), and `status` reports
it as ``leftover of pid N``. With the lock taken, the load waits at most ``vram_s``
(`AUDITION_VRAM_S`, 120 s) for room on the card, ``waiting`` and cancellable, then fails
naming the processes on it: `fukikae studio` sets no run VRAM timeout, and the tts stage's
own wait would hold the lock forever. The lock is held for as long as the model is loaded and
passed to the worker process, and never with nothing loaded: a job that stops or fails before
its model loads lets go of it at once. Between jobs the model stays loaded; after ``idle_s``
(`AUDITION_IDLE_S`, 300 s) with nothing queued, on `release` (which also cancels the queued
jobs and a job that hasn't started rendering) and on `close` it is unloaded and the lock
released. A job for another backend unloads the current one only when it has a line to
render: a line cached on it unloads nothing. Jobs run one at a time, in order; at most
`MAX_QUEUE` wait. A cancel before the first render (preparing, waiting, loading) stops the
job; a render already running is not interrupted: it finishes into the cache and the job
reports ``cancelled``. A worker process that hangs is killed by its own timeouts
(`fukikae.backends.tts.remote.WorkerSynthesizer`, 900 s to load, 300 s a render), and the job
fails; the server's request threads never wait on the worker.

Each finished audition is appended to its draft version, matched by number and card
(`fukikae.studio.model.Audition`, under the server's drafts lock, never the decisions lock),
and its wav is hard-linked (or copied) from the cache to ``<ws>/studio/auditions/<key>.wav``
for the server to serve at ``/audition/<key>.wav``. The link shares the cache file's inode,
so its mode is left alone; the auditions directory is made 0700 on every use. Emotion clips
cut into ``auditions/ref/`` that no draft's auditions use are pruned whenever the worker
saves a draft.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from fukikae import gpulock
from fukikae.gpu import VramTimeout, free_mb, gpu_processes
from fukikae.stages.tts import LazySynth
from fukikae.studio.drafts import DraftError, drafts_dir, language_of, load_draft, save_draft
from fukikae.studio.model import (
    Audition,
    AuditionLine,
    Draft,
    DraftVersion,
    RealLine,
    RefSpan,
    TypedLine,
)
from fukikae.workspace import Workspace

if TYPE_CHECKING:
    from fukikae.backends.base import Synthesizer
    from fukikae.cards.card import Card
    from fukikae.config import Config
    from fukikae.models import DubCue
    from fukikae.pitch import Gender, GenderEstimate
    from fukikae.stages.tts import RenderJob

log = logging.getLogger(__name__)

AUDITION_IDLE_S = 300.0  # the model is unloaded after this long with nothing queued
AUDITION_VRAM_S = 120.0  # how long a load waits for the card to have room, lock held
CLONE_CHECK_TTL_S = 60.0  # how long a backend's `clone_problem` answer is reused
LOCK_POLL_S = 2.0  # how often a waiting job tries the GPU lock again
MAX_QUEUE = 16  # jobs waiting at most; past that a submit is refused (QueueFull)
MAX_JOBS = 500  # finished jobs remembered for polling, oldest forgotten first
MAX_LINES = 4096  # (draft version, line) pairs whose last job `latest` remembers past MAX_JOBS
TYPED_CUE_ID = "typed"

JobState = Literal["queued", "waiting", "loading", "running", "done", "failed", "cancelled"]
FINISHED = frozenset({"done", "failed", "cancelled"})
_LINE: TypeAdapter[TypedLine | RealLine] = TypeAdapter(AuditionLine)


class AuditionRefused(Exception):  # noqa: N818 (a refusal, the server's 422)
    """The audition can't be made: a backend that can't render here, a line that isn't the
    character's, an episode without translated cues."""


class NoSuchDraft(AuditionRefused):
    """No such character, draft or version (the server's 404)."""


class QueueFull(Exception):  # noqa: N818
    """`MAX_QUEUE` jobs are already waiting."""


class _Cancelled(Exception):  # noqa: N818
    pass


class AuditionJob(BaseModel):
    """A job as the page polls it: its state, the error message of a failed one (``refused``
    when the worker found what `submit` would refuse, the server's 422: a backend that can't
    run here), how long it waited for the GPU, loaded the model and rendered, and the audition
    once done. A job waiting for room on the card says so in ``message``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    character: int
    language: str
    version: int
    line: AuditionLine
    state: JobState
    message: str | None = None
    refused: bool = False
    submitted: str
    waited_s: float | None = None
    load_s: float | None = None
    render_s: float | None = None
    audition: Audition | None = None


class GpuStatus(BaseModel):
    """The worker and the card: the backend loaded (None: none), its device, what the worker
    is doing, who holds the card's lock (``holder``, from the lock file: ``pid 4121 (tts
    indextts2)``; ``ours`` when it is this worker; ``leftover`` when the file names a process
    that exited and the lock is still held, ``leftover of pid N (...); kill it or wait``), the
    jobs queued, the job running and how long until an idle unload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    loaded_backend: str | None
    device: str | None
    state: Literal["idle", "waiting", "loading", "rendering"]
    holder: str | None
    holder_pid: int | None
    holder_job: str | None
    ours: bool
    leftover: bool = False
    queued: int
    running: str | None
    unload_in_s: float | None


@dataclass
class _Job:
    id: str
    character: int
    language: str
    version: int
    line: TypedLine | RealLine
    submitted: str
    state: JobState = "queued"
    message: str | None = None
    refused: bool = False
    cancel: bool = False
    rendering: bool = False  # the first render began: a cancel lets it finish into the cache
    waited_s: float | None = None
    load_s: float | None = None
    render_s: float | None = None
    audition: Audition | None = None
    events: list[str] = field(default_factory=list)

    def snapshot(self) -> AuditionJob:
        return AuditionJob(
            id=self.id, character=self.character, language=self.language, version=self.version,
            line=self.line, state=self.state, message=self.message, refused=self.refused,
            submitted=self.submitted,
            waited_s=self.waited_s, load_s=self.load_s, render_s=self.render_s,
            audition=self.audition,
        )  # fmt: skip


@dataclass(frozen=True)
class _Prepared:
    job: RenderJob
    device: str
    source: RefSpan | None
    line: GenderEstimate | None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def auditions_dir(ws: Workspace | Path) -> Path:
    root = ws.root if isinstance(ws, Workspace) else Path(ws)
    return root / "studio" / "auditions"


class AuditionWorker:
    """One workspace's audition worker (see the module notes). ``drafts_lock`` is the lock the
    server takes for drafts; the worker takes it to read a draft and to record an audition."""

    def __init__(
        self,
        ws: Workspace | Path,
        cfg: Config,
        *,
        drafts_lock: threading.RLock | None = None,
        idle_s: float = AUDITION_IDLE_S,
        poll_s: float = LOCK_POLL_S,
        vram_s: float = AUDITION_VRAM_S,
    ) -> None:
        self.ws = ws if isinstance(ws, Workspace) else Workspace(Path(ws))
        self.cfg = cfg
        self.idle_s = idle_s
        self.poll_s = poll_s
        self.vram_s = vram_s
        self._drafts_lock = drafts_lock if drafts_lock is not None else threading.RLock()
        self._cond = threading.Condition()
        self._queue: collections.deque[_Job] = collections.deque()
        self._jobs: collections.OrderedDict[str, _Job] = collections.OrderedDict()
        # the last job per (character, language, version, line), kept after the job ring
        # forgets it, so a compare never takes a forgotten failure for a line never tried
        self._by_line: collections.OrderedDict[tuple[Any, ...], _Job] = collections.OrderedDict()
        self._current: _Job | None = None
        self._release = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._last_used = time.monotonic()
        self._clone_checked: dict[str, tuple[float, str | None]] = {}  # name -> (when, problem)
        # worker-thread state: the synth, what it is for, and the GPU lock it holds
        self._synth: LazySynth | None = None
        self._synth_for: tuple[str, str] | None = None
        self._gpu = contextlib.ExitStack()
        self._gpu_held = False

    # -- the API the server calls ---------------------------------------------------------------

    def submit(self, character: int, language: str, version: int, line: Any) -> str:
        """Queue an audition of the draft version on ``line`` (a `TypedLine`, `RealLine` or
        their dict); returns the job id. `NoSuchDraft` when the draft or version doesn't
        exist, `AuditionRefused` when its backend isn't installed, is known (within
        `CLONE_CHECK_TTL_S`) not to run here, or the line isn't usable, `QueueFull` when
        `MAX_QUEUE` jobs wait. Never waits for the worker, and never runs `clone_problem`
        (``docker inspect`` can take 30 s): the worker does, and a job it refuses fails with
        ``refused`` set."""
        parsed: TypedLine | RealLine = (
            line if isinstance(line, TypedLine | RealLine) else _LINE.validate_python(line)
        )
        try:
            language = language_of(language)
        except DraftError as e:
            raise NoSuchDraft(str(e)) from None
        v = self._version(character, language, version)
        self._renderer(character, language, v, probe=False)
        self._resolve_line(character, language, parsed)
        job = _Job(
            id=secrets.token_hex(8), character=character, language=language, version=version,
            line=parsed, submitted=_now(),
        )  # fmt: skip
        with self._cond:
            if self._closed:
                raise AuditionRefused("the studio is shutting down")
            if len(self._queue) >= MAX_QUEUE:
                raise QueueFull(f"{MAX_QUEUE} auditions are already queued; wait for some")
            self._queue.append(job)
            self._remember(job)
            key = _line_key(character, language, version, parsed)
            self._by_line[key] = job
            self._by_line.move_to_end(key)
            while len(self._by_line) > MAX_LINES:
                self._by_line.popitem(last=False)
            self._start()
            self._cond.notify_all()
        return job.id

    def job(self, job_id: str) -> AuditionJob | None:
        with self._cond:
            j = self._jobs.get(job_id)
            return j.snapshot() if j is not None else None

    def wait(self, job_id: str, timeout: float | None = None) -> AuditionJob:
        """The job once finished (done, failed or cancelled), or as it stands after
        ``timeout`` seconds. KeyError for an unknown job."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            j = self._jobs[job_id]
            while j.state not in FINISHED:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    break
                self._cond.wait(left)
            return j.snapshot()

    def cancel(self, job_id: str) -> AuditionJob | None:
        """Cancel a job: a queued one is dropped, a waiting one stops waiting for the GPU,
        and a running render finishes into the cache with the job reported cancelled."""
        with self._cond:
            j = self._jobs.get(job_id)
            if j is None:
                return None
            if j.state == "queued":
                self._queue.remove(j)
                j.state, j.message = "cancelled", "cancelled before it started"
            elif j.state not in FINISHED:
                j.cancel = True
            self._cond.notify_all()
            return j.snapshot()

    def release(self) -> None:
        """Let go of the GPU: the queued jobs and a job that hasn't started rendering
        (preparing, waiting for the lock or for room on the card, loading) are cancelled, the
        model is unloaded (after a render already running) and the lock released."""
        with self._cond:
            for j in self._queue:
                j.state, j.message = "cancelled", "the GPU was released"
            self._queue.clear()
            cur = self._current
            if cur is not None and cur.state not in FINISHED and not cur.rendering:
                cur.cancel = True
            self._release = True
            self._cond.notify_all()

    def close(self, timeout: float = 30.0) -> None:
        """Cancel everything, unload and stop the thread (waiting up to ``timeout`` for a
        render already running)."""
        with self._cond:
            self._closed = True
            for j in self._queue:
                j.state, j.message = "cancelled", "the studio stopped"
            self._queue.clear()
            if self._current is not None:
                self._current.cancel = True
            self._cond.notify_all()
            t = self._thread
        if t is not None:
            t.join(timeout)

    def status(self) -> GpuStatus:
        index = int(self.cfg.gpu_device)
        holder = gpulock.current_holder(index)
        left = gpulock.leftover_holder(index) if holder is None else None
        named: str | None = f"pid {holder[0]} ({holder[1]})" if holder else None
        if left is not None:
            pid, what, found = left
            listed = ", ".join(f"pid {p} ({cmd[:200]})" for p, cmd in found)
            named = (
                f"leftover of pid {pid} ({what}){': ' + listed if listed else ''}; kill it or wait"
            )
            holder = (pid, what)
        with self._cond:
            synth = self._synth
            loaded = synth is not None and (synth.inst is not None or self._gpu_held)
            cur = self._current
            state: Literal["idle", "waiting", "loading", "rendering"] = "idle"
            if cur is not None:
                state = {"waiting": "waiting", "loading": "loading"}.get(cur.state, "rendering")  # type: ignore[assignment]
            unload = None
            if loaded and cur is None and not self._queue:
                unload = max(0.0, self.idle_s - (time.monotonic() - self._last_used))
            return GpuStatus(
                index=index,
                loaded_backend=synth.bc.backend if loaded and synth is not None else None,
                device=synth.device if loaded and synth is not None else None,
                state=state,
                holder=named,
                holder_pid=holder[0] if holder else None,
                holder_job=holder[1] if holder else None,
                ours=left is None
                and holder is not None
                and holder[0] == os.getpid()
                and self._gpu_held,
                leftover=left is not None,
                queued=len(self._queue),
                running=cur.id if cur is not None else None,
                unload_in_s=None if unload is None else round(unload, 1),
            )

    def made(self) -> set[str]:
        """The wav names of the auditions this worker rendered."""
        with self._cond:
            return {j.audition.wav for j in self._jobs.values() if j.audition is not None}

    def latest(
        self, character: int, language: str, version: int, line: TypedLine | RealLine
    ) -> AuditionJob | None:
        """The last job submitted for this draft version and line (equal lines: the same
        kind, cue or text and emotion prompt), or None; what a compare polls instead of
        submitting the line again. Remembered for the last `MAX_LINES` pairs, after the job
        ring (`MAX_JOBS`) forgets the job."""
        with self._cond:
            j = self._by_line.get(_line_key(character, language, version, line))
            return j.snapshot() if j is not None else None

    # -- checks shared by submit and the worker ------------------------------------------------

    def _version(self, character: int, language: str, n: int) -> DraftVersion:
        with self._drafts_lock:
            try:
                draft = load_draft(self.ws, character, language)
            except DraftError as e:
                raise NoSuchDraft(str(e)) from None
            except (OSError, ValueError) as e:
                raise AuditionRefused(f"the draft can't be read ({type(e).__name__})") from None
        if draft is None:
            raise NoSuchDraft(
                f"character {character} has no {language} draft; GET /api/drafts/{character}/"
                f"{language} opens one"
            )
        v = next((v for v in draft.versions if v.n == n), None)
        if v is None:
            raise NoSuchDraft(f"draft {draft.name} has no version {n}")
        return v

    def _renderer(
        self, character: int, language: str, v: DraftVersion, *, probe: bool
    ) -> tuple[Card, type[Synthesizer], str]:
        """The version's card, the class that renders it (``[tts] engines`` applied) and its
        device; `AuditionRefused` for a card that isn't a clone (D41, a hand-edited draft
        included), or a backend that isn't installed or can't run here. ``probe`` runs
        `clone_problem` when its answer for the backend is stale (the worker); without it only
        a fresh answer counts (`submit`, on the server's request thread)."""
        from fukikae.backends import loader
        from fukikae.cards import auto
        from fukikae.cards.card import CardError
        from fukikae.cards.store import CardStore
        from fukikae.config import ConfigError
        from fukikae.stages.tts import check_loaded_card, synth_device, tts_engines

        who = f"draft {character}.{language} version {v.n}"
        never = (
            "an audition renders only with its version's own backend, never another one or a "
            "stock voice (D41)"
        )
        store = CardStore(self.ws.root)
        try:
            card = store.get(v.card_id)
            engines = tts_engines(self.cfg)
        except (CardError, ConfigError) as e:
            raise AuditionRefused(f"{who}: {e}") from None
        if card.voice.source != "clone":
            raise AuditionRefused(
                f"{who}: card {v.card_id} is a {card.voice.source} voice, not a clone; {never}"
            )
        name = engines.get(card.voice.backend, card.voice.backend)
        try:
            loader.backend_class("tts", name)
        except Exception as e:  # noqa: BLE001 (unknown, or its extra isn't installed)
            raise AuditionRefused(
                f"{who}: its backend {name} isn't installed here ({type(e).__name__}: {e}); {never}"
            ) from None
        try:
            cls = check_loaded_card(store, card, v.card_id, who, language, engines)
        except CardError as e:
            raise AuditionRefused(f"{e}; {never}") from None
        problem = self._clone_problem(auto.clone_problem, cls, probe)
        if problem is not None:
            raise AuditionRefused(
                f"{who}: the clone backend {cls.name} can't run here ({problem}); run `fukikae "
                f"backends setup {cls.name}`; {never}"
            )
        try:
            device = synth_device(self.cfg, cls)
        except ConfigError as e:
            raise AuditionRefused(f"{who}: {e}") from None
        return card, cls, device

    def _clone_problem(self, check: Any, cls: type[Synthesizer], probe: bool) -> str | None:
        """``check(cls, cfg)`` (`clone_problem`), its answer reused for `CLONE_CHECK_TTL_S`;
        without ``probe`` a stale or missing answer is None (not known to fail)."""
        with self._cond:
            hit = self._clone_checked.get(cls.name)
        if hit is not None and time.monotonic() - hit[0] < CLONE_CHECK_TTL_S:
            return hit[1]
        if not probe:
            return None
        problem = check(cls, self.cfg)
        with self._cond:
            self._clone_checked[cls.name] = (time.monotonic(), problem)
        return problem

    def _cues(self, episode: str, language: str) -> list[DubCue]:
        from fukikae.artifacts import read_json
        from fukikae.models import DubCues

        path = self.ws.work_dir(episode) / "06-translate" / f"cues.{language}.json"
        if not path.is_file():
            raise AuditionRefused(
                f"{episode} has no {language} cues: run `fukikae run {self.ws.root} {episode} "
                "--to translate` first"
            )
        try:
            return read_json(path, DubCues).cues
        except (OSError, ValueError) as e:
            raise AuditionRefused(f"{episode}'s {language} cues can't be read ({e})") from None

    def _cue(self, episode: str, language: str, cue_id: str) -> DubCue:
        cue = next((c for c in self._cues(episode, language) if c.id == cue_id), None)
        if cue is None:
            raise AuditionRefused(f"{episode} has no cue {cue_id!r} in its {language} cues")
        return cue

    def _resolve_line(
        self, character: int, language: str, line: TypedLine | RealLine
    ) -> tuple[DubCue | None, str | None]:
        """(the cue the line reads, its episode): a real line's own cue, which must be the
        character's and spoken; a typed line's emotion prompt cue, which needs a source span."""
        from fukikae.stages.tts import MIN_REFERENCE_S, TtsStage

        if isinstance(line, RealLine):
            cue = self._cue(line.episode, language, line.cue_id)
            if cue.character_id != character:
                whose = (
                    f"character {cue.character_id}'s" if cue.character_id is not None
                    else f"an extra's ({cue.extra or 'no cluster'})"
                )  # fmt: skip
                raise AuditionRefused(
                    f"cue {cue.id} of {line.episode} is {whose}, not character {character}'s"
                )
            why = TtsStage._skipped(cue)
            if why is not None:
                raise AuditionRefused(f"cue {cue.id} of {line.episode} is not spoken ({why})")
            return cue, line.episode
        if line.emotion_from is None:
            return None, None
        ref = line.emotion_from
        cue = self._cue(ref.episode, language, ref.cue_id)
        a, b = cue.delivery.reference
        if b - a < MIN_REFERENCE_S:
            raise AuditionRefused(
                f"cue {cue.id} of {ref.episode} has no source span to prompt with"
            )
        return cue, ref.episode

    # -- the worker thread --------------------------------------------------------------------

    def _start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="audition", daemon=True)
            self._thread.start()

    def _remember(self, job: _Job) -> None:
        self._jobs[job.id] = job
        while len(self._jobs) > MAX_JOBS:
            old = next(iter(self._jobs))
            if self._jobs[old].state not in FINISHED:
                break
            del self._jobs[old]

    def _loaded(self) -> bool:
        return self._synth is not None or self._gpu_held

    def _loop(self) -> None:
        while True:
            job: _Job | None = None
            with self._cond:
                while not (self._closed or self._release):
                    if self._queue:
                        job = self._queue.popleft()
                        job.state = "running"
                        self._current = job
                        self._cond.notify_all()
                        break
                    if not self._loaded():
                        self._cond.wait()
                        continue
                    left = self.idle_s - (time.monotonic() - self._last_used)
                    if left <= 0:
                        break
                    self._cond.wait(left)
                closing = self._closed
                if job is None:
                    self._release = False
            if job is None:
                self._unload()
                if closing:
                    return
                continue
            self._process(job)
            with self._cond:
                self._current = None
                self._last_used = time.monotonic()
                self._cond.notify_all()

    def _set(self, job: _Job, state: JobState) -> None:
        with self._cond:
            if job.state not in FINISHED:
                job.state = state
                if state != "waiting":
                    job.message = None  # a waiting job's note goes once it moves on
            self._cond.notify_all()

    def _check_cancel(self, job: _Job) -> None:
        with self._cond:
            if job.cancel or self._closed:
                raise _Cancelled

    def _process(self, job: _Job) -> None:
        from fukikae.stages.tts import (
            SpeechRates,
            TtsStage,
            backend_config,
            max_speedup,
            render_line,
        )

        started = time.monotonic()
        state: JobState = "failed"
        message: str | None = None
        refused = False
        synth: _AuditionSynth | None = None
        try:
            self._check_cancel(job)
            v = self._version(job.character, job.language, job.version)
            prep = self._prepare(job, v)
            self._check_cancel(job)  # cancelled or released while preparing
            r = prep.job
            want = (r.cls.name, prep.device)
            if isinstance(self._synth, _AuditionSynth) and self._synth_for == want:
                synth = self._synth
            else:
                # Another backend: the current one is unloaded only if a render needs this one
                # (`_AuditionSynth.get`), so a line cached on it unloads nothing.
                synth = _AuditionSynth(
                    self, want, backend_config(self.cfg, r.cls.name), prep.device, self.cfg
                )
            synth.job = job
            cache = self.ws.root / ".fukikae" / "tts-cache"
            cache.mkdir(parents=True, exist_ok=True)
            out = auditions_dir(self.ws)
            out.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(out, 0o700)  # owner-only on every use, as save_draft does the drafts dir
            ref_dir = out / "ref"
            ref_dir.mkdir(exist_ok=True, mode=0o700)
            os.chmod(ref_dir, 0o700)
            got = render_line(
                r, synth, cache, _store(self.ws), speedup=max_speedup(self.cfg),
                ref_dir=ref_dir, rates=SpeechRates(),
            )  # fmt: skip
            wav = f"{got.key}.wav"
            # a hard link shares the cache file's inode, so its mode is left alone: the
            # auditions dir (0700) is what keeps it owner-only
            TtsStage._link(cache / wav, out / wav)
            est = prep.line
            audition = Audition(
                job=job.id, line=job.line, backend=r.cls.name, card_id=r.card_id,
                cue_key=r.key, key=got.key, seconds=round(got.seconds, 4),
                rendered_seconds=round(got.rendered_seconds, 4), rate_factor=got.rate_factor,
                suspect=got.guard.suspect, retry_seed=got.guard.retry_seed, passed=r.passed,
                source_gender=est.gender if est is not None else None,
                gender_gate=r.passed.get("emotion_dropped") == "gender",
                source=prep.source, renders=got.renders, wav=wav, created=_now(),
            )  # fmt: skip
            with self._cond:
                job.audition = audition
            if job.cancel:
                state, message = "cancelled", "cancelled; the render finished and is cached"
            else:
                self._record(job, audition)
                state = "done"
        except _Cancelled:
            state = "cancelled"
            message = "the GPU was released" if self._release else "cancelled"
        except AuditionRefused as e:
            message, refused = str(e), True
        except (gpulock.GpuBusy, VramTimeout) as e:
            message = str(e)
        except Exception as e:  # noqa: BLE001 (the job fails; the worker goes on)
            log.exception("audition %s failed", job.id)
            message = f"{type(e).__name__}: {e}"
            self._unload()  # a worker that failed may be dead: start clean
        if synth is not None and synth is not self._synth:
            synth.close()  # a backend never loaded (every line cached): nothing to keep
        if not (self._synth is not None and self._synth.inst is not None):
            self._unload()  # never the lock held with no model loaded
        total = time.monotonic() - started
        with self._cond:
            job.render_s = round(max(0.0, total - (job.waited_s or 0.0) - (job.load_s or 0.0)), 3)
            job.state, job.message, job.refused = state, message, refused
            self._cond.notify_all()
        a = job.audition
        log.info(
            "audition %s: %s%s (waited %.1f s, loaded %.1f s, rendered %.1f s%s)",
            job.id, state, f": {message}" if message else "", job.waited_s or 0.0,
            job.load_s or 0.0, job.render_s,
            f"; {a.seconds:.2f} s of audio, {a.renders} renders" if a is not None else "",
        )  # fmt: skip

    def _prepare(self, job: _Job, v: DraftVersion) -> _Prepared:
        """The line's RenderJob, as the tts stage would prepare it (see the module notes)."""
        from fukikae.models import Delivery, DubCue
        from fukikae.stages.tts import (
            GenderGate,
            TtsStage,
            gender_gate,
            line_genders,
            prepare_cue,
            tag_events,
        )

        card, cls, device = self._renderer(job.character, job.language, v, probe=True)
        cue_in, episode = self._resolve_line(job.character, job.language, job.line)
        takes_audio = "emotion_audio" in cls.capabilities_for(card.voice.source)
        operator = TtsStage.operator_state(self.ws.root)[0]
        line: GenderEstimate | None = None
        voice: Gender = "u"
        source: RefSpan | None = None
        vocals: Path | None = None
        events: list[tuple[float, float, str]] = []
        if episode is not None and cue_in is not None:
            work = self.ws.work_dir(episode)
            vocals = work / "01-separate" / "vocals.wav"
            if takes_audio and not vocals.is_file():
                raise AuditionRefused(f"{episode} has no separated vocals to prompt with")
            events = tag_events(work)
            source = RefSpan(episode=episode, start=float(cue_in.delivery.reference[0]),
                             end=float(cue_in.delivery.reference[1]))  # fmt: skip
        if isinstance(job.line, RealLine):
            assert cue_in is not None and episode is not None
            cue = cue_in
            if takes_audio:
                spoken = [
                    c for c in self._cues(episode, job.language)
                    if c.character_id == job.character and TtsStage._skipped(c) is None
                ]  # fmt: skip
                gate = gender_gate(self.ws.work_dir(episode), spoken, self.ws.root,
                                   operator=operator)  # fmt: skip
            else:
                gate = GenderGate({}, {})
            line, voice = gate.line(cue.id), gate.voice(cue.id)
        else:
            text = job.line.text.strip()
            delivery = Delivery(reference=(0.0, 0.0))
            if cue_in is not None and episode is not None:
                d = cue_in.delivery
                delivery = Delivery(reference=d.reference, emotion=d.emotion,
                                    emotion_confidence=d.emotion_confidence,
                                    direction=d.direction)  # fmt: skip
                if takes_audio:
                    assert vocals is not None
                    line = line_genders(vocals, [cue_in], self.ws.root)[cue_in.id]
                    voice = self._voice(job, episode, operator)
            cue = DubCue(
                id=TYPED_CUE_ID, start=0.0, end=0.0, character_id=job.character, extra=None,
                speaker="studio", source_text="", text=text, budget=len(text), delivery=delivery,
            )  # fmt: skip
        r = prepare_cue(
            cue, v.card_id, self.cfg, _store(self.ws), who=f"character {job.character}",
            checked=(card, cls), vocals=vocals, events=events, line=line, voice=voice,
        )  # fmt: skip
        return _Prepared(r, device, source, line)

    def _voice(self, job: _Job, episode: str, operator: dict[int, Gender]) -> Gender:
        """The character's established gender for a typed line's emotion prompt from
        ``episode``: the operator's; else its register in that episode, as the stage reads
        it (its turns, else its spoken cues' spans); else across every episode's turns."""
        from fukikae.artifacts import read_json
        from fukikae.cards.auto import character_spans
        from fukikae.models import Assignments
        from fukikae.stages.tts import GATE_BANDS, TtsStage, established_gender

        cid = job.character
        if cid in operator:
            return operator[cid]
        work = self.ws.work_dir(episode)
        asg = work / "05-characters" / "assignments.json"
        spans = character_spans(read_json(asg, Assignments)).get(cid, []) if asg.is_file() else []
        if not spans:
            spans = [
                (float(c.delivery.reference[0]), float(c.delivery.reference[1]))
                for c in self._cues(episode, job.language)
                if c.character_id == cid and TtsStage._skipped(c) is None
            ]
        if spans:
            return established_gender(work / "01-separate" / "vocals.wav", spans, self.ws.root)
        return self._season_gender(cid, GATE_BANDS)

    def _season_gender(self, cid: int, bands: tuple[float, float]) -> Gender:
        from fukikae.artifacts import read_json
        from fukikae.cards.auto import character_spans
        from fukikae.models import Assignments
        from fukikae.pitch import BIMODAL_GATE_RATIO, band_seconds, cached_track

        m = f = 0.0
        episodes = self.ws.root / ".fukikae" / "episodes"
        for work in sorted(episodes.iterdir()) if episodes.is_dir() else []:
            asg, vocals = (
                work / "05-characters" / "assignments.json",
                work / "01-separate" / "vocals.wav",
            )
            if not (asg.is_file() and vocals.is_file()):
                continue
            spans = character_spans(read_json(asg, Assignments)).get(cid)
            if spans:
                secs = band_seconds(cached_track(self.ws.root, vocals, spans), bands)
                m, f = m + secs["m"], f + secs["f"]
        if m > 0 and m >= BIMODAL_GATE_RATIO * f:
            return "m"
        if f > 0 and f >= BIMODAL_GATE_RATIO * m:
            return "f"
        return "u"

    def _take_gpu(self, job: _Job, device: str) -> None:
        """Take the card's lock before a model loads on it, waiting while another job holds
        it (the job ``waiting``, cancellable). A busy lock whose file names a live holder, or
        nobody yet (a holder between its ``flock`` and its line, or letting go), is tried
        again; only what is left of a job that exited fails the job (`gpulock.GpuLeftover`,
        named as gpulock's leftover scan names it)."""
        if not device.startswith("cuda") or self._gpu_held:
            return
        index = int(self.cfg.gpu_device)
        started = time.monotonic()
        logged: float | None = None
        waited = False
        while True:
            self._check_cancel(job)
            try:
                self._gpu.enter_context(
                    gpulock.gpu_lock(index, f"studio audition ({self.ws.root.name})", timeout_s=0)
                )
                self._gpu_held = True
                break
            except gpulock.GpuLeftover:
                raise  # waiting won't free it: say who to kill
            except gpulock.GpuBusy:
                holder = gpulock.current_holder(index)
                who = f"pid {holder[0]} ({holder[1]})" if holder else "another fukikae job"
                waited = True
                self._set(job, "waiting")
                now = time.monotonic()
                if logged is None or now - logged >= gpulock.WAIT_LOG_S:
                    log.warning("audition %s: waiting for GPU %d, held by %s", job.id, index, who)
                    logged = now
                with self._cond:
                    if not (job.cancel or self._closed):
                        self._cond.wait(self.poll_s)
        if waited:
            with self._cond:
                job.waited_s = round(time.monotonic() - started, 3)

    def _unload(self) -> None:
        """Unload the model and let go of the lock (worker thread only: the lock is held per
        thread)."""
        synth = self._synth
        try:
            if synth is not None:
                synth.close()
        except Exception:  # noqa: BLE001 (an unload that fails still lets go of the lock)
            log.exception("audition worker: unloading failed")
        finally:
            gpu, self._gpu = self._gpu, contextlib.ExitStack()
            try:
                gpu.close()
            finally:
                with self._cond:  # reported unloaded only once the lock is let go
                    self._gpu_held = False
                    self._synth, self._synth_for = None, None
                    self._cond.notify_all()

    def _record(self, job: _Job, audition: Audition) -> None:
        """Append the audition to its draft version (the same number and card: a draft
        replaced meanwhile gets nothing), under the drafts lock; then prune the emotion clips
        no draft's auditions use (`_prune_refs`)."""

        def ours(v: DraftVersion) -> bool:
            return v.n == job.version and v.card_id == audition.card_id

        with self._drafts_lock:
            try:
                draft = load_draft(self.ws, job.character, job.language)
            except (DraftError, OSError, ValueError):
                log.exception("audition %s: its draft can't be read; not recorded", job.id)
                return
            if draft is None or not any(ours(v) for v in draft.versions):
                log.info("audition %s: its draft version is gone; not recorded", job.id)
                return
            versions = [
                v.model_copy(update={"auditions": [*v.auditions, audition]}) if ours(v) else v
                for v in draft.versions
            ]  # fmt: skip
            save_draft(self.ws, draft.model_copy(update={"versions": versions}))
            self._prune_refs()

    def _prune_refs(self) -> None:
        """Delete the files of ``auditions/ref/`` that no saved draft's auditions use. A
        render cuts its emotion prompt there as ``<cue id>.wav`` (``typed.wav`` for a typed
        line's), so the names kept are those of every audition's line. Runs on the worker
        thread, under the drafts lock, after the worker saves a draft: never during one of
        its renders, which read those files."""
        from fukikae.artifacts import read_json

        ref = auditions_dir(self.ws) / "ref"
        if not ref.is_dir():
            return
        keep: set[str] = set()
        for path in sorted(drafts_dir(self.ws).glob("*.json")):
            try:
                draft = read_json(path, Draft)
            except (OSError, ValueError):
                return  # a draft that can't be read may use any of them: keep all
            for v in draft.versions:
                for a in v.auditions:
                    if isinstance(a.line, RealLine):
                        keep.add(f"{a.line.cue_id}.wav")
                    elif a.line.emotion_from is not None:
                        keep.add(f"{TYPED_CUE_ID}.wav")
        for f in ref.iterdir():
            if f.name not in keep and not f.is_dir():
                with contextlib.suppress(OSError):
                    f.unlink()


def cached_lines(
    ws: Workspace | Path,
    cfg: Config,
    character: int,
    card_id: str,
    language: str,
    episodes: list[str],
) -> dict[str, list[str]]:
    """Per episode, the character's spoken cues whose render with ``card_id`` is already in
    the workspace's tts cache under the stage's own ``cue_key`` (prepared as the stage
    prepares them, the line gate measured over every spoken cue of the episode as the stage
    measures it), so the next tts run takes them as cache hits: what an accept's response
    lists beside the episodes it makes stale. ``cfg`` is the config that run will read (the
    server re-reads it). An episode without ``cues.<language>.json`` lists none. Reads only;
    raises what the card check or a read raises."""
    from fukikae.artifacts import read_json
    from fukikae.models import DubCues
    from fukikae.stages.tts import (
        GenderGate,
        TtsStage,
        check_loaded_card,
        gender_gate,
        prepare_cue,
        tag_events,
        tts_engines,
    )

    w = ws if isinstance(ws, Workspace) else Workspace(Path(ws))
    store = _store(w)
    card = store.get(card_id)
    who = f"character {character}"
    cls = check_loaded_card(store, card, card_id, who, language, tts_engines(cfg))
    takes_audio = "emotion_audio" in cls.capabilities_for(card.voice.source)
    operator = TtsStage.operator_state(w.root)[0]
    cache = w.root / ".fukikae" / "tts-cache"
    out: dict[str, list[str]] = {}
    for ep in episodes:
        work = w.work_dir(ep)
        path = work / "06-translate" / f"cues.{language}.json"
        if not path.is_file():
            out[ep] = []
            continue
        spoken = [c for c in read_json(path, DubCues).cues if TtsStage._skipped(c) is None]
        mine = [c for c in spoken if c.character_id == character]
        # the gate over every spoken cue of the episode, as the stage measures it: the pitch
        # tracks are cached by their spans, so this reads the stage's entry
        gate = (
            gender_gate(work, spoken, w.root, operator=operator) if takes_audio and mine
            else GenderGate({}, {})
        )  # fmt: skip
        events = tag_events(work)
        vocals = work / "01-separate" / "vocals.wav"
        hits: list[str] = []
        for c in mine:
            r = prepare_cue(
                c, card_id, cfg, store, who=who, checked=(card, cls), vocals=vocals,
                events=events, line=gate.line(c.id), voice=gate.voice(c.id),
            )  # fmt: skip
            if (cache / f"{r.key}.wav").is_file():
                hits.append(c.id)
        out[ep] = hits
    return out


def _line_key(
    character: int, language: str, version: int, line: TypedLine | RealLine
) -> tuple[Any, ...]:
    return (character, language, version, line.model_dump_json())


def _store(ws: Workspace) -> Any:
    from fukikae.cards.store import CardStore

    return CardStore(ws.root)


class _AuditionSynth(LazySynth):
    """The tts stage's `LazySynth` for one backend and device (``want``), loaded only when a
    render needs it: then it unloads the worker's current backend (a backend switch), takes
    the GPU lock through the worker (a wait it can report and cancel), waits a finite,
    cancellable time for room on the card (`AuditionWorker.vram_s`) and times the load. A
    cancel before the first render stops the job; once rendering, the render finishes."""

    def __init__(
        self, worker: AuditionWorker, want: tuple[str, str], bc: Any, device: str, cfg: Config
    ) -> None:
        super().__init__(bc, device, cfg)
        self.worker = worker
        self.want = want
        self.job: _Job | None = None

    def get(self) -> Synthesizer:
        w, job = self.worker, self.job
        assert job is not None
        if not job.rendering:
            w._check_cancel(job)
            if self.inst is None:
                if w._synth is not self:
                    w._unload()
                    with w._cond:
                        w._synth, w._synth_for = self, self.want
                w._take_gpu(job, self.device)
                w._check_cancel(job)
                w._set(job, "loading")
                t = time.monotonic()
                super().get()
                with w._cond:
                    job.load_s = round(time.monotonic() - t, 3)
            with w._cond:  # cancelled while loading: stop before the render
                if job.cancel or w._closed:
                    raise _Cancelled
                job.rendering = True
            w._set(job, "running")
        elif self.inst is None:
            super().get()
        assert self.inst is not None
        return self.inst

    def wait_vram(self, inst: Synthesizer) -> None:
        """Wait, lock held, for ``inst.vram_mb`` free on the card, polling every ``poll_s``
        and stopping on a cancel, a release or close; after ``vram_s`` the job fails
        (`VramTimeout`) naming the processes on the card, and the worker lets go of the
        lock. The tts stage's wait (``runctx``) is forever when `fukikae studio` sets none."""
        w, job = self.worker, self.job
        assert job is not None
        need = int(inst.vram_mb)
        if need <= 0:
            return
        index = int(self.cfg.gpu_device)
        started = time.monotonic()
        logged = False
        while True:
            free = free_mb(index)
            if free >= need:
                w._set(job, "loading")
                return
            waited = time.monotonic() - started
            if waited >= w.vram_s:
                using = ", ".join(f"pid {p} ({n})" for p, n in gpu_processes(index))
                raise VramTimeout(
                    f"GPU {index} never had room for {inst.name}: it needs {need} MiB free and "
                    f"had {free} MiB after {waited:.0f} s; using the card: "
                    f"{using or 'nothing this host can name'}. Free the card and try again"
                )
            note = f"waiting for {need} MiB free on GPU {index} ({free} MiB free now)"
            with w._cond:
                if job.state not in FINISHED:
                    job.state, job.message = "waiting", note
                w._cond.notify_all()
            if not logged:
                log.warning("audition %s: %s", job.id, note)
                logged = True
            with w._cond:
                if job.cancel or w._closed:
                    raise _Cancelled
                w._cond.wait(max(0.0, min(w.poll_s, w.vram_s - waited)))


__all__ = [
    "AUDITION_IDLE_S",
    "AUDITION_VRAM_S",
    "MAX_QUEUE",
    "AuditionJob",
    "AuditionRefused",
    "AuditionWorker",
    "GpuStatus",
    "NoSuchDraft",
    "QueueFull",
    "auditions_dir",
    "cached_lines",
]
