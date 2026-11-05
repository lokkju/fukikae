import sys
import types

import pytest

from doppio import gpu
from doppio.config import Config
from doppio.gpu import VramTimeout, device_string, free_mb, wait_for_vram


def _seq(values):
    it = iter(values)
    return lambda _device: next(it)


def test_returns_when_enough_free():
    sleeps: list[float] = []
    wait_for_vram(0, 100, 60, query=_seq([50, 80, 120]), sleep=sleeps.append, log=lambda _m: None)
    assert sleeps == [15.0, 15.0]


def test_returns_at_once_when_free_already():
    sleeps: list[float] = []
    wait_for_vram(0, 100, 60, query=_seq([100]), sleep=sleeps.append, log=lambda _m: None)
    assert sleeps == []


def test_times_out_with_details():
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    with pytest.raises(VramTimeout) as exc:
        wait_for_vram(
            1,
            9000,
            30,
            poll_s=10,
            query=lambda _d: 1234,
            sleep=sleep,
            log=lambda _m: None,
            clock=lambda: clock["t"],
        )
    msg = str(exc.value)
    assert "1" in msg and "9000" in msg and "1234" in msg


def test_zero_timeout_fails_without_sleeping():
    sleeps: list[float] = []
    with pytest.raises(VramTimeout):
        wait_for_vram(0, 100, 0, query=lambda _d: 10, sleep=sleeps.append, log=lambda _m: None)
    assert sleeps == []


def test_logs_once_per_wait():
    logs: list[str] = []
    wait_for_vram(0, 100, 60, query=_seq([10, 20, 30, 200]), sleep=lambda _s: None, log=logs.append)
    assert len(logs) == 1
    assert "100" in logs[0] and "10" in logs[0]


def test_no_log_when_no_wait():
    logs: list[str] = []
    wait_for_vram(0, 100, 60, query=_seq([200]), sleep=lambda _s: None, log=logs.append)
    assert logs == []


def _fake_nvml(count=2, init_error=False):
    m = types.SimpleNamespace()

    def init():
        if init_error:
            raise RuntimeError("no driver")

    m.nvmlInit = init
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetCount = lambda: count

    def handle(i):
        if i >= count:
            raise RuntimeError("bad index")
        return i

    m.nvmlDeviceGetHandleByIndex = handle
    m.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(free=(h + 1) * 1024 * 1024 * 100)
    return m


def test_device_string_cuda(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml())
    assert device_string(Config(gpu_device=1)) == "cuda:1"


def test_device_string_unset_env_uses_nvml_index(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml())
    assert device_string(Config(gpu_device=0)) == "cuda:0"


def test_device_string_maps_through_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml())
    assert device_string(Config(gpu_device=0)) == "cuda:1"
    assert device_string(Config(gpu_device=1)) == "cuda:0"


def test_device_string_card_not_visible(monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml())
    assert device_string(Config(gpu_device=0)) == "cpu"
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


def test_device_string_missing_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(count=1))
    assert device_string(Config(gpu_device=3)) == "cpu"


def test_device_string_init_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml(init_error=True))
    assert device_string(Config()) == "cpu"


def test_device_string_import_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert device_string(Config()) == "cpu"


def test_free_mb_with_fake(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", _fake_nvml())
    assert free_mb(1) == 200


@pytest.mark.gpu
def test_free_mb_real():
    s = device_string(Config())
    if s == "cpu":
        pytest.skip("NVML unavailable")
    assert free_mb(int(s.split(":")[1])) > 0


def test_module_importable_without_pynvml():
    assert gpu.VramTimeout is VramTimeout
