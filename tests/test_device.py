"""Device resolution logic, with torch mocked so no GPU is required."""

from __future__ import annotations

import sys
import types

import pytest

from ase_umlip_kit.device import resolve_device


def _fake_torch(*, cuda: bool, mps_built: bool = False, mps_avail: bool = False):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(
            is_built=lambda: mps_built,
            is_available=lambda: mps_avail,
        )
    )
    return torch


@pytest.mark.parametrize("explicit", ["cuda", "cpu", "mps"])
def test_explicit_device_passthrough(explicit):
    assert resolve_device(explicit) == explicit


def test_auto_prefers_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert resolve_device("auto") == "cuda"
    assert resolve_device("auto", allow_mps=True) == "cuda"


def test_auto_uses_mps_only_when_allowed(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda=False, mps_built=True, mps_avail=True)
    )
    # CHGNet (allow_mps=True) gets mps; others fall back to cpu.
    assert resolve_device("auto", allow_mps=True) == "mps"
    assert resolve_device("auto") == "cpu"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert resolve_device("auto", allow_mps=True) == "cpu"


def test_auto_without_torch_returns_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on import
    assert resolve_device("auto") == "cpu"
