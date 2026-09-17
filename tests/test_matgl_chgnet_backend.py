"""The provisional correction must be scoped and fail on unfamiliar source."""

from __future__ import annotations

import hashlib
import inspect
import sys
import textwrap
import types
from contextlib import contextmanager

import pytest

from ase_calculator_kit import get_calculator
from ase_calculator_kit.backends.mlip import matgl_chgnet as backend


class TorchStub:
    enabled = True

    @contextmanager
    def no_grad(self):
        old = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = old


torch = TorchStub()


def create_directed_line_graph(value):
    return value, torch.enabled


class CHGNet:
    def forward(self, value):
        with torch.no_grad():
            result = create_directed_line_graph(value)
        return result


@pytest.fixture
def fake_matgl(monkeypatch):
    digest = hashlib.sha256(textwrap.dedent(inspect.getsource(CHGNet.forward)).encode()).hexdigest()
    monkeypatch.setattr(backend, "_FORWARD_SHA256", digest)
    monkeypatch.setattr(backend, "version", lambda _: "4.0.3")
    seen = {}

    class Potential:
        def __init__(self):
            self.model = CHGNet()

        def float(self):
            seen["cast"] = True
            return self

        def to(self, device):
            seen["device"] = device
            return self

        def eval(self):
            return self

    class PESCalculator:
        def __init__(self, **kwargs):
            seen["calculator"] = kwargs
            self.parameters = {}
            self._atoms2graph = object()

    def load(name, **kwargs):
        seen["load"] = (name, kwargs)
        return Potential()

    for name in ["matgl", "matgl.ext", "matgl.ext.ase", "matgl.models", "matgl.models._chgnet"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["matgl"].load_model = load
    sys.modules["matgl.ext.ase"].PESCalculator = PESCalculator
    sys.modules["matgl.models._chgnet"].CHGNet = CHGNet
    return seen


@pytest.mark.parametrize("xc", ["pbe", "r2scan"])
@pytest.mark.parametrize("model_form", ["short", "full", "suffix"])
def test_routing_pinned_weights_and_instance_isolation(fake_matgl, xc, model_form):
    from ase_calculator_kit.backends.mlip.matgl import MATPES_MODELS

    key = f"matpes-{xc}"
    full = MATPES_MODELS["matgl-chgnet"][key]
    model = {"short": key, "full": full, "suffix": full.split("-", 1)[1]}[model_form]
    with pytest.warns(UserWarning, match="provisional"):
        calc = get_calculator("matgl-chgnet", model=model, device="cpu")
    assert fake_matgl["load"] == ("materialyze/" + full, {"revision": backend._REVISIONS[key]})
    assert fake_matgl["calculator"]["stress_unit"] == "eV/A3"
    assert fake_matgl["calculator"]["use_voigt"] is True
    assert fake_matgl["calculator"]["potential"].model.forward(3) == (3, True)
    assert CHGNet().forward(3) == (3, False)
    assert torch.enabled is True
    assert calc.parameters["revision"] == backend._REVISIONS[key]


@pytest.mark.parametrize("xc", ["pbe", "r2scan"])
def test_d3_and_mps_and_explicit_revision(fake_matgl, monkeypatch, xc):
    def wrap(bare, **kwargs):
        fake_matgl["d3"] = kwargs
        return bare

    monkeypatch.setattr(backend, "wrap_with_d3", wrap)
    with pytest.warns(UserWarning, match="not retrained"):
        get_calculator("matgl-chgnet", model=f"matpes-{xc}", device="mps",
                       revision="explicit-sha", dispersion=True)
    assert fake_matgl["load"][1] == {"revision": "explicit-sha"}
    assert fake_matgl["cast"] is True
    assert fake_matgl["d3"]["xc"] == xc
    assert fake_matgl["d3"]["device"] == "mps"


@pytest.mark.parametrize("failure", ["source", "version"])
def test_unrecognized_implementation_fails_before_download(fake_matgl, monkeypatch, failure):
    if failure == "source":
        monkeypatch.setattr(backend, "_FORWARD_SHA256", "changed")
    else:
        monkeypatch.setattr(backend, "version", lambda _: "4.0.4")
    with pytest.raises(RuntimeError, match="4.0.3"):
        get_calculator("matgl-chgnet", device="cpu")
    assert "load" not in fake_matgl


@pytest.mark.parametrize("kwargs", [
    {"model": "TensorNet-PES-MatPES-PBE-2025.2"}, {"stress_unit": "GPa"},
    {"stress_weight": 0.1}, {"use_voigt": False}, {"potential": object()},
])
def test_invalid_selectors_and_units_fail_before_download(fake_matgl, kwargs):
    with pytest.raises((ValueError, TypeError)):
        get_calculator("matgl-chgnet", device="cpu", **kwargs)
    assert "load" not in fake_matgl
