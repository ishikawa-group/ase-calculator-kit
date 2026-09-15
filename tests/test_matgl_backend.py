"""MatGL model routing, device placement and ASE conventions without weights."""

from __future__ import annotations

import sys
import types

import pytest

from ase_calculator_kit import DispersionError, MissingDependencyError, get_calculator
from ase_calculator_kit.backends.mlip import matgl as backend

MODELS = [
    ("tensornet", "matpes-pbe", "TensorNet-PES-MatPES-PBE-2025.2", "pbe"),
    ("tensornet", "matpes-r2scan", "TensorNet-PES-MatPES-r2SCAN-2025.2", "r2scan"),
]


@pytest.fixture
def fake_matgl(monkeypatch):
    seen = {"operations": []}

    class Potential:
        def float(self):
            seen["operations"].append("float")
            return self

        def to(self, device):
            seen["operations"].append(device)
            return self

        def eval(self):
            seen["operations"].append("eval")
            return self

    class PESCalculator:
        def __init__(self, **kwargs):
            seen["calculator_kwargs"] = kwargs

    potential = Potential()

    def load_model(name, **kwargs):
        seen["load"] = (name, kwargs)
        return potential

    module = types.ModuleType("matgl")
    module.load_model = load_model
    ext = types.ModuleType("matgl.ext")
    ase = types.ModuleType("matgl.ext.ase")
    ase.PESCalculator = PESCalculator
    for name, value in (("matgl", module), ("matgl.ext", ext), ("matgl.ext.ase", ase)):
        monkeypatch.setitem(sys.modules, name, value)
    return seen, potential, PESCalculator


@pytest.mark.parametrize("name,key,full,xc", MODELS)
@pytest.mark.parametrize("spelling", ["short", "suffix", "full"])
def test_model_aliases_and_ase_units(fake_matgl, name, key, full, xc, spelling):
    seen, potential, cls = fake_matgl
    model = {"short": key, "suffix": full.split("-", 1)[1], "full": full}[spelling]
    calc = get_calculator(name.upper(), model=model.upper(), device="cpu", label="run")
    assert isinstance(calc, cls)
    assert seen["load"] == ("materialyze/" + full, {})
    assert seen["operations"] == ["cpu", "eval"]
    assert seen["calculator_kwargs"] == {
        "potential": potential, "stress_unit": "eV/A3", "use_voigt": True, "label": "run",
    }


@pytest.mark.parametrize("name,key,full,xc", MODELS)
def test_mps_casts_before_moving_and_revision_reaches_loader(fake_matgl, name, key, full, xc):
    seen, _, _ = fake_matgl
    get_calculator(name, model=key, device="mps", revision="commit-sha")
    assert seen["operations"] == ["float", "mps", "eval"]
    assert seen["load"] == ("materialyze/" + full, {"revision": "commit-sha"})


@pytest.mark.parametrize("name,key,full,xc", MODELS)
@pytest.mark.parametrize("override", [None, "pbesol"])
def test_d3_uses_resolved_model(fake_matgl, monkeypatch, name, key, full, xc, override):
    seen, _, cls = fake_matgl

    def wrap(bare, **kwargs):
        assert isinstance(bare, cls)
        seen["d3"] = kwargs
        return "SUM"

    monkeypatch.setattr(backend, "wrap_with_d3", wrap)
    assert get_calculator(
        name, model=full, device="mps", dispersion=True, dispersion_xc=override,
        dispersion_damping="zero", dispersion_cutoff=16, dispersion_cutoff_smoothing="none",
    ) == "SUM"
    assert seen["d3"] == {
        "xc": override or xc, "device": "mps", "damping": "zero",
        "cutoff": 16, "cutoff_smoothing": "none",
    }


def test_default_is_pbe(fake_matgl):
    seen, _, _ = fake_matgl
    get_calculator("tensornet", device="cpu")
    assert "-PBE-" in seen["load"][0]


def test_invalid_model_fails_without_download(fake_matgl):
    seen, _, _ = fake_matgl
    for model in ("matpes-typo", "TensorNet-PES-MatPES-PBE-2025.99"):
        with pytest.raises(ValueError, match="Unknown .* MatGL model"):
            get_calculator("tensornet", model=model, device="cpu")
    assert "load" not in seen


def test_wrong_architecture_is_rejected(fake_matgl):
    with pytest.raises(ValueError, match="Unknown tensornet"):
        get_calculator("tensornet", model="M3GNet-PES-MatPES-PBE-2025.2")


@pytest.mark.parametrize("kwargs,exc", [
    ({"dispersion": True, "dispersion_damping": "typo"}, DispersionError),
    ({"dispersion": True, "dispersion_cutoff": -1}, DispersionError),
    ({"stress_unit": "GPa"}, ValueError),
    ({"stress_weight": 0.00624}, ValueError),
    ({"use_voigt": False}, ValueError),
    ({"potential": object()}, TypeError),
])
def test_bad_settings_fail_before_loading(fake_matgl, kwargs, exc):
    seen, _, _ = fake_matgl
    with pytest.raises(exc):
        get_calculator("tensornet", device="cpu", **kwargs)
    assert "load" not in seen


def test_missing_matgl_names_correct_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "matgl", None)
    with pytest.raises(MissingDependencyError, match=r"ase-calculator-kit\[matgl\]"):
        get_calculator("tensornet", model="matpes-pbe", device="cpu")
