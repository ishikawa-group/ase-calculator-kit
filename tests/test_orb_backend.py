"""OrbMol backend routing and guard behavior (no real checkpoints loaded)."""

from __future__ import annotations

import sys
import types

import pytest

from ase_calculator_kit import DispersionError, MissingDependencyError, get_calculator


@pytest.fixture
def fake_orb(monkeypatch):
    seen: dict = {}

    class ORBCalculator:
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, model, *, atoms_adapter, device=None, **kwargs):
            seen["model"] = model
            seen["adapter"] = atoms_adapter
            seen["calculator_device"] = device
            seen["calculator_kwargs"] = kwargs

    def orbmol_v2(device=None, precision=None, compile=None, **kwargs):
        seen["load"] = {
            "device": device, "precision": precision, "compile": compile, **kwargs
        }
        return "potential", "adapter"

    orb_models = types.ModuleType("orb_models")
    forcefield = types.ModuleType("orb_models.forcefield")
    pretrained = types.ModuleType("orb_models.forcefield.pretrained")
    pretrained.orbmol_v2 = orbmol_v2
    inference = types.ModuleType("orb_models.forcefield.inference")
    calculator = types.ModuleType("orb_models.forcefield.inference.calculator")
    calculator.ORBCalculator = ORBCalculator

    for name, module in (
        ("orb_models", orb_models),
        ("orb_models.forcefield", forcefield),
        ("orb_models.forcefield.pretrained", pretrained),
        ("orb_models.forcefield.inference", inference),
        ("orb_models.forcefield.inference.calculator", calculator),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return seen, ORBCalculator


def test_default_is_orbmol_v2_on_the_resolved_device(fake_orb):
    seen, cls = fake_orb

    calc = get_calculator("orb", device="cpu")

    assert isinstance(calc, cls)
    assert seen["load"] == {
        "device": "cpu", "precision": "float32-high", "compile": None
    }
    assert seen["calculator_device"] == "cpu"
    assert seen["calculator_kwargs"] == {}


@pytest.mark.parametrize("spelling", ["orbmol-v2", "orbmol_v2", "OrbMol-v2", " orbmol-v2 "])
def test_both_spellings_of_the_model_name_are_accepted(fake_orb, spelling):
    seen, _ = fake_orb

    get_calculator("orb", device="cpu", model=spelling)

    assert seen["load"]["device"] == "cpu"


def test_another_orb_checkpoint_is_refused_by_name(fake_orb):
    """The rest of orb-models' registry is not wired up, and says so.

    Passing it through to upstream would load a model at a different level of
    theory than the one `dispersion.py` documents for this backend.
    """
    with pytest.raises(ValueError, match="orb-v3-conservative-inf-omat"):
        get_calculator("orb", device="cpu", model="orb-v3-conservative-inf-omat")


def test_precision_is_validated_before_the_download(fake_orb):
    seen, _ = fake_orb

    with pytest.raises(ValueError, match="float32-high"):
        get_calculator("orb", device="cpu", precision="float16")
    assert "load" not in seen


def test_precision_and_compile_reach_the_loader(fake_orb):
    seen, _ = fake_orb

    get_calculator("orb", device="cpu", precision="float64", compile=False)

    assert seen["load"]["precision"] == "float64"
    assert seen["load"]["compile"] is False


def test_extra_keywords_go_to_the_calculator_not_the_loader(fake_orb):
    seen, _ = fake_orb

    get_calculator("orb", device="cpu", max_num_neighbors=40, edge_method="knn_scipy")

    assert seen["calculator_kwargs"] == {
        "max_num_neighbors": 40, "edge_method": "knn_scipy"
    }
    assert "max_num_neighbors" not in seen["load"]


def test_orb_rejects_mps():
    """Graph construction runs through NVIDIA Warp, which has no Metal backend.

    Measured on Apple Silicon: `knn_alchemi` raises `Unsupported Torch device
    type mps`, `knn_brute_force` needs float64 that MPS cannot hold, and
    `knn_scipy` refuses a non-CPU device itself. So the refusal happens here,
    before orb-models is even imported.
    """
    with pytest.raises(ValueError, match="MPS-validated"):
        get_calculator("orb", device="mps")


def test_dispersion_is_always_refused(fake_orb):
    seen, _ = fake_orb

    with pytest.raises(DispersionError, match="VV10"):
        get_calculator("orb", device="cpu", dispersion=True)
    assert "load" not in seen


def test_an_explicit_xc_cannot_re_enable_dispersion(fake_orb):
    with pytest.raises(DispersionError, match="VV10"):
        get_calculator("orb", device="cpu", dispersion=True, dispersion_xc="pbe")


def test_missing_orb_models_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "orb_models", None)

    with pytest.raises(MissingDependencyError, match=r"ase-calculator-kit\[orb\]"):
        get_calculator("orb", device="cpu")


def test_an_orb_models_without_orbmol_v2_says_which_version_added_it(monkeypatch):
    """orb-models 0.6 imports fine but has no OrbMol-v2 loader."""
    orb_models = types.ModuleType("orb_models")
    monkeypatch.setitem(sys.modules, "orb_models", orb_models)
    for name in (
        "orb_models.forcefield.pretrained",
        "orb_models.forcefield.inference.calculator",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ImportError, match="orb-models 0.7.0"):
        get_calculator("orb", device="cpu")
