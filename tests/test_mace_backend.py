"""MACE backend construction without installing mace-torch or loading weights.

MACE cannot share an environment with the other backends (e3nn 0.4.4 vs >=0.5),
so these tests inject a fake ``mace.calculators`` module exactly as the other
backend tests do. They are the only coverage that runs in CI, where mace-torch
is not installed at all.
"""

from __future__ import annotations

import sys
import types

import pytest

from ase_calculator_kit import DispersionError, get_calculator
from ase_calculator_kit.backends.mlip.mace import MH1_HEADS
from ase_calculator_kit.dispersion import _POLICIES
from ase_calculator_kit.errors import MissingDependencyError


class FakeMACECalculator:
    implemented_properties = ["energy"]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # MACECalculator exposes the checkpoint's heads under this name.
        self.available_heads = list(MH1_HEADS)


def _install_fake_mace(monkeypatch, seen: dict, *, available_heads=None):
    def fake_mace_mp(**kwargs):
        seen["kwargs"] = kwargs
        calc = FakeMACECalculator(**kwargs)
        if available_heads is not None:
            calc.available_heads = list(available_heads)
        return calc

    mace = types.ModuleType("mace")
    calculators = types.ModuleType("mace.calculators")
    calculators.mace_mp = fake_mace_mp
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)


def test_defaults_are_mh1_omat_pbe_float64(monkeypatch):
    """The documented default: MACE-MH-1 with its PBE materials head."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    calc = get_calculator("mace", device="cpu")

    assert isinstance(calc, FakeMACECalculator)
    assert seen["kwargs"] == {
        "model": "mh-1",
        "device": "cpu",
        "default_dtype": "float64",
        "head": "omat_pbe",
    }


def test_extra_keywords_reach_mace(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    get_calculator(
        "mace", device="cpu", head="matpes_r2scan",
        default_dtype="float32", enable_cueq=True,
    )

    assert seen["kwargs"]["head"] == "matpes_r2scan"
    assert seen["kwargs"]["default_dtype"] == "float32"
    assert seen["kwargs"]["enable_cueq"] is True


def test_unknown_head_is_refused_before_the_model_is_loaded(monkeypatch):
    """MACE answers an unknown head with a *warning* and the wrong head.

    ``MACECalculator`` logs "Head <x> not found ... defaulting to the last head"
    and returns energies from that head, so a typo produces a plausible number
    at a different level of theory. Catch it here, and before the download.
    """
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    with pytest.raises(ValueError) as exc:
        get_calculator("mace", device="cpu", head="rgd1_b3lyp")

    message = str(exc.value)
    assert "rgd1_b3lyp" in message
    assert "omat_pbe" in message
    assert seen == {}, "the checkpoint must not be loaded to reject a head"


def test_head_none_lets_mace_pick_a_single_head_checkpoint(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["Default"])

    get_calculator("mace", device="cpu", model="medium-omat-0", head=None)

    assert "head" not in seen["kwargs"]


def test_head_missing_from_an_unlisted_checkpoint_is_refused(monkeypatch):
    """The safety net for checkpoints whose heads this package cannot know."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["a_head", "another"])

    with pytest.raises(ValueError, match="a_head"):
        get_calculator("mace", device="cpu", model="mh-0", head="omat_pbe")


def test_mps_is_refused(monkeypatch):
    """Measured: torch.load(map_location='mps') on MH-1 fails at float64."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    with pytest.raises(ValueError, match="mps"):
        get_calculator("mace", device="mps")


def test_dispersion_is_refused_for_the_molecular_heads(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    for head in ("omol", "spice_wB97M"):
        with pytest.raises(DispersionError, match="double-counting"):
            get_calculator("mace", device="cpu", head=head, dispersion=True)
    assert seen == {}, "the policy is checked before the checkpoint is loaded"


def test_dispersion_wraps_the_pbe_head(monkeypatch):
    seen: dict = {}
    d3_kwargs: dict = {}
    _install_fake_mace(monkeypatch, seen)

    class FakeD3:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            d3_kwargs.update(kwargs)

    torch_dftd = types.ModuleType("torch_dftd")
    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeD3
    monkeypatch.setitem(sys.modules, "torch_dftd", torch_dftd)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    from ase.calculators.mixing import SumCalculator

    calc = get_calculator("mace", device="cpu", dispersion=True)

    assert isinstance(calc, SumCalculator)
    assert d3_kwargs["xc"] == "pbe"
    assert d3_kwargs["damping"] == "bj"


def test_missing_mace_names_the_separate_environment(monkeypatch):
    """The install hint has to mention the second environment, not just the extra."""
    monkeypatch.setitem(sys.modules, "mace", None)
    monkeypatch.setitem(sys.modules, "mace.calculators", None)

    with pytest.raises(MissingDependencyError) as exc:
        get_calculator("mace", device="cpu")

    message = str(exc.value)
    assert "ase-calculator-kit[mace]" in message
    assert "e3nn" in message
    assert "virtual environment" in message


@pytest.mark.parametrize("head", MH1_HEADS)
def test_every_mh1_head_has_a_dispersion_policy(head):
    """A head with no policy row would be refused as "unverified" at runtime."""
    assert ("mace", head) in _POLICIES
