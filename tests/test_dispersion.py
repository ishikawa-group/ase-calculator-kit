"""Dispersion (D3) policy logic. Fast: no model weights or torch-dftd needed."""

from __future__ import annotations

import sys

import pytest

from ase_calculator_kit import DispersionError, get_calculator, get_dispersion_policy
from ase_calculator_kit.dispersion import (
    precheck_dispersion_xc,
    resolve_dispersion_xc,
    wrap_with_d3,
)


@pytest.mark.parametrize(
    "backend,key,expected_xc",
    [
        ("chgnet", "default", "pbe"),
        ("chgnet", "0.3.0", "pbe"),
        ("chgnet", "0.2.0", "pbe"),
        ("chgnet", "r2scan", "r2scan"),
        ("mattersim", "1M", "pbe"),
        ("mattersim", "5M", "pbe"),
        ("sevennet", "mpa", "pbe"),
        ("sevennet", "omat24", "pbe"),
        ("sevennet", "matpes_pbe", "pbe"),
        ("sevennet", "oc20", "rpbe"),
        ("sevennet", "oc22", "pbe"),
        ("sevennet", "matpes_r2scan", "r2scan"),
        ("sevennet", "mp_r2scan", "r2scan"),
        ("sevennet", "pet_mad", "pbesol"),
        ("sevennet", "default", "pbe"),
        ("nequip", "S", "pbe"),
        ("nequip", "M", "pbe"),
        ("nequip", "L", "pbe"),
        ("nequip", "XL", "pbe"),
        ("uma", "omat", "pbe"),
        ("uma", "oc20", "rpbe"),
        ("uma", "oc22", "pbe"),
    ],
)
def test_allowed_returns_default_xc(backend, key, expected_xc):
    assert resolve_dispersion_xc(backend, key, dispersion_xc=None) == expected_xc


@pytest.mark.parametrize(
    "backend,key",
    [
        ("uma", "oc25"),
        ("uma", "omol"),
        ("uma", "odac"),
        ("uma", "omc"),
        ("sevennet", "omol25_low"),
        ("sevennet", "omol25_high"),
        ("sevennet", "spice"),
        ("sevennet", "qcml"),
        ("sevennet", "odac23"),
    ],
)
def test_included_models_always_error(backend, key):
    # Even an explicit dispersion_xc cannot override an already-dispersive model.
    with pytest.raises(DispersionError, match="double-counting"):
        resolve_dispersion_xc(backend, key, dispersion_xc=None)
    with pytest.raises(DispersionError):
        resolve_dispersion_xc(backend, key, dispersion_xc="pbe")


def test_unlisted_modal_stays_unverified():
    # The escape hatch still exists for tasks this table does not cover yet.
    with pytest.raises(DispersionError, match="not verified"):
        resolve_dispersion_xc("sevennet", "some_future_task", dispersion_xc=None)
    assert (
        resolve_dispersion_xc("sevennet", "some_future_task", dispersion_xc="pbe")
        == "pbe"
    )


def test_precheck_returns_none_when_disabled():
    assert (
        precheck_dispersion_xc("uma", "oc25", dispersion=False, dispersion_xc=None)
        is None
    )


def test_policy_exposes_training_reference_for_researchers():
    pbe = get_dispersion_policy("sevennet", "matpes_pbe")
    assert pbe is not None
    assert pbe.reference_level == "PBE"
    assert pbe.d3_xc == "pbe"
    assert pbe.includes_dispersion is False

    molecular = get_dispersion_policy("sevennet", "omol25_low")
    assert molecular is not None
    assert molecular.reference_level == "ωB97M-V"
    assert molecular.includes_dispersion is True


@pytest.mark.parametrize(
    "backend,key",
    [
        ("uma", "omol"),
        ("uma", "omc"),
        ("uma", "odac"),
        ("sevennet", "omol25_low"),
        ("sevennet", "omol25_high"),
        ("sevennet", "spice"),
        ("sevennet", "qcml"),
    ],
)
def test_molecular_references_are_all_verified(backend, key):
    """Every molecular task has a documented functional, not an unverified one.

    Molecular reference data is the easiest place to double-count dispersion,
    so leaving one of these on the overridable "unverified" tier would let
    ``dispersion_xc=...`` silently add D3 on top of a dispersive functional.
    """
    policy = get_dispersion_policy(backend, key)
    assert policy is not None
    assert policy.includes_dispersion is True
    assert policy.note


def test_get_calculator_included_task_fails_fast(monkeypatch):
    # Policy is checked before the (heavy) predictor is built, so this raises
    # without any model download.
    with pytest.raises(DispersionError):
        get_calculator("uma", task="oc25", device="cpu", dispersion=True)


def test_wrap_with_d3_missing_torch_dftd(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_dftd", None)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", None)
    with pytest.raises(DispersionError, match="torch-dftd"):
        wrap_with_d3(object(), xc="pbe", device="cpu")


def test_damping_defaults_to_bj():
    from ase_calculator_kit.dispersion import resolve_dispersion_damping

    assert resolve_dispersion_damping(None) == "bj"


def test_zero_damping_is_accepted():
    from ase_calculator_kit.dispersion import resolve_dispersion_damping

    assert resolve_dispersion_damping("zero") == "zero"


def test_unknown_damping_is_refused():
    """A typo must not silently fall back to a parameter set nobody fitted."""
    from ase_calculator_kit.dispersion import resolve_dispersion_damping
    from ase_calculator_kit.errors import DispersionError

    with pytest.raises(DispersionError, match="dispersion_damping"):
        resolve_dispersion_damping("bjm")


def test_damping_is_validated_before_the_model_is_built():
    """The check belongs in the precheck: loading a checkpoint costs GBs."""
    from ase_calculator_kit.dispersion import precheck_dispersion_xc
    from ase_calculator_kit.errors import DispersionError

    with pytest.raises(DispersionError, match="dispersion_damping"):
        precheck_dispersion_xc(
            "sevennet", "mpa",
            dispersion=True, dispersion_xc=None, dispersion_damping="nope",
        )


def test_damping_is_ignored_when_dispersion_is_off():
    from ase_calculator_kit.dispersion import precheck_dispersion_xc

    assert precheck_dispersion_xc(
        "sevennet", "mpa",
        dispersion=False, dispersion_xc=None, dispersion_damping="nope",
    ) is None


def test_damping_reaches_torch_dftd(monkeypatch):
    """get_calculator(..., dispersion_damping="zero") must land on the D3 term."""
    import sys
    import types

    seen = {}

    class FakeD3:
        # SumCalculator intersects implemented_properties across its members.
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            seen.update(kwargs)

    class FakeSevenNetCalculator:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            pass

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator
    torch_dftd = types.ModuleType("torch_dftd")
    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeD3

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)
    monkeypatch.setitem(sys.modules, "torch_dftd", torch_dftd)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    get_calculator(
        "sevennet", modal="oc20", device="cpu",
        dispersion=True, dispersion_damping="zero",
    )

    assert seen["damping"] == "zero"
    assert seen["xc"] == "rpbe"


def _fake_d3_kwargs(monkeypatch, **call_kwargs):
    """Build a SevenNet calculator against fakes and return the D3 kwargs."""
    import sys
    import types

    seen = {}

    class FakeD3:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            seen.update(kwargs)

    class FakeSevenNetCalculator:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            pass

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator
    torch_dftd = types.ModuleType("torch_dftd")
    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeD3

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)
    monkeypatch.setitem(sys.modules, "torch_dftd", torch_dftd)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    get_calculator("sevennet", device="cpu", dispersion=True, **call_kwargs)
    return seen


def test_d3_numerics_default_to_pfp_not_torch_dftd(monkeypatch):
    """The whole point of 0.5.3: an unqualified dispersion=True must be PFP's.

    torch-dftd's own defaults are 95 Bohr (50.3 A) and no smoothing. Comparing
    an NNP+D3 against PFP+D3 with those in place compares two different
    dispersion terms on top of the model difference we mean to measure.
    """
    seen = _fake_d3_kwargs(monkeypatch)

    assert seen["cutoff"] == 14.0
    assert seen["cutoff_smoothing"] == "poly"


def test_torch_dftd_defaults_can_be_restored(monkeypatch):
    """Reproducing a dataset built on torch-dftd's defaults stays possible."""
    seen = _fake_d3_kwargs(
        monkeypatch, dispersion_cutoff=50.3, dispersion_cutoff_smoothing="none",
    )

    assert seen["cutoff"] == 50.3
    assert seen["cutoff_smoothing"] == "none"


def test_cutoff_defaults_to_pfps_14_angstrom():
    from ase_calculator_kit.dispersion import resolve_dispersion_cutoff

    assert resolve_dispersion_cutoff(None) == 14.0


def test_cutoff_smoothing_defaults_to_poly():
    from ase_calculator_kit.dispersion import resolve_dispersion_cutoff_smoothing

    assert resolve_dispersion_cutoff_smoothing(None) == "poly"


def test_cutoff_smoothing_none_is_accepted():
    """PFP v6.0.0 and earlier, and torch-dftd's own default."""
    from ase_calculator_kit.dispersion import resolve_dispersion_cutoff_smoothing

    assert resolve_dispersion_cutoff_smoothing("none") == "none"


def test_unknown_cutoff_smoothing_is_refused():
    from ase_calculator_kit.dispersion import resolve_dispersion_cutoff_smoothing

    with pytest.raises(DispersionError, match="dispersion_cutoff_smoothing"):
        resolve_dispersion_cutoff_smoothing("polynomial")


@pytest.mark.parametrize("bad", [0, -1.0, "far"])
def test_a_cutoff_that_is_not_a_positive_distance_is_refused(bad):
    from ase_calculator_kit.dispersion import resolve_dispersion_cutoff

    with pytest.raises(DispersionError, match="dispersion_cutoff"):
        resolve_dispersion_cutoff(bad)


def test_cutoff_is_validated_before_the_model_is_built():
    with pytest.raises(DispersionError, match="dispersion_cutoff"):
        precheck_dispersion_xc(
            "sevennet", "mpa",
            dispersion=True, dispersion_xc=None, dispersion_cutoff=-1,
        )


def test_cutoff_smoothing_is_validated_before_the_model_is_built():
    with pytest.raises(DispersionError, match="dispersion_cutoff_smoothing"):
        precheck_dispersion_xc(
            "sevennet", "mpa",
            dispersion=True, dispersion_xc=None,
            dispersion_cutoff_smoothing="polynomial",
        )
