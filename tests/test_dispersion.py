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
