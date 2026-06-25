"""Dispersion (D3) policy logic. Fast: no model weights or torch-dftd needed."""

from __future__ import annotations

import sys

import pytest

from ase_calculator_kit import DispersionError, get_calculator
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
        ("sevennet", "omol25_low"),
        ("sevennet", "omol25_high"),
    ],
)
def test_included_models_always_error(backend, key):
    # Even an explicit dispersion_xc cannot override an already-dispersive model.
    with pytest.raises(DispersionError, match="double-counting"):
        resolve_dispersion_xc(backend, key, dispersion_xc=None)
    with pytest.raises(DispersionError):
        resolve_dispersion_xc(backend, key, dispersion_xc="pbe")


@pytest.mark.parametrize(
    "backend,key",
    [
        ("uma", "odac"),
        ("uma", "omc"),
    ],
)
def test_unverified_requires_explicit_xc(backend, key):
    with pytest.raises(DispersionError, match="not verified"):
        resolve_dispersion_xc(backend, key, dispersion_xc=None)
    # Explicit override unlocks it.
    assert resolve_dispersion_xc(backend, key, dispersion_xc="pbe") == "pbe"


def test_precheck_returns_none_when_disabled():
    assert (
        precheck_dispersion_xc("uma", "oc25", dispersion=False, dispersion_xc=None)
        is None
    )


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
