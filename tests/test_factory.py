"""Factory / registry behaviour (no NNP weights required)."""

from __future__ import annotations

import pytest

import ase_calculator_kit as kit
from ase_calculator_kit.registry import BACKENDS, DFT_BACKENDS, MLIP_BACKENDS


def test_available_models():
    assert kit.available_models() == [
        "chgnet",
        "espresso",
        "fairchem",
        "mattersim",
        "qe",
        "quantum-espresso",
        "sevennet",
        "uma",
        "vasp",
    ]
    assert kit.available_mlip_models() == [
        "chgnet",
        "fairchem",
        "mattersim",
        "sevennet",
        "uma",
    ]
    assert kit.available_dft_calculators() == [
        "espresso",
        "qe",
        "quantum-espresso",
        "vasp",
    ]


def test_unknown_model_raises_valueerror():
    with pytest.raises(ValueError) as exc:
        kit.get_calculator("does-not-exist")
    msg = str(exc.value)
    assert "Unknown calculator" in msg
    # Error lists the valid names.
    for name in ("chgnet", "sevennet", "mattersim", "uma"):
        assert name in msg


def test_name_is_case_insensitive(monkeypatch):
    seen = {}

    class FakeBackend:
        def create_calculator(self, *, device="auto", **kwargs):
            seen["device"] = device
            seen["kwargs"] = kwargs
            return "CALC"

    monkeypatch.setitem(BACKENDS, "chgnet", FakeBackend)
    assert kit.get_calculator("ChGNeT", device="cpu", foo=1) == "CALC"
    assert seen == {"device": "cpu", "kwargs": {"foo": 1}}


def test_uma_and_fairchem_share_backend():
    assert BACKENDS["uma"] is BACKENDS["fairchem"]


def test_registry_groups_are_split():
    assert MLIP_BACKENDS["uma"] is BACKENDS["uma"]
    assert DFT_BACKENDS["vasp"] is BACKENDS["vasp"]


def test_get_mlip_calculator_routes_to_mlip_backend(monkeypatch):
    seen = {}

    class FakeBackend:
        def create_calculator(self, **kwargs):
            seen["kwargs"] = kwargs
            return "CALC"

    monkeypatch.setitem(MLIP_BACKENDS, "chgnet", FakeBackend)
    assert kit.get_mlip_calculator("chgnet", device="cpu") == "CALC"
    assert seen["kwargs"] == {"device": "cpu"}


def test_attach_calculator_sets_calc(monkeypatch):
    from ase.build import bulk

    class FakeBackend:
        def create_calculator(self, *, device="auto", **kwargs):
            return "CALC"

    monkeypatch.setitem(BACKENDS, "chgnet", FakeBackend)
    atoms = bulk("Cu", "fcc", a=3.6)
    returned = kit.attach_calculator(atoms, "chgnet")
    assert returned is atoms
    assert atoms.calc == "CALC"
