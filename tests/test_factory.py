"""Factory / registry behaviour (no NNP weights required)."""

from __future__ import annotations

import pytest

import ase_umlip_kit as kit
from ase_umlip_kit.registry import BACKENDS


def test_available_models():
    assert kit.available_models() == ["chgnet", "fairchem", "mattersim", "sevennet", "uma"]


def test_unknown_model_raises_valueerror():
    with pytest.raises(ValueError) as exc:
        kit.get_calculator("does-not-exist")
    msg = str(exc.value)
    assert "Unknown MLIP" in msg
    # Error lists the valid names.
    for name in ("chgnet", "sevennet", "mattersim", "uma"):
        assert name in msg


def test_name_is_case_insensitive(monkeypatch):
    seen = {}

    class FakeBackend:
        def create_calculator(self, *, device, **kwargs):
            seen["device"] = device
            seen["kwargs"] = kwargs
            return "CALC"

    monkeypatch.setitem(BACKENDS, "chgnet", FakeBackend)
    assert kit.get_calculator("ChGNeT", device="cpu", foo=1) == "CALC"
    assert seen == {"device": "cpu", "kwargs": {"foo": 1}}


def test_uma_and_fairchem_share_backend():
    assert BACKENDS["uma"] is BACKENDS["fairchem"]


def test_aliases_point_to_get_calculator():
    assert kit.build_calculator is kit.get_calculator
    assert kit.utils_uMLIP_calculator is kit.get_calculator


def test_attach_calculator_sets_calc(monkeypatch):
    from ase.build import bulk

    class FakeBackend:
        def create_calculator(self, *, device, **kwargs):
            return "CALC"

    monkeypatch.setitem(BACKENDS, "chgnet", FakeBackend)
    atoms = bulk("Cu", "fcc", a=3.6)
    returned = kit.attach_calculator(atoms, "chgnet")
    assert returned is atoms
    assert atoms.calc == "CALC"
