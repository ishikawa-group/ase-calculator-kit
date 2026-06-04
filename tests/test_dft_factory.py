"""DFT factory routing and config-only API behaviour."""

from __future__ import annotations

import pytest

import ase_calculator_kit as kit
from ase_calculator_kit.registry import BACKENDS, DFT_BACKENDS


def test_dft_requires_config():
    with pytest.raises(TypeError, match="requires config"):
        kit.get_calculator("vasp")


def test_dft_rejects_unexpected_kwargs():
    with pytest.raises(TypeError, match="Unexpected arguments: \\['encut'\\]"):
        kit.get_calculator("vasp", encut=520)


def test_get_dft_calculator_rejects_mlip_name():
    with pytest.raises(ValueError, match="Unknown DFT calculator"):
        kit.get_dft_calculator("chgnet")


@pytest.mark.parametrize("name", ["vasp", "qe", "espresso", "quantum-espresso"])
def test_dft_aliases_route_to_backend(monkeypatch, name):
    seen = {}

    class FakeBackend:
        def create_calculator(self, **kwargs):
            seen["kwargs"] = kwargs
            return "CALC"

    monkeypatch.setitem(DFT_BACKENDS, name, FakeBackend)
    monkeypatch.setitem(BACKENDS, name, FakeBackend)

    assert kit.get_calculator(name, config={"calculator": name}) == "CALC"
    assert seen["kwargs"] == {"config": {"calculator": name}}


def test_compatibility_import_exposes_new_api():
    import ase_umlip_kit as old_kit

    assert old_kit.get_calculator is kit.get_calculator
    assert old_kit.available_dft_calculators is kit.available_dft_calculators
