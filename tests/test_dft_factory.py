"""DFT factory routing and config-only API behaviour."""

from __future__ import annotations

import pytest

import ase_calculator_kit as kit
from ase_calculator_kit.registry import BACKENDS, DFT_BACKENDS


@pytest.mark.parametrize("name", ["vasp", "qe"])
def test_dft_requires_config(name):
    with pytest.raises(TypeError, match="requires config"):
        kit.get_calculator(name)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("vasp", {"encut": 520}),
        ("qe", {"pseudo_dir": "/path/to/pseudos"}),
    ],
)
def test_dft_rejects_unexpected_kwargs(name, kwargs):
    with pytest.raises(TypeError, match="accepts only config"):
        kit.get_calculator(name, **kwargs)


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
