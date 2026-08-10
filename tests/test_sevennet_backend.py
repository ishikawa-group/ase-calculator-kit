"""SevenNet backend construction without loading real checkpoints."""

from __future__ import annotations

import sys
import types

import pytest

from ase_calculator_kit import get_calculator


def test_sevennet_backend_allows_mps(monkeypatch):
    seen = {}

    class FakeSevenNetCalculator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)

    calc = get_calculator("sevennet", modal="mpa", device="mps")

    assert isinstance(calc, FakeSevenNetCalculator)
    assert seen["kwargs"]["device"] == "mps"


@pytest.mark.parametrize("model", ["7net-omni", "7net-omni-i8", "7net-omni-i12"])
def test_every_omni_capacity_is_selectable_with_a_modal(monkeypatch, model):
    """i8 and i12 are the same recipe at larger capacity, so `modal` still applies."""
    seen = {}

    class FakeSevenNetCalculator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)

    get_calculator("sevennet", model=model, modal="matpes_r2scan", device="cpu")

    assert seen["kwargs"]["model"] == model
    assert seen["kwargs"]["modal"] == "matpes_r2scan"


def test_molecular_modal_is_forwarded_without_charge_or_spin(monkeypatch):
    """sevenn has no charge/spin input: the modal is the only molecular handle.

    Guards against "helpfully" inventing a charge=/spin= keyword here — it would
    have to be dropped on the floor, which is worse than not offering it.
    """
    seen = {}

    class FakeSevenNetCalculator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)

    get_calculator("sevennet", modal="omol25_high", device="cpu")

    assert seen["kwargs"]["modal"] == "omol25_high"
    assert "charge" not in seen["kwargs"]
    assert "spin" not in seen["kwargs"]
