"""SevenNet backend construction without loading real checkpoints."""

from __future__ import annotations

import sys
import types

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
