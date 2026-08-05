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


def test_sevennet_debug_prints_are_suppressed(monkeypatch, capsys):
    class NoisySevenNetCalculator:
        def __init__(self, **kwargs):
            # Mirrors sevenn 0.12.1 calculator.py.
            print("cueq")
            print(False)
            print("flash")
            print(False)
            print("Loading 7net-omni")

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = NoisySevenNetCalculator

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)

    get_calculator("sevennet", device="cpu")

    assert capsys.readouterr().out == "Loading 7net-omni\n"


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
