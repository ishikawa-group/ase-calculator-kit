"""MatterSim backend construction without loading real checkpoints."""

from __future__ import annotations

import sys
import types

from ase_calculator_kit import get_calculator


def test_mattersim_backend_allows_mps(monkeypatch):
    seen = {}

    class FakeMatterSimCalculator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    mattersim = types.ModuleType("mattersim")
    forcefield = types.ModuleType("mattersim.forcefield")
    forcefield.MatterSimCalculator = FakeMatterSimCalculator

    monkeypatch.setitem(sys.modules, "mattersim", mattersim)
    monkeypatch.setitem(sys.modules, "mattersim.forcefield", forcefield)

    calc = get_calculator("mattersim", model="1M", device="mps")

    assert isinstance(calc, FakeMatterSimCalculator)
    assert seen["kwargs"] == {"device": "mps"}
