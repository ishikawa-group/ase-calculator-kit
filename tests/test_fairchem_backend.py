"""UMA / fairchem backend guard behavior (no real checkpoints loaded)."""

from __future__ import annotations

import pytest

from ase_calculator_kit import get_calculator


def test_uma_rejects_mps():
    # fairchem-core only supports cpu/cuda, so the kit refuses device="mps"
    # before importing fairchem.
    with pytest.raises(ValueError, match="MPS-validated"):
        get_calculator("uma", device="mps")
