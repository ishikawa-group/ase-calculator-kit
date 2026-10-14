"""Opt-in real nonperiodic H2 calculation; binary locations are test fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from ase_calculator_kit import get_calculator


@pytest.mark.slow
def test_cp2k_h2_singlepoint(tmp_path):
    command = os.environ.get("ASE_CALCULATOR_KIT_TEST_CP2K_COMMAND")
    data_dir = os.environ.get("ASE_CALCULATOR_KIT_TEST_CP2K_DATA_DIR")
    if not command or not data_dir:
        pytest.skip("Set ASE_CALCULATOR_KIT_TEST_CP2K_COMMAND and ..._DATA_DIR to run CP2K.")
    config = Path(__file__).resolve().parents[1] / "examples/dft/cp2k_pbe_h2.yaml"
    atoms = Atoms("H2", positions=[[0, 0, -.37], [0, 0, .37]], cell=[12, 12, 12], pbc=False)
    atoms.center()
    with get_calculator("cp2k", config=config, overrides={
        "profile": {"command": command, "data_dir": data_dir}, "directory": str(tmp_path),
    }) as calc:
        atoms.calc = calc
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()
        assert atoms.get_forces().shape == (2, 3)
        assert atoms.get_potential_energy(force_consistent=True) == atoms.get_potential_energy()
