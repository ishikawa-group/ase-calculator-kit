"""Fast unit tests for PySCF 0.6.0 advanced features:

SCF control, checkpoints, diagnostics, PCM radii, D3zero, and Hessian API.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Hartree

from ase_calculator_kit.backends.dft._pyscf_support import (
    CheckpointManager,
    DiagnosticsCollector,
    build_pcm_radii_table,
    format_hessian,
    validate_pyscf_parameters,
)
from ase_calculator_kit.backends.dft.pyscf import PySCFCalculator


BASE_CONFIG = {"basis": "sto-3g", "charge": 0, "multiplicity": 1}


def test_parameter_validation_scf_and_diis():
    # Valid algorithms and init_guess
    p1 = validate_pyscf_parameters(BASE_CONFIG | {"scf_algorithm": "cdiis", "init_guess": "atom"})
    assert p1["scf_algorithm"] == "cdiis"
    assert p1["init_guess"] == "atom"

    p2 = validate_pyscf_parameters(BASE_CONFIG | {"scf_algorithm": "newton", "init_guess": "1e"})
    assert p2["scf_algorithm"] == "newton"

    # Newton rejects DIIS-specific parameters
    for diis_key in ("diis_space", "diis_start_cycle", "damp"):
        with pytest.raises(ValueError, match="does not accept"):
            validate_pyscf_parameters(BASE_CONFIG | {"scf_algorithm": "newton", diis_key: 10})

    # Invalid scf_algorithm or init_guess
    with pytest.raises(ValueError, match="scf_algorithm"):
        validate_pyscf_parameters(BASE_CONFIG | {"scf_algorithm": "ediis"})
    with pytest.raises(ValueError, match="init_guess"):
        validate_pyscf_parameters(BASE_CONFIG | {"init_guess": "random"})


def test_parameter_validation_dispersion():
    # Valid dispersion specifications
    for disp in ("d3bj", "d3zero", "d4", "d3zero:b3lyp", "d3bj:pbe", "d4:wb97x"):
        p = validate_pyscf_parameters(BASE_CONFIG | {"disp": disp})
        assert p["disp"].startswith(disp.split(":")[0])

    # Unknown dispersion prefix
    with pytest.raises(ValueError, match="disp prefix must be one of"):
        validate_pyscf_parameters(BASE_CONFIG | {"disp": "d2"})
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_pyscf_parameters(BASE_CONFIG | {"disp": "d3bj:"})


def test_parameter_validation_checkpoint():
    # Valid checkpoint setup
    p = validate_pyscf_parameters(BASE_CONFIG | {
        "checkpoint": {"read": "init.chk", "write": "final.chk", "allow_unconverged": True}
    })
    assert p["checkpoint"]["read"] == "init.chk"
    assert p["checkpoint"]["write"] == "final.chk"
    assert p["checkpoint"]["allow_unconverged"] is True

    # Same read and write file rejected
    with pytest.raises(ValueError, match="cannot point to the same file"):
        validate_pyscf_parameters(BASE_CONFIG | {
            "checkpoint": {"read": "same.chk", "write": "same.chk"}
        })


def test_parameter_validation_pcm_solvent():
    # Valid PCM options
    p = validate_pyscf_parameters(BASE_CONFIG | {
        "solvent": {
            "model": "pcm", "eps": 78.4, "method": "COSMO",
            "equilibrium_solvation": True, "lebedev_order": 29,
            "vdw_scale": 1.1, "r_probe": 0.5,
            "radii": {"H": 1.2, "O": 1.5}
        }
    })
    assert p["solvent"]["method"] == "COSMO"
    assert p["solvent"]["equilibrium_solvation"] is True
    assert p["solvent"]["radii"] == {"H": 1.2, "O": 1.5}

    # Invalid eps
    with pytest.raises(ValueError, match="eps > 1"):
        validate_pyscf_parameters(BASE_CONFIG | {"solvent": {"model": "pcm", "eps": 0.5}})


def test_parameter_validation_hessian():
    # Valid hessian options
    p = validate_pyscf_parameters(BASE_CONFIG | {
        "density_fit": True,
        "hessian": {"conv_tol_cpscf": 1e-8, "grid_response": True, "auxbasis_response": 2}
    })
    assert p["hessian"]["conv_tol_cpscf"] == 1e-8
    assert p["hessian"]["grid_response"] is True
    assert p["hessian"]["auxbasis_response"] == 2

    # auxbasis_response without density_fit must be rejected
    with pytest.raises(ValueError, match="auxbasis_response requires density_fit=true"):
        validate_pyscf_parameters(BASE_CONFIG | {
            "density_fit": False,
            "hessian": {"auxbasis_response": 2}
        })


def test_build_pcm_radii_table(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyscf.solvent.pcm", SimpleNamespace(
        modified_Bondi=np.full(120, 2.0)))
    monkeypatch.setitem(sys.modules, "pyscf.data.radii", SimpleNamespace(BOHR=Bohr))
    mol = SimpleNamespace(atom_symbols=lambda: ["H", "O"])
    solvent_cfg = {
        "vdw_scale": 1.2,
        "r_probe": 0.0,
        "radii": {"O": 1.52},  # explicit radius for O
    }
    radii_table_bohr, effective_radii = build_pcm_radii_table(mol, solvent_cfg)
    assert isinstance(radii_table_bohr, np.ndarray)
    assert effective_radii["O"] == 1.52
    assert effective_radii["H"] > 0
    # O radius in bohr must equal 1.52 / radii.BOHR
    assert radii_table_bohr[8] > 0


def test_format_hessian():
    natm = 2
    # Mock PySCF Hessian in Hartree / Bohr^2 of shape (2, 2, 3, 3)
    h_pyscf = np.zeros((2, 2, 3, 3))
    h_pyscf[0, 1, 0, 2] = 0.5  # Atom 0 coord x (0), Atom 1 coord z (2)
    h_ev_ang2, meta = format_hessian(h_pyscf, natm)
    assert h_ev_ang2.shape == (6, 6)
    assert meta["unit"] == "eV/Angstrom^2"
    # Row for atom 0 coord x is index 0. Column for atom 1 coord z is index 3*1 + 2 = 5.
    expected_val = 0.5 * Hartree / (Bohr ** 2)
    assert h_ev_ang2[0, 5] == pytest.approx(expected_val)


def test_checkpoint_compatibility_verification():
    mol = SimpleNamespace(atom_symbols=lambda: ["H", "H"], charge=0, spin=0,
                          _basis={"H": [[0, [1.0, 1.0]]]}, _ecp={}, cart=False, nao=2)
    settings = validate_pyscf_parameters(BASE_CONFIG)
    meta = CheckpointManager.metadata(mol, settings)
    assert CheckpointManager.verify_compatibility(meta, mol, settings) is None
    for key, value in (("atom_symbols", ["H", "O"]), ("basis", "def2-svp"),
                       ("spin", 2), ("basis_definition", {}), ("physics", {})):
        assert key in CheckpointManager.verify_compatibility(meta | {key: value}, mol, settings)
    assert "Missing" in CheckpointManager.verify_compatibility({}, mol, settings)


def test_get_hessian_calculator_method(tmp_path):
    h_mock = np.zeros((2, 2, 3, 3))
    h_mock[0, 0, 2, 2] = 0.25
    h_mock[1, 1, 2, 2] = 0.25
    mf_mock = SimpleNamespace(
        converged=True,
        kernel=lambda dm0=None: -1.0,
        nuc_grad_method=lambda: SimpleNamespace(kernel=lambda: np.zeros((2, 3))),
        Hessian=lambda: SimpleNamespace(kernel=lambda: h_mock),
        mol=SimpleNamespace(atom_symbols=lambda: ["H", "H"], charge=0, spin=0, natm=2),
    )

    class MockPySCFCalc(PySCFCalculator):
        def calculate(self, atoms=None, properties=("energy",), system_changes=None):
            self.atoms = atoms.copy() if atoms is not None else None
            self.results = {"energy": -1.0}
            self._scf = mf_mock
            self._scf_energy = -1.0
            self._scf_mol = mf_mock.mol
            self._state_key = self._state(atoms)[2]
            self._collector = DiagnosticsCollector(self.settings, self.directory)
            self.metadata = {"scf": {}}

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = MockPySCFCalc(parameters=BASE_CONFIG | {"retain_scf": True}, directory=tmp_path)
    atoms.calc = calc

    hess = calc.get_hessian(atoms)
    assert hess.shape == (6, 6)
    assert "hessian" in calc.metadata
    assert calc.metadata["hessian"]["unit"] == "eV/Angstrom^2"
    # SCF retained because retain_scf=True
    assert calc._scf is mf_mock

    # With retain_scf=False, get_hessian discards SCF
    calc_noretain = MockPySCFCalc(parameters=BASE_CONFIG | {"retain_scf": False}, directory=tmp_path)
    atoms.calc = calc_noretain
    hess2 = calc_noretain.get_hessian(atoms)
    assert hess2.shape == (6, 6)
    assert calc_noretain._scf is None
