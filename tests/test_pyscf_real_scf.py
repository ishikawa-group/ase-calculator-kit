"""Real PySCF calculations validating 0.6.0 features on CPU.

Skipped if pyscf is not installed (e.g. In standard lightweight environment).
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms

from ase_calculator_kit import get_calculator
from ase_calculator_kit.backends.dft.pyscf import PySCFCalculator
from ase_calculator_kit.backends.dft._pyscf_support import CheckpointManager

try:
    import pyscf  # noqa: F401
    has_pyscf = True
except ImportError:
    has_pyscf = False

pytestmark = pytest.mark.skipif(not has_pyscf, reason="Real PySCF is required for this suite")


def test_real_cdiis_and_newton_and_hessian(tmp_path):
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])

    # 1. CDIIS calculation + Hessian
    cfg_cdiis = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "cdiis"),
        "parameters": {
            "basis": "sto-3g", "charge": 0, "multiplicity": 1,
            "scf_algorithm": "cdiis", "conv_tol": 1e-9, "retain_scf": True,
        }
    }
    calc1 = get_calculator("pyscf", config=cfg_cdiis)
    h2.calc = calc1
    e1 = h2.get_potential_energy()
    f1 = h2.get_forces()
    h1 = calc1.get_hessian(h2)

    assert np.isfinite(e1)
    assert f1.shape == (2, 3)
    assert h1.shape == (6, 6)
    # Hessian should be symmetric
    np.testing.assert_allclose(h1, h1.T, atol=1e-5)
    assert calc1.metadata["converged"] is True
    assert calc1.metadata["scf"]["scf_algorithm"] == "cdiis"
    assert "energy_components" in calc1.metadata["scf"]

    # 2. Newton calculation
    cfg_newton = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "newton"),
        "parameters": {
            "basis": "sto-3g", "charge": 0, "multiplicity": 1,
            "scf_algorithm": "newton", "conv_tol": 1e-9,
        }
    }
    calc2 = get_calculator("pyscf", config=cfg_newton)
    h2.calc = calc2
    e2 = h2.get_potential_energy()
    h2_hess = calc2.get_hessian(h2)
    assert calc2.metadata["scf"]["scf_algorithm"] == "newton"
    assert e2 == pytest.approx(e1, abs=1e-6)
    np.testing.assert_allclose(h2_hess, h1, atol=1e-4)


def test_real_open_shell_spin_and_diagnostics(tmp_path):
    oh = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
    cfg = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "oh"),
        "parameters": {
            "basis": "sto-3g", "charge": 0, "multiplicity": 2,
            "method": "uks", "xc": "pbe",
            "diagnostics": {"save": True, "iterations": True},
        }
    }
    calc = get_calculator("pyscf", config=cfg)
    oh.calc = calc
    e = oh.get_potential_energy()
    assert np.isfinite(e)
    diag = calc.metadata["scf"]
    assert diag["converged"] is True
    assert "spin_analysis" in diag
    # <S^2> should be close to 0.75 for doublet
    assert diag["spin_analysis"]["s2"] == pytest.approx(0.75, abs=0.05)
    assert "mulliken_spin_populations" in diag["spin_analysis"]
    spin_pops = diag["spin_analysis"]["mulliken_spin_populations"]
    assert len(spin_pops) == 2
    # Total spin population should sum to roughly 1.0 (Nalpha - Nbeta)
    assert sum(spin_pops) == pytest.approx(1.0, abs=0.01)

    # Check written diagnostics JSON and JSONL
    diag_json = (tmp_path / "oh" / "scf_diagnostics.json")
    iter_jsonl = (tmp_path / "oh" / "scf_iterations.jsonl")
    assert diag_json.exists()
    assert iter_jsonl.exists()
    loaded_diag = json.loads(diag_json.read_text(encoding="utf-8"))
    assert loaded_diag["converged"] is True


def test_real_checkpoint_save_and_resume(tmp_path):
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    chk_file = tmp_path / "h2.chk"

    # 1. Run and write checkpoint
    cfg_write = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "run1"),
        "parameters": {
            "basis": "sto-3g", "charge": 0, "multiplicity": 1,
            "checkpoint": {"write": str(chk_file)},
        }
    }
    calc1 = get_calculator("pyscf", config=cfg_write)
    h2.calc = calc1
    e1 = h2.get_potential_energy()
    assert chk_file.exists()

    # 2. Resume in a fresh calculator from checkpoint
    cfg_read = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "run2"),
        "parameters": {
            "basis": "sto-3g", "charge": 0, "multiplicity": 1,
            "checkpoint": {"read": str(chk_file)},
        }
    }
    calc2 = get_calculator("pyscf", config=cfg_read)
    h2_copy = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    h2_copy.calc = calc2
    e2 = h2_copy.get_potential_energy()
    assert e2 == pytest.approx(e1, abs=1e-7)
    assert calc2.metadata["scf"]["init_guess_source"] == "checkpoint"

    # 3. Incompatible checkpoint should raise ValueError
    cfg_incompat = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "run3"),
        "parameters": {
            "basis": "sto-3g", "charge": 1, "multiplicity": 2,  # different charge/spin
            "checkpoint": {"read": str(chk_file)},
        }
    }
    calc3 = get_calculator("pyscf", config=cfg_incompat)
    h2_plus = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    h2_plus.info.update(charge=1, spin=2)
    h2_plus.calc = calc3
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        h2_plus.get_potential_energy()


def test_real_reuse_density_between_geometries(tmp_path):
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    cfg = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "density_reuse"),
        "parameters": {
            "basis": "sto-3g", "charge": 0, "multiplicity": 1,
            "reuse_density": True,
        }
    }
    calc = get_calculator("pyscf", config=cfg)
    h2.calc = calc

    # Step 1: initial calculation (init_guess="minao")
    e1 = h2.get_potential_energy()
    assert calc.metadata["scf"]["init_guess_source"] == "default"

    # Step 2: displaced geometry, same electronic state -> orbital projection
    h2.positions[1, 2] = 0.76
    e2 = h2.get_potential_energy()
    assert calc.metadata["scf"]["init_guess_source"] == "projected_mo"
    assert e2 != e1


def test_real_pcm_cosmo_and_radii(tmp_path):
    h2o = Atoms("H2O", positions=[
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587],
    ])
    cfg = {
        "calculator": "pyscf",
        "directory": str(tmp_path / "pcm"),
        "parameters": {
            "basis": "sto-3g", "charge": 0, "multiplicity": 1,
            "solvent": {
                "model": "pcm", "eps": 78.4, "method": "COSMO",
                "equilibrium_solvation": True,
                "vdw_scale": 1.2,
                "radii": {"O": 1.50},  # custom radius for O in Angstrom
            }
        }
    }
    calc = get_calculator("pyscf", config=cfg)
    h2o.calc = calc
    e = h2o.get_potential_energy()
    assert np.isfinite(e)
    assert calc.metadata["converged"] is True
    assert np.isfinite(calc.metadata["scf"]["final_orbital_gradient_norm"])


def test_unconverged_checkpoint_and_failure_diagnostics(tmp_path):
    from ase.build import molecule
    atoms = molecule("H2O")
    parameters = {"basis": "sto-3g", "charge": 0, "spin": 0, "max_cycle": 1,
                  "checkpoint": {"write": "state.chk"},
                  "diagnostics": {"save": True, "iterations": True}}
    calc = PySCFCalculator(parameters=parameters, directory=tmp_path)
    atoms.calc = calc
    with pytest.raises(Exception, match="did not converge"):
        atoms.get_potential_energy()
    assert not calc.results and calc._scf_log is None
    assert calc.metadata["scf"]["failure_stage"] == "scf_convergence"
    assert json.loads((tmp_path / "scf_diagnostics.json").read_text())["converged"] is False
    checkpoint = tmp_path / "state.chk"
    with pytest.raises(ValueError, match="unconverged"):
        CheckpointManager.load(checkpoint)
    values, meta = CheckpointManager.load(checkpoint, allow_unconverged=True)
    assert not meta["converged"]
    assert values["e_tot"] < -70  # the completed iteration, not mf.e_tot's initial zero
    assert values["mo_coeff"].shape == (7, 7)
    assert len((tmp_path / "scf_iterations.jsonl").read_text().splitlines()) == 1

    # Resume a partial checkpoint through Newton in a different Python process.
    code = '''
from ase.build import molecule
from ase_calculator_kit.backends.dft.pyscf import PySCFCalculator
import sys
atoms=molecule("H2O")
atoms.calc=PySCFCalculator(parameters={"basis":"sto-3g", "charge":0, "spin":0,
    "scf_algorithm":"newton", "checkpoint":{"read":sys.argv[1], "allow_unconverged":True}},
    directory=sys.argv[2])
assert atoms.get_potential_energy() < -1900
assert atoms.calc.metadata["scf"]["init_guess_source"] == "checkpoint"
'''
    subprocess.run([sys.executable, "-c", code, str(checkpoint), str(tmp_path / "resume")], check=True)


def test_checkpoint_physics_and_projection(tmp_path):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    p = {"basis": "sto-3g", "charge": 0, "spin": 0, "checkpoint": {"write": "a.chk"}}
    atoms.calc = PySCFCalculator(parameters=p, directory=tmp_path)
    atoms.get_potential_energy()
    checkpoint = tmp_path / "a.chk"
    original_bytes = checkpoint.read_bytes()
    for change in ({"density_fit": True}, {"solvent": {"model": "pcm", "eps": 10.}},
                   {"grids": {"level": 1}}, {"method": "uks"}):
        atoms.calc = PySCFCalculator(parameters=p | change | {"checkpoint": {"read": str(checkpoint)}},
                                     directory=tmp_path / "invalid")
        with pytest.raises(ValueError, match="Incompatible checkpoint"):
            atoms.get_potential_energy()
    atoms.positions[1, 2] = 0.8
    atoms.calc = PySCFCalculator(parameters=p | {"checkpoint": {"read": str(checkpoint)}},
                                 directory=tmp_path / "projected")
    energy = atoms.get_potential_energy()
    assert atoms.calc.metadata["scf"]["init_guess_source"] == "checkpoint"
    atoms.calc = PySCFCalculator(parameters=p | {"checkpoint": None}, directory=tmp_path / "fresh")
    assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-7)
    assert checkpoint.read_bytes() == original_bytes


def test_density_reuse_rejects_changed_electronic_state(tmp_path):
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.97]])
    atoms.info.update(charge=0, spin=2)
    calc = PySCFCalculator(parameters={"basis": "sto-3g", "reuse_density": True}, directory=tmp_path)
    atoms.calc = calc
    atoms.get_potential_energy()
    atoms.info.update(charge=-1, spin=1)
    changed_energy = atoms.get_potential_energy()
    assert calc.metadata["scf"]["init_guess_source"] == "default"
    assert "charge mismatch" in calc.metadata["scf"]["density_reuse_fallback"]
    atoms.calc = PySCFCalculator(parameters={"basis": "sto-3g"}, directory=tmp_path / "fresh")
    assert atoms.get_potential_energy() == pytest.approx(changed_energy, abs=1e-7)


def test_hessian_after_forces_and_failure_cleanup(tmp_path, monkeypatch):
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    calc = PySCFCalculator(parameters={"basis": "sto-3g", "method": "rhf", "charge": 0,
                                      "spin": 0}, directory=tmp_path)
    atoms.calc = calc
    atoms.get_forces()
    hessian = calc.get_hessian(atoms)
    np.testing.assert_allclose(calc.get_hessian(atoms), hessian, atol=1e-7)
    step = 1e-3
    atoms.positions[1, 2] += step
    plus = atoms.get_forces().copy()
    atoms.positions[1, 2] -= 2 * step
    minus = atoms.get_forces().copy()
    np.testing.assert_allclose(-(plus-minus).ravel() / (2*step), hessian[:, 5], atol=1e-3)

    calc.settings["retain_scf"] = True
    atoms.get_potential_energy()
    if calc._scf is None:
        calc.calculate(atoms)
    from types import SimpleNamespace
    stream = calc._scf_log
    monkeypatch.setattr(calc._scf, "Hessian", lambda: SimpleNamespace(
        kernel=lambda: np.full((2, 2, 3, 3), np.nan)))
    with pytest.raises(ValueError, match="nonfinite Hessian"):
        calc.get_hessian(atoms)
    assert stream.closed and calc._scf is None and not calc.results
    assert calc.metadata["scf"]["failure_stage"] == "hessian"


def test_checkpoint_write_errors_are_not_hidden(tmp_path, monkeypatch):
    from ase.build import molecule
    def fail(*args, **kwargs):
        raise PermissionError("checkpoint write denied")
    monkeypatch.setattr(CheckpointManager, "save", fail)
    calc = PySCFCalculator(parameters={"basis": "sto-3g", "charge": 0, "spin": 0,
        "checkpoint": {"write": "state.chk"}}, directory=tmp_path)
    atoms = molecule("H2O")
    atoms.calc = calc
    with pytest.raises(PermissionError, match="checkpoint write denied"):
        atoms.get_potential_energy()
    assert not calc.results and calc._scf_log is None
    assert calc.metadata["scf"]["failure_stage"] == "checkpoint"
    assert calc.metadata["scf"]["cycles"] == 1
