#!/usr/bin/env python3
"""Validation suite for ase-calculator-kit 0.6.0 features on TSUBAME4 (GPU4PySCF)."""

import json
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
from ase.build import molecule
from ase.units import Bohr, Hartree
from ase_calculator_kit import get_calculator
from pyscf import lib

lib.num_threads(2)
root = Path.cwd()
log_dir = root / "log"
log_dir.mkdir(exist_ok=True, parents=True)

report = {
    "system_info": {},
    "basic_parity": {},
    "new_features": {},
    "all_passed": True,
}

print("=== Starting TSUBAME4 Validation for ase-calculator-kit 0.6.0 ===", flush=True)

# ---------------------------------------------------------
# Test cases for basic energy & force parity (CPU vs GPU)
# ---------------------------------------------------------
wb_common = {
    "basis": "def2-tzvpd",
    "xc": "wb97m_v",
    "nlc": "vv10",
    "density_fit": True,
    "auxbasis": "def2-universal-jkfit",
    "grids": {"atom_grid": [99, 590], "prune": None},
    "nlcgrids": {"atom_grid": [50, 194]},
    "conv_tol": 1e-9,
    "max_cycle": 200,
    "verbose": 0,
}

parity_cases = [
    ("water_wb", molecule("H2O"), dict(wb_common, charge=0, multiplicity=1)),
    ("oh_wb", molecule("OH"), dict(wb_common, charge=0, multiplicity=2)),
    (
        "water_d3bj",
        molecule("H2O"),
        {"basis": "def2-svp", "xc": "pbe", "density_fit": True, "charge": 0, "spin": 0, "disp": "d3bj", "verbose": 0},
    ),
    (
        "water_d3zero",
        molecule("H2O"),
        {"basis": "def2-svp", "xc": "pbe", "density_fit": True, "charge": 0, "spin": 0, "disp": "d3zero", "verbose": 0},
    ),
    (
        "water_pcm_custom",
        molecule("H2O"),
        {
            "basis": "def2-svp",
            "xc": "pbe",
            "density_fit": True,
            "charge": 0,
            "spin": 0,
            "solvent": {
                "model": "pcm",
                "vdw_scale": 1.1,
                "r_probe": 0.4,
                "radii": {"H": 1.2, "O": 1.52},
                "eps": 78.3553,
            },
            "verbose": 0,
        },
    ),
]

for label, atoms, p in parity_cases:
    print(f"\n--- Parity Test: {label} ---", flush=True)
    records = {}
    for backend in ("pyscf", "gpu4pyscf"):
        t0 = time.monotonic()
        try:
            calc = get_calculator(
                backend,
                config={
                    "calculator": backend,
                    "directory": str(root / label / backend),
                    "parameters": p,
                },
            )
            atoms.calc = calc
            f = atoms.get_forces()
            e = atoms.get_potential_energy()
            scf_meta = calc.metadata.get("scf", {})
            records[backend] = {
                "energy_hartree": e / Hartree,
                "forces_au": (f / (Hartree / Bohr)).tolist(),
                "seconds": time.monotonic() - t0,
                "converged": scf_meta.get("converged"),
                "cycles": scf_meta.get("cycles"),
                "spin_s2": scf_meta.get("spin_analysis", {}).get("s2"),
            }
            print(f"  {backend}: E = {e/Hartree:.8f} Ha, time = {records[backend]['seconds']:.2f}s, conv = {records[backend]['converged']}", flush=True)
        except Exception as ex:
            traceback.print_exc()
            records[backend] = {"error": str(ex)}
            report["all_passed"] = False

    if "energy_hartree" in records.get("pyscf", {}) and "energy_hartree" in records.get("gpu4pyscf", {}):
        de = abs(records["pyscf"]["energy_hartree"] - records["gpu4pyscf"]["energy_hartree"])
        df = float(np.max(np.abs(np.array(records["pyscf"]["forces_au"]) - np.array(records["gpu4pyscf"]["forces_au"]))))
        passed = (de <= 1e-6) and (df <= 1e-5)
        records["delta_energy_hartree"] = de
        records["max_delta_force_au"] = df
        records["passed"] = passed
        if not passed:
            report["all_passed"] = False
        print(f"  Result: ΔE = {de:.2e} Ha, ΔF = {df:.2e} au, passed = {passed}", flush=True)
    else:
        records["passed"] = False
        report["all_passed"] = False

    report["basic_parity"][label] = records

# ---------------------------------------------------------
# Test 0.6.0 New Features on GPU4PySCF
# ---------------------------------------------------------
print("\n=== Feature Tests: 0.6.0 specific on GPU4PySCF ===", flush=True)

# 1. Hessian API (analytical and finite difference fallback)
print("\n--- Feature 1: Hessian API (GPU4PySCF) ---", flush=True)
try:
    atoms_h2o = molecule("H2O")
    # Analytical Hessian (DFT without dispersion)
    calc_h_ana = get_calculator(
        "gpu4pyscf",
        config={
            "calculator": "gpu4pyscf",
            "directory": str(root / "feat_hessian_ana"),
            "parameters": {"basis": "def2-svp", "xc": "pbe", "density_fit": True, "charge": 0, "spin": 0, "verbose": 0},
        },
    )
    atoms_h2o.calc = calc_h_ana
    hess_ana = calc_h_ana.get_hessian(atoms_h2o)
    n_atoms = len(atoms_h2o)
    assert hess_ana.shape == (3 * n_atoms, 3 * n_atoms), f"Expected shape {(3*n_atoms, 3*n_atoms)}, got {hess_ana.shape}"
    # Check symmetry H = H^T
    symm_err = np.max(np.abs(hess_ana - hess_ana.T))
    assert symm_err < 1e-5, f"Hessian not symmetric: max err {symm_err}"

    # Also compare with CPU analytical Hessian
    calc_h_cpu = get_calculator(
        "pyscf",
        config={
            "calculator": "pyscf",
            "directory": str(root / "feat_hessian_cpu"),
            "parameters": {"basis": "def2-svp", "xc": "pbe", "density_fit": True, "charge": 0, "spin": 0, "verbose": 0},
        },
    )
    atoms_h2o.calc = calc_h_cpu
    hess_cpu = calc_h_cpu.get_hessian(atoms_h2o)
    delta_hess = np.max(np.abs(hess_ana - hess_cpu))

    # Analytical Hessian with D3 dispersion (which falls back to finite difference of D3)
    calc_h_d3 = get_calculator(
        "gpu4pyscf",
        config={
            "calculator": "gpu4pyscf",
            "directory": str(root / "feat_hessian_d3"),
            "parameters": {"basis": "def2-svp", "xc": "pbe", "disp": "d3bj", "density_fit": True, "charge": 0, "spin": 0, "verbose": 0},
        },
    )
    atoms_h2o.calc = calc_h_d3
    hess_d3 = calc_h_d3.get_hessian(atoms_h2o)
    assert hess_d3.shape == (3 * n_atoms, 3 * n_atoms)

    report["new_features"]["hessian"] = {
        "passed": True,
        "shape": list(hess_ana.shape),
        "symmetry_error": float(symm_err),
        "delta_gpu_cpu_max": float(delta_hess),
        "d3_hessian_shape": list(hess_d3.shape),
    }
    print(f"  Hessian test passed: shape={hess_ana.shape}, symm_err={symm_err:.2e}, delta_gpu_cpu={delta_hess:.2e}", flush=True)
except Exception as ex:
    traceback.print_exc()
    report["new_features"]["hessian"] = {"passed": False, "error": str(ex)}
    report["all_passed"] = False

# 2. Checkpoint Save & Resume with kit metadata
print("\n--- Feature 2: Checkpoint write and read ---", flush=True)
try:
    chk_file = root / "chk_test.h5"
    if chk_file.exists():
        chk_file.unlink()

    atoms_h2o = molecule("H2O")
    # Step A: write checkpoint
    calc_chk_write = get_calculator(
        "gpu4pyscf",
        config={
            "calculator": "gpu4pyscf",
            "directory": str(root / "feat_chk_write"),
            "parameters": {
                "basis": "def2-svp",
                "xc": "pbe",
                "density_fit": True,
                "charge": 0,
                "spin": 0,
                "checkpoint": {"write": str(chk_file)},
                "verbose": 0,
            },
        },
    )
    atoms_h2o.calc = calc_chk_write
    e_chk1 = atoms_h2o.get_potential_energy()

    assert chk_file.exists(), "Checkpoint file was not created"

    # Verify kit metadata inside HDF5
    with h5py.File(chk_file, "r") as f:
        assert "ase_calculator_kit" in f, "Missing kit group in HDF5"
        meta_grp = f["ase_calculator_kit"]
        assert "metadata_json" in meta_grp.attrs, "Missing metadata_json in HDF5"
        meta = json.loads(meta_grp.attrs["metadata_json"])
        assert meta["converged"] is True, "Checkpoint not marked converged"
        assert meta["basis"] == "def2-svp", f"Basis mismatch in chk: {meta.get('basis')}"
        assert meta["xc"] == "pbe", f"XC mismatch in chk: {meta.get('xc')}"
        print("  Kit metadata verified in HDF5 checkpoint", flush=True)

    # Step B: read checkpoint
    calc_chk_read = get_calculator(
        "gpu4pyscf",
        config={
            "calculator": "gpu4pyscf",
            "directory": str(root / "feat_chk_read"),
            "parameters": {
                "basis": "def2-svp",
                "xc": "pbe",
                "density_fit": True,
                "charge": 0,
                "spin": 0,
                "checkpoint": {"read": str(chk_file)},
                "verbose": 0,
            },
        },
    )
    atoms_h2o.calc = calc_chk_read
    e_chk2 = atoms_h2o.get_potential_energy()
    diff_e = abs(e_chk1 - e_chk2)
    assert diff_e < 1e-9, f"Energy mismatch after checkpoint read: {diff_e}"

    report["new_features"]["checkpoint"] = {
        "passed": True,
        "energy_step1": float(e_chk1),
        "energy_step2": float(e_chk2),
        "delta_energy": float(diff_e),
        "metadata_verified": True,
    }
    print(f"  Checkpoint test passed: delta_E = {diff_e:.2e} eV", flush=True)
except Exception as ex:
    traceback.print_exc()
    report["new_features"]["checkpoint"] = {"passed": False, "error": str(ex)}
    report["all_passed"] = False

# 3. Density Reuse (reuse_density)
print("\n--- Feature 3: Density reuse (reuse_density) ---", flush=True)
try:
    atoms_h2o = molecule("H2O")
    calc_reuse = get_calculator(
        "gpu4pyscf",
        config={
            "calculator": "gpu4pyscf",
            "directory": str(root / "feat_reuse"),
            "parameters": {
                "basis": "def2-svp",
                "xc": "pbe",
                "density_fit": True,
                "charge": 0,
                "spin": 0,
                "reuse_density": True,
                "verbose": 0,
            },
        },
    )
    atoms_h2o.calc = calc_reuse
    e1 = atoms_h2o.get_potential_energy()
    cycles1 = calc_reuse.metadata.get("scf", {}).get("cycles")

    # Small displacement
    atoms_h2o.positions[0, 0] += 0.01
    e2 = atoms_h2o.get_potential_energy()
    cycles2 = calc_reuse.metadata.get("scf", {}).get("cycles")
    converged2 = calc_reuse.metadata.get("scf", {}).get("converged")

    assert converged2, "Second step did not converge with reused density"

    report["new_features"]["reuse_density"] = {
        "passed": True,
        "step1_cycles": cycles1,
        "step2_cycles": cycles2,
        "step2_converged": converged2,
    }
    print(f"  Reuse density test passed: step1_cycles={cycles1}, step2_cycles={cycles2}, converged={converged2}", flush=True)
except Exception as ex:
    traceback.print_exc()
    report["new_features"]["reuse_density"] = {"passed": False, "error": str(ex)}
    report["all_passed"] = False

# 4. SCF Algorithm Control (cdiis)
print("\n--- Feature 4: SCF Algorithm (CDIIS) ---", flush=True)
try:
    atoms_h2o = molecule("H2O")
    calc_cdiis = get_calculator(
        "gpu4pyscf",
        config={
            "calculator": "gpu4pyscf",
            "directory": str(root / "feat_cdiis"),
            "parameters": {
                "basis": "def2-svp",
                "xc": "pbe",
                "density_fit": True,
                "charge": 0,
                "spin": 0,
                "scf_algorithm": "cdiis",
                "diis_space": 10,
                "verbose": 0,
            },
        },
    )
    atoms_h2o.calc = calc_cdiis
    e_cdiis = atoms_h2o.get_potential_energy()
    meta_cdiis = calc_cdiis.metadata.get("scf", {})
    assert meta_cdiis.get("converged"), "CDIIS did not converge"
    assert meta_cdiis.get("scf_algorithm") == "cdiis"

    report["new_features"]["scf_algorithm_cdiis"] = {
        "passed": True,
        "cycles": meta_cdiis.get("cycles"),
        "converged": True,
        "scf_algorithm": meta_cdiis.get("scf_algorithm"),
    }
    print(f"  CDIIS test passed: cycles={meta_cdiis.get('cycles')}, conv={meta_cdiis.get('converged')}", flush=True)
except Exception as ex:
    traceback.print_exc()
    report["new_features"]["scf_algorithm_cdiis"] = {"passed": False, "error": str(ex)}
    report["all_passed"] = False

# 5. Open-shell Diagnostics (S^2 and Mulliken Spin Population)
print("\n--- Feature 5: Open-shell Diagnostics (OH doublet) ---", flush=True)
try:
    atoms_oh = molecule("OH")
    calc_diag = get_calculator(
        "gpu4pyscf",
        config={
            "calculator": "gpu4pyscf",
            "directory": str(root / "feat_diag"),
            "parameters": {
                "basis": "def2-svp",
                "xc": "pbe",
                "density_fit": True,
                "charge": 0,
                "multiplicity": 2,
                "verbose": 0,
            },
        },
    )
    atoms_oh.calc = calc_diag
    atoms_oh.get_potential_energy()
    meta_diag = calc_diag.metadata.get("scf", {})
    spin_analysis = meta_diag.get("spin_analysis", {})
    mulliken_spin = spin_analysis.get("mulliken_spin_populations")

    assert spin_analysis.get("s2") is not None, "Missing S^2"
    assert spin_analysis.get("ideal_s2") == 0.75, f"Expected ideal S^2 = 0.75, got {spin_analysis.get('ideal_s2')}"
    assert mulliken_spin is not None and len(mulliken_spin) == 2, f"Missing Mulliken spin population: {mulliken_spin}"

    report["new_features"]["open_shell_diagnostics"] = {
        "passed": True,
        "s2_total": spin_analysis.get("s2"),
        "s2_ideal": spin_analysis.get("ideal_s2"),
        "s2_deviation": spin_analysis.get("s2_deviation"),
        "mulliken_spin": mulliken_spin,
    }
    print(f"  Open-shell diagnostics passed: S^2={spin_analysis.get('s2'):.4f} (ideal {spin_analysis.get('ideal_s2')}), Mulliken={mulliken_spin}", flush=True)
except Exception as ex:
    traceback.print_exc()
    report["new_features"]["open_shell_diagnostics"] = {"passed": False, "error": str(ex)}
    report["all_passed"] = False

# Write summary JSON
out_path = root / "validation_060.json"
out_path.write_text(json.dumps(report, indent=2))
log_out = log_dir / "validation_060.json"
log_out.write_text(json.dumps(report, indent=2))

print("\n=== Validation Complete ===")
print(f"All passed: {report['all_passed']}")
print(f"Report written to: {out_path} and {log_out}")
