"""Manual CPU/H100 release checks; run only on a scheduled compute node.

Usage: python tests/tsubame_validation_060.py --output /absolute/temp/review
Every claimed check has an assertion. JSON is saved even on failure; failures exit 1.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import traceback

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.units import Bohr, Hartree

import ase_calculator_kit
from ase_calculator_kit.backends.dft.pyscf import PySCFCalculator
from ase_calculator_kit.backends.dft._pyscf_support import CheckpointManager, to_numpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="*")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    common = {"basis": "def2-svp", "density_fit": True, "auxbasis": "def2-universal-jkfit",
              "xc": "pbe", "grids": {"level": 3}, "nlcgrids": {"atom_grid": [50, 194], "prune": None},
              "conv_tol": 1e-10, "conv_tol_grad": 1e-5, "verbose": 0,
              "retain_scf": True, "hessian": {"grid_response": False, "auxbasis_response": 2,
                                               "conv_tol_cpscf": 1e-9}}
    cases = {
        "water_pbe": (molecule("H2O"), 0, 1, {}),
        "oh_vv10": (molecule("OH"), 0, 2, {"xc": "wb97m_v", "nlc": "vv10"}),
        "water_d3zero": (molecule("H2O"), 0, 1, {"disp": "d3zero:pbe"}),
        "water_d4": (molecule("H2O"), 0, 1, {"disp": "d4:pbe"}),
        "water_cosmo": (molecule("H2O"), 0, 1, {"solvent": {"model": "pcm", "method": "COSMO",
            "eps": 78.4, "equilibrium_solvation": True, "r_probe": 0.4,
            "vdw_scale": 1.1, "radii": {"O": 1.6}, "lebedev_order": 17}}),
        "water_newton_cosmo": (molecule("H2O"), 0, 1, {"scf_algorithm": "newton",
            "solvent": {"model": "pcm", "method": "COSMO", "eps": 78.4,
                        "equilibrium_solvation": True}}),
        "oh_newton_vv10": (molecule("OH"), 0, 2,
            {"scf_algorithm": "newton", "xc": "wb97m_v", "nlc": "vv10"}),
        "water_smd": (molecule("H2O"), 0, 1, {"solvent": {"model": "smd", "solvent": "water"}}),
        "rbh_ecp": (Atoms("RbH", positions=[[0, 0, 0], [0, 0, 2.3]]), 0, 1,
                    {"ecp": {"Rb": "def2-svp"}}),
    }
    package = Path(ase_calculator_kit.__file__).parent
    report = {"host": platform.node(), "versions": {n: version(n) for n in
              ("ase", "pyscf", "gpu4pyscf-cuda12x", "cupy-cuda12x", "pyscf-dispersion")},
              "source_sha256": {str(f.relative_to(package)): hashlib.sha256(f.read_bytes()).hexdigest()
                                for f in (package / "backends/dft").glob("*.py")}, "checks": {}}

    def check(name, callback):
        print(name, flush=True)
        try:
            report["checks"][name] = {"passed": True, **callback()}
        except Exception as exc:
            traceback.print_exc()
            report["checks"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        (root / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def parity(name):
        atoms, charge, mult, extra = cases[name]
        atoms.info.update(charge=charge, spin=mult)
        p = common | extra
        output = {}
        for gpu in (False, True):
            label = "gpu" if gpu else "cpu"
            calc = PySCFCalculator(parameters=p, gpu=gpu, directory=root / name / label)
            atoms.calc = calc
            try:
                energy = atoms.get_potential_energy()
            except Exception:
                (root / name / label / "failure.json").write_text(json.dumps(calc.metadata, indent=2))
                raise
            mf = calc._scf
            stream = calc._scf_log
            forces = atoms.get_forces()
            assert calc._scf is mf and not stream.closed
            direct_force = -to_numpy(mf.nuc_grad_method().kernel()) * Hartree / Bohr
            np.testing.assert_allclose(forces, direct_force, atol=1e-8, rtol=0)
            try:
                hessian = calc.get_hessian(atoms)
            except NotImplementedError as exc:
                assert not gpu and mult == 2 and extra.get("nlc") == "vv10"
                assert "CPU UKS Hessians with NLC" in str(exc)
                hessian = reference = None
                symmetry_error = None
            else:
                direct_hess = mf.Hessian()
                direct_hess.grid_response = False
                direct_hess.auxbasis_response = 2
                reference = to_numpy(direct_hess.kernel()).transpose(0, 2, 1, 3).reshape(hessian.shape) * Hartree / Bohr**2
                np.testing.assert_allclose(hessian, reference, atol=1e-7, rtol=0)
                symmetry_error = float(np.max(abs(hessian-hessian.T)))
                assert np.isfinite(hessian).all()
            assert calc.metadata["scf"].get("final_orbital_gradient_norm") is not None
            if mult == 2:
                spin = calc.metadata["scf"]["spin_analysis"]
                assert abs(sum(spin["mulliken_spin_populations"]) - 1) < 1e-7
            output[label] = {"energy_eV": energy, "forces": forces.tolist(), "hessian": hessian.tolist() if hessian is not None else None,
                             "hessian_status": "supported" if hessian is not None else "CPU UKS NLC unsupported (expected error)",
                             "symmetry_error": symmetry_error, "same_scf_force_delta": float(np.max(abs(forces-direct_force))),
                             "same_scf_hessian_delta": float(np.max(abs(hessian-reference))) if hessian is not None else None,
                             "diagnostics": calc.metadata}
            (root / name / "observations.json").write_text(json.dumps(output, indent=2))
            calc.reset()
            assert stream.closed
        e_delta = abs(output["gpu"]["energy_eV"] - output["cpu"]["energy_eV"])
        f_delta = float(np.max(abs(np.array(output["gpu"]["forces"]) - output["cpu"]["forces"])))
        h_delta = (float(np.max(abs(np.array(output["gpu"]["hessian"]) - output["cpu"]["hessian"])))
                   if output["cpu"]["hessian"] is not None else None)
        assert e_delta < 3e-5 and f_delta < 6e-4, (e_delta, f_delta)
        assert max(v["symmetry_error"] for v in output.values() if v["symmetry_error"] is not None) < 1e-3
        assert h_delta is None or h_delta < 0.01, h_delta  # Cross-device/grid implementation comparison, not adapter tolerance.
        output.update(energy_delta_eV=e_delta, force_delta_eV_A=f_delta, hessian_delta_eV_A2=h_delta)
        return output

    def restart_and_newton():
        atoms = molecule("H2O")
        atoms.info.update(charge=0, spin=1)
        p = common | {"checkpoint": {"write": str((root / "cpu.chk").resolve())}}
        cpu = PySCFCalculator(parameters=p, directory=root / "checkpoint_cpu")
        atoms.calc = cpu
        baseline = atoms.get_potential_energy()
        cpu.reset()
        answers = {}
        for gpu in (False, True):
            label = "gpu" if gpu else "cpu"
            # Save a completed but unaccepted first iteration; then use GPU/CPU Newton.
            checkpoint = (root / f"{label}_partial.chk").resolve()
            partial = PySCFCalculator(parameters=common | {"max_cycle": 1,
                "checkpoint": {"write": str(checkpoint)}}, gpu=gpu, directory=root / (label + "_partial"))
            atoms.calc = partial
            try:
                atoms.get_potential_energy()
            except Exception:
                assert partial.metadata["scf"]["failure_stage"] == "scf_convergence"
            else:
                raise AssertionError("One-cycle water should not converge")
            values, meta = CheckpointManager.load(checkpoint, True)
            assert not meta["converged"] and values["e_tot"] < -70
            restart = PySCFCalculator(parameters=common | {"scf_algorithm": "newton",
                "checkpoint": {"read": str(checkpoint), "allow_unconverged": True}},
                gpu=gpu, directory=root / (label + "_newton"))
            atoms.calc = restart
            energy = atoms.get_potential_energy()
            assert abs(energy-baseline) < 3e-5
            assert restart.metadata["scf"]["init_guess_source"] == "checkpoint"
            assert np.isfinite(atoms.get_forces()).all()
            assert np.isfinite(restart.get_hessian(atoms)).all()
            answers[label] = restart.metadata
            restart.reset()
        cross = PySCFCalculator(parameters=common | {"checkpoint": {"read": str((root / "cpu.chk").resolve())}},
                                gpu=True, directory=root / "cross_device")
        atoms.calc = cross
        assert abs(atoms.get_potential_energy()-baseline) < 3e-5
        cross.reset()
        return answers

    def reuse_open_shell():
        atoms = molecule("OH")
        atoms.info.update(charge=0, spin=2)
        calc = PySCFCalculator(parameters=common | {"reuse_density": True, "scf_algorithm": "newton"},
                               gpu=True, directory=root / "reuse_newton")
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.positions[1, 2] += 0.01
        atoms.get_potential_energy()
        assert calc.metadata["scf"]["init_guess_source"] == "projected_mo"
        np.testing.assert_allclose(sum(calc.metadata["scf"]["spin_analysis"]["mulliken_spin_populations"]), 1, atol=1e-7)
        atoms.info.update(charge=-1, spin=1)
        atoms.get_potential_energy()
        assert calc.metadata["scf"]["init_guess_source"] == "default"
        assert "charge mismatch" in calc.metadata["scf"]["density_reuse_fallback"]
        meta = calc.metadata
        calc.reset()
        return meta

    names = args.cases or list(cases) + ["restart_newton", "reuse_open_shell"]
    for name in names:
        if name == "restart_newton":
            check(name, restart_and_newton)
        elif name == "reuse_open_shell":
            check(name, reuse_open_shell)
        else:
            check(name, lambda name=name: parity(name))
    report["all_passed"] = all(item["passed"] for item in report["checks"].values())
    (root / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("All passed:", report["all_passed"], flush=True)
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
