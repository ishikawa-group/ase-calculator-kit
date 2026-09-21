"""Run an XYZ single point or ASE optimization with a molecular DFT YAML.

Example: python examples/dft/run_pyscf.py water.xyz --config \
examples/dft/pyscf_wb97mv.yaml --output runs/water --optimize
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS

from ase_calculator_kit import get_calculator
from ase_calculator_kit.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--fmax", type=float, default=0.05, help="eV/Angstrom")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not np.isfinite(args.fmax) or args.fmax <= 0 or args.steps < 1:
        parser.error("fmax and steps must be positive")
    cfg = load_config(args.config)
    if cfg.get("calculator", "").lower() not in {"pyscf", "gpu4pyscf"}:
        parser.error("config must select pyscf or gpu4pyscf")
    # Refuse stale output rather than mixing different charge/spin calculations.
    args.output.mkdir(parents=True, exist_ok=False)
    cfg["directory"] = str(args.output)
    atoms = read(args.xyz)
    calc = get_calculator(cfg["calculator"], config=cfg, write_resolved_config=True)
    atoms.calc = calc
    cfg = load_config(args.output / "resolved_calculator_config.yaml")
    versions = {}
    for name in ("ase-calculator-kit", "ase", "pyscf", "gpu4pyscf-cuda12x",
                 "cupy-cuda12x", "cutensor-cu12", "pyscf-dispersion"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    result = {"config": cfg, "versions": versions,
              "units": {"energy": "eV", "forces": "eV/Angstrom", "positions": "Angstrom"},
              "optimization_requested": args.optimize, "success": False}
    try:
        if args.optimize:
            optimizer = BFGS(atoms, logfile=str(args.output / "optimization.log"),
                             trajectory=str(args.output / "optimization.traj"))
            converged = optimizer.run(fmax=args.fmax, steps=args.steps)
            result.update(optimization_converged=bool(converged),
                          optimization_steps=optimizer.nsteps, fmax=args.fmax)
        else:
            converged = True
        # Asking for forces first avoids a second SCF for the energy.
        forces = atoms.get_forces()
        energy = atoms.get_potential_energy()
        result.update(calc.metadata, energy=energy, success=bool(converged),
                      config=load_config(args.output / "resolved_calculator_config.yaml"))
        np.savez(args.output / "results.npz", positions=atoms.positions,
                 atomic_numbers=atoms.numbers, energy=energy, forces=forces)
        write(args.output / "final.xyz", atoms, format="xyz")
        final = atoms.copy()
        final.info.update(charge=calc.metadata["charge"], spin=calc.metadata["multiplicity"])
        write(args.output / "final.extxyz", final, format="extxyz")
        if not converged:
            raise RuntimeError("Geometry optimization did not reach fmax within the step limit.")
    except Exception as exc:
        result.update(success=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
