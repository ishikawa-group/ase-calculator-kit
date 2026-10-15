"""Manual H100 check of SevenNet FlashTP; run only on a scheduled GPU node.

Usage: python tests/tsubame_validation_062_flash.py --output /absolute/temp/flash

Builds the same 7net-omni model with and without ``enable_flash`` and compares
energies and forces on a 309-atom Pt/Co icosahedron with an O/OH overlayer,
plain and with D3(BJ). Every claimed check has an assertion; the JSON summary is
written even on failure, and a failure exits 1.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback

import numpy as np
from ase import Atoms
from ase.cluster import Icosahedron

from ase_calculator_kit import get_calculator
from ase_calculator_kit.backends.mlip.sevennet import _uses_flash

#: float32 kernels summed in a different order: anything beyond round-off is a bug.
ENERGY_TOL_EV_PER_ATOM = 1e-5
FORCE_TOL_EV_PER_A = 1e-3


def _structure() -> Atoms:
    atoms = Icosahedron("Pt", noshells=5, latticeconstant=3.92)
    rng = np.random.default_rng(0)
    center = atoms.positions.mean(axis=0)
    radius = np.linalg.norm(atoms.positions - center, axis=1)
    inner = np.argsort(radius)[:147]
    symbols = atoms.get_chemical_symbols()
    for i in inner:
        symbols[i] = "Co"
    atoms.set_chemical_symbols(symbols)
    surface = np.argsort(radius)[-162:]
    ads = []
    for k, i in enumerate(rng.choice(surface, size=40, replace=False)):
        normal = (atoms.positions[i] - center) / radius[i]
        site = atoms.positions[i] + 1.9 * normal
        ads.append(("O", site))
        if k % 2:
            ads.append(("H", site + 0.98 * normal))
    for symbol, pos in ads:
        atoms += Atoms(symbol, positions=[pos])
    atoms.positions += rng.normal(scale=0.02, size=atoms.positions.shape)
    atoms.center(vacuum=10.0)
    atoms.pbc = True
    return atoms


def _evaluate(calc, atoms: Atoms, repeats: int = 5) -> dict:
    atoms = atoms.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    times = []
    for _ in range(repeats):
        atoms.positions[0, 0] += 1e-6  # defeat the ASE cache
        start = time.perf_counter()
        atoms.get_forces()
        times.append(time.perf_counter() - start)
    return {"energy": float(energy), "forces": forces, "seconds": float(np.median(times))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="7net-omni")
    parser.add_argument("--modal", default="matpes_pbe")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch

    report: dict = {
        "versions": {p: version(p) for p in ("sevenn", "torch", "ase", "ase-calculator-kit")},
        "device": torch.cuda.get_device_name(0),
        "model": args.model,
        "modal": args.modal,
        "cases": {},
    }
    failures = 0
    atoms = _structure()
    report["n_atoms"] = len(atoms)
    for dispersion in (False, True):
        name = "d3" if dispersion else "plain"
        case: dict = {}
        try:
            common = dict(model=args.model, modal=args.modal, device="cuda",
                          dispersion=dispersion)
            ref_calc = get_calculator("sevennet", **common)
            flash_calc = get_calculator("sevennet", enable_flash=True, **common)
            bare = flash_calc.mixer.calcs[0] if dispersion else flash_calc
            assert _uses_flash(bare), "FlashTP layers missing from the flash model"
            ref = _evaluate(ref_calc, atoms)
            fla = _evaluate(flash_calc, atoms)
            de = abs(ref["energy"] - fla["energy"]) / len(atoms)
            df = float(np.abs(ref["forces"] - fla["forces"]).max())
            case.update(
                energy_e3nn_eV=ref["energy"], energy_flash_eV=fla["energy"],
                dE_eV_per_atom=de, max_dF_eV_per_A=df,
                seconds_e3nn=ref["seconds"], seconds_flash=fla["seconds"],
                speedup=ref["seconds"] / fla["seconds"],
            )
            assert de < ENERGY_TOL_EV_PER_ATOM, f"energy differs by {de} eV/atom"
            assert df < FORCE_TOL_EV_PER_A, f"forces differ by {df} eV/A"
            case["ok"] = True
        except Exception:  # noqa: BLE001 - record and continue to the next case
            case["ok"] = False
            case["traceback"] = traceback.format_exc()
            failures += 1
        report["cases"][name] = case
    (args.output / "flash_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
