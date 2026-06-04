#!/usr/bin/env python
"""Run a CPU single-point with every supported uMLIP and print the energies.

This doubles as a usage demo: each calculator is created in one line via
``get_calculator`` and attached to an ASE ``Atoms`` object exactly as you would
in your own scripts. A tqdm progress bar tracks the run (weights download on
first use, so the first run is slow).

    python examples/run_all_models.py            # all variants on CPU
    python examples/run_all_models.py --device auto
    python examples/run_all_models.py --only chgnet sevennet
"""

from __future__ import annotations

import argparse

from ase import Atoms
from ase.build import bulk, molecule
from tqdm import tqdm

from ase_umlip_kit import get_calculator


def make_bulk() -> Atoms:
    return bulk("Cu", "fcc", a=3.6)


def make_molecule(*, charge: int | None = None, spin: int | None = None) -> Atoms:
    atoms = molecule("H2O")
    if charge is not None:
        atoms.info["charge"] = charge
    if spin is not None:
        atoms.info["spin"] = spin
    return atoms


# (label, model name, kwargs for get_calculator, system factory)
VARIANTS = [
    ("chgnet", "chgnet", {}, make_bulk),
    ("sevennet 7net-omni/mpa", "sevennet", {"modal": "mpa"}, make_bulk),
    ("sevennet 7net-omni/omat24", "sevennet", {"modal": "omat24"}, make_bulk),
    ("sevennet 7net-omni/matpes_pbe", "sevennet", {"modal": "matpes_pbe"}, make_bulk),
    ("sevennet 7net-omni/matpes_r2scan", "sevennet", {"modal": "matpes_r2scan"}, make_bulk),
    ("sevennet 7net-omni/omol25_low", "sevennet", {"modal": "omol25_low"}, make_molecule),
    ("sevennet 7net-omni/omol25_high", "sevennet", {"modal": "omol25_high"}, make_molecule),
    ("mattersim 1M", "mattersim", {"model": "1M"}, make_bulk),
    ("mattersim 5M", "mattersim", {"model": "5M"}, make_bulk),
    ("uma-s-1p2/omat", "uma", {"task": "omat"}, make_bulk),
    ("uma-s-1p2/oc20", "uma", {"task": "oc20"}, make_bulk),
    ("uma-s-1p2/oc22", "uma", {"task": "oc22"}, make_bulk),
    ("uma-s-1p2/oc25", "uma", {"task": "oc25"}, make_bulk),
    ("uma-s-1p2/odac", "uma", {"task": "odac"}, make_bulk),
    ("uma-s-1p2/omol", "uma", {"task": "omol"}, lambda: make_molecule(charge=0, spin=1)),
    ("uma-s-1p2/omc", "uma", {"task": "omc"}, lambda: make_molecule(charge=0, spin=1)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda, mps, or auto")
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="restrict to these model names (chgnet sevennet mattersim uma)",
    )
    args = parser.parse_args()

    variants = VARIANTS
    if args.only:
        wanted = {m.lower() for m in args.only}
        variants = [v for v in VARIANTS if v[1] in wanted]

    results: list[tuple[str, str]] = []
    for label, name, kwargs, make_system in tqdm(
        variants, desc="uMLIP single-point", unit="variant"
    ):
        try:
            atoms = make_system()
            atoms.calc = get_calculator(name, device=args.device, **kwargs)
            energy = float(atoms.get_potential_energy())
            results.append((label, f"{energy:.4f} eV"))
        except Exception as exc:  # noqa: BLE001 - demo: report and continue
            results.append((label, f"SKIPPED ({type(exc).__name__}: {exc})"))

    width = max(len(label) for label, _ in results)
    print("\nResults:")
    for label, value in results:
        print(f"  {label:<{width}}  {value}")


if __name__ == "__main__":
    main()
