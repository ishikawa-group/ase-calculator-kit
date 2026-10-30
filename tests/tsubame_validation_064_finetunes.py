"""Manual H100 check of the UMA fine-tunes; run only on a scheduled GPU node.

Usage: python tests/tsubame_validation_064_finetunes.py --output /absolute/temp/ft

For ``uma-s-1p2p1-ft-r2scan`` (rattled fcc Cu, its only task ``omat``) and
``uma-s-1p2p1-ft-d4rev`` (ethanol singlet and OH radical, its only task
``omol``), the kit's default must select the fine-tune's task, run UMA Nitro and
agree with ``"high-precision"`` (FP32) to TF32 accuracy. ``ft-r2scan`` with
``dispersion=True`` must add the r2SCAN D3(BJ) term, not the PBE one that its
``omat`` task would select on ``uma-s-1p2p1``. Every claimed check has an
assertion; the JSON summary is written even on failure, and a failure exits 1.
"""
from __future__ import annotations

import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator, all_changes

from ase_calculator_kit import get_calculator
from ase_calculator_kit.dispersion import wrap_with_d3

N_FRAMES = 12
WARMUP = 4
TF32_ENERGY_TOL_EV_PER_ATOM = 1e-3
TF32_FORCE_TOL_EV_PER_A = 2e-2


def _frames(base: Atoms, info: dict | None = None) -> list:
    out = []
    for i in range(N_FRAMES):
        atoms = base.copy()
        atoms.info.update(info or {})
        atoms.rattle(0.02, seed=i)
        out.append(atoms)
    return out


def _model_kind(calc) -> dict:
    bare = calc.mixer.calcs[0] if hasattr(calc, "mixer") else calc
    predictor = bare.predictor
    return {
        "nitro": bool(predictor.inference_settings.nitro),
        "model_class": type(predictor.model.module).__name__,
        "task_name": bare.task_name,
    }


def _run(frames: list, **kwargs) -> dict:
    import torch

    calc = get_calculator("uma", device="cuda", **kwargs)
    energies, forces, times = [], [], []
    for atoms in frames:
        atoms = atoms.copy()
        atoms.calc = calc
        start = time.perf_counter()
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    out = {
        "first_step_seconds": times[0],
        "steady_ms": 1e3 * float(np.mean(times[WARMUP:])),
        "energies": np.array(energies),
        "forces": np.array(forces),
        **_model_kind(calc),
    }
    del calc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def _compare(model: str, frames: list, expected_task: str) -> dict:
    natoms = len(frames[0])
    ref = _run(frames, model=model, inference_settings="high-precision")
    nitro = _run(frames, model=model)  # the kit's default: task "auto", nitro
    de = np.abs(nitro["energies"] - ref["energies"]).max() / natoms
    df = nitro["forces"] - ref["forces"]
    case = {
        "n_atoms": natoms,
        "high_precision": {k: ref[k] for k in ("first_step_seconds", "steady_ms", "model_class")},
        "nitro": {k: nitro[k] for k in ("first_step_seconds", "steady_ms", "model_class")},
        "energy_frame0_eV": float(ref["energies"][0]),
        "max_dE_meV_per_atom": 1e3 * float(de),
        "rmse_dF_eV_per_A": float(np.sqrt((df**2).mean())),
        "max_dF_eV_per_A": float(np.abs(df).max()),
    }
    assert nitro["task_name"] == expected_task, nitro["task_name"]
    assert nitro["nitro"] and nitro["model_class"] == "UMANitroModel", nitro
    assert not ref["nitro"], "high-precision ran Nitro"
    assert de < TF32_ENERGY_TOL_EV_PER_ATOM, case
    assert case["max_dF_eV_per_A"] < TF32_FORCE_TOL_EV_PER_A, case
    return case


class _Zero(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "free_energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def _r2scan_d3(frames: list) -> dict:
    """dispersion=True on ft-r2scan adds exactly the r2SCAN D3(BJ) term."""
    atoms = frames[0].copy()
    atoms.calc = get_calculator("uma", device="cuda", model="uma-s-1p2p1-ft-r2scan")
    e_bare = float(atoms.get_potential_energy())
    atoms.calc = get_calculator(
        "uma", device="cuda", model="uma-s-1p2p1-ft-r2scan", dispersion=True
    )
    e_d3 = float(atoms.get_potential_energy())
    terms = {}
    for xc in ("r2scan", "pbe"):
        atoms.calc = wrap_with_d3(_Zero(), xc=xc, device="cuda")
        terms[xc] = float(atoms.get_potential_energy())
    added = e_d3 - e_bare
    out = {"added_eV": added, "d3_r2scan_eV": terms["r2scan"], "d3_pbe_eV": terms["pbe"]}
    assert abs(added - terms["r2scan"]) < 1e-4, out
    assert abs(terms["r2scan"] - terms["pbe"]) > 1e-3, out
    return out


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - a copied tree has no .git
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch

    report: dict = {
        "versions": {p: version(p) for p in ("fairchem-core", "torch", "ase", "torch-dftd")},
        "kit_commit": _git_commit(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cases": {},
    }
    cu = _frames(bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3)))
    cases = {
        "r2scan_cu27": lambda: _compare("uma-s-1p2p1-ft-r2scan", cu, "omat"),
        "r2scan_cu27_d3": lambda: _r2scan_d3(cu),
        "d4rev_ethanol": lambda: _compare(
            "uma-s-1p2p1-ft-d4rev",
            _frames(molecule("CH3CH2OH"), {"charge": 0, "spin": 1}), "omol",
        ),
        "d4rev_oh_radical": lambda: _compare(
            "uma-s-1p2p1-ft-d4rev",
            _frames(molecule("OH"), {"charge": 0, "spin": 2}), "omol",
        ),
    }
    failures = 0
    for name, run in cases.items():
        try:
            report["cases"][name] = {**run(), "ok": True}
        except Exception:  # noqa: BLE001 - record and continue to the next case
            report["cases"][name] = {"ok": False, "traceback": traceback.format_exc()}
            failures += 1
    (args.output / "finetune_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
