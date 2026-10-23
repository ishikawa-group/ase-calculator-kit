"""Manual H100 check of UMA Nitro as the default; run only on a scheduled GPU node.

Usage: python tests/tsubame_validation_063_nitro.py --output /absolute/temp/nitro

Runs one fixed trajectory (rattled fcc Cu, ``task="omat"``) through the kit's
default and the other fast presets, with ``"high-precision"`` (FP32
merge_mole + compile) as the reference, and reports first-step time, steady
per-step time and the energy/force differences. The 27-atom case is the system
documented for fairchem-core 2.22 in ``docs/backends.md``. Every claimed check
has an assertion; the JSON summary is written even on failure, and a failure
exits 1. Timings on a shared MIG slice are indicative; the energies are exact.
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
from ase.build import bulk

from ase_calculator_kit import get_calculator

N_FRAMES = 12
WARMUP = 4
#: None is the kit's default, which must be Nitro; high-precision is the reference.
PRESETS = ("high-precision", None, "multigpu", "batch")
#: TF32 against FP32: anything beyond these is a bug, not reduced precision.
TF32_ENERGY_TOL_EV_PER_ATOM = 1e-3
TF32_FORCE_TOL_EV_PER_A = 2e-2
#: Two FP32 paths (merged vs unmerged MOLE) differ only by summation order.
FP32_ENERGY_TOL_EV_PER_ATOM = 1e-5
FP32_FORCE_TOL_EV_PER_A = 1e-3


def _frames(repeat: int) -> list:
    base = bulk("Cu", "fcc", a=3.6).repeat((repeat,) * 3)
    out = []
    for i in range(N_FRAMES):
        atoms = base.copy()
        atoms.rattle(0.02, seed=i)
        out.append(atoms)
    return out


def _model_kind(calc) -> dict:
    predictor = calc.predictor
    return {
        "nitro": bool(predictor.inference_settings.nitro),
        "model_class": type(predictor.model.module).__name__,
    }


def _run(preset, frames: list) -> dict:
    import torch

    kwargs = {} if preset is None else {"inference_settings": preset}
    start = time.perf_counter()
    calc = get_calculator("uma", task="omat", device="cuda", **kwargs)
    build = time.perf_counter() - start
    energies, forces, times = [], [], []
    for atoms in frames:
        atoms = atoms.copy()
        atoms.calc = calc
        start = time.perf_counter()
        energies.append(float(atoms.get_potential_energy()))
        forces.append(np.asarray(atoms.get_forces(), dtype=float))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {
        "calc": calc,
        "build_seconds": build,
        "first_step_seconds": times[0],
        "steady_ms": 1e3 * float(np.mean(times[WARMUP:])),
        "energies": np.array(energies),
        "forces": np.array(forces),
        **_model_kind(calc),
        "matmul_precision_after": torch.get_float32_matmul_precision(),
    }


def _compare(cur: dict, ref: dict, natoms: int) -> dict:
    de = np.abs(cur["energies"] - ref["energies"])
    df = cur["forces"] - ref["forces"]
    return {
        "max_dE_meV_per_atom": 1e3 * float(de.max()) / natoms,
        "rmse_dF_eV_per_A": float(np.sqrt((df**2).mean())),
        "max_dF_eV_per_A": float(np.abs(df).max()),
    }


def _size_case(repeat: int) -> dict:
    import torch

    frames = _frames(repeat)
    natoms = len(frames[0])
    runs = {}
    case: dict = {"n_atoms": natoms, "presets": {}}
    for preset in PRESETS:
        name = preset or "kit-default"
        runs[name] = run = _run(preset, frames)
        case["presets"][name] = {
            k: run[k] for k in (
                "build_seconds", "first_step_seconds", "steady_ms", "nitro",
                "model_class", "matmul_precision_after",
            )
        }
        case["presets"][name]["energy_frame0_eV"] = float(run["energies"][0])
        if preset is not None:  # free GPU memory; keep only the Nitro calculator
            del run["calc"]
            gc.collect()
            torch.cuda.empty_cache()
    ref = runs["high-precision"]
    for name, run in runs.items():
        if name != "high-precision":
            case["presets"][name]["vs_high_precision"] = _compare(run, ref, natoms)

    nitro = case["presets"]["kit-default"]
    assert nitro["nitro"] and nitro["model_class"] == "UMANitroModel", (
        f"kit default did not run UMA Nitro: {nitro}"
    )
    for name in ("high-precision", "multigpu", "batch"):
        assert not case["presets"][name]["nitro"], f"{name} ran Nitro"
    for name, entry in case["presets"].items():
        assert entry["matmul_precision_after"] == "highest", (
            f"{name} left float32 matmul precision at {entry['matmul_precision_after']}"
        )
    for name in ("kit-default", "multigpu"):
        cmp = case["presets"][name]["vs_high_precision"]
        assert cmp["max_dE_meV_per_atom"] < 1e3 * TF32_ENERGY_TOL_EV_PER_ATOM, (name, cmp)
        assert cmp["max_dF_eV_per_A"] < TF32_FORCE_TOL_EV_PER_A, (name, cmp)
    cmp = case["presets"]["batch"]["vs_high_precision"]
    assert cmp["max_dE_meV_per_atom"] < 1e3 * FP32_ENERGY_TOL_EV_PER_ATOM, cmp
    assert cmp["max_dF_eV_per_A"] < FP32_FORCE_TOL_EV_PER_A, cmp
    case["_nitro_calc"] = runs["kit-default"]["calc"]
    return case


def _fallback_case(calc) -> dict:
    """Nitro keeps serving a vacancy, and falls back on a new element.

    Removing a Cu atom leaves the composition (element fractions) of pure Cu
    unchanged, so Nitro must keep running. Substituting Ni changes it, so
    fairchem must fall back from Nitro and still answer.
    """
    out = {}
    vacancy = _frames(3)[0]
    del vacancy[0]
    alloy = _frames(3)[0]
    alloy.symbols[0] = "Ni"
    for name, atoms, nitro_expected in (("vacancy", vacancy, True), ("ni", alloy, False)):
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        kind = _model_kind(calc)
        assert np.isfinite(energy) and np.isfinite(forces).all(), name
        assert kind["nitro"] is nitro_expected, f"{name}: {kind}"
        out[name] = {"n_atoms": len(atoms), "energy_eV": energy, **kind}
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
    parser.add_argument("--repeats", type=int, nargs="+", default=[3, 8])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import torch

    report: dict = {
        "versions": {p: version(p) for p in ("fairchem-core", "torch", "ase")},
        "kit_commit": _git_commit(),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cases": {},
    }
    failures = 0
    nitro_calc = None
    for repeat in args.repeats:
        name = f"cu{repeat**3}"
        try:
            case = _size_case(repeat)
            calc = case.pop("_nitro_calc")
            nitro_calc = nitro_calc or calc
            case["ok"] = True
        except Exception:  # noqa: BLE001 - record and continue to the next case
            case = {"ok": False, "traceback": traceback.format_exc()}
            failures += 1
        report["cases"][name] = case
    try:
        assert nitro_calc is not None, "no Nitro calculator to test the fallback on"
        report["cases"]["fallback"] = {**_fallback_case(nitro_calc), "ok": True}
    except Exception:  # noqa: BLE001
        report["cases"]["fallback"] = {"ok": False, "traceback": traceback.format_exc()}
        failures += 1
    (args.output / "nitro_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
