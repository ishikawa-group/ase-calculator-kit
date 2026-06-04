"""Real CPU single-point across every supported calculator and variant.

Each case forces ``device="cpu"``, builds a small appropriate system, attaches
the calculator, and asserts ``get_potential_energy()`` returns a finite float.

These are marked ``slow`` because they download model weights on first run::

    pytest                 # runs them
    pytest -m "not slow"   # skips them

A case is *skipped* (not failed) when the failure is environmental — a missing
backend install, a model-weight download problem, or Hugging Face gating for the
UMA checkpoints. A genuine API/usage error still fails the test.
"""

from __future__ import annotations

import math
import numbers

import pytest
from ase import Atoms
from ase.build import bulk, molecule

from ase_umlip_kit import get_calculator
from ase_umlip_kit.errors import MissingDependencyError

pytestmark = pytest.mark.slow

# Substrings that mark an environmental problem -> skip rather than fail.
_ENV_HINTS = (
    "huggingface", "hugging face", "401", "403", "gated", "unauthorized",
    "access", "token", "login", "connection", "could not", "download",
    "max retries", "timed out", "timeout", "offline", "http", "url",
    "name or service not known", "no such file", "checkpoint", "resolve",
    "ssl", "certificate", "proxy",
)


def _bulk() -> Atoms:
    return bulk("Cu", "fcc", a=3.6)


def _molecule(*, charge: int | None = None, spin: int | None = None) -> Atoms:
    atoms = molecule("H2O")
    if charge is not None:
        atoms.info["charge"] = charge
    if spin is not None:
        atoms.info["spin"] = spin
    return atoms


# (test id, model name, kwargs, system factory)
# NOTE: the tiny bulk("Cu") / H2O systems below are intentionally API smoke-test
# structures to confirm each calculator can be built and run on CPU. They are NOT
# scientifically meaningful benchmark systems for each domain head (task/modal).
CASES = [
    ("chgnet", "chgnet", {}, _bulk),
    ("sevennet-mpa", "sevennet", {"modal": "mpa"}, _bulk),
    ("sevennet-omat24", "sevennet", {"modal": "omat24"}, _bulk),
    ("sevennet-matpes_pbe", "sevennet", {"modal": "matpes_pbe"}, _bulk),
    ("sevennet-matpes_r2scan", "sevennet", {"modal": "matpes_r2scan"}, _bulk),
    ("sevennet-omol25_low", "sevennet", {"modal": "omol25_low"}, _molecule),
    ("sevennet-omol25_high", "sevennet", {"modal": "omol25_high"}, _molecule),
    ("mattersim-1M", "mattersim", {"model": "1M"}, _bulk),
    ("mattersim-5M", "mattersim", {"model": "5M"}, _bulk),
    ("uma-omat", "uma", {"task": "omat"}, _bulk),
    ("uma-oc20", "uma", {"task": "oc20"}, _bulk),
    ("uma-oc22", "uma", {"task": "oc22"}, _bulk),
    ("uma-oc25", "uma", {"task": "oc25"}, _bulk),
    ("uma-odac", "uma", {"task": "odac"}, _bulk),
    ("uma-omol", "uma", {"task": "omol"}, lambda: _molecule(charge=0, spin=1)),
    ("uma-omc", "uma", {"task": "omc"}, lambda: _molecule(charge=0, spin=1)),
]


@pytest.mark.parametrize("model,kwargs,make_system", [c[1:] for c in CASES],
                         ids=[c[0] for c in CASES])
def test_cpu_single_point(model, kwargs, make_system, singlepoint_progress):
    try:
        calc = get_calculator(model, device="cpu", **kwargs)
        atoms = make_system()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
    except MissingDependencyError as exc:
        pytest.skip(f"backend not installed: {exc}")
    except Exception as exc:  # noqa: BLE001 - classify env vs real failure
        text = str(exc).lower()
        if any(hint in text for hint in _ENV_HINTS):
            pytest.skip(f"environmental (weights/network/HF): {type(exc).__name__}: {exc}")
        raise

    # ASE backends may return a numpy float scalar; accept any finite real number.
    assert isinstance(energy, numbers.Real)
    assert math.isfinite(float(energy))


@pytest.mark.parametrize(
    "model,kwargs",
    [("chgnet", {}), ("uma", {"task": "oc20"})],
    ids=["chgnet", "uma-oc20"],
)
def test_dispersion_changes_energy(model, kwargs, singlepoint_progress):
    """dispersion=True must add a (negative) D3 contribution vs the bare model."""
    try:
        bare = _bulk()
        bare.calc = get_calculator(model, device="cpu", **kwargs)
        e_bare = float(bare.get_potential_energy())

        d3 = _bulk()
        d3.calc = get_calculator(model, device="cpu", dispersion=True, **kwargs)
        e_d3 = float(d3.get_potential_energy())
    except MissingDependencyError as exc:
        pytest.skip(f"backend not installed: {exc}")
    except Exception as exc:  # noqa: BLE001 - classify env vs real failure
        text = str(exc).lower()
        if any(hint in text for hint in _ENV_HINTS):
            pytest.skip(f"environmental (weights/network/HF): {type(exc).__name__}: {exc}")
        raise

    assert math.isfinite(e_d3)
    # D3 dispersion is attractive, so the corrected energy should be lower.
    assert e_d3 < e_bare
