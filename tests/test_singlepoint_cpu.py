"""Real CPU single-point across every supported calculator and variant.

Each case forces ``device="cpu"``, builds a small appropriate system, attaches
the calculator, and asserts ``get_potential_energy()`` returns a finite float.

These are marked ``slow`` because they download model weights on first run::

    pytest -m slow         # runs them
    pytest                 # skips them

A case is *skipped* (not failed) when the failure is environmental — a missing
backend install, a model-weight download problem, or Hugging Face gating for the
UMA checkpoints. A genuine API/usage error still fails the test.
"""

from __future__ import annotations

import importlib
import math
import numbers
import socket
import ssl
from urllib.error import HTTPError, URLError

import pytest
from ase import Atoms
from ase.build import bulk, molecule

from ase_calculator_kit import get_calculator
from ase_calculator_kit.errors import MissingDependencyError

pytestmark = pytest.mark.slow

def _is_environment_error(exc, seen=None):
    """Skip identified transport/auth failures, never guess from model error text."""
    seen = set() if seen is None else seen
    if id(exc) in seen:
        return False
    seen.add(id(exc))
    http_types = [HTTPError]
    network_types = [ConnectionError, TimeoutError, socket.gaierror, ssl.SSLError]
    offline_types = []
    for module_name, http_names, network_names, offline_names in (
        ("requests.exceptions", ("HTTPError",), ("ConnectionError", "Timeout"), ()),
        ("httpx", ("HTTPStatusError",), ("NetworkError", "TimeoutException"), ()),
        ("huggingface_hub.errors", ("HfHubHTTPError",), (),
         ("GatedRepoError", "LocalEntryNotFoundError", "OfflineModeIsEnabled")),
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for target, names in ((http_types, http_names), (network_types, network_names),
                              (offline_types, offline_names)):
            target.extend(getattr(module, name) for name in names if hasattr(module, name))
    if isinstance(exc, tuple(offline_types)):
        return True
    if isinstance(exc, tuple(http_types)):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", getattr(exc, "code", None))
        return status in {401, 403, 408, 429} or (isinstance(status, int) and status >= 500)
    if isinstance(exc, URLError):
        return isinstance(exc.reason, tuple(network_types))
    if isinstance(exc, tuple(network_types)):
        return True
    # MACE wraps download failures in RuntimeError; classify the explicit cause,
    # so a wrapped invalid-model ValueError still fails.
    return (isinstance(exc, RuntimeError) and exc.__cause__ is not None
            and _is_environment_error(exc.__cause__, seen))


def _bulk() -> Atoms:
    return bulk("Cu", "fcc", a=3.6)


def _molecule(*, charge: int | None = None, spin: int | None = None) -> Atoms:
    atoms = molecule("H2O")
    if charge is not None:
        atoms.info["charge"] = charge
    if spin is not None:
        atoms.info["spin"] = spin
    return atoms


def _hydroxide(*, charge: int, spin: int) -> Atoms:
    """OH as an anion (charge=-1, spin=1) or a radical (charge=0, spin=2)."""
    atoms = molecule("OH")
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    return atoms


_UMA_BATCH = {"inference_settings": "batch"}


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
    ("sevennet-omni-i8", "sevennet", {"model": "7net-omni-i8", "modal": "mpa"}, _bulk),
    ("sevennet-omni-i12", "sevennet", {"model": "7net-omni-i12", "modal": "mpa"}, _bulk),
    # Single-fidelity: modal="auto" has to send no modal at all for this to run.
    ("sevennet-omat", "sevennet", {"model": "7net-omat"}, _bulk),
    # MACE only ever runs from its own environment (e3nn 0.4.4 vs >=0.5), where
    # every other case here skips as "backend not installed" — and vice versa.
    ("mace-mh-1-omat_pbe", "mace", {"head": "omat_pbe"}, _bulk),
    ("mace-mh-1-mp_pbe_refit_add", "mace", {"head": "mp_pbe_refit_add"}, _bulk),
    ("mace-mh-1-oc20_usemppbe", "mace", {"head": "oc20_usemppbe"}, _bulk),
    ("mace-mh-1-matpes_r2scan", "mace", {"head": "matpes_r2scan"}, _bulk),
    ("mace-mh-1-omol", "mace", {"head": "omol"}, _molecule),
    ("mace-mh-1-spice_wB97M", "mace", {"head": "spice_wB97M"}, _molecule),
    # Single-head: head="auto" has to send no head at all for this to run.
    ("mace-omat-0", "mace", {"model": "medium-omat-0"}, _bulk),
    # MACE-Polar: the model name alone has to route to mace_polar, and these
    # read charge/spin like UMA's omol head, so both are set explicitly.
    ("mace-polar-1-s", "mace", {"model": "polar-1-s"},
     lambda: _molecule(charge=0, spin=1)),
    ("mace-polar-1-m", "mace", {"model": "polar-1-m"},
     lambda: _molecule(charge=0, spin=1)),
    ("mace-polar-1-l", "mace", {"model": "polar-1-l"},
     lambda: _molecule(charge=0, spin=1)),
    ("mattersim-1M", "mattersim", {"model": "1M"}, _bulk),
    ("mattersim-5M", "mattersim", {"model": "5M"}, _bulk),
    ("nequip-OAM-S", "nequip", {"model": "S"}, _bulk),
    ("nequip-OAM-M", "nequip", {"model": "M"}, _bulk),
    ("nequip-OAM-L", "nequip", {"model": "L"}, _bulk),
    ("nequip-OAM-XL", "nequip", {"model": "XL"}, _bulk),
    # OrbMol-v2 raises when charge/spin are missing, so every case sets both.
    # The first case exercises the default (compiled) path; the other two turn
    # compilation off, since re-paying it per case buys nothing.
    ("orb-orbmol-v2", "orb", {}, lambda: _molecule(charge=0, spin=1)),
    ("orb-orbmol-v2-anion", "orb", {"compile": False},
     lambda: _hydroxide(charge=-1, spin=1)),
    ("orb-orbmol-v2-radical", "orb", {"compile": False},
     lambda: _hydroxide(charge=0, spin=2)),
    # uma-omat keeps the "default" preset so the MD fast path is exercised;
    # the rest use "batch", which skips a MOLE merge and a compile this suite
    # discards after one single point anyway (18.8 s vs 2.6 s on CPU).
    ("uma-omat", "uma", {"task": "omat"}, _bulk),
    ("uma-omat-turbo", "uma", {"task": "omat", "inference_settings": "turbo"}, _bulk),
    ("uma-oc20", "uma", {"task": "oc20", **_UMA_BATCH}, _bulk),
    ("uma-oc22", "uma", {"task": "oc22", **_UMA_BATCH}, _bulk),
    ("uma-oc25", "uma", {"task": "oc25", **_UMA_BATCH}, _bulk),
    ("uma-odac", "uma", {"task": "odac", **_UMA_BATCH}, _bulk),
    ("uma-omol", "uma", {"task": "omol", **_UMA_BATCH},
     lambda: _molecule(charge=0, spin=1)),
    ("uma-omc", "uma", {"task": "omc", **_UMA_BATCH},
     lambda: _molecule(charge=0, spin=1)),
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
        if _is_environment_error(exc):
            pytest.skip(f"environmental (weights/network/HF): {type(exc).__name__}: {exc}")
        raise

    # ASE backends may return a numpy float scalar; accept any finite real number.
    assert isinstance(energy, numbers.Real)
    assert math.isfinite(float(energy))


@pytest.mark.parametrize(
    "model,kwargs",
    [("chgnet", {}), ("uma", {"task": "oc20"}), ("mace", {"head": "omat_pbe"})],
    ids=["chgnet", "uma-oc20", "mace-omat_pbe"],
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
        if _is_environment_error(exc):
            pytest.skip(f"environmental (weights/network/HF): {type(exc).__name__}: {exc}")
        raise

    assert math.isfinite(e_d3)
    # D3 dispersion is attractive, so the corrected energy should be lower.
    assert e_d3 < e_bare
