"""Molecular PySCF adapters: upstream electronic structure, ASE units and caching."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, CalculationFailed, all_changes
from ase.units import Bohr, Hartree

from ...config import resolve_calculator_config, write_resolved_config_file
from ...errors import DispersionError, MissingDependencyError
from ..base import BaseBackend

_DEFAULTS = {
    "method": "auto", "xc": "pbe", "ecp": None,
    "density_fit": False, "auxbasis": None, "nlc": None, "disp": None,
    "grids": {}, "nlcgrids": {}, "conv_tol": 1e-9, "max_cycle": 200,
    "max_memory": 4000, "verbose": 4, "solvent": None,
}
_GRID_KEYS = {"level", "atom_grid", "prune"}


def _mapping(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")


def _parameters(raw):
    _mapping(raw, set(_DEFAULTS) | {"basis", "charge", "spin", "multiplicity"}, "parameters")
    p = deepcopy(_DEFAULTS) | deepcopy(raw)
    if not p.get("basis"):
        raise ValueError("PySCF requires parameters.basis.")
    for key in ("charge", "spin", "multiplicity"):
        if key in p:
            value = p[key]
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{key} must be an integer.")
            p[key] = int(value)
    if "multiplicity" in p:
        if p["multiplicity"] < 1:
            raise ValueError("multiplicity must be a positive integer.")
        if "spin" in p and p["spin"] != p["multiplicity"] - 1:
            raise ValueError("spin must equal multiplicity - 1.")
        p["spin"] = p["multiplicity"] - 1
    if "spin" in p:
        if p["spin"] < 0:
            raise ValueError("spin must be nonnegative (Nalpha - Nbeta).")
        p["multiplicity"] = p["spin"] + 1
    if p["method"] not in {"auto", "rks", "uks", "rhf", "uhf"}:
        raise ValueError("method must be auto, rks, uks, rhf or uhf.")
    if p["method"] in {"rks", "rhf"} and p.get("spin", 0):
        raise ValueError("Restricted closed-shell methods require spin=0.")
    if type(p["density_fit"]) is not bool:
        raise ValueError("density_fit must be true or false.")
    if p["auxbasis"] and not p["density_fit"]:
        raise ValueError("auxbasis requires density_fit=true.")
    if p["nlc"] not in (None, False, "vv10"):
        raise ValueError("nlc must be null (automatic), false, or 'vv10'.")
    if p["disp"] not in (None, "d3bj", "d4"):
        raise ValueError("disp must be null, 'd3bj', or 'd4'.")
    if not isinstance(p["xc"], str) or not p["xc"]:
        raise ValueError("xc must be a nonempty PySCF functional name.")
    if "-d3" in p["xc"].lower() or "-d4" in p["xc"].lower():
        raise ValueError("Specify dispersion with disp, not an xc suffix.")
    if p["method"] in {"rhf", "uhf"}:
        if p["nlc"] or "xc" in raw or p["grids"] or p["nlcgrids"]:
            raise ValueError("HF does not accept xc, nlc or DFT grids.")
    for name in ("conv_tol", "max_memory"):
        if (isinstance(p[name], bool) or not isinstance(p[name], (int, float))
                or not np.isfinite(p[name]) or p[name] <= 0):
            raise ValueError(f"{name} must be a finite positive number.")
    if type(p["max_cycle"]) is not int or p["max_cycle"] < 1:
        raise ValueError("max_cycle must be a positive integer.")
    if type(p["verbose"]) is not int or not 0 <= p["verbose"] <= 9:
        raise ValueError("verbose must be an integer from 0 to 9.")
    for name in ("grids", "nlcgrids"):
        grid = p[name]
        _mapping(grid, _GRID_KEYS, name)
        if "prune" in grid and grid["prune"] is not None:
            raise ValueError(f"{name}.prune only accepts null (disable pruning); omit for default.")
        if "level" in grid and (type(grid["level"]) is not int or not 0 <= grid["level"] <= 9):
            raise ValueError(f"{name}.level must be an integer from 0 to 9.")
        if "atom_grid" in grid:
            pair = grid["atom_grid"]
            if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                    or any(type(n) is not int or n <= 0 for n in pair)):
                raise ValueError(f"{name}.atom_grid must contain two positive integers.")
    if p["solvent"] is not None:
        s = p["solvent"]
        _mapping(s, {"model", "solvent", "eps", "method"}, "solvent")
        if s.get("model") == "smd":
            if set(s) != {"model", "solvent"} or not isinstance(s["solvent"], str):
                raise ValueError("SMD requires exactly model=smd and solvent=<name>.")
        elif s.get("model") == "pcm":
            eps = s.get("eps")
            if (set(s) - {"model", "eps", "method"} or isinstance(eps, bool)
                    or not isinstance(eps, (int, float)) or not np.isfinite(eps) or eps <= 1):
                raise ValueError("PCM requires eps > 1 and optional method.")
            if s.get("method", "IEF-PCM") not in {"C-PCM", "COSMO", "IEF-PCM", "SS(V)PE"}:
                raise ValueError("Unsupported PCM method.")
        else:
            raise ValueError("solvent.model must be smd or pcm.")
    return p


def _electronic_state(settings, info):
    """Resolve explicit total charge and multiplicity, rejecting disagreements."""
    state, sources = {}, {}
    for key in ("charge", "spin"):
        av = info.get(key)
        if key in info:
            if isinstance(av, bool) or not isinstance(av, Integral):
                raise ValueError(f"atoms.info[{key!r}] must be an integer.")
            av = int(av)
            if key == "spin":
                if av < 1:
                    raise ValueError("atoms.info['spin'] is multiplicity and must be >= 1.")
                av -= 1
        yv = settings.get(key)
        if key in info and yv is not None and av != yv:
            raise ValueError(
                f"Electronic-state mismatch for {key}: atoms.info={info[key]}, "
                f"YAML={yv}. atoms.info['spin'] is multiplicity; YAML spin is multiplicity - 1."
            )
        if key not in info and yv is None:
            raise ValueError(f"Missing {key}: set atoms.info charge/spin or YAML charge/spin/multiplicity.")
        state[key] = av if key in info else yv
        sources[key] = "atoms.info+yaml" if key in info and yv is not None else (
            "atoms.info" if key in info else "yaml"
        )
    if settings["method"] in {"rks", "rhf"} and state["spin"]:
        raise ValueError("Restricted closed-shell methods require spin=0.")
    state["multiplicity"] = state["spin"] + 1
    return state, sources


class PySCFCalculator(Calculator):
    """Translate molecular HF/DFT energies and analytic gradients to ASE.

    YAML ``spin`` is Nalpha-Nbeta; atoms.info["spin"] is multiplicity.
    Only nonperiodic structures are accepted, even when an ASE cell is present.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, *, parameters, gpu=False, directory="."):
        super().__init__(directory=directory)
        self.settings = _parameters(parameters)
        self.gpu = gpu
        self.metadata: dict[str, Any] = {}
        self._state_key = None
        self._write_config = False

    def _state(self, atoms):
        state, sources = _electronic_state(self.settings, atoms.info)
        key = (state["charge"], state["spin"], sources["charge"], sources["spin"])
        return state, sources, key

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol)
        if self._state(atoms)[2] != self._state_key:
            changes.append("electronic_state")
        return changes

    def get_property(self, name, atoms=None, allow_calculation=True):
        candidate = atoms if atoms is not None else self.atoms
        if candidate is not None:
            try:
                if self._state(candidate)[2] != self._state_key:
                    self.results = {}
            except ValueError:
                self.results = {}
                self.metadata = {}
                raise
        return super().get_property(name, atoms, allow_calculation)

    def _save_config(self, parameters):
        if self._write_config:
            saved = deepcopy(parameters)
            if saved["method"] in {"rhf", "uhf"}:
                saved.pop("xc", None)
            write_resolved_config_file(
                {"calculator": "gpu4pyscf" if self.gpu else "pyscf",
                 "directory": self.directory, "parameters": saved}, self.directory
            )

    def set(self, **kwargs):
        if kwargs:
            raise TypeError("Configure PySCF through config/overrides in the factory, not calc.set().")
        return {}

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {}
        self.metadata = {}
        if self.atoms.pbc.any():
            raise ValueError("PySCF backend supports molecules only; all atoms.pbc must be false.")
        if not len(self.atoms) or not np.isfinite(self.atoms.positions).all():
            raise ValueError("PySCF requires a nonempty structure with finite coordinates.")
        from pyscf import dft, gto, scf

        state, sources, state_key = self._state(self.atoms)
        p = self.settings | state
        self._save_config(p)
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        # A context manager closes PySCF's output even when SCF/gradients fail.
        with (Path(self.directory) / "pyscf.log").open("a", encoding="utf-8") as log:
            mol = gto.Mole()
            mol.stdout = log
            mol.verbose = p["verbose"]
            mol.atom = list(zip(self.atoms.get_chemical_symbols(), self.atoms.positions))
            mol.unit = "Angstrom"
            mol.basis, mol.charge, mol.spin = p["basis"], p["charge"], p["spin"]
            mol.max_memory = p["max_memory"]
            if p["ecp"] is not None:
                mol.ecp = p["ecp"]
            mol.build()
            if mol.nelectron < mol.spin or (mol.nelectron - mol.spin) % 2:
                raise ValueError("charge/spin are inconsistent with the effective electron count.")
            method = p["method"]
            if method == "auto":
                method = "uks" if mol.spin else "rks"
            is_dft = method.endswith("ks")
            mf = getattr(dft if is_dft else scf, method.upper())(mol)
            if is_dft:
                mf.xc = p["xc"]
                if p["nlc"] is not None:
                    mf.nlc = 0 if p["nlc"] is False else p["nlc"]
                if p["disp"] and (dft.libxc.is_nlc(mf.xc) or p["nlc"]):
                    raise DispersionError("D3/D4 cannot be combined with a VV10/NLC functional.")
            if p["density_fit"]:
                mf = mf.density_fit(auxbasis=p["auxbasis"])
            if p["disp"]:
                try:
                    import pyscf.dispersion  # noqa: F401
                except ImportError as exc:
                    raise MissingDependencyError("pyscf-dispersion") from exc
                mf.disp = p["disp"]
            if is_dft:
                for name in ("grids", "nlcgrids"):
                    for key, value in p[name].items():
                        setattr(getattr(mf, name), key, tuple(value) if key == "atom_grid" else value)
            mf.conv_tol, mf.max_cycle = p["conv_tol"], p["max_cycle"]
            if p["solvent"]:
                s = p["solvent"]
                if s["model"] == "smd":
                    mf = mf.SMD()
                    mf.with_solvent.solvent = s["solvent"]
                else:
                    mf = mf.PCM()
                    mf.with_solvent.eps = s["eps"]
                    mf.with_solvent.method = s.get("method", "IEF-PCM")
            if self.gpu:
                mf = mf.to_gpu()
                if not type(mf).__module__.startswith("gpu4pyscf"):
                    raise RuntimeError("GPU4PySCF did not produce a GPU SCF object.")
            energy = float(mf.kernel())
            if not mf.converged or not np.isfinite(energy):
                raise CalculationFailed("PySCF SCF did not converge to a finite energy.")
            results = {"energy": energy * Hartree}
            if "forces" in properties:
                grad = mf.nuc_grad_method().kernel()
                grad = np.asarray(grad.get() if hasattr(grad, "get") else grad, dtype=float)
                if grad.shape != (len(self.atoms), 3) or not np.isfinite(grad).all():
                    raise CalculationFailed("PySCF returned invalid analytic gradients.")
                results["forces"] = -grad * Hartree / Bohr
            self.results = results
            self._state_key = state_key
            self.metadata = {"converged": True, "method": method, "gpu": self.gpu,
                             "charge": mol.charge, "spin": mol.spin,
                             "multiplicity": mol.spin + 1, "energy_hartree": energy,
                             "state_sources": sources}


class PySCFBackend(BaseBackend):
    name = "pyscf"
    gpu = False

    def create_calculator(self, *, config, overrides=None, write_resolved_config=False):
        """Create a molecular calculator from explicit, reviewable conditions."""
        resolved = resolve_calculator_config(self.name, config=config, overrides=overrides)
        _mapping(resolved, {"calculator", "directory", "parameters"}, "config")
        parameters = resolved.get("parameters", {})
        calc = PySCFCalculator(parameters=parameters, gpu=self.gpu,
                               directory=resolved.get("directory", "."))
        try:
            import pyscf  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError("pyscf") from exc
        if self.gpu:
            try:
                import gpu4pyscf  # noqa: F401
            except ImportError as exc:
                raise MissingDependencyError("gpu4pyscf") from exc
        calc._write_config = write_resolved_config
        calc._save_config(calc.settings)
        return calc


class GPU4PySCFBackend(PySCFBackend):
    name = "gpu4pyscf"
    gpu = True
