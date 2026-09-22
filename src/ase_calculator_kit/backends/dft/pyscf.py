"""Molecular PySCF adapters: upstream electronic structure, ASE units and caching."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, CalculationFailed, all_changes
from ase.units import Bohr, Hartree

from ...config import resolve_calculator_config, write_resolved_config_file
from ...errors import DispersionError, MissingDependencyError
from ..base import BaseBackend
from ._pyscf_support import (
    CheckpointManager,
    DiagnosticsCollector,
    build_pcm_radii_table,
    format_hessian,
    validate_pyscf_parameters,
    projected_density,
    to_numpy,
)

_parameters = validate_pyscf_parameters


def _electronic_state(settings: dict[str, Any], info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve explicit total charge and multiplicity, rejecting disagreements."""
    state, sources = {}, {}
    for key in ("charge", "spin"):
        av = info.get(key)
        if key in info:
            if isinstance(av, bool) or not isinstance(av, int) and not isinstance(av, np.integer):
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
    """Translate molecular HF/DFT energies, analytic gradients and Hessians to ASE.

    YAML ``spin`` is Nalpha-Nbeta; atoms.info["spin"] is multiplicity.
    Only nonperiodic structures are accepted, even when an ASE cell is present.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, *, parameters: dict[str, Any], gpu: bool = False, directory: str | Path = "."):
        super().__init__(directory=directory)
        self.settings = validate_pyscf_parameters(parameters)
        self.gpu = gpu
        checkpoint = self.settings.get("checkpoint") or {}
        for key in ("read", "write"):
            if key in checkpoint:
                checkpoint[key] = str((Path(self.directory) / checkpoint[key]).resolve())
        if checkpoint.get("read") == checkpoint.get("write") and checkpoint.get("read"):
            raise ValueError("checkpoint.read and checkpoint.write cannot point to the same file.")
        diagnostics = self.settings.get("diagnostics") or {}
        paths = [Path(self.directory) / "pyscf.log",
                 Path(self.directory) / "resolved_calculator_config.yaml"]
        paths.extend(Path(v) for k, v in checkpoint.items() if k in {"read", "write"})
        if diagnostics.get("save"):
            paths.append(Path(self.directory) / diagnostics.get("summary_file", "scf_diagnostics.json"))
            if diagnostics.get("iterations"):
                paths.append(Path(self.directory) / diagnostics.get("iteration_file", "scf_iterations.jsonl"))
        if len({p.resolve() for p in paths}) != len(paths):
            raise ValueError("Checkpoint, diagnostic and calculation log paths must be distinct.")
        self.metadata: dict[str, Any] = {}
        self._state_key = None
        self._write_config = False
        self._scf = None
        self._scf_energy = None
        self._scf_mol = None
        self._scf_log = None
        self._scf_diag = None
        self._last_converged_mol = None
        self._last_converged_mo_coeff = None
        self._last_converged_mo_occ = None
        self._last_converged_meta = None
        self._checkpoint_read_done = False

    def _discard_scf(self) -> None:
        self._scf = None
        self._scf_energy = None
        self._scf_mol = None
        log = getattr(self, "_scf_log", None)
        self._scf_log = None
        if log is not None:
            log.close()

    def reset(self) -> None:
        super().reset()
        self._discard_scf()
        self._state_key = None
        self.metadata = {}
        self._scf_diag = None
        self._last_converged_mol = None
        self._last_converged_mo_coeff = None
        self._last_converged_mo_occ = None
        self._last_converged_meta = None
        self._checkpoint_read_done = False

    def _state(self, atoms):
        state, sources = _electronic_state(self.settings, atoms.info)
        key = (state["charge"], state["spin"], sources["charge"], sources["spin"])
        return state, sources, key

    def check_state(self, atoms, tol: float = 1e-15) -> list[str]:
        changes = super().check_state(atoms, tol)
        if self._state(atoms)[2] != self._state_key:
            changes.append("electronic_state")
        return changes

    def get_property(self, name: str, atoms=None, allow_calculation: bool = True):
        candidate = atoms if atoms is not None else self.atoms
        if candidate is not None:
            try:
                if self._state(candidate)[2] != self._state_key:
                    self.results = {}
                    self._discard_scf()
            except ValueError:
                self._discard_scf()
                self.results = {}
                self.metadata = {}
                raise
        return super().get_property(name, atoms, allow_calculation)

    def _save_config(self, parameters: dict[str, Any]) -> None:
        if self._write_config:
            saved = deepcopy(parameters)
            if saved["method"] in {"rhf", "uhf"}:
                saved.pop("xc", None)
            write_resolved_config_file(
                {"calculator": "gpu4pyscf" if self.gpu else "pyscf",
                 "directory": str(self.directory), "parameters": saved}, str(self.directory)
            )

    def set(self, **kwargs):
        if kwargs:
            raise TypeError("Configure PySCF through config/overrides in the factory, not calc.set().")
        return {}

    def _record_failure(self, collector, mf, mol, source, stage, error):
        """Keep diagnostic evidence while invalidating all usable ASE results."""
        diag = collector.collect(mf, mol, bool(getattr(mf, "converged", False)), source, stage)
        diag["error"] = f"{type(error).__name__}: {error}"
        diag["results_valid"] = False
        self.metadata = {"converged": diag["converged"], "scf": diag}
        self.results = {}
        self._discard_scf()
        try:
            collector.write_reports(diag)
        except Exception as report_error:
            diag["report_error"] = str(report_error)

    def get_hessian(self, atoms=None) -> np.ndarray:
        """Return upstream Cartesian Hessian as (3N, 3N), in eV/Angstrom^2.

        Dispersion and SMD CDS retain upstream's finite-difference corrections.
        No finite-difference SCF scan or automatic CPU fallback is performed.
        """
        candidate = atoms if atoms is not None else self.atoms
        if candidate is None:
            raise ValueError("get_hessian requires an Atoms object.")
        try:
            required = self.check_state(candidate) or self._scf is None
        except ValueError:
            self.reset()
            raise
        if required:
            self.calculate(candidate, properties=["energy"])
        mf = self._scf
        collector = self._collector
        try:
            if not self.gpu and self.metadata.get("method") == "uks" and mf.do_nlc():
                raise NotImplementedError(
                    "PySCF 2.14 does not implement CPU UKS Hessians with NLC/VV10; "
                    "energy and forces remain supported. Select GPU4PySCF explicitly for this Hessian."
                )
            h = mf.Hessian()
            h_params = self.settings.get("hessian") or {}
            for key, value in h_params.items():
                target = mf if key == "conv_tol_cpscf" else h
                if not hasattr(target, key):
                    raise NotImplementedError(f"This Hessian does not support {key}.")
                setattr(target, key, value)
            h_ev_ang2, meta = format_hessian(h.kernel(), len(candidate))
            meta["response"] = {
                key: getattr(mf if key == "conv_tol_cpscf" else h, key, None)
                for key in ("conv_tol_cpscf", "grid_response", "auxbasis_response")
            }
            if self.settings.get("disp"):
                meta["dispersion"] = {"method": "finite_difference_of_gradients",
                                      "step_size_bohr": 1e-5, "step_size_angstrom": 1e-5 * Bohr}
            if (self.settings.get("solvent") or {}).get("model") == "smd":
                meta["smd_cds"] = {"method": "finite_difference_of_cds_gradients",
                                   "step_size_bohr": 1e-4, "step_size_angstrom": 1e-4 * Bohr}
            self.metadata["hessian"] = meta
            self.metadata["scf"]["hessian"] = meta
            collector.write_reports(self.metadata["scf"])
        except Exception as exc:
            self._record_failure(collector, mf, mf.mol,
                                 self.metadata.get("scf", {}).get("init_guess_source"), "hessian", exc)
            raise
        if not self.settings["retain_scf"]:
            self._discard_scf()
        return h_ev_ang2

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        candidate = atoms if atoms is not None else self.atoms
        if candidate is None:
            raise ValueError("PySCF requires an Atoms object.")
        try:
            reuse = self._scf is not None and not self.check_state(candidate)
        except ValueError:
            self.reset()
            raise
        if not reuse:
            self._discard_scf()
        super().calculate(candidate, properties, system_changes)
        self.results = {}
        self.metadata = {}
        collector = self._collector if reuse else DiagnosticsCollector(self.settings, self.directory)
        mf, mol, source, stage = self._scf, self._scf_mol, "default", "input"
        fallback = None
        try:
            if self.atoms.pbc.any():
                raise ValueError("PySCF backend supports molecules only; all atoms.pbc must be false.")
            if not len(self.atoms) or not np.isfinite(self.atoms.positions).all():
                raise ValueError("PySCF requires a nonempty structure with finite coordinates.")
            from pyscf import dft, gto, scf
            state, sources, state_key = self._state(self.atoms)
            p = self.settings | state
            method = p["method"]
            if method == "auto":
                method = "uks" if state["spin"] else "rks"
            self._save_config(p)
            if reuse:
                energy = self._scf_energy
                diag = self._scf_diag
                source = diag["init_guess_source"]
            else:
                stage = "build"
                Path(self.directory).mkdir(parents=True, exist_ok=True)
                self._scf_log = (Path(self.directory) / "pyscf.log").open("a", encoding="utf-8")
                mol = gto.Mole()
                mol.stdout, mol.verbose = self._scf_log, p["verbose"]
                mol.atom = list(zip(self.atoms.get_chemical_symbols(), self.atoms.positions))
                mol.unit = "Angstrom"
                mol.basis, mol.charge, mol.spin = p["basis"], p["charge"], p["spin"]
                mol.max_memory = p["max_memory"]
                if p["ecp"] is not None:
                    mol.ecp = p["ecp"]
                mol.build()
                if mol.nelectron < mol.spin or (mol.nelectron - mol.spin) % 2:
                    raise ValueError("charge/spin are inconsistent with the effective electron count.")
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
                mf.conv_tol, mf.max_cycle, mf.init_guess = p["conv_tol"], p["max_cycle"], p["init_guess"]
                for key in ("conv_tol_grad", "diis_space", "diis_start_cycle", "damp", "level_shift"):
                    if p[key] is not None:
                        if not hasattr(mf, key):
                            raise NotImplementedError(f"SCF does not support {key}.")
                        setattr(mf, key, p[key])
                effective_radii = None
                if p["solvent"]:
                    solvent = p["solvent"]
                    if solvent["model"] == "smd":
                        mf = mf.SMD()
                        mf.with_solvent.solvent = solvent["solvent"]
                    else:
                        mf = mf.PCM()
                        mf.with_solvent.eps = solvent["eps"]
                        mf.with_solvent.method = solvent.get("method", "IEF-PCM")
                        for key in ("equilibrium_solvation", "lebedev_order"):
                            if key in solvent:
                                setattr(mf.with_solvent, key, solvent[key])
                        table, effective_radii = build_pcm_radii_table(mol, solvent)
                        mf.with_solvent.radii_table = table
                        collector.details["effective_radii_angstrom"] = effective_radii
                if self.gpu:
                    mf = mf.to_gpu()
                    if not type(mf).__module__.startswith("gpu4pyscf"):
                        raise RuntimeError("GPU4PySCF did not produce a GPU SCF object.")
                # Validate on the actual device implementation; never set ignored attributes.
                if (p.get("solvent") or {}).get("model") == "pcm":
                    solvent = p["solvent"]
                    if "surface_method" in solvent:
                        if not hasattr(mf.with_solvent, "surface_discretization_method"):
                            raise NotImplementedError("PCM implementation has no surface_method setting.")
                        mf.with_solvent.surface_discretization_method = solvent["surface_method"]
                checkpoint = p.get("checkpoint") or {}
                snapshot = None
                if checkpoint or p["reuse_density"]:
                    snapshot = CheckpointManager.metadata(mol, p)
                dm0 = None
                stage = "initial_guess"
                if checkpoint.get("read") and not self._checkpoint_read_done:
                    orbitals, meta = CheckpointManager.load(checkpoint["read"], checkpoint.get("allow_unconverged", False))
                    collector.details["checkpoint_source"] = {"path": checkpoint["read"],
                                                               "converged": meta["converged"]}
                    error = CheckpointManager.verify_compatibility(meta, mol, p)
                    if error:
                        raise ValueError(f"Incompatible checkpoint: {error}")
                    dm0 = projected_density(mf, orbitals["mol"], orbitals["mo_coeff"], orbitals["mo_occ"], self.gpu)
                    source = "checkpoint"
                elif p["reuse_density"] and self._last_converged_mol is not None:
                    error = CheckpointManager.verify_compatibility(self._last_converged_meta, mol, p)
                    if error:
                        fallback = error
                    else:
                        try:
                            dm0 = projected_density(mf, self._last_converged_mol,
                                                    self._last_converged_mo_coeff,
                                                    self._last_converged_mo_occ, self.gpu)
                            source = "projected_mo"
                        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                            fallback = f"Orbital projection failed: {exc}"
                def callback(envs):
                    nonlocal stage
                    collector.callback(envs)
                    if checkpoint.get("write"):
                        stage = "checkpoint"
                        # mf.mo_* are not necessarily updated until kernel returns.
                        # The callback must not capture mf and create a cycle
                        # retaining GPU arrays after _discard_scf().
                        CheckpointManager.save(checkpoint["write"], None, mol, False, snapshot, envs)
                        stage = "scf_execution"
                mf.callback = callback
                stage = "scf_execution"
                if p["scf_algorithm"] == "newton":
                    base = mf
                    if dm0 is None:
                        dm0 = base.get_init_guess(mol, p["init_guess"])
                    # Build target-device MOs explicitly: upstream Newton.from_dm
                    # can return CPU/list orbitals even for a GPU solver.
                    # dm is named dm_or_wfn by GPU solvent wrappers.
                    fock = base.get_fock(None, None, None, dm0)
                    overlap = base.get_ovlp()
                    if self.gpu:
                        import cupy
                        overlap = cupy.asarray(overlap)
                        # Match GPU4PySCF's own SCF/Newton kernel, including its
                        # orthogonalization rather than the generalized eigensolver.
                        mo_energy, mo_coeff = base.eig(fock, overlap,
                                                      x=base.check_linear_dependency(overlap))
                    else:
                        mo_energy, mo_coeff = base.eig(fock, overlap)
                    mo_occ = base.get_occ(mo_energy, mo_coeff)
                    mf = base.newton()
                    mf.callback = callback
                    if self.gpu:
                        import cupy
                        mo_coeff, mo_occ = cupy.asarray(to_numpy(mo_coeff)), cupy.asarray(to_numpy(mo_occ))
                    energy = float(mf.kernel(mo_coeff=mo_coeff, mo_occ=mo_occ))
                    mf = mf.undo_soscf()
                else:
                    energy = float(mf.kernel(dm0=dm0) if dm0 is not None else mf.kernel())
                stage = "scf_convergence"
                if not mf.converged or not np.isfinite(energy):
                    raise CalculationFailed("PySCF SCF did not converge to a finite energy.")
                if checkpoint.get("write"):
                    stage = "checkpoint"
                    CheckpointManager.save(checkpoint["write"], mf, mol, True, snapshot)
                self._checkpoint_read_done = True
                if p["reuse_density"]:
                    self._last_converged_mol = mol.copy()
                    self._last_converged_mo_coeff = to_numpy(mf.mo_coeff).copy()
                    self._last_converged_mo_occ = to_numpy(mf.mo_occ).copy()
                    self._last_converged_meta = snapshot
                diag = collector.collect(mf, mol, True, source)
                if effective_radii is not None:
                    diag["effective_radii_angstrom"] = effective_radii
                if fallback:
                    diag["density_reuse_fallback"] = fallback
                self._scf, self._scf_energy, self._scf_mol = mf, energy, mol
                self._scf_diag, self._collector = diag, collector
                self._scf_log.flush()
            result = {"energy": energy * Hartree}
            if "forces" in properties:
                stage = "forces"
                grad = to_numpy(mf.nuc_grad_method().kernel())
                if grad.shape != (len(self.atoms), 3) or not np.isfinite(grad).all():
                    raise CalculationFailed("PySCF returned invalid analytic gradients.")
                result["forces"] = -grad * Hartree / Bohr
            self.metadata = {
                "converged": True, "method": method, "gpu": self.gpu,
                "charge": mol.charge, "spin": mol.spin, "multiplicity": mol.spin + 1,
                "energy_hartree": energy, "state_sources": sources, "scf": diag,
            }
            stage = "diagnostics"
            collector.write_reports(diag)
            self.results, self._state_key = result, state_key
            if "forces" in result and not p["retain_scf"]:
                self._discard_scf()
        except Exception as exc:
            self._record_failure(collector, mf, mol, source, stage, exc)
            raise


class PySCFBackend(BaseBackend):
    name = "pyscf"
    gpu = False

    def create_calculator(self, *, config, overrides=None, write_resolved_config=False):
        """Create a molecular calculator from explicit, reviewable conditions."""
        resolved = resolve_calculator_config(self.name, config=config, overrides=overrides)
        if not isinstance(resolved, dict):
            raise ValueError("config must be a mapping.")
        unknown = set(resolved) - {"calculator", "directory", "parameters"}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
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
