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
    _get_atom_symbols,
    build_pcm_radii_table,
    format_hessian,
    validate_pyscf_parameters,
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

    def get_hessian(self, atoms=None) -> np.ndarray:
        """Compute and return the analytic Hessian matrix in eV/Angstrom^2.

        Returns
        -------
        np.ndarray
            Shape (3N, 3N) array of second derivatives of energy with respect to
            atomic positions, in eV/Angstrom^2. Atomic ordering follows atoms
            sequence, with x, y, z blocks per atom.
        """
        candidate = atoms if atoms is not None else self.atoms
        if candidate is None:
            raise ValueError("get_hessian requires an Atoms object.")

        if self.calculation_required(candidate, ["energy"]):
            self.calculate(candidate, properties=["energy"])

        mf = self._scf
        if mf is None:
            raise CalculationFailed("No active converged SCF available for Hessian calculation.")

        h = mf.Hessian()
        h_params = self.settings.get("hessian") or {}

        if "conv_tol_cpscf" in h_params:
            if hasattr(mf, "conv_tol_cpscf"):
                mf.conv_tol_cpscf = h_params["conv_tol_cpscf"]
        if "grid_response" in h_params:
            if hasattr(h, "grid_response"):
                h.grid_response = h_params["grid_response"]
        if "auxbasis_response" in h_params:
            if hasattr(h, "auxbasis_response"):
                h.auxbasis_response = h_params["auxbasis_response"]

        try:
            h_raw = h.kernel()
        except Exception as exc:
            if not self.settings.get("retain_scf", False):
                self._discard_scf()
            raise CalculationFailed(f"PySCF Hessian calculation failed: {exc}") from exc

        h_ev_ang2, h_meta = format_hessian(h_raw, len(self.atoms))

        # Record finite-difference details for dispersion and SMD
        if self.settings.get("disp"):
            disp_str = self.settings["disp"]
            if disp_str.startswith("d3"):
                h_meta["dispersion"] = {
                    "method": "finite_difference_of_gradients",
                    "step_size_bohr": 1e-5,
                    "step_size_angstrom": 1e-5 * Bohr,
                }
        if self.settings.get("solvent") and self.settings["solvent"].get("model") == "smd":
            h_meta["smd_cds"] = {
                "method": "finite_difference_of_cds_gradients",
                "step_size_bohr": 1e-4,
                "step_size_angstrom": 1e-4 * Bohr,
            }

        self.metadata["hessian"] = h_meta

        if not self.settings.get("retain_scf", False):
            self._discard_scf()

        return h_ev_ang2

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        candidate = atoms if atoms is not None else self.atoms
        try:
            reuse = self._scf is not None and not self.check_state(candidate)
        except ValueError:
            self.reset()
            raise
        if not reuse:
            self._discard_scf()

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
        method = p["method"]
        if method == "auto":
            method = "uks" if state["spin"] else "rks"

        diagnostics = DiagnosticsCollector(p, self.directory)
        init_guess_source = "default"
        density_reuse_fallback = None

        try:
            self._save_config(p)
            if reuse:
                mf, energy = self._scf, self._scf_energy
                mol = self._scf_mol
                scf_diag = self._scf_diag or {}
            else:
                Path(self.directory).mkdir(parents=True, exist_ok=True)
                self._scf_log = (Path(self.directory) / "pyscf.log").open("a", encoding="utf-8")
                mol = gto.Mole()
                mol.stdout = self._scf_log
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

                # Explicit SCF convergence controls
                if p["conv_tol_grad"] is not None:
                    mf.conv_tol_grad = p["conv_tol_grad"]
                if p["diis_space"] is not None and hasattr(mf, "diis_space"):
                    mf.diis_space = p["diis_space"]
                if p["diis_start_cycle"] is not None and hasattr(mf, "diis_start_cycle"):
                    mf.diis_start_cycle = p["diis_start_cycle"]
                if p["damp"] is not None and hasattr(mf, "damp"):
                    mf.damp = p["damp"]
                if p["level_shift"] is not None and hasattr(mf, "level_shift"):
                    mf.level_shift = p["level_shift"]

                mf.init_guess = p["init_guess"]

                # Solvation handling
                if p["solvent"]:
                    s = p["solvent"]
                    if s["model"] == "smd":
                        mf = mf.SMD()
                        mf.with_solvent.solvent = s["solvent"]
                    else:
                        mf = mf.PCM()
                        if s.get("eps") is not None:
                            mf.with_solvent.eps = s["eps"]
                        mf.with_solvent.method = s.get("method", "IEF-PCM")
                        if "equilibrium_solvation" in s:
                            mf.with_solvent.equilibrium_solvation = s["equilibrium_solvation"]
                        if "lebedev_order" in s:
                            mf.with_solvent.lebedev_order = s["lebedev_order"]
                        if "surface_method" in s:
                            mf.with_solvent.surface_discretization_method = s["surface_method"]
                        # Construct unified Bondi radii table in Bohr
                        radii_table_bohr, eff_radii = build_pcm_radii_table(mol, s)
                        mf.with_solvent.radii_table = radii_table_bohr

                # Second-order SCF (Newton)
                if p["scf_algorithm"] == "newton":
                    mf = mf.newton()

                if self.gpu:
                    mf = mf.to_gpu()
                    if not type(mf).__module__.startswith("gpu4pyscf"):
                        raise RuntimeError("GPU4PySCF did not produce a GPU SCF object.")

                # Initial density dm0 resolution
                dm0 = None
                if p.get("checkpoint") and p["checkpoint"].get("read"):
                    chk_read = p["checkpoint"]["read"]
                    scf_dict, chk_meta = CheckpointManager.load(
                        chk_read, allow_unconverged=p["checkpoint"].get("allow_unconverged", False)
                    )
                    chk_err = CheckpointManager.verify_compatibility(chk_meta, mol, p)
                    if chk_err:
                        raise ValueError(f"Incompatible checkpoint {chk_read}: {chk_err}")
                    mo_c = scf_dict.get("mo_coeff")
                    mo_o = scf_dict.get("mo_occ")
                    if mo_c is not None and mo_o is not None:
                        if self.gpu:
                            import cupy
                            mo_c = cupy.asarray(mo_c)
                            mo_o = cupy.asarray(mo_o)
                        dm0 = mf.make_rdm1(mo_c, mo_o)
                        init_guess_source = "checkpoint"
                elif p["reuse_density"] and self._last_converged_mol is not None:
                    compat_meta = {
                        "atom_symbols": _get_atom_symbols(self._last_converged_mol),
                        "basis": p.get("basis"),
                        "ecp": p.get("ecp"),
                        "charge": mol.charge,
                        "spin": mol.spin,
                        "method": method,
                        "xc": p.get("xc"),
                    }
                    compat_err = CheckpointManager.verify_compatibility(compat_meta, mol, p)
                    if compat_err is None:
                        try:
                            # Project MOs onto new geometry
                            prev_mo = self._last_converged_mo_coeff
                            prev_occ = self._last_converged_mo_occ
                            if method in {"uks", "uhf"} and isinstance(prev_mo, np.ndarray) and prev_mo.ndim == 3:
                                mo_a = scf.project_mo_nr2nr(self._last_converged_mol, prev_mo[0], mol)
                                mo_b = scf.project_mo_nr2nr(self._last_converged_mol, prev_mo[1], mol)
                                mo_proj = np.stack([mo_a, mo_b])
                            else:
                                mo_proj = scf.project_mo_nr2nr(self._last_converged_mol, prev_mo, mol)
                            if self.gpu:
                                import cupy
                                mo_proj = cupy.asarray(mo_proj)
                                prev_occ = cupy.asarray(prev_occ)
                            dm0 = mf.make_rdm1(mo_proj, prev_occ)
                            init_guess_source = "projected_mo"
                        except Exception as proj_exc:
                            density_reuse_fallback = f"Orbital projection failed: {proj_exc}"
                            dm0 = None
                            init_guess_source = "default"
                    else:
                        density_reuse_fallback = compat_err
                        dm0 = None
                        init_guess_source = "default"

                # Checkpoint write configuration
                chk_write = p.get("checkpoint", {}).get("write") if p.get("checkpoint") else None

                # Attach iteration callback
                def _scf_callback(envs):
                    diagnostics.callback(envs)
                    if chk_write:
                        try:
                            CheckpointManager.save(
                                chk_write, mf, mol, converged=False,
                                extra_meta={"method": method, "xc": p.get("xc")}
                            )
                        except Exception:
                            pass

                mf.callback = _scf_callback

                # Run SCF calculation
                try:
                    energy = float(mf.kernel(dm0=dm0) if dm0 is not None else mf.kernel())
                except Exception as exc:
                    scf_diag = diagnostics.collect(mf, mol, False, init_guess_source, "scf_execution")
                    if density_reuse_fallback:
                        scf_diag["density_reuse_fallback"] = density_reuse_fallback
                    diagnostics.write_reports(scf_diag)
                    self.metadata = {"converged": False, "scf": scf_diag}
                    raise CalculationFailed(f"PySCF SCF kernel execution failed: {exc}") from exc

                if not mf.converged or not np.isfinite(energy):
                    scf_diag = diagnostics.collect(mf, mol, False, init_guess_source, "scf_convergence")
                    if density_reuse_fallback:
                        scf_diag["density_reuse_fallback"] = density_reuse_fallback
                    diagnostics.write_reports(scf_diag)
                    self.metadata = {"converged": False, "scf": scf_diag}
                    raise CalculationFailed("PySCF SCF did not converge to a finite energy.")

                # Save final converged checkpoint
                if chk_write:
                    CheckpointManager.save(
                        chk_write, mf, mol, converged=True,
                        extra_meta={"method": method, "xc": p.get("xc")}
                    )

                # Cache converged orbitals for reuse_density
                try:
                    self._last_converged_mol = mol.copy()
                except Exception:
                    self._last_converged_mol = None
                mo_c = getattr(mf, "mo_coeff", None)
                if mo_c is not None:
                    mo_c = mo_c.get() if hasattr(mo_c, "get") else mo_c
                    self._last_converged_mo_coeff = np.asarray(mo_c).copy()
                mo_o = getattr(mf, "mo_occ", None)
                if mo_o is not None:
                    mo_o = mo_o.get() if hasattr(mo_o, "get") else mo_o
                    self._last_converged_mo_occ = np.asarray(mo_o).copy()

                scf_diag = diagnostics.collect(mf, mol, True, init_guess_source)
                if density_reuse_fallback:
                    scf_diag["density_reuse_fallback"] = density_reuse_fallback
                diagnostics.write_reports(scf_diag)

                self._scf, self._scf_energy, self._scf_mol = mf, energy, mol
                self._scf_diag = scf_diag
                self._scf_log.flush()

            results = {"energy": energy * Hartree}

            if "forces" in properties:
                try:
                    grad = mf.nuc_grad_method().kernel()
                except Exception as exc:
                    if "scf_diag" in locals():
                        scf_diag["failure_stage"] = "forces"
                    raise CalculationFailed(f"PySCF nuclear gradient calculation failed: {exc}") from exc
                grad = np.asarray(grad.get() if hasattr(grad, "get") else grad, dtype=float)
                if grad.shape != (len(self.atoms), 3) or not np.isfinite(grad).all():
                    raise CalculationFailed("PySCF returned invalid analytic gradients.")
                results["forces"] = -grad * Hartree / Bohr

            mol = mf.mol
            self.results = results
            self._state_key = state_key
            self.metadata = {
                "converged": True, "method": method, "gpu": self.gpu,
                "charge": mol.charge, "spin": mol.spin,
                "multiplicity": mol.spin + 1, "energy_hartree": energy,
                "state_sources": sources,
                "scf": scf_diag if "scf_diag" in locals() else getattr(self, "metadata", {}).get("scf", {}),
            }

            # Check if SCF should be retained
            if "forces" in results and not p.get("retain_scf", False):
                self._discard_scf()
        except Exception:
            self.results = {}
            if "scf_diag" not in self.metadata:
                self.metadata = {}
            self._discard_scf()
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
