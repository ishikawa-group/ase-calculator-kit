"""Auxiliary support functions for molecular PySCF / GPU4PySCF calculators.

Handles parameter validation, checkpoint management, SCF diagnostics,
PCM/COSMO cavity radii construction, and Hessian formatting.
"""

from __future__ import annotations

from copy import deepcopy
import json
from numbers import Integral
import os
from pathlib import Path
from typing import Any

import numpy as np
from ase.units import Bohr, Hartree


_DEFAULT_PARAMETERS = {
    "method": "auto", "xc": "pbe", "ecp": None,
    "density_fit": False, "auxbasis": None, "nlc": None, "disp": None,
    "grids": {}, "nlcgrids": {}, "conv_tol": 1e-9, "max_cycle": 200,
    "max_memory": 4000, "verbose": 4, "solvent": None,
    # 0.6.0 additions:
    "scf_algorithm": "cdiis",
    "init_guess": "minao",
    "conv_tol_grad": None,
    "diis_space": None,
    "diis_start_cycle": None,
    "damp": None,
    "level_shift": None,
    "reuse_density": False,
    "checkpoint": None,
    "diagnostics": None,
    "hessian": None,
    "retain_scf": False,
}

_GRID_KEYS = {"level", "atom_grid", "prune"}
_INIT_GUESSES = {"minao", "atom", "1e", "huckel", "mod_huckel"}
_DISP_VERSIONS = {"d3bj", "d3zero", "d4"}


def _mapping(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")


def validate_pyscf_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize PySCF calculator parameters."""
    _mapping(raw, set(_DEFAULT_PARAMETERS) | {"basis", "charge", "spin", "multiplicity"}, "parameters")
    p = deepcopy(_DEFAULT_PARAMETERS) | deepcopy(raw)

    if not p.get("basis"):
        raise ValueError("PySCF requires parameters.basis.")

    for key in ("charge", "spin", "multiplicity"):
        if key in p:
            val = p[key]
            if isinstance(val, bool) or not isinstance(val, Integral):
                raise ValueError(f"{key} must be an integer.")
            p[key] = int(val)

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

    # Dispersion validation
    disp_val = p["disp"]
    if disp_val is not None:
        if not isinstance(disp_val, str) or not disp_val.strip():
            raise ValueError("disp must be null, or a dispersion specification like 'd3bj', 'd3zero', 'd4'.")
        disp_raw = disp_val.strip()
        disp_base = disp_raw.split(":", 1)[0].lower()
        if disp_base not in _DISP_VERSIONS:
            raise ValueError(f"disp prefix must be one of {sorted(_DISP_VERSIONS)}; got {disp_base!r}.")
        if ":" in disp_raw:
            func = disp_raw.split(":", 1)[1].strip()
            if not func:
                raise ValueError("disp functional specification after colon cannot be empty.")
            p["disp"] = f"{disp_base}:{func}"
        else:
            p["disp"] = disp_base

    if not isinstance(p["xc"], str) or not p["xc"]:
        raise ValueError("xc must be a nonempty PySCF functional name.")
    if "-d3" in p["xc"].lower() or "-d4" in p["xc"].lower():
        raise ValueError("Specify dispersion with disp, not an xc suffix.")

    if p["method"] in {"rhf", "uhf"}:
        if p["nlc"] or "xc" in raw or p["grids"] or p["nlcgrids"]:
            raise ValueError("HF does not accept xc, nlc or DFT grids.")

    # Numerical tolerances and sizes
    for name in ("conv_tol", "max_memory"):
        if (isinstance(p[name], bool) or not isinstance(p[name], (int, float))
                or not np.isfinite(p[name]) or p[name] <= 0):
            raise ValueError(f"{name} must be a finite positive number.")
    if type(p["max_cycle"]) is not int or p["max_cycle"] < 1:
        raise ValueError("max_cycle must be a positive integer.")
    if type(p["verbose"]) is not int or not 0 <= p["verbose"] <= 9:
        raise ValueError("verbose must be an integer from 0 to 9.")

    # Grids
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

    # SCF algorithm and DIIS options
    if p["scf_algorithm"] not in {"cdiis", "newton"}:
        raise ValueError("scf_algorithm must be 'cdiis' or 'newton'.")
    if p["scf_algorithm"] == "newton":
        for diis_key in ("diis_space", "diis_start_cycle", "damp"):
            if p.get(diis_key) is not None:
                raise ValueError(f"scf_algorithm 'newton' does not accept {diis_key}.")

    if p["init_guess"] not in _INIT_GUESSES:
        raise ValueError(f"init_guess must be one of {sorted(_INIT_GUESSES)}; got {p['init_guess']!r}.")

    if p["conv_tol_grad"] is not None:
        v = p["conv_tol_grad"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
            raise ValueError("conv_tol_grad must be a finite positive number.")
    if p["diis_space"] is not None:
        v = p["diis_space"]
        if type(v) is not int or v < 1:
            raise ValueError("diis_space must be a positive integer.")
    if p["diis_start_cycle"] is not None:
        v = p["diis_start_cycle"]
        if type(v) is not int or v < 1:
            raise ValueError("diis_start_cycle must be a positive integer.")
    if p["damp"] is not None:
        v = p["damp"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError("damp must be a finite nonnegative number.")
    if p["level_shift"] is not None:
        v = p["level_shift"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError("level_shift must be a finite nonnegative number.")

    if type(p["reuse_density"]) is not bool:
        raise ValueError("reuse_density must be true or false.")
    if type(p["retain_scf"]) is not bool:
        raise ValueError("retain_scf must be true or false.")

    # Solvent settings
    if p["solvent"] is not None:
        s = p["solvent"]
        _mapping(s, {
            "model", "solvent", "eps", "method", "equilibrium_solvation",
            "lebedev_order", "vdw_scale", "r_probe", "radii", "surface_method"
        }, "solvent")
        if s.get("model") == "smd":
            if not isinstance(s.get("solvent"), str) or not s["solvent"].strip():
                raise ValueError("SMD requires model=smd and solvent=<name>.")
            allowed_smd = {"model", "solvent"}
            if set(s) - allowed_smd:
                raise ValueError(f"SMD only accepts {allowed_smd}.")
        elif s.get("model") == "pcm":
            eps = s.get("eps")
            if eps is not None:
                if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not np.isfinite(eps) or eps <= 1:
                    raise ValueError("PCM requires eps > 1.")
            if s.get("method", "IEF-PCM") not in {"C-PCM", "COSMO", "IEF-PCM", "SS(V)PE"}:
                raise ValueError("Unsupported PCM method.")
            if "equilibrium_solvation" in s and type(s["equilibrium_solvation"]) is not bool:
                raise ValueError("solvent.equilibrium_solvation must be true or false.")
            if "lebedev_order" in s and (type(s["lebedev_order"]) is not int or s["lebedev_order"] < 1):
                raise ValueError("solvent.lebedev_order must be a positive integer.")
            if "vdw_scale" in s:
                v = s["vdw_scale"]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                    raise ValueError("solvent.vdw_scale must be a finite positive number.")
            if "r_probe" in s:
                v = s["r_probe"]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                    raise ValueError("solvent.r_probe must be a finite nonnegative number in Angstrom.")
            if "radii" in s:
                r_dict = s["radii"]
                if not isinstance(r_dict, dict):
                    raise ValueError("solvent.radii must be a mapping of element symbols to radii in Angstrom.")
                for elem, val in r_dict.items():
                    if not isinstance(elem, str) or isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or val <= 0:
                        raise ValueError(f"Invalid radius for element {elem!r}: must be a positive float in Angstrom.")
            if "surface_method" in s and not isinstance(s["surface_method"], str):
                raise ValueError("solvent.surface_method must be a string.")
        else:
            raise ValueError("solvent.model must be smd or pcm.")

    # Checkpoint settings
    if p["checkpoint"] is not None:
        c = p["checkpoint"]
        _mapping(c, {"read", "write", "allow_unconverged"}, "checkpoint")
        for k in ("read", "write"):
            if k in c and c[k] is not None and not isinstance(c[k], (str, Path)):
                raise ValueError(f"checkpoint.{k} must be a file path.")
        if "allow_unconverged" in c and type(c["allow_unconverged"]) is not bool:
            raise ValueError("checkpoint.allow_unconverged must be true or false.")
        if c.get("read") and c.get("write") and str(Path(c["read"]).resolve()) == str(Path(c["write"]).resolve()):
            raise ValueError("checkpoint.read and checkpoint.write cannot point to the same file.")

    # Diagnostics settings
    if p["diagnostics"] is not None:
        d = p["diagnostics"]
        _mapping(d, {"save", "iterations", "summary_file", "iteration_file"}, "diagnostics")
        for k in ("save", "iterations"):
            if k in d and type(d[k]) is not bool:
                raise ValueError(f"diagnostics.{k} must be true or false.")

    # Hessian settings
    if p["hessian"] is not None:
        h = p["hessian"]
        _mapping(h, {"conv_tol_cpscf", "grid_response", "auxbasis_response"}, "hessian")
        if "conv_tol_cpscf" in h:
            v = h["conv_tol_cpscf"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
                raise ValueError("hessian.conv_tol_cpscf must be a finite positive number.")
        if "grid_response" in h and type(h["grid_response"]) is not bool:
            raise ValueError("hessian.grid_response must be true or false.")
        if "auxbasis_response" in h:
            if type(h["auxbasis_response"]) is not bool:
                raise ValueError("hessian.auxbasis_response must be true or false.")
            if h["auxbasis_response"] and not p["density_fit"]:
                raise ValueError("hessian.auxbasis_response requires density_fit=true.")

    return p


def _get_atom_symbols(mol) -> list[str]:
    """Extract list of element symbols from PySCF mol or mock object."""
    if hasattr(mol, "atom_symbols") and callable(mol.atom_symbols):
        return list(mol.atom_symbols())
    if hasattr(mol, "natm") and hasattr(mol, "atom_symbol"):
        return [mol.atom_symbol(i) for i in range(mol.natm)]
    if hasattr(mol, "_atom"):
        return [atom[0] for atom in mol._atom]
    return []


def build_pcm_radii_table(mol, solvent_settings: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    """Build a PySCF-compatible Bondi radii table in Bohr, absorbing CPU/GPU differences.

    All user length inputs (vdw_scale, r_probe, element radii) are in Angstrom.
    Element-specific radii are treated as final values (no vdw_scale applied).
    Returns (radii_table_bohr, effective_radii_angstrom).
    """
    from ase.data import atomic_numbers

    try:
        from pyscf.data import radii
    except ImportError:
        radii_vdw = np.full(120, 2.0)
        radii_bohr_val = Bohr
    else:
        radii_vdw = radii.VDW.copy()
        radii_vdw[1] = 1.1 / radii.BOHR
        radii_bohr_val = radii.BOHR

    vdw_scale = float(solvent_settings.get("vdw_scale", 1.2))
    r_probe_ang = float(solvent_settings.get("r_probe", 0.0))
    user_radii = solvent_settings.get("radii") or {}

    max_z = max(len(radii_vdw), 120)
    radii_table_bohr = np.zeros(max_z, dtype=float)
    effective_radii_ang = {}

    unique_symbols = set(_get_atom_symbols(mol))
    for symb in unique_symbols:
        z = atomic_numbers.get(symb, 1)
        if symb in user_radii:
            final_r_ang = float(user_radii[symb])
        else:
            base_bondi_bohr = float(radii_vdw[z]) if z < len(radii_vdw) else 2.0
            base_bondi_ang = base_bondi_bohr * radii_bohr_val
            final_r_ang = vdw_scale * base_bondi_ang + r_probe_ang

        effective_radii_ang[symb] = final_r_ang
        if z < max_z:
            radii_table_bohr[z] = final_r_ang / radii_bohr_val

    return radii_table_bohr, effective_radii_ang


class CheckpointManager:
    """Read, write, and verify PySCF-compatible HDF5 checkpoints."""

    @staticmethod
    def verify_compatibility(meta: dict[str, Any], mol, settings: dict[str, Any]) -> str | None:
        """Verify that checkpoint metadata matches current mol and settings.

        Returns None if compatible, or an explanation string if incompatible.
        """
        current_symbols = _get_atom_symbols(mol)
        chk_symbols = meta.get("atom_symbols")
        if chk_symbols is not None and chk_symbols != current_symbols:
            return f"Atom symbols mismatch: current={current_symbols}, checkpoint={chk_symbols}"

        if meta.get("basis") is not None and meta["basis"] != settings.get("basis"):
            return f"Basis mismatch: current={settings.get('basis')}, checkpoint={meta['basis']}"

        current_ecp = settings.get("ecp") or {}
        chk_ecp = meta.get("ecp") or {}
        if current_ecp != chk_ecp:
            return f"ECP mismatch: current={settings.get('ecp')}, checkpoint={meta.get('ecp')}"

        if meta.get("charge") is not None and meta["charge"] != mol.charge:
            return f"Charge mismatch: current={mol.charge}, checkpoint={meta['charge']}"

        if meta.get("spin") is not None and meta["spin"] != mol.spin:
            return f"Spin mismatch: current={mol.spin}, checkpoint={meta['spin']}"

        current_method = settings.get("method")
        chk_method = meta.get("method")
        if chk_method and current_method not in {"auto", chk_method}:
            return f"Method mismatch: current={current_method}, checkpoint={chk_method}"

        if settings.get("xc") and meta.get("xc") and settings["xc"].lower() != meta["xc"].lower():
            return f"XC functional mismatch: current={settings['xc']}, checkpoint={meta['xc']}"

        return None

    @staticmethod
    def save(filepath: str | Path, mf, mol, converged: bool, extra_meta: dict[str, Any] | None = None) -> None:
        """Save PySCF checkpoint safely using temporary file and atomic replace."""
        target = Path(filepath).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")

        try:
            import h5py
            from pyscf.scf import chkfile

            # Extract arrays from GPU or CPU
            def _to_numpy(val):
                if val is None:
                    return None
                if hasattr(val, "get"):
                    return val.get()
                return np.asarray(val)

            mo_energy = _to_numpy(getattr(mf, "mo_energy", None))
            mo_coeff = _to_numpy(getattr(mf, "mo_coeff", None))
            mo_occ = _to_numpy(getattr(mf, "mo_occ", None))
            e_tot = float(getattr(mf, "e_tot", 0.0))

            chkfile.dump_scf(mol, str(tmp), e_tot, mo_energy, mo_coeff, mo_occ)

            meta = {
                "converged": bool(converged),
                "atom_symbols": _get_atom_symbols(mol),
                "charge": int(getattr(mol, "charge", 0)),
                "spin": int(getattr(mol, "spin", 0)),
                "basis": getattr(mol, "basis", None),
                "ecp": getattr(mol, "ecp", None),
            }
            if extra_meta:
                meta.update(extra_meta)

            with h5py.File(str(tmp), "a") as f:
                grp = f.require_group("ase_calculator_kit")
                grp.attrs["metadata_json"] = json.dumps(meta, default=str)

            os.replace(tmp, target)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    @staticmethod
    def load(filepath: str | Path, allow_unconverged: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load PySCF checkpoint and kit metadata.

        Returns (scf_dict, meta_dict).
        """
        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            import h5py
            from pyscf.scf import chkfile
        except ImportError as exc:
            raise RuntimeError("Loading checkpoints requires pyscf and h5py.") from exc

        meta: dict[str, Any] = {}
        with h5py.File(str(path), "r") as f:
            if "ase_calculator_kit" in f and "metadata_json" in f["ase_calculator_kit"].attrs:
                meta = json.loads(f["ase_calculator_kit"].attrs["metadata_json"])

        converged = meta.get("converged", True)
        if not converged and not allow_unconverged:
            raise ValueError(
                f"Checkpoint {path} contains unconverged SCF results. "
                "Set checkpoint.allow_unconverged: true to resume from an unconverged checkpoint."
            )

        _, scf_dict = chkfile.load_scf(str(path))
        return scf_dict, meta


class DiagnosticsCollector:
    """Collect structured SCF diagnostics and write optional JSON/JSONL reports."""

    def __init__(self, parameters: dict[str, Any], directory: str | Path):
        self.parameters = parameters
        self.directory = Path(directory)
        self.iterations: list[dict[str, Any]] = []
        self.last_cycle_info: dict[str, Any] = {}

    def callback(self, envs: dict[str, Any]) -> None:
        """Callback attached to mf.callback to record per-iteration progress."""
        cycle = int(envs.get("cycle", len(self.iterations)))
        e_tot = float(envs.get("e_tot", np.nan))
        last_e = float(envs.get("last_hf_e", np.nan))
        delta_e = float(e_tot - last_e) if np.isfinite(e_tot) and np.isfinite(last_e) else None

        norm_gorb = envs.get("norm_gorb")
        if norm_gorb is not None:
            try:
                norm_gorb = float(norm_gorb.get() if hasattr(norm_gorb, "get") else norm_gorb)
            except Exception:
                norm_gorb = None

        norm_ddm = envs.get("norm_ddm")
        if norm_ddm is not None:
            try:
                norm_ddm = float(norm_ddm.get() if hasattr(norm_ddm, "get") else norm_ddm)
            except Exception:
                norm_ddm = None

        iter_data = {
            "cycle": cycle,
            "energy_hartree": e_tot,
            "delta_energy_hartree": delta_e,
            "norm_gorb": norm_gorb,
            "norm_ddm": norm_ddm,
        }
        self.iterations.append(iter_data)
        self.last_cycle_info = iter_data

    def collect(self, mf, mol, converged: bool, init_guess_source: str, failure_stage: str | None = None) -> dict[str, Any]:
        """Compile complete structured metadata["scf"] dictionary."""
        natm = getattr(mol, "natm", 0)
        if hasattr(natm, "__call__"):
            natm = natm()
        nao = getattr(mol, "nao", 0)
        if hasattr(nao, "__call__"):
            nao = nao()
        diag: dict[str, Any] = {
            "converged": bool(converged),
            "cycles": len(self.iterations),
            "scf_algorithm": self.parameters.get("scf_algorithm", "cdiis"),
            "init_guess_source": init_guess_source,
            "failure_stage": failure_stage,
            "dimensions": {
                "natm": int(natm),
                "nao": int(nao),
            },
            "memory": {
                "max_memory_mb": float(self.parameters.get("max_memory", 4000)),
            },
        }

        # Energy components from upstream scf_summary
        summary = getattr(mf, "scf_summary", {}) or {}
        components = {}
        for k, v in summary.items():
            try:
                components[k] = float(v.get() if hasattr(v, "get") else v)
            except Exception:
                pass
        diag["energy_components"] = components

        # Final cycle metrics
        diag["final_cycle"] = self.last_cycle_info

        # Density fit / auxiliary basis
        if self.parameters.get("density_fit") and getattr(mf, "with_df", None):
            auxmol = getattr(mf.with_df, "auxmol", None)
            if auxmol is not None:
                diag["dimensions"]["naux"] = int(getattr(auxmol, "nao", 0))

        # DFT grids
        if getattr(mf, "grids", None) and hasattr(mf.grids, "coords"):
            coords = getattr(mf.grids, "coords", None)
            if coords is not None:
                diag["dimensions"]["ngrids"] = int(coords.shape[0])

        # Spin analysis
        mol_spin = getattr(mol, "spin", 0)
        if hasattr(mf, "spin_square"):
            try:
                ss, mult = mf.spin_square()
                ss = float(ss.get() if hasattr(ss, "get") else ss)
                mult = float(mult.get() if hasattr(mult, "get") else mult)
                s_ideal = float(mol_spin) / 2.0
                ideal_ss = s_ideal * (s_ideal + 1.0)
                diag["spin_analysis"] = {
                    "s2": ss,
                    "ideal_s2": ideal_ss,
                    "s2_deviation": ss - ideal_ss,
                    "multiplicity_from_s2": mult,
                }
            except Exception:
                pass

        # Mulliken spin population for open-shell
        if mol_spin > 0 and hasattr(mf, "mulliken_pop"):
            try:
                dm = mf.make_rdm1()
                if hasattr(dm, "get"):
                    dm = dm.get()
                s_ovlp = mf.get_ovlp()
                if hasattr(s_ovlp, "get"):
                    s_ovlp = s_ovlp.get()
                if isinstance(dm, np.ndarray) and dm.ndim == 3 and dm.shape[0] == 2:
                    pop_a = np.einsum("ij,ji->i", dm[0], s_ovlp).real
                    pop_b = np.einsum("ij,ji->i", dm[1], s_ovlp).real
                    nelec_a = np.zeros(mol.natm)
                    nelec_b = np.zeros(mol.natm)
                    for i, (ia, *_) in enumerate(mol.ao_labels(fmt=None)):
                        nelec_a[ia] += pop_a[i]
                        nelec_b[ia] += pop_b[i]
                    spin_pop = (nelec_a - nelec_b).tolist()
                    if "spin_analysis" not in diag:
                        diag["spin_analysis"] = {}
                    diag["spin_analysis"]["mulliken_spin_populations"] = spin_pop
            except Exception:
                pass

        # GPU memory if available
        try:
            import cupy
            free_b, total_b = cupy.cuda.Device().mem_info
            diag["memory"]["gpu_vram_free_mb"] = float(free_b / 1024**2)
            diag["memory"]["gpu_vram_total_mb"] = float(total_b / 1024**2)
        except Exception:
            pass

        # Newton notes on NLC response
        if self.parameters.get("scf_algorithm") == "newton" and self.parameters.get("nlc"):
            diag["newton_notes"] = (
                "Newton AH second-order response does not include NLC (VV10) response "
                "per upstream PySCF implementation."
            )

        return diag

    def write_reports(self, scf_diag: dict[str, Any]) -> None:
        """Write summary JSON and/or iteration JSONL if configured in diagnostics settings."""
        d_cfg = self.parameters.get("diagnostics")
        if not d_cfg or not d_cfg.get("save", False):
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        summary_name = d_cfg.get("summary_file") or "scf_diagnostics.json"
        summary_path = self.directory / summary_name

        def _serialize(v):
            if isinstance(v, float) and not np.isfinite(v):
                return repr(v)
            if isinstance(v, dict):
                return {str(k): _serialize(item) for k, item in v.items()}
            if isinstance(v, (list, tuple)):
                return [_serialize(item) for item in v]
            return v

        summary_path.write_text(json.dumps(_serialize(scf_diag), indent=2) + "\n", encoding="utf-8")

        if d_cfg.get("iterations", False):
            iter_name = d_cfg.get("iteration_file") or "scf_iterations.jsonl"
            iter_path = self.directory / iter_name
            with iter_path.open("w", encoding="utf-8") as f:
                for line in self.iterations:
                    f.write(json.dumps(_serialize(line)) + "\n")


def format_hessian(h_raw: Any, natm: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert upstream PySCF Hessian from (natm, natm, 3, 3) in Hartree/Bohr^2

    to (3N, 3N) in eV/Angstrom^2, fixed atomic/xyz block ordering.
    """
    if hasattr(h_raw, "get"):
        h_arr = h_raw.get()
    else:
        h_arr = np.asarray(h_raw, dtype=float)

    if h_arr.shape != (natm, natm, 3, 3):
        raise ValueError(f"Unexpected Hessian shape from PySCF: expected {(natm, natm, 3, 3)}, got {h_arr.shape}")

    # Transpose from (atom1, atom2, coord1, coord2) to (atom1, coord1, atom2, coord2)
    # and reshape into (3*natm, 3*natm)
    h_block = h_arr.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)

    # Unit conversion: Hartree / Bohr^2 -> eV / Angstrom^2
    conversion = Hartree / (Bohr ** 2)
    h_ev_ang2 = h_block * conversion

    meta = {
        "shape": list(h_ev_ang2.shape),
        "unit": "eV/Angstrom^2",
        "atomic_ordering": "fixed_atom_xyz",
    }
    return h_ev_ang2, meta
