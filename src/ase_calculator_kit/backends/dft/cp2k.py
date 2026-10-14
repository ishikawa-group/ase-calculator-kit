"""Config-only CP2K selection and validation; electronic structure stays in ASE."""

from __future__ import annotations

import re
import shlex
from numbers import Integral
from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator, all_changes

from ...config import (
    resolve_calculator_config, validate_external_dft_config, validate_mapping_keys,
    write_resolved_config_file,
)
from ..base import BaseBackend


def _integer(value: Any, location: str, *, multiplicity: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{location} must be an integer.")
    if multiplicity and value < 1:
        raise ValueError(f"{location} is multiplicity and must be >= 1.")
    return int(value)


def _validate_electronic_state(parameters: dict[str, Any], info: dict[str, Any]) -> None:
    for key, parameter in (("charge", "charge"), ("spin", "multiplicity")):
        if key not in info:
            continue
        value = _integer(info[key], f"atoms.info[{key!r}]", multiplicity=key == "spin")
        configured = parameters.get(parameter)
        if configured is None:
            raise ValueError(
                f"CP2K atoms.info[{key!r}] requires explicit YAML parameters.{parameter}; "
                "template electronic states cannot be checked. atoms.info spin is multiplicity."
            )
        if value != configured:
            raise ValueError(
                f"Electronic-state mismatch for {key}: atoms.info={value}, "
                f"parameters.{parameter}={configured}. CP2K electronic state is YAML-only; "
                "atoms.info spin is multiplicity."
            )


class _YamlElectronicState:
    """Check info even when ASE would return a cached property without calculate()."""

    def _validate_info(self, atoms):
        state = {key: self.parameters.get(key) for key in self._kit_state_keys}
        _validate_electronic_state(state, atoms.info)

    def get_property(self, name, atoms=None, allow_calculation=True):
        candidate = atoms if atoms is not None else self.atoms
        if candidate is not None:
            self._validate_info(candidate)
        return super().get_property(name, atoms, allow_calculation)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        candidate = atoms if atoms is not None else self.atoms
        if candidate is not None:
            self._validate_info(candidate)
        return super().calculate(atoms, properties, system_changes)


class _ShortProject:
    """Keep GLOBAL/PROJECT within CP2K's 80-character string keywords.

    ASE writes ``PROJECT <directory>/<label>``, which CP2K rejects once the absolute
    run directory is long. The shell instead starts inside ``directory`` (see
    ``create_calculator``), so the bare label puts PROJECT-derived files (restart
    files, wavefunctions) in the same place.
    """

    def _generate_input(self):
        text = super()._generate_input()
        name = Path(self.label).name
        text, count = re.subn(
            rf"^(\s*)PROJECT {re.escape(self.label)}$", rf"\g<1>PROJECT {name}", text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError("Could not shorten ASE's GLOBAL/PROJECT line for CP2K.")
        return text


def _input_template(value: Any) -> str:
    if isinstance(value, Path):
        return value.read_text(encoding="utf-8")
    if not isinstance(value, str):
        raise ValueError("CP2K inp must be a template string or file path.")
    if not value.strip() or "\n" in value or value.lstrip().startswith(("&", "!", "#", "@")):
        return value
    return Path(value).read_text(encoding="utf-8")


class Cp2kBackend(BaseBackend):
    name = "cp2k"

    def create_calculator(
        self,
        *,
        config: str | Path | dict[str, Any],
        overrides: dict[str, Any] | None = None,
        write_resolved_config: bool = False,
    ) -> Calculator:
        """Create ASE CP2K for molecular or periodic Gaussian/plane-wave DFT.

        Parameters
        ----------
        config : str, Path or dict
            Reviewable YAML with explicit profile.command and profile.data_dir.
            parameters must contain xc, basis_set, pseudo_potential and cutoff
            (eV). None delegates a setting to inp; pseudo_potential='auto' is
            refused to prevent a silent change of reference functional.
        overrides : dict, optional
            Recursively merged configuration changes.
        write_resolved_config : bool
            Save merged settings, resolved data paths and template text.

        Returns
        -------
        ase.calculators.cp2k.CP2K
            A thin validation subclass. Charge/multiplicity/uks are YAML-only;
            atoms.info charge/spin (multiplicity) must agree before cache reuse.

        Notes
        -----
        Construction starts a CP2K shell. Use close() or a with statement.
        ASE reports free_energy == energy, including with smearing; no separate
        E0/free-energy convention is available. ASE always requests stress:
        stress_tensor=False can yield zeros or a shell error, not valid stress.
        The shell runs inside directory and GLOBAL/PROJECT is the bare label
        "cp2k" (CP2K limits PROJECT to 80 characters); relative file names in
        inp are therefore relative to directory.
        Partial-PBC slabs need an explicit inp Poisson section, e.g. PERIODIC XY
        with POISSON_SOLVER MT, and parameters.poisson_solver=None. ASE generates
        CELL periodicity but its automatic Poisson setup covers only molecules.
        Avoid template keywords also generated by ASE; set their parameters to
        None. GLOBAL/PROJECT is always generated.
        """
        import ase
        from ase.calculators.cp2k import CP2K
        from ase.utils import tokenize_version

        resolved = resolve_calculator_config("cp2k", config=config, overrides=overrides)
        validate_external_dft_config(resolved, "cp2k")
        profile = resolved["profile"]
        if tokenize_version(ase.__version__) < tokenize_version("3.29.0"):
            if "cp2k_shell" not in profile["command"]:
                raise ValueError("ASE < 3.29 requires profile.command to contain 'cp2k_shell'.")

        parameters = resolved.get("parameters", {})
        validate_mapping_keys(parameters, set(CP2K.default_parameters), "cp2k parameters")
        for key in ("xc", "basis_set", "pseudo_potential", "cutoff"):
            if key not in parameters:
                raise ValueError(f"CP2K requires explicit parameters.{key} (None delegates to inp).")
        potential = parameters["pseudo_potential"]
        if isinstance(potential, str) and potential.strip().lower() == "auto":
            raise ValueError("CP2K parameters.pseudo_potential='auto' is not reproducible.")
        if "inp" in resolved and "inp" in parameters:
            raise ValueError("CP2K inp cannot appear both at top-level and in parameters.inp.")
        template = _input_template(resolved.pop("inp", parameters.get("inp", "")))
        parameters["inp"] = template

        data_dir = Path(profile["data_dir"]).resolve()
        if not data_dir.is_dir():
            raise ValueError(f"CP2K profile.data_dir is not a directory: {data_dir}")
        profile["data_dir"] = str(data_dir)
        for key in ("basis_set_file", "potential_file"):
            filename = parameters.get(key, CP2K.default_parameters[key])
            if filename is None:
                parameters[key] = None
                continue
            if not isinstance(filename, str | Path) or not str(filename).strip():
                raise ValueError(f"CP2K parameters.{key} must be a nonempty path or None.")
            path = Path(filename)
            if not path.is_absolute():
                path = (data_dir / path).resolve()
            if not path.is_file():
                raise ValueError(f"CP2K parameters.{key} file does not exist: {path}")
            parameters[key] = str(path)

        for key in ("charge", "multiplicity"):
            if parameters.get(key) is not None:
                parameters[key] = _integer(
                    parameters[key], f"parameters.{key}", multiplicity=key == "multiplicity",
                )
        if parameters.get("uks") is not None and not isinstance(parameters["uks"], bool):
            raise ValueError("CP2K parameters.uks must be a Boolean or None.")
        if (parameters.get("multiplicity") or 1) > 1:
            if parameters.get("uks") is False:
                raise ValueError("CP2K multiplicity > 1 requires uks=True.")
            parameters["uks"] = True

        directory = str(Path(resolved.get("directory", ".")).resolve())
        resolved["directory"] = directory
        resolved["parameters"] = parameters

        class ValidatedCP2K(_ShortProject, _YamlElectronicState, CP2K):
            pass

        # CP2K resolves PROJECT and relative template paths from its working directory.
        Path(directory).mkdir(parents=True, exist_ok=True)
        command = (f"cd {shlex.quote(directory)} && "
                   f"exec /bin/sh -c {shlex.quote(profile['command'])}")
        calculator = ValidatedCP2K(
            command=command, directory=directory, label="cp2k", **parameters,
        )
        calculator._kit_state_keys = set(parameters) & {"charge", "multiplicity"}
        if write_resolved_config:
            try:
                write_resolved_config_file(resolved, directory)
            except Exception:
                calculator.close()
                raise
        return calculator
