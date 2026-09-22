"""Quantum ESPRESSO backend using ASE's Espresso calculator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from ...config import (
    resolve_calculator_config, validate_external_dft_config, write_resolved_config_file,
)
from ..base import BaseBackend


class EspressoBackend(BaseBackend):
    name = "qe"

    def create_calculator(
        self,
        *,
        config: str | Path | dict[str, Any],
        overrides: dict[str, Any] | None = None,
        write_resolved_config: bool = False,
    ) -> Calculator:
        """Create an ASE :class:`ase.calculators.espresso.Espresso` from config."""
        from ase.calculators.espresso import Espresso, EspressoProfile

        resolved = resolve_calculator_config(
            "qe",
            config=config,
            overrides=overrides,
        )
        validate_external_dft_config(resolved, "qe")
        profile_cfg = resolved.get("profile", {})
        if "command" not in profile_cfg:
            raise ValueError("QE config requires profile.command.")
        if "pseudo_dir" not in profile_cfg:
            raise ValueError("QE config requires profile.pseudo_dir.")

        pseudopotentials = resolved.get("pseudopotentials")
        if not pseudopotentials:
            raise ValueError("QE config requires pseudopotentials.")

        parameters = resolved.get("parameters", {})
        _validate_parameters(parameters)
        directory = resolved.get("directory", ".")

        profile = EspressoProfile(
            command=profile_cfg["command"],
            pseudo_dir=profile_cfg["pseudo_dir"],
        )
        calculator = Espresso(
            profile=profile,
            directory=directory,
            pseudopotentials=pseudopotentials,
            **parameters,
        )
        if write_resolved_config:
            write_resolved_config_file(resolved, directory)
        return calculator


def _validate_parameters(parameters):
    """Use ASE's pw.x vocabulary, including indexed and flat namelist keys."""
    from ase.io.espresso_namelist.keys import ALL_KEYS
    from ase.io.espresso_namelist.namelist import Namelist

    keys = ALL_KEYS["pw"]
    writer_keys = {"input_data", "kspacing", "kpts", "koffset",
                   "crystal_coordinates", "additional_cards"}
    for key in parameters:
        if not isinstance(key, str) or (
            key not in writer_keys and Namelist.search_key(key.lower(), keys) is None
        ):
            raise ValueError(f"Unknown QE parameters key: {key!r}")
    data = parameters.get("input_data", {})
    if not isinstance(data, dict):
        raise ValueError("QE parameters.input_data must be a mapping.")
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("QE input_data keys must be strings.")
        section = key.lower()
        if section in keys:
            if not isinstance(value, dict):
                raise ValueError(f"QE input_data.{key} must be a mapping.")
            for subkey in value:
                if not isinstance(subkey, str) or subkey.lower().split("(")[0] not in keys[section]:
                    raise ValueError(f"Unknown QE input_data.{key} key: {subkey!r}")
        elif isinstance(value, dict) or Namelist.search_key(section, keys) is None:
            raise ValueError(f"Unknown QE input_data key: {key!r}")
