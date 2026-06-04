"""Quantum ESPRESSO backend using ASE's Espresso calculator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from ...config import resolve_calculator_config, write_resolved_config_file
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
        profile_cfg = resolved.get("profile", {})
        if "command" not in profile_cfg:
            raise ValueError("QE config requires profile.command.")
        if "pseudo_dir" not in profile_cfg:
            raise ValueError("QE config requires profile.pseudo_dir.")

        pseudopotentials = resolved.get("pseudopotentials")
        if not pseudopotentials:
            raise ValueError("QE config requires pseudopotentials.")

        parameters = resolved.get("parameters", {})
        directory = resolved.get("directory", ".")
        if write_resolved_config:
            write_resolved_config_file(resolved, directory)

        profile = EspressoProfile(
            command=profile_cfg["command"],
            pseudo_dir=profile_cfg["pseudo_dir"],
        )
        return Espresso(
            profile=profile,
            directory=directory,
            pseudopotentials=pseudopotentials,
            **parameters,
        )
