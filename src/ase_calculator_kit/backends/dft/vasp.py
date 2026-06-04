"""VASP backend using ASE's Vasp calculator."""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator

from ...config import resolve_calculator_config, write_resolved_config_file
from ..base import BaseBackend


class VaspBackend(BaseBackend):
    name = "vasp"

    def create_calculator(
        self,
        *,
        config: str | dict[str, Any],
        overrides: dict[str, Any] | None = None,
        write_resolved_config: bool = False,
    ) -> Calculator:
        """Create an ASE :class:`ase.calculators.vasp.Vasp` from config."""
        from ase.calculators.vasp import Vasp

        resolved = resolve_calculator_config(
            "vasp",
            config=config,
            overrides=overrides,
        )
        profile = resolved.get("profile", {})
        if "command" not in profile:
            raise ValueError("VASP config requires profile.command.")

        parameters = resolved.get("parameters", {})
        directory = resolved.get("directory", ".")
        if write_resolved_config:
            write_resolved_config_file(resolved, directory)

        return Vasp(
            command=profile["command"],
            directory=directory,
            txt=profile.get("txt", "vasp.out"),
            **parameters,
        )
