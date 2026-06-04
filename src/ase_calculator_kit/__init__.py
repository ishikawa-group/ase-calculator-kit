"""ase-calculator-kit: a unified ASE calculator factory.

Call supported MLIP and DFT ASE calculators from one public factory::

    from ase_calculator_kit import get_calculator
    atoms.calc = get_calculator("uma", task="omat")
    atoms.calc = get_calculator("vasp", config="examples/dft/vasp_pbe_static.yaml")
    energy = atoms.get_potential_energy()
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .config import resolve_calculator_config
from .errors import CalculatorKitError, DispersionError, MissingDependencyError, UMLIPError
from .factory import (
    attach_calculator,
    available_dft_calculators,
    available_models,
    available_umlip_models,
    get_calculator,
    get_dft_calculator,
    get_umlip_calculator,
)

try:
    __version__ = version("ase-calculator-kit")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

# Aliases kept for continuity with existing scripts / earlier sketches.
build_calculator = get_calculator
utils_uMLIP_calculator = get_calculator

__all__ = [
    "get_calculator",
    "get_umlip_calculator",
    "get_dft_calculator",
    "attach_calculator",
    "available_models",
    "available_umlip_models",
    "available_dft_calculators",
    "resolve_calculator_config",
    "build_calculator",
    "utils_uMLIP_calculator",
    "CalculatorKitError",
    "UMLIPError",
    "MissingDependencyError",
    "DispersionError",
    "__version__",
]
