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
from .errors import CalculatorKitError, DispersionError, MissingDependencyError
from .factory import (
    attach_calculator,
    available_calculators,
    available_dft_calculators,
    available_mlip_models,
    available_models,
    get_calculator,
    get_dft_calculator,
    get_mlip_calculator,
)

try:
    __version__ = version("ase-calculator-kit")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "0.0.0"

__all__ = [
    "get_calculator",
    "get_mlip_calculator",
    "get_dft_calculator",
    "attach_calculator",
    "available_calculators",
    "available_models",
    "available_mlip_models",
    "available_dft_calculators",
    "resolve_calculator_config",
    "CalculatorKitError",
    "MissingDependencyError",
    "DispersionError",
    "__version__",
]
