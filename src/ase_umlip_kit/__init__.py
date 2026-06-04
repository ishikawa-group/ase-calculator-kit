"""ase-umlip-kit: a unified ASE calculator factory for universal MLIPs.

Call any of the supported universal machine-learning interatomic potentials
(CHGNet, SevenNet, MatterSim, UMA/fairchem) in one line::

    from ase_umlip_kit import get_calculator
    atoms.calc = get_calculator("uma", task="omat")
    energy = atoms.get_potential_energy()
"""

from __future__ import annotations

from .errors import MissingDependencyError, UMLIPError
from .factory import attach_calculator, available_models, get_calculator

__version__ = "0.1.0"

# Aliases kept for continuity with existing scripts / earlier sketches.
build_calculator = get_calculator
utils_uMLIP_calculator = get_calculator

__all__ = [
    "get_calculator",
    "attach_calculator",
    "available_models",
    "build_calculator",
    "utils_uMLIP_calculator",
    "UMLIPError",
    "MissingDependencyError",
    "__version__",
]
