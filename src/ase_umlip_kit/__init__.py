"""Compatibility shim for the old ``ase_umlip_kit`` import name.

Use ``ase_calculator_kit`` for new code. This namespace is kept temporarily so
existing scripts continue to import the public API.
"""

from __future__ import annotations

from ase_calculator_kit import *  # noqa: F403
from ase_calculator_kit import __all__ as __all__
from ase_calculator_kit import __version__ as __version__
