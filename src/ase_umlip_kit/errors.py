"""Exception types for :mod:`ase_umlip_kit`."""

from __future__ import annotations


class UMLIPError(Exception):
    """Base class for all errors raised by ``ase_umlip_kit``."""


class MissingDependencyError(UMLIPError, ImportError):
    """Raised when the backend package for a requested model is not installed.

    The four supported backends (CHGNet, SevenNet, MatterSim, fairchem/UMA) are
    installed by default with ``pip install ase-umlip-kit``. If you installed
    only a subset via extras, the missing one raises this with an install hint.
    """

    def __init__(self, backend: str, extra: str) -> None:
        self.backend = backend
        self.extra = extra
        super().__init__(
            f"{backend} is not installed. "
            f"Install with: pip install ase-umlip-kit[{extra}]"
        )
