"""Exception types for :mod:`ase_umlip_kit`."""

from __future__ import annotations


class UMLIPError(Exception):
    """Base class for all errors raised by ``ase_umlip_kit``."""


class MissingDependencyError(UMLIPError, ImportError):
    """Raised when the backend package for a requested model is not installed.

    All four supported backends (CHGNet, SevenNet, MatterSim, fairchem/UMA) are
    installed by default with ``pip install ase-umlip-kit``; this is raised only
    if a backend's package is missing or broken in the environment.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(
            f"{backend} is not installed. "
            f"Install or repair with: pip install ase-umlip-kit"
        )
