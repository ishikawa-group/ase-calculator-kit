"""Exception types for :mod:`ase_umlip_kit`."""

from __future__ import annotations


class UMLIPError(Exception):
    """Base class for all errors raised by ``ase_umlip_kit``."""


class DispersionError(UMLIPError, ValueError):
    """Raised when a requested dispersion (D3) correction is not allowed.

    Either the model already includes dispersion in its training functional
    (so adding D3 would double-count), or the functional is unverified and no
    explicit ``dispersion_xc`` override was provided. Subclasses ``ValueError``
    so it is catchable as an ordinary usage error.
    """


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
