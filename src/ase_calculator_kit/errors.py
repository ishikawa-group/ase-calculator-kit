"""Exception types for :mod:`ase_calculator_kit`."""

from __future__ import annotations


class CalculatorKitError(Exception):
    """Base class for all errors raised by ``ase_calculator_kit``."""


class UMLIPError(CalculatorKitError):
    """Backward-compatible base class for MLIP-related errors."""


class DispersionError(UMLIPError, ValueError):
    """Raised when a requested dispersion (D3) correction is not allowed.

    Either the model already includes dispersion in its training functional
    (so adding D3 would double-count), or the functional is unverified and no
    explicit ``dispersion_xc`` override was provided. Subclasses ``ValueError``
    so it is catchable as an ordinary usage error.
    """


class MissingDependencyError(UMLIPError, ImportError):
    """Raised when the backend package for a requested model is not installed.

    MLIP backends are installed by default with ``pip install
    ase-calculator-kit``; this is raised only if a backend's package is missing
    or broken in the environment.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(
            f"{backend} is not installed. "
            f"Install or repair with: pip install ase-calculator-kit"
        )
