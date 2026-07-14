"""Exception types for :mod:`ase_calculator_kit`."""

from __future__ import annotations


class CalculatorKitError(Exception):
    """Base class for all errors raised by ``ase_calculator_kit``."""


class DispersionError(CalculatorKitError, ValueError):
    """Raised when a requested dispersion (D3) correction is not allowed.

    Either the model already includes dispersion in its training functional
    (so adding D3 would double-count), or the functional is unverified and no
    explicit ``dispersion_xc`` override was provided. Subclasses ``ValueError``
    so it is catchable as an ordinary usage error.
    """


class MissingDependencyError(CalculatorKitError, ImportError):
    """Raised when the backend package for a requested model is not installed.

    NNP packages are optional. The error points to the smallest matching
    packaging extra instead of asking users to install every backend.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        extra = {
            "chgnet": "chgnet",
            "sevennet": "sevennet",
            "mattersim": "mattersim",
            "nequip": "nequip",
            "fairchem-core": "uma",
        }.get(backend.lower(), backend.lower())
        super().__init__(
            f"{backend} is not installed. "
            f"Install it with: pip install 'ase-calculator-kit[{extra}]'"
        )
