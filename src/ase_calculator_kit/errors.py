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


#: Backends that cannot share an environment with the rest, and why.
#:
#: The note is appended to the install hint because this is where a user
#: actually lands: they run ``pip install 'ase-calculator-kit[mace]'`` into the
#: environment they already have, pip fails on the e3nn conflict, and nothing
#: has told them that a second environment was the intended answer.
_SEPARATE_ENVIRONMENT_NOTES = {
    "mace": (
        " Install it into a virtual environment of its own: mace-torch pins "
        "e3nn==0.4.4, while sevenn, fairchem-core, mattersim and nequip all "
        "require e3nn>=0.5, so MACE cannot coexist with the other backends."
    ),
}


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
            "mace-torch": "mace",
            "fairchem-core": "uma",
        }.get(backend.lower(), backend.lower())
        super().__init__(
            f"{backend} is not installed. "
            f"Install it with: pip install 'ase-calculator-kit[{extra}]'"
            + _SEPARATE_ENVIRONMENT_NOTES.get(extra, "")
        )
