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


#: Packages no extra of this package can install, and the command that does.
#:
#: PyPI rejects a direct reference (``pkg @ git+https://...``) in an uploaded
#: project's metadata, so a dependency that exists only as a git repository can
#: never be an extra here. It still has to be *nameable* when something needs
#: it, which is what this table is for.
#:
#: ``graph_longrange`` is the module namespace of WillBaldwin0's
#: ``graph_electrostatics``, and MACE-Polar's published checkpoints unpickle
#: classes from it. Note the mismatch: the repository is called
#: ``graph_electrostatics`` while the distribution it builds is named
#: ``graph_longrange``, so pip refuses the ``graph_electrostatics @ git+...``
#: spelling ("has inconsistent name") and the bare URL is the form that works.
_DIRECT_INSTALLS = {
    "graph_longrange": (
        "pip install "
        "git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0"
    ),
}


class MissingDependencyError(CalculatorKitError, ImportError):
    """Raised when the backend package for a requested model is not installed.

    NNP packages are optional. The error points to the smallest matching
    packaging extra instead of asking users to install every backend — or, for
    a package that no extra can reach, to the command that installs it.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        direct = _DIRECT_INSTALLS.get(backend)
        if direct is not None:
            super().__init__(f"{backend} is not installed. Install it with: {direct}")
            return
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
