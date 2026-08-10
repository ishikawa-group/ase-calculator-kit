"""Mapping of calculator names to backend classes."""

from __future__ import annotations

from .backends import (
    BaseBackend,
    CHGNetBackend,
    EspressoBackend,
    FairChemBackend,
    MACEBackend,
    MatterSimBackend,
    NequIPBackend,
    SevenNetBackend,
    VaspBackend,
)

#: Public MLIP names accepted by :func:`ase_calculator_kit.get_calculator`.
#:
#: ``mace`` is listed here like any other backend, but it cannot be installed
#: next to the rest: ``mace-torch`` pins ``e3nn==0.4.4`` against the others'
#: ``e3nn>=0.5``. It needs its own virtual environment — see
#: ``backends/mlip/mace.py``.
MLIP_BACKENDS: dict[str, type[BaseBackend]] = {
    "chgnet": CHGNetBackend,
    "sevennet": SevenNetBackend,
    "mattersim": MatterSimBackend,
    "nequip": NequIPBackend,
    "mace": MACEBackend,
    "uma": FairChemBackend,
    "fairchem": FairChemBackend,
}

#: Public DFT names accepted by :func:`ase_calculator_kit.get_calculator`.
DFT_BACKENDS: dict[str, type[BaseBackend]] = {
    "vasp": VaspBackend,
    "qe": EspressoBackend,
    "espresso": EspressoBackend,
    "quantum-espresso": EspressoBackend,
}

BACKENDS: dict[str, type[BaseBackend]] = {
    **MLIP_BACKENDS,
    **DFT_BACKENDS,
}
