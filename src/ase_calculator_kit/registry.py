"""Mapping of calculator names to backend classes."""

from __future__ import annotations

from .backends import (
    BaseBackend,
    CHGNetBackend,
    EspressoBackend,
    FairChemBackend,
    MatterSimBackend,
    SevenNetBackend,
    VaspBackend,
)

#: Public MLIP names accepted by :func:`ase_calculator_kit.get_calculator`.
MLIP_BACKENDS: dict[str, type[BaseBackend]] = {
    "chgnet": CHGNetBackend,
    "sevennet": SevenNetBackend,
    "mattersim": MatterSimBackend,
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
