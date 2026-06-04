"""Mapping of model names to backend classes."""

from __future__ import annotations

from .backends import (
    BaseBackend,
    CHGNetBackend,
    FairChemBackend,
    MatterSimBackend,
    SevenNetBackend,
)

#: Public model names accepted by :func:`ase_umlip_kit.get_calculator`.
BACKENDS: dict[str, type[BaseBackend]] = {
    "chgnet": CHGNetBackend,
    "sevennet": SevenNetBackend,
    "mattersim": MatterSimBackend,
    "uma": FairChemBackend,
    "fairchem": FairChemBackend,
}
