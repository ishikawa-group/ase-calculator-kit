"""Backend base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ase.calculators.calculator import Calculator


class BaseBackend(ABC):
    """Abstract base for a uMLIP backend.

    Subclasses lazily import their underlying package inside
    :meth:`create_calculator` and raise
    :class:`ase_umlip_kit.errors.MissingDependencyError` if it is not installed.
    """

    #: Canonical backend name (e.g. ``"chgnet"``).
    name: str

    @abstractmethod
    def create_calculator(self, *, device: str = "auto", **kwargs) -> Calculator:
        """Build and return a fresh ASE calculator."""
        raise NotImplementedError
