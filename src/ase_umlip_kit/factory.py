"""The public factory: build an ASE calculator for a named uMLIP."""

from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator

from .registry import BACKENDS


def get_calculator(name: str, *, device: str = "auto", **kwargs) -> Calculator:
    """Create an ASE calculator for a supported universal MLIP.

    Parameters
    ----------
    name:
        One of ``available_models()``: ``"chgnet"``, ``"sevennet"``,
        ``"mattersim"``, ``"uma"`` (alias ``"fairchem"``). Case-insensitive.
    device:
        ``"auto"`` (default) or an explicit ``"cuda"`` / ``"cpu"`` / ``"mps"``.
        ``"mps"`` is only supported by CHGNet.
    **kwargs:
        Backend-specific options, e.g. ``model=``/``modal=`` for SevenNet,
        ``task=`` for UMA, ``model="5M"`` for MatterSim. See each backend's
        ``create_calculator`` docstring.

    Examples
    --------
    >>> calc = get_calculator("chgnet", device="mps")
    >>> calc = get_calculator("sevennet", model="7net-omni", modal="mpa")
    >>> calc = get_calculator("mattersim", model="5M")
    >>> calc = get_calculator("uma", model="uma-s-1p2", task="omat")
    """
    key = name.lower()
    try:
        backend_cls = BACKENDS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown MLIP '{name}'. Available models: {valid}"
        ) from exc
    return backend_cls().create_calculator(device=device, **kwargs)


def attach_calculator(
    atoms: Atoms, name: str, *, device: str = "auto", **kwargs
) -> Atoms:
    """Attach a calculator to ``atoms`` and return the same ``atoms`` object."""
    atoms.calc = get_calculator(name, device=device, **kwargs)
    return atoms


def available_models() -> list[str]:
    """Return the sorted list of accepted model names."""
    return sorted(BACKENDS)
