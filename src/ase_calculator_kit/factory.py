"""The public factory: build an ASE calculator by name."""

from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator

from .registry import BACKENDS, DFT_BACKENDS, UMLIP_BACKENDS

_DFT_ALLOWED_KWARGS = {"config", "overrides", "write_resolved_config"}


def get_calculator(name: str, **kwargs) -> Calculator:
    """Create an ASE calculator for a supported MLIP or DFT backend.

    Parameters
    ----------
    name:
        One of ``available_models()``. Case-insensitive.
    **kwargs:
        MLIP backends accept lightweight backend-specific options such as
        ``device=`` and ``model=``. DFT backends accept only ``config=`` plus
        optional ``overrides=`` and ``write_resolved_config=``.

    Examples
    --------
    >>> calc = get_calculator("chgnet", device="mps")
    >>> calc = get_calculator("sevennet", model="7net-omni", modal="mpa")
    >>> calc = get_calculator("mattersim", model="5M")
    >>> calc = get_calculator("uma", model="uma-s-1p2", task="omat")
    >>> calc = get_calculator("vasp", config="examples/dft/vasp_pbe_static.yaml")
    """
    key = name.lower()
    try:
        backend_cls = BACKENDS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown calculator '{name}'. Available calculators: {valid}"
        ) from exc
    if key in DFT_BACKENDS:
        _validate_dft_kwargs(key, kwargs)
    return backend_cls().create_calculator(**kwargs)


def get_umlip_calculator(name: str, **kwargs) -> Calculator:
    """Create an ASE calculator for a supported universal MLIP."""
    key = name.lower()
    try:
        backend_cls = UMLIP_BACKENDS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(UMLIP_BACKENDS))
        raise ValueError(
            f"Unknown uMLIP calculator '{name}'. Available: {valid}"
        ) from exc
    return backend_cls().create_calculator(**kwargs)


def get_dft_calculator(name: str, **kwargs) -> Calculator:
    """Create an ASE calculator for a supported DFT backend."""
    key = name.lower()
    try:
        backend_cls = DFT_BACKENDS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DFT_BACKENDS))
        raise ValueError(
            f"Unknown DFT calculator '{name}'. Available: {valid}"
        ) from exc
    _validate_dft_kwargs(key, kwargs)
    return backend_cls().create_calculator(**kwargs)


def attach_calculator(atoms: Atoms, name: str, **kwargs) -> Atoms:
    """Attach a calculator to ``atoms`` and return the same ``atoms`` object."""
    atoms.calc = get_calculator(name, **kwargs)
    return atoms


def available_models() -> list[str]:
    """Return the sorted list of accepted calculator names."""
    return sorted(BACKENDS)


def available_umlip_models() -> list[str]:
    """Return the sorted list of accepted MLIP calculator names."""
    return sorted(UMLIP_BACKENDS)


def available_dft_calculators() -> list[str]:
    """Return the sorted list of accepted DFT calculator names."""
    return sorted(DFT_BACKENDS)


def _validate_dft_kwargs(name: str, kwargs: dict) -> None:
    unexpected = set(kwargs) - _DFT_ALLOWED_KWARGS
    if unexpected:
        raise TypeError(
            f"DFT calculator '{name}' accepts only config=, overrides=, "
            f"and write_resolved_config=. Unexpected arguments: "
            f"{sorted(unexpected)}"
        )

    if "config" not in kwargs:
        raise TypeError(
            f"DFT calculator '{name}' requires config=. "
            f"Example: get_calculator('{name}', config='examples/dft/{name}.yaml')"
        )
