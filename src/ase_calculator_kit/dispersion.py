"""Grimme-D3 dispersion correction policy and application.

This module is the single source of truth for *whether* a Grimme-D3 van-der-Waals
correction may be added on top of a given model, and *which* exchange-correlation
functional parameters to use. The decision is non-trivial: some models already
include dispersion in their training functional, so adding D3 would double-count
it. See ``docs/models.md`` for the human-readable version of these tables — the
two MUST be kept in sync.

Three tiers (keyed by ``(backend, key)`` where ``key`` is the model's functional
discriminator — CHGNet model, MatterSim model, NequIP model, SevenNet modal,
or UMA task):

1. Allowed   -> a default D3 ``xc`` is known; ``dispersion=True`` wraps the model.
2. Included  -> dispersion is already in the training functional; always an error.
3. Unverified -> the training functional is not confirmed; an error UNLESS the
   caller passes an explicit ``dispersion_xc`` to take responsibility.

The mechanism is ASE's :class:`~ase.calculators.mixing.SumCalculator` plus
``torch_dftd``'s ``TorchDFTD3Calculator`` (D3 with Becke-Johnson damping).
"""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from .errors import DispersionError

#: ``(backend, key)`` whose training functional already includes dispersion.
#: Adding D3 would double-count it, so ``dispersion=True`` is always rejected.
_DISPERSION_INCLUDED: dict[tuple[str, str], str] = {
    ("uma", "oc25"): "the oc25 task is trained at RPBE+D3(BJ); D3 is already included",
    ("uma", "omol"): "the omol task is trained at wB97M-V, which already includes "
    "nonlocal (VV10) dispersion",
    ("sevennet", "omol25_low"): "the omol25 modals are trained at wB97M-V, which "
    "already includes nonlocal (VV10) dispersion",
    ("sevennet", "omol25_high"): "the omol25 modals are trained at wB97M-V, which "
    "already includes nonlocal (VV10) dispersion",
}

#: ``(backend, key)`` -> default D3 ``xc`` for models whose functional is known
#: and does NOT include dispersion.
_DEFAULT_XC: dict[tuple[str, str], str] = {
    ("chgnet", "default"): "pbe",      # MPtrj, PBE+U
    ("mattersim", "1M"): "pbe",        # PBE
    ("mattersim", "5M"): "pbe",        # PBE
    ("mattersim", "default"): "pbe",   # custom checkpoint, assumed PBE
    ("sevennet", "mpa"): "pbe",        # MPtrj + sAlex, PBE
    ("sevennet", "omat24"): "pbe",     # OMat24, PBE
    ("sevennet", "matpes_pbe"): "pbe",  # MatPES, PBE
    ("sevennet", "default"): "pbe",    # single-fidelity 7net-* (e.g. 7net-0), PBE
    ("uma", "omat"): "pbe",            # OMat24, PBE+U
    ("uma", "oc20"): "rpbe",           # OC20, RPBE
    ("uma", "oc22"): "pbe",            # OC22, PBE(+U)
}


def resolve_dispersion_xc(
    backend: str, key: str, *, dispersion_xc: str | None
) -> str:
    """Return the D3 ``xc`` to use for ``(backend, key)``, or raise.

    Raises
    ------
    DispersionError
        If the model already includes dispersion (always), or if the model's
        functional is unverified and no explicit ``dispersion_xc`` was given.
    """
    pair = (backend, key)

    if pair in _DISPERSION_INCLUDED:
        raise DispersionError(
            f"dispersion=True is not allowed for {backend} '{key}': "
            f"{_DISPERSION_INCLUDED[pair]}. Remove dispersion=True to avoid "
            "double-counting."
        )

    if dispersion_xc is not None:
        # Explicit override: the caller takes responsibility (also the escape
        # hatch for the unverified tier).
        return dispersion_xc

    if pair in _DEFAULT_XC:
        return _DEFAULT_XC[pair]

    raise DispersionError(
        f"dispersion inclusion for {backend} '{key}' is not verified, so a D3 "
        "correction is refused by default. If you are certain the base model "
        "excludes dispersion, pass an explicit dispersion_xc (e.g. "
        "dispersion_xc='pbe') to override."
    )


def wrap_with_d3(base_calc: Calculator, *, xc: str, device: str) -> Calculator:
    """Return ``base_calc`` summed with a torch-dftd D3(BJ) correction."""
    try:
        from ase.calculators.mixing import SumCalculator
        from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
    except ImportError as exc:  # pragma: no cover - torch-dftd is a core dep
        raise DispersionError(
            "Dispersion requires 'torch-dftd'. Install or repair with: "
            "pip install ase-calculator-kit"
        ) from exc

    # torch-dftd on MPS is unreliable; run the D3 part on CPU in that case.
    d3_device = "cpu" if device == "mps" else device
    d3 = TorchDFTD3Calculator(damping="bj", xc=xc, device=d3_device)
    return SumCalculator([base_calc, d3])


def precheck_dispersion_xc(
    backend: str, key: str, *, dispersion: bool, dispersion_xc: str | None
) -> str | None:
    """Validate the dispersion policy up front and return the D3 ``xc`` to use.

    Returns ``None`` when ``dispersion`` is False (no correction). Otherwise
    delegates to :func:`resolve_dispersion_xc`, which raises
    :class:`~ase_calculator_kit.errors.DispersionError` for models that already
    include dispersion or whose functional is unverified — *before* the (heavy)
    base calculator is built, so the error is fast and offline.
    """
    if not dispersion:
        return None
    return resolve_dispersion_xc(backend, key, dispersion_xc=dispersion_xc)
