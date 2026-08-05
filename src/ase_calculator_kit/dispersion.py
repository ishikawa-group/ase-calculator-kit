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

from dataclasses import dataclass

from ase.calculators.calculator import Calculator

from .errors import DispersionError

@dataclass(frozen=True)
class DispersionPolicy:
    """Chemical meaning of an optional D3 correction for one model variant.

    ``reference_level`` records the DFT level used for training. ``d3_xc`` is
    the matching torch-dftd parameter when a correction may be added. If the
    training reference already includes dispersion, ``includes_dispersion`` is
    true and an additional D3 term is always rejected.
    """

    reference_level: str
    d3_xc: str | None = None
    includes_dispersion: bool = False
    note: str = ""


def _allowed(reference_level: str, d3_xc: str) -> DispersionPolicy:
    return DispersionPolicy(reference_level=reference_level, d3_xc=d3_xc)


def _included(reference_level: str, note: str) -> DispersionPolicy:
    return DispersionPolicy(
        reference_level=reference_level,
        includes_dispersion=True,
        note=note,
    )


# One row represents one chemically distinct checkpoint, task, or modal. Keep
# this table synchronized with docs/models.md.
_POLICIES: dict[tuple[str, str], DispersionPolicy] = {
    # CHGNet: MPtrj is PBE(+U); the transfer-learning checkpoint is r2SCAN.
    ("chgnet", "default"): _allowed("PBE+U", "pbe"),
    ("chgnet", "0.3.0"): _allowed("PBE+U", "pbe"),
    ("chgnet", "0.2.0"): _allowed("PBE+U", "pbe"),
    ("chgnet", "r2scan"): _allowed("r2SCAN", "r2scan"),
    # MatterSim and NequIP OAM use PBE-level materials reference data.
    ("mattersim", "1M"): _allowed("PBE", "pbe"),
    ("mattersim", "5M"): _allowed("PBE", "pbe"),
    ("mattersim", "default"): _allowed("assumed PBE", "pbe"),
    ("nequip", "S"): _allowed("PBE(+U)", "pbe"),
    ("nequip", "M"): _allowed("PBE(+U)", "pbe"),
    ("nequip", "L"): _allowed("PBE(+U)", "pbe"),
    ("nequip", "XL"): _allowed("PBE(+U)", "pbe"),
    # SevenNet: the modal selects both dataset and reference functional. The
    # fidelity of every 7net-omni task is listed in SevenNet's "Pretrained
    # models" documentation; the rows below follow it.
    ("sevennet", "mpa"): _allowed("PBE(+U)", "pbe"),
    ("sevennet", "omat24"): _allowed("PBE(+U)", "pbe"),
    ("sevennet", "matpes_pbe"): _allowed("PBE", "pbe"),
    ("sevennet", "matpes_r2scan"): _allowed("r2SCAN", "r2scan"),
    ("sevennet", "mp_r2scan"): _allowed("r2SCAN", "r2scan"),
    ("sevennet", "oc20"): _allowed("RPBE", "rpbe"),
    ("sevennet", "oc22"): _allowed("PBE(+U)", "pbe"),
    ("sevennet", "pet_mad"): _allowed("PBEsol", "pbesol"),
    ("sevennet", "default"): _allowed("PBE", "pbe"),
    ("sevennet", "omol25_low"): _included(
        "ωB97M-V", "the OMol25 modal already includes nonlocal VV10 dispersion"
    ),
    ("sevennet", "omol25_high"): _included(
        "ωB97M-V", "the OMol25 modal already includes nonlocal VV10 dispersion"
    ),
    ("sevennet", "spice"): _included(
        "ωB97M-D3(BJ)", "SPICE is computed at ωB97M-D3(BJ)/def2-TZVPPD, so D3(BJ) "
        "is already in the reference data"
    ),
    ("sevennet", "qcml"): _included(
        "PBE0+MBD-NL", "QCML applies the MBD-NL many-body dispersion correction"
    ),
    ("sevennet", "odac23"): _included(
        "PBE-D3", "ODAC23 is computed at the PBE-D3 level"
    ),
    # UMA: each task is a separate chemical domain and DFT reference level.
    ("uma", "omat"): _allowed("PBE+U", "pbe"),
    ("uma", "oc20"): _allowed("RPBE", "rpbe"),
    ("uma", "oc22"): _allowed("PBE(+U)", "pbe"),
    ("uma", "oc25"): _included(
        "RPBE+D3(BJ)", "the OC25 task already includes D3(BJ) dispersion"
    ),
    ("uma", "omol"): _included(
        "ωB97M-V", "the OMol task already includes nonlocal VV10 dispersion"
    ),
    ("uma", "odac"): _included(
        "PBE-D3", "ODAC23 is computed at the PBE-D3 level"
    ),
    ("uma", "omc"): _included(
        "PBE+D3", "OMC25 is computed at the PBE+D3 level"
    ),
}


def get_dispersion_policy(backend: str, key: str) -> DispersionPolicy | None:
    """Return the documented chemistry for a model variant, if verified."""

    return _POLICIES.get((backend, key))


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
    policy = get_dispersion_policy(backend, key)
    if policy is not None and policy.includes_dispersion:
        raise DispersionError(
            f"dispersion=True is not allowed for {backend} '{key}': "
            f"{policy.note}. Remove dispersion=True to avoid "
            "double-counting."
        )

    if dispersion_xc is not None:
        # Explicit override: the caller takes responsibility (also the escape
        # hatch for the unverified tier).
        return dispersion_xc

    if policy is not None and policy.d3_xc is not None:
        return policy.d3_xc

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
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise DispersionError(
            "Dispersion requires 'torch-dftd'. Install it with: "
            "pip install 'ase-calculator-kit[dispersion]'"
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
