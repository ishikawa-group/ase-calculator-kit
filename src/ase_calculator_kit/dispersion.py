"""Grimme-D3 dispersion correction policy and application.

This module is the single source of truth for *whether* a Grimme-D3 van-der-Waals
correction may be added on top of a given model, and *which* exchange-correlation
functional parameters to use. The decision is non-trivial: some models already
include dispersion in their training functional, so adding D3 would double-count
it. See ``docs/models.md`` for the human-readable version of these tables — the
two MUST be kept in sync.

Three tiers (keyed by ``(backend, key)`` where ``key`` is the model's functional
discriminator — CHGNet model, MatterSim model, NequIP model, SevenNet modal,
MACE head, or UMA task):

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
    # MACE-MH-1: the head *is* the level of theory. The model's own authors
    # evaluate the PBE-trained heads with torch-dftd D3(BJ) using the PBE
    # parametrisation, and run the OMol head with no added dispersion
    # (arXiv:2510.25380, "Dispersion corrections (D3)").
    ("mace", "omat_pbe"): _allowed("PBE(+U)", "pbe"),
    ("mace", "mp_pbe_refit_add"): _allowed("PBE(+U)", "pbe"),
    ("mace", "oc20_usemppbe"): DispersionPolicy(
        reference_level="PBE",
        d3_xc="pbe",
        note=(
            "MACE-MH-1 describes this head as OC20 'computed at the PBE level' "
            "and applies its PBE D3(BJ) parameters to it; the OC20 dataset "
            "itself is published as RPBE, which is what the sevennet and uma "
            "oc20 rows use. Pass dispersion_xc='rpbe' to follow the dataset."
        ),
    ),
    ("mace", "matpes_r2scan"): _allowed("r2SCAN", "r2scan"),
    ("mace", "omol"): _included(
        "ωB97M-VV10", "the OMol head already includes nonlocal VV10 dispersion"
    ),
    ("mace", "spice_wB97M"): _included(
        "ωB97M-D3(BJ)", "SPICE-1 is computed at ωB97M-D3(BJ), so D3(BJ) is "
        "already in the reference data"
    ),
    # Single-head MACE checkpoints have no head to key on, so the model name
    # does it: for these the functional is a property of the whole model.
    ("mace", "medium-omat-0"): _allowed("PBE(+U)", "pbe"),
    ("mace", "small-omat-0"): _allowed("PBE(+U)", "pbe"),
    ("mace", "medium-mpa-0"): _allowed("PBE(+U)", "pbe"),
    ("mace", "mace-matpes-pbe-0"): _allowed("PBE", "pbe"),
    ("mace", "mace-matpes-r2scan-0"): _allowed("r2SCAN", "r2scan"),
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
    # Single-fidelity checkpoints take no modal, so the model name keys them.
    ("sevennet", "7net-omat"): _allowed("PBE(+U)", "pbe"),
    ("sevennet", "sevennet-omat"): _allowed("PBE(+U)", "pbe"),
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
        "RPBE+D3(zero)",
        "the OC25 task already includes D3 dispersion with zero damping",
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


#: Damping functions this package will hand to torch-dftd.
#:
#: torch-dftd also accepts ``zerom``, ``bjm`` and ``dftd2``, but those have
#: published parameters for only a handful of functionals, and none for the
#: ones in the policy table above. Restricting the choice keeps a typo from
#: silently selecting a parameter set that was never fitted for the model's
#: reference functional.
DAMPING_FUNCTIONS = ("bj", "zero")

DEFAULT_DAMPING = "bj"

#: Distance cutoff of the D3 sum, in angstrom.
#:
#: This is **PFP's setting, not torch-dftd's**. torch-dftd defaults to 95 Bohr
#: (50.3 Å); PFP v7.0.0 and later run the same torch-dftd code at 14 Å, and
#: Matlantis published the validation behind that number: on the Wellendorff
#: adsorption benchmark the change moves adsorption energies by an MAE of
#: 0.0024 eV — negligible against the 0.01 eV scale those numbers are discussed
#: at — while the 90th percentile of the COD unit-cell-volume error is
#: unchanged. What it buys is size: the D3 term, not the network, was the limit,
#: and 14 Å raises the reachable system from ~5000 atoms to about three times
#: that, faster as well.
#:
#: Matching PFP matters here because this package's whole point is comparing
#: NNPs against each other. Leaving torch-dftd's 50 Å in place would have made
#: every non-PFP model's dispersion term a different quantity from PFP's, on top
#: of the model difference we mean to measure.
#:
#: Source: "DFT-D3補正に関するアップデート", Matlantis documentation
#: (matlantis/matlantis-agent-skills, references/theory/d3-validation).
DEFAULT_CUTOFF = 14.0

#: Smoothing applied to the D3 term near ``cutoff``.
#:
#: torch-dftd defaults to ``"none"``, which leaves a step in the energy — and a
#: discontinuity in the force — at the cutoff radius. PFP shipped ``"none"``
#: too, but as an acknowledged bug: v7.0.0 fixed it to ``"poly"`` and kept
#: ``"none"`` only on v6.0.0 and earlier for backward compatibility. A relaxation
#: or MD run is exactly where the missing smoothing shows up, so ``"poly"`` is
#: the default here.
DEFAULT_CUTOFF_SMOOTHING = "poly"

#: Smoothings torch-dftd implements. ``"none"`` reproduces PFP v6.0.0 and
#: earlier, and torch-dftd's own default.
CUTOFF_SMOOTHINGS = ("none", "poly")


def resolve_dispersion_damping(damping: str | None) -> str:
    """Validate the damping choice, returning the name to pass to torch-dftd."""
    if damping is None:
        return DEFAULT_DAMPING
    if damping not in DAMPING_FUNCTIONS:
        valid = ", ".join(repr(name) for name in DAMPING_FUNCTIONS)
        raise DispersionError(
            f"Unknown dispersion_damping {damping!r}. Supported: {valid}."
        )
    return damping


def resolve_dispersion_cutoff(cutoff: float | None) -> float:
    """Validate the D3 cutoff, returning the value to pass to torch-dftd."""
    if cutoff is None:
        return DEFAULT_CUTOFF
    try:
        value = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise DispersionError(
            f"dispersion_cutoff must be a distance in angstrom, got {cutoff!r}."
        ) from exc
    if not value > 0:
        raise DispersionError(
            f"dispersion_cutoff must be positive, got {value!r}."
        )
    return value


def resolve_dispersion_cutoff_smoothing(smoothing: str | None) -> str:
    """Validate the cutoff smoothing, returning the name for torch-dftd."""
    if smoothing is None:
        return DEFAULT_CUTOFF_SMOOTHING
    if smoothing not in CUTOFF_SMOOTHINGS:
        valid = ", ".join(repr(name) for name in CUTOFF_SMOOTHINGS)
        raise DispersionError(
            f"Unknown dispersion_cutoff_smoothing {smoothing!r}. "
            f"Supported: {valid}."
        )
    return smoothing


def wrap_with_d3(
    base_calc: Calculator,
    *,
    xc: str,
    device: str,
    damping: str | None = None,
    cutoff: float | None = None,
    cutoff_smoothing: str | None = None,
) -> Calculator:
    """Return ``base_calc`` summed with a torch-dftd D3 correction.

    The numerical settings default to PFP's (14 Å, ``"poly"``) rather than
    torch-dftd's (50.3 Å, ``"none"``) — see :data:`DEFAULT_CUTOFF`.

    Parameters
    ----------
    xc:
        Exchange-correlation functional whose D3 parameters to use. Normally
        comes from the policy table, i.e. the model's training functional.
    device:
        Where to run the D3 term. ``"mps"`` is redirected to CPU.
    damping:
        ``"bj"`` (Becke-Johnson, the default) or ``"zero"``. The two are
        separately fitted parameter sets, not a numerical detail: for a given
        functional they can differ by a factor of two on molecule-metal
        systems, where D3 has no screening of the metal's C6 coefficients.
        ``"zero"`` is the right choice when matching a reference dataset that
        used it — OC25, for instance, is RPBE + D3 with zero damping.
    cutoff:
        Distance cutoff in angstrom; defaults to PFP's 14 Å. Pass ``50.3`` to
        get torch-dftd's own default back.
    cutoff_smoothing:
        ``"poly"`` (the default, and PFP v7.0.0+) or ``"none"``.

    Notes
    -----
    ``cnthr``, the coordination-number cutoff, is left at torch-dftd's default
    of 40 Bohr (21.2 Å). torch-dftd itself clamps it to ``cutoff`` when it is
    larger and says so, so at 14 Å the effective ``cnthr`` is 14 Å — the same
    clamp PFP's stack goes through. Setting it here instead would silently
    change what a caller who *raises* ``cutoff`` gets.
    """
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
    d3 = TorchDFTD3Calculator(
        damping=resolve_dispersion_damping(damping),
        xc=xc,
        device=d3_device,
        cutoff=resolve_dispersion_cutoff(cutoff),
        cutoff_smoothing=resolve_dispersion_cutoff_smoothing(cutoff_smoothing),
    )
    return SumCalculator([base_calc, d3])


def precheck_dispersion_xc(
    backend: str,
    key: str,
    *,
    dispersion: bool,
    dispersion_xc: str | None,
    dispersion_damping: str | None = None,
    dispersion_cutoff: float | None = None,
    dispersion_cutoff_smoothing: str | None = None,
) -> str | None:
    """Validate the dispersion policy up front and return the D3 ``xc`` to use.

    Returns ``None`` when ``dispersion`` is False (no correction). Otherwise
    delegates to :func:`resolve_dispersion_xc`, which raises
    :class:`~ase_calculator_kit.errors.DispersionError` for models that already
    include dispersion or whose functional is unverified — *before* the (heavy)
    base calculator is built, so the error is fast and offline.

    ``dispersion_damping``, ``dispersion_cutoff`` and
    ``dispersion_cutoff_smoothing`` are validated here for the same reason: a
    typo should surface before a multi-gigabyte checkpoint is loaded, not after.
    """
    if not dispersion:
        return None
    resolve_dispersion_damping(dispersion_damping)
    resolve_dispersion_cutoff(dispersion_cutoff)
    resolve_dispersion_cutoff_smoothing(dispersion_cutoff_smoothing)
    return resolve_dispersion_xc(backend, key, dispersion_xc=dispersion_xc)
