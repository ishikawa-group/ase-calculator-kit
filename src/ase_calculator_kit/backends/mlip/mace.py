"""MACE backend (https://github.com/ACEsuit/mace).

**MACE needs a virtual environment of its own.** ``mace-torch`` pins
``e3nn==0.4.4``, while ``sevenn``, ``fairchem-core``, ``mattersim`` and
``nequip`` all require ``e3nn>=0.5``. pip cannot satisfy both, so the ``mace``
extra is deliberately *not* part of ``[all]``::

    python -m venv .venv-mace
    .venv-mace/bin/pip install "ase-calculator-kit[mace]"

Everything else in this package works in that environment; only the other MLIP
backends are unavailable there, and they report it as
:class:`~ase_calculator_kit.errors.MissingDependencyError` as usual.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend

#: Heads carried by the MACE-MH-1 checkpoint, read back from the real file
#: (``MACECalculator.available_heads`` for ``mace-mh-1.model``).
#:
#: This is not the list printed on the model card, which advertises an
#: ``rgd1_b3lyp`` head: the published checkpoint ships ``mp_pbe_refit_add``
#: instead. The names below are the ones the shipped model actually answers to.
MH1_HEADS = (
    "omat_pbe",
    "mp_pbe_refit_add",
    "oc20_usemppbe",
    "matpes_r2scan",
    "omol",
    "spice_wB97M",
)

#: Multi-head checkpoints whose head names are known without downloading them.
_KNOWN_HEADS: dict[str, tuple[str, ...]] = {"mh-1": MH1_HEADS}

_HEAD_FALLBACK_WARNING = (
    "MACE does not raise on an unknown head: it logs a warning and silently "
    "computes with the last head in the checkpoint, so the energy comes back "
    "from a different level of theory than the one asked for."
)

#: Values accepted by ``accelerator=``.
ACCELERATORS = ("auto", "cueq", "oeq", "none")

#: ``accelerator`` -> the MACECalculator flag it turns on.
_ACCELERATOR_FLAG = {"cueq": "enable_cueq", "oeq": "enable_oeq"}

#: Agreement an accelerated model must show on the probe cell before ``"auto"``
#: keeps it. Generous next to float32 noise (~1e-6 eV), tight next to the
#: failure it exists to catch: cuequivariance on a multi-head checkpoint has
#: been reported returning +5500 eV where the plain model returns -200 eV
#: (ACEsuit/mace#1298).
_PROBE_ATOL_EV = 1e-3
_PROBE_RTOL = 1e-5


def _validate_known_head(model: str | Path, head: str | None) -> None:
    """Reject a head the selected checkpoint does not carry, before loading it."""
    known = _KNOWN_HEADS.get(str(model))
    if known is None or head is None or head in known:
        return
    raise ValueError(
        f"Unknown MACE head '{head}' for model '{model}'. "
        f"Use one of: {', '.join(known)}. {_HEAD_FALLBACK_WARNING}"
    )


def _resolve_accelerator(accelerator: str, kwargs: dict) -> str:
    """Validate ``accelerator=`` and let an explicit ``enable_*`` flag win."""
    normalized = accelerator.lower()
    if normalized not in ACCELERATORS:
        valid = ", ".join(repr(name) for name in ACCELERATORS)
        raise ValueError(
            f"Unknown accelerator {accelerator!r}. Supported: {valid}."
        )
    explicit = [flag for flag in _ACCELERATOR_FLAG.values() if flag in kwargs]
    if explicit:
        if normalized != "auto":
            raise ValueError(
                f"Pass either accelerator={accelerator!r} or {explicit[0]}=, "
                "not both."
            )
        # An explicit enable_cueq=/enable_oeq= is a deliberate choice; honour it
        # unprobed rather than second-guessing it.
        return "none"
    return normalized


def _probe_cell(calc) -> Atoms | None:
    """A two-atom periodic cell built from an element the model actually knows.

    Two atoms 2.5 A apart are inside every foundation-model cutoff, so the
    equivariant tensor products — the part an accelerator replaces — really run.
    """
    zs = getattr(getattr(calc, "z_table", None), "zs", None)
    if not zs:
        return None
    z = 29 if 29 in zs else int(zs[0])  # Cu when available, else whatever is
    return Atoms(
        numbers=[z, z],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def _single_point(calc, atoms: Atoms) -> float:
    probe = atoms.copy()
    probe.calc = calc
    return float(probe.get_potential_energy())


def _agrees(reference: float, candidate: float) -> bool:
    if not math.isfinite(candidate):
        return False
    tolerance = max(_PROBE_ATOL_EV, _PROBE_RTOL * abs(reference))
    return abs(candidate - reference) <= tolerance


def _autoselect_accelerator(build, params: dict, plain: Calculator) -> Calculator:
    """Return a cuequivariance-accelerated model only if it is installed *and* right.

    Installed is not the same as usable, and usable is not the same as correct:

    * cuequivariance's kernels are shipped for recent architectures only. On a
      V100 (sm_70) the calculator builds fine and then dies with
      ``cudaErrorNoKernelImageForDevice`` at the *first energy evaluation* —
      measured, not guessed — so a mere import check would hand back a
      calculator that explodes later, in the middle of someone's run.
    * ACEsuit/mace#1298 reports cuequivariance returning wildly wrong energies
      on a multi-head checkpoint (+5500 eV against -200 eV) without raising at
      all, which no import check can catch.

    So ``"auto"`` builds both, compares them on a tiny cell, and keeps the
    accelerated one only when it agrees. The cost is one extra model build and
    two two-atom single points, and only when cuequivariance is installed at
    all; anything unexpected falls back to the plain model with a warning.
    """
    if params.get("device") != "cuda":
        return plain
    try:
        import cuequivariance_torch  # noqa: F401
    except ImportError:
        return plain

    atoms = _probe_cell(plain)
    if atoms is None:
        return plain

    try:
        reference = _single_point(plain, atoms)
        accelerated = build(**params, enable_cueq=True)
        candidate = _single_point(accelerated, atoms)
    except Exception as exc:  # noqa: BLE001 - any failure means "do not use it"
        warnings.warn(
            f"cuequivariance acceleration is installed but did not run on this "
            f"GPU ({type(exc).__name__}: {exc}). Falling back to the plain MACE "
            "model. Pass accelerator='none' to skip this probe.",
            RuntimeWarning,
            stacklevel=3,
        )
        return plain

    if not _agrees(reference, candidate):
        warnings.warn(
            f"cuequivariance acceleration disagrees with the plain MACE model "
            f"on a two-atom probe ({candidate:.6f} eV against {reference:.6f} "
            "eV), so it is disabled. This is the failure mode reported in "
            "ACEsuit/mace#1298 for multi-head checkpoints.",
            RuntimeWarning,
            stacklevel=3,
        )
        return plain
    return accelerated


def _reject_silent_head_fallback(calc, head: str | None) -> None:
    """Same check against the loaded model, for checkpoints not listed above."""
    available = getattr(calc, "available_heads", None)
    if head is None or not available or head in available:
        return
    raise ValueError(
        f"MACE head '{head}' is not in this checkpoint's heads: "
        f"{', '.join(available)}. {_HEAD_FALLBACK_WARNING}"
    )


class MACEBackend(BaseBackend):
    name = "mace"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str | Path = "mh-1",
        head: str | None = "omat_pbe",
        default_dtype: str = "float64",
        accelerator: str = "auto",
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        dispersion_damping: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create a :class:`mace.calculators.MACECalculator` foundation model.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > cpu) or explicit ``"cuda"`` / ``"cpu"``. Apple
            Silicon ``"mps"`` is not supported: the MH-1 checkpoint stores
            float64 tensors, and ``torch.load(..., map_location="mps")`` fails
            with ``Cannot convert a MPS Tensor to float64`` before
            ``default_dtype`` is ever applied — measured locally with both
            ``float32`` and ``float64``, so lowering the precision does not help.
        model:
            MACE foundation model name or a local checkpoint path. Defaults to
            ``"mh-1"`` (MACE-MH-1), the multi-head cross-learning model, which
            is downloaded once and cached under ``~/.cache/mace``. Other names
            accepted by ``mace_mp`` include ``"medium-mpa-0"``,
            ``"medium-omat-0"``, ``"mace-matpes-pbe-0"`` and
            ``"mace-matpes-r2scan-0"``.
        head:
            Which readout head of a multi-head checkpoint to evaluate. Each head
            is a different *level of theory*, not a different accuracy setting:

            ==================== ================================================
            ``head``             Trained on
            ==================== ================================================
            ``omat_pbe``         OMat24 replay; PBE(+U) inorganic crystals
                                 (default, best cross-domain behaviour)
            ``mp_pbe_refit_add`` MPtrj; PBE(+U) Materials Project trajectories
            ``oc20_usemppbe``    OC20 surface slabs and adsorbates
            ``matpes_r2scan``    MatPES; r2SCAN without Hubbard U
            ``omol``             OMol25 subset; molecules and organometallics
            ``spice_wB97M``      SPICE-1; small to medium organic molecules
            ==================== ================================================

            Pass ``head=None`` for a single-head checkpoint (MACE then picks its
            own ``"Default"`` head). An unknown head is rejected here because
            MACE itself does not reject it — it warns and falls back to the last
            head in the file, which returns a plausible number from the wrong
            level of theory.
        default_dtype:
            ``"float64"`` (default, and what the MH-1 model card recommends —
            geometry optimisation and phonons need it) or ``"float32"`` for
            faster MD.
        accelerator:
            Equivariant-kernel acceleration on CUDA. ``"auto"`` (default) uses
            cuequivariance when it is installed *and* demonstrably correct,
            ``"cueq"`` / ``"oeq"`` force cuequivariance / openequivariance, and
            ``"none"`` disables the whole mechanism, probe included.

            ``"auto"`` does not stop at "is it importable", because that
            question has been measured to be the wrong one twice over: on a
            V100 (sm_70) cuequivariance builds happily and then raises
            ``cudaErrorNoKernelImageForDevice`` at the first energy evaluation,
            and ACEsuit/mace#1298 reports it returning +5500 eV where the plain
            model returns -200 eV on a multi-head checkpoint, without raising
            at all. So ``"auto"`` builds both models and compares them on a
            two-atom cell, keeping the accelerated one only when it agrees, and
            warning when it does not. The extra cost — one model build and two
            tiny single points — is paid only when cuequivariance is installed,
            which on CPU-only or plain-CUDA environments means never.

            ``enable_cueq=`` / ``enable_oeq=`` may still be passed directly; an
            explicit flag is taken as a deliberate choice and skips the probe.
        dispersion, dispersion_xc:
            Add a Grimme-D3 correction. The D3 ``xc`` follows the head's
            reference functional (``omat_pbe``/``mp_pbe_refit_add``/
            ``oc20_usemppbe``→pbe, ``matpes_r2scan``→r2scan), which is also what
            the MACE-MH-1 authors do: they evaluate their PBE-trained heads with
            torch-dftd D3(BJ) and the PBE parametrisation. Rejected for the two
            heads whose reference already contains dispersion — ``omol``
            (ωB97M-VV10) and ``spice_wB97M`` (ωB97M-D3(BJ)). A model or head
            outside the policy table is refused until you pass an explicit
            ``dispersion_xc``. See ``docs/models.md``.
        dispersion_damping:
            ``"bj"`` (Becke-Johnson, the default) or ``"zero"``. The two are
            separately fitted parameter sets. On molecule-metal systems the
            choice is not cosmetic: D3 does not screen a metal's C6
            coefficients, so for RPBE the two dampings differ by roughly a
            factor of two. Match the reference dataset when there is one --
            OC25, for example, is RPBE + D3 with zero damping.
        """
        _validate_known_head(model, head)
        resolved_accelerator = _resolve_accelerator(accelerator, kwargs)

        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, head if head is not None else "default",
            dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
        )
        resolved_device = resolve_device(device)

        try:
            from mace.calculators import mace_mp
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("mace-torch") from exc

        params: dict = {
            "model": model,
            "device": resolved_device,
            "default_dtype": default_dtype,
        }
        if head is not None:
            params["head"] = head
        if resolved_accelerator in _ACCELERATOR_FLAG:
            params[_ACCELERATOR_FLAG[resolved_accelerator]] = True
        params.update(kwargs)
        # `dispersion=` is handled here, not by mace_mp: the policy table is this
        # package's single source of truth for which xc may be added to what.
        bare = mace_mp(**params)
        if resolved_accelerator == "auto":
            bare = _autoselect_accelerator(mace_mp, params, bare)
        _reject_silent_head_fallback(bare, head)

        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device,
                damping=dispersion_damping,
            )
        return bare
