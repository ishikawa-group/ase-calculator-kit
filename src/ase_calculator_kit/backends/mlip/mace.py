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

from pathlib import Path

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


def _validate_known_head(model: str | Path, head: str | None) -> None:
    """Reject a head the selected checkpoint does not carry, before loading it."""
    known = _KNOWN_HEADS.get(str(model))
    if known is None or head is None or head in known:
        return
    raise ValueError(
        f"Unknown MACE head '{head}' for model '{model}'. "
        f"Use one of: {', '.join(known)}. {_HEAD_FALLBACK_WARNING}"
    )


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
        params.update(kwargs)
        # `dispersion=` is handled here, not by mace_mp: the policy table is this
        # package's single source of truth for which xc may be added to what.
        bare = mace_mp(**params)
        _reject_silent_head_fallback(bare, head)

        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device,
                damping=dispersion_damping,
            )
        return bare
