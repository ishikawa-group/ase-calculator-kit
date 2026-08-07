"""SevenNet backend (https://github.com/MDIL-SNU/SevenNet)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend


class SevenNetBackend(BaseBackend):
    name = "sevennet"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "7net-omni",
        modal: str | None = "mpa",
        enable_cueq: bool = False,
        enable_flash: bool = False,
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        dispersion_damping: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create a :class:`sevenn.calculator.SevenNetCalculator`.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > mps > cpu), or explicit ``"cuda"`` / ``"mps"`` /
            ``"cpu"``. SevenNet supports Apple Silicon ``"mps"`` (validated
            locally with the ``7net-omni`` model).
        model:
            Pretrained keyword. Defaults to ``"7net-omni"``. Other options
            include ``"7net-mf-ompa"``, ``"7net-omat"``, ``"7net-l3i5"`` and
            ``"7net-0"``.
        modal:
            Inference task for the multi-fidelity models ``7net-omni`` and
            ``7net-mf-ompa``. Set to ``None`` for single-fidelity models such as
            ``7net-0`` (which reject ``modal``).

            Choosing ``modal`` for ``7net-omni``:

            ================= ===============================================
            ``modal``         Use for
            ================= ===============================================
            ``mpa``           General PBE(+U)-level materials (default)
            ``omat24``        Broad / high-force PBE(+U) configurations
            ``matpes_pbe``    PBE without Hubbard U
            ``matpes_r2scan`` r2SCAN-level materials
            ``mp_r2scan``     r2SCAN-level Materials Project data
            ``oc20``          Catalyst surfaces and adsorption (RPBE)
            ``oc22``          Oxide catalysis (PBE(+U))
            ``odac23``        MOFs / direct air capture (PBE-D3)
            ``omol25_low``    Low-spin molecular systems (ωB97M-V)
            ``omol25_high``   High-spin molecular systems only (ωB97M-V)
            ``spice``         Drug-like molecules and peptides (ωB97M-D3(BJ))
            ``qcml``          Small molecules, wide element coverage (PBE0)
            ``pet_mad``       PBEsol-level data
            ================= ===============================================

            ``omol25_low`` and ``omol25_high`` split OMol25 by **spin state**,
            not by accuracy: pick the one matching your system's spin
            configuration. SevenNet's own guidance is that ``mpa`` stays the
            recommended default even for molecules, organic crystals, and
            molecular liquids; select another task only when consistency with a
            specific functional or benchmark protocol is required.

            **SevenNet accepts no total charge or spin multiplicity.** sevenn
            has no such input, so the modal embedding is the only handle on the
            molecular reference data, and ions or open-shell systems cannot be
            specified. Use ``get_calculator("uma", task="omol")`` when the
            charge and spin of the system must be set explicitly.
        enable_cueq, enable_flash:
            Acceleration flags; only enable when the local SevenNet/CUDA stack
            supports them.
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction. Allowed for the modals whose
            reference functional excludes dispersion (``mpa``, ``omat24``,
            ``matpes_pbe``, ``matpes_r2scan``, ``mp_r2scan``, ``oc20``, ``oc22``,
            ``pet_mad``); rejected for ``omol25_*``, ``spice``, ``qcml`` and
            ``odac23``, whose reference data already accounts for dispersion.
            See ``docs/models.md``.
        dispersion_damping:
            ``"bj"`` (Becke-Johnson, the default) or ``"zero"``. The two are
            separately fitted parameter sets. On molecule-metal systems the
            choice is not cosmetic: D3 does not screen a metal's C6
            coefficients, so for RPBE the two dampings differ by roughly a
            factor of two. Match the reference dataset when there is one --
            OC25, for example, is RPBE + D3 with zero damping.
        """
        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, modal if modal is not None else "default",
            dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
        )
        resolved_device = resolve_device(device, allow_mps=True)

        try:
            from sevenn.calculator import SevenNetCalculator
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("SevenNet") from exc

        params: dict = {
            "model": model,
            "device": resolved_device,
            "enable_cueq": enable_cueq,
            "enable_flash": enable_flash,
        }
        if modal is not None:
            params["modal"] = modal
        params.update(kwargs)
        bare = SevenNetCalculator(**params)

        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device,
                damping=dispersion_damping,
            )
        return bare
