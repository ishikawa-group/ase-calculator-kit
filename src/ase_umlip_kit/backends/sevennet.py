"""SevenNet backend (https://github.com/MDIL-SNU/SevenNet)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ..device import resolve_device
from ..dispersion import precheck_dispersion_xc, wrap_with_d3
from ..errors import MissingDependencyError
from .base import BaseBackend


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
        **kwargs,
    ) -> Calculator:
        """Create a :class:`sevenn.calculator.SevenNetCalculator`.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > cpu) or explicit ``"cuda"`` / ``"cpu"``.
        model:
            Pretrained keyword. Defaults to ``"7net-omni"``. Other options
            include ``"7net-mf-ompa"``, ``"7net-omat"``, ``"7net-l3i5"`` and
            ``"7net-0"``.
        modal:
            Inference task for the multi-fidelity models ``7net-omni`` and
            ``7net-mf-ompa``. Set to ``None`` for single-fidelity models such as
            ``7net-0`` (which reject ``modal``).

            Choosing ``modal`` for ``7net-omni``:

            ============== ===============================================
            ``modal``      Use for
            ============== ===============================================
            ``mpa``        General PBE(+U)-level materials (default)
            ``omat24``     Broad / high-force PBE(+U) configurations
            ``matpes_pbe`` PBE without Hubbard U
            ``matpes_r2scan`` r2SCAN-level materials
            ``omol25_low`` Molecular / high-fidelity molecular systems
            ``omol25_high`` High-spin molecular configurations only
            ============== ===============================================
        enable_cueq, enable_flash:
            Acceleration flags; only enable when the local SevenNet/CUDA stack
            supports them.
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction. Allowed for the PBE-level modals
            (``mpa``/``omat24``/``matpes_pbe``); rejected for ``omol25_*`` (already
            includes nonlocal dispersion); ``matpes_r2scan`` needs an explicit
            ``dispersion_xc``. See ``docs/models.md``.
        """
        try:
            from sevenn.calculator import SevenNetCalculator
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("SevenNet") from exc

        resolved_device = resolve_device(device)
        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, modal if modal is not None else "default",
            dispersion=dispersion, dispersion_xc=dispersion_xc,
        )

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
            return wrap_with_d3(bare, xc=d3_xc, device=resolved_device)
        return bare
