"""SevenNet backend (https://github.com/MDIL-SNU/SevenNet)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ..device import resolve_device
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
        """
        try:
            from sevenn.calculator import SevenNetCalculator
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("SevenNet") from exc

        params: dict = {
            "model": model,
            "device": resolve_device(device),
            "enable_cueq": enable_cueq,
            "enable_flash": enable_flash,
        }
        if modal is not None:
            params["modal"] = modal
        params.update(kwargs)
        return SevenNetCalculator(**params)
