"""MatterSim backend (https://github.com/microsoft/mattersim)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ..device import resolve_device
from ..dispersion import precheck_dispersion_xc, wrap_with_d3
from ..errors import MissingDependencyError
from .base import BaseBackend

#: Friendly model keys -> checkpoint file names shipped with MatterSim.
_MODEL_TO_LOAD_PATH = {
    "1M": "MatterSim-v1.0.0-1M.pth",
    "5M": "MatterSim-v1.0.0-5M.pth",
    "MatterSim-v1.0.0-1M": "MatterSim-v1.0.0-1M.pth",
    "MatterSim-v1.0.0-5M": "MatterSim-v1.0.0-5M.pth",
}
_DEFAULT_MODELS = {"1M", "MatterSim-v1.0.0-1M"}


class MatterSimBackend(BaseBackend):
    name = "mattersim"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "1M",
        load_path: str | None = None,
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create a :class:`mattersim.forcefield.MatterSimCalculator`.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > cpu) or explicit ``"cuda"`` / ``"cpu"``.
        model:
            ``"1M"`` (default, fast screening) or ``"5M"`` (more accurate).
            Keep the checkpoint fixed across a campaign.
        load_path:
            Explicit checkpoint path; overrides ``model``. When ``model="1M"``
            and no ``load_path`` is given, MatterSim's default 1M checkpoint is
            used (no ``load_path`` passed).
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction (MatterSim is PBE, so ``xc="pbe"`` by
            default). See ``docs/models.md``.
        """
        try:
            from mattersim.forcefield import MatterSimCalculator
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("MatterSim") from exc

        resolved_device = resolve_device(device)
        if model in {"1M", "MatterSim-v1.0.0-1M"}:
            key = "1M"
        elif model in {"5M", "MatterSim-v1.0.0-5M"}:
            key = "5M"
        else:
            key = "default"
        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, key, dispersion=dispersion, dispersion_xc=dispersion_xc,
        )

        params: dict = {"device": resolved_device}

        resolved = load_path or _MODEL_TO_LOAD_PATH.get(model)
        if load_path is not None or model not in _DEFAULT_MODELS:
            if resolved is None:
                raise ValueError(
                    f"Unknown MatterSim model '{model}'. Use '1M', '5M', or pass "
                    "an explicit load_path."
                )
            params["load_path"] = resolved

        params.update(kwargs)
        bare = MatterSimCalculator(**params)

        if d3_xc is not None:
            return wrap_with_d3(bare, xc=d3_xc, device=resolved_device)
        return bare
