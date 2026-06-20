"""NequIP OAM backend (https://www.nequip.net/)."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend

_MODEL_IDS = {
    "S": "mir-group/NequIP-OAM-S:0.1",
    "M": "mir-group/NequIP-OAM-M:0.1",
    "L": "mir-group/NequIP-OAM-L:0.1",
    "XL": "mir-group/NequIP-OAM-XL:0.1",
}


class NequIPBackend(BaseBackend):
    name = "nequip"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "L",
        model_path: str | Path | None = None,
        chemical_species_to_atom_type_map: dict[str, str] | bool | None = True,
        compile_mode: str = "eager",
        model_name: str = "sole_model",
        neighborlist_backend: str = "matscipy",
        allow_tf32: bool = False,
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create a NequIP ASE calculator for one of the OAM foundation models.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > cpu) or explicit ``"cuda"`` / ``"cpu"``. Apple
            Silicon ``"mps"`` is intentionally not enabled for NequIP OAM:
            local testing fails with ``Cannot convert a MPS Tensor to float64``
            because the packaged OAM models use float64 buffers and PyTorch MPS
            does not support float64.
        model:
            OAM model size: ``"S"``, ``"M"``, ``"L"`` (default), or ``"XL"``.
            Case-insensitive.
        model_path:
            Optional local ``.nequip.zip`` or checkpoint path. When omitted, the
            selected OAM model is loaded through NequIP's ``nequip.net:`` loader
            and cached by NequIP.
        chemical_species_to_atom_type_map:
            Passed to NequIP's ASE integration. ``True`` selects the identity
            map and silences the upstream fallback warning, which is correct for
            OAM model type names.
        compile_mode, model_name, neighborlist_backend, allow_tf32:
            Forwarded to NequIP's saved-model ASE loader.
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction only when an explicit
            ``dispersion_xc`` is provided. The OAM dispersion policy is kept
            conservative until the training functional / dispersion inclusion is
            confirmed in this project.
        """
        normalized_model = model.upper()
        if normalized_model not in _MODEL_IDS:
            valid = ", ".join(_MODEL_IDS)
            raise ValueError(
                f"Unknown NequIP OAM model '{model}'. Use one of: {valid}."
            )

        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, normalized_model,
            dispersion=dispersion, dispersion_xc=dispersion_xc,
        )
        resolved_device = resolve_device(device)

        try:
            from nequip.integrations.ase import NequIPCalculator
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("NequIP") from exc

        load_target = (
            str(model_path)
            if model_path is not None
            else f"nequip.net:{_MODEL_IDS[normalized_model]}"
        )
        bare = NequIPCalculator._from_saved_model(
            load_target,
            device=resolved_device,
            chemical_species_to_atom_type_map=chemical_species_to_atom_type_map,
            allow_tf32=allow_tf32,
            model_name=model_name,
            compile_mode=compile_mode,
            neighborlist_backend=neighborlist_backend,
            **kwargs,
        )

        if d3_xc is not None:
            return wrap_with_d3(bare, xc=d3_xc, device=resolved_device)
        return bare
