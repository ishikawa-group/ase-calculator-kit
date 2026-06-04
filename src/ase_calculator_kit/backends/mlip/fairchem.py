"""fairchem / UMA backend (https://github.com/facebookresearch/fairchem)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend


class FairChemBackend(BaseBackend):
    name = "uma"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "uma-s-1p2",
        task: str = "omat",
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create a :class:`fairchem.core.FAIRChemCalculator` for a UMA model.

        UMA checkpoints are gated on Hugging Face. If creation fails with an
        authorization error, request access to the model repository and run
        ``huggingface-cli login``.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > cpu) or explicit ``"cuda"`` / ``"cpu"``.
        model:
            UMA model name. Defaults to ``"uma-s-1p2"``.
        task:
            The ``task_name`` selecting the domain-specific head. A single UMA
            model serves many domains; pick the task matching your system:

            ======= ==================================================
            ``task`` Use for
            ======= ==================================================
            ``omat`` Inorganic bulk/materials, stress, cell optimization
            ``omol`` Molecules and polymers
            ``oc20`` Catalyst surfaces and adsorption
            ``oc22`` Oxide catalysis
            ``oc25`` Electrochemistry / solid-liquid interfaces
            ``odac`` MOFs and direct air capture
            ``omc``  Molecular crystals
            ======= ==================================================

            For molecular tasks (``omol``), set ``atoms.info["charge"]`` and
            ``atoms.info["spin"]`` before computing.
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction. The D3 ``xc`` depends on the task's
            DFT level (e.g. ``omat``→pbe, ``oc20``→rpbe). Rejected for ``oc25``
            (RPBE+D3 already included) and ``omol`` (wB97M-V nonlocal dispersion);
            ``odac``/``omc`` need an explicit ``dispersion_xc``. See ``docs/models.md``.
        """
        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("fairchem-core") from exc

        resolved_device = resolve_device(device)
        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, task, dispersion=dispersion, dispersion_xc=dispersion_xc,
        )

        predictor = pretrained_mlip.get_predict_unit(model, device=resolved_device)
        bare = FAIRChemCalculator(predictor, task_name=task, **kwargs)

        if d3_xc is not None:
            return wrap_with_d3(bare, xc=d3_xc, device=resolved_device)
        return bare
