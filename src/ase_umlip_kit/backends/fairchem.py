"""fairchem / UMA backend (https://github.com/facebookresearch/fairchem)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ..device import resolve_device
from ..errors import MissingDependencyError
from .base import BaseBackend


class FairChemBackend(BaseBackend):
    name = "uma"
    extra = "uma"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "uma-s-1p2",
        task: str = "omat",
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
        """
        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("fairchem-core", self.extra) from exc

        predictor = pretrained_mlip.get_predict_unit(model, device=resolve_device(device))
        return FAIRChemCalculator(predictor, task_name=task, **kwargs)
