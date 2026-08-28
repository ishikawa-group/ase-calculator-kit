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
        model: str = "uma-s-1p2p1",
        task: str = "omat",
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        dispersion_damping: str | None = None,
        dispersion_cutoff: float | None = None,
        dispersion_cutoff_smoothing: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create a :class:`fairchem.core.FAIRChemCalculator` for a UMA model.

        UMA checkpoints are gated on Hugging Face. If creation fails with an
        authorization error, request access to the model repository and run
        ``huggingface-cli login``.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > cpu) or explicit ``"cuda"`` / ``"cpu"``. Apple
            Silicon ``"mps"`` is not supported: fairchem-core's predict unit
            asserts ``device in {"cpu", "cuda"}``, so ``"mps"`` is rejected
            before this wrapper runs.
        model:
            UMA model name. Defaults to ``"uma-s-1p2p1"``, the newest small
            UMA checkpoint. It is only in fairchem-core's model registry from
            **2.22.0** onwards — an older install answers this name with
            ``KeyError: Model 'uma-s-1p2p1' not found``, listing the
            checkpoints it does carry — which is why the ``uma`` extra requires
            ``fairchem-core>=2.22``. The earlier checkpoints stay selectable by
            name: ``"uma-s-1p2"``, ``"uma-s-1p1"``, ``"uma-m-1p1"``. Anything
            not in the registry is rejected by fairchem itself; this package
            adds no aliases, so one checkpoint has exactly one name.
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

            For the molecular task (``omol``), set ``atoms.info["charge"]`` (total
            charge) and ``atoms.info["spin"]`` (spin multiplicity, ``2S+1``)
            *before* computing::

                atoms.info["charge"] = -1
                atoms.info["spin"] = 2
                atoms.calc = get_calculator("uma", task="omol")

            This is not optional in practice, only in form: fairchem does **not**
            raise when they are missing. It logs a warning, writes
            ``charge=0`` / ``spin=1`` into ``atoms.info`` (mutating the object you
            passed in), and returns a neutral closed-shell result. An anion or a
            radical therefore comes back silently wrong unless both keys are set.
            ``charge`` and ``spin`` are read only by the ``omol`` head; other
            tasks ignore them.
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction. The D3 ``xc`` depends on the task's
            DFT level (e.g. ``omat``→pbe, ``oc20``→rpbe). Rejected for the tasks
            whose reference data already accounts for dispersion: ``oc25``
            (RPBE+D3 with zero damping), ``omol`` (ωB97M-V nonlocal),
            ``odac`` (PBE-D3) and ``omc``
            (PBE+D3). See ``docs/models.md``.
        dispersion_damping:
            ``"bj"`` (Becke-Johnson, the default) or ``"zero"``. The two are
            separately fitted parameter sets. On molecule-metal systems the
            choice is not cosmetic: D3 does not screen a metal's C6
            coefficients, so for RPBE the two dampings differ by roughly a
            factor of two. Match the reference dataset when there is one --
            OC25, for example, is RPBE + D3 with zero damping.
        dispersion_cutoff, dispersion_cutoff_smoothing:
            The D3 term's numerical settings. They default to PFP's 14 Å
            and ``"poly"``, not torch-dftd's own 50.3 Å and ``"none"``,
            so that the correction added here is the same quantity PFP
            adds. See ``ase_calculator_kit.dispersion.DEFAULT_CUTOFF``.
        """
        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, task, dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
            dispersion_cutoff=dispersion_cutoff,
            dispersion_cutoff_smoothing=dispersion_cutoff_smoothing,
        )
        resolved_device = resolve_device(device)

        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("fairchem-core") from exc

        predictor = pretrained_mlip.get_predict_unit(model, device=resolved_device)
        bare = FAIRChemCalculator(predictor, task_name=task, **kwargs)

        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device,
                damping=dispersion_damping,
                cutoff=dispersion_cutoff,
                cutoff_smoothing=dispersion_cutoff_smoothing,
            )
        return bare
