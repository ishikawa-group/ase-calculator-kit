"""fairchem / UMA backend (https://github.com/facebookresearch/fairchem)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend
from ._molecular_state import molecular_calculator_type

if TYPE_CHECKING:  # never imported at runtime - see invariant 3
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

#: fairchem's named inference presets, checked here rather than upstream.
#:
#: ``guess_inference_settings`` validates the name with a bare ``assert``, which
#: ``python -O`` strips: the typo then reaches ``NAME_TO_INFERENCE_SETTING[...]``
#: as a ``KeyError`` with no list of what was valid. Checking the name here also
#: means the error arrives before the checkpoint download. The tuple matches
#: fairchem-core 2.24's ``NAME_TO_INFERENCE_SETTING``, which is why the ``uma``
#: extra requires ``>=2.24``: an older install has no ``nitro``, ``multigpu`` or
#: ``high-precision`` and still accepts the ``turbo`` refused below.
INFERENCE_SETTINGS = (
    "nitro", "default", "multigpu", "high-precision", "batch", "traineval",
)

#: Removed in fairchem-core 2.24. Refused here with the same advice upstream
#: gives, because the name check above runs before fairchem is imported.
_REMOVED_INFERENCE_SETTINGS = {
    "turbo": (
        "inference_settings='turbo' was removed in fairchem-core 2.24. Use the "
        "default 'nitro' for single-GPU MD, or 'multigpu' (turbo's former "
        "settings) for multi-worker inference."
    ),
}


#: Single-task fine-tunes of uma-s-1p2p1 (fairchem-core 2.24), mapped to the one
#: task each carries. A fine-tune changes the level of theory, so its model name
#: rather than its task keys the D3 policy: ``omat`` on the r2SCAN fine-tune is
#: not ``omat`` on PBE+U, and ``omol`` on the D4rev fine-tune is ωB97M-D4rev.
FINE_TUNED_MODELS = {
    "uma-s-1p2p1-ft-d4rev": "omol",
    "uma-s-1p2p1-ft-r2scan": "omat",
}


class FairChemBackend(BaseBackend):
    name = "uma"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "uma-s-1p2p1",
        task: str = "auto",
        inference_settings: str | InferenceSettings = "nitro",
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
            UMA checkpoint and the one UMA Nitro is built for. It entered
            fairchem-core's model registry in 2.22.0 and Nitro arrived in
            2.24.0, which is why the ``uma`` extra requires
            ``fairchem-core>=2.24``. The earlier checkpoints stay selectable by
            name: ``"uma-s-1p2"``, ``"uma-s-1p1"``, ``"uma-m-1p1"``; they run
            without Nitro (fairchem warns and uses ``multigpu``). Anything
            not in the registry is rejected by fairchem itself; this package
            adds no aliases, so one checkpoint has exactly one name.

            Two single-task fine-tunes of ``uma-s-1p2p1`` (fairchem-core 2.24)
            move the level of theory, and both run on UMA Nitro:

            ========================= ======== ===============================
            ``model``                 ``task`` Level of theory
            ========================= ======== ===============================
            ``uma-s-1p2p1-ft-d4rev``  ``omol`` ωB97M-D4rev, dispersion included
            ``uma-s-1p2p1-ft-r2scan`` ``omat`` r²SCAN, no dispersion, not
                                               spin-polarized
            ========================= ======== ===============================

            ``ft-d4rev`` was fine-tuned on about 4M OMol25 structures relabelled
            at ωB97M-D4rev: molecules, noncovalent interactions, molecular
            liquids and crystals. Charge and multiplicity work as for the
            ``omol`` task, and ``dispersion=True`` is refused. ``ft-r2scan`` was
            trained on MAD-1.6 (arXiv:2603.02089) and treats every system as
            non-magnetic: do not use it for magnetic systems or for cohesive
            energies (isolated atoms are often magnetic) — use ``uma-s-1p2p1``
            with ``omat`` there. ``dispersion=True`` adds D3(BJ) with r²SCAN
            parameters, as fairchem recommends. Both were trained on less data
            than ``uma-s-1p2p1``, which is often more accurate at its own
            levels of theory, and neither ships isolated-atom reference
            energies, so single-atom energies may be unreliable.
        task:
            The ``task_name`` selecting the domain-specific head. ``"auto"``
            (the default) means ``"omat"`` for the multi-task checkpoints and
            the one task a fine-tune carries; any other task on a fine-tune is
            a ``ValueError`` before the download. A single UMA model serves
            many domains; pick the task matching your system:

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

            For ``omol``, the kit rejects missing/noninteger charge or
            multiplicity and inconsistent electron counts before inference or
            cached results. Upstream's neutral-singlet fallback is not used.
            Other tasks keep their existing input contract.
        inference_settings:
            How the predict unit is built, which is where UMA's speed/precision
            trade lives. One of the names below, or a ``fairchem.core``
            ``InferenceSettings`` object for anything they do not cover
            (``max_atoms``, ``edge_chunk_size``, ``execution_mode`` ...).

            ================== ==== ====================== ==================
            name               CUDA Model                  Use for
            ================== ==== ====================== ==================
            ``nitro``          TF32 UMA Nitro kernels      MD and relaxation
            ``default``        TF32 fairchem's alias of    (same as ``nitro``)
                                    ``nitro``
            ``multigpu``       TF32 merge_mole + compile   ``workers > 1``
            ``high-precision`` FP32 merge_mole + compile   model comparisons
            ``batch``          FP32 unmerged               varying structures
            ``traineval``      FP32 unmerged, graph v1     training numbers
            ================== ==== ====================== ==================

            **The default is UMA Nitro** (since 0.6.3, fairchem-core 2.24):
            a standalone ``uma-s-1p2p1`` model with fused Triton/CuTe kernels
            and CUDA graphs, for one system per call on a CUDA GPU of compute
            capability 8.x or newer. Where it cannot run — another checkpoint,
            the CPU, an older GPU, a change of composition, task, charge or
            spin — fairchem warns and falls back to ``multigpu``, and from there
            to the unmerged model, so the call still succeeds. ``turbo`` was
            removed upstream in 2.24 and is refused here with that advice.

            **Nitro and multigpu compute in TF32 on CUDA.** That is reduced
            precision, and fairchem logs ``Running in a reduced-precision
            mode`` when it applies. Use ``"high-precision"`` (merge_mole +
            compile in FP32) when the number is going into a comparison
            between models, and for Hessians: Nitro has no double backward.
            fairchem scopes TF32 to its own forward pass and restores
            ``torch.get_float32_matmul_precision()`` afterwards, so it does not
            leak into other calculators in the same process. TF32 does not
            apply on the CPU.

            **The fast paths assume a fixed system.** Composition, task, total
            charge and spin must all stay constant, which MD satisfies and a
            loop over different structures does not. fairchem notices and falls
            back with a warning, but paying for a Nitro build or a merge and a
            compile before discarding them is waste: use ``"batch"`` when the
            structures vary.
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction. The D3 ``xc`` depends on the task's
            DFT level (e.g. ``omat``→pbe, ``oc20``→rpbe), and for a fine-tune on
            the model's (``ft-r2scan``→r2scan; ``ft-d4rev`` is rejected). Rejected for the tasks
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
        if isinstance(inference_settings, str) and (
            inference_settings in _REMOVED_INFERENCE_SETTINGS
        ):
            raise ValueError(_REMOVED_INFERENCE_SETTINGS[inference_settings])
        if isinstance(inference_settings, str) and (
            inference_settings not in INFERENCE_SETTINGS
        ):
            valid = ", ".join(repr(name) for name in INFERENCE_SETTINGS)
            raise ValueError(
                f"Unknown inference_settings {inference_settings!r}. "
                f"Supported: {valid}, or an InferenceSettings object."
            )
        task, policy_key = _resolve_task(model, task)
        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, policy_key, dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
            dispersion_cutoff=dispersion_cutoff,
            dispersion_cutoff_smoothing=dispersion_cutoff_smoothing,
        )
        resolved_device = resolve_device(device)

        try:
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("fairchem-core") from exc

        predictor = pretrained_mlip.get_predict_unit(
            model, device=resolved_device, inference_settings=inference_settings
        )
        calculator_type = molecular_calculator_type(FAIRChemCalculator) if task == "omol" else (
            FAIRChemCalculator
        )
        bare = calculator_type(predictor, task_name=task, **kwargs)

        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device,
                damping=dispersion_damping,
                cutoff=dispersion_cutoff,
                cutoff_smoothing=dispersion_cutoff_smoothing,
            )
        return bare


def _resolve_task(model: str, task: str) -> tuple[str, str]:
    """Return ``(task, dispersion policy key)`` for a UMA checkpoint.

    A fine-tune carries exactly one task, so ``"auto"`` resolves to it and any
    other task is refused before the download. Its model name keys the D3
    policy, because the fine-tune's level of theory is not its task's.
    """
    fine_tune_task = FINE_TUNED_MODELS.get(model)
    if fine_tune_task is None:
        task = "omat" if task == "auto" else task
        return task, task
    if task not in ("auto", fine_tune_task):
        raise ValueError(
            f"{model} has only the {fine_tune_task!r} task; got task={task!r}. "
            "Omit task, or use uma-s-1p2p1 for the other tasks."
        )
    return fine_tune_task, model
