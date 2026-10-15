"""SevenNet backend (https://github.com/MDIL-SNU/SevenNet)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import get_dispersion_policy, precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend

#: Pretrained keywords that carry a modal map, and the modal to use by default.
#:
#: sevenn accepts both the ``7net-`` and ``sevennet-`` spellings for each of
#: these (``sevenn.util.pretrained_name_to_path``), so both are listed.
_MULTI_FIDELITY_DEFAULT_MODAL = "mpa"
#: ``7net-mf-0`` is deliberately absent: it is multi-fidelity, but its modal
#: names are not verified here, and guessing ``mpa`` for it would be exactly the
#: kind of unchecked assumption this table exists to avoid. Left unlisted, it
#: gets no modal and sevenn answers with its own list of the real ones.
_MULTI_FIDELITY_MODELS = frozenset(
    f"{prefix}-{name}"
    for prefix in ("7net", "sevennet")
    for name in ("omni", "omni-i8", "omni-i12", "mf-ompa")
)

#: Pretrained keywords with no modal map. sevenn does not reject a ``modal`` for
#: these — it warns and drops it (``calculator.py``: "modal=... is ignored as
#: model has no modal_map") — which is exactly why this package has to know:
#: a dropped modal must not go on to choose the model's D3 parameters.
_SINGLE_FIDELITY_MODELS = frozenset(
    f"{prefix}-{name}"
    for prefix in ("7net", "sevennet")
    for name in (
        "omat", "0", "0_11Jul2024", "0_22May2024", "l3i5",
        "nano-4.5", "nano-5.0", "nano-5.5", "nano-6.0",
    )
)


def _resolve_modal(model: str, modal: str | None) -> str | None:
    """Turn ``modal="auto"`` into the modal this checkpoint can actually use."""
    if modal != "auto":
        if modal is not None and model in _SINGLE_FIDELITY_MODELS:
            raise ValueError(
                f"SevenNet model '{model}' is single-fidelity and takes no "
                f"modal, but modal={modal!r} was given. sevenn would drop it "
                "with a warning and this package would still have used it to "
                "pick the D3 functional. Pass modal=None (or leave the default "
                "modal='auto')."
            )
        return modal
    if model in _MULTI_FIDELITY_MODELS:
        return _MULTI_FIDELITY_DEFAULT_MODAL
    # Unknown keyword: send no modal, so a multi-fidelity model released after
    # this version answers with sevenn's own "modal argument missing (avail:
    # [...])" instead of being handed a guess.
    return None


#: Class sevenn swaps in for every ``IrrepsConvolution`` when FlashTP is active
#: (``sevenn.nn.flash_helper.patch_convolution``). Its presence in the built
#: model is the only reliable evidence that FlashTP is really in use: sevenn
#: answers a missing ``flashTP_e3nn`` or a CPU device with a warning and falls
#: back to plain e3nn, so a run asked to be fast would silently not be.
_FLASH_CONVOLUTION = "IrrepsScatterGatterFusedConvolution"


def _check_flash_request(
    resolved_device: str, *, enable_cueq: bool, enable_oeq: bool
) -> None:
    """Refuse a FlashTP request that sevenn would quietly downgrade."""
    if enable_cueq or enable_oeq:
        raise ValueError(
            "enable_flash=True cannot be combined with enable_cueq or "
            "enable_oeq: sevenn replaces the tensor products with exactly one "
            "accelerator."
        )
    if not resolved_device.startswith("cuda"):
        raise ValueError(
            f"enable_flash=True needs a CUDA device, got device={resolved_device!r}. "
            "FlashTP is a set of CUDA kernels; on CPU or MPS sevenn would fall "
            "back to e3nn with only a warning."
        )
    try:
        import flashTP_e3nn  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError("flashTP_e3nn") from exc


def _uses_flash(calc: Calculator) -> bool:
    """Whether the calculator's model actually runs FlashTP convolutions."""
    model = getattr(calc, "model", None)
    modules = getattr(model, "modules", None)
    if modules is None:
        return False
    return any(type(m).__name__ == _FLASH_CONVOLUTION for m in modules())


def _policy_key(model: str, modal: str | None) -> str:
    """The dispersion-policy key: the effective modal, else the model itself.

    With no modal there is nothing but the checkpoint to go on, so a model with
    its own row uses it (``7net-omat`` is OMat24, not MPtrj) and everything else
    falls back to the single-fidelity ``"default"`` row as before.
    """
    if modal is not None:
        return modal
    if get_dispersion_policy("sevennet", model) is not None:
        return model
    return "default"


class SevenNetBackend(BaseBackend):
    name = "sevennet"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "7net-omni",
        modal: str | None = "auto",
        enable_cueq: bool = False,
        enable_flash: bool = False,
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        dispersion_damping: str | None = None,
        dispersion_cutoff: float | None = None,
        dispersion_cutoff_smoothing: str | None = None,
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
            Pretrained keyword. Defaults to ``"7net-omni"``. The Omni family is
            one training recipe at three capacities, so the ``modal`` table
            below applies unchanged to all three:

            ================= ===============================================
            ``model``         Capacity
            ================= ===============================================
            ``7net-omni``     Recommended default
            ``7net-omni-i8``  Larger; more accurate, slower
            ``7net-omni-i12`` Largest of the family
            ================= ===============================================

            Beyond the Omni family, ``"7net-mf-ompa"`` is the other
            multi-fidelity model, while ``"7net-omat"`` (OMat24-only, PBE(+U)),
            ``"7net-l3i5"``, ``"7net-0"`` and the ``"7net-nano-*"`` models are
            single-fidelity and take no ``modal`` — ``modal="auto"`` handles
            that for you. Keep the model fixed across a campaign: i8 and i12 are
            not drop-in refinements of an ``omni`` number, they are separate
            models. sevenn's ``sevennet-`` spellings work too.
        modal:
            Inference task for the multi-fidelity models. ``"auto"`` (the
            default) sends ``"mpa"`` to the Omni family and ``7net-mf-ompa``,
            and nothing at all to a single-fidelity model, so
            ``model="7net-omat"`` needs no second argument.

            Passing an explicit ``modal`` to a single-fidelity model raises.
            That is deliberate: sevenn only *warns* and drops it, and this
            package would otherwise have gone on to pick the D3 functional from
            a modal the model never saw — ``model="7net-0",
            modal="matpes_r2scan"`` would have applied r2SCAN parameters to a
            PBE model.

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
        enable_cueq:
            Use cuEquivariance tensor products; only enable when the local
            SevenNet/CUDA stack supports them.
        enable_flash:
            Use FlashTP's fused tensor-product kernels (CUDA only). FlashTP is
            not on PyPI: build it against the installed torch with
            ``pip install --no-build-isolation
            git+https://github.com/SNU-ARC/flashTP.git@v0.1.1``. sevenn itself
            only *warns* and falls back to e3nn when FlashTP is missing or the
            device is not CUDA, so here the request is checked before the model
            is loaded (``MissingDependencyError`` / ``ValueError``) and the
            built model is checked afterwards (``RuntimeError`` if its
            convolutions were not replaced). Cannot be combined with
            ``enable_cueq`` or ``enable_oeq``. FlashTP changes the kernels,
            not the model: energies agree with the e3nn path to float32
            round-off.
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
        dispersion_cutoff, dispersion_cutoff_smoothing:
            The D3 term's numerical settings. They default to PFP's 14 Å
            and ``"poly"``, not torch-dftd's own 50.3 Å and ``"none"``,
            so that the correction added here is the same quantity PFP
            adds. See ``ase_calculator_kit.dispersion.DEFAULT_CUTOFF``.
        """
        resolved_modal = _resolve_modal(model, modal)

        # Validate the dispersion policy before loading the model (fail fast).
        d3_xc = precheck_dispersion_xc(
            self.name, _policy_key(model, resolved_modal),
            dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
            dispersion_cutoff=dispersion_cutoff,
            dispersion_cutoff_smoothing=dispersion_cutoff_smoothing,
        )
        resolved_device = resolve_device(device, allow_mps=True)
        if enable_flash:
            _check_flash_request(
                resolved_device,
                enable_cueq=enable_cueq,
                enable_oeq=bool(kwargs.get("enable_oeq", False)),
            )

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
        if resolved_modal is not None:
            params["modal"] = resolved_modal
        params.update(kwargs)
        bare = SevenNetCalculator(**params)
        if enable_flash and not _uses_flash(bare):
            raise RuntimeError(
                "enable_flash=True was requested, but the loaded SevenNet model "
                f"contains no {_FLASH_CONVOLUTION} layer: sevenn fell back to "
                "e3nn. Check its warnings (GPU visibility, flashTP_e3nn build)."
            )

        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device,
                damping=dispersion_damping,
                cutoff=dispersion_cutoff,
                cutoff_smoothing=dispersion_cutoff_smoothing,
            )
        return bare
