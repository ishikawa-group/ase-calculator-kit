"""CHGNet backend (https://github.com/CederGroupHub/chgnet)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend


class CHGNetBackend(BaseBackend):
    name = "chgnet"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str | None = None,
        checkpoint: str | None = None,
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        dispersion_damping: str | None = None,
        dispersion_cutoff: float | None = None,
        dispersion_cutoff_smoothing: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create a :class:`chgnet.model.dynamics.CHGNetCalculator`.

        CHGNet is best for inorganic crystalline materials and fast
        pre-relaxation. Be careful with isolated atoms and chemistry far from
        its training domain.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > mps > cpu), or ``"cuda"`` / ``"mps"`` / ``"cpu"``.
            CHGNet supports Apple Silicon ``"mps"``.
        model:
            Optional pretrained model name passed to ``CHGNet.load(model_name=...)``.
            When omitted, CHGNet's bundled default model is used.
        checkpoint:
            Optional path to a ``.pth`` checkpoint, loaded via
            ``CHGNetCalculator.from_file``.
        dispersion, dispersion_xc:
            Add a Grimme-D3(BJ) correction (CHGNet is PBE+U, so ``xc="pbe"`` by
            default). See ``docs/models.md`` for the per-model policy.
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
            self.name, model or "default",
            dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
            dispersion_cutoff=dispersion_cutoff,
            dispersion_cutoff_smoothing=dispersion_cutoff_smoothing,
        )
        use_device = resolve_device(device, allow_mps=True)

        try:
            from chgnet.model.dynamics import CHGNetCalculator
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("CHGNet") from exc

        if checkpoint is not None:
            bare = CHGNetCalculator.from_file(
                checkpoint, use_device=use_device, **kwargs
            )
        elif model is not None:
            from chgnet.model.model import CHGNet

            loaded = CHGNet.load(model_name=model)
            bare = CHGNetCalculator(model=loaded, use_device=use_device, **kwargs)
        else:
            bare = CHGNetCalculator(use_device=use_device, **kwargs)

        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=use_device,
                damping=dispersion_damping,
                cutoff=dispersion_cutoff,
                cutoff_smoothing=dispersion_cutoff_smoothing,
            )
        return bare
