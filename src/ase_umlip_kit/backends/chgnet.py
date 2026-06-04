"""CHGNet backend (https://github.com/CederGroupHub/chgnet)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ..device import resolve_device
from ..errors import MissingDependencyError
from .base import BaseBackend


class CHGNetBackend(BaseBackend):
    name = "chgnet"
    extra = "chgnet"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str | None = None,
        checkpoint: str | None = None,
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
        """
        try:
            from chgnet.model.dynamics import CHGNetCalculator
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("CHGNet", self.extra) from exc

        use_device = resolve_device(device, allow_mps=True)

        if checkpoint is not None:
            return CHGNetCalculator.from_file(
                checkpoint, use_device=use_device, **kwargs
            )

        if model is not None:
            from chgnet.model.model import CHGNet

            loaded = CHGNet.load(model_name=model)
            return CHGNetCalculator(model=loaded, use_device=use_device, **kwargs)

        return CHGNetCalculator(use_device=use_device, **kwargs)
