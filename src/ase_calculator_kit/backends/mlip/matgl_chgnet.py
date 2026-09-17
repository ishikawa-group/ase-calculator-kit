"""Temporary, instance-local MatGL 4.0.3 CHGNet three-body gradient correction.

Remove this compatibility implementation after validating upstream retrained
checkpoints. This is the explicit exception to the kit's thin-factory policy;
the original Ceder CHGNet backend and other MatGL instances are unchanged.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
import types
import warnings
from importlib.metadata import version

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc
from ...errors import MissingDependencyError
from ..base import BaseBackend
from ._matgl_pbc import make_pes_calculator, wrap_with_d3
from .matgl import resolve_matpes_model


_REVISIONS = {
    "matpes-pbe": "4ec3c2c323cde07a31ca99d6719cca45919b5e5f",
    "matpes-r2scan": "4447783f53387df4642d13062cafc9571f1668d2",
}
_FORWARD_SHA256 = "2246c0b812df353d59e409e0eee400c157e8abd8469158a3561faa9a59066931"


def _corrected_forward(original):
    """Remove only the audited geometry block, refusing unfamiliar source."""
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != _FORWARD_SHA256:
        raise RuntimeError(
            "Unrecognized MatGL CHGNet.forward; the temporary correction supports "
            "the unmodified MatGL 4.0.3 source only."
        )
    tree = ast.parse(source)

    class RestoreGeometry(ast.NodeTransformer):
        count = 0

        def visit_With(self, node):
            if (
                len(node.items) == 1
                and ast.unparse(node.items[0].context_expr) == "torch.no_grad()"
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Assign)
                and isinstance(node.body[0].value, ast.Call)
                and ast.unparse(node.body[0].value.func) == "create_directed_line_graph"
            ):
                self.count += 1
                return node.body
            return self.generic_visit(node)

    patch = RestoreGeometry()
    tree = ast.fix_missing_locations(patch.visit(tree))
    if patch.count != 1:
        raise RuntimeError("Expected exactly one CHGNet three-body no_grad block.")
    # Private globals and an instance-bound method avoid a process-wide patch.
    namespace = dict(original.__globals__)
    exec(compile(tree, "<ase-calculator-kit-chgnet-three-body-gradients>", "exec"), namespace)
    return namespace[original.__name__]


class MatGLCHGNetBackend(BaseBackend):
    """Existing MatPES weights with consistent three-body energy derivatives."""

    name = "matgl-chgnet"

    def create_calculator(
        self,
        *,
        model: str = "matpes-pbe",
        device: str = "auto",
        revision: str | None = None,
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        dispersion_damping: str | None = None,
        dispersion_cutoff: float | None = None,
        dispersion_cutoff_smoothing: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Load a provisional gradient-corrected MatGL CHGNet calculator.

        Parameters
        ----------
        model:
            ``"matpes-pbe"`` or ``"matpes-r2scan"``, a full dated name, or its
            dated ``pes-matpes-...`` suffix. These weights have not been retrained.
        device:
            ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
        revision:
            Hugging Face revision. None selects the frozen benchmark revision,
            not the repository's moving main branch.
        dispersion, dispersion_xc:
            Add D3 with parameters matching the selected functional by default.
        dispersion_damping, dispersion_cutoff, dispersion_cutoff_smoothing:
            Shared kit defaults: BJ, 14 Angstrom and polynomial smoothing.
        **kwargs:
            Passed to MatGL's PESCalculator with ASE stress conventions.

        Notes
        -----
        Requires MatGL 4.0.3. Only the no_grad block around three-body geometry
        is removed on this model instance. Energy derivatives become consistent;
        benchmark accuracy is not guaranteed to improve for every property.
        Upstream retrained models will replace this provisional implementation
        after validation in a future kit release, with the change documented.
        """
        key, full_name = resolve_matpes_model(self.name, model)
        d3_xc = precheck_dispersion_xc(
            self.name, key, dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping, dispersion_cutoff=dispersion_cutoff,
            dispersion_cutoff_smoothing=dispersion_cutoff_smoothing,
        )
        for parameter, required in (
            ("stress_unit", "eV/A3"), ("stress_weight", 1.0), ("use_voigt", True),
        ):
            if kwargs.pop(parameter, required) != required:
                raise ValueError(f"MatGL requires {parameter}={required!r} for ASE stress.")
        if "potential" in kwargs:
            raise TypeError("Select the MatGL potential with model=, not potential=.")
        resolved_device = resolve_device(device, allow_mps=True)
        try:
            import matgl
            from matgl.ext.ase import PESCalculator
            from matgl.models._chgnet import CHGNet
        except ImportError as exc:
            raise MissingDependencyError("MatGL") from exc
        if version("matgl") != "4.0.3":
            raise RuntimeError(
                "The provisional matgl-chgnet backend requires matgl==4.0.3; "
                "install that version or use a kit release validated with newer MatGL."
            )
        forward = _corrected_forward(CHGNet.forward)
        resolved_revision = _REVISIONS[key] if revision is None else revision
        potential = matgl.load_model(
            f"materialyze/{full_name}", revision=resolved_revision,
        )
        if type(potential.model) is not CHGNet:
            raise RuntimeError("Expected the MatGL PyG CHGNet architecture.")
        potential.model.forward = types.MethodType(forward, potential.model)
        if resolved_device == "mps":
            potential = potential.float()
        potential = potential.to(resolved_device).eval()
        bare = make_pes_calculator(
            PESCalculator, potential=potential, stress_unit="eV/A3", use_voigt=True, **kwargs,
        )
        bare.parameters.update(
            model=full_name, revision=resolved_revision,
            three_body_gradients="kit-temporary-matgl-4.0.3",
        )
        warnings.warn(
            "matgl-chgnet is provisional: existing 2025.2.10 weights with the "
            "three-body no_grad block removed, not retrained weights. Upstream "
            "retrained models will replace this implementation after validation "
            "in a future kit release. See docs/matgl-validation.md.",
            UserWarning, stacklevel=2,
        )
        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device, damping=dispersion_damping,
                cutoff=dispersion_cutoff, cutoff_smoothing=dispersion_cutoff_smoothing,
            )
        return bare
