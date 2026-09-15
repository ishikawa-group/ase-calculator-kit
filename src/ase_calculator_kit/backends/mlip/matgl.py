"""MatPES potentials through MatGL's PyTorch Geometric implementation."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc, wrap_with_d3
from ...errors import MissingDependencyError
from ..base import BaseBackend


# Short names select these exact dated model names, never an upstream default.
MATPES_MODELS = {
    "tensornet": {
        "matpes-pbe": "TensorNet-PES-MatPES-PBE-2025.2",
        "matpes-r2scan": "TensorNet-PES-MatPES-r2SCAN-2025.2",
    },
}


def resolve_matpes_model(backend: str, model: str) -> tuple[str, str]:
    """Resolve a short, dated suffix, or full name within one architecture."""
    for key, full_name in MATPES_MODELS[backend].items():
        suffix = full_name.split("-", 1)[1]
        if model.lower() in {key, suffix.lower(), full_name.lower()}:
            return key, full_name
    choices = ", ".join(MATPES_MODELS[backend])
    raise ValueError(
        f"Unknown {backend} MatGL model {model!r}. Use {choices}, or the "
        "corresponding full dated MatPES model name."
    )


class TensorNetBackend(BaseBackend):
    """TensorNet MatPES potentials through MatGL PyG."""

    name = "tensornet"

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
        """Create a MatGL PyG potential trained on MatPES without Hubbard U.

        Parameters
        ----------
        model:
            ``"matpes-pbe"`` (default), ``"matpes-r2scan"``, or the full dated
            name. Choose the functional consistent with your reference data.
        device:
            ``"auto"`` (CUDA > MPS > CPU), ``"cpu"``, ``"cuda"``, or ``"mps"``.
            MPS uses float32 for parameters and floating buffers.
        revision:
            Optional Hugging Face commit for reproducible model weights.
        dispersion, dispersion_xc:
            Add D3(BJ), with PBE or r2SCAN parameters selected by the model.
        dispersion_damping, dispersion_cutoff, dispersion_cutoff_smoothing:
            Shared D3 settings; defaults are ``"bj"``, 14 Angstrom, ``"poly"``.
        **kwargs:
            Forwarded to ``matgl.ext.ase.PESCalculator``. Stress uses ASE units
            (eV/Angstrom cubed) and Voigt notation.
        """
        key, full_name = resolve_matpes_model(self.name, model)
        d3_xc = precheck_dispersion_xc(
            self.name, key, dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
            dispersion_cutoff=dispersion_cutoff,
            dispersion_cutoff_smoothing=dispersion_cutoff_smoothing,
        )
        # ASE's stress must be eV/A^3; MatGL defaults to GPa. Voigt output also
        # matches torch-dftd when SumCalculator adds the two stresses.
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
        except ImportError as exc:
            raise MissingDependencyError("MatGL") from exc

        load_kwargs = {} if revision is None else {"revision": revision}
        potential = matgl.load_model(f"materialyze/{full_name}", **load_kwargs)
        if resolved_device == "mps":
            # TensorNet ships float64 buffers although its weights are float32.
            # Cast before moving: MPS cannot even store a float64 tensor. CPU/MPS
            # energies, forces and stresses are recorded in docs/matgl-validation.md.
            potential = potential.float()
        potential = potential.to(resolved_device).eval()
        bare = PESCalculator(
            potential=potential, stress_unit="eV/A3", use_voigt=True, **kwargs,
        )
        if d3_xc is not None:
            return wrap_with_d3(
                bare, xc=d3_xc, device=resolved_device,
                damping=dispersion_damping, cutoff=dispersion_cutoff,
                cutoff_smoothing=dispersion_cutoff_smoothing,
            )
        return bare
