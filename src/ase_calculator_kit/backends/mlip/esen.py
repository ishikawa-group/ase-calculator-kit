"""OMol25-only eSEN checkpoints using the current fairchem inference API."""

from __future__ import annotations

from .fairchem import FairChemBackend

ESEN_MODELS = (
    "esen-sm-conserving-all-omol",
    "esen-sm-direct-all-omol",
    "esen-md-direct-all-omol",
)


class ESENBackend(FairChemBackend):
    """Molecular eSEN; the conserving checkpoint is suited to ASE optimization.

    All three models use OMol25's omegaB97M-V/def2-TZVPD reference, including
    VV10. The direct models predict forces separately from energy gradients.
    Set atoms.info['charge'] and atoms.info['spin'] (multiplicity), as for UMA
    omol. Legacy eSEN-OMat checkpoints are not supported by this backend.
    """

    name = "esen"

    def create_calculator(
        self, *, model="esen-sm-conserving-all-omol", task="omol",
        inference_settings="batch", **kwargs,
    ):
        """Load an OMol25 eSEN checkpoint without changing UMA's defaults.

        ``batch`` avoids UMA-specific merge/compile assumptions and supports
        a sequence of molecules with different charge and spin. Other
        arguments, including device and dispersion validation, follow UMA.
        """
        if model not in ESEN_MODELS:
            raise ValueError(f"Unknown eSEN model {model!r}. Supported: {ESEN_MODELS}")
        if task != "omol":
            raise ValueError("The eSEN backend supports only task='omol'.")
        return super().create_calculator(
            model=model, task=task, inference_settings=inference_settings, **kwargs
        )
