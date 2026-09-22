"""OrbMol backend (https://github.com/orbital-materials/orb-models)."""

from __future__ import annotations

from ase.calculators.calculator import Calculator

from ...device import resolve_device
from ...dispersion import precheck_dispersion_xc
from ...errors import MissingDependencyError
from ..base import BaseBackend

#: Accepted spellings of the one OrbMol checkpoint this backend loads.
#:
#: ``orbmol-v2`` is the key in orb-models' own ``ORB_PRETRAINED_MODELS``
#: registry; ``orbmol_v2`` is the loader function's name. Both read naturally,
#: so both are accepted and normalized to the registry spelling, which is what
#: keys ``dispersion.py``.
_MODEL_ALIASES = {"orbmol-v2": "orbmol-v2", "orbmol_v2": "orbmol-v2"}

#: Precisions ``orb_models.forcefield.pretrained`` accepts.
PRECISIONS = ("float32-high", "float32-highest", "float64")


class OrbBackend(BaseBackend):
    """OrbMol-v2, Orbital Materials' molecular potential with electrostatics."""

    name = "orb"

    def create_calculator(
        self,
        *,
        device: str = "auto",
        model: str = "orbmol-v2",
        precision: str = "float32-high",
        compile: bool | None = None,
        dispersion: bool = False,
        dispersion_xc: str | None = None,
        dispersion_damping: str | None = None,
        dispersion_cutoff: float | None = None,
        dispersion_cutoff_smoothing: str | None = None,
        **kwargs,
    ) -> Calculator:
        """Create an ``orb_models`` ``ORBCalculator`` for OrbMol-v2.

        OrbMol-v2 continues the Orb-v3 architecture with **learnable
        electrostatics**: a latent-charge head predicts per-atom charges
        constrained to sum to the system's total charge, and a Coulomb module
        adds the long-range term — a bare 1/r sum for isolated systems, particle
        mesh Ewald for periodic ones. It is trained on OMol25 and OPoly26 at
        ωB97M-V/def2-TZVPD, the same reference level as UMA's ``omol`` task and
        SevenNet's ``omol25_*`` modals, and unlike OrbMol-v1 it is trained on
        periodic systems as well.

        The per-atom charges are a *latent* feature: the model never saw
        reference charges during training, and upstream says so explicitly.
        Read them (``calc.results``) as a diagnostic, not as a population
        analysis.

        Parameters
        ----------
        device:
            ``"auto"`` (cuda > cpu) or explicit ``"cuda"`` / ``"cpu"``. Apple
            Silicon ``"mps"`` is rejected: graph construction goes through
            NVIDIA Warp (``nvalchemiops``), which has no Metal backend and
            raises ``Unsupported Torch device type mps``. The legacy edge
            methods are no escape — ``knn_brute_force`` builds the neighbor
            list in float64, which MPS cannot hold, and ``knn_scipy`` refuses
            any non-CPU device itself.
        model:
            ``"orbmol-v2"`` (default), also spelled ``"orbmol_v2"``. The rest of
            orb-models' registry — the Orb-v3 OMat/MPA models, Orb-v2, OrbMol-v1
            — is deliberately not wired up here yet: each is a different
            reference level and needs its own dispersion-policy row.
        precision:
            ``"float32-high"`` (upstream's default), ``"float32-highest"``, or
            ``"float64"``. The first two differ only in matmul precision;
            ``"float64"`` runs the whole model in double precision and is
            correspondingly slower.
        compile:
            Whether to ``torch.compile`` the model. ``None`` (default) follows
            orb-models, which compiles on every device but MPS. Compilation
            moves cost into the *first* single point: measured on CPU, ~7 s for
            the first call against ~1.4 s uncompiled, both settling at ~0.02 s
            afterwards. It pays off across a trajectory and not in a one-shot
            script, where ``False`` is the better choice. The very first run on
            a machine also pays NVIDIA Warp's one-off neighbor-list kernel
            compilation, which this flag does not control and which is cached.
        dispersion:
            Always refused. OMol25 and OPoly26 are computed with ωB97M-V, whose
            VV10 nonlocal term already carries long-range dispersion, so a D3
            correction on top would double-count it. ``dispersion=True`` raises
            :class:`~ase_calculator_kit.errors.DispersionError`, and no
            ``dispersion_xc`` overrides that — see ``docs/models.md``.
        dispersion_xc, dispersion_damping, dispersion_cutoff,
        dispersion_cutoff_smoothing:
            Accepted for API symmetry with the other backends and validated,
            but unreachable while ``dispersion`` is refused.
        **kwargs:
            Forwarded to ``ORBCalculator`` — ``edge_method``,
            ``max_num_neighbors``, ``half_supercell``, ``directory``.

        Notes
        -----
        **Set ``atoms.info["charge"]`` and ``atoms.info["spin"]`` before
        computing.** OrbMol-v2 is conditioned on both::

            atoms.info["charge"] = -1   # total charge
            atoms.info["spin"] = 2      # multiplicity, 2S+1
            atoms.calc = get_calculator("orb")

        ``ORBCalculator`` raises ``ValueError: atoms.info must contain both
        'charge' and 'spin'`` when either is absent. Kit-created UMA/eSEN
        omol calculators also require both fields since 0.5.10.

        Energy, forces and stress all come from autograd — OrbMol-v2 is a
        conservative model, and orb-models enables the stress derivative when
        preparing it for inference. In a periodic cell the Coulomb term is an
        Ewald sum, so a *charged* system is computed against a neutralizing
        background and its energy is not comparable with the isolated one.
        """
        key = _MODEL_ALIASES.get(model.lower().strip())
        if key is None:
            choices = ", ".join(sorted(set(_MODEL_ALIASES.values())))
            raise ValueError(
                f"Unknown orb model {model!r}. This backend loads {choices}; "
                "other orb-models checkpoints are not supported yet."
            )
        if precision not in PRECISIONS:
            valid = ", ".join(repr(name) for name in PRECISIONS)
            raise ValueError(f"Unknown precision {precision!r}. Supported: {valid}.")
        # Validate the dispersion policy before loading the model (fail fast).
        # OrbMol-v2 sits in the always-refused tier, so this only ever returns
        # None or raises — there is no D3 branch below on purpose.
        precheck_dispersion_xc(
            self.name, key, dispersion=dispersion, dispersion_xc=dispersion_xc,
            dispersion_damping=dispersion_damping,
            dispersion_cutoff=dispersion_cutoff,
            dispersion_cutoff_smoothing=dispersion_cutoff_smoothing,
        )
        resolved_device = resolve_device(device)

        try:
            import orb_models  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via tests with mocks
            raise MissingDependencyError("orb-models") from exc

        try:
            from orb_models.forcefield.inference.calculator import ORBCalculator
            from orb_models.forcefield.pretrained import orbmol_v2
        except ImportError as exc:
            raise ImportError(
                "This orb-models install has no OrbMol-v2. It arrived in "
                "orb-models 0.7.0; upgrade with: "
                "pip install -U 'ase-calculator-kit[orb]'"
            ) from exc

        potential, atoms_adapter = orbmol_v2(
            device=resolved_device, precision=precision, compile=compile
        )
        return ORBCalculator(
            potential, atoms_adapter=atoms_adapter, device=resolved_device, **kwargs
        )
