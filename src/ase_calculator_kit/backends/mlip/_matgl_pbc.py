"""ASE boundary conditions at the MatGL graph-conversion boundary."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
from ase.neighborlist import neighbor_list

from ...dispersion import wrap_with_d3 as _wrap_with_d3


def _validate_periodic_cell(atoms: Atoms) -> None:
    if np.any(atoms.pbc & ~np.asarray(atoms.cell.any(axis=1))):
        raise ValueError("Every periodic direction needs a nonzero cell vector.")


class _ASEAtoms2Graph:
    """Keep MatGL's full-PBC route; honor individual axes for all other cells."""

    def __init__(self, upstream):
        self.upstream = upstream

    def get_graph(self, atoms: Atoms):
        periodic = atoms.get_pbc()
        _validate_periodic_cell(atoms)
        if periodic.all():
            return self.upstream.get_graph(atoms)

        # ASE completes only missing, nonperiodic vectors for coordinate algebra;
        # neighbor_list still uses the original cell and its actual PBC flags.
        cell = atoms.cell.complete()
        try:
            fractional = np.linalg.solve(cell.T, atoms.positions.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("The supplied cell vectors must be linearly independent.") from exc
        src, dst, images = neighbor_list(
            "ijS", atoms, self.upstream.cutoff, self_interaction=False,
        )
        return self.upstream.get_graph_from_processed_structure(
            atoms, src, dst, images, np.asarray(cell)[None, :, :],
            self.upstream.element_types, fractional, is_atoms=True,
        )


def make_pes_calculator(calculator_cls: type[Calculator], **kwargs) -> Calculator:
    """Adapt one MatGL calculator without modifying MatGL module globals.

    The caller imports MatGL lazily and supplies its PESCalculator class, so
    importing the lightweight kit never imports torch or MatGL.
    """
    class ASEPESCalculator(calculator_cls):
        def calculate(self, atoms, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            # A completed unit-length nonperiodic axis permits E/F calculation,
            # but does not define a physical volume for stress in eV/Angstrom^3.
            if atoms.cell.rank < 3:
                self.results.pop("stress", None)

    calc = ASEPESCalculator(**kwargs)
    calc._atoms2graph = _ASEAtoms2Graph(calc._atoms2graph)
    return calc


class _ASESumCalculator(SumCalculator):
    """Complete missing open axes for D3 without assigning a physical volume."""

    def calculate(self, atoms, properties, system_changes):
        _validate_periodic_cell(atoms)
        calculation_atoms = atoms
        if atoms.cell.rank < 3:
            # torch-dftd's periodic neighbor builder requires an invertible
            # cell. Only missing nonperiodic vectors are filled; PBC is kept.
            calculation_atoms = atoms.copy()
            calculation_atoms.set_cell(atoms.cell.complete())
        super().calculate(calculation_atoms, properties, system_changes)
        self.atoms = atoms.copy()  # Cache against the caller's original cell.
        if atoms.cell.rank < 3:
            self.results.pop("stress", None)
            self.results.pop("stress_contributions", None)


def wrap_with_d3(base_calc: Calculator, **kwargs) -> Calculator:
    """Use the shared D3 policy with ASE cell handling for MatGL backends only."""
    summed = _wrap_with_d3(base_calc, **kwargs)
    return _ASESumCalculator(summed.mixer.calcs)
