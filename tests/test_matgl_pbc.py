"""Axis-specific ASE boundary conditions without torch or pretrained weights."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, PropertyNotImplementedError

from ase_calculator_kit.backends.mlip._matgl_pbc import (
    _ASEAtoms2Graph,
    _ASESumCalculator,
    make_pes_calculator,
)


class GraphRecorder:
    cutoff = 1.0
    element_types = ("Cu",)

    def get_graph(self, atoms):
        return "upstream", atoms

    def get_graph_from_processed_structure(
        self, atoms, src, dst, images, cell, elements, fractional, *, is_atoms,
    ):
        assert is_atoms and elements == self.element_types
        return src, dst, images, cell, fractional


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("pbc", list(product([False, True], repeat=3)))
def test_all_pbc_masks_keep_the_requested_periodic_images(axis, pbc):
    positions = np.ones((2, 3))
    positions[:, axis] = [0.2, 3.8]
    cell = np.full(3, 20.)
    cell[axis] = 4.
    atoms = Atoms("Cu2", positions=positions, cell=cell, pbc=pbc)
    before = atoms.copy()
    result = _ASEAtoms2Graph(GraphRecorder()).get_graph(atoms)
    if all(pbc):
        assert result[0] == "upstream" and result[1] is atoms
    else:
        src, dst, images, lattice, frac = result
        expected = set()
        if pbc[axis]:
            shift = np.zeros(3, dtype=int)
            shift[axis] = -1
            expected = {(0, 1, *shift), (1, 0, *-shift)}
        actual = {(i, j, *s) for i, j, s in zip(src, dst, images)}
        assert actual == expected
        assert lattice.shape == (1, 3, 3)
        np.testing.assert_allclose(frac @ lattice[0], atoms.positions)
        assert np.all(images[:, ~np.asarray(pbc)] == 0)
    assert atoms == before


@pytest.mark.parametrize("pbc,cell", [
    ([False, False, False], [0, 0, 0]),
    ([True, False, False], [4, 0, 0]),
    ([True, True, False], [4, 4, 0]),
])
def test_missing_nonperiodic_vectors_are_completed_without_changing_atoms(pbc, cell):
    atoms = Atoms("Cu2", positions=[[.2, .3, .4], [3.8, .3, .4]], cell=cell, pbc=pbc)
    before = atoms.copy()
    _, _, _, lattice, frac = _ASEAtoms2Graph(GraphRecorder()).get_graph(atoms)
    np.testing.assert_allclose(frac @ lattice[0], atoms.positions)
    assert np.linalg.det(lattice[0]) != 0
    assert atoms == before


def test_periodic_axis_without_a_cell_vector_is_rejected():
    atoms = Atoms("Cu", pbc=[True, False, False])
    with pytest.raises(ValueError, match="periodic direction"):
        _ASEAtoms2Graph(GraphRecorder()).get_graph(atoms)


def test_periodic_images_of_the_same_atom_are_kept():
    atoms = Atoms("Cu", cell=[.75, 0, 0], pbc=[True, False, False])
    src, dst, images, _, _ = _ASEAtoms2Graph(GraphRecorder()).get_graph(atoms)
    assert {(i, j, *s) for i, j, s in zip(src, dst, images)} == {
        (0, 0, -1, 0, 0), (0, 0, 1, 0, 0),
    }


def test_skew_cell_preserves_cartesian_geometry_and_nonperiodic_axes():
    cell = np.array([[4., 0., 0.], [1.2, 5., 0.], [.4, .7, 10.]])
    atoms = Atoms("Cu2", scaled_positions=[[.05, .4, .3], [.95, .4, .3]],
                  cell=cell, pbc=[True, False, False])
    src, dst, images, lattice, frac = _ASEAtoms2Graph(GraphRecorder()).get_graph(atoms)
    displacement = (frac[dst] - frac[src] + images) @ lattice[0]
    np.testing.assert_allclose(np.linalg.norm(displacement, axis=1), .4)
    np.testing.assert_allclose(frac @ lattice[0], atoms.positions)


class FakePESCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._atoms2graph = GraphRecorder()

    def calculate(self, atoms, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 1., "forces": np.zeros((len(atoms), 3)), "stress": np.ones(6)}


def test_stress_requires_real_volume_but_energy_and_forces_do_not():
    atoms = Atoms("Cu", cell=[4, 4, 0], pbc=[True, True, False])
    calc = make_pes_calculator(FakePESCalculator)
    atoms.calc = calc
    assert atoms.get_potential_energy() == 1.
    np.testing.assert_array_equal(atoms.get_forces(), np.zeros((1, 3)))
    with pytest.raises(PropertyNotImplementedError, match="stress"):
        atoms.get_stress()
    atoms.cell[2, 2] = 20.
    np.testing.assert_array_equal(atoms.get_stress(), np.ones(6))
    # Conversion is local to the returned calculator, not the upstream class.
    assert isinstance(calc, FakePESCalculator)
    assert isinstance(calc._atoms2graph, _ASEAtoms2Graph)
    assert isinstance(FakePESCalculator()._atoms2graph, GraphRecorder)


def test_sum_completes_only_open_axes_and_caches_against_original_cell():
    atoms = Atoms("Cu", positions=[[.2, .3, .4]], cell=[4, 4, 0], pbc=[True, True, False])
    before = atoms.copy()
    parts = [FakePESCalculator(), FakePESCalculator()]
    calc = _ASESumCalculator(parts)
    atoms.calc = calc
    assert atoms.get_potential_energy() == 2.
    np.testing.assert_array_equal(atoms.get_forces(), np.zeros((1, 3)))
    assert calc.check_state(atoms) == []
    for part in parts:
        assert part.atoms.cell.rank == 3
        np.testing.assert_array_equal(part.atoms.pbc, before.pbc)
        np.testing.assert_array_equal(part.atoms.positions, before.positions)
        np.testing.assert_array_equal(part.atoms.cell[:2], before.cell[:2])
    with pytest.raises(PropertyNotImplementedError, match="stress"):
        atoms.get_stress()
    assert "stress_contributions" not in calc.results
    assert atoms == before
    atoms.cell[2, 2] = 20.
    np.testing.assert_array_equal(atoms.get_stress(), np.full(6, 2.))


def test_sum_does_not_complete_a_missing_periodic_vector():
    atoms = Atoms("Cu", pbc=[True, False, False])
    atoms.calc = _ASESumCalculator([FakePESCalculator(), FakePESCalculator()])
    with pytest.raises(ValueError, match="periodic direction"):
        atoms.get_potential_energy()
