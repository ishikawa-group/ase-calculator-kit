"""Validation of molecular calculator cache invalidation and state tracking."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from ase_calculator_kit.backends.mlip._molecular_state import molecular_calculator_type


class MockMolecularCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calculate_calls = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        keys = getattr(self, "info_keys", None) or {
            "total_charge": "charge", "total_spin": "spin", "external_field": "external_field"
        }
        q_key = keys.get("total_charge", "charge")
        s_key = keys.get("total_spin", "spin")
        f_key = keys.get("external_field", "external_field")
        received_charge = atoms.info.get(q_key)
        received_spin = atoms.info.get(s_key)
        received_field = atoms.info.get(f_key)
        self.calculate_calls.append({
            "charge": received_charge,
            "charge_type": type(received_charge),
            "spin": received_spin,
            "spin_type": type(received_spin),
            "field": received_field.copy() if received_field is not None else None,
            "positions": atoms.positions.copy(),
        })
        # Distinct energy depending on charge, spin, and field
        ef_norm = np.linalg.norm(received_field) if received_field is not None else 0.0
        e = -10.0 + 1.0 * received_charge + 0.1 * received_spin + 0.01 * ef_norm
        self.results = {"energy": float(e), "forces": np.zeros((len(atoms), 3))}


def test_oh_charge_spin_switching_forces_recalculation():
    """OH doublet -> OH- singlet -> OH doublet must recalculate each time."""
    calc_cls = molecular_calculator_type(MockMolecularCalc)
    calc = calc_cls()
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
    atoms.calc = calc

    # State 1: OH neutral doublet (charge=0, multiplicity=2)
    atoms.info.update(charge=0, spin=2)
    e1 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 1
    assert e1 == pytest.approx(-9.8)

    # Calling again at same coordinates & state must reuse cache
    e1_cached = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 1
    assert e1_cached == e1

    # State 2: OH- anion singlet (charge=-1, multiplicity=1)
    atoms.info.update(charge=-1, spin=1)
    e2 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 2
    assert e2 == pytest.approx(-10.9)
    assert e2 != e1

    # State 3: Switch back to OH neutral doublet (charge=0, multiplicity=2)
    atoms.info.update(charge=0, spin=2)
    e3 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 3
    assert e3 == pytest.approx(e1)


def test_allow_calculation_false_does_not_return_stale_cache():
    """When allow_calculation=False, state change must invalidate cache and not return stale values."""
    calc_cls = molecular_calculator_type(MockMolecularCalc)
    calc = calc_cls()
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
    atoms.calc = calc

    atoms.info.update(charge=0, spin=2)
    e = atoms.get_potential_energy()
    assert e == pytest.approx(-9.8)

    # With allow_calculation=False and same state: returns cached value
    res = calc.get_property("energy", atoms, allow_calculation=False)
    assert res == pytest.approx(-9.8)

    # Change charge to -1, spin to 1
    atoms.info.update(charge=-1, spin=1)
    res_stale = calc.get_property("energy", atoms, allow_calculation=False)
    # Cache must be invalidated, so None is returned (property not present in calculation)
    assert res_stale is None
    assert not calc.results


def test_external_field_variations_and_inplace_mutation():
    """external_field: list, numpy array, inplace mutation, omission as zero field."""
    calc_cls = molecular_calculator_type(MockMolecularCalc)
    calc = calc_cls()
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
    atoms.calc = calc
    atoms.info.update(charge=0, spin=2)

    # 1. Omitted field is treated as zero field
    e_zero1 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 1
    assert "external_field" not in calc.calculate_calls[-1] or calc.calculate_calls[-1]["field"] is None

    # Explicit [0, 0, 0] as list should match omitted zero field (no recalculation)
    atoms.info["external_field"] = [0.0, 0.0, 0.0]
    e_zero2 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 1
    assert e_zero2 == e_zero1

    # Explicit np.zeros(3) should match zero field (no recalculation)
    atoms.info["external_field"] = np.zeros(3)
    e_zero3 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 1
    assert e_zero3 == e_zero1

    # 2. Field changed via new list
    atoms.info["external_field"] = [0.1, 0.0, 0.0]
    e_f1 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 2
    assert e_f1 != e_zero1

    # Same field as numpy array should reuse cache
    atoms.info["external_field"] = np.array([0.1, 0.0, 0.0])
    e_f1_cached = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 2
    assert e_f1_cached == e_f1

    # 3. Inplace mutation of the numpy array
    atoms.info["external_field"][0] = 0.2
    e_f2 = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 3
    assert e_f2 != e_f1

    # Inplace mutation with allow_calculation=False should not return stale cache
    atoms.info["external_field"][1] = 0.5
    res = calc.get_property("energy", atoms, allow_calculation=False)
    assert res is None


def test_numpy_integer_conversion_for_upstream():
    """NumPy integers in atoms.info must be converted to Python int on the computation copy."""
    calc_cls = molecular_calculator_type(MockMolecularCalc)
    calc = calc_cls()
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
    atoms.calc = calc

    atoms.info.update(charge=np.int64(0), spin=np.int32(2))
    atoms.get_potential_energy()

    call = calc.calculate_calls[-1]
    assert call["charge"] == 0
    assert call["charge_type"] is int
    assert call["spin"] == 2
    assert call["spin_type"] is int


def test_custom_info_keys():
    """calc.info_keys must be respected for mapping charge and spin keys."""
    class CustomKeyCalc(MockMolecularCalc):
        info_keys = {"total_charge": "q_tot", "total_spin": "mult", "external_field": "efield"}

    calc_cls = molecular_calculator_type(CustomKeyCalc)
    calc = calc_cls()
    atoms = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.96]])
    atoms.calc = calc

    atoms.info.update(q_tot=0, mult=2, efield=[0.0, 0.0, 0.1])
    e = atoms.get_potential_energy()
    assert len(calc.calculate_calls) == 1
    assert e is not None
