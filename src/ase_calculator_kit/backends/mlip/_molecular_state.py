"""Explicit electronic-state checks for molecular MLIP calculators."""

from __future__ import annotations

from numbers import Integral

from ase.calculators.calculator import all_changes


def _require_molecular_state(atoms, info_keys=None):
    """OMol uses total charge and spin multiplicity, never per-atom guesses."""
    if atoms is None:
        raise ValueError("Molecular calculator requires Atoms with charge and spin multiplicity.")
    info_keys = info_keys if info_keys is not None else {
        "total_charge": "charge", "total_spin": "spin",
    }
    try:
        charge_key, spin_key = info_keys["total_charge"], info_keys["total_spin"]
    except KeyError as exc:
        raise ValueError("Molecular info_keys must map total_charge and total_spin.") from exc
    for key in (charge_key, spin_key):
        if key not in atoms.info:
            raise ValueError(f"OMol requires atoms.info[{key!r}]; set charge and spin multiplicity.")
        value = atoms.info[key]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"atoms.info[{key!r}] must be an integer.")
    if atoms.info[spin_key] < 1:
        raise ValueError("atoms.info['spin'] is multiplicity and must be >= 1.")
    electrons = int(atoms.numbers.sum()) - int(atoms.info[charge_key])
    spin = int(atoms.info[spin_key]) - 1
    if electrons < spin or (electrons - spin) % 2:
        raise ValueError("OMol charge and spin multiplicity are inconsistent with the electron count.")


def molecular_calculator_type(upstream):
    class ValidatedMolecularCalculator(upstream):
        def get_property(self, name, atoms=None, allow_calculation=True):
            candidate = atoms if atoms is not None else self.atoms
            if candidate is not None:
                try:
                    _require_molecular_state(candidate, getattr(self, "info_keys", None))
                except ValueError:
                    self.results = {}
                    raise
            return super().get_property(name, atoms, allow_calculation)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            atoms = atoms if atoms is not None else self.atoms
            self.results = {}
            _require_molecular_state(atoms, getattr(self, "info_keys", None))
            return super().calculate(atoms, properties, system_changes)

    # ASE derives its calculator name from the class name; preserve that metadata.
    ValidatedMolecularCalculator.__name__ = upstream.__name__
    return ValidatedMolecularCalculator
