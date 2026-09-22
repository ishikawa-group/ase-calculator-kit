"""Explicit electronic-state and external-field checks for molecular MLIP calculators."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from ase.calculators.calculator import all_changes


def _extract_molecular_state(atoms, info_keys=None) -> dict[str, Any]:
    """Validate and extract total charge, spin multiplicity, and external field."""
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

    charge = int(atoms.info[charge_key])
    spin = int(atoms.info[spin_key])
    if spin < 1:
        raise ValueError("atoms.info['spin'] is multiplicity and must be >= 1.")

    electrons = int(atoms.numbers.sum()) - charge
    spin_s = spin - 1
    if electrons < spin_s or (electrons - spin_s) % 2:
        raise ValueError("OMol charge and spin multiplicity are inconsistent with the electron count.")

    field_key = info_keys.get("external_field", "external_field")
    raw_field = atoms.info.get(field_key, None)
    if raw_field is None:
        field = np.zeros(3, dtype=float)
        has_field = False
    else:
        try:
            arr = np.asarray(raw_field, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"atoms.info[{field_key!r}] must be a 3-vector of floats.") from exc
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise ValueError(f"atoms.info[{field_key!r}] must be a 3-vector of finite floats.")
        field = arr.copy()
        has_field = True

    return {
        "charge": charge,
        "spin": spin,
        "charge_key": charge_key,
        "spin_key": spin_key,
        "field_key": field_key,
        "has_field": has_field,
        "external_field": field,
    }


def _states_equal(s1: dict[str, Any] | None, s2: dict[str, Any] | None) -> bool:
    if s1 is None or s2 is None:
        return False
    if s1["charge"] != s2["charge"] or s1["spin"] != s2["spin"]:
        return False
    return bool(np.allclose(s1["external_field"], s2["external_field"], atol=1e-12, rtol=1e-12))


def molecular_calculator_type(upstream):
    class ValidatedMolecularCalculator(upstream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_molecular_state = None

        def reset(self):
            super().reset()
            self._last_molecular_state = None

        def check_state(self, atoms, tol=1e-15):
            changes = super().check_state(atoms, tol)
            if atoms is not None:
                try:
                    curr = _extract_molecular_state(atoms, getattr(self, "info_keys", None))
                    if not _states_equal(curr, getattr(self, "_last_molecular_state", None)):
                        changes.append("electronic_state")
                except ValueError:
                    changes.append("electronic_state")
            return changes

        def get_property(self, name, atoms=None, allow_calculation=True):
            candidate = atoms if atoms is not None else self.atoms
            if candidate is not None:
                try:
                    curr = _extract_molecular_state(candidate, getattr(self, "info_keys", None))
                except ValueError:
                    self.results = {}
                    self._last_molecular_state = None
                    raise
                if not _states_equal(curr, getattr(self, "_last_molecular_state", None)):
                    self.results = {}
            return super().get_property(name, atoms, allow_calculation)

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            target = atoms if atoms is not None else self.atoms
            state = _extract_molecular_state(target, getattr(self, "info_keys", None))

            self.results = {}

            # Prepare computation copy: ensure Python int for charge/spin, independent copy of field
            calc_atoms = target.copy()
            calc_atoms.info[state["charge_key"]] = state["charge"]
            calc_atoms.info[state["spin_key"]] = state["spin"]
            if state["has_field"]:
                calc_atoms.info[state["field_key"]] = state["external_field"].copy()

            try:
                super().calculate(calc_atoms, properties, system_changes)
            except Exception:
                self.results = {}
                self._last_molecular_state = None
                raise

            self._last_molecular_state = {
                "charge": state["charge"],
                "spin": state["spin"],
                "external_field": state["external_field"].copy(),
            }

    # ASE derives its calculator name from the class name; preserve that metadata.
    ValidatedMolecularCalculator.__name__ = upstream.__name__
    return ValidatedMolecularCalculator
