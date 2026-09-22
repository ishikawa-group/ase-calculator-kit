"""Small boundary checks; numerical validation is documented separately."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import CalculationFailed
from ase.units import Bohr, Hartree

from ase_calculator_kit.backends.dft.pyscf import PySCFCalculator, _electronic_state, _parameters
from ase_calculator_kit.errors import DispersionError

BASE = {"basis": "sto-3g", "charge": 0, "multiplicity": 1}


def test_electronic_state_and_unknown_options():
    assert _parameters(dict(BASE, multiplicity=3))["spin"] == 2
    for extra in ({"spin": 1}, {"charge": 0.5}, {"multiplicity": 0},
                  {"typo": True}, {"grids": {"prune": "typo"}},
                  {"auxbasis": "def2-universal-jkfit"}):
        with pytest.raises(ValueError):
            _parameters(BASE | extra)
    p = _parameters({"basis": "sto-3g"})
    state, sources = _electronic_state(p, {"charge": -1, "spin": 2})
    assert state == {"charge": -1, "spin": 1, "multiplicity": 2}
    assert sources == {"charge": "atoms.info", "spin": "atoms.info"}
    with pytest.raises(ValueError, match="Missing spin"):
        _electronic_state(p, {"charge": 0})


def test_periodic_input_is_rejected_without_loading_pyscf(tmp_path):
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, .75]], pbc=True)
    a.calc = PySCFCalculator(parameters=BASE, directory=tmp_path)
    with pytest.raises(ValueError, match="molecules only"):
        a.get_potential_energy()


@pytest.mark.parametrize("converged", [True, False])
def test_upstream_units_cache_and_scf_failure(monkeypatch, tmp_path, converged):
    calls = []
    gradient = np.array([[0., 0., .1], [0., 0., -.1]])
    mf = SimpleNamespace(converged=converged, kernel=lambda: calls.append(1) or -1.,
                         nuc_grad_method=lambda: SimpleNamespace(kernel=lambda: gradient))
    gto = SimpleNamespace(Mole=lambda: SimpleNamespace(build=lambda: None, nelectron=2))
    def make_scf(mol):
        mf.mol = mol
        return mf
    dft = SimpleNamespace(RKS=make_scf)
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(gto=gto, dft=dft, scf=None))
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, .75]])
    a.calc = PySCFCalculator(parameters=BASE, directory=tmp_path)
    if not converged:
        with pytest.raises(CalculationFailed, match="did not converge"):
            a.get_forces()
        assert not a.calc.results
        return
    assert a.get_potential_energy() == -Hartree
    retained_log = a.calc._scf_log
    np.testing.assert_allclose(a.get_forces(), -gradient * Hartree / Bohr)
    assert retained_log.closed
    assert a.calc._scf is None
    assert a.get_potential_energy() == -Hartree
    assert len(calls) == 1
    a.positions[1, 2] += .01
    a.get_forces()
    assert len(calls) == 2
    a.info.update(charge=0, spin=1)
    a.get_forces()
    with pytest.raises(ValueError, match="mismatch"):
        a.info["spin"] = 3
        a.get_potential_energy()  # must reject even with a cached energy
    a.info.update(charge=0, spin=1)
    # An info-only state change must run SCF again, even at fixed coordinates.
    dft.UKS = dft.RKS
    a.calc = PySCFCalculator(parameters={"basis": "sto-3g"}, directory=tmp_path)
    a.get_forces()
    before = len(calls)
    a.info["spin"] = 3
    a.get_potential_energy()
    assert len(calls) == before + 1
    assert a.calc.metadata["spin"] == 2
    a.info.clear()
    # The NLC/dispersion guard runs before SCF or any dispersion import.
    dft.libxc = SimpleNamespace(is_nlc=lambda xc: True)
    a.calc = PySCFCalculator(parameters=BASE | {"xc": "wb97m_v", "disp": "d3bj"},
                             directory=tmp_path)
    with pytest.raises(DispersionError):
        a.get_forces()


def test_reset_and_failed_gradient_discard_scf(monkeypatch, tmp_path):
    mf = SimpleNamespace(converged=True, kernel=lambda: -1.)
    def make(mol):
        mf.mol = mol
        return mf
    monkeypatch.setitem(sys.modules, "pyscf", SimpleNamespace(
        gto=SimpleNamespace(Mole=lambda: SimpleNamespace(build=lambda: None, nelectron=2)),
        dft=SimpleNamespace(RKS=make), scf=None,
    ))
    a = Atoms("H2", positions=[[0, 0, 0], [0, 0, .75]])
    a.calc = PySCFCalculator(parameters=BASE, directory=tmp_path)
    a.get_potential_energy()
    log = a.calc._scf_log
    a.calc.reset()
    assert log.closed and a.calc._scf is None
    a.get_potential_energy()
    log = a.calc._scf_log
    mf.nuc_grad_method = lambda: SimpleNamespace(kernel=lambda: np.full((2, 3), np.nan))
    with pytest.raises(CalculationFailed, match="invalid analytic"):
        a.get_forces()
    assert log.closed and not a.calc.results and a.calc._scf is None
