"""Release metadata should keep heavy NNP stacks opt-in."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _project_metadata() -> dict:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def test_default_install_has_no_nnp_or_dispersion_packages():
    dependencies = _project_metadata()["dependencies"]
    assert dependencies == ["ase==3.28.0", "pyyaml==6.0.3"]


def test_individual_and_all_extras_are_consistent():
    extras = _project_metadata()["optional-dependencies"]
    individual = ("chgnet", "sevennet", "mattersim", "nequip", "uma", "dispersion")
    assert set(individual).issubset(extras)
    expected_all = {requirement for name in individual for requirement in extras[name]}
    assert set(extras["all"]) == expected_all


def test_release_version_is_0_3_0():
    assert _project_metadata()["version"] == "0.3.0"
