"""Release metadata should keep heavy NNP stacks opt-in."""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: Individual backend extras, in the order they appear in pyproject.toml.
_BACKEND_EXTRAS = ("chgnet", "sevennet", "mattersim", "nequip", "uma", "dispersion")


def _project_metadata() -> dict:
    path = _ROOT / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _distribution_names(requirements: list[str]) -> set[str]:
    return {requirement.split(">")[0].split("=")[0].split("<")[0] for requirement in requirements}


def test_default_install_has_no_nnp_or_dispersion_packages():
    dependencies = _project_metadata()["dependencies"]
    assert _distribution_names(dependencies) == {"ase", "pyyaml"}


def test_individual_and_all_extras_are_consistent():
    extras = _project_metadata()["optional-dependencies"]
    assert set(_BACKEND_EXTRAS).issubset(extras)
    expected_all = {req for name in _BACKEND_EXTRAS for req in extras[name]}
    assert set(extras["all"]) == expected_all


def test_published_requirements_are_ranges_not_exact_pins():
    """Exact pins belong in constraints.txt, not in the published metadata."""
    project = _project_metadata()
    extras = project["optional-dependencies"]
    published = list(project["dependencies"])
    for name in (*_BACKEND_EXTRAS, "all"):
        published.extend(extras[name])
    assert [req for req in published if "==" in req] == []


def test_constraints_cover_every_published_requirement():
    lines = (_ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines()
    pinned = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert all("==" in requirement for requirement in pinned)

    project = _project_metadata()
    extras = project["optional-dependencies"]
    required = _distribution_names(project["dependencies"]) | _distribution_names(extras["all"])
    assert required.issubset(_distribution_names(pinned))


def test_release_version_is_0_3_2():
    assert _project_metadata()["version"] == "0.3.2"


def test_requires_python_has_no_upper_bound():
    """An upper cap makes the package invisible to a newer interpreter.

    It is also permanent: the value is baked into every file uploaded to PyPI,
    so a cap can only be lifted by cutting a new release. Backends that lag a
    Python release cap themselves, and pip then names the backend.
    """
    requires_python = _project_metadata()["requires-python"]
    assert "<" not in requires_python, requires_python
