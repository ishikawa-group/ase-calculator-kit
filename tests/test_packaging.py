"""Release metadata should keep heavy NNP stacks opt-in."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]

#: Individual backend extras that can share one environment, in pyproject order.
_BACKEND_EXTRAS = ("chgnet", "matgl", "sevennet", "mattersim", "nequip", "uma", "esen", "dispersion")

#: Extras deliberately excluded from `all`, and why each one is.
#:
#: mace-torch pins e3nn==0.4.4 against everyone else's e3nn>=0.5, so
#: `pip install '...[all,mace]'` has no solution at all. orb-models resolves
#: alongside every other backend, but pins dm-tree==0.1.8, whose newest wheels
#: are cp312 — putting it in `all` would make `all` stop installing on 3.13
#: and 3.14.
_EXTRAS_OUTSIDE_ALL = ("mace", "orb", "pyscf", "pyscf-dispersion", "gpu4pyscf-cuda12x")


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


def test_extras_outside_all_are_installable_but_never_part_of_all():
    """Both exclusions are the reason the extra exists, not a reason to drop it.

    Before 0.5.0 the answer to the e3nn conflict was to leave MACE out of the
    package entirely. It is supported now, in its own environment — the extra
    exists, and `all` must keep not containing it. `orb` is the same bargain
    for a Python-version reason: orb-models pins dm-tree==0.1.8, so an `all`
    carrying it would fail to install on 3.13 and 3.14.
    """
    extras = _project_metadata()["optional-dependencies"]
    for name in _EXTRAS_OUTSIDE_ALL:
        assert name in extras, f"the '{name}' extra is how it gets installed"
        excluded = _distribution_names(extras[name])
        assert excluded.isdisjoint(_distribution_names(extras["all"]))


def test_published_requirements_are_ranges_not_exact_pins():
    """Exact pins belong in constraints.txt, not in the published metadata."""
    project = _project_metadata()
    extras = project["optional-dependencies"]
    published = list(project["dependencies"])
    for name in (*_BACKEND_EXTRAS, *_EXTRAS_OUTSIDE_ALL, "all"):
        published.extend(extras[name])
    assert [req for req in published if "==" in req] == []


def test_constraints_cover_every_published_requirement():
    lines = (_ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines()
    pinned = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert all("==" in requirement for requirement in pinned)

    project = _project_metadata()
    extras = project["optional-dependencies"]
    required = _distribution_names(project["dependencies"]) | _distribution_names(extras["all"])
    for name in _EXTRAS_OUTSIDE_ALL:
        required |= _distribution_names(extras[name])
    assert required.issubset(_distribution_names(pinned))


def test_version_comes_from_the_git_tag():
    """A hand-written version is a second source of truth that can drift.

    0.3.3 shipped a CITATION.cff still announcing 0.3.2 because the number
    lived in several files at once. setuptools-scm derives it from the tag, so
    a release cannot disagree with the tag it was built from.
    """
    project = _project_metadata()
    assert "version" not in project
    assert "version" in project["dynamic"]


def test_citation_version_matches_the_newest_changelog_entry():
    """CITATION.cff is the one place a version is still written by hand.

    Zenodo reads it when minting the DOI, so a stale value misdescribes the
    archived record rather than merely looking untidy.
    """
    changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    released = re.findall(r"^## (\d+\.\d+\.\d+)$", changelog, flags=re.MULTILINE)
    assert released, "CHANGELOG.md has no released version heading"

    citation = yaml.safe_load((_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert str(citation["version"]) == released[0]


def test_requires_python_has_no_upper_bound():
    """An upper cap makes the package invisible to a newer interpreter.

    It is also permanent: the value is baked into every file uploaded to PyPI,
    so a cap can only be lifted by cutting a new release. Backends that lag a
    Python release cap themselves, and pip then names the backend.
    """
    requires_python = _project_metadata()["requires-python"]
    assert "<" not in requires_python, requires_python
