"""`docs/models.md` and the dispersion policy table must agree.

`dispersion.py` calls that document its human-readable form and says the two
"MUST be kept in sync". Until now nothing checked that. The table is the only
place a reader learns *why* a model refuses `dispersion=True`, so a drift here
is a chemistry documentation bug: someone reads "allowed — D3 xc=pbe", writes
it into a paper's methods section, and the code did something else.

The parser is deliberately strict about formatting. Column 1 of every row must
name its policy keys in backticks, which is also what makes the table readable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ase_calculator_kit.dispersion import _POLICIES

_MODELS_MD = Path(__file__).resolve().parents[1] / "docs" / "models.md"

#: Bold display name in the table -> backend key used in ``_POLICIES``.
_DISPLAY_TO_BACKEND = {
    "CHGNet": "chgnet",
    "MatterSim": "mattersim",
    "NequIP OAM": "nequip",
    "SevenNet": "sevennet",
    "UMA": "uma",
}


def _parse_table() -> dict[tuple[str, str], tuple[bool, str | None]]:
    """Return ``{(backend, key): (includes_dispersion, d3_xc)}`` from the doc."""
    parsed: dict[tuple[str, str], tuple[bool, str | None]] = {}
    for line in _MODELS_MD.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 5 or cells[0] == "Model / task":
            continue

        model_cell, behavior = cells[0], cells[4]
        display = re.match(r"\*\*(.+?)\*\*", model_cell)
        if display is None or display.group(1) not in _DISPLAY_TO_BACKEND:
            raise AssertionError(f"row does not start with a known model: {line}")
        backend = _DISPLAY_TO_BACKEND[display.group(1)]

        keys = re.findall(r"`([^`]+)`", model_cell)
        assert keys, f"row names no policy key in backticks: {line}"

        includes = "⛔" in behavior
        assert includes or "✅" in behavior, f"row has no ⛔/✅ verdict: {line}"
        xc_match = re.search(r"`xc=([^`]+)`", behavior)
        d3_xc = xc_match.group(1) if xc_match else None

        for key in keys:
            parsed[(backend, key)] = (includes, d3_xc)
    return parsed


def test_every_policy_has_a_documented_row():
    documented = _parse_table()
    missing = sorted(set(_POLICIES) - set(documented))
    assert not missing, (
        f"in dispersion.py but not in docs/models.md: {missing}. "
        "Adding a model/task/modal means adding a table row too."
    )


def test_no_documented_row_without_a_policy():
    documented = _parse_table()
    extra = sorted(set(documented) - set(_POLICIES))
    assert not extra, (
        f"in docs/models.md but not in dispersion.py: {extra}. "
        "The table would promise behavior the code does not implement."
    )


@pytest.mark.parametrize("entry", sorted(_POLICIES), ids=lambda e: f"{e[0]}-{e[1]}")
def test_documented_verdict_matches_the_policy(entry):
    documented = _parse_table()
    if entry not in documented:
        pytest.skip("covered by test_every_policy_has_a_documented_row")

    policy = _POLICIES[entry]
    doc_includes, doc_xc = documented[entry]

    assert doc_includes == policy.includes_dispersion, (
        f"{entry}: docs/models.md says "
        f"{'refused' if doc_includes else 'allowed'} but dispersion.py says "
        f"{'refused' if policy.includes_dispersion else 'allowed'}"
    )
    if not policy.includes_dispersion:
        assert doc_xc == policy.d3_xc, (
            f"{entry}: docs/models.md documents xc={doc_xc!r}, "
            f"dispersion.py applies xc={policy.d3_xc!r}"
        )
