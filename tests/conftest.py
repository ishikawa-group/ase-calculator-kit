"""Shared pytest fixtures, incl. a tqdm progress bar for the slow matrix.

The bar is visible when output capturing is disabled::

    pytest -s            # watch the single-point matrix progress live
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def _singlepoint_bar(request):
    """A single tqdm bar sized to the number of selected ``slow`` tests."""
    from tqdm import tqdm

    total = sum(
        1 for item in request.session.items if item.get_closest_marker("slow")
    )
    bar = tqdm(total=total, desc="uMLIP CPU single-point", unit="variant")
    yield bar
    bar.close()


@pytest.fixture
def singlepoint_progress(request, _singlepoint_bar):
    """Per-test hook: label the bar with the current variant, then advance it."""
    callspec = getattr(request.node, "callspec", None)
    if callspec is not None:
        _singlepoint_bar.set_postfix_str(callspec.id)
    yield _singlepoint_bar
    _singlepoint_bar.update(1)
