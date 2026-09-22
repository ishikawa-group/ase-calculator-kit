"""Regression for slow-test exception classification, without loading models."""

from __future__ import annotations

from urllib.error import HTTPError

import pytest
from ase import Atoms

import test_singlepoint_cpu as smoke


def test_model_errors_fail_but_transport_errors_skip(monkeypatch):
    download = RuntimeError("Model download failed")
    download.__cause__ = HTTPError("https://example.org/model", 403, "gated", {}, None)
    incompatible = RuntimeError("Model download failed")
    incompatible.__cause__ = ValueError("invalid model specification")
    for error, expected in (
        (download, pytest.skip.Exception),
        (incompatible, RuntimeError),
        (ValueError("checkpoint architecture is incompatible"), ValueError),
        (RuntimeError("could not resolve token shape"), RuntimeError),
        (FileNotFoundError("missing checkpoint"), FileNotFoundError),
        (HTTPError("https://example.org/model", 404, "missing", {}, None), HTTPError),
        (HTTPError("https://example.org/model", 403, "gated", {}, None), pytest.skip.Exception),
        (ConnectionError("unavailable"), pytest.skip.Exception),
    ):
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(smoke, "get_calculator", fail)
        with pytest.raises(expected):
            smoke.test_cpu_single_point("stub", {}, lambda: Atoms("H"), None)
