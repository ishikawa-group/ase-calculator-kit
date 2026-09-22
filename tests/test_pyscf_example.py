"""Input failures must leave a retryable path or a useful failure record."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys

import pytest


def test_preflight_and_setup_failures(monkeypatch, tmp_path):
    main = runpy.run_path(str(Path(__file__).parents[1] / "examples/dft/run_pyscf.py"))["main"]
    config = tmp_path / "input.yaml"
    config.write_text("calculator: pyscf\nparameters: {basis: sto-3g}\n")
    xyz, output = tmp_path / "input.xyz", tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run_pyscf.py", str(xyz), "--config", str(config),
                                     "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        main()
    assert not output.exists()
    xyz.write_text("1\n\nH 0 0 0\n")
    config.write_text("calculator: pyscf\nparameters: {basis: sto-3g, conv_tol: .nan}\n")
    def fail(*args, **kwargs):
        raise ValueError("invalid backend setup")
    monkeypatch.setitem(main.__globals__, "get_calculator", fail)
    with pytest.raises(ValueError, match="invalid backend setup"):
        main()
    report = json.loads((output / "results.json").read_text())
    assert report["config"]["parameters"]["conv_tol"] == "nan"
    assert report["success"] is False and "invalid backend setup" in report["error"]
    with pytest.raises(FileExistsError):
        main()
    assert json.loads((output / "results.json").read_text()) == report
