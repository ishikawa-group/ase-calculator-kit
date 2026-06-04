"""Config loading and resolution for DFT calculators."""

from __future__ import annotations

import pytest
import yaml

from ase_calculator_kit.config import (
    deep_merge,
    load_config,
    resolve_calculator_config,
    write_resolved_config_file,
)


def test_load_config_from_yaml_path(tmp_path):
    path = tmp_path / "vasp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "calculator": "vasp",
                "directory": "runs/vasp/default",
                "profile": {"command": "vasp_std"},
                "parameters": {"xc": "PBE", "encut": 520},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_config(path)

    assert loaded["calculator"] == "vasp"
    assert loaded["parameters"]["encut"] == 520


def test_load_config_from_dict_returns_copy():
    config = {"calculator": "vasp", "parameters": {"encut": 520}}

    loaded = load_config(config)
    loaded["parameters"]["encut"] = 600

    assert config["parameters"]["encut"] == 520


def test_deep_merge_preserves_nested_values():
    merged = deep_merge(
        {"parameters": {"xc": "PBE", "encut": 520}, "directory": "old"},
        {"parameters": {"encut": 600}},
    )

    assert merged == {
        "parameters": {"xc": "PBE", "encut": 600},
        "directory": "old",
    }


def test_resolve_calculator_config_applies_overrides():
    resolved = resolve_calculator_config(
        "vasp",
        config={
            "calculator": "vasp",
            "directory": "runs/vasp/default",
            "parameters": {"xc": "PBE", "encut": 520},
        },
        overrides={"directory": "runs/vasp/Cu", "parameters": {"encut": 600}},
    )

    assert resolved["directory"] == "runs/vasp/Cu"
    assert resolved["parameters"] == {"xc": "PBE", "encut": 600}


def test_resolve_calculator_config_rejects_mismatch():
    with pytest.raises(ValueError, match="does not match requested"):
        resolve_calculator_config("vasp", config={"calculator": "qe"})


def test_resolve_calculator_config_accepts_qe_aliases():
    resolved = resolve_calculator_config(
        "espresso",
        config={"calculator": "qe", "profile": {"command": "pw.x"}},
    )

    assert resolved["calculator"] == "qe"


def test_write_resolved_config_file(tmp_path):
    resolved = {
        "calculator": "vasp",
        "directory": str(tmp_path),
        "parameters": {"encut": 520},
    }

    path = write_resolved_config_file(resolved, tmp_path)

    assert path.name == "resolved_calculator_config.yaml"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == resolved
