"""DFT backend calculator construction without running external binaries."""

from __future__ import annotations

from ase_calculator_kit import get_calculator


def test_vasp_backend_constructs_ase_calculator(monkeypatch, tmp_path):
    seen = {}

    class FakeVasp:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    monkeypatch.setattr("ase.calculators.vasp.Vasp", FakeVasp)

    calc = get_calculator(
        "vasp",
        config={
            "calculator": "vasp",
            "profile": {"command": "mpirun -np 32 vasp_std", "txt": "out.log"},
            "directory": str(tmp_path / "vasp"),
            "parameters": {"xc": "PBE", "encut": 520},
        },
    )

    assert isinstance(calc, FakeVasp)
    assert seen["kwargs"] == {
        "command": "mpirun -np 32 vasp_std",
        "directory": str(tmp_path / "vasp"),
        "txt": "out.log",
        "xc": "PBE",
        "encut": 520,
    }


def test_vasp_backend_writes_resolved_config(monkeypatch, tmp_path):
    class FakeVasp:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("ase.calculators.vasp.Vasp", FakeVasp)

    run_dir = tmp_path / "vasp"
    get_calculator(
        "vasp",
        config={
            "calculator": "vasp",
            "profile": {"command": "vasp_std"},
            "directory": str(run_dir),
            "parameters": {"encut": 520},
        },
        overrides={"parameters": {"encut": 600}},
        write_resolved_config=True,
    )

    assert (run_dir / "resolved_calculator_config.yaml").exists()


def test_qe_backend_constructs_ase_calculator(monkeypatch, tmp_path):
    seen = {}

    class FakeProfile:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeEspresso:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    monkeypatch.setattr("ase.calculators.espresso.EspressoProfile", FakeProfile)
    monkeypatch.setattr("ase.calculators.espresso.Espresso", FakeEspresso)

    calc = get_calculator(
        "qe",
        config={
            "calculator": "qe",
            "profile": {
                "command": "mpirun -np 32 pw.x",
                "pseudo_dir": "/path/to/pseudos",
            },
            "directory": str(tmp_path / "qe"),
            "pseudopotentials": {"Cu": "Cu.UPF"},
            "parameters": {
                "kpts": [4, 4, 4],
                "input_data": {"system": {"ecutwfc": 60}},
            },
        },
    )

    assert isinstance(calc, FakeEspresso)
    assert seen["kwargs"]["directory"] == str(tmp_path / "qe")
    assert seen["kwargs"]["pseudopotentials"] == {"Cu": "Cu.UPF"}
    assert seen["kwargs"]["kpts"] == [4, 4, 4]
    assert seen["kwargs"]["input_data"] == {"system": {"ecutwfc": 60}}
    assert seen["kwargs"]["profile"].kwargs == {
        "command": "mpirun -np 32 pw.x",
        "pseudo_dir": "/path/to/pseudos",
    }


def test_qe_backend_requires_pseudo_settings():
    config = {
        "calculator": "qe",
        "profile": {"command": "pw.x"},
        "pseudopotentials": {"Cu": "Cu.UPF"},
    }

    try:
        get_calculator("qe", config=config)
    except ValueError as exc:
        assert "profile.pseudo_dir" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
