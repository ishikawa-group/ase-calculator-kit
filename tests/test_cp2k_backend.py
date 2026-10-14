"""CP2K configuration and electronic-state checks without an external binary."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import ase
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.cp2k import CP2K

from ase_calculator_kit import get_calculator


@pytest.fixture
def config(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("BASIS_MOLOPT", "POTENTIAL"):
        (data / name).write_text("test data", encoding="utf-8")
    return {
        "calculator": "cp2k",
        "profile": {"command": "env OMP_NUM_THREADS=4 mpirun -np 2 cp2k_shell.psmp",
                    "data_dir": str(data)},
        "directory": str(tmp_path / "run"),
        "parameters": {"xc": "PBE", "basis_set": "DZVP-MOLOPT-SR-GTH",
                       "pseudo_potential": "GTH-PBE", "cutoff": 600,
                       "charge": 0, "multiplicity": 1},
    }


@pytest.fixture
def fake_cp2k(monkeypatch):
    class FakeCP2K(Calculator):
        default_parameters = deepcopy(CP2K.default_parameters)
        implemented_properties = ["energy", "forces"]

        def __init__(self, *, command, directory, label, **parameters):
            self.kwargs = dict(command=command, directory=directory, label=label, **parameters)
            self.calls = 0
            self.closed = False
            super().__init__(directory=directory, label=label, **parameters)

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results = {"energy": -1.0, "forces": np.zeros((len(self.atoms), 3))}

        def close(self):
            self.closed = True

    monkeypatch.setattr("ase.calculators.cp2k.CP2K", FakeCP2K)
    return FakeCP2K


def test_construction_paths_and_resolved_config(config, fake_cp2k, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config["directory"] = "relative-run"
    config["profile"]["data_dir"] = "data"
    original = deepcopy(config)
    calc = get_calculator("CP2K", config=config, overrides={"parameters": {"cutoff": 800}},
                          write_resolved_config=True)
    assert isinstance(calc, fake_cp2k)
    assert config == original
    assert calc.kwargs["label"] == "cp2k"
    assert calc.directory == str(tmp_path / "relative-run")
    assert calc.label == str(tmp_path / "relative-run" / "cp2k")
    for key, name in (("basis_set_file", "BASIS_MOLOPT"), ("potential_file", "POTENTIAL")):
        assert calc.parameters[key] == str(tmp_path / "data" / name)
    saved = yaml.safe_load((Path(calc.directory) / "resolved_calculator_config.yaml").read_text())
    assert saved["parameters"]["cutoff"] == 800
    assert saved["parameters"]["basis_set_file"] == calc.parameters.basis_set_file
    assert saved["directory"] == calc.directory
    assert saved["profile"]["data_dir"] == str(tmp_path / "data")


@pytest.mark.parametrize("key", ["xc", "basis_set", "pseudo_potential", "cutoff"])
def test_required_keys(config, fake_cp2k, key):
    del config["parameters"][key]
    with pytest.raises(ValueError, match=rf"parameters\.{key}"):
        get_calculator("cp2k", config=config)


def test_none_delegates_to_template(config, fake_cp2k):
    keys = ("xc", "basis_set", "pseudo_potential", "cutoff", "basis_set_file", "potential_file")
    config["parameters"].update(dict.fromkeys(keys))
    config["inp"] = "&FORCE_EVAL\n&DFT\n&END DFT\n&END FORCE_EVAL"
    calc = get_calculator("cp2k", config=config)
    assert all(calc.kwargs[key] is None for key in keys)


@pytest.mark.parametrize("value", ["auto", "AUTO", "Auto", " auto "])
def test_auto_potential_rejected(config, fake_cp2k, value):
    config["parameters"]["pseudo_potential"] = value
    with pytest.raises(ValueError, match="pseudo_potential='auto'"):
        get_calculator("cp2k", config=config)


@pytest.mark.parametrize("bad", ["paramters", "profile", "parameters"])
def test_unknown_keys(config, fake_cp2k, bad):
    if bad == "paramters":
        config[bad] = {}
    else:
        config[bad]["typo"] = 1
    with pytest.raises(ValueError, match="Unknown cp2k"):
        get_calculator("cp2k", config=config, write_resolved_config=True)
    assert not Path(config["directory"]).exists()


@pytest.mark.parametrize("key", ["command", "data_dir"])
@pytest.mark.parametrize("value", [None, "", " ", 1])
def test_profile_required(config, fake_cp2k, key, value):
    config["profile"][key] = value
    with pytest.raises(ValueError, match=f"profile.{key}"):
        get_calculator("cp2k", config=config)


@pytest.mark.parametrize("as_path", [False, True])
def test_input_file_and_string(config, fake_cp2k, tmp_path, as_path):
    template = "&FORCE_EVAL\n&DFT\n&POISSON\nPERIODIC NONE\n&END\n&END\n&END\n"
    path = tmp_path / "template.inp"
    path.write_text(template)
    config["inp"] = path if as_path else str(path)
    calc = get_calculator("cp2k", config=config, write_resolved_config=True)
    assert calc.parameters.inp == template
    saved = yaml.safe_load((Path(calc.directory) / "resolved_calculator_config.yaml").read_text())
    assert saved["parameters"]["inp"] == template
    config["inp"] = template
    assert get_calculator("cp2k", config=config).parameters.inp == template


def test_parameters_input_and_duplicate(config, fake_cp2k):
    config["parameters"]["inp"] = "&FORCE_EVAL\n&END FORCE_EVAL"
    assert get_calculator("cp2k", config=config).parameters.inp == config["parameters"]["inp"]
    config["inp"] = ""
    with pytest.raises(ValueError, match="both at top-level"):
        get_calculator("cp2k", config=config)


def test_missing_or_invalid_template(config, fake_cp2k, tmp_path):
    config["inp"] = str(tmp_path / "missing.inp")
    with pytest.raises(FileNotFoundError):
        get_calculator("cp2k", config=config)
    config["inp"] = None
    with pytest.raises(ValueError, match="inp must"):
        get_calculator("cp2k", config=config)


@pytest.mark.parametrize("key", ["basis_set_file", "potential_file"])
def test_absolute_and_missing_data_files(config, fake_cp2k, tmp_path, key):
    path = tmp_path / "absolute-data"
    path.write_text("data")
    config["parameters"][key] = str(path)
    assert get_calculator("cp2k", config=config).parameters[key] == str(path)
    path.unlink()
    with pytest.raises(ValueError, match="file does not exist"):
        get_calculator("cp2k", config=config)
    config["parameters"][key] = "missing-relative-file"
    with pytest.raises(ValueError, match="file does not exist"):
        get_calculator("cp2k", config=config)


def test_missing_data_directory(config, fake_cp2k):
    config["profile"]["data_dir"] += "/missing"
    with pytest.raises(ValueError, match="not a directory"):
        get_calculator("cp2k", config=config)


@pytest.mark.parametrize("version,allowed", [("3.28.0", False), ("3.29.0", True), ("3.30.0", True)])
def test_version_dependent_command(config, fake_cp2k, monkeypatch, version, allowed):
    monkeypatch.setattr(ase, "__version__", version)
    config["profile"]["command"] = "mpirun -np 2 cp2k.psmp -s"
    if allowed:
        command = get_calculator("cp2k", config=config).kwargs["command"]
        assert command.endswith("exec /bin/sh -c 'mpirun -np 2 cp2k.psmp -s'")
    else:
        with pytest.raises(ValueError, match="contain 'cp2k_shell'"):
            get_calculator("cp2k", config=config)


@pytest.mark.parametrize("key,value", [("charge", True), ("charge", 0.5), ("multiplicity", 0),
                                      ("multiplicity", 1.5), ("multiplicity", False), ("uks", 1)])
def test_invalid_yaml_state(config, fake_cp2k, key, value):
    config["parameters"][key] = value
    with pytest.raises(ValueError, match="integer|multiplicity|Boolean"):
        get_calculator("cp2k", config=config)


def test_open_shell_uks(config, fake_cp2k):
    config["parameters"]["multiplicity"] = np.int64(2)
    config["parameters"]["charge"] = np.int64(1)
    calc = get_calculator("cp2k", config=config)
    assert calc.parameters.uks is True
    assert type(calc.parameters.multiplicity) is int
    atoms = Atoms("H", info={"charge": np.int64(1), "spin": np.int64(2)})
    assert calc.get_potential_energy(atoms) == -1
    config["parameters"]["uks"] = False
    with pytest.raises(ValueError, match="requires uks=True"):
        get_calculator("cp2k", config=config)


@pytest.mark.parametrize("key,value", [("charge", 1), ("spin", 2), ("charge", True),
                                      ("spin", 0), ("spin", 1.5)])
def test_info_only_conflict_before_cached_or_direct_calculation(config, fake_cp2k, key, value):
    calc = get_calculator("cp2k", config=config)
    atoms = Atoms("H2", info={"charge": 0, "spin": 1})
    assert calc.get_potential_energy(atoms) == -1
    atoms.info[key] = value
    with pytest.raises(ValueError, match="mismatch|integer|multiplicity"):
        calc.get_potential_energy(atoms)
    with pytest.raises(ValueError):
        calc.get_forces(atoms)
    with pytest.raises(ValueError):
        calc.get_property("energy", atoms, allow_calculation=False)
    with pytest.raises(ValueError):
        calc.calculate(atoms, system_changes=[])
    assert calc.calls == 1


@pytest.mark.parametrize("key,info", [("charge", {"charge": 0}), ("multiplicity", {"spin": 1})])
@pytest.mark.parametrize("null", [False, True])
def test_info_requires_checkable_yaml_state(config, fake_cp2k, key, info, null):
    if null:
        config["parameters"][key] = None
    else:
        del config["parameters"][key]
    calc = get_calculator("cp2k", config=config)
    with pytest.raises(ValueError, match="requires explicit YAML"):
        calc.get_potential_energy(Atoms("H2", info=info))


def test_resolved_write_failure_closes_shell(config, fake_cp2k, monkeypatch):
    closed = []
    monkeypatch.setattr(fake_cp2k, "close", lambda self: closed.append(True))

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr("ase_calculator_kit.backends.dft.cp2k.write_resolved_config_file", fail)
    with pytest.raises(OSError, match="disk full"):
        get_calculator("cp2k", config=config, write_resolved_config=True)
    assert closed == [True]


def test_real_ase_input_label_and_cached_info_guard(config, monkeypatch):
    """Exercise upstream input writing and shell protocol without starting CP2K."""
    class Shell:
        version = 1.0
        isready = True

        def __init__(self, command, debug):
            self.sent = []
            self.responses = []

        def send(self, line):
            self.sent.append(line)
            if line.startswith("LOAD "):
                self.responses = ["1"]
            elif line == "*END":
                self.responses = ["0"]
            elif line.startswith("GET_E "):
                self.responses = ["-1"]
            elif line.startswith("GET_F "):
                self.responses = ["6", "0 0 0", "0 0 0"]
            elif line.startswith("GET_STRESS "):
                self.responses = ["0 0 0 0 0 0 0 0 0"]

        def recv(self):
            return self.responses.pop(0)

        def expect(self, line):
            pass

        def close(self):
            pass

    monkeypatch.setattr("ase.calculators.cp2k.Cp2kShell", Shell)
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]], cell=[10, 10, 10],
                  info={"charge": 0, "spin": 1})
    config["parameters"]["poisson_solver"] = None
    config["inp"] = ("&FORCE_EVAL\n&DFT\n&POISSON\nPERIODIC NONE\n"
                     "POISSON_SOLVER MT\n&END\n&END\n&END")
    with get_calculator("cp2k", config=config) as calc:
        assert calc.get_potential_energy(atoms) == -1
        label = str(Path(config["directory"]) / "cp2k")
        text = Path(label + ".inp").read_text()
        assert "PROJECT cp2k" in text.splitlines()[1:] or "  PROJECT cp2k" in text
        assert "PROJECT " + label not in text
        assert "LOAD " + label + ".inp " + label + ".out" in calc._shell.sent
        assert "MULTIPLICITY 1" in text
        assert "PERIODIC NONE" in text
        assert "POISSON_SOLVER MT" in text
        assert calc.results["free_energy"] == calc.results["energy"]
        assert calc._force_env_id == 1
        count = len(calc._shell.sent)
        atoms.info["charge"] = 1
        with pytest.raises(ValueError, match="mismatch"):
            calc.get_potential_energy(atoms)
        assert len(calc._shell.sent) == count
    assert calc._shell is None


def test_examples_generate_ase_inputs_without_shell(config, monkeypatch):
    class Shell:
        version = 7.0

        def __init__(self, command, debug):
            pass

        def close(self):
            pass

    monkeypatch.setattr("ase.calculators.cp2k.Cp2kShell", Shell)
    examples = Path(__file__).resolve().parents[1] / "examples/dft"
    for filename, pbc, periodic in (("cp2k_pbe_h2.yaml", False, "NONE"),
                                     ("cp2k_pbe_slab.yaml", (True, True, False), "XY")):
        with get_calculator("cp2k", config=examples / filename, overrides={
            "profile": config["profile"], "directory": config["directory"],
        }) as calc:
            calc.atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]], cell=[12, 12, 12], pbc=pbc)
            text = calc._generate_input()
            assert text.count("PERIODIC " + periodic) == 2  # CELL and POISSON
            assert "POISSON_SOLVER MT" in text
            assert "BASIS_SET DZVP-MOLOPT-SR-GTH" in text
            assert "POTENTIAL GTH-PBE" in text
            assert text.count("CUTOFF [") == 1  # one MGRID cutoff, from YAML or template
            assert text.count("XC_FUNCTIONAL") == 2  # opening and closing section


def test_long_directory_keeps_project_short_and_runs_inside_directory(config, monkeypatch, tmp_path):
    """CP2K aborts on string keywords over 80 characters (TSUBAME H2 run, 2026-10-14)."""
    started = {}

    class Shell:
        version = 1.0
        isready = True

        def __init__(self, command, debug):
            started["command"] = command

        def close(self):
            pass

    monkeypatch.setattr("ase.calculators.cp2k.Cp2kShell", Shell)
    directory = tmp_path / ("x" * 60) / "a run with spaces"
    config["directory"] = str(directory)
    calc = get_calculator("cp2k", config=config)
    assert directory.is_dir()
    assert started["command"].startswith(f"cd '{directory}' && exec /bin/sh -c ")
    assert "cp2k_shell.psmp" in started["command"]
    calc.atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, .74]], cell=[10, 10, 10])
    lines = [line.strip() for line in calc._generate_input().splitlines()]
    assert "PROJECT cp2k" in lines
    assert len(str(directory)) > 80
