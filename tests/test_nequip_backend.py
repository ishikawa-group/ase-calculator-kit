"""NequIP backend construction without downloading OAM model packages."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ase_calculator_kit import get_calculator

_MODEL_IDS = {
    "S": "mir-group/NequIP-OAM-S:0.1",
    "M": "mir-group/NequIP-OAM-M:0.1",
    "L": "mir-group/NequIP-OAM-L:0.1",
    "XL": "mir-group/NequIP-OAM-XL:0.1",
}


def _install_fake_nequip(monkeypatch, seen):
    class FakeNequIPCalculator:
        @classmethod
        def _from_saved_model(cls, model_path, **kwargs):
            seen["model_path"] = model_path
            seen["kwargs"] = kwargs
            return "NEQUIP_CALC"

    nequip = types.ModuleType("nequip")
    integrations = types.ModuleType("nequip.integrations")
    ase_module = types.ModuleType("nequip.integrations.ase")
    ase_module.NequIPCalculator = FakeNequIPCalculator

    monkeypatch.setitem(sys.modules, "nequip", nequip)
    monkeypatch.setitem(sys.modules, "nequip.integrations", integrations)
    monkeypatch.setitem(sys.modules, "nequip.integrations.ase", ase_module)


@pytest.mark.parametrize("model,model_id", _MODEL_IDS.items())
def test_nequip_oam_models_route_to_nequip_net(monkeypatch, model, model_id):
    seen = {}
    _install_fake_nequip(monkeypatch, seen)

    calc = get_calculator("nequip", model=model.lower(), device="cpu")

    assert calc == "NEQUIP_CALC"
    assert seen["model_path"] == f"nequip.net:{model_id}"
    assert seen["kwargs"] == {
        "device": "cpu",
        "chemical_species_to_atom_type_map": True,
        "allow_tf32": False,
        "model_name": "sole_model",
        "compile_mode": "eager",
        "neighborlist_backend": "matscipy",
    }


def test_nequip_local_model_path_is_supported(monkeypatch, tmp_path):
    seen = {}
    _install_fake_nequip(monkeypatch, seen)
    model_path = tmp_path / "local.nequip.zip"

    calc = get_calculator("nequip", model="L", model_path=model_path, device="cpu")

    assert calc == "NEQUIP_CALC"
    assert seen["model_path"] == str(model_path)


def test_nequip_rejects_unknown_oam_model():
    with pytest.raises(ValueError, match="Unknown NequIP OAM model"):
        get_calculator("nequip", model="XXL", device="cpu")


def test_nequip_rejects_mps():
    with pytest.raises(ValueError, match="MPS-validated"):
        get_calculator("nequip", model="S", device="mps")


def test_nequip_accepts_pathlike_model_path(monkeypatch):
    seen = {}
    _install_fake_nequip(monkeypatch, seen)

    get_calculator("nequip", model_path=Path("model.nequip.zip"), device="cpu")

    assert seen["model_path"] == "model.nequip.zip"
