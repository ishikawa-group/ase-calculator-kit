"""UMA / fairchem backend guard behavior (no real checkpoints loaded)."""

from __future__ import annotations

import sys
import types

import pytest

from ase_calculator_kit import get_calculator


def _install_fake_fairchem(monkeypatch, seen: dict):
    class FakeFAIRChemCalculator:
        implemented_properties = ["energy"]

        def __init__(self, predictor, **kwargs):
            seen["predictor"] = predictor
            seen["kwargs"] = kwargs

    def fake_get_predict_unit(model_name, device=None, **kwargs):
        seen["model"] = model_name
        seen["device"] = device
        seen["predict_unit_kwargs"] = kwargs
        return object()

    pretrained_mlip = types.ModuleType("fairchem.core.pretrained_mlip")
    pretrained_mlip.get_predict_unit = fake_get_predict_unit

    core = types.ModuleType("fairchem.core")
    core.FAIRChemCalculator = FakeFAIRChemCalculator
    core.pretrained_mlip = pretrained_mlip

    fairchem = types.ModuleType("fairchem")
    fairchem.core = core

    monkeypatch.setitem(sys.modules, "fairchem", fairchem)
    monkeypatch.setitem(sys.modules, "fairchem.core", core)
    return FakeFAIRChemCalculator


def test_uma_rejects_mps():
    # fairchem-core only supports cpu/cuda, so the kit refuses device="mps"
    # before importing fairchem.
    with pytest.raises(ValueError, match="MPS-validated"):
        get_calculator("uma", device="mps")


def test_default_checkpoint_is_uma_s_1p2p1(monkeypatch):
    """The documented default, and the reason the extra needs fairchem >=2.22.

    `uma-s-1p2p1` entered fairchem-core's registry in 2.22.0; an older install
    answers the name with a KeyError listing what it does have. Pinning the
    string here means a silent revert to an earlier checkpoint cannot pass.
    """
    seen: dict = {}
    fake = _install_fake_fairchem(monkeypatch, seen)

    calc = get_calculator("uma", device="cpu")

    assert isinstance(calc, fake)
    assert seen["model"] == "uma-s-1p2p1"
    assert seen["device"] == "cpu"
    assert seen["kwargs"] == {"task_name": "omat"}
    assert seen["predict_unit_kwargs"] == {"inference_settings": "default"}


def test_an_earlier_checkpoint_stays_selectable(monkeypatch):
    seen: dict = {}
    _install_fake_fairchem(monkeypatch, seen)

    get_calculator("uma", device="cpu", model="uma-s-1p2", task="oc20")

    assert seen["model"] == "uma-s-1p2"
    assert seen["kwargs"] == {"task_name": "oc20"}


@pytest.mark.parametrize("name", ["default", "turbo", "batch", "traineval"])
def test_every_inference_preset_reaches_the_predict_unit(monkeypatch, name):
    """The preset belongs to `get_predict_unit`, not to the ASE calculator.

    It used to be unreachable for exactly that reason: `**kwargs` went to
    `FAIRChemCalculator`, which takes only `predict_unit`, `task_name` and a
    deprecated `seed`, so `inference_settings=` came back as a TypeError.
    """
    seen: dict = {}
    _install_fake_fairchem(monkeypatch, seen)

    get_calculator("uma", device="cpu", inference_settings=name)

    assert seen["predict_unit_kwargs"] == {"inference_settings": name}
    assert "inference_settings" not in seen["kwargs"]


def test_an_inference_settings_object_passes_through(monkeypatch):
    """Anything the four names do not cover goes through untouched."""
    seen: dict = {}
    _install_fake_fairchem(monkeypatch, seen)
    settings = object()   # stands in for fairchem's InferenceSettings

    get_calculator("uma", device="cpu", inference_settings=settings)

    assert seen["predict_unit_kwargs"]["inference_settings"] is settings


def test_a_misspelled_preset_is_refused_before_the_download(monkeypatch):
    """fairchem checks the name with a bare `assert`, which `python -O` strips."""
    seen: dict = {}
    _install_fake_fairchem(monkeypatch, seen)

    with pytest.raises(ValueError, match="'default', 'turbo', 'batch', 'traineval'"):
        get_calculator("uma", device="cpu", inference_settings="trubo")
    assert "model" not in seen
