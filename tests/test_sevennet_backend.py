"""SevenNet backend construction without loading real checkpoints."""

from __future__ import annotations

import sys
import types

import pytest

from ase_calculator_kit import get_calculator, get_dispersion_policy
from ase_calculator_kit.errors import MissingDependencyError


def test_sevennet_backend_allows_mps(monkeypatch):
    seen = {}

    class FakeSevenNetCalculator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)

    calc = get_calculator("sevennet", modal="mpa", device="mps")

    assert isinstance(calc, FakeSevenNetCalculator)
    assert seen["kwargs"]["device"] == "mps"


@pytest.mark.parametrize("model", ["7net-omni", "7net-omni-i8", "7net-omni-i12"])
def test_every_omni_capacity_is_selectable_with_a_modal(monkeypatch, model):
    """i8 and i12 are the same recipe at larger capacity, so `modal` still applies."""
    seen = {}

    class FakeSevenNetCalculator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)

    get_calculator("sevennet", model=model, modal="matpes_r2scan", device="cpu")

    assert seen["kwargs"]["model"] == model
    assert seen["kwargs"]["modal"] == "matpes_r2scan"


def _install_fake_sevenn(monkeypatch, seen: dict):
    class FakeSevenNetCalculator:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator
    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)
    return FakeSevenNetCalculator


@pytest.mark.parametrize(
    "model,expected_modal",
    [
        ("7net-omni", "mpa"),
        ("7net-omni-i12", "mpa"),
        ("7net-mf-ompa", "mpa"),
        ("sevennet-omni", "mpa"),
        ("7net-omat", None),
        ("7net-0", None),
        ("7net-l3i5", None),
        ("sevennet-omat", None),
        ("7net-some-future-model", None),
    ],
)
def test_modal_auto_matches_the_checkpoint(monkeypatch, model, expected_modal):
    """Single-fidelity models take no modal; 7net-omat is the one asked for."""
    seen: dict = {}
    _install_fake_sevenn(monkeypatch, seen)

    get_calculator("sevennet", model=model, device="cpu")

    assert seen["kwargs"].get("modal") == expected_modal


def test_explicit_modal_on_a_single_fidelity_model_is_refused(monkeypatch):
    """sevenn only warns and drops it — and the dropped modal used to pick D3.

    `model="7net-0", modal="matpes_r2scan"` silently applied r2SCAN D3
    parameters to a PBE model before this was an error.
    """
    seen: dict = {}
    _install_fake_sevenn(monkeypatch, seen)

    with pytest.raises(ValueError, match="single-fidelity"):
        get_calculator("sevennet", model="7net-0", modal="matpes_r2scan", device="cpu")
    with pytest.raises(ValueError, match="single-fidelity"):
        get_calculator("sevennet", model="7net-omat", modal="mpa", device="cpu")
    assert seen == {}, "the model must not be loaded to reject the modal"


def test_omat_model_gets_its_own_dispersion_row(monkeypatch):
    """7net-omat is OMat24, not the generic single-fidelity MPtrj default."""
    seen: dict = {}
    d3_kwargs: dict = {}
    _install_fake_sevenn(monkeypatch, seen)

    class FakeD3:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            d3_kwargs.update(kwargs)

    torch_dftd = types.ModuleType("torch_dftd")
    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeD3
    monkeypatch.setitem(sys.modules, "torch_dftd", torch_dftd)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    get_calculator("sevennet", model="7net-omat", device="cpu", dispersion=True)

    assert d3_kwargs["xc"] == "pbe"
    assert get_dispersion_policy("sevennet", "7net-omat").reference_level == "PBE(+U)"


def test_molecular_modal_is_forwarded_without_charge_or_spin(monkeypatch):
    """sevenn has no charge/spin input: the modal is the only molecular handle.

    Guards against "helpfully" inventing a charge=/spin= keyword here — it would
    have to be dropped on the floor, which is worse than not offering it.
    """
    seen = {}

    class FakeSevenNetCalculator:
        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator

    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)

    get_calculator("sevennet", modal="omol25_high", device="cpu")

    assert seen["kwargs"]["modal"] == "omol25_high"
    assert "charge" not in seen["kwargs"]
    assert "spin" not in seen["kwargs"]


def _install_fake_flash_sevenn(monkeypatch, seen: dict, *, fused: bool):
    """A fake sevenn whose model does, or does not, carry FlashTP layers."""

    class IrrepsScatterGatterFusedConvolution:
        pass

    class IrrepsConvolution:
        pass

    class FakeModel:
        def modules(self):
            yield self
            yield IrrepsScatterGatterFusedConvolution() if fused else IrrepsConvolution()

    class FakeSevenNetCalculator:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            seen["kwargs"] = kwargs
            self.model = FakeModel()

    sevenn = types.ModuleType("sevenn")
    calculator = types.ModuleType("sevenn.calculator")
    calculator.SevenNetCalculator = FakeSevenNetCalculator
    monkeypatch.setitem(sys.modules, "sevenn", sevenn)
    monkeypatch.setitem(sys.modules, "sevenn.calculator", calculator)
    monkeypatch.setitem(sys.modules, "flashTP_e3nn", types.ModuleType("flashTP_e3nn"))
    return FakeSevenNetCalculator


def test_flash_is_forwarded_and_verified(monkeypatch):
    seen: dict = {}
    fake = _install_fake_flash_sevenn(monkeypatch, seen, fused=True)

    calc = get_calculator(
        "sevennet", modal="matpes_pbe", device="cuda", enable_flash=True
    )

    assert isinstance(calc, fake)
    assert seen["kwargs"]["enable_flash"] is True
    assert seen["kwargs"]["enable_cueq"] is False


def test_flash_survives_the_d3_wrapper(monkeypatch):
    """dispersion=True returns a SumCalculator around the FlashTP model."""
    seen: dict = {}
    fake = _install_fake_flash_sevenn(monkeypatch, seen, fused=True)

    class FakeD3:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            pass

    torch_dftd = types.ModuleType("torch_dftd")
    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeD3
    monkeypatch.setitem(sys.modules, "torch_dftd", torch_dftd)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    calc = get_calculator(
        "sevennet", modal="matpes_pbe", device="cuda",
        enable_flash=True, dispersion=True,
    )

    assert isinstance(calc.mixer.calcs[0], fake)
    assert isinstance(calc.mixer.calcs[1], FakeD3)
    assert seen["kwargs"]["enable_flash"] is True


def test_flash_fallback_inside_sevenn_is_an_error(monkeypatch):
    """sevenn only warns when it cannot apply FlashTP; that must not pass."""
    seen: dict = {}
    _install_fake_flash_sevenn(monkeypatch, seen, fused=False)

    with pytest.raises(RuntimeError, match="fell back to e3nn"):
        get_calculator("sevennet", modal="matpes_pbe", device="cuda", enable_flash=True)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_flash_needs_cuda(monkeypatch, device):
    seen: dict = {}
    _install_fake_flash_sevenn(monkeypatch, seen, fused=True)

    with pytest.raises(ValueError, match="CUDA device"):
        get_calculator("sevennet", modal="matpes_pbe", device=device, enable_flash=True)
    assert seen == {}, "the model must not be loaded to reject the request"


@pytest.mark.parametrize("other", ["enable_cueq", "enable_oeq"])
def test_flash_excludes_other_accelerators(monkeypatch, other):
    seen: dict = {}
    _install_fake_flash_sevenn(monkeypatch, seen, fused=True)

    with pytest.raises(ValueError, match="exactly one"):
        get_calculator(
            "sevennet", modal="matpes_pbe", device="cuda",
            enable_flash=True, **{other: True},
        )
    assert seen == {}


def test_missing_flashtp_names_the_install_command(monkeypatch):
    seen: dict = {}
    _install_fake_flash_sevenn(monkeypatch, seen, fused=True)
    monkeypatch.setitem(sys.modules, "flashTP_e3nn", None)

    with pytest.raises(MissingDependencyError, match="SNU-ARC/flashTP"):
        get_calculator("sevennet", modal="matpes_pbe", device="cuda", enable_flash=True)
    assert seen == {}
