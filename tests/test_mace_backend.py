"""MACE backend construction without installing mace-torch or loading weights.

MACE cannot share an environment with the other backends (e3nn 0.4.4 vs >=0.5),
so these tests inject a fake ``mace.calculators`` module exactly as the other
backend tests do. They are the only coverage that runs in CI, where mace-torch
is not installed at all.
"""

from __future__ import annotations

import sys
import types

import pytest

from ase_calculator_kit import DispersionError, get_calculator
from ase_calculator_kit.backends.mlip.mace import MH1_HEADS
from ase_calculator_kit.dispersion import _POLICIES
from ase_calculator_kit.errors import MissingDependencyError


class FakeMACECalculator:
    implemented_properties = ["energy"]

    def __init__(self, *, energy=-1.0, fails=False, **kwargs):
        self.kwargs = kwargs
        # MACECalculator exposes the checkpoint's heads and elements under
        # these names; `accelerator="auto"` builds its probe cell from z_table.
        self.available_heads = list(MH1_HEADS)
        self.z_table = types.SimpleNamespace(zs=[1, 8, 29])
        self._energy = energy
        self._fails = fails

    def get_potential_energy(self, atoms=None, **_):
        if self._fails:
            raise RuntimeError(
                "CUDA error: cudaErrorNoKernelImageForDevice: no kernel image "
                "is available for execution on the device"
            )
        return self._energy


def _install_fake_mace(
    monkeypatch,
    seen: dict,
    *,
    available_heads=None,
    plain_energy=-1.0,
    cueq_energy=None,
    cueq_fails=False,
    cuequivariance_installed=False,
):
    calls = seen.setdefault("calls", [])

    def fake_mace_mp(**kwargs):
        calls.append(kwargs)
        seen["kwargs"] = kwargs
        accelerated = kwargs.get("enable_cueq", False)
        energy = plain_energy
        if accelerated and cueq_energy is not None:
            energy = cueq_energy
        calc = FakeMACECalculator(
            energy=energy, fails=accelerated and cueq_fails, **kwargs
        )
        if available_heads is not None:
            calc.available_heads = list(available_heads)
        return calc

    mace = types.ModuleType("mace")
    calculators = types.ModuleType("mace.calculators")
    calculators.mace_mp = fake_mace_mp
    monkeypatch.setitem(sys.modules, "mace", mace)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_torch",
        types.ModuleType("cuequivariance_torch") if cuequivariance_installed else None,
    )


def test_defaults_are_mh1_omat_pbe_float64(monkeypatch):
    """The documented default: MACE-MH-1 with its PBE materials head."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    calc = get_calculator("mace", device="cpu")

    assert isinstance(calc, FakeMACECalculator)
    assert seen["kwargs"] == {
        "model": "mh-1",
        "device": "cpu",
        "default_dtype": "float64",
        "head": "omat_pbe",
    }


def test_extra_keywords_reach_mace(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    get_calculator(
        "mace", device="cpu", head="matpes_r2scan",
        default_dtype="float32", enable_cueq=True,
    )

    assert seen["kwargs"]["head"] == "matpes_r2scan"
    assert seen["kwargs"]["default_dtype"] == "float32"
    assert seen["kwargs"]["enable_cueq"] is True


def test_unknown_head_is_refused_before_the_model_is_loaded(monkeypatch):
    """MACE answers an unknown head with a *warning* and the wrong head.

    ``MACECalculator`` logs "Head <x> not found ... defaulting to the last head"
    and returns energies from that head, so a typo produces a plausible number
    at a different level of theory. Catch it here, and before the download.
    """
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    with pytest.raises(ValueError) as exc:
        get_calculator("mace", device="cpu", head="rgd1_b3lyp")

    message = str(exc.value)
    assert "rgd1_b3lyp" in message
    assert "omat_pbe" in message
    assert seen["calls"] == [], "the checkpoint must not be loaded to reject a head"


def test_head_none_lets_mace_pick_a_single_head_checkpoint(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["Default"])

    get_calculator("mace", device="cpu", model="medium-omat-0", head=None)

    assert "head" not in seen["kwargs"]


def test_head_missing_from_an_unlisted_checkpoint_is_refused(monkeypatch):
    """The safety net for checkpoints whose heads this package cannot know."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["a_head", "another"])

    with pytest.raises(ValueError, match="a_head"):
        get_calculator("mace", device="cpu", model="mh-0", head="omat_pbe")


def test_mps_is_refused(monkeypatch):
    """Measured: torch.load(map_location='mps') on MH-1 fails at float64."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    with pytest.raises(ValueError, match="mps"):
        get_calculator("mace", device="mps")


def test_dispersion_is_refused_for_the_molecular_heads(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    for head in ("omol", "spice_wB97M"):
        with pytest.raises(DispersionError, match="double-counting"):
            get_calculator("mace", device="cpu", head=head, dispersion=True)
    assert seen["calls"] == [], "the policy is checked before the model is loaded"


def test_dispersion_wraps_the_pbe_head(monkeypatch):
    seen: dict = {}
    d3_kwargs: dict = {}
    _install_fake_mace(monkeypatch, seen)

    class FakeD3:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            d3_kwargs.update(kwargs)

    torch_dftd = types.ModuleType("torch_dftd")
    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeD3
    monkeypatch.setitem(sys.modules, "torch_dftd", torch_dftd)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    from ase.calculators.mixing import SumCalculator

    calc = get_calculator("mace", device="cpu", dispersion=True)

    assert isinstance(calc, SumCalculator)
    assert d3_kwargs["xc"] == "pbe"
    assert d3_kwargs["damping"] == "bj"


def test_missing_mace_names_the_separate_environment(monkeypatch):
    """The install hint has to mention the second environment, not just the extra."""
    monkeypatch.setitem(sys.modules, "mace", None)
    monkeypatch.setitem(sys.modules, "mace.calculators", None)

    with pytest.raises(MissingDependencyError) as exc:
        get_calculator("mace", device="cpu")

    message = str(exc.value)
    assert "ase-calculator-kit[mace]" in message
    assert "e3nn" in message
    assert "virtual environment" in message


def test_single_head_model_needs_no_head_argument(monkeypatch):
    """MACE-OMAT-0 is the point of head="auto": it carries only "Default"."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["Default"])

    calc = get_calculator("mace", device="cpu", model="medium-omat-0")

    assert "head" not in seen["kwargs"]
    assert seen["kwargs"]["model"] == "medium-omat-0"
    assert calc.kwargs["default_dtype"] == "float64"


def test_multi_head_model_still_defaults_to_omat_pbe(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    get_calculator("mace", device="cpu", model="mh-1")

    assert seen["kwargs"]["head"] == "omat_pbe"


def test_unlisted_model_sends_no_head(monkeypatch):
    """MACE's own error lists the checkpoint's heads; do not pre-empt it."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["Default"])

    get_calculator("mace", device="cpu", model="/some/local/model.model")

    assert "head" not in seen["kwargs"]


def test_explicit_head_still_wins_over_auto(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    get_calculator("mace", device="cpu", model="mh-1", head="matpes_r2scan")

    assert seen["kwargs"]["head"] == "matpes_r2scan"


@pytest.mark.parametrize(
    "model,expected_xc",
    [
        ("medium-omat-0", "pbe"),
        ("small-omat-0", "pbe"),
        ("medium-mpa-0", "pbe"),
        ("mace-matpes-pbe-0", "pbe"),
        ("mace-matpes-r2scan-0", "r2scan"),
    ],
)
def test_single_head_models_key_the_dispersion_policy_by_model(
    monkeypatch, model, expected_xc
):
    """With no head to key on, the checkpoint's own functional has to decide."""
    seen: dict = {}
    d3_kwargs: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["Default"])

    class FakeD3:
        implemented_properties = ["energy"]

        def __init__(self, **kwargs):
            d3_kwargs.update(kwargs)

    torch_dftd = types.ModuleType("torch_dftd")
    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeD3
    monkeypatch.setitem(sys.modules, "torch_dftd", torch_dftd)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    get_calculator("mace", device="cpu", model=model, dispersion=True)

    assert d3_kwargs["xc"] == expected_xc


def test_unknown_model_keeps_dispersion_unverified(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, available_heads=["Default"])

    with pytest.raises(DispersionError, match="not verified"):
        get_calculator(
            "mace", device="cpu", model="/some/local/model.model", dispersion=True
        )


def test_accelerator_auto_does_nothing_without_cuda(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, cuequivariance_installed=True)

    get_calculator("mace", device="cpu")

    assert len(seen["calls"]) == 1
    assert "enable_cueq" not in seen["kwargs"]


def test_accelerator_auto_skips_when_cuequivariance_is_absent(monkeypatch):
    """The common case: CUDA, no cuequivariance. It must cost nothing."""
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen, cuequivariance_installed=False)

    get_calculator("mace", device="cuda")

    assert len(seen["calls"]) == 1
    assert "enable_cueq" not in seen["kwargs"]


def test_accelerator_auto_enables_cueq_when_it_agrees(monkeypatch):
    seen: dict = {}
    _install_fake_mace(
        monkeypatch, seen,
        cuequivariance_installed=True, plain_energy=-200.0, cueq_energy=-200.0,
    )

    calc = get_calculator("mace", device="cuda")

    assert calc.kwargs["enable_cueq"] is True
    assert [c.get("enable_cueq", False) for c in seen["calls"]] == [False, True]


def test_accelerator_auto_falls_back_when_the_gpu_cannot_run_it(monkeypatch):
    """Measured on a V100: it builds, then dies at the first evaluation."""
    seen: dict = {}
    _install_fake_mace(
        monkeypatch, seen, cuequivariance_installed=True, cueq_fails=True
    )

    with pytest.warns(RuntimeWarning, match="did not run on this GPU"):
        calc = get_calculator("mace", device="cuda")

    assert calc.kwargs.get("enable_cueq", False) is False


def test_accelerator_auto_falls_back_when_the_energy_disagrees(monkeypatch):
    """ACEsuit/mace#1298: cuequivariance returns +5500 eV and raises nothing."""
    seen: dict = {}
    _install_fake_mace(
        monkeypatch, seen,
        cuequivariance_installed=True, plain_energy=-200.885, cueq_energy=5499.825,
    )

    with pytest.warns(RuntimeWarning, match="disagrees"):
        calc = get_calculator("mace", device="cuda")

    assert calc.kwargs.get("enable_cueq", False) is False


def test_accelerator_auto_keeps_float32_rounding(monkeypatch):
    """The probe must not mistake float32 noise for a broken kernel."""
    seen: dict = {}
    _install_fake_mace(
        monkeypatch, seen,
        cuequivariance_installed=True, plain_energy=-200.0, cueq_energy=-200.0000004,
    )

    calc = get_calculator("mace", device="cuda", default_dtype="float32")

    assert calc.kwargs["enable_cueq"] is True


def test_explicit_accelerator_skips_the_probe(monkeypatch):
    seen: dict = {}
    _install_fake_mace(
        monkeypatch, seen, cuequivariance_installed=True, cueq_fails=True
    )

    calc = get_calculator("mace", device="cuda", accelerator="cueq")

    # No probe: the caller asked for it, so a failure is theirs to see.
    assert len(seen["calls"]) == 1
    assert calc.kwargs["enable_cueq"] is True


def test_accelerator_oeq_sets_the_openequivariance_flag(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    get_calculator("mace", device="cuda", accelerator="oeq")

    assert seen["kwargs"]["enable_oeq"] is True


def test_accelerator_none_disables_the_probe(monkeypatch):
    seen: dict = {}
    _install_fake_mace(
        monkeypatch, seen, cuequivariance_installed=True, cueq_fails=True
    )

    get_calculator("mace", device="cuda", accelerator="none")

    assert len(seen["calls"]) == 1
    assert "enable_cueq" not in seen["kwargs"]


def test_unknown_accelerator_is_refused(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    with pytest.raises(ValueError, match="Unknown accelerator"):
        get_calculator("mace", device="cuda", accelerator="cuequivariance")
    assert seen["calls"] == []


def test_explicit_enable_flag_wins_over_the_probe(monkeypatch):
    seen: dict = {}
    _install_fake_mace(
        monkeypatch, seen, cuequivariance_installed=True, cueq_fails=True
    )

    calc = get_calculator("mace", device="cuda", enable_cueq=True)

    assert len(seen["calls"]) == 1
    assert calc.kwargs["enable_cueq"] is True


def test_accelerator_and_enable_flag_together_are_refused(monkeypatch):
    seen: dict = {}
    _install_fake_mace(monkeypatch, seen)

    with pytest.raises(ValueError, match="not both"):
        get_calculator("mace", device="cuda", accelerator="cueq", enable_cueq=True)


@pytest.mark.parametrize("head", MH1_HEADS)
def test_every_mh1_head_has_a_dispersion_policy(head):
    """A head with no policy row would be refused as "unverified" at runtime."""
    assert ("mace", head) in _POLICIES
