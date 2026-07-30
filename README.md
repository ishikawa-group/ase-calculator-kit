# ase-calculator-kit

A thin, unified [ASE](https://wiki.fysik.dtu.dk/ase/) calculator factory for
machine-learning interatomic potentials and external DFT calculators. Every call
returns a standard `ase.Calculator`, so the rest of your ASE workflow stays
unchanged.

Supported MLIP backends:

- [SevenNet](https://github.com/MDIL-SNU/SevenNet) (installed by default)
- [CHGNet](https://github.com/CederGroupHub/chgnet)
- [MatterSim](https://github.com/microsoft/mattersim)
- [NequIP OAM](https://www.nequip.net/)
- [UMA / fairchem](https://github.com/facebookresearch/fairchem)

Supported DFT backends:

- VASP
- Quantum ESPRESSO (`qe`, `espresso`, `quantum-espresso`)

## Install

```bash
pip install ase-calculator-kit
```

The default installation stays small: ASE, PyYAML, and **SevenNet as the default
NNP backend**. Every other backend is an explicit extra, so install only what
your workflow needs:

```bash
# One extra backend
pip install "ase-calculator-kit[chgnet]"
pip install "ase-calculator-kit[mattersim]"
pip install "ase-calculator-kit[nequip]"
pip install "ase-calculator-kit[uma]"

# Several selected backends
pip install "ase-calculator-kit[chgnet,mattersim]"

# Every supported NNP backend and the optional D3 correction
pip install "ase-calculator-kit[all]"

# D3 correction without installing every NNP backend
pip install "ase-calculator-kit[dispersion]"
```

Python `>=3.12,<3.14` is supported. Missing backend packages are reported only
when that calculator is requested, with the matching extra to install.

Use this import for new code:

```python
from ase_calculator_kit import get_calculator
```

## Usage

MLIP calculators keep the lightweight keyword API:

```python
from ase.build import bulk
from ase_calculator_kit import get_calculator

atoms = bulk("Cu", "fcc", a=3.6)

atoms.calc = get_calculator("sevennet", model="7net-omni", modal="mpa")
print(atoms.get_potential_energy())

atoms.calc = get_calculator("chgnet", device="mps")
print(atoms.get_potential_energy())

atoms.calc = get_calculator("mattersim", model="5M")
print(atoms.get_potential_energy())

atoms.calc = get_calculator("nequip", model="L")
print(atoms.get_potential_energy())

atoms.calc = get_calculator("uma", model="uma-s-1p2", task="omat")
print(atoms.get_potential_energy())
```

DFT calculators are config-only:

```python
from ase_calculator_kit import get_calculator

atoms.calc = get_calculator("vasp", config="examples/dft/vasp_pbe_static.yaml")
atoms.calc = get_calculator("qe", config="examples/dft/qe_pbe_static.yaml")
```

For VASP and QE, arbitrary keyword arguments are intentionally rejected to keep
calculation conditions explicit and reproducible:

```python
get_calculator("vasp", encut=520)  # TypeError
```

For reproducibility, VASP configs must explicitly specify `profile.command`
(QE additionally requires `profile.pseudo_dir` and `pseudopotentials`);
environment-variable-only execution is intentionally not used by this wrapper.

Use `overrides=` for small dynamic changes:

```python
atoms.calc = get_calculator(
    "vasp",
    config="examples/dft/vasp_pbe_static.yaml",
    overrides={"directory": "runs/vasp/Cu_001"},
)
```

Write the final merged config for auditability:

```python
atoms.calc = get_calculator(
    "qe",
    config="examples/dft/qe_pbe_static.yaml",
    overrides={"directory": "runs/qe/Cu_001"},
    write_resolved_config=True,
)
```

## API Reference

### Calculator names

`get_calculator(name, **kwargs)` takes one of these names (case-insensitive);
`available_calculators()` returns the same list at runtime.

| `name` | Kind | Aliases |
|---|---|---|
| `sevennet` | MLIP | — |
| `chgnet` | MLIP | — |
| `mattersim` | MLIP | — |
| `nequip` | MLIP | — |
| `uma` | MLIP | `fairchem` |
| `vasp` | DFT | — |
| `qe` | DFT | `espresso`, `quantum-espresso` |

An unknown name raises `ValueError` listing the valid names.

### MLIP keyword arguments

All MLIP backends accept `device=` (`"auto"` by default; see
[Apple Silicon (MPS) support](#apple-silicon-mps-support)), `dispersion=False`,
`dispersion_xc=None`, and forward any extra keywords to the underlying
calculator.

| `name` | Backend-specific keywords (defaults) |
|---|---|
| `sevennet` | `model="7net-omni"`, `modal="mpa"`, `enable_cueq=False`, `enable_flash=False` |
| `chgnet` | `model=None` (bundled default), `checkpoint=None` (path to a `.pth`) |
| `mattersim` | `model="1M"` (or `"5M"`), `load_path=None` |
| `nequip` | `model="L"` (`S`/`M`/`L`/`XL`), `model_path=None`, `compile_mode="eager"`, `neighborlist_backend="matscipy"`, `allow_tf32=False` |
| `uma` | `model="uma-s-1p2"`, `task="omat"` |

### DFT keyword arguments

DFT backends accept **only** these three; anything else raises `TypeError`.

| Keyword | Default | Meaning |
|---|---|---|
| `config` | *required* | YAML path or `dict` of calculation conditions |
| `overrides` | `None` | `dict` deep-merged over `config` |
| `write_resolved_config` | `False` | Write the merged config into the run directory |

### Public helpers

```python
from ase_calculator_kit import (
    attach_calculator,
    available_calculators,
    available_dft_calculators,
    available_mlip_models,
    available_models,
    get_dft_calculator,
    get_mlip_calculator,
    resolve_calculator_config,
)

available_mlip_models()     # ['chgnet', 'fairchem', 'mattersim', 'nequip', 'sevennet', 'uma']
available_dft_calculators() # ['espresso', 'qe', 'quantum-espresso', 'vasp']
available_calculators()     # both of the above; available_models() is an alias
attach_calculator(atoms, "uma", task="omat")  # sets atoms.calc, returns atoms
```

### Exceptions

```python
from ase_calculator_kit import CalculatorKitError, DispersionError, MissingDependencyError
```

| Exception | Also a | Raised when |
|---|---|---|
| `CalculatorKitError` | `Exception` | Base class for everything below |
| `MissingDependencyError` | `ImportError` | The backend package is not installed; the message names the extra to install |
| `DispersionError` | `ValueError` | `dispersion=True` is not allowed for that model (see [Dispersion](#dispersion)) |
| `ValueError` | — | Unknown calculator name, unsupported `device`, or an incomplete DFT config |
| `TypeError` | — | A DFT backend was given a keyword other than the three above, or `config=` was omitted |

## Examples

Run a CPU single point with every MLIP model/variant:

```bash
python examples/run_all_models.py
python examples/run_all_models.py --device auto
python examples/run_all_models.py --only chgnet sevennet nequip
```

Create DFT calculator objects from YAML without running VASP/QE:

```bash
python examples/dft/create_dft_calculator_from_config.py vasp \
  examples/dft/vasp_pbe_static.yaml
```

DFT YAML examples live in [`examples/dft`](examples/dft).

## Apple Silicon (MPS) support

Every MLIP backend was run on a single point (`bulk("Cu")`) with `device="mps"`
on an Apple Silicon Mac (arm64, PyTorch 2.8, MPS available). Results:

| Backend | `device="mps"` | Notes |
|---|---|---|
| SevenNet | ✅ supported | validated locally (`7net-omni`) |
| CHGNet | ✅ supported | validated locally |
| MatterSim | ✅ supported | validated locally |
| NequIP OAM | ❌ not supported | PyTorch MPS lacks float64; the packaged OAM models use float64 buffers |
| UMA / fairchem | ❌ not supported | `fairchem-core` asserts `device in {"cpu", "cuda"}` |

For the MPS-supported backends, `device="auto"` resolves to `mps` on Apple
Silicon when no CUDA device is present. NequIP and UMA accept only `"cpu"` /
`"cuda"`; passing `device="mps"` raises a clear `ValueError`, and `device="auto"`
falls back to `cpu`.

## Choosing an MLIP Variant

### SevenNet `modal`

| `modal` | Use for |
|---|---|
| `mpa` (default) | General PBE(+U)-level materials |
| `omat24` | Broad / high-force PBE(+U) configurations |
| `matpes_pbe` | PBE without Hubbard U |
| `matpes_r2scan` | r2SCAN-level materials |
| `omol25_low` | Molecular / high-fidelity molecular systems |
| `omol25_high` | High-spin molecular configurations only |

Single-fidelity models such as `7net-0` do not take `modal`; pass `modal=None`.

### NequIP OAM `model`

| `model` | Use for |
|---|---|
| `S` | Smallest OAM model for quick checks |
| `M` | Medium OAM model |
| `L` (default) | Recommended general OAM model for inorganic solids |
| `XL` | Largest OAM model when higher capacity is worth the cost |

NequIP OAM models are loaded through NequIP's `nequip.net:` loader and cached by
NequIP. To avoid a download, pass `model_path="path/to/model.nequip.zip"`.

### MatterSim `model`

`1M` (default) is for fast screening, `5M` is more accurate. Keep the checkpoint
fixed across a campaign.

### UMA `task`

| `task` | Use for |
|---|---|
| `omat` (default) | Inorganic bulk/materials, stress, cell optimization |
| `omol` | Molecules and polymers |
| `oc20` | Catalyst surfaces and adsorption |
| `oc22` | Oxide catalysis |
| `oc25` | Electrochemistry / solid-liquid interfaces |
| `odac` | MOFs and direct air capture |
| `omc` | Molecular crystals |

For molecular tasks (`omol`), set `atoms.info["charge"]` and
`atoms.info["spin"]` before computing.

## Dispersion

Add a Grimme-D3(BJ) correction on top of MLIP models with `dispersion=True`:

```python
atoms.calc = get_calculator("uma", task="omat", dispersion=True)
atoms.calc = get_calculator("uma", task="oc20", dispersion=True)
atoms.calc = get_calculator("chgnet", dispersion=True)
atoms.calc = get_calculator("uma", task="odac", dispersion=True, dispersion_xc="pbe")
```

With `dispersion=True` the returned object is an ASE
`SumCalculator([backend_calculator, d3_calculator])`, not the backend calculator
itself — it satisfies the same `ase.Calculator` interface, but do not rely on
backend-specific attributes or `isinstance` checks against the backend class.

Some models already include dispersion in their training functional, so
`dispersion=True` is refused for them with `DispersionError`. See
[`docs/models.md`](docs/models.md) for the full per-model table.

## Why no MACE?

MACE is intentionally excluded. `mace-torch` requires an `e3nn` version that
conflicts with the `e3nn` pinned by SevenNet (`sevenn`) and UMA
(`fairchem-core`). If you need MACE, use dedicated MACE tooling in a separate
environment.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

`pytest` runs only the fast tests by default. Slow tests
(`pytest -m slow`) run real MLIP CPU single-point calculations and may download
model weights; install `.[dev,all]` first so every backend is importable.

## Further Reading

- [`docs/models.md`](docs/models.md) — per-model dispersion policy and training
  functionals.
- [`docs/code-guide_ja.md`](docs/code-guide_ja.md) — 実装の化学的な判断と
  モジュールの責務（日本語）.
- [`AGENTS.md`](AGENTS.md) — repository map, invariants, and conventions for
  AI coding agents (Claude, GPT, and others).
- [`CHANGELOG.md`](CHANGELOG.md) — release history.

## License

MIT
