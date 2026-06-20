# ase-calculator-kit

A thin, unified [ASE](https://wiki.fysik.dtu.dk/ase/) calculator factory for
machine-learning interatomic potentials and external DFT calculators. Every call
returns a standard `ase.Calculator`, so the rest of your ASE workflow stays
unchanged.

Supported MLIP backends:

- [CHGNet](https://github.com/CederGroupHub/chgnet)
- [SevenNet](https://github.com/MDIL-SNU/SevenNet)
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

This installs ASE, PyYAML, and all supported MLIP backends. It requires
**Python >=3.12,<3.14** because MatterSim requires Python >=3.12 and
fairchem-core caps at <3.14.

For a lightweight / custom environment, install without dependencies and manage
the backend packages yourself:

```bash
pip install --no-deps ase-calculator-kit
pip install ase pyyaml chgnet
```

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

atoms.calc = get_calculator("chgnet", device="mps")
print(atoms.get_potential_energy())

atoms.calc = get_calculator("sevennet", model="7net-omni", modal="mpa")
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

For reproducibility, VASP configs must explicitly specify `profile.command`;
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

## Public Helpers

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

available_mlip_models()
available_dft_calculators()
available_calculators()
available_models()
attach_calculator(atoms, "uma", task="omat")
```

## Examples

Run a CPU single point with every MLIP model/variant:

```bash
.venv/bin/python examples/run_all_models.py
.venv/bin/python examples/run_all_models.py --device auto
.venv/bin/python examples/run_all_models.py --only chgnet sevennet nequip
```

Create DFT calculator objects from YAML without running VASP/QE:

```bash
.venv/bin/python examples/dft/create_dft_calculator_from_config.py vasp \
  examples/dft/vasp_pbe_static.yaml
```

DFT YAML examples live in [`examples/dft`](examples/dft).

## Apple Silicon (MPS) support

Every MLIP backend was run on a single point (`bulk("Cu")`) with `device="mps"`
on an Apple Silicon Mac (arm64, PyTorch 2.8, MPS available). Results:

| Backend | `device="mps"` | Notes |
|---|---|---|
| CHGNet | ✅ supported | validated locally |
| SevenNet | ✅ supported | validated locally (`7net-omni`) |
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

NequIP OAM supports `device="cpu"` and `device="cuda"`. `device="mps"` is not
supported: Apple Silicon testing fails with `Cannot convert a MPS Tensor to
float64` because the packaged OAM models use float64 buffers and PyTorch MPS has
no float64. See the [Apple Silicon (MPS) support](#apple-silicon-mps-support)
matrix above.

### MatterSim `device`

MatterSim supports `device="cpu"`, `device="cuda"`, `device="mps"`, and
`device="auto"`. Apple Silicon testing confirmed `MatterSimCalculator(device="mps")`
computes energy and forces for a small bulk structure.

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

UMA supports `device="cpu"` and `device="cuda"` only; `fairchem-core` asserts
`device in {"cpu", "cuda"}`, so `device="mps"` is rejected. See the
[Apple Silicon (MPS) support](#apple-silicon-mps-support) matrix above.

## Dispersion

Add a Grimme-D3(BJ) correction on top of MLIP models with `dispersion=True`:

```python
atoms.calc = get_calculator("uma", task="omat", dispersion=True)
atoms.calc = get_calculator("uma", task="oc20", dispersion=True)
atoms.calc = get_calculator("chgnet", dispersion=True)
atoms.calc = get_calculator("uma", task="odac", dispersion=True, dispersion_xc="pbe")
```

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
.venv/bin/pip install -e .[dev]
.venv/bin/pytest
.venv/bin/pytest -m slow
.venv/bin/pytest -m slow -s
```

`pytest` runs only the fast tests by default. Slow tests run real MLIP CPU
single-point calculations and may download model weights.

## License

MIT
