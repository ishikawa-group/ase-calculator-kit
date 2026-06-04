# ase-umlip-kit

A thin, unified [ASE](https://wiki.fysik.dtu.dk/ase/) calculator factory for
**universal machine-learning interatomic potentials (uMLIPs)**. Switch between
several published models with a single line — every call returns a standard
`ase.Calculator`, so the rest of your ASE workflow stays unchanged.

Supported backends:

- [CHGNet](https://github.com/CederGroupHub/chgnet)
- [SevenNet](https://github.com/MDIL-SNU/SevenNet)
- [MatterSim](https://github.com/microsoft/mattersim)
- [UMA / fairchem](https://github.com/facebookresearch/fairchem)

### Why no MACE?

MACE is **intentionally excluded**. `mace-torch` requires an `e3nn` version that
conflicts with the `e3nn` pinned by SevenNet (`sevenn`) and UMA (`fairchem-core`).
Because this package installs all backends together in one environment, adding
MACE would break dependency resolution and the "install all four and they just
work" guarantee. If you need MACE, use the dedicated MACE tooling in a separate
environment rather than this kit.

## Install

```bash
pip install ase-umlip-kit
```

This installs all supported backends: CHGNet, SevenNet, MatterSim, and
UMA/fairchem. Requires **Python >=3.12,<3.14** (MatterSim requires ≥ 3.12;
fairchem-core caps at < 3.14).

For a lightweight / custom environment, install without dependencies and manage
the backend packages yourself:

```bash
pip install --no-deps ase-umlip-kit
pip install ase chgnet   # then add only the backends you need
```

UMA checkpoints are gated on Hugging Face — request access to the model and run
`huggingface-cli login` before first use.

## Usage

```python
from ase.build import bulk
from ase_umlip_kit import get_calculator

atoms = bulk("Cu", "fcc", a=3.6)

atoms.calc = get_calculator("chgnet", device="mps")          # Apple Silicon ok
print(atoms.get_potential_energy())

atoms.calc = get_calculator("sevennet", model="7net-omni", modal="mpa")
print(atoms.get_potential_energy())

atoms.calc = get_calculator("mattersim", model="5M")
print(atoms.get_potential_energy())

atoms.calc = get_calculator("uma", model="uma-s-1p2", task="omat")
print(atoms.get_potential_energy())
```

`device` defaults to `"auto"` (CUDA → MPS for CHGNet → CPU). Pass an explicit
`"cuda"`, `"cpu"`, or `"mps"` for reproducible scripts.

### Convenience helpers

```python
from ase_umlip_kit import attach_calculator, available_models

available_models()                       # ['chgnet', 'fairchem', 'mattersim', 'sevennet', 'uma']
attach_calculator(atoms, "uma", task="omat")   # sets atoms.calc, returns atoms
```

`build_calculator` and `utils_uMLIP_calculator` are aliases of `get_calculator`.

### Full ASE workflow

`get_calculator` returns a plain `ase.Calculator`, so everything in ASE works
unchanged — single points, relaxations, MD, EOS, etc.:

```python
from ase.build import bulk
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase_umlip_kit import get_calculator

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = get_calculator("mattersim", model="5M", device="cpu")

# single point
print("E  =", atoms.get_potential_energy(), "eV")
print("F  =", atoms.get_forces())
print("S  =", atoms.get_stress())

# relax atoms + cell
opt = BFGS(FrechetCellFilter(atoms))
opt.run(fmax=0.02, steps=200)
print("E_relaxed =", atoms.get_potential_energy(), "eV")
```

### Runnable example

[`examples/run_all_models.py`](examples/run_all_models.py) runs a CPU single
point with every model/variant and prints the energies, with a tqdm progress
bar (weights download on first use):

```bash
.venv/bin/python examples/run_all_models.py                 # all variants, CPU
.venv/bin/python examples/run_all_models.py --device auto
.venv/bin/python examples/run_all_models.py --only chgnet sevennet
```

## Choosing a model variant

### SevenNet `modal` (for the multi-fidelity `7net-omni` / `7net-mf-ompa`)

| `modal` | Use for |
|---|---|
| `mpa` (default) | General PBE(+U)-level materials |
| `omat24` | Broad / high-force PBE(+U) configurations |
| `matpes_pbe` | PBE without Hubbard U |
| `matpes_r2scan` | r2SCAN-level materials |
| `omol25_low` | Molecular / high-fidelity molecular systems |
| `omol25_high` | High-spin molecular configurations only |

Single-fidelity models such as `7net-0` do not take `modal` — pass `modal=None`.

### UMA `task` (for `uma-s-1p2`)

| `task` | Use for |
|---|---|
| `omat` (default) | Inorganic bulk/materials, stress, cell optimization |
| `omol` | Molecules and polymers |
| `oc20` | Catalyst surfaces and adsorption |
| `oc22` | Oxide catalysis |
| `oc25` | Electrochemistry / solid–liquid interfaces |
| `odac` | MOFs and direct air capture |
| `omc` | Molecular crystals |

For molecular tasks (`omol`), set `atoms.info["charge"]` and
`atoms.info["spin"]` before computing.

### MatterSim checkpoint

| `model` | Notes |
|---|---|
| `1M` (default) | Fast screening |
| `5M` | More accurate; keep fixed across a campaign |

### CHGNet device

CHGNet supports `device="mps"` on Apple Silicon in addition to `"cuda"`,
`"cpu"`, and `"auto"`.

## Dispersion (D3)

Add a Grimme-D3(BJ) van-der-Waals correction on top of any model with
`dispersion=True` (off by default):

```python
atoms.calc = get_calculator("uma", task="omat", dispersion=True)        # auto xc=pbe
atoms.calc = get_calculator("uma", task="oc20", dispersion=True)        # auto xc=rpbe
atoms.calc = get_calculator("chgnet", dispersion=True)                  # auto xc=pbe
atoms.calc = get_calculator("uma", task="odac",
                            dispersion=True, dispersion_xc="pbe")        # override
```

The correction is applied as `SumCalculator([model, TorchDFTD3Calculator(...)])`
via [`torch-dftd`](https://github.com/pfnet-research/torch-dftd) (a core
dependency). The D3 `xc` defaults to the model's training functional; override it
with `dispersion_xc`.

**Double-counting guard.** Some models already include dispersion in their
training functional, so `dispersion=True` is refused for them with a
`DispersionError`:

- **Always refused** (dispersion already baked in): UMA `oc25` (RPBE+D3), UMA
  `omol` and SevenNet `omol25_*` (ωB97M-V nonlocal dispersion).
- **Refused unless you pass an explicit `dispersion_xc`** (training functional
  unverified): UMA `odac`/`omc`, SevenNet `matpes_r2scan`, any unknown checkpoint.

See **[docs/models.md](docs/models.md)** for the full per-model table (training
dataset, DFT level, dispersion status). It is kept in sync with the policy in
`src/ase_umlip_kit/dispersion.py`.

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pytest             # fast tests (factory + device); the default
.venv/bin/pytest -m slow     # real calculator smoke tests (downloads weights)
.venv/bin/pytest -m slow -s  # + live tqdm progress bar for the matrix
```

`pytest` runs only the fast tests by default (`addopts = -m 'not slow'`). The
`slow` tests run a real CPU single-point for each backend and intra-model
variant (CHGNet, every SevenNet `modal`, MatterSim 1M & 5M, every UMA `task`).
A tqdm progress bar tracks the matrix — pass `-s` to see it live (pytest
captures output otherwise). Cases that need a model download / Hugging Face
access are **skipped** (not failed) when the weights or credentials are
unavailable.

## License

MIT
