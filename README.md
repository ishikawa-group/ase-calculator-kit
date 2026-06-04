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

## Install

All four backends are installed by default:

```bash
pip install ase-umlip-kit
```

Requires **Python ≥ 3.12** (driven by MatterSim; fairchem caps at < 3.14).

If you only want a subset, install the package without its heavy deps and pick
extras instead:

```bash
pip install ase-umlip-kit[chgnet]      # just CHGNet
pip install ase-umlip-kit[sevennet]    # just SevenNet
pip install ase-umlip-kit[mattersim]   # just MatterSim
pip install ase-umlip-kit[uma]         # just UMA / fairchem
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

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e .[dev]
.venv/bin/pytest -m "not slow"   # fast: factory + device logic
.venv/bin/pytest                 # full: real CPU single-point across every variant
```

The `slow` tests run a real CPU single-point for each backend and intra-model
variant (CHGNet, every SevenNet `modal`, MatterSim 1M & 5M, every UMA `task`).
Cases that need a model download / Hugging Face access are **skipped** (not
failed) when the weights or credentials are unavailable.

## License

MIT
