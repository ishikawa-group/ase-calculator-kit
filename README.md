# ase-calculator-kit

[![PyPI](https://img.shields.io/pypi/v/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![Python](https://img.shields.io/pypi/pyversions/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21807793.svg)](https://doi.org/10.5281/zenodo.21807793)

A thin, unified [ASE](https://wiki.fysik.dtu.dk/ase/) calculator factory for foundation machine-learning interatomic potentials (MLIPs) and molecular DFT engines. Every call returns a standard `ase.Calculator`, keeping the rest of your ASE simulation workflow intact.

---

## Key Features

- **Unified Factory API**: Seamlessly instantiate calculators via `get_calculator(name, config=...)` using dictionaries or YAML files.
- **Accurate State & Cache Management**: Automatically detects changes to molecular electronic states (`charge`, `spin`) and external electric fields (`external_field`), preventing stale cache reuse across spin/charge switches.
- **Comprehensive Foundation MLIP Suite**: Supports SevenNet, CHGNet, MatGL (TensorNet & MatPES), MatterSim, NequIP OAM, OrbMol, UMA, eSEN, and MACE.
- **Automated Dispersion Protection**: Enforces model-specific dispersion policies to prevent double-counting empirical DFT-D3/D4 corrections.
- **Production Molecular DFT (PySCF & GPU4PySCF)**: Energy, analytical forces, Cartesian Hessians, density reuse (`reuse_density`), CDIIS/Newton solvers, atomic checkpointing, and continuum solvation (PCM, SMD).

---

## Installation

The core library is intentionally lightweight and does not install PyTorch or heavy dependencies by default:

```bash
pip install ase-calculator-kit
```

Install backend dependencies as optional extras depending on your workflow:

```bash
# Specific backends
pip install "ase-calculator-kit[sevennet]"
pip install "ase-calculator-kit[chgnet]"
pip install "ase-calculator-kit[matgl]"       # TensorNet and MatGL CHGNet
pip install "ase-calculator-kit[mattersim]"
pip install "ase-calculator-kit[nequip]"
pip install "ase-calculator-kit[orb]"          # Molecular OrbMol (Python 3.12)
pip install "ase-calculator-kit[uma]"
pip install "ase-calculator-kit[esen]"

# Co-installable MLIP suite and DFT-D3 correction
pip install "ase-calculator-kit[all]"

# Molecular DFT backends
pip install "ase-calculator-kit[pyscf,pyscf-dispersion]"      # CPU
pip install "ase-calculator-kit[gpu4pyscf-cuda12x]"           # GPU (CUDA 12)
```

> [!WARNING]
> **MACE requires an isolated virtual environment**: `mace-torch` pins `e3nn==0.4.4`, whereas other modern MLIPs require `e3nn>=0.5`. Install MACE in its own virtual environment (`pip install "ase-calculator-kit[mace]"`). See [Getting Started](docs/getting-started.md).

---

## Quickstart

### 1. Machine Learning Potential (OrbMol / SevenNet)

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

mol = molecule("H2O")
mol.info["charge"] = 0
mol.info["spin"] = 0  # 2S (singlet)

calc = get_calculator("orb", config={"model_name": "orb-d3-v2", "device": "cpu"})
mol.calc = calc

print("Energy (eV):", mol.get_potential_energy())
print("Forces (eV/Å):\n", mol.get_forces())
```

### 2. Molecular DFT (PySCF / GPU4PySCF)

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
calc = get_calculator(
    "pyscf",  # or "gpu4pyscf"
    config={
        "parameters": {
            "basis": "def2-svp",
            "xc": "pbe",
            "disp": "d3bj",
            "charge": 0,
            "spin": 0,
        }
    }
)
atoms.calc = calc

print("DFT Energy (eV):", atoms.get_potential_energy())
hessian = calc.get_hessian(atoms)  # Shape (3N, 3N) in eV/Å^2
```

---

## Supported Backends

| Backend Key | Model / Engine | Target Domain | Key Capabilities | Documentation |
|---|---|---|---|:---:|
| `sevennet` | SevenNet (`7net-0`, `7net-meta`) | Materials / Bulk | Fast bulk screening, zero-shot MD | [Guide](docs/backends.md#sevennet-sevennet) |
| `chgnet` | CHGNet (original) | Materials / Batteries | Charge-informed crystal stability | [Guide](docs/backends.md#chgnet-chgnet--matgl-matgl) |
| `matgl` / `tensornet` | MatGL (TensorNet, MatPES) | Materials / Molecules | Tensor embeddings, D3-compatible | [Guide](docs/backends.md#chgnet-chgnet--matgl-matgl) |
| `mattersim` | MatterSim | Materials / Crystals | High-precision multi-property models | [Guide](docs/backends.md#mattersim-mattersim) |
| `nequip` | NequIP OAM | Catalysis / Surfaces | Strict E(3)-equivariance | [Guide](docs/backends.md#nequip-oam-nequip) |
| `orb` | OrbMol (`orb-d3-v2`) | Molecules / Radicals | Electronic state & electric field tracking | [Guide](docs/backends.md#orbmol-orb) |
| `uma` | UMA 1.x / Fairchem | Catalysis / Surfaces | Multi-domain foundation model | [Guide](docs/backends.md#uma--esen-uma-esen) |
| `esen` | eSEN OMol25 | Molecules | Conserving/direct-force checkpoints | [Guide](docs/backends.md#uma--esen-uma-esen) |
| `mace` | MACE-MP-0 / MACE-OFF | Materials / Molecules | Higher-order message passing | [Guide](docs/backends.md#mace-mace) |
| `vasp` | VASP | Solid-State DFT | Standard periodic plane-wave DFT | [Guide](docs/backends.md#vasp-vasp) |
| `qe` | Quantum ESPRESSO | Solid-State DFT | Open-source plane-wave DFT | [Guide](docs/backends.md#quantum-espresso-qe-espresso-quantum-espresso) |
| `pyscf` / `gpu4pyscf` | PySCF / GPU4PySCF | Molecular DFT & HF | Hessians, CDIIS, density reuse, PCM/SMD | [Guide](docs/pyscf.md) |

---

## Documentation

For full guides and references, please visit the **[Documentation Portal](docs/README.md)**:

- **[Getting Started](docs/getting-started.md)**: Installation, virtual environments, and configuration.
- **[Backends Reference](docs/backends.md)**: Detailed configuration and parameters for all MLIP/DFT engines.
- **[PySCF & GPU4PySCF Guide](docs/pyscf.md)**: Complete guide to molecular DFT, Hessian calculations, and SCF control.
- **[Dispersion Policies](docs/models.md)**: DFT-D3/D4 dispersion compatibility across foundation MLIPs.
- **[Validation Records](docs/validation-records.md)**: Hardware benchmarks and CPU/GPU parity results (TSUBAME4 H100).
- **[Developer & Release Guide](docs/releasing.md)**: Instructions for development and releases.
- **[Architecture Guide (Japanese)](docs/code-guide_ja.md)**: Implementation design and lifecycle details.

---

## Citation

If you use `ase-calculator-kit` in your research, please cite:

```bibtex
@software{ase_calculator_kit,
  author = {Taishiro Wakamiya and Naoki Ishikawa},
  title = {ase-calculator-kit: A unified ASE calculator factory for MLIPs and DFT engines},
  url = {https://github.com/ishikawa-group/ase-calculator-kit},
  year = {2026},
  doi = {10.5281/zenodo.21807793}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
