# Getting Started

## Installation

`ase-calculator-kit` has an intentionally lightweight core: it pulls in ASE and PyYAML, and **no neural-network potential backends**, so it does not drag in PyTorch by default.

```bash
pip install ase-calculator-kit
```

Install backend dependencies as optional extras depending on your computational needs:

```bash
# Specific backends
pip install "ase-calculator-kit[sevennet]"
pip install "ase-calculator-kit[chgnet]"
pip install "ase-calculator-kit[matgl]"       # TensorNet and MatGL CHGNet
pip install "ase-calculator-kit[mattersim]"
pip install "ase-calculator-kit[nequip]"
pip install "ase-calculator-kit[orb]"          # Molecular OrbMol (Python 3.12 wheels)
pip install "ase-calculator-kit[uma]"
pip install "ase-calculator-kit[esen]"

# Co-installable NNP suite and DFT-D3 correction
pip install "ase-calculator-kit[all]"

# D3 dispersion support only
pip install "ase-calculator-kit[dispersion]"
```

### Python Version Support

Python 3.12 and newer are supported. Backend compatibility varies:

| Component | 3.12 | 3.13 | 3.14 | Notes |
|---|:---:|:---:|:---:|---|
| Core, `all`, MLIPs (except `orb`) | ✅ | ✅ | ✅ | Standard pip install |
| `pyscf`, `pyscf-dispersion` (Linux/macOS) | ✅ | ✅ | ✅ | Pre-built wheels available on Linux |
| `gpu4pyscf-cuda12x` (Linux x86_64) | ✅ | ✅ | ❌ | Requires CUDA 12.x runtime |
| `orb` | ✅ | ⚠️ | ⚠️ | Pins `dm-tree==0.1.8`; requires build tools on ≥3.13 |

### Dedicated Environments

#### MACE Requires an Isolated Environment
`mace-torch` strictly pins `e3nn==0.4.4`, whereas `sevenn`, `fairchem-core`, `mattersim`, and `nequip` require `e3nn>=0.5`. Because these versions conflict, MACE cannot be co-installed into the `[all]` environment:

```bash
python -m venv .venv-mace
.venv-mace/bin/pip install "ase-calculator-kit[mace]"
```

#### OrbMol on Python 3.13+
To install OrbMol on Python 3.13+, supply build dependencies for `dm-tree`:

```bash
pip install "dm-tree>=0.1.8" "ase-calculator-kit[orb]" --no-build-isolation
```

---

## Basic Usage

### Factory API

Calculators are instantiated using `get_calculator()`:

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")

# Machine-learning potential (e.g. SevenNet)
calc_ml = get_calculator(
    "sevennet",
    config={"model_name": "7net-0", "device": "cpu"}
)
atoms.calc = calc_ml
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# Molecular DFT (e.g. PySCF)
calc_dft = get_calculator(
    "pyscf",
    config={
        "parameters": {
            "basis": "def2-svp",
            "xc": "pbe",
            "charge": 0,
            "spin": 0,
        }
    }
)
atoms.calc = calc_dft
print("DFT Energy (eV):", atoms.get_potential_energy())
```

### Configuration via YAML

You can pass configurations from YAML files or dictionaries:

```python
calc = get_calculator("orb", config="config.yaml")
```

Example `config.yaml`:
```yaml
calculator: orb
model_name: orb-d3-v2
device: cuda
```

### Molecular Electronic State Tracking

For molecular models (`orb`, `pyscf`, `gpu4pyscf`), electronic state changes (`charge`, `spin`, `external_field`) automatically invalidate the ASE cache to prevent returning stale energies:

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

mol = molecule("OH")
calc = get_calculator("orb", config={"model_name": "orb-d3-v2", "device": "cpu"})
mol.calc = calc

# Neutral doublet
mol.info["charge"] = 0
mol.info["spin"] = 1
e_doublet = mol.get_potential_energy()

# Hydroxide anion singlet - cache correctly invalidates and recomputes!
mol.info["charge"] = -1
mol.info["spin"] = 0
e_anion = mol.get_potential_energy()
```
