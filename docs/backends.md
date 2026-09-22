# Backends Reference

This document provides configuration options, supported models, parameter tables, and usage examples for all MLIP and DFT engines supported by `ase-calculator-kit`.

---

## Machine Learning Interatomic Potentials (MLIP)

### Supported MLIP Summary

| Backend Key | Model / Task | Default Model | Dispersion D3 | Molecular States | Isolated Env |
|---|---|---|:---:|:---:|:---:|
| `sevennet` | SevenNet (`7net-0`, `7net-0-mf`, `7net-meta`) | `7net-0` | ⛔ Included | ❌ | No |
| `chgnet` | CHGNet (original pretrained) | `0.3.0` | ⛔ Included | ❌ | No |
| `matgl` / `tensornet` | TensorNet / MatPES CHGNet | `TensorNet-MatPES-PBE-v2025.1-OAM` | Configurable | ❌ | No |
| `mattersim` | MatterSim (`MatterSim-v1.0.0-1M`, `5M`) | `MatterSim-v1.0.0-1M` | ⛔ Included | ❌ | No |
| `nequip` | NequIP OAM | `OAM_NequIP_v1` | ⛔ Included | ❌ | No |
| `orb` | OrbMol (`orb-d3-v2`, `orb-v2`) | `orb-d3-v2` | ✅ Policy check | ✅ `charge`, `spin`, `field` | Python 3.12 |
| `uma` | UMA 1.x / Fairchem | `uma-1p5-ommat` | ⛔ Included | ✅ `charge`, `spin` | No |
| `esen` | eSEN OMol25 | `esen-30m-omol` | ⛔ Included | ❌ | No |
| `mace` | MACE-MP-0 / MACE-OFF / MACE-Polar | `medium` | ✅ Model-dependent | ✅ (Polar model) | **Yes** (e3nn pin) |

---

### SevenNet (`sevennet`)

SevenNet is an equivariant graph neural network potential trained on large MP and OMAT datasets.

```python
from ase_calculator_kit import get_calculator

calc = get_calculator(
    "sevennet",
    config={
        "model_name": "7net-0",  # or "7net-0-mf"
        "device": "cuda",        # or "cpu", "mps"
        "modal": "omnia",       # optional modal selection
    }
)
```

- **Dispersion**: Built-in (DFT-D3 is blocked by default).
- **Periodic Boundary Conditions**: Supports 3D bulk and non-periodic clusters.

---

### CHGNet (`chgnet`) & MatGL (`matgl`)

`chgnet` provides the original Ceder group implementation. `matgl` provides PyTorch Geometric implementations of TensorNet and MatPES-retrained CHGNet models.

```python
# CHGNet original
calc_chg = get_calculator("chgnet", config={"device": "cpu"})

# MatGL TensorNet
calc_tn = get_calculator("tensornet", config={"model_name": "TensorNet-MatPES-PBE-v2025.1-OAM"})
```

---

### MatterSim (`mattersim`)

Microsoft's deep learning potential for materials:

```python
calc = get_calculator("mattersim", config={"model_name": "MatterSim-v1.0.0-1M", "device": "cuda"})
```

---

### NequIP OAM (`nequip`)

OpenCatalyst / Alexandria foundation model:

```python
calc = get_calculator("nequip", config={"model_name": "OAM_NequIP_v1", "device": "cuda"})
```

---

### OrbMol (`orb`)

Orb-models molecular foundation potential (`OrbMol-v2`).

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

mol = molecule("H2O")
mol.info["charge"] = 0
mol.info["spin"] = 0  # 2S (number of unpaired electrons)

calc = get_calculator("orb", config={"model_name": "orb-d3-v2", "device": "cpu"})
mol.calc = calc
energy = mol.get_potential_energy()
```

- **Electronic State Tracking**: Requires `charge` and `spin` in `atoms.info`. Supports `external_field`.
- **Cache Management**: Changes to charge, spin, or electric field invalidate the cache automatically.

---

### UMA & eSEN (`uma`, `esen`)

Fairchem-based foundation models.

```python
# UMA
calc_uma = get_calculator(
    "uma",
    config={
        "model_name": "uma-1p5-ommat",
        "task": "omat",
        "device": "cuda"
    }
)

# eSEN
calc_esen = get_calculator("esen", config={"model_name": "esen-30m-omol", "device": "cuda"})
```

---

### MACE (`mace`)

MACE foundation models must be run in an isolated virtual environment (`e3nn` pin conflict).

```python
from ase_calculator_kit import get_calculator

calc = get_calculator(
    "mace",
    config={
        "model_name": "medium",
        "device": "cuda",
        "default_dtype": "float64"
    }
)
```

---

## DFT Backends

### VASP (`vasp`)

Integrates with ASE's standard VASP calculator via structured parameters and commands.

```python
calc = get_calculator(
    "vasp",
    config={
        "directory": "run_vasp",
        "parameters": {
            "xc": "PBE",
            "encut": 520,
            "kpts": [4, 4, 4],
            "ediff": 1e-6,
        }
    }
)
```

### Quantum ESPRESSO (`qe`, `espresso`, `quantum-espresso`)

PWscf calculator interface:

```python
calc = get_calculator(
    "qe",
    config={
        "directory": "run_qe",
        "parameters": {
            "ecutwfc": 60.0,
            "conv_thr": 1e-8,
            "pseudopotentials": {"Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF"}
        }
    }
)
```

### PySCF & GPU4PySCF (`pyscf`, `gpu4pyscf`)

For full details on PySCF molecular calculations, see **[PySCF & GPU4PySCF Guide](pyscf.md)**.
