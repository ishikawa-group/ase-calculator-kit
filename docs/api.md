# Factory API and examples

MLIP settings are keyword arguments; only DFT uses YAML or configuration dictionaries.

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

atoms.calc = get_calculator("uma", model="uma-s-1p2p1", task="omat")
print(atoms.get_potential_energy())
```

OrbMol-v2 is a molecular model, so it takes a molecule and needs the system's
total charge and spin multiplicity:

```python
from ase.build import molecule

mol = molecule("H2O")
mol.info["charge"] = 0
mol.info["spin"] = 1
mol.calc = get_calculator("orb")
print(mol.get_potential_energy())
```

In the separate MACE environment, the same call shape applies:

```python
atoms.calc = get_calculator("mace", model="mh-1", head="omat_pbe")
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
| `tensornet` | MLIP (MatGL PyG) | — |
| `matgl-chgnet` | MLIP (MatGL PyG; provisional gradient correction) | — |
| `mattersim` | MLIP | — |
| `nequip` | MLIP | — |
| `orb` | MLIP (OrbMol-v2; Python 3.12) | — |
| `mace` | MLIP (separate environment) | — |
| `uma` | MLIP | `fairchem` |
| `esen` | MLIP (OMol25) | — |
| `vasp` | DFT | — |
| `qe` | DFT | `espresso`, `quantum-espresso` |
| `pyscf` | Molecular HF/DFT (CPU) | — |
| `gpu4pyscf` | Molecular HF/DFT (CUDA) | — |

An unknown name raises `ValueError` listing the valid names.

### MLIP keyword arguments

All MLIP backends accept `device=` (`"auto"` by default; see
[Apple Silicon (MPS) support](devices.md)), `dispersion=False`,
`dispersion_xc=None`, `dispersion_damping=None`, `dispersion_cutoff=None`,
`dispersion_cutoff_smoothing=None` (see [Dispersion](dispersion.md)), and forward
any extra keywords to the underlying calculator.

| `name` | Backend-specific keywords (defaults) |
|---|---|
| `sevennet` | `model="7net-omni"`, `modal="auto"`, `enable_cueq=False`, `enable_flash=False` |
| `chgnet` | `model=None` (bundled default), `checkpoint=None` (path to a `.pth`) |
| `tensornet` | `model="matpes-pbe"` (or `"matpes-r2scan"`), `revision=None` (Hugging Face commit) |
| `matgl-chgnet` | same model selectors; `revision=None` selects the frozen benchmark revision |
| `mattersim` | `model="1M"` (or `"5M"`), `load_path=None` |
| `nequip` | `model="L"` (`S`/`M`/`L`/`XL`), `model_path=None`, `compile_mode="eager"`, `neighborlist_backend="matscipy"`, `allow_tf32=False` |
| `orb` | `model="orbmol-v2"`, `precision="float32-high"`, `compile=None` |
| `mace` | `model="mh-1"`, `head="auto"`, `default_dtype="float64"`, `accelerator="auto"` |
| `uma` | `model="uma-s-1p2p1"`, `task="omat"`, `inference_settings="default"` |

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

available_mlip_models()     # ['chgnet', 'fairchem', 'mace', 'matgl-chgnet', 'mattersim', 'nequip', 'orb', 'sevennet', 'tensornet', 'uma']
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
| `DispersionError` | `ValueError` | `dispersion=True` is not allowed for that model (see [Dispersion](dispersion.md)) |
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

DFT YAML examples live in [`examples/dft`](https://github.com/ishikawa-group/ase-calculator-kit/tree/main/examples/dft).
