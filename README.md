# ase-calculator-kit

[![PyPI](https://img.shields.io/pypi/v/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![Python](https://img.shields.io/pypi/pyversions/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21807793.svg)](https://doi.org/10.5281/zenodo.21807793)

A thin ASE calculator factory for machine-learning interatomic potentials,
VASP, Quantum ESPRESSO, and molecular PySCF/GPU4PySCF. It selects upstream
calculators, validates inputs, and records calculation conditions.

**MLIP settings are Python keyword arguments. DFT settings use YAML or a
configuration dictionary.** No heavy backend is imported until requested.

## Install

```bash
pip install ase-calculator-kit                 # ASE + PyYAML core
pip install 'ase-calculator-kit[sevennet]'      # choose a backend
pip install 'ase-calculator-kit[orb]'           # OrbMol-v2; Python 3.12
pip install 'ase-calculator-kit[uma]'           # UMA (Hugging Face access)
pip install 'ase-calculator-kit[esen]'          # eSEN OMol25
pip install 'ase-calculator-kit[pyscf]'         # CPU molecular HF/DFT
```

For Linux x86_64 with NVIDIA CUDA 12:

```bash
pip install 'ase-calculator-kit[gpu4pyscf-cuda12x]'
```

`[all]` installs the co-installable MLIP extras and D3. MACE, Orb, and PySCF
extras are separate. MACE requires its own environment because its e3nn pin
conflicts with several other backends:

```bash
python -m venv .venv-mace
.venv-mace/bin/pip install 'ase-calculator-kit[mace]'
```

MACE-POLAR additionally needs graph_longrange; Orb has a documented dm-tree
override on Python 3.13+. Follow [installation instructions](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/getting-started.md).

## Quickstart: molecular MLIP

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
atoms.info.update(charge=0, spin=1)  # spin is MULTIPLICITY: 2S+1
atoms.calc = get_calculator("orb", model="orbmol-v2", device="cpu")
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())            # eV/Angstrom
```

UMA/eSEN OMol, OrbMol, MACE-OMOL-0 and MACE-POLAR require explicit electronic
states. MACE-POLAR also reads `atoms.info["external_field"]` in V/Angstrom.
See [molecular inputs and cache behavior](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/molecular.md).

For a material, the same ASE interface accepts other backends:

```python
from ase.build import bulk

atoms = bulk("Cu", "fcc", a=3.6)
atoms.calc = get_calculator("sevennet", model="7net-omni", modal="mpa")
print(atoms.get_potential_energy())
```

## Quickstart: molecular DFT

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
atoms.info.update(charge=0, spin=1)
calc = get_calculator("pyscf", config={
    "calculator": "pyscf",  # use gpu4pyscf in both places for GPU
    "directory": "runs/water",
    "parameters": {
        "basis": "def2-svp",
        "xc": "pbe",
        "density_fit": True,
        "retain_scf": True,
    },
})
atoms.calc = calc
print(atoms.get_potential_energy())
print(atoms.get_forces())
hessian = calc.get_hessian(atoms)  # (3N, 3N), eV/Angstrom^2
print(calc.metadata["scf"])
calc.reset()  # release retained SCF/log resources
```

YAML `spin` is 2S; `atoms.info["spin"]` is multiplicity. Conflicts raise an
error. Density fitting defaults to false on both CPU and GPU; the GPU examples
explicitly enable it. Density reuse between geometries is opt-in too.

See [PySCF settings, checkpoints and Hessians](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/pyscf.md) and
[complete DFT examples](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/examples/dft/README.md). VASP/QE require explicit
execution commands and QE requires explicit pseudopotential settings.

## Supported backends

| Backend | Models / methods | Details |
|---|---|---|
| `sevennet` | SevenNet, including selectable multi-fidelity modals | [Models](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md) |
| `chgnet` | Original CHGNet | [API](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/api.md) |
| `matgl`, `tensornet` | TensorNet MatPES | [Models](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md) |
| `matgl-chgnet` | Provisional corrected MatGL CHGNet MatPES | [Validation](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/matgl-validation.md) |
| `mattersim` | MatterSim 1M / 5M | [Models](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md) |
| `nequip` | NequIP OAM L / XL | [Models](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md) |
| `orb` | OrbMol-v2 | [Molecules](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/molecular.md) |
| `uma` | UMA with task selection | [Models](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md) |
| `esen` | OMol25 conserving / direct-force checkpoints | [Models](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md) |
| `mace` | MH-1, OMAT/MatPES, OMOL-0, POLAR-1 | [Models](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md) |
| `vasp`, `qe` | ASE VASP / Quantum ESPRESSO | [Examples](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/examples/dft/README.md) |
| `pyscf`, `gpu4pyscf` | Molecular HF/DFT, gradients and Hessians | [Guide](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/pyscf.md) |

Dispersion is model-dependent; the kit rejects double-counting.
See the [training levels and dispersion policy](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/models.md).

## Documentation

Start at the [documentation index](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/README.md).

- [Installation and compatibility](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/getting-started.md)
- [Factory API](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/api.md) and [model selection](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/backends.md)
- [Charge, spin and electric fields](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/molecular.md)
- [D3 settings](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/dispersion.md) and [Apple Silicon](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/devices.md)
- [PySCF/GPU4PySCF](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/pyscf.md)
- [Validation records](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/validation-records.md)
- [Development](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/development.md), [Japanese implementation guide](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/code-guide_ja.md), and [releases](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/releasing.md)
- [Changes by version](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/CHANGELOG.md)

## Citation


If this package contributed to published work, please cite the archived
release. [`CITATION.cff`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/CITATION.cff) holds the machine-readable metadata —
GitHub renders it under "Cite this repository", and Zenodo reads it when
minting the DOI.

```bibtex
@software{ase_calculator_kit,
  title  = {ase-calculator-kit: a unified ASE calculator factory for MLIP and DFT calculators},
  author = {Wakamiya, Taishiro and Ishikawa, Atsushi},
  year   = {2026},
  doi    = {10.5281/zenodo.21807793},
  url    = {https://github.com/ishikawa-group/ase-calculator-kit}
}
```

`10.5281/zenodo.21807793` is the *concept* DOI: it always resolves to the newest
archived version. To cite one specific release instead, use its version DOI from
the [Zenodo record](https://doi.org/10.5281/zenodo.21807793) — 0.3.4 is
`10.5281/zenodo.21807794`.

## License

[MIT](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/LICENSE). Upstream software and model weights retain their own licenses;
consult the backend documentation before selecting a checkpoint.
