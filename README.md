# ase-calculator-kit

[![PyPI](https://img.shields.io/pypi/v/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![Python](https://img.shields.io/pypi/pyversions/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21807793.svg)](https://doi.org/10.5281/zenodo.21807793)

A thin, unified [ASE](https://wiki.fysik.dtu.dk/ase/) calculator factory for
machine-learning interatomic potentials and DFT calculators. Every call
returns a standard `ase.Calculator`, so the rest of your ASE workflow stays
unchanged.

Supported MLIP backends:

- [SevenNet](https://github.com/MDIL-SNU/SevenNet)
- [CHGNet](https://github.com/CederGroupHub/chgnet)
- [MatGL (PyG)](https://github.com/materialyzeai/matgl): TensorNet and provisional CHGNet MatPES potentials
- [MatterSim](https://github.com/microsoft/mattersim)
- [NequIP OAM](https://www.nequip.net/)
- [OrbMol (orb-models)](https://github.com/orbital-materials/orb-models):
  OrbMol-v2, a molecular potential with learnable electrostatics — installs
  from wheels **only on Python 3.12**, see [OrbMol (orb-models)](#orbmol-orb-models)
- [UMA / fairchem](https://github.com/facebookresearch/fairchem)
- eSEN OMol25 (`esen`): conserving and direct-force checkpoints through fairchem
- [MACE](https://github.com/ACEsuit/mace) — **must be installed in a separate
  virtual environment**, see [MACE needs its own environment](#mace-needs-its-own-environment)

Supported DFT backends:

- VASP
- Quantum ESPRESSO (`qe`, `espresso`, `quantum-espresso`)
- Molecular PySCF (`pyscf`) and GPU4PySCF (`gpu4pyscf`): HF/DFT energy and forces,
  VV10, D3/D4, ECP, SMD/PCM; see [configuration](docs/pyscf.md)

## Install

```bash
pip install ase-calculator-kit
```

The default installation is intentionally lightweight: it pulls in ASE and
PyYAML and **no NNP backend**, so it does not drag in torch. Each backend is an
explicit extra — install only what your workflow needs:

```bash
# One backend
pip install "ase-calculator-kit[sevennet]"
pip install "ase-calculator-kit[chgnet]"
pip install "ase-calculator-kit[matgl]"  # TensorNet and provisional MatGL CHGNet
pip install "ase-calculator-kit[mattersim]"
pip install "ase-calculator-kit[nequip]"
pip install "ase-calculator-kit[orb]"        # Python 3.12; see the note below
pip install "ase-calculator-kit[uma]"
pip install "ase-calculator-kit[esen]"        # same fairchem dependency

# Several selected backends
pip install "ase-calculator-kit[chgnet,mattersim]"

# Every co-installable NNP backend and the optional D3 correction
pip install "ase-calculator-kit[all]"

# D3 correction without installing every NNP backend
pip install "ase-calculator-kit[dispersion]"
```

> ⚠️ **MACE is the one exception: it needs a virtual environment of its own.**
> `mace-torch` pins `e3nn==0.4.4`, while `sevenn`, `fairchem-core`, `mattersim`
> and `nequip` all require `e3nn>=0.5`. There is no resolution that satisfies
> both, so `mace` is **not** part of `[all]`, and
> `pip install "ase-calculator-kit[all,mace]"` cannot succeed.
>
> ```bash
> python -m venv .venv-mace
> .venv-mace/bin/pip install "ase-calculator-kit[mace]"
> ```
>
> See [MACE needs its own environment](#mace-needs-its-own-environment).

> ⚠️ **`orb` is also outside `[all]`, for an unrelated reason.** It co-installs
> with every other backend, but `orb-models` pins `dm-tree==0.1.8`, whose newest
> wheels are cp312 — so on Python 3.13 and 3.14 the install stops in a source
> build. Putting it in `[all]` would take `[all]` down with it. The model itself
> runs fine on 3.13; see [OrbMol (orb-models)](#orbmol-orb-models) for the
> one-line override.

Missing backend packages are reported only when that calculator is requested,
with the matching extra to install.

### Python versions

Python 3.12 and newer. The package itself has no upper bound; backend
compatibility is checked separately:

| | 3.12 | 3.13 | 3.14 |
|---|:--:|:--:|:--:|
| Core, `all`, and MLIP extras except `orb` | ✅ | ✅ | ✅ |
| `pyscf`, `pyscf-dispersion` (Linux) | ✅ | ✅ | ✅ |
| `gpu4pyscf-cuda12x` (Linux x86_64) | ✅ | ✅ | ❌ |
| `orb` | ✅ | ⚠️ override | ⚠️ override |

For `orb`, the cause is one line of
upstream metadata rather than the model: `orb-models` pins `dm-tree==0.1.8`,
whose newest wheels are cp312. OrbMol-v2 itself runs on 3.13 — verified against
the 3.12 results, to the last digit — once `dm-tree` is allowed to be newer.
[OrbMol (orb-models)](#orbmol-orb-models) has the command.

The molecular GPU extra uses a validated CuPy version without cp314 wheels.
`uma` (and therefore `all`) used to be
❌ on 3.14, because `fairchem-core` declared `requires-python = ">=3.11,<3.14"`
and pinned `torch~=2.8.0`, which has no cp314 wheels; fairchem-core 2.22.0 —
the floor this release requires — lifted the cap and moved to `torch~=2.13.0`.

A backend that lags a Python release is left to cap itself rather than hidden
behind an environment marker: pip then names the backend in the error, while a
marker would make the install *succeed* and silently leave the backend out.

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
[Apple Silicon (MPS) support](#apple-silicon-mps-support)), `dispersion=False`,
`dispersion_xc=None`, `dispersion_damping=None`, `dispersion_cutoff=None`,
`dispersion_cutoff_smoothing=None` (see [Dispersion](#dispersion)), and forward
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

DFT YAML examples live in [`examples/dft`](https://github.com/ishikawa-group/ase-calculator-kit/tree/main/examples/dft).

## Apple Silicon (MPS) support

Every MLIP backend was run on a single point (`bulk("Cu")`) with `device="mps"`
on Apple Silicon Macs (arm64, MPS available). MatGL models use the environment
recorded in their validation report below. Results:

| Backend | `device="mps"` | Notes |
|---|---|---|
| SevenNet | ✅ supported | validated locally (`7net-omni`) |
| CHGNet | ✅ supported | validated locally |
| MatGL TensorNet | ✅ supported | both MatPES checkpoints; MPS uses float32, D3 runs on CPU; see the [validation record](docs/matgl-validation.md) |
| MatGL CHGNet (provisional) | ✅ supported | both corrected MatPES checkpoints; MPS uses float32, D3 runs on CPU; see the [validation record](docs/matgl-validation.md) |
| MatterSim | ✅ supported | validated locally |
| NequIP OAM | ❌ not supported | PyTorch MPS lacks float64; the packaged OAM models use float64 buffers |
| OrbMol | ❌ not supported | graph construction runs through NVIDIA Warp, which has no Metal backend; the legacy edge methods need float64 or refuse a non-CPU device |
| MACE | ❌ not supported | same float64 problem: loading `mace-mh-1.model` with `map_location="mps"` raises `Cannot convert a MPS Tensor to float64`, with `default_dtype="float32"` as well |
| UMA / fairchem | ❌ not supported | `fairchem-core` asserts `device in {"cpu", "cuda"}` |

For the MPS-supported backends, `device="auto"` resolves to `mps` on Apple
Silicon when no CUDA device is present. NequIP, OrbMol, MACE and UMA accept only
`"cpu"` / `"cuda"`; passing `device="mps"` raises a clear `ValueError`, and
`device="auto"` falls back to `cpu`.

## Choosing an MLIP Variant

### MatGL PyG MatPES models

Install `ase-calculator-kit[matgl]`, or `[matgl,dispersion]` to add D3. The
`[all]` extra includes both. MatGL 4.0.3 or newer (within 4.x) uses PyG only;
DGL is not installed or selected. The native `chgnet` extra remains independent.

```python
atoms.calc = get_calculator("tensornet", model="matpes-pbe", device="mps")
atoms.calc = get_calculator("tensornet", model="matpes-r2scan", dispersion=True)
```

| `name` | Short `model` | Official model name |
|---|---|---|
| `tensornet` | `matpes-pbe` (default) | `TensorNet-PES-MatPES-PBE-2025.2` |
| `tensornet` | `matpes-r2scan` | `TensorNet-PES-MatPES-r2SCAN-2025.2` |
| `matgl-chgnet` | `matpes-pbe` (default) | `CHGNet-PES-MatPES-PBE-2025.2.10` |
| `matgl-chgnet` | `matpes-r2scan` | `CHGNet-PES-MatPES-r2SCAN-2025.2.10` |

The full model names and dated suffixes such as
`model="pes-matpes-pbe-2025.2"` are also accepted, case-insensitively.
An unknown version or a name belonging to a different architecture is rejected
before downloading. Short names always map to the dated names above.
The weights come from the official
[`materialyze` organization](https://huggingface.co/materialyze); pass
`revision="<Hugging Face commit SHA>"` to freeze a specific weight revision.
Upstream has replaced weights under existing dated names, so record the revision
as well as the model name for reproducible comparisons.

**`matgl-chgnet` is a temporary compatibility backend requiring exactly MatGL
4.0.3.** It uses the existing 2025.2.10 weights, with only the `no_grad()` block
around three-body geometry removed on the loaded model instance. These are
**not retrained checkpoints**. This restores consistency between energy and
its force/stress derivatives; it does not guarantee better accuracy for every
property or dataset. The original `chgnet` backend uses the Ceder CHGNet package.

```bash
pip install "ase-calculator-kit[matgl,dispersion]" "matgl==4.0.3"
```

```python
atoms.calc = get_calculator("matgl-chgnet", model="matpes-pbe", device="mps")
atoms.calc = get_calculator("matgl-chgnet", model="matpes-r2scan", dispersion=True)
```

By default, provisional CHGNet weights are frozen to HF revisions
`4ec3c2c323cde07a31ca99d6719cca45919b5e5f` (PBE) and
`4447783f53387df4642d13062cafc9571f1668d2` (r2SCAN); an explicit `revision=`
overrides this choice. The resolved model/revision and correction are recorded
in the underlying PESCalculator's `parameters`. With D3 this is
`calc.mixer.calcs[0].parameters`. Unrecognized MatGL source or versions fail
before downloading instead of receiving an unverified patch.

**Once upstream retrained models are released and validated, a future kit
release will replace this provisional implementation and document the new
weights and behavior.** No automatic weight replacement occurs in this release.
For reproducibility, record the kit/MatGL versions and HF revision. See the
[validation record](docs/matgl-validation.md) and
[upstream fix #835](https://github.com/materialyzeai/matgl/pull/835).
MatGL M3GNet remains deferred.

`dispersion=True` selects D3 parameters `xc="pbe"` or `xc="r2scan"` from the
resolved model. The shared defaults remain BJ damping, 14 Å cutoff and `poly`
smoothing. When the network runs on MPS, the D3 term runs on CPU.

The calculator returns total-cell energy in eV, forces in eV/Å with shape
`(N, 3)`, and stress in eV/Å³ with shape `(6,)`, ordered
`xx, yy, zz, yz, xz, xy`. MatGL's GPa default is explicitly changed through its
own API; incompatible `stress_unit`, `stress_weight` or `use_voigt` overrides
are rejected. TensorNet's float64 buffers are cast to float32 before moving to
MPS. No global PyTorch dtype or graph backend setting is changed.

Both MatGL backends honor each axis of `atoms.pbc`: bulk `[True, True, True]`,
slabs such as `[True, True, False]`, wires and nonperiodic molecules. An input
adapter corrects MatGL 4.0.3's handling of partial PBC, including with D3;
the supplied atoms are not modified. Missing **nonperiodic** cell vectors are
allowed for energy/forces. Stress requires three independent cell vectors;
otherwise ASE raises `PropertyNotImplementedError`. With a vacuum cell, stress
uses the full cell volume, as in ASE. Fully nonperiodic torch-dftd calculations
provide energy/forces only, so stress is unavailable with D3 in that case.
Previously computed partial-PBC MatGL results need recomputation. Full-PBC
graph construction remains unchanged.

See [the validation record](docs/matgl-validation.md) for the common-system
checks against SevenNet-Omni, finite differences and measured CPU/MPS agreement.

### SevenNet `model`

| `model` | Fidelity | Use for |
|---|---|---|
| `7net-omni` (default) | multi | Recommended general model |
| `7net-omni-i8` | multi | Larger capacity: more accurate, slower |
| `7net-omni-i12` | multi | Largest of the family |
| `7net-mf-ompa` | multi | MPtrj + sAlex + OMat24 multi-fidelity |
| `7net-omat` | single | **OMat24 only, PBE(+U)** — SevenNet-omat |
| `7net-l3i5`, `7net-0` | single | Earlier models |

```python
atoms.calc = get_calculator("sevennet", model="7net-omni-i12")  # modal="mpa"
atoms.calc = get_calculator("sevennet", model="7net-omat")      # no modal
```

The Omni family is one training recipe at three capacities, so the `modal` table
below applies unchanged to all three. Keep the model fixed across a campaign —
i8 and i12 are separate models, not refinements of an `7net-omni` number, so
energies are not comparable between them.

**Multi- vs single-fidelity is handled for you.** `modal` defaults to `"auto"`,
which sends `mpa` to a multi-fidelity model and nothing at all to a
single-fidelity one, so `model="7net-omat"` needs no second argument. Passing an
explicit `modal` to a single-fidelity model now raises: sevenn only *warns* and
drops it, and the dropped modal was still being used to pick the D3 functional —
`model="7net-0", modal="matpes_r2scan"` used to apply r2SCAN dispersion
parameters to a PBE model.

### SevenNet `modal`

| `modal` | Use for | Reference level |
|---|---|---|
| `mpa` (default) | General-purpose, including molecules | PBE(+U) |
| `omat24` | Broad / high-force configurations | PBE(+U) |
| `matpes_pbe` | PBE without Hubbard U | PBE |
| `matpes_r2scan` | r2SCAN-level materials | r2SCAN |
| `mp_r2scan` | r2SCAN-level Materials Project data | r2SCAN |
| `oc20` | Catalyst surfaces and adsorption | RPBE |
| `oc22` | Oxide catalysis | PBE(+U) |
| `odac23` | MOFs / direct air capture | PBE-D3 |
| `omol25_low` | **Low-spin** molecular systems | ωB97M-V |
| `omol25_high` | **High-spin** molecular systems only | ωB97M-V |
| `spice` | Drug-like molecules and peptides | ωB97M-D3(BJ) |
| `qcml` | Small molecules, wide element coverage | PBE0 + MBD-NL |
| `pet_mad` | PBEsol-level data | PBEsol |

`omol25_low` and `omol25_high` split OMol25 by **spin state**, not by accuracy —
pick the one matching your system. SevenNet's own guidance is that `mpa` stays
the recommended default even for molecules, organic crystals, and molecular
liquids; choose another task only when you need consistency with a specific
functional or benchmark protocol.

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

### MACE `model` and `head`

| `model` | Heads | Trained on |
|---|---|---|
| `mh-1` (default) | 6, see below | Multi-head cross-learning |
| `medium-omat-0` | single | **OMat24, PBE(+U)** — MACE-OMAT-0 |
| `small-omat-0` | single | OMat24, smaller |
| `medium-mpa-0` | single | MPtrj + sAlex, PBE(+U) |
| `mace-matpes-pbe-0` | single | MatPES, PBE |
| `mace-matpes-r2scan-0` | single | MatPES, r2SCAN |

```python
atoms.calc = get_calculator("mace")                            # mh-1 / omat_pbe
atoms.calc = get_calculator("mace", model="medium-omat-0")     # no head needed
```

`head` defaults to `"auto"`: `omat_pbe` for `mh-1`, and no head at all for the
single-head checkpoints above (they carry one head named `Default`, and handing
them a head name from another model makes MACE quietly compute with the head it
does have). The omat-0 and matpes checkpoints are released under the **ASL**
license, not MIT — MACE prints a notice when it downloads one.

MACE-MH-1 is a single checkpoint with six readout heads. The head *is* the level
of theory, not an accuracy setting: the same structure returns PBE, r2SCAN or
ωB97M energies depending on which one you pick.

| `head` | Use for | Reference level |
|---|---|---|
| `omat_pbe` (default) | Inorganic materials; best cross-domain behaviour | PBE(+U) |
| `mp_pbe_refit_add` | Materials Project trajectories | PBE(+U) |
| `oc20_usemppbe` | Catalyst surfaces and adsorption | PBE (see below) |
| `matpes_r2scan` | r2SCAN-level materials | r2SCAN |
| `omol` | Molecules and organometallics | ωB97M-VV10 |
| `spice_wB97M` | Small to medium organic molecules | ωB97M-D3(BJ) |

```python
atoms.calc = get_calculator("mace", head="matpes_r2scan")
```

Two things worth knowing before trusting a number from this model:

- **An unknown head does not raise upstream.** MACE logs a warning and computes
  with the last head in the file, so a typo returns a plausible energy from the
  wrong level of theory. This package rejects unknown heads with a `ValueError`
  instead, before the checkpoint is downloaded. The head names above are read
  back from the shipped `mace-mh-1.model`; the model card advertises an
  `rgd1_b3lyp` head that the published checkpoint does not carry.
- **`default_dtype` defaults to `"float64"`**, following the MH-1 model card —
  the right choice for geometry optimisation and phonons. Pass
  `default_dtype="float32"` for faster MD.

### OrbMol `model`

`get_calculator("orb")` loads **OrbMol-v2**, and that is the only checkpoint
this backend accepts (`model="orbmol-v2"`, also spelled `"orbmol_v2"`). The rest
of orb-models' registry — the Orb-v3 OMat/MPA models, Orb-v2, OrbMol-v1 — is not
wired up: each is a different reference level and needs its own row in the
dispersion policy, so asking for one raises `ValueError` rather than quietly
computing at a level this package does not document.

OrbMol-v2 continues the Orb-v3 architecture with **learnable electrostatics**: a
latent-charge head predicts per-atom charges constrained to sum to the system's
total charge, and a Coulomb module adds the long-range term — a bare 1/r sum for
isolated systems, particle mesh Ewald for periodic ones. It is trained on OMol25
and OPoly26 at ωB97M-V/def2-TZVPD, the same reference level as UMA's `omol` task
and SevenNet's `omol25_*` modals, and unlike OrbMol-v1 it has seen periodic
systems.

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.calc = get_calculator("orb")
print(atoms.get_potential_energy())
```

| Keyword | Default | Meaning |
|---|---|---|
| `precision` | `"float32-high"` | also `"float32-highest"` (same dtype, exact matmuls) or `"float64"` |
| `compile` | `None` | `None` follows orb-models and compiles; `False` skips it |

- **The first single point is the slow one, twice over.** On a machine that has
  never run orb-models, NVIDIA Warp compiles its neighbor-list kernels once —
  about a minute and a half here, then cached under `~/.cache/warp` and
  unrelated to `compile=`. With those caches warm, measured on CPU:
  `compile=None` costs ~7 s on the first call against ~1.4 s with
  `compile=False`, and both settle at ~0.02 s per call afterwards. Compiling is
  worth it across a trajectory, not in a one-shot script.
- **Energy, forces and stress are all derivatives.** OrbMol-v2 is conservative,
  and orb-models turns the stress derivative on when preparing it for
  inference. `calc.results` also carries `confidence` (the model's own
  uncertainty estimate), `rotational_grad`, and `grad_forces` / `grad_stress`
  under their upstream names.
- **A charged system in a periodic cell is not the isolated one.** The periodic
  Coulomb term is an Ewald sum against a neutralizing background, so acetate⁻
  in a 14 Å box came out 1.45 eV below the same anion with `pbc=False` here.
  That is the Ewald convention doing its job, not a model error — but it means
  `pbc=True` on a charged molecule answers a different question. Neutral
  molecules in the same box moved by under 3 meV.
- `dispersion=True` is refused: ωB97M-V already carries the nonlocal VV10 term.

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

For the molecular task (`omol`), set `atoms.info["charge"]` and
`atoms.info["spin"]` before computing — see
[Molecular systems](#molecular-systems-charge-and-spin) for why this matters.

### UMA `inference_settings`

`inference_settings=` chooses how the predict unit is built, which is where
UMA's speed/precision trade lives. It reaches fairchem's `get_predict_unit()`,
so it must be set when the calculator is created, not afterwards.

| `inference_settings` | TF32 | `merge_mole` | `compile` | Use for |
|---|:--:|:--:|:--:|---|
| `"default"` | ✗ | ✅ | ✅ | MD and relaxation — one system, many steps |
| `"turbo"` | ✅ | ✅ | ✅ | the same, trading a little precision for speed |
| `"batch"` | ✗ | ✗ | ✗ | many different structures |
| `"traineval"` | ✗ | ✗ | ✗ | reproducing fairchem's training/eval numbers |

```python
atoms.calc = get_calculator("uma", task="omat")                          # MD
atoms.calc = get_calculator("uma", task="omat", inference_settings="turbo")
atoms.calc = get_calculator("uma", task="omat", inference_settings="batch")
```

Two things about this are easy to get backwards.

**The default is already the MD fast path.** Since fairchem-core 2.22,
`"default"` merges the MOLE experts *and* compiles the model; `"turbo"` is that
same path with TF32 switched on. So `turbo` is not what turns compilation on.

Measured on an H100 (MIG 4g.47gb), 27-atom fcc Cu, `task="omat"`, twelve frames
of the same system, averaged over the last eight:

| `inference_settings` | first step | steady state | ΔE vs `default` | ΔF rmse |
|---|--:|--:|--:|--:|
| `"default"` | 59.4 s | **13.5 ms** | — | — |
| `"turbo"` | 48.2 s | **12.8 ms** | 3.69 meV (0.137 meV/atom) | 0.0004 eV/Å |
| `"batch"` | 0.38 s | **56.7 ms** | 0.005 meV | 0.000001 eV/Å |

So on this hardware `turbo` bought about 5 %, and cost 0.137 meV/atom. That
error is the same size as the gap between two *different* models — OrbMol-v2 and
UMA differ by 0.08–0.8 meV/atom on small molecules — so turning it on by default
would put a numerical artefact where a model difference is supposed to be. It
stays opt-in here, as it does in fairchem. Turn it on when you are running one
model and want throughput; leave it off when the number is going into a
comparison. On CPU it does nothing at all.

The same table says `merge_mole` + `compile` is numerically honest: `"batch"`,
which uses neither, lands within 0.005 meV of `"default"`. The 4× per-step cost
is what you give up for skipping them.

TF32 is applied inside a context manager around UMA's own forward pass, which
restores `torch.get_float32_matmul_precision()` afterwards — so `turbo` does not
leak reduced precision into other calculators sharing the process.

**`merge_mole` assumes the system never changes.** Composition, task, total
charge and spin must all stay fixed — true of an MD trajectory, false of a loop
over structures. fairchem notices and falls back, logging `The UMA fast path
(merge_mole + compile) is only available for fixed composition, task, charge,
and spin`. Nothing is wrong when you see that, but a merge and a compile were
paid for and thrown away: pass `inference_settings="batch"` when the structures
vary. On CPU that was 18.8 s per single point against 2.6 s; on the H100 above,
0.38 s to the first result against 59.4 s. Which is why
[`examples/run_all_models.py`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/examples/run_all_models.py)
uses `"batch"` for its UMA lines.

Anything the four names do not cover — `max_atoms`, `edge_chunk_size`,
`execution_mode`, a different `base_precision_dtype` — goes through as a
`fairchem.core` `InferenceSettings` object:

```python
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

atoms.calc = get_calculator(
    "uma", task="omat",
    inference_settings=InferenceSettings(merge_mole=True, compile=True, max_atoms=2048),
)
```

A misspelled name raises `ValueError` listing the four, before the checkpoint is
downloaded. fairchem checks it with a bare `assert`, which `python -O` removes.

### eSEN-30M-OMat is not available through this package

`eSEN-30M-OMat` is an OMat24-trained fairchem model, so it looks like it should
sit next to UMA here. It does not, and the reason is structural rather than an
oversight:

- the checkpoint (`esen_30m_omat.pt`, in the gated repo
  [`facebook/OMAT24`](https://huggingface.co/facebook/OMAT24)) is a
  **fairchem-core 1.x** artefact, loaded with `OCPCalculator(checkpoint_path=…)`;
- fairchem-core 2.x — what the `uma` extra installs — ships no eSEN
  architecture, and `pretrained_mlip.available_models` lists only UMA plus the
  OMol25 / OC25 / ODAC eSEN checkpoints, none of them OMat24;
- fairchem-core 1.10 requires `torch~=2.4`, `numpy<2` and Python `<3.13`, so it
  cannot share an environment with the 2.x line, with MACE, or with much else.

Supporting it would mean a third isolated environment, as MACE has. If you need
it today, install `fairchem-core<2` separately and use its own `OCPCalculator`.
For an OMat24-level model inside this package, use
`get_calculator("mace", model="medium-omat-0")` or
`get_calculator("sevennet", model="7net-omat")` — same dataset, same PBE(+U)
reference level.

## Molecular systems (charge and spin)

Molecular models need two inputs that no bulk model does: the **total charge**
of the system and its **spin multiplicity** (`2S+1`). ASE has no standard place
for either, so they are passed through `atoms.info`, and the backends differ in
whether they read them at all.

| Backend | Molecular option | Takes charge / spin? |
|---|---|---|
| `esen` | OMol25 checkpoints | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` |
| `pyscf` / `gpu4pyscf` | Molecular HF/DFT | ✅ `atoms.info` or YAML; mismatches raise |
| `uma` | `task="omol"` | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` |
| `sevennet` | `modal="omol25_low"` / `"omol25_high"` / `"spice"` / `"qcml"` | ❌ not supported by sevenn |
| `mace` | `head="omol"` / `"spice_wB97M"` | ❌ the MH-1 molecular heads are fitted to neutral closed-shell data |
| `mace` | `model="polar-1-s"` / `"polar-1-m"` / `"polar-1-l"` | ✅ `atoms.info["charge"]`, `atoms.info["spin"]`, `atoms.info["external_field"]` |
| `orb` | `model="orbmol-v2"` (the default) | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` — **required**, not defaulted |

### UMA: set both keys explicitly

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
atoms.info["charge"] = 0   # total charge
atoms.info["spin"] = 1     # spin multiplicity, 2S+1 (1 = closed shell)
atoms.calc = get_calculator("uma", task="omol")
print(atoms.get_potential_energy())
```

A hydroxide anion and a neutral radical are the cases that actually bite:

```python
oh_minus = molecule("OH")
oh_minus.info["charge"] = -1   # anion
oh_minus.info["spin"] = 1      # closed shell
oh_minus.calc = get_calculator("uma", task="omol")

oh_radical = molecule("OH")
oh_radical.info["charge"] = 0
oh_radical.info["spin"] = 2    # doublet — one unpaired electron
oh_radical.calc = get_calculator("uma", task="omol")
```

> **Do not rely on the defaults.** fairchem does *not* raise when `charge` or
> `spin` is missing. It logs a warning, writes `charge=0` / `spin=1` into the
> `atoms.info` dict you passed in, and returns a neutral closed-shell result.
> An ion or an open-shell species then comes back **silently wrong**. Set both
> keys on every molecular structure, including the ones you think are obvious.

Both keys are integers. `charge` may range from -100 to 100 and `spin` from 0 to
100; they are read only by the `omol` head, and other UMA tasks ignore them.

### SevenNet: no charge or spin input

sevenn has no charge or spin argument, so the `modal` embedding is the only
handle on the molecular reference data. Charged species and a chosen open-shell
state **cannot be expressed** — `omol25_high` selects a model trained on
high-spin configurations, but it is not a multiplicity you set per structure.
Use `get_calculator("uma", task="omol")` when the charge and spin of the system
matter.

### MACE-MH-1: molecular heads, no charge or spin

MACE-MH-1's `omol` head was fine-tuned on a subsample of OMol25 that is
explicitly **neutral and closed-shell**, and `spice_wB97M` on SPICE-1, which is
likewise neutral. There is no charge or spin input to set, and no head that
covers ions or a chosen open-shell state. Use `get_calculator("uma",
task="omol")`, or MACE-Polar below, when the charge and spin of the system
matter.

### MACE-Polar: charge, spin, and an applied field

`polar-1-s` / `polar-1-m` / `polar-1-l` are MACE's **electrostatics** foundation
models, trained on OMol25 at ωB97M-V. They are the one group of checkpoints
loaded through upstream's `mace_polar` rather than `mace_mp` — the model name is
what switches the loader, so nothing else changes at the call site.

They need one package on top of `mace-torch`. It is not on PyPI, so no extra of
this project can pull it in and you install it yourself, into the MACE
environment:

```bash
.venv-mace/bin/pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0
```

That repository builds a distribution named `graph_longrange` — the module the
checkpoints unpickle. The name mismatch matters at the command line: pip rejects
the `graph_electrostatics @ git+…` spelling as an inconsistent name, so pass the
bare URL as above. Ask for a polar checkpoint without it and
`get_calculator` raises `MissingDependencyError` with this command, rather than
letting MACE fail with `No module named 'graph_longrange'` after the download.

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.01]   # V/Å, along z
atoms.calc = get_calculator("mace", model="polar-1-m")
print(atoms.get_potential_energy())
print(atoms.calc.results["dipole"])               # non-periodic cells only
```

Beyond energy, forces and stress, `calc.results` carries `dipole`, `charges`,
`spins` and an electrostatic energy breakdown (`electrostatic_energy`,
`electron_energy`, `interaction_energy`). **Read them from `calc.results`
directly**: MACE leaves them out of `implemented_properties`, so ASE's own
accessors do not see them and `atoms.get_dipole_moment()` raises
`PropertyNotImplementedError` even though `calc.results["dipole"]` is there.
Partial charges are also **not a uniquely defined quantity** — read them as a
decomposition of the model's electrostatics, not as a measurement.

> **The same silent-default trap as UMA's `omol`.** MACE substitutes
> `charge=0`, `spin=1` and a zero field when the keys are absent, without a
> warning. An ion, a radical, or a field-on calculation then comes back
> **silently** neutral, closed-shell and unpolarised. Set the keys you mean.

`dispersion=True` is refused for all three: OMol25's ωB97M-V reference already
carries the nonlocal VV10 term, so a D3 correction would double-count it.

`polar-1-m` and `polar-1-l` carry 12 Å and 18 Å receptive fields; `polar-1-s` is
the small model. All three live in the MACE environment like every other MACE
checkpoint, and all three are stored in float32 — with this package's
`default_dtype="float64"` default, MACE logs that it is converting them, which
is the conversion and not a failure.

### OrbMol-v2: charge and spin, enforced

OrbMol-v2 reads the same two `atoms.info` keys, with one difference that makes
it the safest of the group: it **raises** when they are missing.

```python
oh_minus = molecule("OH")
oh_minus.info["charge"] = -1
oh_minus.info["spin"] = 1
oh_minus.calc = get_calculator("orb")
print(oh_minus.get_potential_energy())

forgot = molecule("OH")
forgot.calc = get_calculator("orb")
forgot.get_potential_energy()
# ValueError: atoms.info must contain both 'charge' and 'spin'
```

UMA and MACE-Polar substitute a neutral closed-shell system and compute on;
orb-models stops.

The two agree closely where both are defined. On eight small molecules and ions
— neutral, anionic and open-shell, isolated — `get_calculator("orb")` and
`get_calculator("uma", task="omol")` with `uma-s-1p2p1` differed by **0.18 to
5.87 meV in total energy**, on absolute energies of 1.5 to 6.3 keV. Both are
OMol25 ωB97M-V models, so their absolute energy scales are directly comparable.
Force directions were identical (direction cosine 1.0000 throughout) and
components agreed to 0.05 eV/Å, except the O₂ triplet at 0.21 eV/Å — that and
the acetate anion were the two largest disagreements.

They also handle a periodic box differently, and that is a real difference
rather than a tolerance. Put each molecule in a 14 Å cell with `pbc=True` and
UMA returns the isolated energy unchanged to 1e-7 eV: its `omol` head has no
long-range electrostatic term, so nothing reaches past the cutoff. OrbMol-v2's
Coulomb term is an Ewald sum and is not cutoff-limited, so it moves — by under
3 meV for the neutral molecules, and by 1.45 eV for acetate⁻, which the Ewald
convention computes against a neutralizing background. See
[OrbMol `model`](#orbmol-model).

### Non-periodic cells

`ase.build.molecule()` returns `pbc=False` with a zero cell, which UMA accepts.
UMA rejects only two ambiguous cases: a fully periodic structure whose cell is
all zeros, and a partially periodic one (`pbc=[True, True, False]`).

## Dispersion

Add a Grimme-D3(BJ) correction on top of MLIP models with `dispersion=True`:

```python
atoms.calc = get_calculator("uma", task="omat", dispersion=True)
atoms.calc = get_calculator("uma", task="oc20", dispersion=True)
atoms.calc = get_calculator("chgnet", dispersion=True)
atoms.calc = get_calculator("sevennet", modal="pet_mad", dispersion=True)
atoms.calc = get_calculator("mace", head="omat_pbe", dispersion=True)
```

With `dispersion=True` the returned object is an ASE
`SumCalculator([backend_calculator, d3_calculator])`, not the backend calculator
itself — it satisfies the same `ase.Calculator` interface, but do not rely on
backend-specific attributes or `isinstance` checks against the backend class.

Some models already include dispersion in their training functional, so
`dispersion=True` is refused for them with `DispersionError`:

```python
get_calculator("uma", task="omol", dispersion=True)      # DispersionError: ωB97M-V includes VV10
get_calculator("sevennet", modal="spice", dispersion=True)  # DispersionError: SPICE is ωB97M-D3(BJ)
get_calculator("mace", head="omol", dispersion=True)     # DispersionError: ωB97M-VV10
```

**Every molecular task falls in this category** — molecular reference data is
almost always dispersion-corrected, each dataset in its own way (VV10, an
explicit D3(BJ) term, or MBD-NL). That verdict cannot be overridden with
`dispersion_xc=`; remove `dispersion=True` instead. A task this table does not
cover yet is refused by default but *can* be unlocked with an explicit
`dispersion_xc` once you have checked its functional yourself.

### Choosing the damping function

`dispersion_damping=` selects between Becke-Johnson (`"bj"`, the default) and
zero damping (`"zero"`):

```python
atoms.calc = get_calculator(
    "uma", task="oc20", dispersion=True, dispersion_damping="zero"
)
```

The two are separately fitted parameter sets, not a numerical detail. D3 does
not screen a metal's C6 coefficients, so on molecule–metal systems the choice
can move the correction by a factor of two — for RPBE, benzene on Pt(111) picks
up −4.6 eV of dispersion with BJ damping against −2.4 eV with zero damping.

Match the reference dataset when the model has one. OC20 and OC22 carry no
dispersion at all, so either damping is a choice you are making rather than
reproducing; OC25, in contrast, is RPBE + D3 with **zero** damping, which is
why `task="oc25"` refuses an added correction outright.

Note also that RPBE's D3 parameters — both dampings — are absent from Grimme's
published fits and carry no citation in the reference parameter tables, unlike
PBE's. Treat RPBE-D3 numbers on metals as indicative.

### Cutoff and smoothing follow PFP

The D3 term runs at **`cutoff=14.0` Å with `cutoff_smoothing="poly"`** — PFP
v7.0.0+'s settings, not torch-dftd's own defaults of 95 Bohr (50.3 Å) and no
smoothing.

This package exists to compare models against each other, and PFP is one of the
models being compared. Left at torch-dftd's defaults, the dispersion term added
to a SevenNet or MACE energy would be a *different quantity* from the one inside
a PFP energy, on top of the model difference you are trying to measure.
[Matlantis published the validation](https://docs.matlantis.com/atomistic-simulation-tutorial/ja/) for the shorter cutoff: an MAE of
0.0024 eV over the Wellendorff adsorption benchmark — negligible against the
0.01 eV scale those numbers live on — and no change in the 90th percentile of
COD unit-cell-volume error, in exchange for roughly three times the reachable
system size. `"none"` smoothing was a PFP bug fixed in v7.0.0; it leaves the
force discontinuous at the cutoff radius, which is exactly what a relaxation or
MD run walks into.

Both are overridable, so a dataset built on torch-dftd's defaults stays
reproducible:

```python
atoms.calc = get_calculator(
    "chgnet", dispersion=True,
    dispersion_cutoff=50.3, dispersion_cutoff_smoothing="none",
)
```

`cnthr`, the coordination-number cutoff, is left at torch-dftd's own default;
torch-dftd clamps it to `cutoff` when it is larger, which is the same path PFP
goes through.

See [`docs/models.md`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/models.md) for the full per-model table.

## MACE needs its own environment

MACE is supported since 0.5.0, but it **cannot share an environment with the
other MLIP backends**, and no amount of pip flags will change that:

| Package | `e3nn` requirement |
|---|---|
| `mace-torch` | `==0.4.4` |
| `sevenn` | `>=0.5.0` |
| `fairchem-core` | `>=0.5` |
| `mattersim` | `>=0.5.0` |
| `nequip` | `>=0.6.0,<0.7.0` |

An exact pin against four lower bounds has no solution, so `mace` is deliberately
excluded from the `all` extra. Give it a second virtual environment:

```bash
python -m venv .venv-mace
.venv-mace/bin/pip install "ase-calculator-kit[mace]"

# with the D3 correction as well
.venv-mace/bin/pip install "ase-calculator-kit[mace,dispersion]"
```

Everything else in this package behaves identically there — the factory, the
dispersion policy, the DFT backends. Only the other MLIP backends are missing,
and asking for one reports the usual `MissingDependencyError`. Conversely, in
your main environment `get_calculator("mace")` raises a `MissingDependencyError`
that names this constraint rather than suggesting an install that cannot work.

```python
from ase.build import bulk
from ase_calculator_kit import get_calculator

atoms = bulk("Cu", "fcc", a=3.6)
atoms.calc = get_calculator("mace", head="omat_pbe")   # model="mh-1" by default
print(atoms.get_potential_energy())
```

The MH-1 checkpoint (~57 MB) is downloaded on first use and cached in
`~/.cache/mace`. Its model card states an **ASL** license, which is not the MIT
license of this package — check the model's own terms before using it in
commercial work.

### GPU acceleration (`accelerator=`)

MACE can replace its equivariant tensor products with
[cuequivariance](https://github.com/NVIDIA/cuEquivariance) or
[openequivariance](https://github.com/PASSIONLab/OpenEquivariance) kernels on
CUDA. `accelerator=` selects that, and defaults to `"auto"`:

| `accelerator` | Behaviour |
|---|---|
| `"auto"` (default) | Use cuequivariance when it is installed **and** demonstrably correct on this GPU; otherwise fall back to the plain model with a `RuntimeWarning` |
| `"cueq"` | Force cuequivariance. Errors are yours to see — no probe, no fallback |
| `"oeq"` | Force openequivariance |
| `"none"` | Plain model, probe included in what is skipped |

```bash
.venv-mace/bin/pip install "mace-torch[cueq,cueq-cuda-12]"
```

```python
atoms.calc = get_calculator("mace")                       # auto
atoms.calc = get_calculator("mace", accelerator="cueq")   # force it
atoms.calc = get_calculator("mace", accelerator="none")   # never
```

**Why `"auto"` measures instead of asking "is it importable".** Both cheaper
questions have been observed to give the wrong answer:

- On a Tesla V100 (sm_70), cuequivariance 0.11 imports, the calculator builds,
  and then the **first energy evaluation** dies with
  `cudaErrorNoKernelImageForDevice` — the shipped kernels do not cover that
  architecture. An import check would hand back a calculator that explodes
  later, in the middle of a run.
- [ACEsuit/mace#1298](https://github.com/ACEsuit/mace/issues/1298) reports
  cuequivariance returning +5500 eV where the plain model returns −200 eV on a
  multi-head checkpoint — **without raising at all**.

So `"auto"` builds both models, compares them on a two-atom cell, and keeps the
accelerated one only if the energies agree. The extra cost is one model build
and two tiny single points, paid only when cuequivariance is installed; if it
is not, `"auto"` is free. `enable_cueq=` / `enable_oeq=` can still be passed
directly, and an explicit flag skips the probe.

Measured on a V100 with cuequivariance 0.11.1 installed: `accelerator="auto"`
warns, falls back, and returns −3.74034995 eV for `bulk("Cu")` — identical to
CPU float64 to 1e-13 eV — while `accelerator="cueq"` raises.

## OrbMol (orb-models)

OrbMol-v2 shares an environment happily with every other backend — there is no
`e3nn` problem here and no second virtual environment to make. What it has is a
Python-version problem, and it is one line of upstream metadata deep:
`orb-models` pins `dm-tree==0.1.8`, and dm-tree 0.1.8's newest wheels are cp312
on every platform. On Python 3.13 or 3.14 pip therefore falls into a source
build that needs a C++ toolchain and a CMake old enough to accept the project,
and stops:

```
ERROR: Failed building wheel for dm-tree
```

That is why `orb` is not part of `[all]`: an `[all]` carrying it would fail to
install on two of the three Pythons this package supports.

**On Python 3.12, nothing special is needed:**

```bash
pip install "ase-calculator-kit[orb]"
```

**On Python 3.13 and 3.14, override that one pin.** dm-tree 0.1.10 ships cp313
and cp314 wheels, orb-models uses exactly two functions from it
(`tree.flatten`, `tree.map_structure`), and OrbMol-v2 on 3.13 with dm-tree
0.1.10 reproduced every one of this package's 3.12 reference energies to the
last digit. With [uv](https://docs.astral.sh/uv/), which has a dependency
override mechanism:

```bash
echo 'dm-tree>=0.1.10' > dm-tree-override.txt
uv pip install --override dm-tree-override.txt "ase-calculator-kit[orb]"
```

pip has no equivalent, so there the override is spelled `--no-deps` plus the
dependency list:

```bash
pip install "ase-calculator-kit" "dm-tree>=0.1.10"
pip install --no-deps "orb-models>=0.7,<0.8"
pip install "cached-path>=1.7.1" "scipy>=1.15.1" "torch>=2.8,<3" \
            "tqdm>=4.67.1" "nvalchemi-toolkit-ops[torch]>=0.3.1,<0.4"
```

`pip check` will then report that `orb-models 0.7.0 requires dm-tree==0.1.8`.
That warning is the override working, not a broken install.

The pin is tracked upstream as
[orbital-materials/orb-models#168](https://github.com/orbital-materials/orb-models/issues/168);
it was introduced deliberately, in
[#78](https://github.com/orbital-materials/orb-models/pull/78), to dodge a macOS
build problem that dm-tree has since fixed. When it is relaxed, CI's
wheel-availability check fails and the Python-versions table above gets updated.

OrbMol-v2 does not run on Apple Silicon GPUs — see
[Apple Silicon (MPS) support](#apple-silicon-mps-support) — and needs
`atoms.info["charge"]` and `atoms.info["spin"]` on every structure, see
[Molecular systems](#molecular-systems-charge-and-spin).

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]" -c constraints.txt
.venv/bin/pytest

# MACE has its own environment; the same suite runs there too
python -m venv .venv-mace
.venv-mace/bin/pip install -e ".[mace,dispersion,dev]" -c constraints.txt
.venv-mace/bin/pytest

# orb shares the main environment, but needs Python 3.12 (see above)
uv venv --python 3.12 .venv-orb
uv pip install --python .venv-orb/bin/python -e ".[orb,dev]" -c constraints.txt
.venv-orb/bin/pytest -m slow -k orb
```

`pyproject.toml` declares compatible version ranges so the package installs
next to whatever ASE/NNP versions you already have; `constraints.txt` pins the
exact combination that is tested, and CI installs with it.

`pytest` runs only the fast tests by default. Slow tests
(`pytest -m slow`) run real MLIP CPU single-point calculations and may download
model weights; install `.[dev,all] -c constraints.txt` first so every backend is
importable.

## Further Reading

- [`docs/models.md`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/models.md) — per-model dispersion policy and training
  functionals.
- [`docs/code-guide_ja.md`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/code-guide_ja.md) — 実装の化学的な判断と
  モジュールの責務（日本語）.
- [`AGENTS.md`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/AGENTS.md) — repository map, invariants, and conventions for
  AI coding agents (Claude, GPT, and others).
- [`CHANGELOG.md`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/CHANGELOG.md) — release history.
- [`docs/releasing.md`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/releasing.md) — how a release reaches PyPI and
  Zenodo.

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

MIT


### Molecular PySCF and GPU4PySCF (0.5.9)

Molecular HF/DFT uses the same config-only factory as VASP/QE, with upstream
PySCF energies and analytic gradients exposed through a small ASE adapter:

```python
from ase.io import read
from ase_calculator_kit import get_calculator

atoms = read("complex.xyz")
atoms.calc = get_calculator(
    "gpu4pyscf", config="examples/dft/gpu4pyscf_wb97mv.yaml",
    overrides={"parameters": {"charge": 2, "multiplicity": 3}},
    write_resolved_config=True,
)
forces = atoms.get_forces()             # eV/Angstrom (request first)
energy = atoms.get_potential_energy()   # eV, reuses the SCF above
```

Install only the required extra, preferably in a dedicated environment:

```bash
pip install 'ase-calculator-kit[pyscf]'
# Linux x86_64, NVIDIA GPU, CUDA Toolkit 12.x; validated with CUDA 12.8:
pip install 'ase-calculator-kit[gpu4pyscf-cuda12x,pyscf-dispersion]' -c constraints.txt
# CPU D3(BJ)/D4 (validated on Linux):
pip install 'ase-calculator-kit[pyscf,pyscf-dispersion]'
```

These extras are excluded from `all`. CPU PySCF and pyscf-dispersion resolve
on Linux with Python 3.12–3.14; actual numerical validation uses Python 3.12.
The CUDA extra supports Python 3.12/3.13, **not 3.14**, because the validated
CuPy 13.4.1/cuTENSOR 2.2.0 pair has no cp314 wheel. GPU wheels do not support
macOS. Bare CPU PySCF was also tested on Apple Silicon; the dispersion extra
requires >=1.5, which is unavailable on macOS at this release. The older macOS
1.0.0 wheel fails during import and is intentionally not accepted.

See [molecular configuration and validation](docs/pyscf.md) for the complete
settings, scope, CPU/GPU comparisons, and limitations. The primary examples
use **omegaB97M-V/def2-TZVPD + VV10**, with explicit electronic states and grids;
SMD water, Ru ECP and PBE-D3 examples are separate files.

```bash
python examples/dft/run_pyscf.py complex.xyz \
  --config examples/dft/gpu4pyscf_wb97mv.yaml --output runs/complex
python examples/dft/run_pyscf.py complex.xyz \
  --config examples/dft/gpu4pyscf_wb97mv_smd.yaml --output runs/complex-smd \
  --optimize --fmax 0.05 --steps 100
```

The example writes energy/forces/coordinates to `results.npz`, settings,
versions and convergence to `results.json`, the effective settings to
`resolved_calculator_config.yaml`, and the final geometry to `final.xyz`.
Optimization also writes a trajectory. The output directory must be new;
SCF failure or incomplete optimization exits with an error. Choose charge and
multiplicity in the YAML for each system: the supplied neutral singlet examples
are examples, not a default inference about your complex.


### eSEN OMol25

```python
atoms.info.update(charge=0, spin=1)  # spin is multiplicity
atoms.calc = get_calculator("esen", device="cuda")
```

The default is `esen-sm-conserving-all-omol`, whose forces are energy gradients.
Also selectable with `model=`: `esen-sm-direct-all-omol` and
`esen-md-direct-all-omol`. Direct models predict forces separately and should
not be assumed to conserve energy. All three use `task="omol"` only and reject
additional D3: OMol25's omegaB97M-V/def2-TZVPD reference already includes VV10.
They use fairchem-core 2.22+, with `inference_settings="batch"` by default,
CPU/CUDA devices, and Hugging Face `facebook/OMol25` access. No MPS support is
claimed. `[esen]` is an alias for the same dependency as `[uma]`, already in
`[all]`; it adds no conflicting stack. Legacy eSEN-30M-OMat remains unsupported.
Set charge/multiplicity explicitly: the upstream FAIRChemCalculator otherwise
warns and supplies neutral-singlet values, just as it does for UMA omol.

### Sharing electronic states between MLIP and PySCF

```python
atoms.info.update(charge=-1, spin=1)  # anion, singlet
atoms.calc = get_calculator("pyscf", config={
    "calculator": "pyscf",
    "parameters": {"basis": "def2-tzvpd", "xc": "wb97m_v"},
})
```

PySCF reads the same `atoms.info` fields as UMA/OrbMol and converts multiplicity
into PySCF's `spin = multiplicity - 1`. You may also supply charge/spin or
multiplicity in YAML, but **any disagreement raises before calculation or
returning a cached result**. Missing values never default to neutral singlet.
Changing charge/spin alone triggers a recalculation. `calc.metadata` records the
actual values and sources; `write_resolved_config=True` records the effective
state in YAML at calculation time. Initial per-atom charges/magnetic moments
are not interpreted as the total charge or multiplicity. The execution example
also saves `final.extxyz` with charge/multiplicity for reuse; ordinary XYZ does
not preserve that metadata.
