# ase-calculator-kit

[![PyPI](https://img.shields.io/pypi/v/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![Python](https://img.shields.io/pypi/pyversions/ase-calculator-kit)](https://pypi.org/project/ase-calculator-kit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21807793.svg)](https://doi.org/10.5281/zenodo.21807793)

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
- [MACE](https://github.com/ACEsuit/mace) — **must be installed in a separate
  virtual environment**, see [MACE needs its own environment](#mace-needs-its-own-environment)

Supported DFT backends:

- VASP
- Quantum ESPRESSO (`qe`, `espresso`, `quantum-espresso`)

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
pip install "ase-calculator-kit[mattersim]"
pip install "ase-calculator-kit[nequip]"
pip install "ase-calculator-kit[uma]"

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

Missing backend packages are reported only when that calculator is requested,
with the matching extra to install.

### Python versions

Python 3.12 and newer. The package itself has no upper bound, but one backend
currently does:

| | 3.12 | 3.13 | 3.14 |
|---|:--:|:--:|:--:|
| Core, `chgnet`, `sevennet`, `mattersim`, `nequip`, `mace`, `dispersion` | ✅ | ✅ | ✅ |
| `uma` (and therefore `all`) | ✅ | ✅ | ❌ |

`fairchem-core` declares `requires-python = ">=3.11,<3.14"` and pins
`torch~=2.8.0`, which has no cp314 wheels, so `pip install
"ase-calculator-kit[uma]"` fails on Python 3.14 with an error naming
`fairchem-core`. Nothing here needs to change once fairchem-core supports 3.14.

This is deliberately *not* hidden behind an environment marker: a marker would
make the install succeed on 3.14 while silently leaving UMA out.

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
| `mattersim` | MLIP | — |
| `nequip` | MLIP | — |
| `mace` | MLIP (separate environment) | — |
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
| `mace` | `model="mh-1"`, `head="omat_pbe"`, `default_dtype="float64"`, `accelerator="auto"` |
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

available_mlip_models()     # ['chgnet', 'fairchem', 'mace', 'mattersim', 'nequip', 'sevennet', 'uma']
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
on an Apple Silicon Mac (arm64, PyTorch 2.8, MPS available). Results:

| Backend | `device="mps"` | Notes |
|---|---|---|
| SevenNet | ✅ supported | validated locally (`7net-omni`) |
| CHGNet | ✅ supported | validated locally |
| MatterSim | ✅ supported | validated locally |
| NequIP OAM | ❌ not supported | PyTorch MPS lacks float64; the packaged OAM models use float64 buffers |
| MACE | ❌ not supported | same float64 problem: loading `mace-mh-1.model` with `map_location="mps"` raises `Cannot convert a MPS Tensor to float64`, with `default_dtype="float32"` as well |
| UMA / fairchem | ❌ not supported | `fairchem-core` asserts `device in {"cpu", "cuda"}` |

For the MPS-supported backends, `device="auto"` resolves to `mps` on Apple
Silicon when no CUDA device is present. NequIP, MACE and UMA accept only
`"cpu"` / `"cuda"`; passing `device="mps"` raises a clear `ValueError`, and
`device="auto"` falls back to `cpu`.

## Choosing an MLIP Variant

### SevenNet `model`

The Omni family is one training recipe at three capacities, so everything in the
`modal` table below applies unchanged to all three:

| `model` | Use for |
|---|---|
| `7net-omni` (default) | Recommended general model |
| `7net-omni-i8` | Larger capacity: more accurate, slower |
| `7net-omni-i12` | Largest of the family |

```python
atoms.calc = get_calculator("sevennet", model="7net-omni-i12", modal="mpa")
```

Keep the model fixed across a campaign — i8 and i12 are separate models, not
refinements of an `7net-omni` number, so energies are not comparable between
them.

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

### MACE `head`

MACE-MH-1 (`model="mh-1"`, the default) is a single checkpoint with six readout
heads. The head *is* the level of theory, not an accuracy setting: the same
structure returns PBE, r2SCAN or ωB97M energies depending on which one you pick.

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

## Molecular systems (charge and spin)

Molecular models need two inputs that no bulk model does: the **total charge**
of the system and its **spin multiplicity** (`2S+1`). ASE has no standard place
for either, so they are passed through `atoms.info`, and the backends differ in
whether they read them at all.

| Backend | Molecular option | Takes charge / spin? |
|---|---|---|
| `uma` | `task="omol"` | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` |
| `sevennet` | `modal="omol25_low"` / `"omol25_high"` / `"spice"` / `"qcml"` | ❌ not supported by sevenn |
| `mace` | `head="omol"` / `"spice_wB97M"` | ❌ the MH-1 molecular heads are fitted to neutral closed-shell data |

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

### MACE: molecular heads, no charge or spin

MACE-MH-1's `omol` head was fine-tuned on a subsample of OMol25 that is
explicitly **neutral and closed-shell**, and `spice_wB97M` on SPICE-1, which is
likewise neutral. There is no charge or spin input to set, and no head that
covers ions or a chosen open-shell state. Use `get_calculator("uma",
task="omol")` when the charge and spin of the system matter.

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

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]" -c constraints.txt
.venv/bin/pytest

# MACE has its own environment; the same suite runs there too
python -m venv .venv-mace
.venv-mace/bin/pip install -e ".[mace,dispersion,dev]" -c constraints.txt
.venv-mace/bin/pytest
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
