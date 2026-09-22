# Getting started

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
[Apple Silicon (MPS) support](devices.md) — and needs
`atoms.info["charge"]` and `atoms.info["spin"]` on every structure, see
[Molecular systems](molecular.md).

See [API and examples](api.md), [backends](backends.md), and [PySCF](pyscf.md).
