# Backends and model selection

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
[validation record](matgl-validation.md) and
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

See [the validation record](matgl-validation.md) for the common-system
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

### SevenNet FlashTP (`enable_flash=`)

[FlashTP](https://github.com/SNU-ARC/flashTP) replaces SevenNet's channel-wise
tensor products with fused, sparsity-aware CUDA kernels. It changes the kernels,
not the model, so energies agree with the plain e3nn path to float32 round-off.
It is not on PyPI; build it against the torch already in the environment, with
a CUDA toolkit (nvcc) of 12 or newer that matches that torch:

```bash
module load cuda/12.8.0          # or any nvcc matching torch.version.cuda
CUDA_ARCH_LIST="90" pip install --no-build-isolation \
    git+https://github.com/SNU-ARC/flashTP.git@v0.1.1
```

```python
atoms.calc = get_calculator(
    "sevennet", model="7net-omni", modal="matpes_pbe",
    device="cuda", enable_flash=True, dispersion=True,
)
```

sevenn answers a FlashTP request it cannot honour (package missing, no GPU)
with a warning and quietly runs plain e3nn. Here the request is strict instead:

| Situation | Result |
|---|---|
| `flashTP_e3nn` not importable | `MissingDependencyError` with the install command, before the model loads |
| `device` is not CUDA | `ValueError`, before the model loads |
| `enable_cueq=True` or `enable_oeq=True` as well | `ValueError` (sevenn takes exactly one accelerator) |
| Model built, but no `IrrepsScatterGatterFusedConvolution` layer in it | `RuntimeError` (sevenn fell back) |

`dispersion=True` wraps the FlashTP model exactly like the plain one; the D3
term is torch-dftd's and does not use FlashTP. `tests/tsubame_validation_062_flash.py`
is the GPU check behind the agreement claim. On an H100 MIG slice (369 atoms,
`matpes_pbe`) it measured identical energies, forces within 1.2e-6 eV/Å, and
a 5.1× faster force call (4.1× with D3).

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
| `omol-0` / `MACE-OMOL-0` | `omol` | OMol25, ωB97M-V |
| `polar-1-s/m/l` / `MACE-POLAR-1` | single | OMol25 with electrostatics; family name selects medium |
| `medium-omat-0` | single | **OMat24, PBE(+U)** — MACE-OMAT-0 |
| `small-omat-0` | single | OMat24, smaller |
| `medium-mpa-0` | single | MPtrj + sAlex, PBE(+U) |
| `mace-matpes-pbe-0` | single | MatPES, PBE |
| `mace-matpes-r2scan-0` | single | MatPES, r2SCAN |

```python
atoms.calc = get_calculator("mace")                            # mh-1 / omat_pbe
atoms.calc = get_calculator("mace", model="medium-omat-0")     # no head needed
```

`head` defaults to `"auto"`: `omat_pbe` for `mh-1`, the loader-owned `omol` head
for `omol-0`, and no head argument for the other single-head checkpoints above
(they carry one head named `Default`, and handing
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
| `omat` (default for the multi-task checkpoints) | Inorganic bulk/materials, stress, cell optimization |
| `omol` | Molecules and polymers |
| `oc20` | Catalyst surfaces and adsorption |
| `oc22` | Oxide catalysis |
| `oc25` | Electrochemistry / solid-liquid interfaces |
| `odac` | MOFs and direct air capture |
| `omc` | Molecular crystals |

For the molecular task (`omol`), set `atoms.info["charge"]` and
`atoms.info["spin"]` before computing — see
[Molecular systems](molecular.md) for why this matters.

### UMA fine-tunes: r²SCAN and ωB97M-D4rev

fairchem-core 2.24 ships two single-task fine-tunes of `uma-s-1p2p1` that move
the level of theory. Both run on UMA Nitro and are selected with `model=`; the
default `task="auto"` picks the one task each carries, and any other task is a
`ValueError` before the download.

| `model` | `task` | Level of theory | Training data | Use for |
|---|---|---|---|---|
| `uma-s-1p2p1-ft-r2scan` | `omat` | r²SCAN, no dispersion, **not spin-polarized** | MAD-1.6 ([arXiv:2603.02089](https://arxiv.org/abs/2603.02089)) | most inorganic and mixed systems |
| `uma-s-1p2p1-ft-d4rev` | `omol` | ωB97M-D4rev, dispersion included | ~4M OMol25 structures relabelled | molecules, noncovalent interactions, molecular liquids and crystals |

```python
atoms.calc = get_calculator("uma", model="uma-s-1p2p1-ft-r2scan", dispersion=True)  # r²SCAN + D3(BJ)

molecule.info.update(charge=0, spin=1)
molecule.calc = get_calculator("uma", model="uma-s-1p2p1-ft-d4rev")                 # ωB97M-D4rev
```

- **Dispersion follows the model, not the task.** `dispersion=True` on
  `ft-r2scan` adds D3(BJ) with **r²SCAN** parameters, as fairchem recommends —
  not the PBE parameters that `omat` selects on `uma-s-1p2p1`. On `ft-d4rev` it
  is refused, because D4rev is already in the reference data.
- **`ft-r2scan` treats every system as non-magnetic.** Do not use it for
  magnetic systems, or for cohesive energies (isolated atoms are often strongly
  magnetic); use `uma-s-1p2p1` with `omat` there.
- **`ft-d4rev` needs charge and multiplicity**, exactly like the `omol` task:
  the kit refuses a missing or inconsistent electronic state before inference.
- Both were fine-tuned on less data than `uma-s-1p2p1`, which is often more
  accurate at its own levels of theory, and neither ships isolated-atom
  reference energies, so single-atom energies may be unreliable.

Checked on an H100 (MIG 4g.47gb), fairchem-core 2.24.0, torch 2.13.0, with
[`tests/tsubame_validation_064_finetunes.py`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/tests/tsubame_validation_064_finetunes.py):
twelve rattled frames per system, the kit default (task `"auto"`, Nitro) against
`"high-precision"` (FP32):

| `model` | system | Nitro ran | steady (Nitro / FP32) | max ΔE | ΔF rmse |
|---|---|:--:|--:|--:|--:|
| `uma-s-1p2p1-ft-r2scan` | 27-atom fcc Cu | ✅ | 3.9 / 13.1 ms | 0.10 meV/atom | 0.0003 eV/Å |
| `uma-s-1p2p1-ft-d4rev` | ethanol, singlet | ✅ | 4.3 / 14.7 ms | 0.05 meV/atom | 0.0006 eV/Å |
| `uma-s-1p2p1-ft-d4rev` | OH radical, doublet | ✅ | 4.3 ms / — | 0.30 meV/atom | 0.0007 eV/Å |

The OH radical's FP32 steady state (184 ms) is left out: on two atoms the
compiled FP32 path kept recompiling across frames, so it is not a comparable
number. On the Cu cell, `dispersion=True` added −7.3053 eV, the r²SCAN D3(BJ) term
computed on its own (−7.3053 eV), not the PBE one (−15.7485 eV).

### UMA `inference_settings`

`inference_settings=` chooses how the predict unit is built, which is where
UMA's speed/precision trade lives. It reaches fairchem's `get_predict_unit()`,
so it must be set when the calculator is created, not afterwards.

| `inference_settings` | on CUDA | model | Use for |
|---|:--:|---|---|
| `"nitro"` (default) | TF32 | UMA Nitro kernels + CUDA graphs | MD and relaxation — one system, many steps |
| `"default"` | TF32 | fairchem's alias of `"nitro"` | — |
| `"multigpu"` | TF32 | `merge_mole` + `compile` | multi-worker inference (`workers > 1`) |
| `"high-precision"` | FP32 | `merge_mole` + `compile` | model comparisons, Hessians |
| `"batch"` | FP32 | unmerged | many different structures |
| `"traineval"` | FP32 | unmerged, graph generation v1 | reproducing fairchem's training/eval numbers |

```python
atoms.calc = get_calculator("uma", task="omat")                          # MD (Nitro)
atoms.calc = get_calculator("uma", task="omat", inference_settings="high-precision")
atoms.calc = get_calculator("uma", task="omat", inference_settings="batch")
```

**The default is UMA Nitro** (since 0.6.3, which requires fairchem-core 2.24).
Nitro is a standalone inference model for `uma-s-1p2p1` — the kit's default
checkpoint — with fused Triton/CuTe kernels and CUDA graphs, for one system per
call on a CUDA GPU of compute capability 8.x or newer. When it cannot serve a
request (another checkpoint, the CPU, an older GPU, a change of composition,
task, charge or spin), fairchem logs `UMA Nitro cannot serve this request` and
runs `"multigpu"` instead, falling further back to the unmerged model if
needed. The call still succeeds; on the CPU every UMA calculator takes this
path. `"turbo"` was removed in fairchem-core 2.24 and is refused here with a
pointer to `"nitro"` / `"multigpu"`.

Measured with fairchem-core 2.24.0 and torch 2.13.0 on an H100 (MIG 3g.47gb)
with [`tests/tsubame_validation_063_nitro.py`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/tests/tsubame_validation_063_nitro.py):
rattled fcc Cu, `task="omat"`, twelve frames of the same system, steady state
averaged over the last eight, differences against `"high-precision"` (FP32):

| `inference_settings` | 27 atoms: first step | steady | 512 atoms: steady | max ΔE | ΔF rmse |
|---|--:|--:|--:|--:|--:|
| `"high-precision"` | 37.5 s | 21.8 ms | 96.6 ms | — | — |
| `"nitro"` (default) | 9.3 s | **5.4 ms** | **31.4 ms** | 0.15 meV/atom | 0.0004 eV/Å |
| `"multigpu"` | 25.4 s | 13.3 ms | 41.7 ms | 0.14 meV/atom | 0.0004 eV/Å |
| `"batch"` | 0.26 s | 55.5 ms | 269 ms | 0.0003 meV/atom | 0.0000005 eV/Å |

First-step times are with the Triton/inductor caches already populated by an
earlier run; from a cold cache the same job took 85.5 s (`high-precision`),
21.3 s (`nitro`) and 60.6 s (`multigpu`). Steady-state `multigpu` also varied
between runs on different GPUs (13.3 vs 26.9 ms); Nitro against FP32 did not
(5.4 vs 21.8 ms here, 4.1 vs 22.8 ms in the first run).

Nitro is 4× faster than the FP32 path for 27 atoms and 3× for 512, and its
error is the TF32 error `multigpu` already has. The run also confirmed that the
default really ran `UMANitroModel`, that a vacancy (same element fractions)
keeps Nitro, and that substituting one Cu with Ni falls back with `Compositions
differ from merged model` and still answers. (The 2.22 `default` — merge +
compile in FP32 — measured 13.5 ms on the same 27-atom system on a MIG
4g.47gb slice.)

**Nitro computes in TF32, which is reduced precision.** Up to 0.6.2 this package
kept TF32 opt-in, because it exists to compare models and a 0.1 meV/atom
artefact is the size of a model difference. With Nitro the speed gain is no
longer a few per cent, so the default follows fairchem; fairchem logs `Running
in a reduced-precision mode` when it applies. **When a number goes into a
comparison between models, pass `inference_settings="high-precision"`**, the
FP32 fast path. Use it for Hessians too: Nitro has no double backward. TF32 is
applied inside a context manager around UMA's own forward pass, which restores
`torch.get_float32_matmul_precision()` afterwards — so it does not leak reduced
precision into other calculators sharing the process. On CPU it does nothing.

**The fast paths assume the system never changes.** Composition, task, total
charge and spin must all stay fixed — true of an MD trajectory, false of a loop
over structures. fairchem notices and falls back with a warning. Nothing is
wrong when you see that, but a Nitro build or a merge and a compile were paid
for and thrown away: pass `inference_settings="batch"` when the structures
vary. With fairchem-core 2.22 on CPU that was 18.8 s per single point against
2.6 s. Which is why
[`examples/run_all_models.py`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/examples/run_all_models.py)
uses `"batch"` for its UMA lines.

Anything the named presets do not cover — `max_atoms`, `edge_chunk_size`,
`execution_mode`, a different `base_precision_dtype` — goes through as a
`fairchem.core` `InferenceSettings` object:

```python
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

atoms.calc = get_calculator(
    "uma", task="omat",
    inference_settings=InferenceSettings(merge_mole=True, compile=True, max_atoms=2048),
)
```

A misspelled name raises `ValueError` listing the presets, before the checkpoint
is downloaded. fairchem checks it with a bare `assert`, which `python -O` removes.

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
They use fairchem-core 2.24+, with `inference_settings="batch"` by default,
CPU/CUDA devices, and Hugging Face `facebook/OMol25` access. No MPS support is
claimed. `[esen]` is an alias for the same dependency as `[uma]`, already in
`[all]`; it adds no conflicting stack. Legacy eSEN-30M-OMat remains unsupported.
Set charge/multiplicity explicitly: since 0.5.10 the kit rejects missing or
inconsistent electronic states for both eSEN and UMA omol before inference.

## VASP and Quantum ESPRESSO

Both require `calculator`, `profile.command`, and reviewed `parameters`.
QE additionally requires `profile.pseudo_dir` and top-level `pseudopotentials`.
Unknown YAML keys are rejected before execution. See the
[DFT examples](../examples/dft/README.md) for complete configurations.

## CP2K

`cp2k` requires YAML `profile.command` and `profile.data_dir`; XC, basis,
pseudopotential and cutoff (eV) must be explicit. Data filenames become absolute.
Charge/multiplicity are YAML-only and info mismatches raise before cache reuse.
Construction starts the CP2K shell: use `with` or `close()`. ASE reports
`free_energy == energy`, disabled stress can return zeros or fail, and slabs
need explicit Poisson settings in `inp`. See [the CP2K guide](cp2k.md) for
templates, lifecycle, source checks and limitations.

## PySCF and GPU4PySCF

See [the molecular DFT guide](pyscf.md) for SCF controls, restart and Hessians.
