# Changelog

## 0.5.3

Aligns the D3 correction's numerical settings with PFP's, so that the dispersion
term added to an NNP energy here is the same quantity PFP adds to its own.

- **`dispersion=True` now runs torch-dftd at `cutoff=14.0` Å with
  `cutoff_smoothing="poly"`,** rather than torch-dftd's defaults of 95 Bohr
  (50.3 Å) and no smoothing. **This changes energies** — recompute anything you
  intend to compare against results from 0.5.2 or earlier.

  The point is comparability. This package exists to put models side by side,
  and PFP is one of the models being compared; with torch-dftd's defaults in
  place, "SevenNet + D3" and "PFP + D3" carried *different dispersion terms*, so
  the difference between them was not only the difference between the models.

  The settings are PFP v7.0.0+'s, and Matlantis published the validation behind
  them: over the Wellendorff adsorption benchmark the shorter cutoff moves
  adsorption energies by an MAE of 0.0024 eV — negligible against the 0.01 eV
  scale those numbers are discussed at — and leaves the 90th percentile of the
  COD unit-cell-volume error unchanged, while raising the reachable system size
  from about 5000 atoms to roughly three times that. `cutoff_smoothing="none"`
  was a bug on PFP's side, fixed in v7.0.0; it leaves the force discontinuous at
  the cutoff radius, which is where a relaxation or an MD run meets it.

- **`dispersion_cutoff=` and `dispersion_cutoff_smoothing=` are new keywords**
  on every MLIP backend, so the previous behaviour — and any dataset built on
  it — stays reproducible with
  `dispersion_cutoff=50.3, dispersion_cutoff_smoothing="none"`. Like
  `dispersion_damping`, both are validated in `precheck_dispersion_xc()` before
  the checkpoint is downloaded.

- `cnthr`, the coordination-number cutoff, is deliberately left at torch-dftd's
  own default of 40 Bohr. torch-dftd clamps it to `cutoff` when it is larger and
  warns that it did, so at 14 Å the effective value is 14 Å — the same clamp
  PFP's stack goes through. Pinning it here would silently change what a caller
  who *raises* `cutoff` gets.

## 0.5.2

Makes the OMat24-trained variant of each backend selectable in one argument, and
stops a variant selector the model ignores from choosing its D3 parameters.

- **`get_calculator("mace", model="medium-omat-0")` (MACE-OMAT-0) now works.**
  It is a single-head checkpoint, and `head=` defaulted to `"omat_pbe"`, so the
  head guard added in 0.5.0 rejected it outright: the model was reachable only
  by knowing to pass `head=None`. `head` now defaults to `"auto"`, which resolves
  to `omat_pbe` for `mh-1` and to no head at all for the single-head checkpoints
  (`medium-omat-0`, `small-omat-0`, `medium-mpa-0`, `mace-matpes-pbe-0`,
  `mace-matpes-r2scan-0`, `small`/`medium`/`large`, …). An unlisted model or a
  local path also sends no head, so MACE's own error — which lists the heads the
  file actually has — is what a mismatch produces.

- **`get_calculator("sevennet", model="7net-omat")` (SevenNet-omat) now works,**
  likewise without a second argument: `modal` defaults to `"auto"`, sending
  `mpa` to the multi-fidelity models (`7net-omni`, `-i8`, `-i12`,
  `7net-mf-ompa`) and nothing to a single-fidelity one.

- **Fixed: a modal the model ignores no longer picks the D3 functional.** sevenn
  accepts `modal=` on a single-fidelity checkpoint, warns
  (`modal=... is ignored as model has no modal_map`), and drops it — but this
  package went on using it as the dispersion policy key. `model="7net-0",
  modal="matpes_r2scan"` therefore computed a PBE model and added **r2SCAN** D3
  parameters to it. The key is now the *resolved* modal, and an explicit modal a
  single-fidelity model cannot use raises instead of being silently dropped.
  MACE had the mirror image of the same defect (a head name the checkpoint does
  not carry); both now resolve through `"auto"`.

- **Dispersion policy rows for the single-variant checkpoints.** With no head or
  modal to key on, the model name does it: MACE `medium-omat-0` / `small-omat-0`
  and SevenNet `7net-omat` are OMat24 PBE(+U) → D3 `xc=pbe`; MACE
  `medium-mpa-0` → `pbe`, `mace-matpes-pbe-0` → `pbe`,
  `mace-matpes-r2scan-0` → `r2scan`. Models without a row keep falling back to
  the previous behaviour (SevenNet's single-fidelity `default` row) or to the
  unverified tier (MACE), so nothing that worked before changes verdict.

  Verified on real weights, `bulk("Cu", "fcc", a=3.6)`: MACE `medium-omat-0`
  −3.73953300 eV, SevenNet `7net-omat` −3.74727941 eV, against MACE
  `mh-1`/`omat_pbe` −3.74034995 eV — three architectures at the same reference
  level, within 8 meV, which is the cross-check that the right head/modal is
  active. `dispersion=True` adds −0.590604 eV in each case, the same D3(BJ)/PBE
  term.

- **Compatibility.** `head="omat_pbe"` and `modal="mpa"` written out explicitly
  keep working, and the defaults resolve to exactly what they were for
  `mh-1` and `7net-omni`. The one behaviour change is the error above, which
  replaces a silently wrong dispersion correction.

- **Documented: eSEN-30M-OMat cannot join the `uma` backend.** It was
  investigated for this release. `esen_30m_omat.pt` is a fairchem-core **1.x**
  checkpoint (`OCPCalculator`) in the gated repo `facebook/OMAT24`; fairchem-core
  2.x ships no eSEN architecture, and 1.10 pins `torch~=2.4`, `numpy<2` and
  Python `<3.13`, so it would need a third isolated environment the way MACE
  does. The finding is written down in the README and `AGENTS.md` rather than
  left to be rediscovered. The same OMat24/PBE(+U) reference level is available
  through MACE `medium-omat-0` and SevenNet `7net-omat`.

## 0.5.1

Gives the MACE backend a GPU accelerator setting, defaulting to `"auto"`.

- **`get_calculator("mace", accelerator=...)`** selects cuequivariance /
  openequivariance acceleration on CUDA: `"auto"` (default), `"cueq"`, `"oeq"`,
  `"none"`. Until now the flags were only reachable by passing `enable_cueq=`
  through `**kwargs`, undocumented, and with nothing to say when they went
  wrong. `enable_cueq=` / `enable_oeq=` still work and are treated as a
  deliberate choice.

- **`"auto"` measures rather than asking whether the package imports**, because
  the cheap question has been observed giving the wrong answer twice:

  - On a Tesla V100 (sm_70) with cuequivariance 0.11.1, the import succeeds,
    the calculator builds, and the *first energy evaluation* dies with
    `cudaErrorNoKernelImageForDevice` — the wheels ship kernels for newer
    architectures only. An import check would hand back a calculator that
    explodes later, mid-run.
  - [ACEsuit/mace#1298](https://github.com/ACEsuit/mace/issues/1298) reports
    cuequivariance returning +5500 eV where the plain model returns -200 eV on
    a multi-head checkpoint, raising nothing at all.

  So `"auto"` builds both models, compares them on a two-atom cell built from
  an element the checkpoint actually knows, and keeps the accelerated one only
  when the energies agree to 1e-3 eV — generous next to float32 noise (~1e-6
  eV), tight next to a 5700 eV error. Anything else falls back to the plain
  model with a `RuntimeWarning` naming the cause. The cost is one extra model
  build and two two-atom single points, and only when cuequivariance is
  installed at all; otherwise `"auto"` does nothing.

  Verified on a V100: `accelerator="auto"` warns, falls back, and returns
  -3.74034995 eV for `bulk("Cu")` — matching CPU float64 to 1e-13 eV — while
  `accelerator="cueq"` raises. The failed attempt does not poison the CUDA
  context: the fallback calculator keeps returning correct energies, checked on
  an 8-atom rattled supercell.

- **Behaviour change on GPUs where cuequivariance works.** 0.5.0 never enabled
  it; 0.5.1 does, when the probe agrees. Pass `accelerator="none"` to keep the
  old behaviour exactly.

## 0.5.0

Adds MACE — in a virtual environment of its own — and makes the larger SevenNet
Omni capacities selectable.

- **`get_calculator("mace", ...)` builds a MACE foundation model.** The default
  is MACE-MH-1 with its `omat_pbe` head at `default_dtype="float64"`, i.e.
  `get_calculator("mace")` == `get_calculator("mace", model="mh-1",
  head="omat_pbe", default_dtype="float64")`. float64 is what the MH-1 model
  card recommends and what geometry optimisation and phonons need; pass
  `"float32"` for faster MD. Verified against the real checkpoint: a CPU single
  point runs on all six heads, and `dispersion=True` on `omat_pbe` moves
  bulk Cu from -3.740350 eV to -4.330954 eV.

- **MACE must be installed in a separate virtual environment, and this release
  says so everywhere it can.** `mace-torch` pins `e3nn==0.4.4` while `sevenn`,
  `fairchem-core` and `mattersim` require `e3nn>=0.5.0` and `nequip`
  `e3nn>=0.6.0`, so no resolution installs MACE next to them:

  ```bash
  python -m venv .venv-mace
  .venv-mace/bin/pip install "ase-calculator-kit[mace]"
  ```

  The `mace` extra is therefore **not** part of `all`, which
  `tests/test_packaging.py` now enforces — `pip install "...[all,mace]"` can
  only fail. Requesting MACE from an environment that lacks it raises the usual
  `MissingDependencyError`, extended to explain the e3nn conflict and the second
  environment instead of suggesting an install that cannot succeed. This
  reverses the previous decision to exclude MACE from the package (AGENTS.md
  invariant 7), which is now "MACE ships, but never in the same environment".

- **An unknown MACE `head` is refused instead of silently answered.** MACE does
  not raise on a head the checkpoint does not have: it logs
  `Head <x> not found ... defaulting to the last head` and returns energies from
  that head, so a typo yields a plausible number at a different level of theory
  — the same failure shape as fairchem's silent `charge=0`/`spin=1`. The head is
  validated before the download, and again against the loaded checkpoint's
  `available_heads`.

  The head names come from the shipped `mace-mh-1.model`, not from the model
  card: the card advertises `rgd1_b3lyp`, the checkpoint carries
  `mp_pbe_refit_add`. The six real heads are `omat_pbe`, `mp_pbe_refit_add`,
  `oc20_usemppbe`, `matpes_r2scan`, `omol` and `spice_wB97M`.

- **Dispersion policy for every MACE head.** The head *is* the level of theory,
  so each one gets its own row: D3 `xc=pbe` for `omat_pbe`,
  `mp_pbe_refit_add` and `oc20_usemppbe`, `xc=r2scan` for `matpes_r2scan`, and a
  refusal for `omol` (ωB97M-VV10 already carries nonlocal dispersion) and
  `spice_wB97M` (ωB97M-D3(BJ) already includes D3). This follows the MH-1
  authors, who evaluate their PBE-trained heads with torch-dftd D3(BJ) and the
  PBE parametrisation and add nothing to the OMol head.

  One row deliberately follows the model paper rather than the dataset:
  `oc20_usemppbe` is described in the MH-1 paper as OC20 "computed at the PBE
  level" and is referenced to MP's PBE data, while OC20 as published is RPBE —
  which is what the SevenNet and UMA `oc20` rows use. Pass
  `dispersion_xc="rpbe"` to follow the dataset convention instead.

- **MACE on Apple Silicon: measured, and not supported.** Loading
  `mace-mh-1.model` with `map_location="mps"` fails with `Cannot convert a MPS
  Tensor to float64` — with `default_dtype="float32"` too, because the failure
  is in the checkpoint's stored tensors, not the compute dtype. Same cause as
  NequIP OAM. `device="mps"` raises, `device="auto"` resolves to CPU.

- **`7net-omni-i8` and `7net-omni-i12` are selectable.**
  `get_calculator("sevennet", model="7net-omni-i12", modal="mpa")`. They are the
  Omni recipe at larger capacity, so the whole `modal` table applies unchanged
  and the dispersion policy is shared. They are separate models, though, not
  refinements of a `7net-omni` number: do not mix them inside one campaign.

## 0.4.0

Adds control over the D3 damping function, corrects the OC25 dispersion entry,
and moves to sevenn 0.13.

- **`dispersion_damping=` selects Becke-Johnson or zero damping.**
  `get_calculator(..., dispersion=True, dispersion_damping="zero")` now reaches
  torch-dftd; the default stays `"bj"`, so existing calls are unchanged. Only
  `"bj"` and `"zero"` are accepted — torch-dftd's `zerom`/`bjm`/`dftd2` have no
  fitted parameters for the functionals in the policy table, and a typo that
  silently picked one would be worse than an error. The choice is validated in
  the precheck, before a multi-gigabyte checkpoint is loaded.

  This is not a cosmetic knob. D3 does not screen a metal's C6 coefficients, so
  on molecule-metal systems the two dampings can differ by a factor of two: for
  RPBE, benzene on Pt(111) picks up -4.6 eV of dispersion with BJ against
  -2.4 eV with zero damping.

- **Corrected: UMA `oc25` is RPBE + D3 with *zero* damping, not D3(BJ).**
  The dispersion policy table and `docs/models.md` both said BJ. The verdict was
  right — `dispersion=True` on `oc25` double-counts and is still refused — but
  the stated reference level was wrong, and it is the line a reader copies into
  a methods section. Confirmed against the OC25 dataset metadata (VASP 6.3.2,
  RPBE, D3 zero damping, 400 eV cutoff, non-spin-polarized).

- **`sevenn>=0.13`** (was `>=0.12,<0.13`). Verified on GPU that 0.13.0 loads
  `7net-omni` and reproduces 0.12.1 bit-for-bit across all six modals
  (`mpa`, `omat24`, `oc20`, `oc22`, `matpes_pbe`, `matpes_r2scan`).

- **Removed the sevenn debug-print filter.** 0.13 no longer prints the
  `cueq`/`flash` pairs from `SevenNetCalculator.__init__`, so
  `_without_sevennet_debug_prints()` had become a no-op.

## 0.3.5

Reworks how a release is cut, following pymatgen's arrangement. No runtime
behavior changes.

- **Publishing a GitHub Release is now the whole release.** It uploads to PyPI
  *and* is the event the Zenodo webhook listens for, so the artifact and its
  DOI can no longer come from different commits. Previously a tag push built
  without uploading and a second manual dispatch did the upload — which is how
  0.3.3 reached PyPI but never got a DOI. A manual run of the workflow is now
  always a TestPyPI dry run; there is no path to PyPI except a published
  release.
- **The test suite gates the upload.** `release.yml` calls `ci.yml` and builds
  nothing until it passes. Until now the two workflows were independent, so a
  red suite could not stop a permanent PyPI upload.
- **The version is derived from the git tag** via `setuptools-scm` instead of
  being written into `pyproject.toml`. It had to be kept in step across three
  files, and in 0.3.3 it was not: that release shipped a `CITATION.cff` still
  announcing 0.3.2. `CITATION.cff` remains hand-written — Zenodo reads it — but
  a test now holds it to the newest `CHANGELOG.md` entry.
- Re-running a partly-failed upload is a no-op rather than an error
  (`skip-existing`).

## 0.3.4

A metadata-only release. No runtime behavior changes.

- Correct `CITATION.cff`, which shipped in 0.3.3 still carrying `version:
  0.3.2`, and add `date-released`. Zenodo reads this file when it mints the
  DOI, so a stale version there would misdescribe the archived record.

## 0.3.3

Turns two documented-but-unenforced invariants into tests, and makes the
package citable. No runtime behavior changes.

- **Citation metadata.** `CITATION.cff` and `.zenodo.json` describe the work for
  Zenodo and for GitHub's "Cite this repository", `docs/releasing.md` documents
  the PyPI and Zenodo publication paths, and the package authors are now named
  individually — 0.3.2 shipped with only the `ishikawa-group` organisation in
  its `Author` field, which PyPI cannot retroactively correct.
- **`dispersion.py` and `docs/models.md` are now checked for drift.** The two
  were required to stay in sync, but nothing verified it — and that table is
  where a reader learns *why* a model refuses `dispersion=True`.
  `tests/test_models_doc_sync.py` parses the table and compares it against the
  policy entries in both directions, including the D3 `xc` for every allowed
  row. The table's first column now names every policy key in backticks, which
  is what the parser reads (`default`, `1M`/`5M`, `S`/`M`/`L`/`XL`).
- **CI resolves every extra on every supported Python** in a new
  `extras-resolve` job, using `uv pip compile` — resolution only, nothing is
  downloaded or installed, so it takes seconds. Until now the fast suite mocked
  the NNP backends, so nothing noticed an upstream cap or conflict until a user
  hit it; the `uma`/Python 3.14 incompatibility shipped in 0.3.2 was found by
  hand. The job's expectations mirror the README support table, so it also
  fails when an extra we document as unavailable starts working and the docs
  need updating.
- `docs/code-guide_ja.md` gains a section on molecular charge and spin — which
  backend accepts them, that UMA silently falls back to a neutral singlet, and
  why `get_calculator` has no `charge=`/`spin=` keyword.

## 0.3.2

- **First release published to PyPI**: `pip install ase-calculator-kit`.
- **Support Python 3.14.** `requires-python` drops its `<3.14` upper bound: a
  cap is baked into every uploaded file and makes the package invisible to a
  newer interpreter even where it works, and it can only be lifted by a new
  release. The core plus the `chgnet`, `sevennet`, `mattersim`, `nequip`, and
  `dispersion` extras are verified on 3.14; CI now runs 3.12, 3.13, and 3.14.
- The `uma` extra (and therefore `all`) does **not** install on Python 3.14:
  `fairchem-core` declares `requires-python = ">=3.11,<3.14"` and pins
  `torch~=2.8.0`, which has no cp314 wheels. Left unmarked on purpose, so pip
  fails with an error naming `fairchem-core` rather than quietly installing
  nothing. No change here will be needed once fairchem-core supports 3.14.
- Fix a stale README claim that SevenNet ships as a base dependency; the default
  install has been ASE + PyYAML only since 0.3.0.

## 0.3.1

### Molecular systems

- **Fix a wrong `modal` description.** SevenNet's `omol25_low` was documented as
  "molecular / high-fidelity"; it is in fact the **low-spin** OMol25 task, the
  counterpart of the high-spin `omol25_high`. Both are ωB97M-V — the split is by
  spin state, not by accuracy.
- Document the remaining `7net-omni` tasks that were missing entirely:
  `mp_r2scan`, `oc20`, `oc22`, `odac23`, `spice`, `qcml`, and `pet_mad`.
- Document that **SevenNet takes no total charge or spin multiplicity** (sevenn
  has no such input), so its molecular modals cannot describe ions or a chosen
  open-shell state. UMA's `omol` task is the option that can.
- Document that **UMA does not fail when `charge`/`spin` are missing**: fairchem
  logs a warning, writes `charge=0` / `spin=1` into the `atoms.info` you passed
  in, and returns a neutral closed-shell result — so an ion or radical comes back
  silently wrong. New README section "Molecular systems (charge and spin)" with
  anion and radical examples.

### Dispersion policy corrections

Molecular reference data is nearly always dispersion-corrected, and four tasks
were sitting on the overridable "unverified" tier where a stray `dispersion_xc=`
could have double-counted it. They are now always refused:

- SevenNet `spice` — SPICE is ωB97M-**D3(BJ)**/def2-TZVPPD.
- SevenNet `qcml` — QCML applies the **MBD-NL** many-body dispersion correction.
- SevenNet `odac23` and UMA `odac` — ODAC23 is **PBE-D3**.
- UMA `omc` — OMC25 is **PBE+D3**.

Newly allowed, with a verified functional: SevenNet `mp_r2scan` (D3 `xc=r2scan`)
and `pet_mad` (D3 `xc=pbesol`). `omat24` and `oc22` are relabelled PBE(+U) to
match SevenNet's own table; their D3 `xc` is unchanged.

No model/task/modal is left on the "unverified" tier. The tier itself stays, for
tasks a future upstream release adds.

### Packaging and docs

- Suppress the `cueq/False/flash/False` debug lines that sevenn 0.12.1 prints
  unconditionally when constructing `SevenNetCalculator`; any other stdout from
  the model load is still forwarded.
- Prepare for PyPI: publish compatible version ranges instead of exact `==`
  pins (the tested combination moves to `constraints.txt`), declare the license
  as a PEP 639 SPDX expression, add `py.typed` so downstream type checkers see
  the annotations, add Changelog/Issues URLs and Python 3.12/3.13 classifiers,
  and add a release workflow that builds and version-checks a `v*` tag. Uploading
  stays a manual `workflow_dispatch` step until PyPI Trusted Publishing is set up.
- Document the public API surface in the README and add `AGENTS.md`.
- `examples/run_all_models.py` covers every documented SevenNet modal, plus an
  OH⁻ anion and an OH radical through UMA `omol`.

## 0.3.0

- Make the default installation lightweight: only ASE and PyYAML are required.
- Move each NNP stack into an explicit extra: `chgnet`, `sevennet`, `mattersim`,
  `nequip`, and `uma`.
- Add `all` for every supported NNP plus D3, and `dispersion` for D3 alone.
- Improve missing-dependency errors so they show the exact extra to install.

## 0.2.2

- Verify dispersion functionals for models that were previously refused, moving
  them from the "unverified" tier to "allowed" (auto `xc`):
  - NequIP OAM (S/M/L/XL): PBE(+U)-level -> D3 `xc=pbe`.
  - SevenNet `oc20` -> `xc=rpbe`, `oc22` -> `xc=pbe`, `matpes_r2scan` -> `xc=r2scan`.
  - CHGNet keyed by model name: `0.3.0`/`0.2.0` (MPtrj, PBE) -> `xc=pbe`,
    `r2scan` (r2SCAN transfer learning) -> `xc=r2scan`. Fixes `dispersion=True`
    being refused for an explicitly-named CHGNet model.
- Keep `docs/models.md` and `dispersion.py` in sync with the above.

## 0.2.1

- Probe every MLIP backend on Apple Silicon (MPS) with a real single point and
  reconcile the `device="mps"` support flags with the measured results.
- Enable SevenNet `device="mps"` (validated locally with `7net-omni`).
- Confirm CHGNet and MatterSim `device="mps"` (already enabled) still compute
  energy and forces on MPS.
- Keep NequIP MPS disabled — PyTorch MPS lacks float64, which the packaged OAM
  models require (`Cannot convert a MPS Tensor to float64`).
- Document that UMA / fairchem does not support MPS: `fairchem-core` asserts
  `device in {"cpu", "cuda"}`.
- Add an "Apple Silicon (MPS) support" matrix to the README and backend
  pass-through tests for SevenNet (mps) and UMA (mps rejected).

## 0.2.0

- Add the NequIP OAM backend with the S, M, L, and XL model variants from
  nequip.net.
- Enable MatterSim `device="mps"` after local Apple Silicon validation.
- Document NequIP and MatterSim MPS behavior.

## 0.1.0

- Initial ase-calculator-kit release with MLIP and config-only DFT calculator
  factories.
