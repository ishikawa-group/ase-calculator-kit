# Changelog

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
