# Reviewed 0.6.0 validation

Validated on 2026-09-22 after review of the release preparation. The source
hashes in the [machine-readable summary](validation-0.6.0.json) match the shipped
DFT source. Reproduce the CPU/GPU checks with
[tsubame_validation_060.py](../tests/tsubame_validation_060.py) on a scheduled
compute node. Earlier results remain in [validation history](validation-history.md).

## Review corrections

The review reproduced and fixed:

- Failure diagnostics were erased by an incorrect metadata-key check.
- Iteration checkpoints read stale/uninitialized mf.mo_* instead of callback
  orbitals; write failures were silently suppressed. Files now contain complete
  iteration data, and I/O failures propagate with their stage recorded.
- Restart lacked full physics/basis provenance and did not project across
  changed geometries. Density reuse compared current settings against themselves,
  so an electronic-state change could inherit the wrong old density.
- Forces followed by Hessian failed without retention. Hessian failures and
  nonfinite data now clear results and close the SCF log even with retention.
- DF auxiliary Hessian response was incorrectly Boolean; full response is
  integer 2. Unsupported options are rejected instead of being ignored.
- GPU Newton initialization needed native orthogonalization and device arrays;
  GPU solvent Fock APIs use dm_or_wfn instead of the CPU dm keyword.
- Molecular wrappers delegated to upstream info comparison that could miss or
  mishandle NumPy arrays. Effective snapshots now control the state cache.
- Reorganized documentation contained incorrect model names, MLIP config calls,
  spin conventions and a citation author. Correct prior material was moved to
  dedicated pages and new examples checked against the implemented API.

## Regression suite and real molecular models

- Lightweight environment: **360 passed, 10 skipped**, with 40 slow cases
  deselected. The skips are the explicitly optional real-PySCF tests.
- PySCF environment: **370 passed**, 40 slow cases deselected. This includes
  10 real CPU tests: failed-SCF diagnostics, atomic partial checkpoints,
  separate-process Newton restart, incompatible-physics rejection, projection,
  state changes, derivative cleanup and RHF Hessian/force finite differences.
- Ruff and documentation file/anchor checks passed.
- On Mac CPU, actual OrbMol and MACE-POLAR models were reused through
  OH doublet → OH-minus singlet → OH doublet. Each result matched a newly constructed
  calculator, including forces. OrbMol used orb-models 0.7.0, Python
  3.12.13, Torch 2.14.0; MACE used mace-torch 0.3.16, graph_longrange 0.4.0,
  Python 3.13.14 and Torch 2.13.0.
- OrbMol OH / OH-minus energies: **-2061.055002689 / -2062.792480946 eV**.
- MACE-POLAR OH energy changed from **-2061.068071241** to **-2061.086745760 eV**
  when a NumPy field was changed in place from zero to 0.05 V/Angstrom in z.
  The changed-field result matched a fresh calculation exactly in this test.

## TSUBAME4 CPU / H100 checks

The final sequential compute-node run on **r4n11** passed all **11 checks**.
Environment: ASE 3.28.0, PySCF 2.14.0, GPU4PySCF 1.8.1,
CuPy 13.4.1, pyscf-dispersion 1.5.0, CUDA module 12.8.0.
Tests run under a 15-minute iqrsh allocation, with two CPU threads.

Common settings: def2-SVP, density fitting with def2-universal-jkfit,
DFT grid level 3, NLC atom grid (50,194) without pruning,
SCF energy tolerance 1e-10 Hartree and orbital-gradient tolerance 1e-5.
Hessian settings: grid_response=false, auxbasis_response=2,
conv_tol_cpscf=1e-9. Exact case overrides are in the committed script.

| Case | CPU/GPU energy difference (eV) | Max force difference (eV/Angstrom) | Max Hessian difference (eV/Angstrom²) |
|---|---:|---:|---|
| `water_pbe` | 3.638e-11 | 8.079e-07 | 3.463e-04 |
| `oh_vv10` | 3.275e-06 | 1.451e-05 | CPU unsupported; GPU direct check passed |
| `water_d3zero` | 3.502e-11 | 8.079e-07 | 3.463e-04 |
| `water_d4` | 3.547e-11 | 8.079e-07 | 3.463e-04 |
| `water_cosmo` | 4.457e-11 | 3.406e-06 | 2.530e-04 |
| `water_newton_cosmo` | 1.605e-10 | 4.672e-05 | 2.341e-04 |
| `oh_newton_vv10` | 5.023e-07 | 3.020e-05 | CPU unsupported; GPU direct check passed |
| `water_smd` | 4.093e-11 | 6.784e-06 | 1.928e-04 |
| `rbh_ecp` | 1.364e-12 | 1.018e-08 | 1.066e-05 |

Every supported Hessian was also compared with a direct call on the **same
converged upstream SCF**, separately from cross-device comparisons. The maximum
same-SCF Hessian difference was **5.47e-10 eV/Angstrom²**, below 1e-7; force
comparisons used 1e-8 eV/Angstrom. Supported CPU/GPU Hessian differences were
below 3.47e-4 eV/Angstrom². The cross-device gates were 3e-5 eV for energy,
6e-4 eV/Angstrom for forces and 1e-2 eV/Angstrom² for Hessians.

Additional checks verified CPU→GPU checkpoint transfer, CPU/GPU Newton restart
from an explicitly permitted one-iteration checkpoint, and GPU open-shell
Newton density projection followed by charge-change rejection of the old guess.
Checkpoint write/read errors, source-file preservation and changed-geometry
restart are additionally covered by the local regression tests.

## Limits and numerical interpretation

- **CPU PySCF 2.14 UKS + NLC/VV10 Hessians are unsupported**, with or without
  DF. Tests require the explicit error; they do not call this CPU Hessian a
  successful calculation. Energy/forces and the GPU Hessian are verified.
- D3zero/D4 and SMD retain upstream finite-difference correction terms, with
  their step sizes recorded in metadata. There is no whole-system numerical
  Hessian implementation or implicit CPU fallback in the kit.
- Symmetry is not exact for every upstream Hessian. The symmetry gate is
  1e-3 eV/Angstrom²; no automatic symmetrization hides an upstream discrepancy.
- An additional water/PBE test with **grid_response=true** matched each
  upstream call, but CPU/GPU Hessians differed by **0.04062 eV/Angstrom²**.
  The upstream grid-response implementations differ; the primary comparison
  table uses false explicitly. Do not infer identical vibrational frequencies
  from matching energies or equal option names. Converge grids and SCF/response
  thresholds for the intended research calculation.
- An initially tighter OH/VV10 SCF test did not converge and correctly failed.
  The final benchmark uses the stated 1e-10 / 1e-5 thresholds; no kit default
  was relaxed to make that test pass.
- These tests establish API behavior and numerical agreement under the stated
  conditions, not convergence of Co/Ru research structures or absence of
  imaginary modes. Previous research OrbMol states affected by stale caching
  still require their own recalculation.
