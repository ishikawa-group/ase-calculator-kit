# TensorNet MatPES validation

Validated on 2026-09-15 for the TensorNet PBE and r2SCAN backends. These checks
verify calculator integration and derivative consistency on small periodic cells;
they are not a benchmark of prediction accuracy against DFT.

## Models and environments

| Model | Hugging Face revision |
|---|---|
| `TensorNet-PES-MatPES-PBE-2025.2` | `7ac2b90400a87eca7f44c1e7956dca6b1c85e418` |
| `TensorNet-PES-MatPES-r2SCAN-2025.2` | `a9934dd4311849373230c253c74c6eab2519f867` |

The weights were loaded from the official `materialyze` Hugging Face repositories.
The package accepts `revision=` to reproduce these revisions.

- Mac: Apple M5, macOS 26.6.2, Python 3.13.14, PyTorch 2.14.0, MatGL 4.0.3,
  PyG 2.8.0.post1, ASE 3.28.0, torch-dftd 0.5.3, SevenNet 0.13.0.
- CUDA: TSUBAME4 NVIDIA H100 MIG 3g.47gb, Python 3.12.11, PyTorch 2.13.0+cu130,
  MatGL 4.0.3, PyG 2.8.0.post1, ASE 3.28.0, torch-dftd 0.5.3; TF32 disabled.
- MPS: CPU fallback disabled; model parameters confirmed on MPS. TensorNet is
  converted to float32 before transfer. The D3 component runs on CPU.

## Checks

Cu (fcc, 4 atoms), Si (diamond, 2 atoms), and NaCl (rocksalt, 2 atoms) were
strained and one atom displaced to avoid relying on zero forces at symmetry.
Each TensorNet checkpoint was evaluated with and without D3: 12 cases per device.
SevenNet-Omni `matpes_pbe` and `matpes_r2scan` were evaluated on the identical
structures with and without D3 on CPU/MPS (12 additional cases).

All TensorNet and SevenNet comparison cases passed the following applicable checks:

- Raw ASE results: scalar total-cell energy in eV, forces `(N, 3)` in eV/Å,
  stress `(6,)` in eV/Å³, with Voigt order `xx, yy, zz, yz, xz, xy`; all finite.
- Force-consistent energy equals energy.
- Kit output matches MatGL output; upstream GPa stresses convert with ASE `units.GPa`.
- TensorNet energy doubles on `(2, 1, 1)` replication; corresponding forces and
  intensive stresses are unchanged within float32 tolerance.
- D3 totals equal the sum of the underlying calculator and D3 results. Matching
  PBE/r2SCAN corrections use BJ damping, 14 Å cutoff and polynomial smoothing.
- All force components and six stress components agree with central finite
  differences. Displacements: 0.002 Å; strain: 0.001. Absolute tolerances:
  0.005 eV/Å for forces and 0.0001 eV/Å³ for stress.
- CPU/MPS and Mac-CPU/H100-CUDA agreement: absolute tolerances 0.0002 eV,
  0.0002 eV/Å and 0.00002 eV/Å³ for E/F/S. Different models are not expected
  to predict identical energies or forces.

## Measured maxima

The finite-difference columns below include both D3 settings and all three
structures on Mac CPU; each row contains six cases.

| Model | Max force FD error (eV/Å) | Max stress FD error (eV/Å³) | Max CPU/MPS force difference (eV/Å) |
|---|---:|---:|---:|
| tensornet matpes-pbe | 0.00147 | 6.99e-05 | 1.61e-06 |
| tensornet matpes-r2scan | 0.00166 | 6.16e-05 | 1.26e-06 |
| sevennet matpes-pbe | 0.000399 | 2.16e-05 | 7.9e-07 |
| sevennet matpes-r2scan | 0.00212 | 5.56e-05 | 5.07e-07 |

Across TensorNet checkpoints, the maximum CPU/MPS differences were
3.81×10⁻⁶ eV (E), 1.61×10⁻⁶ eV/Å (F), and 1.99×10⁻⁷ eV/Å³ (S).
The maximum Mac-CPU/H100-CUDA differences were 2.78×10⁻⁶ eV,
1.37×10⁻⁶ eV/Å, and 1.21×10⁻⁷ eV/Å³. The CUDA job completed successfully
in about 59 seconds, including a separate M3GNet diagnostic.

## Provisional MatGL CHGNet in 0.5.6

`matgl-chgnet` provides the existing PBE/r2SCAN 2025.2.10 checkpoints with
three-body geometry gradients restored. It is **not a retrained model**.
This explicitly authorized exception to the thin-factory policy is isolated in
`backends/mlip/matgl_chgnet.py` and requires the unmodified MatGL 4.0.3 source.
The helper checks the SHA-256 of `CHGNet.forward`, removes exactly the one
`torch.no_grad()` block around `create_directed_line_graph`, and binds the new
method to the loaded instance. Other instances, module globals and weights
are unchanged; an unknown version/source is rejected before loading weights.
The upstream fix is [PR #835](https://github.com/materialyzeai/matgl/pull/835).

Default Hugging Face revisions:

| Model | Frozen revision |
|---|---|
| CHGNet-PES-MatPES-PBE-2025.2.10 | `4ec3c2c323cde07a31ca99d6719cca45919b5e5f` |
| CHGNet-PES-MatPES-r2SCAN-2025.2.10 | `4447783f53387df4642d13062cafc9571f1668d2` |

The returned PESCalculator records the resolved model, revision and temporary
correction in its `parameters`. An explicit `revision=` overrides the default;
only the frozen revisions above have been validated here.

**After upstream retrained models are published and validated, a future kit
release will replace this temporary implementation and document the changed
weights and behavior.** The current release never silently switches to new
weights. Record kit and MatGL versions as well as HF revision to reproduce runs.

The release checks use the same displaced/strained Cu, Si and NaCl structures,
D3 settings, finite-difference steps and tolerances as above. All 12 CPU/MPS
cases passed with MPS fallback disabled; the D3 term uses CPU on MPS.

| Model | Max force FD error (eV/Å) | Max stress FD error (eV/Å³) | Max CPU/MPS force difference (eV/Å) |
|---|---:|---:|---:|
| matgl-chgnet matpes-pbe | 0.000260 | 1.77e-05 | 2.65e-06 |
| matgl-chgnet matpes-r2scan | 0.000837 | 2.65e-05 | 2.38e-06 |

All 12 cases also passed on H100 MIG 3g.47gb / CUDA with TF32 disabled.
The maximum Mac-CPU/CUDA differences were 9.54e-07 eV,
1.07e-06 eV/Å and 5.42e-08 eV/Å³. CUDA finite-difference
errors were at most 0.000838 eV/Å (force) and 2.55e-05 eV/Å³
(stress). The validation process completed in about 34 seconds.

These checks cover total-cell energy, F `(N,3)`, ASE Voigt stress `(6,)`,
finite differences, supercell scaling and D3 addition. Numerical inputs,
raw outputs, scripts and hashes are retained under ignored `temp/matgl_v056/`.

Earlier H100/CUDA experiments with the same weights and removal of this block
found improved MatPES force/stress accuracy and NVE energy conservation.
Results varied for elastic/adsorption properties; the 5,651-reaction D3
adsorption comparison had worse mean energy errors after the correction.
Energy-gradient consistency is therefore not a promise of universal accuracy
improvement. Adsorption references also differ from the model functionals.
See development artifacts under `temp/matgl_chgnet_oss_bench_20260916/` and
`temp/matgl_chgnet_catbench_d3_20260917/` for the scope and raw results.

## ASE PBC compatibility correction (unreleased)

MatGL 4.0.3's usual `Atoms2Graph` path checks `atoms.pbc.all()` and treats
partial PBC as fully nonperiodic. The shared `_matgl_pbc.py` adapter now uses
ASE neighbor lists, periodic image offsets and the actual cell for partial
and nonperiodic inputs. Full-PBC graph construction remains upstream's.
The adapter preserves the supplied atoms and does not modify model weights,
the provisional CHGNet gradient correction, or D3 parameters.

Missing nonperiodic cell vectors are completed internally for energy/forces,
including torch-dftd. Stress is unavailable without a physical cell volume;
fully nonperiodic torch-dftd also does not provide stress. A supplied vacuum
cell uses its full volume. Nonperiodic MatGL without D3 now uses the supplied
volume for stress, correcting the upstream unit-volume normalization.

Validation on 2026-09-17 used MatGL 4.0.3, ASE 3.28.0, torch 2.14.0 and
torch-dftd 0.5.3 on Apple Silicon CPU/MPS, with MPS fallback disabled:

- Actual graph comparison with ASE: all 8 PBC masks × 3 boundary-crossing
  directions, 24/24 passed. Fast regression tests also cover skew cells,
  periodic images of the same atom, missing cell vectors and stress caching.
- TensorNet/CHGNet × PBE/r2SCAN × D3 off/on × 8 masks × CPU/MPS: 128 baseline
  calculations, with periodic translations and cell replication checks.
- Missing open-axis cell vectors: 48 energy/force cases passed; stress was
  correctly unavailable. The input atoms were unchanged.
- Float64 finite differences on 24 nonperiodic/1D/2D systems, using coordinate
  and strain steps of `1e-5` and `5e-6`: maximum force error **9.35e-8 eV/Å**
  and stress error **3.55e-7 eV/Å³**. A coarse 0.002 Å step on one CHGNet case
  was not converged; the smaller steps resolved that numerical discrepancy.

Maximum absolute differences across the tested models and D3 settings:

| Comparison | E (eV) | F (eV/Å) | S (eV/Å³) |
|---|---:|---:|---:|
| Full PBC versus upstream conversion | 3.81e-6 | 1.79e-7 | 3.73e-8 |
| Cu(111), 20 Å vacuum gap: partial versus full PBC | 3.81e-6 | 3.73e-7 | 6.05e-9 |
| CPU versus MPS | 3.81e-6 | 2.26e-6 | 1.42e-7 |

The fast suite passed 305 tests; ruff passed. CUDA was not rerun for this input
adapter change. Scripts, raw outputs, fixed HF revisions, source hashes and
the Japanese report are in ignored `temp/matgl_pbc_fix_20260917/`.
Earlier partial-PBC MatGL results require recomputation. The archived CatBench
inputs were all fully periodic, including in the original source data, so
that benchmark is unaffected by this partial-PBC defect.

## Deferred architecture

- **MatGL M3GNet** remains deferred. MatGL 4.0.3 direct calls showed force FD
  discrepancies and dependence on supercell representation in both CPU float64
  and H100/CUDA float64. For the strained/displaced Si cell, doubling the cell
  changed the expected doubled energy by 0.01154 eV (PBE) and 0.04637 eV (r2SCAN).
  Experimental remapping of pruned bond indices to parent graph indices removed
  these discrepancies on the tested structures. This was a diagnostic only;
  no M3GNet physics modifications are included in this package.

Full input structures, raw E/F/S arrays, model hashes, scripts and experimental
diagnostics are retained under the development checkout’s ignored
`temp/matgl_v055/` directory. Numerical experiments are not part of the test suite.
