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

## Deferred architectures

- **MatGL CHGNet** is deferred pending
  [upstream issue #834](https://github.com/materialyzeai/matgl/issues/834):
  three-body bond geometry was detached from energy derivatives.
- **MatGL M3GNet** is also deferred. MatGL 4.0.3 direct calls showed force FD
  discrepancies and dependence on supercell representation in both CPU float64
  and H100/CUDA float64. For the strained/displaced Si cell, doubling the cell
  changed the expected doubled energy by 0.01154 eV (PBE) and 0.04637 eV (r2SCAN).
  Experimental remapping of pruned bond indices to parent graph indices removed
  these discrepancies on the tested structures. This was a diagnostic only;
  no upstream physics modifications are included in this package.

Full input structures, raw E/F/S arrays, model hashes, scripts and experimental
diagnostics are retained under the development checkout’s ignored
`temp/matgl_v055/` directory. Numerical experiments are not part of the test suite.
