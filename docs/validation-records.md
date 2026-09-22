# Validation Records

This document summarizes hardware-level validation records, parity benchmarks between CPU and GPU backends, and numerical consistency checks conducted on supercomputing infrastructure.

---

## TSUBAME4 (Tokyo Tech / Science Tokyo ISCT)

### Hardware & Software Environment
- **Node**: TSUBAME4 Compute Node (`r4n11`)
- **GPU**: NVIDIA H100 PCIe (MIG 3g.47gb / 4g.47gb partitioned)
- **CUDA Toolkit**: 12.8.0
- **Environment**: Python 3.12.11, PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 13.4.1, cuTENSOR 2.2.0, pyscf-dispersion 1.5.0, ASE 3.28.0

---

### 1. Parity Benchmarks: PySCF (CPU) vs GPU4PySCF (NVIDIA H100)

| System | Method / Functional | Basis / Grids / Dispersion | $\Delta E$ (Hartree) | $\text{Max } \Delta F$ (au) | Status |
|---|---|---|:---:|:---:|:---:|
| $\text{H}_2\text{O}$ | RKS $\omega\text{B97M-V}$ + VV10 | def2-TZVPD / (99,590) / RI-JK | $3.13 \times 10^{-8}$ | $2.10 \times 10^{-7}$ | ✅ Passed |
| $\text{OH}^\bullet$ | UKS $\omega\text{B97M-V}$ + VV10 (doublet) | def2-TZVPD / (99,590) / RI-JK | $2.93 \times 10^{-8}$ | $5.35 \times 10^{-8}$ | ✅ Passed |
| $\text{H}_2\text{O}$ | RKS PBE-D3(BJ) | def2-SVP / RI-J | $2.37 \times 10^{-12}$ | $2.01 \times 10^{-7}$ | ✅ Passed |
| $\text{H}_2\text{O}$ | RKS PBE-D3(0) | def2-SVP / RI-J | $2.37 \times 10^{-12}$ | $2.01 \times 10^{-7}$ | ✅ Passed |
| $\text{H}_2\text{O}$ | RKS PBE + PCM (UFF radii, $r_{\text{probe}}=0.4$ Å) | def2-SVP / RI-J / $\varepsilon=78.3553$ | $1.59 \times 10^{-12}$ | $1.25 \times 10^{-7}$ | ✅ Passed |

- **Tolerance Criteria**: Energy difference $\Delta E \le 1 \times 10^{-6}\,\text{Ha}$, Force difference $\Delta F \le 1 \times 10^{-5}\,\text{au}$.
- **Observations**: Both closed-shell and open-shell systems match within sub-microhartree precision across CPU and GPU implementations.

---

### 2. 0.6.0 Feature Verification on GPU4PySCF

#### Cartesian Hessian API (`calc.get_hessian(atoms)`)
- **Matrix Dimension**: $(9, 9)$ for $\text{H}_2\text{O}$ ($(3N, 3N)$ Cartesian coordinates in $x, y, z$).
- **Symmetry Error**: $\|H - H^T\|_{\infty} = 0.00\,\text{eV/Å}^2$ (exact matrix symmetry).
- **GPU vs CPU Analytical Hessian Deviation**: $\|H_{\text{GPU}} - H_{\text{CPU}}\|_{\infty} = 3.76 \times 10^{-3}\,\text{eV/Å}^2$.
- **Hybrid Finite-Difference**: Dispersion (D3BJ / D3zero) and implicit solvation (SMD) contributions successfully combined via finite-difference gradient steps.

#### Checkpoint Management (`checkpoint: {write: ..., read: ...}`)
- **Atomic File Serialization**: Written to temporary PID-stamped files and atomically committed.
- **Kit Metadata Embedding**: Embedded directly into the PySCF HDF5 root group `/ase_calculator_kit/attrs/metadata_json`.
- **Restart Verification**: Restarting an identical geometry from checkpoint reproduced the potential energy with $\Delta E = 3.59 \times 10^{-11}\,\text{eV}$.

#### Wavefunction & Density Reuse (`reuse_density: true`)
- **First Step**: Initial SCF converged from scratch in 7 iterations.
- **Coordinate Perturbation**: Geometry perturbed by $\Delta x = +0.01$ Å.
- **Second Step**: Projected molecular orbitals via `project_mo_nr2nr` reached full convergence in 6 iterations without SCF oscillations.

#### SCF Convergence Algorithm Control (`scf_algorithm: "cdiis"`)
- CDIIS subspace dimension control (`diis_space: 10`) validated on GPU4PySCF, reaching target convergence in 7 cycles.

#### Open-Shell Spin Diagnostics
- Evaluated on doublet $\text{OH}^\bullet$ molecule:
  - $\langle S^2 \rangle_{\text{calculated}} = 0.7516$
  - $\langle S^2 \rangle_{\text{ideal}} = 0.7500$
  - Spin Contamination: $\Delta \langle S^2 \rangle = 0.0016$
  - Net Mulliken Spin Population: $\text{O} = +1.025$, $\text{H} = -0.025$ (total spin = $1.000$).
