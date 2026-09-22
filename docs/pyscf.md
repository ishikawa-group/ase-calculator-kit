# Molecular PySCF & GPU4PySCF Guide (0.6.0)

The `pyscf` and `gpu4pyscf` backends provide direct ASE interfaces to molecular Hartree-Fock and Density Functional Theory (DFT) calculations powered by PySCF and GPU4PySCF. Calculations return standard `ase.Calculator` instances with exact unit conversions and explicit CPU/GPU selection.

---

## Configuration Reference

Configurations are passed via dictionaries or YAML files to `get_calculator("pyscf", config=...)`.

### Core Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `basis` | str / dict | *required* | Basis set name (e.g. `"def2-tzvpd"`) or per-element dictionary. |
| `xc` | str | `"pbe"` | Exchange-correlation functional. Omit or set to empty for HF methods. |
| `method` | str | `"auto"` | `"auto"` (RKS for closed-shell, UKS for open-shell), `"rks"`, `"uks"`, `"rhf"`, `"uhf"`. |
| `charge` | int | from atoms | Total system charge. Can be specified in `atoms.info["charge"]`. |
| `multiplicity` / `spin` | int | from atoms | Spin state. `spin = 2S = multiplicity - 1`. Can be set in `atoms.info["spin"]`. |
| `ecp` | str / dict | `None` | Effective core potential (e.g. `{"Ru": "def2-tzvpd"}`). |
| `density_fit` | bool | `False` | Density fitting (RI-J / RI-JK). Recommended on GPUs for large basis sets. |
| `auxbasis` | str | `None` | Auxiliary basis for density fitting (e.g. `"def2-universal-jkfit"`). |
| `disp` | str | `None` | Empirical dispersion: `"d3bj"`, `"d3zero"`, or `"d4"` (requires `pyscf-dispersion`). |
| `nlc` | str / bool | `None` | Non-local correlation (e.g. `"vv10"` for wB97M-V). |
| `grids` | dict | `None` | DFT numerical grids, e.g. `{"atom_grid": [99, 590], "prune": None}`. |
| `conv_tol` | float | `1e-9` | SCF energy convergence tolerance in Hartree. |
| `max_cycle` | int | `200` | Maximum number of SCF iterations. |
| `max_memory` | int | `4000` | Host memory limit in MB. |

---

## 0.6.0 Advanced SCF & Acceleration Features

### SCF Algorithm & Convergence Acceleration

Control SCF convergence behaviour directly from configuration:

```yaml
parameters:
  basis: def2-svp
  xc: pbe
  scf_algorithm: cdiis       # "cdiis" (default) or "newton"
  diis_space: 12             # Subspace dimension for DIIS (default: 8)
  diis_start_cycle: 2        # SCF cycle to begin DIIS extrapolation
  level_shift: 0.1           # Energy shift to unoccupied orbitals in Hartree
  damp: 0.2                  # Density matrix damping factor (0.0 to 1.0)
  init_guess: minao          # "minao", "1e", "atom", "huckel", "vsap"
```

- **Newton Solver (`scf_algorithm: "newton"`)**: Second-order co-iterative solver for difficult open-shell or transition metal systems.

### Wavefunction & Density Reuse (`reuse_density`)

Accelerate geometry optimizations and molecular dynamics by projecting converged molecular orbitals across coordinate steps:

```yaml
parameters:
  basis: def2-tzvpd
  xc: wb97m_v
  density_fit: true
  reuse_density: true        # Projects MOs from previous step via project_mo_nr2nr
```

When atomic positions shift slightly, the calculator projects MO coefficients onto the new atomic basis, significantly reducing the required SCF cycles. If geometry shifts drastically or atom types change, it gracefully falls back to the configured `init_guess`.

### Checkpoint Management (`checkpoint`)

Save and restart calculations from standard PySCF HDF5 checkpoint files, augmented with kit metadata validation:

```yaml
# Save checkpoint upon convergence
parameters:
  basis: def2-tzvpd
  xc: pbe
  checkpoint:
    write: "/path/to/water_chk.h5"

# Restart or initialize from checkpoint
parameters:
  basis: def2-tzvpd
  xc: pbe
  checkpoint:
    read: "/path/to/water_chk.h5"
    allow_unconverged: false
```

- **Atomic File Writing**: Checkpoints are written to a unique PID-tagged temporary file and atomically renamed upon completion to prevent file corruption.
- **Compatibility Verification**: The kit validates that the saved basis, functional, charge, and spin match the active configuration before reading.

---

## Implicit Solvation Models (PCM, COSMO, SMD)

Support for continuum dielectric environments with fine-grained control:

```yaml
parameters:
  basis: def2-svp
  xc: pbe
  solvent:
    model: pcm               # "pcm" or "smd"
    method: COSMO            # "IEF-PCM" (default), "C-PCM", "COSMO", "SS(V)PE"
    eps: 78.3553             # Dielectric constant (e.g. water)
    vdw_scale: 1.1           # Van der Waals radius scaling factor
    r_probe: 0.4             # Solvent probe radius in Angstrom
    radii:                   # Optional per-element atomic radii in Angstrom
      H: 1.20
      O: 1.52
    lebedev_order: 29        # Surface discretization grid order
```

---

## Hessian API (`get_hessian`)

Retrieve the full Cartesian Hessian matrix (`(3N, 3N)` in `eV/Å²`):

```python
calc = get_calculator("pyscf", config=...)
atoms.calc = calc

hessian = calc.get_hessian(atoms)  # Shape: (3*N, 3*N), unit: eV/Å^2
```

- **Analytical Hessians**: Available for RHF, RKS, UHF, UKS (without dispersion).
- **Automated Hybrid Finite-Difference**: When empirical dispersion (`d3bj`, `d3zero`, `d4`) or SMD solvation is requested, the calculator evaluates the analytical electronic Hessian and augments it with finite-difference gradients of the dispersion/CDS terms.

---

## Structured Diagnostics (`calc.metadata["scf"]`)

Every calculation populates detailed diagnostics in `calc.metadata["scf"]`:

- `converged`: Boolean convergence indicator.
- `cycles`: Number of SCF iterations taken.
- `scf_algorithm`: Active convergence accelerator.
- `spin_analysis`: For open-shell systems:
  - `s2`: Expectation value $\langle S^2 \rangle$.
  - `ideal_s2`: Theoretical $S(S+1)$.
  - `s2_deviation`: Spin contamination $\langle S^2 \rangle - S_{\mathrm{ideal}}(S_{\mathrm{ideal}}+1)$.
- `mulliken_spin_population`: Net Mulliken spin per atom.
- `energy_components`: Electronic, nuclear repulsion, and xc energy breakdown.
- `failure_stage`: Detailed stage info if convergence fails.
