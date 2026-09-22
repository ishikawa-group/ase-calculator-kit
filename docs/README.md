# ase-calculator-kit Documentation

Welcome to the documentation for **ase-calculator-kit**. This kit provides a thin, unified ASE calculator factory for machine-learning interatomic potentials (MLIPs) and molecular DFT backends.

## Documentation Index

- **[Getting Started](getting-started.md)**: Installation options, Python version support, isolated environment instructions (e.g. MACE), and basic usage.
- **[Backends Reference](backends.md)**: Detailed configuration, parameter tables, and examples for all supported MLIP and DFT engines (SevenNet, CHGNet, MatGL, MatterSim, NequIP, OrbMol, UMA, eSEN, MACE, VASP, Quantum ESPRESSO).
- **[PySCF & GPU4PySCF Guide](pyscf.md)**: Complete guide to molecular DFT and Hartree-Fock calculations, SCF algorithms (CDIIS, Newton), density reuse, checkpoints, diagnostics, implicit solvation (PCM, SMD), dispersion corrections, and Hessian computation.
- **[Dispersion Policies](models.md)**: Ground-truth reference for DFT-D3/D4 dispersion compatibility across foundation MLIP models.
- **[Validation Records](validation-records.md)**: Benchmarks and parity verification results across CPU and GPU hardware (including TSUBAME4 supercomputer records).
- **[Developer & Release Guide](releasing.md)**: Release checklists and packaging instructions.
- **[Architecture Guide (Japanese)](code-guide_ja.md)**: Deep dive into design philosophy, state tracking, and lifecycle management.
