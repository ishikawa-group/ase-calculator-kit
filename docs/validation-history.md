# Validation before 0.6.0

These checks describe their stated versions, not every new 0.6.0 path.

## Validation

Numerical validation uses small systems on TSUBAME4 H100 MIG via `iqrsh`, with
two CPU threads. This shared allocation is a correctness check, not a speed
benchmark. Acceptance limits are `1e-6` Eh for CPU/GPU energy differences and
`1e-5` Eh/Bohr for the largest force-component difference. The finite-difference
check uses a 0.001 Angstrom central displacement and a `1e-4` eV/Angstrom limit.

The unit suite adds only four cases for boundary behavior: electronic-state
validation, nonperiodic scope, unit conversion/cache invalidation, and SCF
failure (with a dispersion double-count check). Existing fast tests and lint
remain release gates. Numerical records below are from actual upstream calls,
not mocked physics.

Recorded on 2026-09-22 (final validated dependency stack):

| Case | CPU/GPU energy difference (Eh) | Maximum force difference (Eh/Bohr) |
| --- | ---: | ---: |
| H2O, omegaB97M-V/def2-TZVPD | 3.129e-08 | 2.079e-07 |
| OH doublet, omegaB97M-V/def2-TZVPD | 2.808e-08 | 6.854e-08 |
| H2O, same + SMD water | 2.833e-08 | 5.813e-07 |
| H2O, PBE/def2-SVP + PCM | 1.705e-12 | 1.025e-07 |
| H2O, PBE/def2-SVP + D3(BJ) | 2.402e-12 | 2.011e-07 |
| H2O, PBE/def2-SVP + D4 | 2.373e-12 | 2.011e-07 |
| RuO4, PBE/def2-TZVPD + Ru def2 ECP | 1.137e-11 | 1.624e-07 |
| H2O, RHF/STO-3G | 8.512e-12 | 1.478e-06 |
| OH doublet, UHF/STO-3G | 6.111e-13 | 4.455e-09 |

The independent CPU call differed by 1.592e-12 eV in energy and 4.247e-12 eV/Angstrom in forces. The central finite difference error was 2.089e-05 eV/Angstrom.

H2O/OH inputs use ASE 3.28 molecule geometries. RuO4 uses Ru at the origin and O at (0.98,0.98,0.98), (0.98,-0.98,-0.98), (-0.98,0.98,-0.98), (-0.98,-0.98,0.98) Angstrom, charge/spin 0. Density fitting is used for DFT. The initial RuH2(+2) probe failed SCF convergence on both CPU and GPU and correctly raised; its energies were not accepted as results.

ASE BFGS on H2O PBE/STO-3G reached 0.05 eV/Angstrom in five steps. Limiting it to one step produced success=false and process exit 1. Validation files/logs remain under TSUBAME4 temp/ase-calculator-kit-0.5.9-pyscf; no calculation outputs are shipped in the package.


The added atoms.info path was separately tested on H2 with (charge,
multiplicity) = (0,1), (0,3), (1,2), reusing the same calculator and coordinates.
CPU/GPU results agreed within the limits above, every state changed the energy,
and cached calls rejected both charge and multiplicity conflicts with YAML.
The example was also exercised from extxyz through optimization, resolved YAML,
JSON/NPZ output and final extxyz, retaining charge/multiplicity.

## 0.5.10 SCF reuse validation

For H2O with omegaB97M-V/def2-SVP, density fitting, VV10 and SMD water,
CPU and H100 GPU runs requested energy first, then forces while disabling
new RKS construction. Both succeeded. Forces agreed with a direct gradient
call on that same SCF to 0 (CPU) and 3.66e-13 eV/Angstrom (GPU).
Independent fresh SCFs differed by 4.68e-8 and 1.70e-8 eV/Angstrom, respectively;
these use a 1e-6 eV/Angstrom repeat-SCF tolerance, separate from the 1e-8
same-SCF gradient tolerance. Energy differences were below 2.3e-12 eV.
Changing geometry created a new SCF and closed the old stream; reset and
failure cleanup are also covered by the regression tests.

Molecular MACE was verified on Mac CPU with mace-torch 0.3.16 and
`graph_longrange` 0.4.0: H2O singlet, OH doublet, and OH-minus singlet returned
finite energies/forces, and missing electronic state was rejected. H2O energies
were -2079.863496759 eV (OMOL-0) and -2079.862579249 eV (POLAR-1 medium).
These are API checks, not a model-accuracy benchmark. Small/large POLAR loader
routing is covered by unit tests; the new numerical check used medium.
