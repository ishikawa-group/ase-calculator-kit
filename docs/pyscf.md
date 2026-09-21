# Molecular PySCF / GPU4PySCF

The `pyscf` and `gpu4pyscf` names are config-only DFT factory backends. They
return an ASE calculator implemented here as a unit/input adapter; all quantum
chemistry comes from upstream PySCF and GPU4PySCF. No external executable or
`profile.command` is needed. CPU and GPU are explicit choices, never automatic
fallbacks. No torch dependency is introduced.

## Configuration

The top level accepts `calculator`, optional `directory` (default `.`), and
`parameters`. Unknown keys are errors. `overrides` deep-merges as for VASP/QE.
`write_resolved_config=True` records effective defaults and spin in
`resolved_calculator_config.yaml`. Change conditions through the factory and
`overrides`, creating a new calculator; `calc.set()` with parameters is rejected.

| Parameter | Meaning / default |
| --- | --- |
| `basis` | Required PySCF basis name or element-to-basis mapping |
| `charge` | Integer total charge; may instead come from atoms.info; positive means electrons removed |
| `multiplicity` / `spin` | May instead come from atoms.info; YAML `spin = Nalpha-Nbeta = multiplicity-1`; both must agree |
| `method` | `auto` (RKS for spin 0, UKS otherwise), `rks`, `uks`, `rhf`, `uhf` |
| `xc` | PySCF functional name, default `pbe`; omit for HF |
| `ecp` | Explicit PySCF ECP name or element mapping; default none |
| `density_fit` | Boolean, default false |
| `auxbasis` | Auxiliary basis, default upstream automatic; requires density fitting |
| `nlc` | `null`: upstream automatic; `vv10`: explicit; `false`: disabled |
| `disp` | `null`, `d3bj`, or `d4`; requires pyscf-dispersion |
| `grids` / `nlcgrids` | Mapping of `level` (0–9), `atom_grid: [radial, angular]`, `prune: null` |
| `conv_tol` | SCF energy convergence tolerance, default `1e-9` Eh |
| `max_cycle` | Maximum SCF cycles, default 200 |
| `max_memory` | PySCF host-memory setting in MB, default 4000; not a hard process or VRAM limit |
| `verbose` | PySCF logging level 0–9, default 4 |
| `solvent` | `null` (gas), `{model: smd, solvent: water}`, or `{model: pcm, eps: 78.3553}` |

PCM additionally accepts `method`: `IEF-PCM` (default), `C-PCM`, `COSMO`, or
`SS(V)PE`. These are upstream models, not an electrolyte/potential treatment.
Grid pruning is left at the upstream default when omitted; `prune: null`
disables it. A global `[radial, angular]` grid pair is supported, not per-element
grid mappings or Python pruning callables. Arbitrary executable Python objects
are not accepted in YAML. Invalid upstream basis/XC/grid choices remain errors.

Electronic state is resolved from `atoms.info["charge"]` (total charge) and
`atoms.info["spin"]` (**multiplicity**, as in UMA/OrbMol). YAML `spin` instead
means multiplicity minus one. YAML `multiplicity` is the same quantity as
`atoms.info["spin"]`. Each field may come from either source; when present in
both, values must agree after conversion or a `ValueError` is raised before
calculation or cache reuse. Missing state is an error at calculation time; no
neutral/singlet defaults are inserted. Integers (including NumPy integers) are
accepted; booleans and floats are rejected. Initial atomic charges/magnetic
moments are never used to infer the total electronic state.

Changing only `atoms.info` charge/spin invalidates the cache even when geometry
is unchanged. The input Atoms is not modified. Effective state and its sources
are available in `calc.metadata`; when requested, the resolved YAML is updated
at calculation time to include the actual charge/spin/multiplicity. Before the
first calculation, an atoms-driven resolved YAML can still lack these fields.
 Electron-count parity is checked by PySCF and by the adapter
after ECP construction. Spin-zero UKS/UHF can be selected explicitly, but this
release does not expose custom initial density matrices or broken-symmetry
occupation control. Convergence is not proof of the lowest-energy electronic
state, especially for transition metals.

Only nonperiodic molecules/clusters are supported: any true component of
`atoms.pbc` raises. A nonperiodic ASE cell is allowed and does not turn the
molecule into a periodic calculation. Properties are energy (eV) and forces
(eV/Angstrom), with `force = -gradient * Hartree / Bohr`. Stress, Hessians,
post-HF methods and periodic k-points are not implemented. SCF failures,
nonfinite energies/gradients and unavailable GPU features propagate as errors.

Request forces first when both energy and forces are needed: the energy is
then cached from the same SCF. Asking for energy first avoids unnecessary
gradient work, but a subsequent force request repeats the SCF. Atomic geometry
changes invalidate ASE's cache. SCF runs start from upstream initial guesses;
this release does not reuse density matrices between optimization steps.
`pyscf.log` is appended in the calculation directory and is closed on failure.
Use separate directories for concurrent calculations and different states.

## Functional, dispersion, ECP and solvent choices

The supplied `pyscf_wb97mv.yaml` and `gpu4pyscf_wb97mv.yaml` reproduce the
requested omegaB97M-V/def2-TZVPD + VV10 settings: density fitting with
`def2-universal-jkfit`, `(99,590)` unpruned XC grids, `(50,194)` NLC grids,
`conv_tol=1e-9`, `max_cycle=200`. NLC pruning is left at the upstream default.

D3(BJ)/D4 is applied through PySCF's `mf.disp`, including its analytic gradient;
it does not use the MLIP torch-D3 wrapper. Do not put a dispersion suffix in
`xc`; use `disp`. A VV10/NLC functional combined with D3/D4 is rejected to avoid
double counting, even if `nlc: false` is explicitly supplied. The supplied
`pyscf_pbe_d3.yaml` demonstrates PBE-D3(BJ); change `disp` to `d4` for PBE-D4.

For Ru, the ECP is explicit, e.g. `ecp: {Ru: def2-tzvpd}`. Merely choosing a
def2 basis does not instruct the adapter to add an ECP. The Ru example's charge
is illustrative: set charge/multiplicity for the actual species.

`gpu4pyscf_wb97mv_smd.yaml` adds SMD water. Keep gas-phase and solvated results
in different directories. SMD total energies contain solvent contributions;
do not mix them directly with gas-phase MLIP energies. GPU4PySCF can execute
some components (including ECP and dispersion) on CPU internally; selecting the
GPU backend means the SCF object is GPU4PySCF, not that every operation is on GPU.

## Installation and compatibility

Use `[pyscf]` for CPU, `[gpu4pyscf-cuda12x]` for NVIDIA CUDA 12 on Linux x86_64,
and `[pyscf-dispersion]` for CPU D3/D4. GPU4PySCF also depends on the dispersion
extension upstream. All are outside the MLIP `all` extra. Package requirements
are compatible ranges; exact validation versions are in `constraints.txt`.

Validated stack: PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 13.4.1, cuTENSOR 2.2.0,
pyscf-dispersion 1.5.0, ASE 3.28.0, Python 3.12, CUDA Toolkit 12.8. The GPU extra
is resolvable on Python 3.12/3.13; CuPy's validated wheels exclude 3.14. CPU
extras resolve on Linux 3.12–3.14, but numerical validation is on 3.12.
No environment marker silently removes a requested backend.

CuPy 14.2.0 with cuTENSOR 2.8.0 fell back to CuPy tensor contractions in the
initial probe. The published extra uses the upstream-recommended 13.4.1/2.2.0
pair instead. CPU PySCF also works on Apple Silicon in the basic smoke test.
The macOS pyscf-dispersion 1.0.0 wheel failed to import with current PySCF/NumPy;
it is below the supported >=1.5 range. CUDA 11/13, multiple GPUs and Windows
are not validated by this release.

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
