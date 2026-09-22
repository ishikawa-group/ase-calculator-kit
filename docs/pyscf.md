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

Energy and forces share one converged SCF at the same geometry/electronic state,
regardless of request order. The SCF object and output stream are retained after
an energy-only request, and released when forces complete, a calculation fails,
or `calc.reset()` is called. Atomic or electronic-state changes invalidate the
SCF. Runs at a new geometry still start from upstream initial guesses; this
release does not reuse density matrices between optimization steps.
`pyscf.log` is appended in the calculation directory.
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

## SCF controls (0.6.0)

Additional keys live under `parameters`; the factory remains config-only.
Existing YAML and defaults remain valid. No solver or device fallback is automatic.

| Setting | Default | Meaning |
|---|---|---|
| `scf_algorithm` | `cdiis` | `cdiis` or `newton` |
| `init_guess` | `minao` | `minao`, `atom`, `1e`, `huckel`, `mod_huckel` |
| `conv_tol_grad` | upstream | Positive orbital-gradient threshold |
| `diis_space` | upstream | Positive integer history size; CDIIS only |
| `diis_start_cycle` | upstream | Nonnegative integer; CDIIS only |
| `damp` | upstream | 0 <= value < 1; CDIIS only |
| `level_shift` | upstream | Nonnegative Hartree shift; CDIIS only |
| `reuse_density` | false | Project the last compatible converged orbitals across geometries |
| `retain_scf` | false | Retain the converged SCF after forces/Hessian for further derivatives |

Newton rejects DIIS/damping/level-shift settings it does not use. Target-device
orbitals initialize the Newton solver; final energy, forces and Hessian use its
converged state. For NLC functionals, upstream's Newton orbital-Hessian response
omits NLC response; this is recorded in `newton_notes`. The full functional is
still used for energy and nuclear derivatives.

`reuse_density: true` uses upstream MO projection and AO-metric normalization.
It is an initial guess, not a cached energy at a new geometry. A saved snapshot
of atom order, resolved basis/ECP, charge, multiplicity, method, functional,
grids, DF and solvent settings controls compatibility. An incompatible state
uses the configured fresh guess and records `density_reuse_fallback`. Reuse
does not guarantee fewer iterations or the same SCF solution for difficult
systems. `reset()` clears both the result cache and the reusable orbitals.

## Checkpoints and SCF restart

```yaml
calculator: gpu4pyscf
directory: runs/water_restart
parameters:
  basis: def2-svp
  xc: pbe
  charge: 0
  multiplicity: 1
  density_fit: true
  checkpoint:
    read: /absolute/path/previous/wavefunction.chk
    write: wavefunction.chk
    allow_unconverged: false
  diagnostics:
    save: true
    iterations: true
```

Read/write are optional, but must point to different files. Relative checkpoint
paths resolve against the calculation directory. The read file is unchanged.
Each completed SCF iteration writes an atomic snapshot with `converged: false`;
a successful final SCF replaces it with a converged snapshot. A write failure
is an error, not a silent loss of restart data. Forced process termination can
lose the current iteration; the last complete snapshot remains readable.

Files contain standard PySCF `mol` and `scf` datasets plus required kit
provenance. Raw upstream checkpoints without that provenance are rejected,
because convergence and physical compatibility cannot be verified. Missing,
corrupt, nonfinite, incompatible and unapproved unconverged inputs raise before
SCF. Set `allow_unconverged: true` explicitly to resume a partial snapshot.
Changing only geometry is allowed: orbitals are projected and normalized.
CPU/GPU transfer is supported. Changing charge/spin, basis/ECP, functional,
solvent, grids or DF settings requires a new initial guess.

An explicit read initializes the first successful SCF of this calculator;
subsequent geometries use `reuse_density` when enabled. SCF restart does not
restore a geometry optimizer, trajectory or scheduler job.

## D3zero and PCM/COSMO details

`disp` accepts `d3bj`, `d3zero`, `d4`, and a suffix specifying the dispersion
parameter functional, for example `d3zero:b3lyp` with `xc: b3lyp5`. The same
specification reaches energy, gradient and Hessian. Additional D3/D4 remains
forbidden when VV10/NLC is active.

PCM uses explicit `eps` and accepts these additional solvent keys:

| Key | Meaning |
|---|---|
| `equilibrium_solvation` | Explicit upstream solvent-response setting |
| `lebedev_order` | Surface Lebedev grid order supported by upstream |
| `surface_method` | `SWIG` or `ISWIG`, if supported by the selected implementation |
| `vdw_scale` | Positive dimensionless multiplier of modified Bondi radii; default 1.2 |
| `r_probe` | Nonnegative additional radius in Angstrom; default 0 |
| `radii` | Element-symbol to final cavity radius in Angstrom |

For example, `radii: {O: 1.6}` overrides O's final radius after the default table
is scaled and the probe radius added. It receives no second scaling. The kit
constructs the table in Bohr explicitly, avoiding CPU/GPU differences in probe
units; `effective_radii_angstrom` records the actual radii. Unknown elements,
invalid units/values and unsupported surface methods raise. SMD uses its own
parameterization and accepts only `model: smd` and `solvent: <name>`.
PySCF COSMO settings do not establish equivalence to a different code's cavity.

## Cartesian Hessian

```python
atoms.calc = calc
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
hessian = calc.get_hessian(atoms)  # NumPy (3N, 3N), eV/Angstrom^2
calc.reset()
```

Rows and columns are atom-major x/y/z, matching ASE's
`VibrationsData.from_2d(atoms, hessian)`. Upstream `(N,N,3,3)` Hartree/Bohr^2
arrays are transposed and converted once. Nonfinite or invalid shapes raise.

With `retain_scf: true`, energy, forces and Hessian share one SCF. Otherwise,
requesting Hessian after forces recreates the SCF; it still works, but costs a
new solve. Successful Hessian calls release the SCF unless retention was requested.
Any derivative failure clears results, closes the log and preserves diagnostics.

Optional `parameters.hessian` keys:

- `conv_tol_cpscf`: positive response convergence threshold.
- `grid_response`: Boolean response of the numerical grid, where supported.
- `auxbasis_response`: integer `0`, `1`, or `2`; **2 is full response**, and a
  Boolean is rejected. Requires density fitting.

Record `grid_response` explicitly when comparing CPU/GPU DFT Hessians, and use
`auxbasis_response: 2` for full DF response. The tested upstream CPU/GPU versions
do not implement identical numerical-grid response terms: equal options do not
guarantee identical Hessians. See the measured differences in the validation record. Unsupported options/configurations raise;
there is no automatic CPU fallback or whole-system finite-difference SCF scan.
The kit calls upstream `Hessian()`, whose D3/D4 correction uses a 1e-5 Bohr
finite-difference step and SMD CDS uses 1e-4 Bohr. These hybrid contributions
are included and recorded under `metadata["hessian"]`, never silently omitted.

## Structured diagnostics

`calc.metadata["scf"]` contains convergence, iteration count and callback count, final
cycle energy change/density residual, final unshifted orbital-gradient norm,
upstream energy components (Hartree), actual SCF controls, initial-guess source,
checkpoint status, and failure stage. Spin diagnostics report S^2, ideal S(S+1),
their difference, and `spin_analysis.mulliken_spin_populations` in atom order.
Mulliken populations are a basis-dependent diagnostic, not unique local spins.
Unavailable diagnostics carry no invented zero value.

Dimensions include atom/AO counts, available auxiliary-basis/grid sizes; memory
reports distinguish host `max_memory_mb` from GPU capacity. Diagnostics survive
unconverged SCF and derivative errors; these trial data are not valid ASE results.

```yaml
parameters:
  diagnostics:
    save: true
    iterations: true
    summary_file: scf_diagnostics.json
    iteration_file: scf_iterations.jsonl
```

Files are optional. The summary is updated for the current evaluation; JSONL
is appended during iterations and includes an evaluation id so successive
geometries are distinguishable. The execution example includes diagnostics in
its failure JSON, but does not implement optimizer resume.

See [reviewed 0.6.0 validation](validation-records.md) and
[earlier CPU/GPU validation](validation-history.md).

### Pinned-version Hessian capability limit

PySCF 2.14 CPU UKS Hessians with NLC/VV10 are not implemented upstream, with
or without DF. The kit raises a clear `NotImplementedError`; it does not drop
NLC or switch devices. Energy and forces are supported. GPU4PySCF 1.8.1 has
the corresponding UKS NLC Hessian path, which is checked against its direct API.
