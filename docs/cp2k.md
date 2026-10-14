# CP2K through ASE

`cp2k` returns a thin validation subclass of ASE's CP2K calculator for
Gaussian/plane-wave DFT of molecules and periodic systems. CP2K performs all
electronic-structure calculations. No extra Python dependency is needed beyond
ASE; install the CP2K executable and its data files separately.

## Configuration and lifecycle

Use [the nonperiodic PBE H2 example](../examples/dft/cp2k_pbe_h2.yaml), replacing
`profile.command` and `profile.data_dir` with reviewed local paths:

```python
from ase import Atoms
from ase_calculator_kit import get_calculator

atoms = Atoms("H2", positions=[[0, 0, -0.37], [0, 0, 0.37]],
              cell=[12, 12, 12], pbc=False)
atoms.center()
with get_calculator("cp2k", config="examples/dft/cp2k_pbe_h2.yaml",
                    write_resolved_config=True) as calc:
    atoms.calc = calc
    print(atoms.get_potential_energy())
    print(atoms.get_forces())
```

**Construction immediately starts CP2K**, even before atoms are attached.
Use `with` as above or call `calc.close()` in a `finally` block. ASE may restart
the shell on a subsequent calculation after closing it.

Both `profile.command` and `profile.data_dir` must be nonempty. The kit always
passes the command explicitly and never consults `ASE_CP2K_COMMAND` or
`CP2K_DATA_DIR`. ASE 3.28.0 requires the literal substring `cp2k_shell` in the
command, for example `env OMP_NUM_THREADS=4 mpirun -np 2 /path/cp2k_shell.psmp`.
The kit raises `ValueError` before ASE's assert. The downloaded PyPI 3.29.0
wheel removes that assert; from 3.29.0 the kit also accepts commands such as
`mpirun -np 2 /path/cp2k.psmp -s`. The command must still launch a compatible
CP2K shell protocol.

Relative `basis_set_file` / `potential_file` names are made absolute relative
to `profile.data_dir`, with file existence checks. Omitted names use ASE's
`BASIS_MOLOPT` / `POTENTIAL` filenames, resolved the same way. Absolute names
are preserved and checked. `null` disables ASE generation of that filename;
then the template must contain an explicit absolute filename. Other template
file references (e.g. additional basis or restart files) must also be explicit:
the kit does not rewrite arbitrary CP2K input text. Relative `data_dir`,
`directory` and template file paths are relative to the Python working
directory, not the YAML location. None of these paths expand environment
variables.

`directory` becomes absolute (and is created) and the constructor receives label
`"cp2k"`. ASE combines them into `<directory>/cp2k`: it writes `.inp` and
requests `.out` there through `LOAD`. CP2K keeps string keywords such as
`GLOBAL/PROJECT` within 80 characters, so an absolute PROJECT path aborts input
parsing once the run directory is long (found in the first real run on TSUBAME).
The kit therefore starts the shell inside `directory`
(`cd <directory> && exec /bin/sh -c '<profile.command>'`) and writes the bare
`PROJECT cp2k`; PROJECT-derived files (restart files, wavefunctions) land in
`directory`, and relative file names in a template are relative to `directory`.
Use a different run directory for each calculator.

## Explicit physical settings and templates

`parameters.xc`, `basis_set`, `pseudo_potential` and `cutoff` must all be
present. `cutoff` is in **eV**, matching ASE, rather than CP2K's usual Ry input
convention. For each of these four settings, `null` delegates it to the
template instead of using ASE's physical default. A template must provide
the delegated setting; its completeness is the caller's responsibility.
`pseudo_potential: auto` is rejected case-insensitively because ASE can silently
substitute `GTH-PBE` for an unsupported functional. Choose consistent XC and
pseudopotentials yourself. Other ASE parameters retain upstream defaults when
omitted; specify them in YAML for a fully reviewed calculation.

Only names in `CP2K.default_parameters` are accepted. The kit does not add a
`spin` parameter: CP2K YAML uses `multiplicity` (2S + 1), unlike PySCF YAML's
`spin` (2S). Constructor options such as `debug`, `restart` and `label` are not
part of this config interface.

Top-level `inp` accepts inline CP2K input or a UTF-8 file path. For compatibility
`parameters.inp` is accepted too, but specifying both raises. Template text and
resolved data paths are saved in `resolved_calculator_config.yaml` when
requested. ASE augments `inp`; it does not treat the template as a complete,
unmodified input file. Avoid specifying the same keyword twice: for example,
set `xc: null` / `cutoff: null` when the template supplies XC/MGRID, and set
`poisson_solver: null` when it supplies POISSON. Do not set `GLOBAL/PROJECT`
in the template, since ASE always adds it. ASE also always adds coordinates
and CELL vectors/periodicity.

For `atoms.pbc=(True, True, False)`, ASE generates `CELL/PERIODIC XY` but does
not supply a matching slab Poisson solver. Explicitly add, for example:

```text
&FORCE_EVAL
  &DFT
    &POISSON
      PERIODIC XY
      POISSON_SOLVER MT
    &END POISSON
  &END DFT
&END FORCE_EVAL
```

Use `parameters.poisson_solver: null`. See the
[slab template example](../examples/dft/cp2k_pbe_slab.yaml). Choose a solver
supporting the intended periodicity and converge cell size/vacuum; see
[CP2K periodicity](https://www.cp2k.org/periodicity) and
[Poisson keywords](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/POISSON.html).
The H2 example uses `PERIODIC NONE` with `MT` and a finite vacuum cell. Neither
example establishes convergence for a production system.

## Charge, multiplicity and result conventions

Electronic state is **YAML-only (option b)** through `parameters.charge`,
`multiplicity` and `uks`. Charge must be an integer when supplied; multiplicity
must be an integer >= 1. NumPy integers are accepted; Booleans and fractional
values are rejected. Multiplicity > 1 enables `uks: true` when unspecified
or null, and rejects an explicit `uks: false`.

`atoms.info['charge']` / `atoms.info['spin']` are optional consistency checks,
with info `spin` interpreted as multiplicity. When present they require
explicit, non-null YAML charge/multiplicity, and must match. States supplied
only in template text cannot be checked, so info state fields are refused in
that case. Without info fields, omitted/null electronic settings retain ASE /
template semantics. For reproducibility set charge and multiplicity explicitly.
Changing info alone never changes the configured electronic state. A mismatch
raises before either a calculation or a cached property is returned, including
an energy-to-forces request. Create a new calculator/config to change state.
The kit does not infer electron counts or interpret arbitrary CP2K templates.

This choice avoids relying on force-env internals: ASE 3.28.0's `calculate()`
releases the force env only for `numbers` changes (lines 293–297); positions and
cell are updated in place, while info-only changes are absent from ASE's state
comparison. The kit adds validation, without updating parameters from info or
patching upstream force-env creation.

ASE sets `free_energy = energy` (3.28.0 line 342), even with smearing. Do not
interpret these as distinct E0 / finite-temperature free energies. ASE always
requests `GET_STRESS` and stores the returned tensor (lines 356–365), even when
`stress_tensor: false` disables generation of the analytical-stress section
(lines 427–430). A disabled calculation can yield zeros or a CP2K shell error;
presence of the result key is no evidence of a computed physical stress.
Template settings also affect stress generation. Keep analytical stress enabled
when stress is needed, and verify the CP2K output.

Source checks: installed ASE 3.28.0 `ase/calculators/cp2k.py` (shell assert
line 536; filenames lines 378–382; PROJECT line 422) and downloaded
`ase-3.29.0-py3-none-any.whl`, same module (launch lines 533–539, no assert;
force-env release lines 291–295). ASE `calculator.py` lines 643–659 and
696–708 resolve directory/label. These are source checks, not numerical
validation of CP2K. The opt-in H2 slow test reads only its binary/data locations
from `ASE_CALCULATOR_KIT_TEST_CP2K_COMMAND` / `..._DATA_DIR`, passes both into
config, and skips if either is absent. Real CP2K validation is pending on
TSUBAME; no accuracy or convergence claim is made here.
