# DFT Calculator Config Examples

VASP, Quantum ESPRESSO and CP2K calculators are config-driven. Pass all calculation
settings through `config=` and use `overrides=` only for small dynamic changes
such as an output directory or a k-point mesh.

```python
from ase_calculator_kit import get_calculator

calc = get_calculator("vasp", config="examples/dft/vasp_pbe_static.yaml")

calc = get_calculator(
    "qe",
    config="examples/dft/qe_pbe_static.yaml",
    overrides={"directory": "runs/qe/Cu_001"},
)
```

Arbitrary DFT keyword arguments are rejected:

```python
get_calculator("vasp", encut=520)  # TypeError
```

The example script creates calculator objects from the YAML files. VASP and QE
start no executable here; CP2K starts its shell and the script closes it:

```bash
python examples/dft/create_dft_calculator_from_config.py vasp \
  examples/dft/vasp_pbe_static.yaml
```


## CP2K

- [PBE H2](cp2k_pbe_h2.yaml): nonperiodic molecule with explicit Poisson settings.
- [PBE slab](cp2k_pbe_slab.yaml): partial PBC `(True, True, False)` and an XC/grid template.

Replace `profile.command` and `profile.data_dir` with your CP2K installation.
The four keys `xc`, `basis_set`, `pseudo_potential`, `cutoff` are mandatory;
`null` delegates them to `inp`. Cutoff is in eV. `pseudo_potential: auto` is
refused. Relative data filenames resolve against data_dir; missing files raise.
`inp` accepts inline input or a UTF-8 file path. Do not duplicate template
keywords that ASE generates; set their parameters to null. Charge/multiplicity
are YAML-only; optional atoms.info charge/spin (multiplicity) must agree before
cache reuse. See [the complete CP2K guide](../../docs/cp2k.md).

CP2K **starts at calculator creation**, including in the creation script:

```python
from ase_calculator_kit import get_calculator

with get_calculator("cp2k", config="examples/dft/cp2k_pbe_h2.yaml") as calc:
    atoms.calc = calc  # supply H2 in a finite vacuum cell with pbc=False
    print(atoms.get_potential_energy())
```

Use `with` or call `close()`. ASE equates energy/free_energy even with smearing,
and disabling analytical stress can yield zeros or fail. Slabs require an
explicit Poisson solver compatible with their periodicity. Examples are starting
inputs, not convergence or accuracy validation.
