# DFT Calculator Config Examples

VASP and Quantum ESPRESSO calculators are config-driven. Pass all calculation
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

The example script creates calculator objects from the YAML files but does not
run VASP or QE:

```bash
python examples/dft/create_dft_calculator_from_config.py vasp \
  examples/dft/vasp_pbe_static.yaml
```
