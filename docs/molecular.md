# Molecular systems: charge and spin

Molecular models need two inputs that no bulk model does: the **total charge**
of the system and its **spin multiplicity** (`2S+1`). ASE has no standard place
for either, so they are passed through `atoms.info`, and the backends differ in
whether they read them at all.

| Backend | Molecular option | Takes charge / spin? |
|---|---|---|
| `esen` | OMol25 checkpoints | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` |
| `pyscf` / `gpu4pyscf` | Molecular HF/DFT | ✅ `atoms.info` or YAML; mismatches raise |
| `uma` | `task="omol"` | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` |
| `sevennet` | `modal="omol25_low"` / `"omol25_high"` / `"spice"` / `"qcml"` | ❌ not supported by sevenn |
| `mace` | `model="omol-0"` / `"MACE-OMOL-0"` | ✅ explicit `atoms.info` charge/multiplicity |
| `mace` | `head="omol"` / `"spice_wB97M"` | ❌ the MH-1 molecular heads are fitted to neutral closed-shell data |
| `mace` | `model="polar-1-s"` / `"polar-1-m"` / `"polar-1-l"` | ✅ `atoms.info["charge"]`, `atoms.info["spin"]`, `atoms.info["external_field"]` |
| `orb` | `model="orbmol-v2"` (the default) | ✅ `atoms.info["charge"]`, `atoms.info["spin"]` — **required**, not defaulted |

### UMA: set both keys explicitly

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
atoms.info["charge"] = 0   # total charge
atoms.info["spin"] = 1     # spin multiplicity, 2S+1 (1 = closed shell)
atoms.calc = get_calculator("uma", task="omol")
print(atoms.get_potential_energy())
```

A hydroxide anion and a neutral radical are the cases that actually bite:

```python
oh_minus = molecule("OH")
oh_minus.info["charge"] = -1   # anion
oh_minus.info["spin"] = 1      # closed shell
oh_minus.calc = get_calculator("uma", task="omol")

oh_radical = molecule("OH")
oh_radical.info["charge"] = 0
oh_radical.info["spin"] = 2    # doublet — one unpaired electron
oh_radical.calc = get_calculator("uma", task="omol")
```

> **Set both fields explicitly.** Since 0.5.10, kit-created UMA/eSEN `omol`
> calculators reject missing or inconsistent states before inference or cache use.
> Upstream fairchem alone still defaults `charge=0` / `spin=1` in the
> `atoms.info` dict you passed in, and returns a neutral closed-shell result.
> An ion or an open-shell species then comes back **silently wrong**. Set both
> keys on every molecular structure, including the ones you think are obvious.

Both keys are integers. `charge` may range from -100 to 100 and `spin` from 1 to
100; they are read only by the `omol` head, and other UMA tasks ignore them.

### SevenNet: no charge or spin input

sevenn has no charge or spin argument, so the `modal` embedding is the only
handle on the molecular reference data. Charged species and a chosen open-shell
state **cannot be expressed** — `omol25_high` selects a model trained on
high-spin configurations, but it is not a multiplicity you set per structure.
Use `get_calculator("uma", task="omol")` when the charge and spin of the system
matter.

### MACE-MH-1: molecular heads, no charge or spin

MACE-MH-1's `omol` head was fine-tuned on a subsample of OMol25 that is
explicitly **neutral and closed-shell**, and `spice_wB97M` on SPICE-1, which is
likewise neutral. There is no charge or spin input to set, and no head that
covers ions or a chosen open-shell state. Use `get_calculator("uma",
task="omol")`, or MACE-Polar below, when the charge and spin of the system
matter.

### MACE-Polar: charge, spin, and an applied field

`polar-1-s` / `polar-1-m` / `polar-1-l` are MACE's **electrostatics** foundation
models, trained on OMol25 at ωB97M-V. They are the one group of checkpoints
loaded through upstream's `mace_polar` rather than `mace_mp` — the model name is
what switches the loader, so nothing else changes at the call site.

They need one package on top of `mace-torch`. It is not on PyPI, so no extra of
this project can pull it in and you install it yourself, into the MACE
environment:

```bash
.venv-mace/bin/pip install git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0
```

That repository builds a distribution named `graph_longrange` — the module the
checkpoints unpickle. The name mismatch matters at the command line: pip rejects
the `graph_electrostatics @ git+…` spelling as an inconsistent name, so pass the
bare URL as above. Ask for a polar checkpoint without it and
`get_calculator` raises `MissingDependencyError` with this command, rather than
letting MACE fail with `No module named 'graph_longrange'` after the download.

```python
from ase.build import molecule
from ase_calculator_kit import get_calculator

atoms = molecule("H2O")
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.01]   # V/Å, along z
atoms.calc = get_calculator("mace", model="polar-1-m")
print(atoms.get_potential_energy())
print(atoms.calc.results["dipole"])               # non-periodic cells only
```

Beyond energy, forces and stress, `calc.results` carries `dipole`, `charges`,
`spins` and an electrostatic energy breakdown (`electrostatic_energy`,
`electron_energy`, `interaction_energy`). **Read them from `calc.results`
directly**: MACE leaves them out of `implemented_properties`, so ASE's own
accessors do not see them and `atoms.get_dipole_moment()` raises
`PropertyNotImplementedError` even though `calc.results["dipole"]` is there.
Partial charges are also **not a uniquely defined quantity** — read them as a
decomposition of the model's electrostatics, not as a measurement.

> **Set charge and multiplicity explicitly.** Since 0.5.10 the kit rejects
> missing or inconsistent electronic states for molecular MACE. An omitted
> `external_field` still means zero field; set it explicitly for field-on runs.

`dispersion=True` is refused for all three: OMol25's ωB97M-V reference already
carries the nonlocal VV10 term, so a D3 correction would double-count it.

`polar-1-m` and `polar-1-l` carry 12 Å and 18 Å receptive fields; `polar-1-s` is
the small model. All three live in the MACE environment like every other MACE
checkpoint, and all three are stored in float32 — with this package's
`default_dtype="float64"` default, MACE logs that it is converting them, which
is the conversion and not a failure.

### OrbMol-v2: charge and spin, enforced

OrbMol-v2 reads the same two `atoms.info` keys and **raises** when they are missing.

```python
oh_minus = molecule("OH")
oh_minus.info["charge"] = -1
oh_minus.info["spin"] = 1
oh_minus.calc = get_calculator("orb")
print(oh_minus.get_potential_energy())

forgot = molecule("OH")
forgot.calc = get_calculator("orb")
forgot.get_potential_energy()
# ValueError: atoms.info must contain both 'charge' and 'spin'
```

Since 0.5.10, kit-created UMA/eSEN omol and molecular MACE also require both keys.

The two agree closely where both are defined. On eight small molecules and ions
— neutral, anionic and open-shell, isolated — `get_calculator("orb")` and
`get_calculator("uma", task="omol")` with `uma-s-1p2p1` differed by **0.18 to
5.87 meV in total energy**, on absolute energies of 1.5 to 6.3 keV. Both are
OMol25 ωB97M-V models, so their absolute energy scales are directly comparable.
Force directions were identical (direction cosine 1.0000 throughout) and
components agreed to 0.05 eV/Å, except the O₂ triplet at 0.21 eV/Å — that and
the acetate anion were the two largest disagreements.

They also handle a periodic box differently, and that is a real difference
rather than a tolerance. Put each molecule in a 14 Å cell with `pbc=True` and
UMA returns the isolated energy unchanged to 1e-7 eV: its `omol` head has no
long-range electrostatic term, so nothing reaches past the cutoff. OrbMol-v2's
Coulomb term is an Ewald sum and is not cutoff-limited, so it moves — by under
3 meV for the neutral molecules, and by 1.45 eV for acetate⁻, which the Ewald
convention computes against a neutralizing background. See
[OrbMol `model`](backends.md#orbmol-model).

### Non-periodic cells

`ase.build.molecule()` returns `pbc=False` with a zero cell, which UMA accepts.
UMA rejects only two ambiguous cases: a fully periodic structure whose cell is
all zeros, and a partially periodic one (`pbc=[True, True, False]`).

## State cache correction in 0.6.0

Kit-created OrbMol calculators now invalidate energy/force caches when valid
charge or multiplicity changes at unchanged coordinates. Results produced by
older shared calculators under that condition need recalculation; successful
execution alone did not guarantee correct electronic-state inference.
MACE-POLAR detects field replacement and in-place array/list changes. Its
omitted field and an explicit zero field represent the same effective input.
NumPy integer charge/multiplicity values are normalized on a computation copy;
the caller's Atoms are not modified. Custom MACE info_keys remain supported.
