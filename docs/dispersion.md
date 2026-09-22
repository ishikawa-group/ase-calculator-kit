# Dispersion

Add a Grimme-D3(BJ) correction on top of MLIP models with `dispersion=True`:

```python
atoms.calc = get_calculator("uma", task="omat", dispersion=True)
atoms.calc = get_calculator("uma", task="oc20", dispersion=True)
atoms.calc = get_calculator("chgnet", dispersion=True)
atoms.calc = get_calculator("sevennet", modal="pet_mad", dispersion=True)
atoms.calc = get_calculator("mace", head="omat_pbe", dispersion=True)
```

With `dispersion=True` the returned object is an ASE
`SumCalculator([backend_calculator, d3_calculator])`, not the backend calculator
itself — it satisfies the same `ase.Calculator` interface, but do not rely on
backend-specific attributes or `isinstance` checks against the backend class.

Some models already include dispersion in their training functional, so
`dispersion=True` is refused for them with `DispersionError`:

```python
get_calculator("uma", task="omol", dispersion=True)      # DispersionError: ωB97M-V includes VV10
get_calculator("sevennet", modal="spice", dispersion=True)  # DispersionError: SPICE is ωB97M-D3(BJ)
get_calculator("mace", head="omol", dispersion=True)     # DispersionError: ωB97M-VV10
```

**Every molecular task falls in this category** — molecular reference data is
almost always dispersion-corrected, each dataset in its own way (VV10, an
explicit D3(BJ) term, or MBD-NL). That verdict cannot be overridden with
`dispersion_xc=`; remove `dispersion=True` instead. A task this table does not
cover yet is refused by default but *can* be unlocked with an explicit
`dispersion_xc` once you have checked its functional yourself.

### Choosing the damping function

`dispersion_damping=` selects between Becke-Johnson (`"bj"`, the default) and
zero damping (`"zero"`):

```python
atoms.calc = get_calculator(
    "uma", task="oc20", dispersion=True, dispersion_damping="zero"
)
```

The two are separately fitted parameter sets, not a numerical detail. D3 does
not screen a metal's C6 coefficients, so on molecule–metal systems the choice
can move the correction by a factor of two — for RPBE, benzene on Pt(111) picks
up −4.6 eV of dispersion with BJ damping against −2.4 eV with zero damping.

Match the reference dataset when the model has one. OC20 and OC22 carry no
dispersion at all, so either damping is a choice you are making rather than
reproducing; OC25, in contrast, is RPBE + D3 with **zero** damping, which is
why `task="oc25"` refuses an added correction outright.

Note also that RPBE's D3 parameters — both dampings — are absent from Grimme's
published fits and carry no citation in the reference parameter tables, unlike
PBE's. Treat RPBE-D3 numbers on metals as indicative.

### Cutoff and smoothing follow PFP

The D3 term runs at **`cutoff=14.0` Å with `cutoff_smoothing="poly"`** — PFP
v7.0.0+'s settings, not torch-dftd's own defaults of 95 Bohr (50.3 Å) and no
smoothing.

This package exists to compare models against each other, and PFP is one of the
models being compared. Left at torch-dftd's defaults, the dispersion term added
to a SevenNet or MACE energy would be a *different quantity* from the one inside
a PFP energy, on top of the model difference you are trying to measure.
[Matlantis published the validation](https://docs.matlantis.com/atomistic-simulation-tutorial/ja/) for the shorter cutoff: an MAE of
0.0024 eV over the Wellendorff adsorption benchmark — negligible against the
0.01 eV scale those numbers live on — and no change in the 90th percentile of
COD unit-cell-volume error, in exchange for roughly three times the reachable
system size. `"none"` smoothing was a PFP bug fixed in v7.0.0; it leaves the
force discontinuous at the cutoff radius, which is exactly what a relaxation or
MD run walks into.

Both are overridable, so a dataset built on torch-dftd's defaults stays
reproducible:

```python
atoms.calc = get_calculator(
    "chgnet", dispersion=True,
    dispersion_cutoff=50.3, dispersion_cutoff_smoothing="none",
)
```

`cnthr`, the coordination-number cutoff, is left at torch-dftd's own default;
torch-dftd clamps it to `cutoff` when it is larger, which is the same path PFP
goes through.

See [`docs/models.md`](https://github.com/ishikawa-group/ase-calculator-kit/blob/main/docs/models.md) for the full per-model table.
