# Models, training data, and dispersion policy

This table is the human-readable form of the dispersion (D3) policy encoded in
[`src/ase_calculator_kit/dispersion.py`](../src/ase_calculator_kit/dispersion.py). **The two
must be kept in sync** — if you change one, change the other.

It records, for each model / task / modal: the training dataset, the DFT level
(exchange–correlation functional) it was trained at, whether that functional
already accounts for dispersion, and what `get_calculator(..., dispersion=True)`
does as a result.

Why this matters: adding a Grimme-D3 correction on top of a model whose training
functional *already* includes dispersion would **double-count** the van-der-Waals
interaction and give wrong energies. So such models reject `dispersion=True`.

## Dispersion policy table

| Model / task | Training dataset | DFT level (functional) | Dispersion in training? | `dispersion=True` behavior |
|---|---|---|---|---|
| **CHGNet** `default` / `0.3.0` / `0.2.0` | MPtrj | PBE+U | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **CHGNet** `r2scan` | MatPES r2SCAN transfer-learning | r2SCAN | ✗ none | ✅ allowed — D3 `xc=r2scan` |
| **MatterSim** `default` / `1M` / `5M` | MatterSim set (MPtrj + T/P-sampled structures) | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **NequIP OAM** `S` / `M` / `L` / `XL` | OMat24 pre-training + sAlex / MPTrj fine-tuning | PBE(+U)-level materials data | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `mpa` | MPtrj + sAlex | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `omat24` | OMat24 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `matpes_pbe` | MatPES | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `oc20` | OC20 | RPBE | ✗ none | ✅ allowed — D3 `xc=rpbe` |
| **SevenNet** `oc22` | OC22 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `default` (single-fidelity, e.g. 7net-0) | MPtrj etc. | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `matpes_r2scan` | MatPES | r2SCAN | ✗ none | ✅ allowed — D3 `xc=r2scan` |
| **SevenNet** `mp_r2scan` | Materials Project r2SCAN | r2SCAN | ✗ none | ✅ allowed — D3 `xc=r2scan` |
| **SevenNet** `pet_mad` | MAD | PBEsol | ✗ none | ✅ allowed — D3 `xc=pbesol` |
| **SevenNet** `omol25_low` / `omol25_high` | OMol25 (low-spin / high-spin) | ωB97M-V | ✓ yes (VV10 nonlocal) | ⛔ error (double-counting) |
| **SevenNet** `spice` | SPICE | ωB97M-D3(BJ)/def2-TZVPPD | ✓ yes (D3(BJ) included) | ⛔ error (double-counting) |
| **SevenNet** `qcml` | QCML | PBE0 + MBD-NL | ✓ yes (MBD-NL many-body) | ⛔ error (double-counting) |
| **SevenNet** `odac23` | ODAC23 | PBE-D3 | ✓ yes (D3 included) | ⛔ error (double-counting) |
| **UMA** `omat` | OMat24 | PBE+U | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **UMA** `oc20` | OC20 | RPBE | ✗ none | ✅ allowed — D3 `xc=rpbe` |
| **UMA** `oc22` | OC22 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **UMA** `oc25` | OC25 | RPBE+D3(BJ) | ✓ yes (D3 included) | ⛔ error (double-counting) |
| **UMA** `omol` | OMol25 | ωB97M-V | ✓ yes (VV10 nonlocal) | ⛔ error (double-counting) |
| **UMA** `odac` | ODAC23 | PBE-D3 | ✓ yes (D3 included) | ⛔ error (double-counting) |
| **UMA** `omc` | OMC25 | PBE+D3 | ✓ yes (D3 included) | ⛔ error (double-counting) |

Legend: ✅ allowed (auto `xc`) · ⚠️ unverified (refused by default, override with an
explicit `dispersion_xc`) · ⛔ always refused (model already includes dispersion).

Every model/task/modal listed above is verified, so no row currently sits in the
⚠️ tier. The tier is not dead: anything *not* in this table — a modal added by a
future upstream release, for instance — lands there and is refused by default.

## Notes

- **Mechanism.** When allowed, the D3 correction is applied as
  `SumCalculator([base_model, TorchDFTD3Calculator(damping="bj", xc=<above>)])`
  via [`torch-dftd`](https://github.com/pfnet-research/torch-dftd). Energies,
  forces, and stress are summed.
- **`xc` choice.** D3 parameters depend on the functional, so the default `xc`
  matches the model's training functional. Override with `dispersion_xc="..."`.
- **Override / escape hatch.** For a model/task *not listed above*, passing an
  explicit `dispersion_xc` (e.g. `dispersion_xc="pbe"`) acknowledges you have
  checked the functional yourself and unlocks the correction. The
  *already-includes-dispersion* rows cannot be overridden — remove
  `dispersion=True` instead.
- **NequIP OAM.** OAM (S/M/L/XL) is trained on PBE(+U)-level materials data
  (OMat24 pre-training + sAlex / MPtrj fine-tuning) and does not include
  dispersion, so D3 with `xc=pbe` is applied.
- **CHGNet** is keyed by the model name: the MPtrj checkpoints `0.3.0`/`0.2.0`
  are PBE (`xc=pbe`), while the `r2scan` transfer-learning checkpoint is r2SCAN
  (`xc=r2scan`).
- **r2SCAN models** (CHGNet `r2scan`, SevenNet `matpes_r2scan`) use D3(BJ) with
  `xc=r2scan`, which torch-dftd supports.
- **ωB97M-V** is a range-separated hybrid with the VV10 nonlocal correlation term,
  which already captures long-range dispersion; adding D3 would double-count it.
- **Molecular tasks are all in the ⛔ tier.** Molecular reference datasets are
  almost always dispersion-corrected, each in a different way: OMol25 through
  VV10, SPICE through an explicit D3(BJ) term, QCML through MBD-NL, and the
  molecular-crystal / MOF sets (OMC25, ODAC23) through PBE+D3. None of them may
  take another D3 correction, and because the ⛔ tier cannot be overridden, a
  stray `dispersion_xc=` cannot re-enable one either.
- **`omol25_low` vs `omol25_high`** select the *spin state*, not the accuracy:
  SevenNet trains the low-spin and high-spin OMol25 configurations as separate
  tasks (low-spin organometallics are oversampled 5×). Both are ωB97M-V.
- **Charge and spin.** Only UMA's `omol` head reads a total charge and spin
  multiplicity, from `atoms.info["charge"]` and `atoms.info["spin"]`. SevenNet
  has no charge/spin input at all, so its molecular modals cannot describe ions
  or a chosen open-shell state. See the README's "Molecular systems" section.
- These functional assignments reflect the datasets as of mid-2026; if upstream
  retrains a task at a different level, update both this table and `dispersion.py`.
