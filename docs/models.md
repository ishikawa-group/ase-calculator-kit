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
| **MACE** `omat_pbe` (MH-1 default) | OMat24 replay (10% of the pre-training set) | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **MACE** `mp_pbe_refit_add` | MPtrj | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **MACE** `oc20_usemppbe` | OC20 (2M subsample) | PBE, as stated by the MACE-MH-1 paper — see the note below | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **MACE** `matpes_r2scan` | MatPES | r2SCAN (no Hubbard U) | ✗ none | ✅ allowed — D3 `xc=r2scan` |
| **MACE** `omol` | OMol25 (1% subsample) | ωB97M-VV10 | ✓ yes (VV10 nonlocal) | ⛔ error (double-counting) |
| **MACE** `spice_wB97M` | SPICE-1 | ωB97M-D3(BJ) | ✓ yes (D3(BJ) included) | ⛔ error (double-counting) |
| **MACE** `medium-omat-0` / `small-omat-0` (MACE-OMAT-0) | OMat24 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **MACE** `medium-mpa-0` | MPtrj + sAlex | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **MACE** `mace-matpes-pbe-0` | MatPES | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **MACE** `mace-matpes-r2scan-0` | MatPES | r2SCAN | ✗ none | ✅ allowed — D3 `xc=r2scan` |
| **MatterSim** `default` / `1M` / `5M` | MatterSim set (MPtrj + T/P-sampled structures) | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **NequIP OAM** `S` / `M` / `L` / `XL` | OMat24 pre-training + sAlex / MPTrj fine-tuning | PBE(+U)-level materials data | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `mpa` | MPtrj + sAlex | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `omat24` | OMat24 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `matpes_pbe` | MatPES | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `oc20` | OC20 | RPBE | ✗ none | ✅ allowed — D3 `xc=rpbe` |
| **SevenNet** `oc22` | OC22 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `default` (single-fidelity, e.g. 7net-0) | MPtrj etc. | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `7net-omat` / `sevennet-omat` (single-fidelity) | OMat24 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
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
| **UMA** `oc25` | OC25 | RPBE+D3(zero) | ✓ yes (D3 included) | ⛔ error (double-counting) |
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
- **MACE-MH-1 heads.** MH-1 is one checkpoint with six readout heads, and the
  head *is* the level of theory — the same structure returns PBE, r2SCAN or
  ωB97M energies depending on which one is selected. The names above are read
  back from the shipped `mace-mh-1.model` file (`available_heads`), not from the
  model card: the card advertises an `rgd1_b3lyp` head, and the published
  checkpoint carries `mp_pbe_refit_add` in its place. This matters because MACE
  does **not** reject an unknown head — it logs a warning and quietly computes
  with the last head in the file — so `get_calculator("mace", head=...)`
  validates the name itself and raises instead.
- **Single-head MACE checkpoints are keyed by model, not head.** MACE-OMAT-0 and
  friends carry one head called `Default`, so there is nothing to select and the
  functional is a property of the whole checkpoint. `head="auto"` (the default)
  passes no head for those, and the model name keys the rows above. The same
  applies to **SevenNet** single-fidelity models such as `7net-omat`, which take
  no `modal`.
- **MACE and D3.** The MH-1 authors evaluate their PBE-trained heads with
  torch-dftd D3(BJ) using the PBE parametrisation and run the OMol head with no
  added dispersion ([arXiv:2510.25380](https://arxiv.org/abs/2510.25380),
  "Dispersion corrections (D3)"), which is exactly the policy above.
- **MACE `oc20_usemppbe` is the one row that follows the model paper rather than
  the dataset.** The MH-1 paper describes this head as OC20 "computed at the PBE
  level", while OC20 as published is RPBE — which is what the **SevenNet**
  `oc20` and **UMA** `oc20` rows use. The head name also says it is referenced
  to MP's PBE data. We follow the model's own paper here, because the D3 term
  should match what the head reproduces; pass `dispersion_xc="rpbe"` to follow
  the dataset convention instead.
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
