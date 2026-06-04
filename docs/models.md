# Models, training data, and dispersion policy

This table is the human-readable form of the dispersion (D3) policy encoded in
[`src/ase_umlip_kit/dispersion.py`](../src/ase_umlip_kit/dispersion.py). **The two
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
| **CHGNet** (default) | MPtrj | PBE+U | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **MatterSim** 1M / 5M | MatterSim set (MPtrj + T/P-sampled structures) | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `mpa` | MPtrj + sAlex | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `omat24` | OMat24 | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `matpes_pbe` | MatPES | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** (single-fidelity, e.g. `7net-0`) | MPtrj etc. | PBE | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **SevenNet** `matpes_r2scan` | MatPES | r2SCAN | ✗ none | ⚠️ unverified — error unless `dispersion_xc` given (torch-dftd r2scan params to confirm) |
| **SevenNet** `omol25_low` / `omol25_high` | OMol25 | ωB97M-V | ✓ yes (VV10 nonlocal) | ⛔ error (double-counting) |
| **UMA** `omat` | OMat24 | PBE+U | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **UMA** `oc20` | OC20 | RPBE | ✗ none | ✅ allowed — D3 `xc=rpbe` |
| **UMA** `oc22` | OC22 | PBE(+U) | ✗ none | ✅ allowed — D3 `xc=pbe` |
| **UMA** `oc25` | OC25 | RPBE+D3(BJ) | ✓ yes (D3 included) | ⛔ error (double-counting) |
| **UMA** `omol` | OMol25 | ωB97M-V | ✓ yes (VV10 nonlocal) | ⛔ error (double-counting) |
| **UMA** `odac` | ODAC23 | (unverified) | ⚠️ unverified | ⚠️ conservative error — error unless `dispersion_xc` given |
| **UMA** `omc` | OMC25 | (unverified) | ⚠️ unverified | ⚠️ conservative error — error unless `dispersion_xc` given |

Legend: ✅ allowed (auto `xc`) · ⚠️ unverified (refused by default, override with an
explicit `dispersion_xc`) · ⛔ always refused (model already includes dispersion).

## Notes

- **Mechanism.** When allowed, the D3 correction is applied as
  `SumCalculator([base_model, TorchDFTD3Calculator(damping="bj", xc=<above>)])`
  via [`torch-dftd`](https://github.com/pfnet-research/torch-dftd). Energies,
  forces, and stress are summed.
- **`xc` choice.** D3 parameters depend on the functional, so the default `xc`
  matches the model's training functional. Override with `dispersion_xc="..."`.
- **Override / escape hatch.** For the *unverified* rows, passing an explicit
  `dispersion_xc` (e.g. `dispersion_xc="pbe"`) acknowledges you have checked the
  functional yourself and unlocks the correction. The *already-includes-dispersion*
  rows cannot be overridden — remove `dispersion=True` instead.
- **ωB97M-V** is a range-separated hybrid with the VV10 nonlocal correlation term,
  which already captures long-range dispersion; adding D3 would double-count it.
- These functional assignments reflect the datasets as of mid-2026; if upstream
  retrains a task at a different level, update both this table and `dispersion.py`.
