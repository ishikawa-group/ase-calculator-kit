# Changelog

## 0.3.1

- Ship SevenNet (`sevenn`) as a base dependency, so `pip install
  ase-calculator-kit` gives a working NNP backend out of the box.
- Keep the `sevennet` extra as a no-op alias so existing
  `ase-calculator-kit[sevennet]` installs keep working.
- Suppress the `cueq/False/flash/False` debug lines that sevenn 0.12.1 prints
  unconditionally when constructing `SevenNetCalculator`; any other stdout from
  the model load is still forwarded.

## 0.3.0

- Make the default installation lightweight: only ASE and PyYAML are required.
- Move each NNP stack into an explicit extra: `chgnet`, `sevennet`, `mattersim`,
  `nequip`, and `uma`.
- Add `all` for every supported NNP plus D3, and `dispersion` for D3 alone.
- Improve missing-dependency errors so they show the exact extra to install.

## 0.2.2

- Verify dispersion functionals for models that were previously refused, moving
  them from the "unverified" tier to "allowed" (auto `xc`):
  - NequIP OAM (S/M/L/XL): PBE(+U)-level -> D3 `xc=pbe`.
  - SevenNet `oc20` -> `xc=rpbe`, `oc22` -> `xc=pbe`, `matpes_r2scan` -> `xc=r2scan`.
  - CHGNet keyed by model name: `0.3.0`/`0.2.0` (MPtrj, PBE) -> `xc=pbe`,
    `r2scan` (r2SCAN transfer learning) -> `xc=r2scan`. Fixes `dispersion=True`
    being refused for an explicitly-named CHGNet model.
- Keep `docs/models.md` and `dispersion.py` in sync with the above.

## 0.2.1

- Probe every MLIP backend on Apple Silicon (MPS) with a real single point and
  reconcile the `device="mps"` support flags with the measured results.
- Enable SevenNet `device="mps"` (validated locally with `7net-omni`).
- Confirm CHGNet and MatterSim `device="mps"` (already enabled) still compute
  energy and forces on MPS.
- Keep NequIP MPS disabled — PyTorch MPS lacks float64, which the packaged OAM
  models require (`Cannot convert a MPS Tensor to float64`).
- Document that UMA / fairchem does not support MPS: `fairchem-core` asserts
  `device in {"cpu", "cuda"}`.
- Add an "Apple Silicon (MPS) support" matrix to the README and backend
  pass-through tests for SevenNet (mps) and UMA (mps rejected).

## 0.2.0

- Add the NequIP OAM backend with the S, M, L, and XL model variants from
  nequip.net.
- Enable MatterSim `device="mps"` after local Apple Silicon validation.
- Document NequIP and MatterSim MPS behavior.

## 0.1.0

- Initial ase-calculator-kit release with MLIP and config-only DFT calculator
  factories.
