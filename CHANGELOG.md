# Changelog

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
