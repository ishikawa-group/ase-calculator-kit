# Apple Silicon (MPS) support

Every MLIP backend was run on a single point (`bulk("Cu")`) with `device="mps"`
on Apple Silicon Macs (arm64, MPS available). MatGL models use the environment
recorded in their validation report below. Results:

| Backend | `device="mps"` | Notes |
|---|---|---|
| SevenNet | ✅ supported | validated locally (`7net-omni`) |
| CHGNet | ✅ supported | validated locally |
| MatGL TensorNet | ✅ supported | both MatPES checkpoints; MPS uses float32, D3 runs on CPU; see the [validation record](matgl-validation.md) |
| MatGL CHGNet (provisional) | ✅ supported | both corrected MatPES checkpoints; MPS uses float32, D3 runs on CPU; see the [validation record](matgl-validation.md) |
| MatterSim | ✅ supported | validated locally |
| NequIP OAM | ❌ not supported | PyTorch MPS lacks float64; the packaged OAM models use float64 buffers |
| OrbMol | ❌ not supported | graph construction runs through NVIDIA Warp, which has no Metal backend; the legacy edge methods need float64 or refuse a non-CPU device |
| MACE | ❌ not supported | same float64 problem: loading `mace-mh-1.model` with `map_location="mps"` raises `Cannot convert a MPS Tensor to float64`, with `default_dtype="float32"` as well |
| UMA / fairchem | ❌ not supported | `fairchem-core` asserts `device in {"cpu", "cuda"}` |

For the MPS-supported backends, `device="auto"` resolves to `mps` on Apple
Silicon when no CUDA device is present. NequIP, OrbMol, MACE and UMA accept only
`"cpu"` / `"cuda"`; passing `device="mps"` raises a clear `ValueError`, and
`device="auto"` falls back to `cpu`.
