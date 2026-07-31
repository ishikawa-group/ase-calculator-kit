# AGENTS.md

Working notes for AI coding agents (Claude Code, Codex/GPT, Copilot, and
others) on `ase-calculator-kit`. Human-facing documentation lives in
[`README.md`](README.md); this file holds the repository map, invariants, and
conventions that are easy to violate without reading the whole codebase.

Read this file before editing. If a change contradicts an invariant below,
say so explicitly instead of silently changing the behavior.

## What this package is

A thin factory layer. It does **not** implement any physics: it maps a name
plus keywords onto an upstream ASE calculator (`chgnet`, `sevenn`, `mattersim`,
`nequip`, `fairchem-core`, `ase.calculators.vasp`, `ase.calculators.espresso`)
and returns it unchanged. New behavior belongs upstream unless it is about
*selection*, *validation*, or *reproducibility*.

## Repository map

```
src/ase_calculator_kit/
  __init__.py        public API surface; __all__ is the contract
  factory.py         get_calculator / get_mlip_calculator / get_dft_calculator,
                     available_* helpers, DFT kwarg validation
  registry.py        name -> backend class maps (MLIP_BACKENDS, DFT_BACKENDS)
  device.py          resolve_device(): "auto"/"cuda"/"cpu"/"mps" + allow_mps gate
  dispersion.py      per-model D3 policy table, wrap_with_d3(), prechecks
  errors.py          CalculatorKitError, MissingDependencyError, DispersionError
  config.py          YAML load / deep_merge / resolve / write-resolved-config
  backends/base.py   BaseBackend: every backend implements create_calculator()
  backends/mlip/     chgnet.py sevennet.py mattersim.py nequip.py fairchem.py
  backends/dft/      vasp.py espresso.py
  py.typed           PEP 561 marker; keep it listed in [tool.setuptools.package-data]
tests/               fast unit tests + test_singlepoint_cpu.py (marked slow)
constraints.txt      exact tested versions behind the pyproject ranges
examples/            run_all_models.py, examples/dft/*.yaml
docs/models.md       per-model training functional and dispersion policy
docs/code-guide_ja.md 実装ガイド（日本語）
```

Adding a backend touches, in order: `backends/mlip/<name>.py`,
`backends/__init__.py`, `registry.py`, `dispersion.py` (policy entry),
`pyproject.toml` (extra + `all`), `constraints.txt` (exact tested version),
`errors.py` (extra mapping), tests, README, `docs/models.md`.

## Invariants

These are deliberate design decisions, not oversights.

1. **DFT is config-only.** `get_calculator("vasp", encut=520)` must raise
   `TypeError`. Only `config=`, `overrides=`, `write_resolved_config=` are
   accepted (`factory.py:_DFT_ALLOWED_KWARGS`). Do not "helpfully" forward
   extra keywords — the point is that DFT conditions live in a reviewable YAML
   file, not in scattered Python call sites.
2. **DFT execution is explicit.** VASP configs require `profile.command`; QE
   requires `profile.command`, `profile.pseudo_dir`, and `pseudopotentials`.
   Never fall back to environment variables (`VASP_COMMAND`, `ASE_*`) — a run
   must be reproducible from the config alone.
3. **Missing NNP packages surface as `MissingDependencyError`,** naming the
   packaging extra. Import backend packages *inside* `create_calculator()`,
   never at module import time, so `import ase_calculator_kit` works without
   torch installed.
4. **Dispersion is policy-gated.** `dispersion=True` is refused with
   `DispersionError` for models whose training functional already includes
   dispersion, and requires an explicit `dispersion_xc` for unverified
   functionals. The table in `dispersion.py` and `docs/models.md` must stay in
   sync; changing one without the other is a bug.
5. **`dispersion=True` changes the return type** to
   `SumCalculator([backend_calc, d3_calc])`. Anything that assumes the backend
   class comes back is wrong.
6. **MPS support is measured, not assumed.** `resolve_device(..., allow_mps=)`
   is `True` only for CHGNet, SevenNet, and MatterSim, because those were
   validated with a real single point on Apple Silicon. Do not flip a flag
   without running the calculation; record the result in the README matrix.
7. **MACE stays out.** `mace-torch` needs an `e3nn` version incompatible with
   the one pinned by `sevenn` and `fairchem-core`. Do not add it.
8. **Published requirements are ranges; exact pins live in `constraints.txt`.**
   `pyproject.toml` uses compatible ranges (`ase>=3.28,<4`) so the package can be
   installed alongside whatever ASE/NNP versions a user already has — an
   `==`-pinned library is uninstallable for half its audience. `constraints.txt`
   holds the exact tested combination and is what CI installs with `-c`. The
   `dev` extra is the one exception and stays `==`-pinned, so a ruff release
   cannot turn CI red on an unrelated PR. `tests/test_packaging.py` enforces
   both halves of this. Widen a range only after testing the new version.
9. **The default install stays lightweight** (since 0.3.0): the base
   dependencies are ASE and PyYAML only, so a bare
   `pip install ase-calculator-kit` does *not* pull in torch. Every NNP stack
   lives behind its own extra. Promoting a backend to a base dependency reverses
   a deliberate release decision — do not do it without agreement.

## Conventions

- Python `>=3.12,<3.14`. `from __future__ import annotations` at the top of
  every module; modern typing (`str | None`, `dict[str, Any]`).
- All `create_calculator()` parameters are keyword-only (`*,`) with defaults,
  ending in `**kwargs` forwarded to the upstream calculator.
- Docstrings are NumPy-style and carry the *chemistry* rationale (when to pick a
  model/modal/task), not just the mechanics. Keep that habit — it is the main
  reason the docstrings exist.
- Comments explain *why*, sparingly. Match the surrounding density.
- Line length and lint: `ruff check src tests examples` must pass.
- Backend names are lowercased before lookup; keep new names lowercase.

## Verification

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]" -c constraints.txt   # add ,all for slow tests
.venv/bin/pytest                        # fast suite; slow tests deselected
.venv/bin/ruff check src tests examples
.venv/bin/pytest -m slow                # real single points, downloads weights
.venv/bin/pytest -m slow -s             # with the tqdm progress bar
```

- `addopts = "-m 'not slow'"` in `pyproject.toml` deselects the slow suite by
  default. Do not run `-m slow` casually: it downloads model weights (GBs) and
  needs `.[dev,all]`.
- CI (`.github/workflows/ci.yml`) runs the fast suite on Python 3.12 and 3.13
  plus `ruff check`. Assume no GPU and no Apple Silicon in CI.
- Releases are cut by pushing a `v*` tag; `.github/workflows/release.yml`
  builds, checks that the tag matches `[project].version`, and uploads to PyPI
  via Trusted Publishing (OIDC, no stored token). PyPI filenames are immutable —
  a version can be yanked but never re-uploaded, so use the `workflow_dispatch`
  TestPyPI path first.
- Backend tests inject fake modules into `sys.modules`
  (see `tests/test_sevennet_backend.py`) rather than importing real NNP
  packages. Follow that pattern for new backends — fast tests must not download
  anything.
- Do not commit or push unless asked.

## Known upstream quirks

- **sevenn 0.12.1** unconditionally prints `cueq / <bool> / flash / <bool>` from
  `SevenNetCalculator.__init__` (`sevenn/calculator.py:83-86`). `sevennet.py`
  filters exactly those lines via `_without_sevennet_debug_prints()` and
  forwards everything else. If sevenn removes the prints, the filter becomes a
  no-op and can be deleted.
- **NequIP OAM** models use float64 buffers; PyTorch MPS has no float64, so
  `device="mps"` fails with `Cannot convert a MPS Tensor to float64`.
- **fairchem-core** asserts `device in {"cpu", "cuda"}`, so UMA rejects MPS
  before this package's own check would matter.
- **CHGNet** uses `use_device=`, not `device=`, and needs
  `CHGNet.load(model_name=...)` when a named model is requested.

## Pitfalls seen in practice

- Copying `.venv/bin/python` from docs into an environment that has no `.venv`.
  Use the interpreter that is actually present.
- Guessing keyword names (`model_name=`, `calculator=`, `xc=`). The accepted
  keywords per backend are tabulated in the README "API Reference" section and
  defined in each `create_calculator()` signature.
- Widening a range in `pyproject.toml` to make a conflict go away. The upper
  bounds mark "not tested above this"; report the conflict instead.
- Adding an exact `==` pin to `pyproject.toml`, or editing `constraints.txt`
  without also checking the range in `pyproject.toml` still contains it.
- Editing `docs/models.md` or `dispersion.py` alone (see invariant 4).
