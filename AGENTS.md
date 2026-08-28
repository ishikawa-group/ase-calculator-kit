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
                     mace.py (separate environment — invariant 7)
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
`errors.py` (extra mapping), tests, README, `docs/models.md`, and the
`extras-resolve` expectations in `.github/workflows/ci.yml`.

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
   sync; changing one without the other is a bug, and
   `tests/test_models_doc_sync.py` now enforces it in both directions. Write
   every policy key in backticks in the table's first column — that is what the
   parser reads.
5. **`dispersion=True` changes the return type** to
   `SumCalculator([backend_calc, d3_calc])`. Anything that assumes the backend
   class comes back is wrong.
6. **MPS support is measured, not assumed.** `resolve_device(..., allow_mps=)`
   is `True` only for CHGNet, SevenNet, and MatterSim, because those were
   validated with a real single point on Apple Silicon. Do not flip a flag
   without running the calculation; record the result in the README matrix.
7. **MACE ships, but never in the same environment.** `mace-torch` pins
   `e3nn==0.4.4`; `sevenn`, `fairchem-core` and `mattersim` require
   `e3nn>=0.5.0` and `nequip` `>=0.6.0`. Since 0.5.0 the backend exists and the
   answer to the conflict is a second virtual environment, not exclusion — so
   the `mace` extra must stay **out of `all`** (`tests/test_packaging.py`
   enforces it), `MissingDependencyError` must keep explaining *why* a second
   environment is needed, and CI resolves `mace` on its own. Do not "fix" the
   conflict by relaxing another backend's `e3nn` floor.
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
10. **`requires-python` carries no upper bound.** A cap is written into every
    file uploaded to PyPI and cannot be edited afterwards, so it makes the
    package invisible to a newer interpreter until a fresh release goes out —
    even when the code runs there fine. Backends that lag a Python release cap
    themselves — `fairchem-core` declared `<3.14` until 2.22.0 lifted it, and as
    of 0.5.4 no backend caps below 3.14 — and pip then names the backend in the
    error. Do not paper over that with an environment marker on the extra: a
    marker makes the install *succeed* while silently omitting the backend.
    `tests/test_packaging.py` enforces the absence of a cap, and the
    `extras-resolve` CI job (expectations mirroring the README table) is what
    catches the change in either direction.
11. **The MACE backend has two upstream loaders, chosen by model name.** Since
    0.5.4, `POLAR_MODELS` (`polar-1-s/m/l`) load through `mace_polar` and
    everything else through `mace_mp`. Neither function accepts the other's
    checkpoint names, so the name *is* the selector and there is no keyword to
    add. `mace_polar` is imported only on the polar path, so an install predating
    it reports a missing mace-torch rather than a missing MACE-MP.

## Conventions

- Python `>=3.12`, with no upper bound — see invariant 10. `from __future__
  import annotations` at the top of every module; modern typing
  (`str | None`, `dict[str, Any]`).
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

# MACE (invariant 7) is verified from its own environment. The fast suite is
# identical there; in the slow suite the MACE cases run and every other case
# skips as "backend not installed", which is exactly the intended split.
python -m venv .venv-mace
.venv-mace/bin/pip install -e ".[mace,dispersion,dev]" -c constraints.txt
.venv-mace/bin/pytest -m slow -k mace
```

- `addopts = "-m 'not slow'"` in `pyproject.toml` deselects the slow suite by
  default. Do not run `-m slow` casually: it downloads model weights (GBs) and
  needs `.[dev,all]`.
- CI (`.github/workflows/ci.yml`) runs the fast suite plus `ruff check` on
  Python 3.12, 3.13, and 3.14. Assume no GPU and no Apple Silicon in CI.
- A second job, `extras-resolve`, resolves every extra against every supported
  Python with `uv pip compile` (resolution only — nothing is downloaded or
  installed). Its expectation list mirrors the "Python versions" table in the
  README, so it fails both when an extra stops resolving and when one we
  document as unavailable starts working. In the latter case, update the README
  table and the expectation list together.
- Backend tests inject fake modules into `sys.modules`
  (see `tests/test_sevennet_backend.py`) rather than importing real NNP
  packages. Follow that pattern for new backends — fast tests must not download
  anything.
- Do not commit or push unless asked.

## Releasing

Never cut a release unless explicitly asked. `docs/releasing.md` has the full
path, including the one-time PyPI and Zenodo setup; this is what an agent needs
in order not to break it.

**Publishing the GitHub Release is the entire release.** That one event runs
`ci.yml`, builds from the tag, uploads to PyPI via Trusted Publishing (OIDC, no
stored token), and — through Zenodo's webhook — mints the DOI. A pushed tag on
its own ships nothing.

```bash
# 1. Land the notes first: a `## X.Y.Z` section in CHANGELOG.md, and the
#    matching version + date-released in CITATION.cff.
git tag -a vX.Y.Z -m "ase-calculator-kit X.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "ase-calculator-kit vX.Y.Z" --notes "..."

# Rehearsal: a manual run always targets TestPyPI, whatever the ref.
gh workflow run release.yml --ref main
```

Three things make this go wrong, and all three have happened here:

1. **Do not write a version number anywhere but `CITATION.cff`.**
   `setuptools-scm` derives it from the tag. When the number lived in several
   files, they drifted — 0.3.3 shipped a `CITATION.cff` still announcing 0.3.2.
   `CITATION.cff` stays hand-written only because Zenodo reads it, and
   `test_citation_version_matches_the_newest_changelog_entry` pins it to the
   newest `CHANGELOG.md` heading.
2. **Do not upload to PyPI by any other route.** Filenames there are immutable:
   a version can be yanked but never re-uploaded, so a mistake costs a version
   number. Rehearse on TestPyPI.
3. **Zenodo only archives releases published after its webhook was installed**,
   and never retroactively. 0.3.3 has no DOI for exactly that reason. The
   concept DOI `10.5281/zenodo.21807793` always resolves to the newest version
   and never changes, so it needs no per-release edit.

## Known upstream quirks

- **eSEN-30M-OMat cannot be added to the `uma` backend** (checked 2026-08-11, so
  it does not get re-investigated). `esen_30m_omat.pt` lives in the gated repo
  `facebook/OMAT24` and is a fairchem-core **1.x** checkpoint used through
  `OCPCalculator`. fairchem-core 2.21 has no eSEN architecture
  (`fairchem/core/models/` = allscaip, base, escaip, uma, utils) and
  `pretrained_mlip.available_models` lists only UMA plus OMol25/OC25/ODAC eSEN
  checkpoints. fairchem-core 1.10 pins `torch~=2.4`, `numpy<2` and
  `requires-python <3.13`, so it needs its own environment exactly like MACE —
  a deliberate decision, not a small addition. The OMat24 reference level is
  already reachable through MACE `medium-omat-0` and SevenNet `7net-omat`.

- **A variant selector that upstream ignores must never key the D3 table.**
  sevenn accepts `modal=` on a single-fidelity model, warns
  (`modal=... is ignored as model has no modal_map`), and drops it — so until
  0.5.2 `model="7net-0", modal="matpes_r2scan"` computed a PBE model and then
  added r2SCAN dispersion parameters. MACE has the mirror image: a head name a
  single-head checkpoint does not have. Both now resolve through `"auto"`
  (`_resolve_modal`, `_resolve_head`), the *resolved* value keys
  `dispersion.py`, and an explicit selector the model cannot use is an error.
  When adding a backend, ask what its variant selector does when the checkpoint
  has no variants — silently ignored is the dangerous answer.

- **cuequivariance is not safe to enable on the strength of an import.**
  Measured on a Tesla V100 (sm_70, cuequivariance 0.11.1, mace-torch 0.3.16):
  the import succeeds, `MACECalculator` builds, and the *first energy
  evaluation* raises `cudaErrorNoKernelImageForDevice` — the wheels ship
  kernels for newer architectures only. Separately, ACEsuit/mace#1298 reports
  cuequivariance returning +5500 eV against the plain model's -200 eV on a
  multi-head checkpoint without raising at all. `accelerator="auto"` therefore
  builds both models and compares them on a two-atom cell instead of trusting
  either signal; do not "simplify" it back into an import check. The failed
  attempt does not poison the CUDA context — the fallback calculator was
  verified to keep returning correct energies afterwards.
- **MACE-Polar's checkpoints need a package pip cannot resolve.** The published
  `polar-1-*` files unpickle classes from `graph_longrange`, which `mace-torch`
  does not depend on and which is not on PyPI: it is built from
  WillBaldwin0's `graph_electrostatics` repository, and the distribution it
  produces is named `graph_longrange`, not `graph_electrostatics` — pip rejects
  the `graph_electrostatics @ git+…` spelling outright, so the documented
  command passes the bare URL. PyPI forbids direct references in uploaded
  metadata, so this can never become an extra here; `_DIRECT_INSTALLS` in
  `errors.py` exists to name it anyway, and `_require_polar_runtime()` checks
  for it before the download. Left alone the failure is `ModuleNotFoundError:
  No module named 'graph_longrange'` from inside `torch.load`, naming neither
  MACE nor this package (ACEsuit/mace#1408).

- **MACE accepts an unknown `head` and computes anyway.** `MACECalculator` logs
  `Head <x> not found in available heads [...], defaulting to the last head` and
  returns energies from that head — a typo yields a plausible number at the
  wrong level of theory, the same failure shape as fairchem's silent
  `charge=0`/`spin=1`. `backends/mlip/mace.py` therefore validates the head
  against `MH1_HEADS` before the download, and against the loaded checkpoint's
  `available_heads` afterwards. `MH1_HEADS` was read out of the shipped
  `mace-mh-1.model`; it deliberately differs from the model card, which lists an
  `rgd1_b3lyp` head the checkpoint does not carry.

- **sevenn 0.12.1** printed `cueq / <bool> / flash / <bool>` from
  `SevenNetCalculator.__init__`, and `sevennet.py` filtered exactly those lines.
  0.13 removed the prints, so with `sevenn>=0.13` the filter was dead code and
  is gone as of 0.4.0. It also prints `Converting model backend...` per
  calculator; that one is informative and is left alone.
- **RPBE's D3 parameters carry no citation.** In the reference parameter tables
  (`dftd3/simple-dftd3`), `rpbe` is one of a handful of entries whose `d3.bj`
  and `d3.zero` rows have no `doi=`, unlike `pbe` and `revpbe`. Its BJ `a1` is
  0.182 against PBE's 0.4289, which makes the correction turn on at much shorter
  range. Combined with D3's unscreened metal C6 that produces very large
  molecule-metal corrections. Do not "fix" this by substituting another
  functional's parameters — that silently renames the method.
