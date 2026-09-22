# AGENTS.md

Working notes for AI coding agents (Claude Code, Codex/GPT, Copilot, and
others) on `ase-calculator-kit`. Human-facing documentation lives in
[`README.md`](README.md); this file holds the repository map, invariants, and
conventions that are easy to violate without reading the whole codebase.

Read this file before editing. If a change contradicts an invariant below,
say so explicitly instead of silently changing the behavior.

## What this package is

A thin factory layer. It does **not** implement any physics: it maps a name
plus keywords onto an upstream ASE calculator (`chgnet`, `matgl`, `sevenn`, `mattersim`,
`nequip`, `fairchem-core`, `ase.calculators.vasp`, `ase.calculators.espresso`)
and returns an ASE calculator. `orb-models` is in that list too. New behavior
belongs upstream unless it is about *selection*, *validation*, or
*reproducibility*.
The molecular PySCF/GPU4PySCF adapter is a user-authorized exception to
returning an existing upstream ASE calculator: it converts ASE structures and
units, and delegates all energies/analytic gradients to upstream. It supports
nonperiodic molecules only, reads explicit charge and spin/multiplicity from YAML and/or atoms.info, and
never silently falls back from GPU to CPU. Factory settings remain YAML-only; atoms.info carries structure-specific state.
Conflicting electronic states raise before cache reuse; info-only state changes
must invalidate the PySCF cache. atoms.info spin is multiplicity, YAML spin is 2S.
PySCF runs in-process, so no external profile.command is required. Its optional
extras stay outside `all`; CUDA 12 wheels target Linux x86_64 and Python 3.12/3.13.

The explicit temporary exception is `matgl_chgnet.py`: a user-authorized,
source-guarded, instance-local three-body gradient correction for MatGL 4.0.3.
Replace it after validating upstream retrained CHGNet weights.
The user-authorized `_matgl_pbc.py` input adapter also makes TensorNet and
MatGL CHGNet honor each axis of `atoms.pbc`, including when D3 is enabled.
It retains the upstream full-PBC graph path and never changes model weights.

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
  backends/mlip/     chgnet.py matgl.py sevennet.py mattersim.py nequip.py fairchem.py
                     matgl_chgnet.py, _matgl_pbc.py (MatGL compatibility adapters)
                     orb.py (OrbMol-v2; outside `all` — invariant 11)
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
   is `True` for CHGNet, TensorNet / provisional CHGNet (MatGL), SevenNet, and
   MatterSim, because those were validated with a real single point on Apple Silicon. Do not flip a flag
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
11. **The `orb` extra is outside `all` for a Python-version reason, not a
    conflict.** Since 0.5.7 the backend exists and `orb-models` co-installs
    happily with every other backend — no e3nn problem, no second environment.
    What it cannot do is install on Python 3.13 or 3.14: it pins
    `dm-tree==0.1.8`, whose newest wheels are cp312, so an `all` containing it
    would stop installing on two of the three supported interpreters. That is
    why the extra exists and why `all` must keep not containing it
    (`tests/test_packaging.py` enforces it, and the `extras-resolve` job's
    second step pins the wheel reality with `--only-binary dm-tree`, because a
    plain resolve is happy to plan a source build). Do not paper over this with
    an environment marker — invariant 10 applies. When upstream relaxes the pin
    (orbital-materials/orb-models#168), the CI step fails, and *that* is when
    the README table and this invariant get revisited.
12. **The MACE backend selects upstream loaders by model name.** Since
    0.5.4, `POLAR_MODELS` (`polar-1-s/m/l`) load through `mace_polar`;
    since 0.5.10, `omol-0` loads through `mace_omol(model="extra_large")`. Other
    models use `mace_mp`. These functions do not accept each other's
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

# orb (invariant 11) needs Python 3.12; it shares the main environment
# otherwise. On 3.13+ add an override for orb-models' dm-tree pin.
uv venv --python 3.12 .venv-orb
uv pip install --python .venv-orb/bin/python -e ".[orb,dev]" -c constraints.txt
.venv-orb/bin/pytest -m slow -k orb

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

- **UMA's `"default"` inference preset is already the fast path; `"turbo"` only
  adds TF32.** In fairchem-core 2.22 `inference_settings_default()` sets
  `merge_mole=True, compile=True`, and `inference_settings_turbo()` is the same
  dataclass with `tf32=True`. Earlier fairchem releases made `turbo` the one
  that enabled compilation, so "switch to turbo for MD" is stale advice worth
  not repeating. `turbo` stays opt-in here for the same reason MACE defaults to
  float64: this package exists to compare models, and a silently reduced
  precision corrupts the comparison. Measured on an H100 (27-atom fcc Cu,
  `task="omat"`): `turbo` 12.8 ms/step against `default` 13.5 ms, for 0.137
  meV/atom of energy shift — a 5 % gain bought with an error the size of a
  model difference. The same run shows the fast path itself is faithful:
  `"batch"`, with neither merge nor compile, lands within 0.005 meV of
  `default`. `merge_mole` also assumes fixed
  composition/task/charge/spin — true for MD, false for a loop over structures,
  where fairchem logs a fallback and `"batch"` is the right preset. The four
  names are validated in `backends/mlip/fairchem.py` because upstream checks
  them with a bare `assert`, which `python -O` strips.

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

- **orb-models pins `dm-tree==0.1.8`, and that is the whole Python-version
  story.** dm-tree 0.1.8's newest wheels are cp312 on every platform, so on
  3.13+ pip falls into a source build that needs a C++ toolchain and a CMake old
  enough to accept the project (orbital-materials/orb-models#168; the pin itself
  came from orbital-materials/orb-models#78). Nothing about the model is
  3.12-only: orb-models uses exactly `tree.flatten` and `tree.map_structure`,
  dm-tree 0.1.10 ships cp313 and cp314 wheels, and OrbMol-v2 on 3.13 with
  dm-tree 0.1.10 reproduced all sixteen 3.12 reference energies to the last
  digit. pip has no dependency-override mechanism and PyPI forbids direct
  references in uploaded metadata, so this cannot be fixed in our metadata —
  the README documents `uv pip install --override` instead. Do not fork
  orb-models for one line.

- **OrbMol-v2 cannot run on MPS, and the reason is three deep.** Graph
  construction defaults to `knn_alchemi`, which goes through `nvalchemiops` and
  NVIDIA Warp; Warp has no Metal backend and raises `Unsupported Torch device
  type mps`. The legacy fallbacks do not rescue it — `knn_brute_force` builds
  the neighbor list in float64, which MPS cannot hold, and `knn_scipy` rejects
  any non-CPU device itself. Measured on Apple Silicon, all three. There is also
  a loader trap: `pretrained.orbmol_v2(device=...)` calls `model.cuda(device)`
  for anything that is not CPU, so even `device="mps"` fails before the graph.
  `resolve_device(..., allow_mps=False)` is therefore correct here, per
  invariant 6 — do not flip it without re-measuring all three paths.

- **`orb` deliberately loads one checkpoint.** `ORB_PRETRAINED_MODELS` carries
  the Orb-v3 OMat/MPA models, Orb-v2, the D3-trained `orb-d3-*` variants and
  OrbMol-v1 as well. Each is a different reference level — `orb-d3-v2` is
  trained *on* D3-corrected targets and would belong in the ⛔ tier for a
  different reason than OrbMol-v2 — so adding one means adding its
  `dispersion.py` row and `docs/models.md` row together. Until then an
  unsupported name is a `ValueError`, not a pass-through.

- **MatGL exposes TensorNet and provisional gradient-corrected CHGNet MatPES.**
  `matgl-chgnet` requires the audited MatGL 4.0.3 source, defaults to frozen HF
  revisions, and removes only the three-body geometry `no_grad()` block on the
  loaded instance. This explicitly authorized temporary exception must be
  replaced after validation of upstream retrained checkpoints, with release
  notes recording changed weights and behavior. Do not silently accept a new
  source or globally patch MatGL. M3GNet remains deferred due to pruned-to-parent
  bond indexing. See `docs/matgl-validation.md` before changing either path.

- **MatGL 4.0.3 treats partial PBC as fully nonperiodic** on its usual graph
  conversion path. `_matgl_pbc.py` uses ASE neighbor lists and lattice image
  offsets for partial/nonperiodic inputs. Keep `atoms.pbc`, positions and cell
  unchanged. Complete missing nonperiodic cell vectors only internally; this
  also permits torch-dftd E/F calculations, but stress requires a real 3D cell.
  Do not replace partial PBC with `pbc=True` or expose stress normalized by an
  artificial completed volume. The adapter applies only to MatGL backends.

- **eSEN OMol25 is separate from UMA and legacy OMat.** `esen` accepts the three
  OMol25 registry checkpoints, defaults to `esen-sm-conserving-all-omol`, fixes
  task to omol and defaults inference_settings to batch. Its `esen` extra shares
  fairchem-core with UMA and `all`; D3 is refused for the OMol reference level.

- Since 0.5.10, UMA/eSEN omol and molecular MACE use an instance-local
  validation subclass: explicit charge/multiplicity is required, with integer
  and electron-count checks before inference and cache reuse. Do not restore
  upstream's neutral-singlet fallback. MACE custom info_keys are respected.
- MACE-OMOL-0 has its own upstream loader/head; never route it through mace_mp
  or pass a second head argument to mace_omol. MACE-POLAR-1 names the medium
  checkpoint; explicit small/medium/large names remain available.
- PySCF may retain an energy-only SCF and its stream until forces or reset.
  All structure/state changes invalidate it; failures must close the stream.
- External DFT configs reject unknown envelope/profile keys; QE namelists use
  ASE's supported-key table. Do not silently drop an unknown YAML field.
