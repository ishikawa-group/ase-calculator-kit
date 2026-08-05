# Releasing to PyPI and Zenodo

This is the whole path from "code is ready" to "there is a PyPI page and a DOI".
Steps marked **(browser)** cannot be automated — they need a logged-in session.

The repository already ships the automation: publishing a GitHub Release runs
the test suite, builds the sdist and wheel from the tag, and uploads them via
PyPI Trusted Publishing (OIDC), so no API token is stored here. The same event
is what Zenodo's webhook archives, so one action produces both the release and
its DOI.

Steps 1 and 3 are one-time setup and are **already done** — they are kept here
for the record, and for whoever sets this up on the next project. A routine
release starts at step 0 and runs through step 4.

---

## 0. Author metadata

Taishiro Wakamiya and Atsushi Ishikawa, in that order, are recorded in three
places — keep them in sync:

| File | Field |
| --- | --- |
| `CITATION.cff` | `authors:` (`family-names` / `given-names`) |
| `.zenodo.json` | `creators:` (`"Family, Given"`) |
| `pyproject.toml` | `authors = [...]` — becomes the Author field on the PyPI page |

Both are affiliated with the Institute of Science Tokyo; the ORCID on record is
Ishikawa's (`0000-0001-6908-831X`). Still open, and worth filling **before** the
first tag: Wakamiya's ORCID, and a contact `email` in `pyproject.toml`. Zenodo
copies these into a permanent citation record, so later corrections mean editing
the deposit by hand.

Note that `.zenodo.json`, when present, overrides whatever Zenodo would infer
from `CITATION.cff`, and both must be committed *before* the GitHub release is
published.

## 1. One-time PyPI setup **(browser)**

The name `ase-calculator-kit` is currently unclaimed on both PyPI and TestPyPI.

Register a **pending publisher** at
<https://pypi.org/manage/account/publishing/> — this works before the project
exists, so no manual first upload is needed:

- PyPI Project Name: `ase-calculator-kit`
- Owner: `ishikawa-group`
- Repository name: `ase-calculator-kit`
- Workflow name: `release.yml`
- Environment name: `pypi`

Then create the matching GitHub environment: repository → Settings →
Environments → **New environment** → `pypi`. (Adding required reviewers there is
a good safety net: the upload then waits for a human click.)

For the optional dry run, repeat both steps on <https://test.pypi.org/> with
environment name `testpypi`.

## 2. Optional dry run to TestPyPI

Actions → Release → **Run workflow** → `target: testpypi`. Then check the
install works from a clean environment:

```bash
python -m venv /tmp/akit && /tmp/akit/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ase-calculator-kit
```

To inspect the artifacts locally instead:

```bash
python -m pip install --upgrade build twine && python -m build && twine check dist/*
```

## 3. One-time Zenodo setup **(browser)**

1. Sign in at <https://zenodo.org/> with the GitHub account that owns the repo
   (or that has admin rights on `ishikawa-group`).
2. Go to <https://zenodo.org/account/settings/github/>, press **Sync now**, and
   flip the toggle for `ishikawa-group/ase-calculator-kit` to **On**.

This installs a webhook. Zenodo only archives releases created **after** the
toggle is on — an existing tag will not be picked up retroactively.

## 4. Cut the release

Commit the release notes first — a `## X.Y.Z` section in `CHANGELOG.md`, plus
the matching `version` and `date-released` in `CITATION.cff`, which a test
checks against each other. There is no version to bump in `pyproject.toml`:
`setuptools-scm` reads it from the tag.

```bash
git tag -a vX.Y.Z -m "ase-calculator-kit X.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "ase-calculator-kit vX.Y.Z" --notes "..."
```

Publishing the release is the whole release. It runs `ci.yml`, builds from the
tag, uploads to PyPI, and — through the webhook — hands the tarball to Zenodo.
Nothing is uploaded if the suite fails.

The tag itself does nothing until the release is published, so a stray tag
cannot ship anything.

To rehearse without touching PyPI, run the workflow by hand: a
`workflow_dispatch` run always targets TestPyPI, whatever the ref.

```bash
gh workflow run release.yml --ref main
```

## 5. Record the DOI

Zenodo mints two DOIs:

- a **concept DOI** that always resolves to the newest version — this is the one
  in the README badge and in citations;
- a **version DOI** for each specific release.

Both are already recorded, from the 0.3.4 archive that first created them:

| | DOI |
| --- | --- |
| Concept (all versions) | [`10.5281/zenodo.21807793`](https://doi.org/10.5281/zenodo.21807793) |
| 0.3.4 | `10.5281/zenodo.21807794` |

The concept DOI never changes, so nothing here needs editing per release — only
the version DOI differs, and Zenodo assigns it automatically. Note that the
0.3.4 tarball does not itself contain its DOI: the archive of the release that
creates a DOI cannot cite it. That resolves from 0.3.5 onward.

## Every subsequent release

1. Add a `## X.Y.Z` section to `CHANGELOG.md`.
2. Set the matching `version` and `date-released` in `CITATION.cff`.
3. Tag, push the tag, and publish the GitHub Release (step 4 above).

Nothing else carries a version number.

## Notes

- **Versions are permanent on PyPI.** A number can be yanked but never reused,
  so prefer the TestPyPI dry run over a corrective patch release.
- **The default install carries no NNP backend** — every one of them is an
  extra, so `pip install ase-calculator-kit` stays torch-free. Keep it that way:
  promoting a backend to a base dependency would make a plain install pull
  hundreds of megabytes of PyTorch.
- The published metadata deliberately uses compatible ranges; the exact tested
  combination stays in `constraints.txt`. `tests/test_packaging.py` enforces
  this, so do not "helpfully" pin `==` in `pyproject.toml`.
