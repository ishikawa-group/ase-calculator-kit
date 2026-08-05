# Releasing to PyPI and Zenodo

This is the whole path from "code is ready" to "there is a PyPI page and a DOI".
Steps marked **(browser)** cannot be automated — they need a logged-in session.

The repository already ships the automation: `.github/workflows/release.yml`
builds the sdist/wheel on every `v*` tag, checks that the tag matches the
version in `pyproject.toml`, and uploads via PyPI Trusted Publishing (OIDC), so
no API token is ever stored here.

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
a good safety net: a tag push then waits for a human click before uploading.)

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

Add the release date to `CITATION.cff` first (`date-released: "YYYY-MM-DD"`,
directly under `version:`) and commit it, so the archived tarball carries it.

```bash
git tag -a v0.3.2 -m "ase-calculator-kit 0.3.2"
git push origin v0.3.2
```

Then **(browser)** publish a GitHub Release for that tag (Releases → Draft a new
release → choose `v0.3.2` → paste the `CHANGELOG.md` entry → Publish).

Both halves are needed and they do different things:

- the **tag push** triggers `release.yml` → PyPI;
- the **published GitHub Release** fires the webhook → Zenodo deposits the
  tarball and mints the DOI.

## 5. Record the DOI

Zenodo mints two DOIs:

- a **concept DOI** that always resolves to the newest version — this is the one
  to put in the README badge and in citations;
- a **version DOI** for this specific release.

Fill both TODO blocks left in the tree:

- `README.md` — the DOI badge near the title, and the BibTeX under `## Citation`.
- `CITATION.cff` — the `identifiers:` block with the concept DOI.

Commit those to `main`. They apply from the *next* release onward, which is
normal and expected: the archived tarball of the release that created the DOI
cannot contain its own DOI.

## Every subsequent release

1. Bump `version` in `pyproject.toml` **and** the assertion in
   `tests/test_packaging.py::test_release_version_is_...`.
2. Update `version` and `date-released` in `CITATION.cff`.
3. Add a `CHANGELOG.md` entry.
4. `git tag -a vX.Y.Z && git push origin vX.Y.Z`, then publish the GitHub
   Release.

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
