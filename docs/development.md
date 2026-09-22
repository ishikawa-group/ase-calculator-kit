# Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]" -c constraints.txt
.venv/bin/pytest

# MACE has its own environment; the same suite runs there too
python -m venv .venv-mace
.venv-mace/bin/pip install -e ".[mace,dispersion,dev]" -c constraints.txt
.venv-mace/bin/pytest

# orb shares the main environment, but needs Python 3.12 (see above)
uv venv --python 3.12 .venv-orb
uv pip install --python .venv-orb/bin/python -e ".[orb,dev]" -c constraints.txt
.venv-orb/bin/pytest -m slow -k orb
```

`pyproject.toml` declares compatible version ranges so the package installs
next to whatever ASE/NNP versions you already have; `constraints.txt` pins the
exact combination that is tested, and CI installs with it.

`pytest` runs only the fast tests by default. Slow tests
(`pytest -m slow`) run real MLIP CPU single-point calculations and may download
model weights; install `.[dev,all] -c constraints.txt` first so every backend is
importable.
