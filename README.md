# SQD_Agent

Sample-based Quantum Diagonalization (SQD) pipeline packaged for reproducible runs on molecules using **Qiskit**, **ffsim**, **PySCF**, and **qiskit-addon-sqd**.  
This repo is designed to be developer-friendly (modular Python package, unit tests, examples) and agent-friendly (integrates with [OpenCode](https://github.com/opencode-ai/opencode) for AI-assisted development).

---

## Quickstart (WSL)

```bash
cd /mnt/d/project/SQD_Agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r requirements.txt
pip install -e .
```

## Repository structure

```

SQD_Agent/
├─ sqd/                   # Core Python package
│  ├─ chemistry.py        # PySCF RHF/MP2/CCSD/CASCI helpers
│  ├─ ansatz.py           # HF, UCJ, LUCJ proxy, HE ansatz builders
│  ├─ active_space.py     # Active space selection and t2 slicing
│  ├─ runner.py           # SamplerV2 sampling + SQD diagonalization loop
│  ├─ compare.py          # Benchmark wrapper with pretty tables
│  ├─ cli.py              # Typer CLI entrypoints
│  └─ __init__.py
├─ examples/              # Ready-to-run molecule demos
│  ├─ run_lithium_hydride.py
│  ├─ run_water.py
│  └─ benchmark_suite.py
├─ data/
│  └─ molecules.json      # Cached geometries + defaults
├─ tests/                 # Pytest unit tests
│  ├─ test_active_space.py
│  ├─ test_ansatz_shapes.py
│  └─ test_runner_toy.py
├─ .vscode/               # Tasks & launch configs for VS Code
├─ .opencode/             # (Optional) OpenCode agent config
├─ requirements.txt
├─ pyproject.toml
└─ README.md

````


## Setup (WSL Ubuntu recommended)

### 1. Install system deps
```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential cmake gfortran libopenblas-dev
````

### 2. Create and activate venv (virtual environment for WSL)

```bash
cd /mnt/d/project/SQD_Agent   # adjust path if needed
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python deps

```bash
pip install -U pip wheel
pip install "qiskit>=1.0" "qiskit-aer>=0.14" qiskit-addon-sqd ffsim pyscf matplotlib typer pytest
```

### 4. (Optional) Limit BLAS threads

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

### 5. Verify imports

```bash
python - << 'PY'
import qiskit, qiskit_aer, qiskit_addon_sqd, ffsim, pyscf
print("Imports OK")
PY
```

### 6. Install project in editable mode

```bash
# ensure __init__.py exists
test -f sqd/__init__.py || printf '__all__ = []\n' > sqd/__init__.py

# add minimal pyproject.toml (if missing)
cat > pyproject.toml << 'EOF'
[project]
name = "sqd-project"
version = "0.1.0"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["sqd"]
EOF

# install locally
pip install -e .
```

---

## Run tests

```bash
python -m pytest -q
```

---

## Example runs

### LiH quick test

```bash
python examples/run_lithium_hydride.py --ansatz ucj --shots 200000
```

### Water molecule (all ansätze)

```bash
python examples/run_water.py --ansatz all
```

### N₂ benchmark

```bash
python -m sqd.cli bench --geom "N 0 0 -0.55; N 0 0 0.55" --basis sto-3g --ansatz all --shots 200000 --samples-per-batch 250
```

### Batch suite

```bash
python examples/benchmark_suite.py --cases N2_1p10A,LiH,H2O --ansatz all
```

## VS Code integration

* `.vscode/tasks.json`:

  * `venv: install` → install requirements
  * `sqd: run example (LiH)`
  * `sqd: bench N2 (all ansatz)`
  * `tests` → run pytest

* `.vscode/launch.json`: debug configs for `sqd.run` and `sqd.bench`.


## OpenCode integration (optional)

For AI-assisted development:

install and steup opencode through : https://github.com/opencode-ai/opencode#

if needed
```bash
sudo apt update
sudo apt install unzip
```
otherwise directly use these commands
```bash
curl -fsSL https://opencode.ai/install | bash
opencode
/init
opencode auth login   # google / openai / anthropic etc.
```

* Config: `.opencode/opencode.json` #if needed
* Custom agent: `.opencode/agents/sqd-dev.md` #if needed
* 
* Example use:

  ```
  /sqd-dev
  @sqd/active_space.py improve choose_active_window for open-shell cases
  ```

---

## Tips

* Use WSL Ubuntu for best compatibility.
* Adjust `OMP_NUM_THREADS` for performance on laptops.
* Add new molecules to `data/molecules.json`.
* Pin versions in `requirements.txt` for reproducibility.

---

## References

* Setup commands adapted from `SQD on WSL running commands.txt`.
* Libraries: [qiskit-addon-sqd](https://github.com/qiskit-community/qiskit-addon-sqd), [ffsim](https://github.com/qiskit-community/ffsim), [PySCF](https://pyscf.org).
