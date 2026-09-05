# Bongo
Quantum Hardware Digital Twin for Noisy Workloads
```
      .-------------.           .-------------.
     /               \         /               \
    |=================|-------|=================|
    |                 |       |                 |
     \               /         \               /
      \             /           \             /
       \           /             \           /
        |_________|               |_________|
```

A device-specific digital twin of one quantum hardware backend, combining a Lindblad-form noise model, a YAQS simulation engine, and an ML surrogate that solves the inverse problem of recovering noise parameters from observable trajectories. The pitch frame is "Synopsys/Cadence for quantum hardware" — software you develop against before paying for real QPU time.

## Prerequisites

- Python 3.11+
- git
- bash (Linux/macOS/WSL)

## Quick Start

```bash
git clone https://github.com/Curba/Bongo.git qel-digital-twin
cd qel-digital-twin
bash scripts/bootstrap.sh
source .venv/bin/activate
```

`bootstrap.sh` clones the YAQS `main` branch as a sibling directory (`../yaqs/`), creates a `.venv`, installs both packages editable, and runs a smoke test.

Run Python commands from the repository root. The checked-in
`.numba_config.yaml` directs YAQS/Numba compilation artifacts to the writable,
gitignored `.cache/numba/` directory. `bootstrap.sh` creates that directory and
exports the same setting for its smoke test, including in environments where the
editable sibling YAQS checkout is read-only.
