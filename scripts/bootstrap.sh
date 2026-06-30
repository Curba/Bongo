#!/usr/bin/env bash
# One-command setup: clone YAQS (char branch), create venv, install editable, smoke test.
# Safe to re-run: skips steps whose result already exists.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}==>${NC} $1"; }
warn()  { echo -e "${YELLOW}==>${NC} $1"; }
error() { echo -e "${RED}==>${NC} $1" >&2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YAQS_DIR="$(cd "${REPO_ROOT}/.." && pwd)/yaqs"
YAQS_REMOTE="https://github.com/munich-quantum-toolkit/yaqs.git"
YAQS_BRANCH="char"

cd "${REPO_ROOT}"

# --- 1. Check prerequisites -------------------------------------------------

if ! command -v git >/dev/null 2>&1; then
    error "git not found. Install git and re-run this script."
    exit 1
fi
info "git found: $(git --version)"

if ! command -v python3 >/dev/null 2>&1; then
    error "python3 not found. Install Python 3.11+ and re-run this script."
    exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
PY_MAJOR="$(echo "${PY_VERSION}" | cut -d. -f1)"
PY_MINOR="$(echo "${PY_VERSION}" | cut -d. -f2)"
if [ "${PY_MAJOR}" -lt 3 ] || { [ "${PY_MAJOR}" -eq 3 ] && [ "${PY_MINOR}" -lt 11 ]; }; then
    error "Python 3.11+ required, found ${PY_VERSION}."
    exit 1
fi
info "python3 found: ${PY_VERSION}"

if ! command -v uv >/dev/null 2>&1; then
    warn "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        error "uv install completed but 'uv' is still not on PATH. Open a new shell or add ~/.local/bin to PATH, then re-run."
        exit 1
    fi
fi
info "uv found: $(uv --version)"

# --- 2. Ensure sibling YAQS checkout exists ---------------------------------

if [ ! -d "${YAQS_DIR}" ]; then
    warn "${YAQS_DIR} not found. Cloning ${YAQS_REMOTE} (branch ${YAQS_BRANCH})..."
    git clone --branch "${YAQS_BRANCH}" "${YAQS_REMOTE}" "${YAQS_DIR}"
    info "Cloned YAQS to ${YAQS_DIR}"
else
    info "Found existing YAQS checkout at ${YAQS_DIR}"
    CURRENT_BRANCH="$(git -C "${YAQS_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    if [ "${CURRENT_BRANCH}" != "${YAQS_BRANCH}" ]; then
        warn "${YAQS_DIR} is on branch '${CURRENT_BRANCH}', not '${YAQS_BRANCH}'."
        warn "qel_twin/physics/ targets the API on '${YAQS_BRANCH}' (custom MPO/MPS/simulator.run, characterization module)."
        warn "If this is unexpected, run: git -C ${YAQS_DIR} checkout ${YAQS_BRANCH}"
    fi
fi

# --- 3. Create venv ----------------------------------------------------------

if [ ! -x "${REPO_ROOT}/.venv/bin/python" ]; then
    info "Creating virtual environment at ${REPO_ROOT}/.venv ..."
    uv venv --python 3.11 "${REPO_ROOT}/.venv"
else
    info "Virtual environment already exists at ${REPO_ROOT}/.venv, skipping creation."
fi

export VIRTUAL_ENV="${REPO_ROOT}/.venv"
export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"

# --- 4. Install dependencies --------------------------------------------------

info "Installing ${YAQS_DIR} (editable)..."
uv pip install --python "${REPO_ROOT}/.venv/bin/python" -e "${YAQS_DIR}"

info "Installing ${REPO_ROOT} (editable)..."
uv pip install --python "${REPO_ROOT}/.venv/bin/python" -e "${REPO_ROOT}"

# --- 5. Smoke test -------------------------------------------------------------

info "Running smoke test..."
if "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/scripts/smoke_test.py"; then
    info "Smoke test passed. Bootstrap complete."
    info "Activate the environment with: source ${REPO_ROOT}/.venv/bin/activate"
else
    error "Smoke test failed. See output above for details."
    exit 1
fi
