#!/usr/bin/env bash
set -euo pipefail

# Usage: bash hpc/bootstrap.sh [environment-directory]
ENV_DIR="${1:-$HOME/.venvs/upgd}"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[hf,test]"
python -m pytest -q
echo "Environment ready: $ENV_DIR"
