#!/usr/bin/env bash
set -euo pipefail

# DI-CC (Decomposition-Induced Context-Memory Conflict) -- first-time setup
# Usage: ./setup.sh

echo "=== DI-CC Setup ==="

# --- Prerequisites ---
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required."; exit 1; }

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
  echo "Error: Python 3.10+ is required (found $PYTHON_VERSION). The codebase uses PEP 604 union-type syntax (e.g. 'int | None')."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi not found. This pipeline requires a CUDA GPU"
  echo "         (bitsandbytes 4-bit quantization for the Qwen2.5 decomposer"
  echo "         model). Stages that only touch already-extracted activations"
  echo "         (03, 07, 30, 31) can run on CPU, but 02/04/05/06/24/25/26"
  echo "         need a GPU."
fi

# --- Virtual environment ---
if [ ! -d .venv ]; then
  echo "Creating virtual environment in .venv..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# --- Dependencies ---
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source .venv/bin/activate"
echo "  2. Reproduce the headline H0 result (needs a CUDA GPU for stages"
echo "     02/04/05/06; skip stage 04 and copy the frozen"
echo "     data/factscore_subclaims_40person.jsonl to"
echo "     data/processed/factscore_subclaims.jsonl to reproduce exactly):"
echo "       python scripts/01_build_nq_swap.py"
echo "       python scripts/02_extract_nqswap_activations.py"
echo "       python scripts/03_train_probe.py"
echo "       python scripts/04_run_decomposition.py   # or use the frozen data file"
echo "       python scripts/05_label_spans.py"
echo "       python scripts/06_extract_decomp_activations.py"
echo "       python scripts/07_evaluate_transfer.py"
echo "  3. See README.md for the full pipeline (including the H3 CAD"
echo "     mitigation path, stages 24-27 and 30-31) and headline results."
