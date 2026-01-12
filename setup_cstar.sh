#!/bin/bash
# Script to clone C-Star, switch to orchestration branch, and install in editable mode

set -e  # Exit on error

# Initialize conda if needed
if ! command -v conda &> /dev/null; then
    echo "Error: conda command not found. Please initialize conda first."
    exit 1
fi

REPO_URL="https://github.com/CWorthy-ocean/C-Star.git"
BRANCH="orchestration"

# Determine code root - use environment variable if set, otherwise use default
# This avoids importing cson_forge which depends on cstar (not yet installed)
if [ -n "$CSTAR_CODE_ROOT" ]; then
    CODE_ROOT="$CSTAR_CODE_ROOT"
else
    # Default matches config.py _layout_unknown: home / "cson-forge-data" / "codes"
    # For CI, we can use a simpler path
    CODE_ROOT="${HOME}/cson-forge-data/codes"
fi

REPO_DIR="$CODE_ROOT/C-Star"
CONDA_ENV="cson-forge"

# Create code root directory if it doesn't exist
mkdir -p "$CODE_ROOT"

pushd "$CODE_ROOT" > /dev/null

# Clone or update the repository
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists. Updating..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    cd ..
else
    echo "Cloning C-Star repository..."
    git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# Activate conda environment and install
echo "Activating conda environment: $CONDA_ENV"
echo "Installing C-Star in editable mode..."

# Activate conda environment and run pip install
# Note: In CI, conda may be a wrapper around micromamba
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

cd "$REPO_DIR"
pip install -e .

echo "C-Star installation completed successfully!"

popd > /dev/null