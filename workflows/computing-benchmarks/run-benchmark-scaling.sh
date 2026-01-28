#!/bin/bash
#SBATCH --job-name=benchmark-scaling
#SBATCH --partition=shared
#SBATCH --output=output/logs/benchmark-scaling-%j.out
#SBATCH --error=output/logs/benchmark-scaling-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=32G
#SBATCH --account=ees250129

# Exit on error (but allow unset variables for conda/micromamba operations)
set -eo pipefail

# Get the directory where this script is located
# Use SLURM_SUBMIT_DIR if available (set by sbatch), otherwise use script location
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

# Create output/logs directory if it doesn't exist
# Note: SLURM will create the directory when writing output files, but creating it
# here ensures it exists for any other operations
mkdir -p output/logs

# Verify we're in the right directory and files exist
if [[ ! -f "benchmark_scaling.py" ]]; then
    echo "Error: benchmark_scaling.py not found in $SCRIPT_DIR" >&2
    echo "Current directory: $(pwd)" >&2
    echo "Files in directory:" >&2
    ls -la >&2
    exit 1
fi

# Activate conda environment
# Allow unset variables for conda operations
set +u
ENV_NAME="cson-forge-v0"

# Load conda via module system
module load conda

# Initialize conda for this shell session
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate "$ENV_NAME"

# Ensure we use the conda environment's Python (not system Python)
# Get the Python executable from the activated environment
PYTHON_EXE="$(which python)"
if [[ -z "$PYTHON_EXE" ]] || [[ ! -f "$PYTHON_EXE" ]]; then
    # Fallback: use Python from CONDA_PREFIX
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON_EXE="$CONDA_PREFIX/bin/python"
    else
        echo "Error: Could not find Python in conda environment $ENV_NAME" >&2
        exit 1
    fi
fi

# Clear PYTHONPATH to avoid conflicts with system-installed packages
# The conda environment's site-packages should be in sys.path automatically
unset PYTHONPATH

echo "Using Python: $PYTHON_EXE"
echo "Python version: $($PYTHON_EXE --version)"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"

# Restore strict error checking
set -u

clobber_inputs_flag=
#clobber_inputs_flag="--clobber-inputs"

# Loop over ensemble IDs
for ensemble_id in 2 3 4; do
    echo "=========================================="
    echo "Running benchmark scaling for ensemble_id=${ensemble_id}"
    echo "Current directory: $(pwd)"
    echo "=========================================="
    
    "$PYTHON_EXE" "$SCRIPT_DIR/benchmark_scaling.py" \
        --ensemble-id "${ensemble_id}" \
        --domains-file "$SCRIPT_DIR/domains-bm-scaling.yml" \
        ${clobber_inputs_flag}
    
    echo ""
    echo "Completed ensemble_id=${ensemble_id}"
    echo ""
done

echo "=========================================="
echo "All ensemble runs completed"
echo "=========================================="
