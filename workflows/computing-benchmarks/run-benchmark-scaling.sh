#!/bin/bash
#SBATCH --job-name=benchmark-scaling
#SBATCH --partition=shared
#SBATCH --output=benchmark-scaling-%j.out
#SBATCH --error=benchmark-scaling-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --account=ees250129

# Exit on error
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cson-forge-v0 2>/dev/null || true
fi

clobber_inputs_flag=
#clobber_inputs_flag="--clobber-inputs"

# Loop over ensemble IDs
for ensemble_id in 2 3 4; do
    echo "=========================================="
    echo "Running benchmark scaling for ensemble_id=${ensemble_id}"
    echo "=========================================="
    
    python benchmark_scaling.py \
        --ensemble-id "${ensemble_id}" \
        --domains-file domains-bm-scaling.yml \
        ${clobber_inputs_flag}
    
    echo ""
    echo "Completed ensemble_id=${ensemble_id}"
    echo ""
done

echo "=========================================="
echo "All ensemble runs completed"
echo "=========================================="
