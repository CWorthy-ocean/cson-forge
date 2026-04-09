#!/usr/bin/env bash
set -euo pipefail

force_recompute=false
workflow_dir=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workflow-dir)
      if [[ -z ${2:-} ]]; then
        echo "Error: --workflow-dir requires a value"
        exit 1
      fi
      workflow_dir="$2"
      shift 2
      ;;
    --force-recompute)
      force_recompute=true
      shift
      ;;
    *)
      echo "Usage: $0 --workflow-dir <directory> [--force-recompute]"
      echo ""
      echo "Arguments:"
      echo "  --workflow-dir <directory>  Directory containing workflow.yml (required)"
      echo "  --force-recompute           Force recomputation of already executed notebooks"
      exit 1
      ;;
  esac
done

# Validate required argument
if [[ -z "$workflow_dir" ]]; then
  echo "Error: --workflow-dir is required"
  echo "Usage: $0 --workflow-dir <directory> [--force-recompute]"
  exit 1
fi

# Check if workflow directory exists (try provided path, then workflows/ subdirectory)
if [[ ! -d "$workflow_dir" ]]; then
  # Try looking in workflows/ subdirectory
  if [[ -d "workflows/$workflow_dir" ]]; then
    workflow_dir="workflows/$workflow_dir"
  else
    echo "Error: Workflow directory does not exist: $workflow_dir"
    echo "       Also checked: workflows/$workflow_dir"
    exit 1
  fi
fi

# Check if workflow.yml exists
workflow_yml="$workflow_dir/workflow.yml"
if [[ ! -f "$workflow_yml" ]]; then
  echo "Error: workflow.yml not found in $workflow_dir"
  exit 1
fi

# Minimal environment check - just verify Python and cstar_forge are available
if ! python - <<'PY'
import importlib.util
import sys

if not importlib.util.find_spec("cstar_forge"):
    print("Error: cstar_forge module not found", file=sys.stderr)
    sys.exit(1)
PY
then
  echo "Error: cstar_forge module is not available."
  echo "Please run ./dev-setup.sh to set up the development environment."
  exit 1
fi

# Build command
# Suppress harmless RuntimeWarning about module already in sys.modules
# This occurs when cstar_forge package imports nb_engine before running as module
export PYTHONWARNINGS="ignore::RuntimeWarning:runpy"

cmd=(python -m cstar_forge.nb_engine)
if $force_recompute; then
  cmd+=(--force-recompute)
fi
cmd+=("$workflow_yml")

# Execute
"${cmd[@]}"
