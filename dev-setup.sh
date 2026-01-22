#!/bin/bash
# Script to setup forge development environment
#
# Usage:
#   ./dev-setup.sh              # Normal setup (creates environment if it doesn't exist)
#   ./dev-setup.sh --clean       # Remove and rebuild the environment
#   ./dev-setup.sh --batch      # Run without user prompts (for CI/automation)
#   ./dev-setup.sh --clean --batch  # Clean rebuild without prompts
#
# Package Manager:
#   Uses micromamba if available (fastest), falls back to conda if not.
#   If micromamba is not found, the script will automatically download and
#   install it locally to ./bin (no root privileges required).
#   Supports macOS (ARM64 and Intel) and Linux automatically.

set -e  # Exit on error

# Parse command line arguments
CLEAN_MODE=false
BATCH_MODE=false
for arg in "$@"; do
  case "$arg" in
    --clean)
      CLEAN_MODE=true
      ;;
    --batch|-f|--force)
      BATCH_MODE=true
      ;;
  esac
done

#--------------------------------------------------------
# Conda environment setup
#--------------------------------------------------------
env_file="environment.yml"
KERNEL_NAME="$(awk -F': *' '$1=="name"{print $2; exit}' "$env_file" 2>/dev/null)"
if [[ -z ${KERNEL_NAME:-} ]]; then
  echo "Error: Could not determine environment name from ${env_file}."
  exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BIN_DIR="$SCRIPT_DIR/bin"
LOCAL_MICROMAMBA="$LOCAL_BIN_DIR/micromamba"

# Discover local Python packages (directories with __init__.py files)
# Exclude test directories and hidden directories
LOCAL_PYTHON_PACKAGES=()
while IFS= read -r package_file; do
  # Get the directory containing the __init__.py file
  package_dir=$(dirname "$package_file")
  # Get the package name (directory name)
  package_name=$(basename "$package_dir")
  # Get relative path from SCRIPT_DIR for easier matching
  rel_path="${package_dir#$SCRIPT_DIR/}"
  # Skip test-related directories, hidden directories, and build directories
  if [[ ! "$rel_path" =~ ^(tests|\.git|_build|external|bin)/ ]] && \
     [[ ! "$package_name" =~ ^\. ]] && \
     [[ "$package_name" != "tests" ]] && \
     [[ "$package_name" != "bin" ]] && \
     [[ "$package_dir" != "$SCRIPT_DIR" ]]; then
    LOCAL_PYTHON_PACKAGES+=("$package_name")
  fi
done < <(find "$SCRIPT_DIR" -maxdepth 2 -type f -name "__init__.py" 2>/dev/null)
# If no packages found, fall back to KERNEL_NAME-based package name
if [[ ${#LOCAL_PYTHON_PACKAGES[@]} -eq 0 ]]; then
  LOCAL_PYTHON_PACKAGES=("${KERNEL_NAME//-/_}")
fi

# Determine OS and architecture for micromamba download
OS_TYPE=""
ARCH_TYPE=""
case "$(uname -s)" in
  Darwin)
    OS_TYPE="osx"
    case "$(uname -m)" in
      arm64) ARCH_TYPE="arm64" ;;
      x86_64) ARCH_TYPE="64" ;;
      *) ARCH_TYPE="64" ;;  # Default to 64-bit
    esac
    ;;
  Linux)
    OS_TYPE="linux"
    case "$(uname -m)" in
      x86_64) ARCH_TYPE="64" ;;
      aarch64) ARCH_TYPE="aarch64" ;;
      *) ARCH_TYPE="64" ;;  # Default to 64-bit
    esac
    ;;
  *)
    OS_TYPE="linux"
    ARCH_TYPE="64"
    ;;
esac

# Determine which package manager to use (micromamba > conda)
PACKAGE_MANAGER=""
MICROMAMBA_CMD=""
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/${OS_TYPE}-${ARCH_TYPE}/latest"

# Report system information and download details
echo ""
echo "Installation Information"
echo "========================="
echo "  System Detection:"
echo "    • Operating System: $(uname -s) ($(uname -m))"
echo "    • OS Type:          $OS_TYPE"
echo "    • Architecture:     $ARCH_TYPE"
echo ""
echo "  Installation Details:"
echo "    • Target Location:  $LOCAL_BIN_DIR"
echo "    • Download URL:     $MICROMAMBA_URL"
echo "    • No root access required"
echo ""
echo "  Environment:"
echo "    • Environment Name: $KERNEL_NAME"
if [[ ${#LOCAL_PYTHON_PACKAGES[@]} -eq 0 ]]; then
  echo "    • Python Packages:  (none found)"
elif [[ ${#LOCAL_PYTHON_PACKAGES[@]} -eq 1 ]]; then
  echo "    • Python Packages:  ${LOCAL_PYTHON_PACKAGES[0]}"
else
  echo "    • Python Packages:"
  for pkg in "${LOCAL_PYTHON_PACKAGES[@]}"; do
    echo "      - $pkg"
  done
fi
echo "    • Environment File: $env_file"
echo ""
echo "  Clean Mode:"
if [[ "$CLEAN_MODE" == "true" ]]; then
  echo "    • Status:           ENABLED (will remove and rebuild environment)"
else
  echo "    • Status:           DISABLED (will reuse existing environment if present)"
fi
echo ""
echo "  Batch Mode:"
if [[ "$BATCH_MODE" == "true" ]]; then
  echo "    • Status:           ENABLED (no user prompts, all operations automatic)"
else
  echo "    • Status:           DISABLED (will prompt for confirmation)"
fi
echo ""
if [[ "$BATCH_MODE" != "true" ]]; then
  echo "  ⚡ READY TO PROCEED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  read -p "  ⏎  Press Enter to continue, or Ctrl+C to cancel:  "
  echo ""
fi

# Check for system micromamba first
if command -v micromamba >/dev/null 2>&1; then
  PACKAGE_MANAGER="micromamba"
  MICROMAMBA_CMD="micromamba"
# Check for local micromamba
elif [[ -f "$LOCAL_MICROMAMBA" ]] && [[ -x "$LOCAL_MICROMAMBA" ]]; then
  PACKAGE_MANAGER="micromamba"
  MICROMAMBA_CMD="$LOCAL_MICROMAMBA"
# Try to install micromamba locally
elif [[ -n "$OS_TYPE" ]] && [[ -n "$ARCH_TYPE" ]]; then
  echo "micromamba not found. Installing locally to $LOCAL_BIN_DIR..."
  mkdir -p "$LOCAL_BIN_DIR"
  
  # Download micromamba
  echo "Downloading micromamba..."
  
  # Extract to temp location first, then move to final location
  TEMP_DIR=$(mktemp -d)
  if curl -Ls "$MICROMAMBA_URL" | tar -xvj -C "$TEMP_DIR" bin/micromamba 2>/dev/null; then
    # Move from temp/bin/micromamba to LOCAL_BIN_DIR/micromamba
    mv "$TEMP_DIR/bin/micromamba" "$LOCAL_MICROMAMBA"
    rm -rf "$TEMP_DIR"
    chmod +x "$LOCAL_MICROMAMBA"
    PACKAGE_MANAGER="micromamba"
    MICROMAMBA_CMD="$LOCAL_MICROMAMBA"
    echo "✓ micromamba installed successfully to $LOCAL_BIN_DIR"
  else
    rm -rf "$TEMP_DIR"
    echo "Warning: Failed to download micromamba. Falling back to conda if available."
  fi
fi

# Fall back to conda if micromamba is not available
if [[ -z "$PACKAGE_MANAGER" ]]; then
  if command -v conda >/dev/null 2>&1; then
    PACKAGE_MANAGER="conda"
  elif command -v module >/dev/null 2>&1; then
    module load conda 2>/dev/null || true
    if command -v conda >/dev/null 2>&1; then
      PACKAGE_MANAGER="conda"
    fi
  fi
fi

if [[ -z "$PACKAGE_MANAGER" ]]; then
  echo "Error: Neither micromamba nor conda is available."
  echo ""
  echo "The script attempted to install micromamba locally but failed."
  echo "Please install miniconda/anaconda for conda support, or install micromamba manually."
  exit 1
fi

echo "Using $PACKAGE_MANAGER as package manager"

# Initialize and activate environment
set +u
if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
  # For micromamba, we need to initialize the shell hook
  # If using local micromamba, we need to make sure the shell hook uses the correct path
  if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
    # For local micromamba, set up an alias so shell hook works
    alias micromamba="$MICROMAMBA_CMD"
  fi
  
  # Initialize micromamba shell hook first (required before any activate/deactivate)
  eval "$("$MICROMAMBA_CMD" shell hook --shell bash)"
  
  # Check if environment exists
  if "$MICROMAMBA_CMD" env list | awk '{print $1}' | grep -q "^$KERNEL_NAME$"; then
    ENV_EXISTS="true"
  else
    ENV_EXISTS="false"
  fi
  
  # Remove environment if --clean is specified and it exists
  if [[ "$CLEAN_MODE" == "true" && "$ENV_EXISTS" == "true" ]]; then
    echo "Removing existing $PACKAGE_MANAGER environment: $KERNEL_NAME"
    "$MICROMAMBA_CMD" env remove -n "$KERNEL_NAME" -y
    ENV_EXISTS="false"
  fi
  
  # Create environment if it doesn't exist
  if [[ "$ENV_EXISTS" == "false" ]]; then
    echo "Creating $PACKAGE_MANAGER environment: $KERNEL_NAME"
    "$MICROMAMBA_CMD" env create -f "$env_file" -y
  fi
  
  # Activate environment (now that shell is initialized)
  echo "Activating $PACKAGE_MANAGER environment: $KERNEL_NAME"
  # After shell hook initialization, we can use 'micromamba' directly (via the hook functions)
  micromamba activate "$KERNEL_NAME"
  
else
  # Initialize conda for this shell session by sourcing conda.sh
  # This makes conda commands (like 'conda activate') available in the script
  source "$(conda info --base)/etc/profile.d/conda.sh"
  
  # Check if environment exists
  if conda env list | awk '{print $1}' | grep -q "^$KERNEL_NAME$"; then
    ENV_EXISTS="true"
  else
    ENV_EXISTS="false"
  fi
  
  # Remove environment if --clean is specified and it exists
  if [[ "$CLEAN_MODE" == "true" && "$ENV_EXISTS" == "true" ]]; then
    echo "Removing existing $PACKAGE_MANAGER environment: $KERNEL_NAME"
    conda env remove -n "$KERNEL_NAME" -y
    ENV_EXISTS="false"
  fi
  
  # Create environment if it doesn't exist
  if [[ "$ENV_EXISTS" == "false" ]]; then
    echo "Creating $PACKAGE_MANAGER environment: $KERNEL_NAME"
    conda env create -f "$env_file"
  fi
  
  # Activate environment
  echo "Activating $PACKAGE_MANAGER environment: $KERNEL_NAME"
  conda activate "$KERNEL_NAME"
fi
# Keep set +u for package manager operations (scripts may reference unset variables)
# We'll restore set -u at the very end of the script

#--------------------------------------------------------
# Install compilers (Mac only)
#--------------------------------------------------------
if [[ "$(uname)" == "Darwin" ]]; then
  # Ensure environment is active
  if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
    if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
      # Shell hook should already be initialized, but ensure alias is set
      if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
        alias micromamba="$MICROMAMBA_CMD"
      fi
      micromamba activate "$KERNEL_NAME"
    fi
  else
    if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "$KERNEL_NAME"
    fi
  fi
  
  echo "Detected macOS - installing compilers from conda-forge..."
  # Package manager install may run deactivation scripts that reference unset variables
  # set +u is already active from the initialization section above
  if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
    micromamba install -y -c conda-forge compilers mpich netcdf-fortran
  else
    conda install -y -c conda-forge compilers mpich netcdf-fortran
  fi
  echo "✓ Compiler installation completed successfully!"
else
  echo "Not macOS - skipping compiler installation"
  echo "Relying on C-Star to manage build environment"
fi

#--------------------------------------------------------
# Local Python package setup
#--------------------------------------------------------
# Ensure environment is active
# set +u is already active from initialization section
if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
    # Shell hook should already be initialized, but ensure alias is set
    if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
      alias micromamba="$MICROMAMBA_CMD"
    fi
    micromamba activate "$KERNEL_NAME"
  fi
else
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$KERNEL_NAME"
  fi
fi

# Install local Python packages in editable mode
# Check if any package needs to be installed
NEEDS_INSTALL=false
for package_name in "${LOCAL_PYTHON_PACKAGES[@]}"; do
  if ! python - "$package_name" <<'PY'
import importlib.util
import sys
package_name = sys.argv[1]
sys.exit(0 if importlib.util.find_spec(package_name) else 1)
PY
  then
    NEEDS_INSTALL=true
    break
  fi
done

if [[ "$NEEDS_INSTALL" == "true" ]]; then
  echo "Installing local Python packages in editable mode..."
  # Install from root directory (uses pyproject.toml or setup.py)
  # This installs all packages defined in the project configuration
  pip install -e .
  echo "✓ Local package installation completed successfully!"
else
  for package_name in "${LOCAL_PYTHON_PACKAGES[@]}"; do
    echo "✓ $package_name is already installed"
  done
fi

#--------------------------------------------------------
# C-Star setup
# This should be unnecessary once C-Star is released to 
# conda-forge and available via conda install
#--------------------------------------------------------
REPO_URL="https://github.com/CWorthy-ocean/C-Star.git"
BRANCH="orchestration"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${SCRIPT_DIR}/external"
REPO_DIR="$CODE_ROOT/C-Star"

# Create code root directory if it doesn't exist
mkdir -p "$CODE_ROOT"

pushd "$CODE_ROOT" > /dev/null

# Clone or update the repository
if [ -d "$REPO_DIR" ]; then
  echo "C-Star repository already exists. Updating..."
  cd "$REPO_DIR"
  git fetch origin
  git checkout "$BRANCH"
  git pull origin "$BRANCH"
  cd ..
else
  echo "Cloning C-Star repository..."
  git clone -b "$BRANCH" "$REPO_URL" "$REPO_DIR"
fi

# Install C-Star in editable mode
# Ensure environment is active
# set +u is already active from initialization section
if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
    # Shell hook should already be initialized, but ensure alias is set
    if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
      alias micromamba="$MICROMAMBA_CMD"
    fi
    micromamba activate "$KERNEL_NAME"
  fi
else
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$KERNEL_NAME"
  fi
fi

echo "Installing C-Star in editable mode..."
cd "$REPO_DIR"
pip install -e .
echo "✓ C-Star installation completed successfully!"

popd > /dev/null

#--------------------------------------------------------
# Jupyter kernel setup
#--------------------------------------------------------
# Ensure environment is active
# set +u is already active from initialization section
if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
    # Shell hook should already be initialized, but ensure alias is set
    if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
      alias micromamba="$MICROMAMBA_CMD"
    fi
    micromamba activate "$KERNEL_NAME"
  fi
else
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$KERNEL_NAME"
  fi
fi

# Check if kernel exists
if python - "$KERNEL_NAME" <<'PY'
from jupyter_client.kernelspec import KernelSpecManager
import sys
name = sys.argv[1]
specs = KernelSpecManager().find_kernel_specs()
sys.exit(0 if name in specs else 1)
PY
then
  KERNEL_EXISTS="true"
else
  KERNEL_EXISTS="false"
fi

# Remove kernel if --clean is specified and it exists
if [[ "$CLEAN_MODE" == "true" && "$KERNEL_EXISTS" == "true" ]]; then
  echo "Removing existing Jupyter kernel: $KERNEL_NAME"
  python -m ipykernel uninstall -y --name "$KERNEL_NAME" 2>/dev/null || true
  KERNEL_EXISTS="false"
fi

# Install kernel if it doesn't exist
if [[ "$KERNEL_EXISTS" == "false" ]]; then
  echo "Installing Jupyter kernel: $KERNEL_NAME"
  python -m ipykernel install --sys-prefix --name "$KERNEL_NAME" --display-name "$KERNEL_NAME"
  echo "✓ Jupyter kernel installation completed successfully!"
fi

echo ""
echo "✓ Environment setup completed successfully!"
echo "  Package manager: $PACKAGE_MANAGER"
echo "  Environment: $KERNEL_NAME"
echo "  C-Star repository: $REPO_DIR"

# Restore strict variable checking now that all conda operations are complete
set -u
