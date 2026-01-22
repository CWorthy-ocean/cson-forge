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

# List of local Python packages to install in editable mode
# Each entry is a directory path relative to SCRIPT_DIR
# Use "." for the current repository root (installs from pyproject.toml)
LOCAL_PYTHON_PACKAGES=(".")

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
if [[ ${#LOCAL_PYTHON_PACKAGES[@]} -eq 1 ]] && [[ "${LOCAL_PYTHON_PACKAGES[0]}" == "." ]]; then
  echo "    • Python Package:   cson-forge (from current directory)"
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
    # Suppress harmless error about mamba_trash.txt (directory may be removed before file write)
    # This error occurs when the conda-meta directory is removed before micromamba can write the trash file
    # Capture stderr, filter out the harmless error, and allow the command to complete
    { "$MICROMAMBA_CMD" env remove -n "$KERNEL_NAME" -y 2>&1; } | grep -v "mamba_trash.txt" || true
    # Small delay to ensure cleanup completes
    sleep 0.5
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
    # Suppress harmless error about mamba_trash.txt (directory may be removed before file write)
    # The environment removal still succeeds despite this error
    conda env remove -n "$KERNEL_NAME" -y 2>&1 | grep -v "mamba_trash.txt" || true
    # Small delay to ensure cleanup completes
    sleep 0.5
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
  
  echo "Detected macOS - installing compilers and ESMF packages from conda-forge..."
  # Package manager install may run deactivation scripts that reference unset variables
  # set +u is already active from the initialization section above
  # These packages are only installed on macOS; on HPC systems, we rely on system modules
  if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
    micromamba install -y -c conda-forge compilers mpich netcdf-fortran esmpy xesmf
  else
    conda install -y -c conda-forge compilers mpich netcdf-fortran esmpy xesmf
  fi
  echo "✓ Compiler installation completed successfully!"
else
  echo "Not macOS - skipping compiler installation"
  echo "Relying on C-Star to manage build environment"
fi

#--------------------------------------------------------
# C-Star setup (install FIRST, as cson_forge depends on it)
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

# Extract roms-tools version from environment.yml and update C-Star's pyproject.toml
echo "Updating C-Star pyproject.toml to pin roms_tools version from environment.yml..."
ENV_YML="${SCRIPT_DIR}/environment.yml"
CSTAR_PYPROJECT="${REPO_DIR}/pyproject.toml"

if [ ! -f "$ENV_YML" ]; then
  echo "Warning: environment.yml not found at $ENV_YML, skipping roms_tools version pinning"
elif [ ! -f "$CSTAR_PYPROJECT" ]; then
  echo "Warning: C-Star pyproject.toml not found at $CSTAR_PYPROJECT, skipping roms_tools version pinning"
else
  # Extract version from environment.yml (format: "roms-tools==3.3.0" or "roms-tools=3.3.0")
  # Matches both conda format (2 spaces) and pip format (4 spaces)
  ROMS_TOOLS_VERSION=$(grep -E "^[[:space:]]{2,4}-[[:space:]]*roms-tools[=]" "$ENV_YML" | sed -E 's/^[[:space:]]*-[[:space:]]*roms-tools[=]+([0-9.]+).*/\1/' | head -1)
  
  if [ -z "$ROMS_TOOLS_VERSION" ]; then
    echo "Warning: Could not extract roms-tools version from environment.yml, skipping version pinning"
  else
    echo "Found roms-tools version: $ROMS_TOOLS_VERSION"
    # Update both occurrences in pyproject.toml (dependencies and optional-dependencies.test)
    # Use sed to replace the version while preserving the [dask] extra
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS uses BSD sed, requires different syntax
      sed -i '' -E "s/roms_tools\[dask\]==[0-9.]+/roms_tools[dask]==${ROMS_TOOLS_VERSION}/g" "$CSTAR_PYPROJECT"
    else
      # Linux uses GNU sed
      sed -i -E "s/roms_tools\[dask\]==[0-9.]+/roms_tools[dask]==${ROMS_TOOLS_VERSION}/g" "$CSTAR_PYPROJECT"
    fi
    echo "✓ Updated C-Star pyproject.toml to use roms_tools[dask]==${ROMS_TOOLS_VERSION}"
  fi
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
# Local Python package setup (install AFTER C-Star)
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
echo "Installing local Python packages in editable mode..."
for package_dir in "${LOCAL_PYTHON_PACKAGES[@]}"; do
  # Resolve to absolute path
  if [[ "$package_dir" == "." ]]; then
    install_dir="$SCRIPT_DIR"
    package_display="cson-forge (current directory)"
  else
    install_dir="$SCRIPT_DIR/$package_dir"
    package_display="$package_dir"
  fi
  
  if [[ ! -d "$install_dir" ]]; then
    echo "  ✗ Warning: Package directory not found: $install_dir"
    continue
  fi
  
  echo "  Installing: $package_display"
  cd "$install_dir"
  pip install -e .
  
  # Verify installation by checking if the package can be imported
  # For the root package, check for cson_forge module
  if [[ "$package_dir" == "." ]]; then
    if python -c "import cson_forge" 2>/dev/null; then
      echo "  ✓ cson-forge installed successfully"
    else
      echo "  ✗ cson-forge installation failed (cannot import cson_forge)"
    fi
  else
    echo "  ✓ $package_display installed"
  fi
done
cd "$SCRIPT_DIR"
echo "✓ Local package installation completed!"

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
  # Use --user flag to make kernel visible globally (not just within the environment)
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_NAME"
  echo "✓ Jupyter kernel installation completed successfully!"
fi

echo ""
echo "✓ Environment setup completed successfully!"
echo "  Package manager: $PACKAGE_MANAGER"
echo "  Environment: $KERNEL_NAME"
echo "  C-Star repository: $REPO_DIR"

# Restore strict variable checking now that all conda operations are complete
set -u
