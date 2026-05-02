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
#   Uses micromamba if available, then mamba, then conda.
#   If none are found, the script will automatically download and
#   install micromamba locally to ./bin (no root privileges required).
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

# Determine which package manager to use (micromamba > mamba > conda > install micromamba)
PACKAGE_MANAGER=""
MICROMAMBA_CMD=""
CONDA_LIKE_CMD="conda"
PM_SOURCE="detected"
MICROMAMBA_URL="https://micro.mamba.pm/api/micromamba/${OS_TYPE}-${ARCH_TYPE}/latest"

# Probe available package manager/binary for reporting (no installs here).
DETECTED_PM="none"
DETECTED_PM_BIN="(none found)"
if command -v micromamba >/dev/null 2>&1; then
  DETECTED_PM="micromamba"
  DETECTED_PM_BIN="$(command -v micromamba)"
elif command -v mamba >/dev/null 2>&1; then
  DETECTED_PM="mamba"
  DETECTED_PM_BIN="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
  DETECTED_PM="conda"
  DETECTED_PM_BIN="$(command -v conda)"
elif [[ -f "$LOCAL_MICROMAMBA" ]] && [[ -x "$LOCAL_MICROMAMBA" ]]; then
  DETECTED_PM="micromamba (local)"
  DETECTED_PM_BIN="$LOCAL_MICROMAMBA"
fi

# Report system information and detected tooling
echo ""
echo "Installation Information"
echo "========================="
echo "  System Detection:"
echo "    • Operating System: $(uname -s) ($(uname -m))"
echo "    • OS Type:          $OS_TYPE"
echo "    • Architecture:     $ARCH_TYPE"
echo ""
echo "  Installation Details:"
echo "    • Package Manager:  $DETECTED_PM"
echo "    • Binary:           $DETECTED_PM_BIN"
echo "    • Fallback Install: local micromamba in $LOCAL_BIN_DIR (if no manager detected)"
if [[ "$DETECTED_PM" == "conda" ]]; then
  echo "    • Note:             conda installation may be slow; mamba/micromamba is recommended."
  echo "                        Consider running with --batch for non-interactive setup."
fi
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
  echo "    • Status:           ENABLED via --clean (will remove and rebuild environment)"
else
  echo "    • Status:           DISABLED (use --clean to remove and rebuild environment)"
fi
echo ""
echo "  Batch Mode:"
if [[ "$BATCH_MODE" == "true" ]]; then
  echo "    • Status:           ENABLED via --batch (no user prompts, all operations automatic)"
else
  echo "    • Status:           DISABLED (use --batch to run non-interactively)"
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
# Check for mamba
elif command -v mamba >/dev/null 2>&1; then
  PACKAGE_MANAGER="mamba"
  CONDA_LIKE_CMD="mamba"
# Check for conda
elif command -v conda >/dev/null 2>&1; then
  PACKAGE_MANAGER="conda"
  CONDA_LIKE_CMD="conda"
# Check for local micromamba
elif [[ -f "$LOCAL_MICROMAMBA" ]] && [[ -x "$LOCAL_MICROMAMBA" ]]; then
  PACKAGE_MANAGER="micromamba"
  MICROMAMBA_CMD="$LOCAL_MICROMAMBA"
# Try to install micromamba locally (last resort)
elif [[ -n "$OS_TYPE" ]] && [[ -n "$ARCH_TYPE" ]]; then
  echo "No micromamba/mamba/conda found. Installing micromamba locally to $LOCAL_BIN_DIR..."
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
    PM_SOURCE="installed"
    echo "✓ micromamba installed successfully to $LOCAL_BIN_DIR"
  else
    rm -rf "$TEMP_DIR"
    echo "Warning: Failed to download micromamba. Falling back to conda if available."
  fi
fi

# Final fallback: try module-based conda (mostly HPC setups)
if [[ -z "$PACKAGE_MANAGER" ]]; then
  if command -v module >/dev/null 2>&1; then
    module load conda 2>/dev/null || true
    if command -v conda >/dev/null 2>&1; then
      PACKAGE_MANAGER="conda"
      CONDA_LIKE_CMD="conda"
    fi
  fi
fi

if [[ -z "$PACKAGE_MANAGER" ]]; then
  echo "Error: None of micromamba, mamba, or conda are available."
  echo ""
  echo "The script attempted to install micromamba locally but failed."
  echo "Please install miniconda/anaconda for conda support, or install micromamba manually."
  exit 1
fi

echo "Using $PACKAGE_MANAGER as package manager"
if [[ "$PACKAGE_MANAGER" == "micromamba" && "$PM_SOURCE" == "installed" ]]; then
  echo "Installing $KERNEL_NAME python environment using $PACKAGE_MANAGER (installed locally in this run)..."
else
  echo "Installing $KERNEL_NAME python environment using $PACKAGE_MANAGER (detected)..."
fi

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
  if "$CONDA_LIKE_CMD" env list | awk '{print $1}' | grep -q "^$KERNEL_NAME$"; then
    ENV_EXISTS="true"
  else
    ENV_EXISTS="false"
  fi
  
  # Remove environment if --clean is specified and it exists
  if [[ "$CLEAN_MODE" == "true" && "$ENV_EXISTS" == "true" ]]; then
    echo "Removing existing $PACKAGE_MANAGER environment: $KERNEL_NAME"
    # Suppress harmless error about mamba_trash.txt (directory may be removed before file write)
    # The environment removal still succeeds despite this error
    "$CONDA_LIKE_CMD" env remove -n "$KERNEL_NAME" -y 2>&1 | grep -v "mamba_trash.txt" || true
    # Small delay to ensure cleanup completes
    sleep 0.5
    ENV_EXISTS="false"
  fi
  
  # Create environment if it doesn't exist
  if [[ "$ENV_EXISTS" == "false" ]]; then
    echo "Creating $PACKAGE_MANAGER environment: $KERNEL_NAME"
    "$CONDA_LIKE_CMD" env create -f "$env_file" -y
  fi
  
  # Activate environment
  echo "Activating $PACKAGE_MANAGER environment: $KERNEL_NAME"
  conda activate "$KERNEL_NAME"
fi
# Keep set +u for package manager operations (scripts may reference unset variables)
# We'll restore set -u at the very end of the script

#--------------------------------------------------------
# Optional compiler/library install
#--------------------------------------------------------
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

INSTALL_FORTRAN_LIBS="false"
if [[ "$BATCH_MODE" == "true" ]]; then
  echo "Batch mode enabled: skipping interactive compiler/library install prompt."
  echo "To install compilers/libraries later, run:"
  echo "  ${CONDA_LIKE_CMD} install -y -c conda-forge compilers mpich netcdf-fortran"
else
  echo ""
  echo "C-Star Forge requires a FORTRAN compiler and supporting libraries (netcdf, MPI)."
  echo "This script can install them in the python environment, however they may conflict with compilers and libraries already installed locally."
  read -r -p 'Please indicated whether to install compilers and fortran libraries [y/N]: ' install_choice
  if [[ "$install_choice" =~ ^[Yy]$ ]]; then
    INSTALL_FORTRAN_LIBS="true"
  fi
fi

if [[ "$INSTALL_FORTRAN_LIBS" == "true" ]]; then
  echo "Installing compilers and library packages from conda-forge..."
  # Package manager install may run deactivation scripts that reference unset variables
  # set +u is already active from the initialization section above
  if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
    micromamba install -y -c conda-forge compilers mpich netcdf-fortran
  else
    "$CONDA_LIKE_CMD" install -y -c conda-forge compilers mpich netcdf-fortran
  fi
  echo "✓ Compiler installation completed successfully!"
else
  echo "Skipping compiler/library installation."
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
echo "Installing local Python package(s) in editable mode..."
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
  echo "Activating and testing kernel in environment $KERNEL_NAME... this may take a few minutes."
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

#--------------------------------------------------------
# Local micromamba convenience messaging (only when local binary is used)
#--------------------------------------------------------
MICROMAMBA_PATH_SH="$LOCAL_BIN_DIR/micromamba-path.sh"
USING_LOCAL_MICROMAMBA="false"
if [[ "$PACKAGE_MANAGER" == "micromamba" ]] && [[ "$MICROMAMBA_CMD" != "micromamba" ]] && [[ -x "$LOCAL_MICROMAMBA" ]]; then
  USING_LOCAL_MICROMAMBA="true"
fi

if [[ "$USING_LOCAL_MICROMAMBA" == "true" ]]; then
  MAMBA_ROOT_PREFIX_EFFECTIVE="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

  # Persist PATH + MAMBA_ROOT_PREFIX for shells that source ./bin/micromamba-path.sh
  cat > "$MICROMAMBA_PATH_SH" <<EOF
# Generated by dev-setup.sh — do not edit by hand (regenerated on each setup).
export PATH="${LOCAL_BIN_DIR}:\${PATH}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX_EFFECTIVE}"
EOF
  chmod a+r "$MICROMAMBA_PATH_SH"

  # Optional: put micromamba on default PATH for new terminals if ~/.local/bin exists or is creatable.
  USER_LOCAL_BIN="${HOME}/.local/bin"
  if mkdir -p "$USER_LOCAL_BIN" 2>/dev/null && ln -sf "$LOCAL_MICROMAMBA" "$USER_LOCAL_BIN/micromamba" 2>/dev/null; then
    echo "micromamba symlink: $USER_LOCAL_BIN/micromamba"
    echo "  (Works in new terminals if $USER_LOCAL_BIN is on your PATH; many setups add it by default.)"
    echo ""
  fi

  echo "micromamba is installed at: $LOCAL_MICROMAMBA"
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "This script was sourced: ./bin is already on PATH for this shell; run: micromamba --help"
  else
    echo "This script was run as a subprocess; your current shell PATH was not changed."
    echo "  In this terminal, run once:"
    echo "    source \"$MICROMAMBA_PATH_SH\""
    echo "  Then you can run: micromamba --help"
  fi
  echo ""

fi

echo "In a new terminal, activate by name with:"
if [[ "$PACKAGE_MANAGER" == "micromamba" ]]; then
  if [[ -f "$MICROMAMBA_PATH_SH" ]]; then
    echo "  source \"$MICROMAMBA_PATH_SH\""
    echo "  eval \"\$(micromamba shell hook -s bash)\"   # or: -s zsh / -s fish"
    echo "  micromamba activate $KERNEL_NAME"
  else
    echo "  eval \"\$(micromamba shell hook -s bash)\"   # or: -s zsh / -s fish"
    echo "  micromamba activate $KERNEL_NAME"
  fi
elif [[ "$PACKAGE_MANAGER" == "mamba" ]]; then
  echo "  eval \"\$(mamba shell hook --shell bash)\"   # or: --shell zsh / --shell fish"
  echo "  mamba activate $KERNEL_NAME"
else
  echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
  echo "  conda activate $KERNEL_NAME"
fi
echo ""

# Restore strict variable checking now that all conda operations are complete
set -u
