#!/bin/bash
# Script to setup forge development environment
#
# Usage:
#   ./dev-setup.sh              # Normal setup (creates environment if it doesn't exist)
#   ./dev-setup.sh --clean       # Remove and rebuild the environment
#   ./dev-setup.sh --batch      # Run without user prompts (for CI/automation)
#   ./dev-setup.sh --clean --batch  # Clean rebuild without prompts
#
# If micromamba is installed under this repo's ./bin, it is not on your PATH unless
# you source bin/micromamba-path.sh (written by this script) or use: source ./dev-setup.sh
#
# Package Manager:
#   Uses micromamba only.
#   If micromamba is not found, the script will automatically download and
#   install it locally to ./bin (no root privileges required).
#   Supports macOS (ARM64 and Intel) and Linux automatically.
#
# Named activation (e.g. micromamba activate cson-forge-v0) requires a stable
# MAMBA_ROOT_PREFIX. This script exports MAMBA_ROOT_PREFIX if unset (default
# ~/micromamba). Add the same export plus "micromamba shell hook" to your shell
# rc file so new terminals can activate by name.

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
# Always use this name so activation is consistently: micromamba activate cson-forge-v0
KERNEL_NAME="cson-forge-v0"

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

# Determine micromamba command (system or local)
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
  MICROMAMBA_CMD="micromamba"
# Check for local micromamba
elif [[ -f "$LOCAL_MICROMAMBA" ]] && [[ -x "$LOCAL_MICROMAMBA" ]]; then
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
    MICROMAMBA_CMD="$LOCAL_MICROMAMBA"
    echo "✓ micromamba installed successfully to $LOCAL_BIN_DIR"
  else
    rm -rf "$TEMP_DIR"
    echo "Warning: Failed to download micromamba."
  fi
fi

if [[ -z "$MICROMAMBA_CMD" ]]; then
  echo "Error: micromamba is not available."
  echo ""
  echo "The script attempted to install micromamba locally but failed."
  echo "Please install micromamba manually and rerun this script."
  exit 1
fi

echo "Using micromamba as package manager"

# Repo-local micromamba is not on PATH by default. Prepend ./bin so this script and
# any shell that *sources* this script can run `micromamba` by name.
if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
  export PATH="$LOCAL_BIN_DIR:$PATH"
fi

# Initialize and activate environment
set +u
# Single root for all micromamba envs so short names resolve in every shell.
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

# For local micromamba, set up an alias so shell hook works
if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
  alias micromamba="$MICROMAMBA_CMD"
fi

# Initialize micromamba shell hook first (required before any activate/deactivate)
eval "$("$MICROMAMBA_CMD" shell hook --shell bash)"

# Prefer filesystem check: "env list" output can omit the name column for some installs.
if [[ -d "$MAMBA_ROOT_PREFIX/envs/$KERNEL_NAME" ]]; then
  ENV_EXISTS="true"
else
  ENV_EXISTS="false"
fi

# Remove environment if --clean is specified and it exists
if [[ "$CLEAN_MODE" == "true" && "$ENV_EXISTS" == "true" ]]; then
  echo "Removing existing micromamba environment: $KERNEL_NAME"
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
  echo "Creating micromamba environment: $KERNEL_NAME"
  # Drop top-level name:/prefix: so -n is the only install target (avoids unnamed / path-only envs).
  tmp_env_file="$(mktemp "${TMPDIR:-/tmp}/cson-forge-env.XXXXXX.yml")"
  trap 'rm -f "$tmp_env_file"' EXIT
  awk '!/^(name|prefix)[[:space:]]*:/' "$env_file" > "$tmp_env_file"
  "$MICROMAMBA_CMD" env create -n "$KERNEL_NAME" -f "$tmp_env_file" -y
  rm -f "$tmp_env_file"
  trap - EXIT
fi

# Activate environment (now that shell is initialized)
echo "Activating micromamba environment: $KERNEL_NAME"
# After shell hook initialization, we can use 'micromamba' directly (via the hook functions)
micromamba activate "$KERNEL_NAME"
# Keep set +u for package manager operations (scripts may reference unset variables)
# We'll restore set -u at the very end of the script

#--------------------------------------------------------
# Install compilers (Mac only)
#--------------------------------------------------------
if [[ "$(uname)" == "Darwin" ]]; then
  # Ensure environment is active
  if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
    # Shell hook should already be initialized, but ensure alias is set
    if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
      alias micromamba="$MICROMAMBA_CMD"
    fi
    micromamba activate "$KERNEL_NAME"
  fi
  
  echo "Detected macOS - installing compilers and ESMF packages from conda-forge..."
  # Package manager install may run deactivation scripts that reference unset variables
  # set +u is already active from the initialization section above
  # These packages are only installed on macOS; on HPC systems, we rely on system modules
  micromamba install -y -c conda-forge compilers mpich netcdf-fortran esmpy xesmf
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
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
  # Shell hook should already be initialized, but ensure alias is set
  if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
    alias micromamba="$MICROMAMBA_CMD"
  fi
  micromamba activate "$KERNEL_NAME"
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
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] || [[ "$CONDA_DEFAULT_ENV" != "$KERNEL_NAME" ]]; then
  # Shell hook should already be initialized, but ensure alias is set
  if [[ "$MICROMAMBA_CMD" != "micromamba" ]]; then
    alias micromamba="$MICROMAMBA_CMD"
  fi
  micromamba activate "$KERNEL_NAME"
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
echo "  Package manager: micromamba"
echo "  Environment: $KERNEL_NAME"
echo "  MAMBA_ROOT_PREFIX: $MAMBA_ROOT_PREFIX"
echo ""

# Persist PATH + MAMBA_ROOT_PREFIX for shells that source ./bin/micromamba-path.sh
MICROMAMBA_PATH_SH="$LOCAL_BIN_DIR/micromamba-path.sh"
if [[ -x "$LOCAL_MICROMAMBA" ]]; then
  cat > "$MICROMAMBA_PATH_SH" <<EOF
# Generated by dev-setup.sh — do not edit by hand (regenerated on each setup).
export PATH="${LOCAL_BIN_DIR}:\${PATH}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}"
EOF
  chmod a+r "$MICROMAMBA_PATH_SH"
fi

# Optional: put micromamba on default PATH for new terminals if ~/.local/bin exists or is creatable.
USER_LOCAL_BIN="${HOME}/.local/bin"
if [[ "$MICROMAMBA_CMD" != "micromamba" ]] && [[ -x "$LOCAL_MICROMAMBA" ]]; then
  if mkdir -p "$USER_LOCAL_BIN" 2>/dev/null && ln -sf "$LOCAL_MICROMAMBA" "$USER_LOCAL_BIN/micromamba" 2>/dev/null; then
    echo "micromamba symlink: $USER_LOCAL_BIN/micromamba"
    echo "  (Works in new terminals if $USER_LOCAL_BIN is on your PATH; many distros add it by default.)"
    echo ""
  fi
fi

if [[ -x "$LOCAL_MICROMAMBA" ]]; then
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
if [[ -f "$MICROMAMBA_PATH_SH" ]]; then
  echo "  source \"$MICROMAMBA_PATH_SH\""
else
  echo "  export MAMBA_ROOT_PREFIX=\"$MAMBA_ROOT_PREFIX\""
fi
if [[ "$MICROMAMBA_CMD" == "micromamba" ]]; then
  echo "  eval \"\$(micromamba shell hook -s bash)\"   # or: -s zsh / -s fish"
else
  echo "  eval \"\$(\"$LOCAL_MICROMAMBA\" shell hook -s bash)\"   # or: -s zsh / -s fish"
fi
echo "  micromamba activate $KERNEL_NAME"

# Restore strict variable checking now that package manager operations are complete
set -u
