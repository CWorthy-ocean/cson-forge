#!/usr/bin/env bash
# Launch the ForgeBlueprint builder as a standalone Voilà web app (code hidden).
#
# Local laptop:
#   ./run-wizard-app.sh
#   -> opens http://localhost:8866 in your browser
#
# HPC login node (no browser there): run it bound to localhost, then SSH
# port-forward from your laptop:
#   # on the login node:
#   ./run-wizard-app.sh --no-browser
#   # on your laptop:
#   ssh -N -L 8866:localhost:8866 <user>@<login-node>
#   # then open http://localhost:8866 locally
#
# Requires: voila (pip install voila) in the active environment.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Steer MPI's libfabric away from the default "sockets" provider: the first
# xESMF regrid in the wizard kernel initializes ESMF/MPI, and the sockets
# provider's progress threads busy-poll at ~100% CPU each (macOS) for the
# life of the kernel. The tcp provider services the same single-process MPI
# without spinning. Respect a pre-set value; batch ROMS runs launched outside
# this script keep their own default.
export FI_PROVIDER="${FI_PROVIDER:-tcp}"
exec voila "${HERE}/cstar_forge/ui/_voila_app.ipynb" \
  --port=8866 \
  --Voila.tornado_settings='{"allow_origin": "*"}' \
  "$@"
