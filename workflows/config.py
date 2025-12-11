from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import socket
from typing import Callable, Dict, Tuple
import argparse
import json
import sys


def _ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class DataPaths:
    """
    Central object holding key paths for data and local assets.

    Includes:
    - source_data
    - input_data
    - run_dir
    - code_root
    - blueprints
    - models_yaml
    - builds_yaml
    """

    here: Path
    model_configs: Path
    source_data: Path
    input_data: Path
    run_dir: Path
    code_root: Path
    blueprints: Path
    models_yaml: Path
    builds_yaml: Path


# --------------------------------------------------------
# Hostname / system detection helpers
# --------------------------------------------------------

def _get_hostname() -> str:
    """Return lowercase hostname from multiple sources."""
    return (
        os.environ.get("HOSTNAME")
        or socket.gethostname()
        or platform.node()
        or "unknown"
    ).lower()


def _detect_system() -> str:
    """
    Return a tag for the current compute environment.

    Tags:
        - "MacOS"
        - "RCAC_anvil"
        - "NERSC_perlmutter"
        - "unknown"

    Extendable via SYSTEM_LAYOUT_REGISTRY.
    """

    system = platform.system().lower()
    if system == "darwin":
        return "MacOS"
    
    host = _get_hostname()
    if "anvil" in host:
        return "RCAC_anvil"

    # Check NERSC_HOST environment variable for Perlmutter
    if os.environ.get("NERSC_HOST", "").lower() == "perlmutter":
        return "NERSC_perlmutter"

    return "unknown"


# --------------------------------------------------------
# System layout registry (pluggable)
# --------------------------------------------------------

# Now each layout returns 4 paths:
# (source_data, input_data, run_dir, code_root)
SystemLayoutFn = Callable[[Path, dict], Tuple[Path, Path, Path, Path]]
SYSTEM_LAYOUT_REGISTRY: Dict[str, SystemLayoutFn] = {}


def register_system(tag: str) -> Callable[[SystemLayoutFn], SystemLayoutFn]:
    """
    Decorator to register a system-specific path layout.

    The decorated function must accept (home: Path, env: dict)
    and return (source_data, input_data, run_dir, code_root).
    """
    def decorator(func: SystemLayoutFn) -> SystemLayoutFn:
        SYSTEM_LAYOUT_REGISTRY[tag] = func
        return func

    return decorator


# --------------------------------------------------------
# Default system layouts
# --------------------------------------------------------

@register_system("MacOS")
def _layout_mac(home: Path, env: dict) -> Tuple[Path, Path, Path, Path]:
    base = home / "cson-forge-data"
    source_data = base / "source-data"
    input_data = base / "input-data"
    run_dir = base / "cson-forge-run"
    code_root = base / "codes"
    return source_data, input_data, run_dir, code_root


@register_system("RCAC_anvil")
def _layout_RCAC_anvil(home: Path, env: dict) -> Tuple[Path, Path, Path, Path]:
    work = Path(env.get("WORK", home / "work"))
    scratch_root = Path(env.get("SCRATCH", work / "scratch"))
    base = work / "cson-forge-data"

    source_data = base / "source-data"
    input_data = base / "input-data"
    run_dir = scratch_root / "cson-forge-run"
    code_root = base / "codes"
    return source_data, input_data, run_dir, code_root


@register_system("NERSC_perlmutter")
def _layout_NERSC_perlmutter(home: Path, env: dict) -> Tuple[Path, Path, Path, Path]:
    scratch_root = Path(env.get("SCRATCH", home / "scratch"))
    base = scratch_root / "cson-forge-data"

    source_data = base / "source-data"
    input_data = base / "input-data"
    run_dir = base / "cson-forge-run"
    code_root = base / "codes"
    return source_data, input_data, run_dir, code_root


@register_system("unknown")
def _layout_unknown(home: Path, env: dict) -> Tuple[Path, Path, Path, Path]:
    base = home / "cson-forge-data"

    source_data = base / "source-data"
    input_data = base / "input-data"
    run_dir = base / "cson-forge-run"
    code_root = base / "codes"
    return source_data, input_data, run_dir, code_root


# --------------------------------------------------------
# Path factory
# --------------------------------------------------------

def get_data_paths() -> DataPaths:
    """
    Return canonical data and project paths adapted to the system we're running on.
    """
    env = os.environ
    home = Path(env.get("HOME", str(Path.home())))
    system_tag = _detect_system()

    layout_fn = SYSTEM_LAYOUT_REGISTRY.get(
        system_tag, SYSTEM_LAYOUT_REGISTRY["unknown"]
    )

    source_data, input_data, run_dir, code_root = layout_fn(home, env)

    here = Path(__file__).resolve().parent
    model_configs = here / "model-configs"
    blueprints_dir = here / "blueprints"
    models_yaml = here / "models.yml"
    builds_yaml = here / "builds.yml"    

    # ensure everything exists
    for p in (source_data, input_data, run_dir, code_root, blueprints_dir, model_configs):
        _ensure_dir(p)

    return DataPaths(
        here=here,
        model_configs=model_configs,
        source_data=source_data,
        input_data=input_data,
        run_dir=run_dir,
        code_root=code_root,
        blueprints=blueprints_dir,
        models_yaml=models_yaml,
        builds_yaml=builds_yaml,
    )


# Initialize canonical instance
paths = get_data_paths()
system = _detect_system()



# --------------------------------------------------------
# CLI
# --------------------------------------------------------

def _paths_to_dict(dp: DataPaths) -> dict:
    return {k: str(v) for k, v in dp.__dict__.items()}


def main(argv: list[str] | None = None) -> int:
    """
    CLI for inspecting detected compute environment and configured paths.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Inspect CSON-Forge data path configuration."
    )

    subparsers = parser.add_subparsers(dest="command")

    # show-paths command
    show_parser = subparsers.add_parser(
        "show-paths",
        help="Show detected system and configured data paths.",
    )
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Output paths as JSON instead of human-readable text.",
    )

    if not argv:
        argv = ["show-paths"]

    args = parser.parse_args(argv)

    if args.command == "show-paths":
        system_tag = _detect_system()
        hostname = _get_hostname()
        dp = paths

        if args.json:
            payload = {
                "system": system_tag,
                "hostname": hostname,
                "paths": _paths_to_dict(dp),
            }
            print(json.dumps(payload, indent=2))
        else:
            print(f"System tag : {system_tag}")
            print(f"Hostname   : {hostname}")
            print("")
            print("Paths:")
            for key, value in _paths_to_dict(dp).items():
                print(f"  {key:12s} -> {value}")

        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
