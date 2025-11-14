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


here = Path(__file__).resolve().parent

default_opt_base_model = "opt_base_roms-marbl-cson-default"


def _ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class DataPaths:
    """Central object holding key paths for data and local assets."""

    source_data: Path
    input_data: Path
    scratch: Path
    logs: Path
    blueprints: Path
    model_config: Path


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

    Tags used by default:
        - "mac"
        - "anvil"
        - "perlmutter"
        - "unknown"

    You can add more tags/layouts via the SYSTEM_LAYOUT_REGISTRY.
    """
    system = platform.system().lower()
    host = _get_hostname()

    # macOS laptops
    if system == "darwin":
        return "mac"

    # Anvil at Purdue (typical hostnames: anvil-*, etc.)
    if "anvil" in host:
        return "anvil"

    # NERSC Perlmutter (login nodes, pm-cpu, pm-gpu, nidxxxxx)
    if (
        "perlmutter" in host
        or "pm-cpu" in host
        or "pm-gpu" in host
        or host.startswith("nid")
    ):
        return "perlmutter"

    return "unknown"


# --------------------------------------------------------
# System layout registry (pluggable)
# --------------------------------------------------------

# signature: layout_func(home: Path, env: dict) -> Tuple[Path, Path, Path]
SystemLayoutFn = Callable[[Path, dict], Tuple[Path, Path, Path]]
SYSTEM_LAYOUT_REGISTRY: Dict[str, SystemLayoutFn] = {}


def register_system(tag: str) -> Callable[[SystemLayoutFn], SystemLayoutFn]:
    """
    Decorator to register a system-specific path layout.

    The decorated function must accept (home: Path, env: dict)
    and return (source_data, input_data, scratch).
    """
    tag = tag.lower()

    def decorator(func: SystemLayoutFn) -> SystemLayoutFn:
        SYSTEM_LAYOUT_REGISTRY[tag] = func
        return func

    return decorator


# ------------------ Default layouts ----------------------


@register_system("mac")
def _layout_mac(home: Path, env: dict) -> Tuple[Path, Path, Path]:
    base = home / "cson-forge-data"
    source_data = base / "source_data"
    input_data = base / "input_data"
    scratch = base / "scratch"
    return source_data, input_data, scratch


@register_system("anvil")
def _layout_anvil(home: Path, env: dict) -> Tuple[Path, Path, Path]:
    work = Path(env.get("WORK", home / "work"))
    scratch_root = Path(env.get("SCRATCH", work / "scratch"))
    base = work / "cson-forge-data"

    source_data = base / "source_data"
    input_data = base / "input_data"
    scratch = scratch_root / "cson-forge"
    return source_data, input_data, scratch


@register_system("perlmutter")
def _layout_perlmutter(home: Path, env: dict) -> Tuple[Path, Path, Path]:
    scratch_root = Path(env.get("SCRATCH", home / "scratch"))
    source_data = scratch_root / "source_data"
    input_data = scratch_root / "input_data"
    scratch = scratch_root / "scratch_work"
    return source_data, input_data, scratch


@register_system("unknown")
def _layout_unknown(home: Path, env: dict) -> Tuple[Path, Path, Path]:
    base = home / "cson-forge-data"
    source_data = base / "source_data"
    input_data = base / "input_data"
    scratch = base / "scratch"
    return source_data, input_data, scratch


# --------------------------------------------------------
# Main factory
# --------------------------------------------------------


def get_data_paths() -> DataPaths:
    """
    Return canonical data and project paths adapted to the system we're running on.
    """
    env = os.environ
    home = Path(env.get("HOME", str(Path.home())))
    system_tag = _detect_system()

    # Pick layout function from registry (fallback to "unknown")
    layout_fn = SYSTEM_LAYOUT_REGISTRY.get(
        system_tag, SYSTEM_LAYOUT_REGISTRY["unknown"]
    )
    source_data, input_data, scratch = layout_fn(home, env)

    # Project-local assets
    model_config = here / "model-configs" / default_opt_base_model
    logs_dir = here / "logs"
    blueprints_dir = here / "blueprints"

    # Ensure dirs exist
    for p in (source_data, input_data, scratch, logs_dir, blueprints_dir):
        _ensure_dir(p)

    return DataPaths(
        source_data=source_data,
        input_data=input_data,
        scratch=scratch,
        logs=logs_dir,
        blueprints=blueprints_dir,
        model_config=model_config,
    )


# Initialize canonical instance (used by rest of code)
paths = get_data_paths()


# --------------------------------------------------------
# Command-line interface
# --------------------------------------------------------


def _paths_to_dict(dp: DataPaths) -> dict:
    return {
        "source_data": str(dp.source_data),
        "input_data": str(dp.input_data),
        "scratch": str(dp.scratch),
        "logs": str(dp.logs),
        "blueprints": str(dp.blueprints),
        "model_config": str(dp.model_config),
    }


def main(argv: list[str] | None = None) -> int:
    """
    Simple CLI for inspecting detected system and configured paths.

    Examples
    --------
    python -m yourpackage.config show-paths
    python config.py show-paths --json
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Inspect ROMS data path configuration."
    )

    subparsers = parser.add_subparsers(dest="command")

    # show-paths subcommand (default)
    show_parser = subparsers.add_parser(
        "show-paths",
        help="Show detected system and configured data paths.",
    )
    show_parser.add_argument(
        "--json",
        action="store_true",
        help="Output paths as JSON instead of human-readable text.",
    )

    # If no command is provided, default to show-paths
    if not argv:
        argv = ["show-paths"]

    args = parser.parse_args(argv)

    if args.command == "show-paths":
        system_tag = _detect_system()
        hostname = _get_hostname()
        dp = paths  # already initialized

        if getattr(args, "json", False):
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
