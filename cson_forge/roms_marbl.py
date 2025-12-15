from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
import uuid

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import roms_tools as rt

from . import config
from . import source_data
from .model import ModelSpec


# =========================================================
# ROMS input generation (from former model_config.py)
# =========================================================


def _path_to_template(path: Union[str, Path]) -> str:
    """
    Convert an actual file path to a template key using config.paths.
    
    Parameters
    ----------
    path : str or Path
        Actual file path to convert.
        
    Returns
    -------
    str
        Template string with config.paths keys (e.g., "{INPUT_DATA}/roms-marbl_ccs-12km/file.nc").
        If the path doesn't match any config.paths, returns the original path as a string.
    """
    if not isinstance(path, (str, Path)):
        return path
    
    path_str = str(path)
    
    # If already a template, return as-is
    if path_str.startswith("{") and "}" in path_str:
        return path_str
    
    paths = config.paths
    
    # Check each config path and replace if it's a prefix
    replacements = {
        str(paths.input_data): "{INPUT_DATA}",
        str(paths.blueprints): "{BLUEPRINTS}",
        str(paths.source_data): "{SOURCE_DATA}",
        str(paths.run_dir): "{RUN_DIR}",
        str(paths.code_root): "{CODE_ROOT}",
        str(paths.model_configs): "{MODEL_CONFIGS}",
    }
    
    # Sort by length (longest first) to match most specific paths first
    for actual_path, template_key in sorted(replacements.items(), key=lambda x: -len(x[0])):
        if path_str.startswith(actual_path):
            # Replace the prefix and return
            relative = path_str[len(actual_path):].lstrip("/")
            if relative:
                return f"{template_key}/{relative}"
            else:
                return template_key
    
    # If no match, return as-is (might be a relative path or external path)
    return path_str


def _template_to_path(template: str) -> str:
    """
    Convert a template string with config.paths keys to an actual path.
    
    Parameters
    ----------
    template : str
        Template string (e.g., "{INPUT_DATA}/roms-marbl_ccs-12km/file.nc").
        
    Returns
    -------
    str
        Actual file path resolved from config.paths.
        If the template doesn't contain any template keys, returns as-is.
    """
    if not isinstance(template, str):
        return template
    
    # If not a template (doesn't start with {), return as-is
    if not template.startswith("{"):
        return template
    
    paths = config.paths
    
    replacements = {
        "{INPUT_DATA}": str(paths.input_data),
        "{BLUEPRINTS}": str(paths.blueprints),
        "{SOURCE_DATA}": str(paths.source_data),
        "{RUN_DIR}": str(paths.run_dir),
        "{CODE_ROOT}": str(paths.code_root),
        "{MODEL_CONFIGS}": str(paths.model_configs),
    }
    
    result = template
    for key, actual_path in replacements.items():
        if result.startswith(key):
            # Replace the key with actual path
            suffix = result[len(key):].lstrip("/")
            if suffix:
                return f"{actual_path}/{suffix}"
            else:
                return actual_path
    
    # If no template key found, return as-is
    return result


def _apply_path_templates_to_value(value: Any, template_to_path: bool = False) -> Any:
    """
    Recursively apply path template conversion to a value (dict, list, or string).
    
    Parameters
    ----------
    value : Any
        Value to process (can be dict, list, str, Path, etc.).
    template_to_path : bool
        If True, convert templates to paths. If False, convert paths to templates.
        
    Returns
    -------
    Any
        Value with paths converted to/from templates.
    """
    if isinstance(value, Path):
        value = str(value)
    
    if isinstance(value, str):
        if template_to_path:
            return _template_to_path(value)
        else:
            return _path_to_template(value)
    
    if isinstance(value, dict):
        return {k: _apply_path_templates_to_value(v, template_to_path) for k, v in value.items()}
    
    if isinstance(value, (list, tuple)):
        return type(value)(_apply_path_templates_to_value(v, template_to_path) for v in value)
    
    return value


class InputStep:
    """Metadata for a single ROMS input generation step."""

    def __init__(self, name: str, order: int, label: str, handler: Callable):
        self.name = name  # canonical key used for filenames & paths
        self.order = order  # execution order
        self.label = label  # human-readable label
        self.handler = handler  # function expecting `self` (inputs instance)


INPUT_REGISTRY: Dict[str, InputStep] = {}


def register_input(name: str, order: int, label: str | None = None):
    """
    Decorator to register an input-generation step.

    Parameters
    ----------
    name : str
        Key for this input (e.g., 'grid', 'initial_conditions', 'surface_forcing').
        This will be used in filenames, and to index `inputs[name]`.
    order : int
        Execution order in `generate_all()`. Lower numbers run first.
    label : str, optional
        Human-readable label for progress messages. If omitted, `name` is used.
    """

    def decorator(func: Callable):
        step_label = label or name
        INPUT_REGISTRY[name] = InputStep(
            name=name,
            order=order,
            label=step_label,
            handler=func,
        )
        return func

    return decorator


@dataclass
class InputObj:
    """
    Structured representation of a single ROMS input product.

    Attributes
    ----------
    input_type : str
        The type/key of this input (e.g., "initial_conditions", "surface_forcing").
    paths : Path | list[Path] | None
        Path or list of paths to the generated NetCDF file(s), if applicable.
    paths_partitioned : Path | list[Path] | None
        Path(s) to the partitioned NetCDF file(s), if applicable.
    yaml_file : Path | None
        Path to the YAML description written for this input, if any.
    """

    input_type: str
    paths: Optional[Union[Path, List[Path]]] = None
    paths_partitioned: Optional[Union[Path, List[Path]]] = None
    yaml_file: Optional[Path] = None


@dataclass
class inputs:
    """
    Generate and manage ROMS input files for a given grid.

    This object is driven by:
      - model specification from `models.yml` (model_spec).

    The list of inputs to generate (`input_list`) is automatically
    derived from the keys in `model_spec.inputs`.

    The defaults from `model_spec.inputs[<key>]` are merged with runtime arguments
    (e.g., start_time, end_time, boundaries). Any "source" or "bgc_source"
    fields in the defaults are resolved through `SourceData`, which injects
    a "path" entry pointing at the prepared dataset file.
    """

    # core config
    model_name: str
    grid_name: str
    grid: object
    start_time: object
    end_time: object
    np_eta: int
    np_xi: int
    boundaries: dict
    source_data: source_data.SourceData

    # model specification from models.yml
    model_spec: ModelSpec

    # which inputs to generate for this run (derived from model_spec.inputs keys)
    input_list: List[str] = field(init=False)

    use_dask: bool = True
    clobber: bool = False

    # derived
    input_data_dir: Path = field(init=False)
    blueprint_dir: Path = field(init=False)
    inputs: Dict[str, InputObj] = field(init=False)
    obj: Dict[str, Any] = field(init=False)  # Maps input keys to roms_tools objects (Grid, InitialConditions, SurfaceForcing, etc.)
    bp_path: Path = field(init=False)

    def __post_init__(self):
        # Path to input directory
        self.input_data_dir = config.paths.input_data / f"{self.model_name}_{self.grid_name}"
        self.input_data_dir.mkdir(exist_ok=True)

        self.blueprint_dir = config.paths.blueprints / f"{self.model_name}_{self.grid_name}"
        self.blueprint_dir.mkdir(parents=True, exist_ok=True)
        self.bp_path = self.blueprint_dir / f"blueprint_{self.model_name}-{self.grid_name}.yml"

        # Storage for detailed per-input objects
        self.inputs = {}
        self.obj = {}
        
        # Derive input_list from model_spec.inputs keys
        input_list = list(self.model_spec.inputs.keys())
        if "grid" not in input_list:
            input_list.insert(0, "grid")
        self.input_list = input_list

    # ----------------------------
    # Public API
    # ----------------------------

    def generate_all(self):
        """
        Generate all ROMS input files for this grid using the registered
        steps whose names appear in `input_list`, then partition and
        write a blueprint.

        If any names in `input_list` lack registered handlers,
        a ValueError is raised.
        """

        if not self._ensure_empty_or_clobber(self.clobber):
            return self

        # Sanity check
        registry_keys = set(INPUT_REGISTRY.keys())
        needed = set(self.input_list)
        missing = sorted(needed - registry_keys)
        if missing:
            raise ValueError(
                "The following ROMS inputs are listed in `input_list` but "
                f"have no registered handlers: {', '.join(missing)}"
            )

        # Use only the selected steps
        steps = [INPUT_REGISTRY[name] for name in self.input_list]
        steps.sort(key=lambda s: s.order)
        total = len(steps) + 1

        # Execute
        for idx, step in enumerate(steps, start=1):
            print(f"\n▶️  [{idx}/{total}] {step.label}...")
            step.handler(self, key=step.name)

        # Partition step
        print(f"\n▶️  [{total}/{total}] Partitioning input files across tiles...")
        self._partition_files()

        print("\n✅ All input files generated and partitioned.\n")
        self._write_inputs_blueprint()
        return self

    # ----------------------------
    # Internals
    # ----------------------------

    def _ensure_empty_or_clobber(self, clobber: bool) -> bool:
        """
        Ensure the input_data_dir is either empty or, if clobber=True,
        remove existing .nc files.
        """
        existing = list(self.input_data_dir.glob("*.nc"))

        if existing and not clobber:
            print(f"⚠️  Found existing ROMS input files in {self.input_data_dir}")
            print("    Not overwriting because clobber=False.")
            print("\nExiting without changes.\n")
            return False

        if existing and clobber:
            print(
                f"⚠️  Clobber=True: removing {len(existing)} existing .nc files in "
                f"{self.input_data_dir}..."
            )
            for f in existing:
                f.unlink()

        return True

    def _forcing_filename(self, key: str) -> Path:
        """Construct the NetCDF filename for a given input key."""
        return self.input_data_dir / f"roms_{key}.nc"

    def _yaml_filename(self, key: str) -> Path:
        """Construct the YAML blueprint filename for a given input key."""
        return self.blueprint_dir / f"_{key}.yml"

    # ----------------------------
    # Helpers for merging YAML defaults
    # ----------------------------

    def _resolve_source_block(self, block: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Normalize a "source"/"bgc_source" block and inject a 'path'
        based on SourceData.

        Parameters
        ----------
        block : str or dict
            Either a simple logical name (e.g., "GLORYS") or a dict
            with at least a "name" field.

        Returns
        -------
        dict
            Source block with a "name" field and optionally a "path" field.
            For streamable sources, "path" is only included if explicitly
            provided in the input block. For non-streamable sources, "path"
            is added from SourceData if available.
        """

        if isinstance(block, str):
            name = block
            out: Dict[str, Any] = {"name": name}
        elif isinstance(block, dict):
            out = dict(block)
            name = out.get("name")
            if not name:
                raise ValueError(
                    f"Source block {block!r} is missing a 'name' field."
                )
        else:
            raise TypeError(f"Unsupported source block type: {type(block)}")

        # Get the mapped dataset key to check if it's streamable
        dataset_key = self.source_data.dataset_key_for_source(name)
        
        # If streamable and no path was explicitly provided in YAML, don't add path field
        if dataset_key in source_data.STREAMABLE_SOURCES:
            # Only return early if path wasn't explicitly provided
            if "path" not in out:
                return out
            # If path was provided, continue to validate it exists (or return as-is)
            return out

        path = self.source_data.path_for_source(name)
        # Only add path if it's not None and wasn't explicitly provided
        if path is not None:
            out.setdefault("path", path)
        return out

    def _build_input_args(self, key: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge per-input defaults (from models.yml) with runtime arguments.

        - Start from `model_spec.inputs.get(key, {})`.
        - If present, resolve "source" and "bgc_source" through SourceData,
          injecting a "path" entry.
        - Merge with `extra`, where `extra` overrides defaults on conflict.
        """
        cfg = dict(self.model_spec.inputs.get(key, {}) or {})

        for field_name in ("source", "bgc_source"):
            if field_name in cfg:
                cfg[field_name] = self._resolve_source_block(cfg[field_name])

        # `extra` overrides defaults
        return {**cfg, **extra}

    # ----------------------------
    # Registry-backed generation steps
    # ----------------------------

    @register_input(name="grid", order=10, label="Writing ROMS grid")
    def _generate_grid(self, key: str = "grid", **kwargs):
        out_path = self._forcing_filename(key)
        yaml_path = self._yaml_filename(key)

        self.grid.save(out_path)
        self.grid.to_yaml(yaml_path)
        self.obj[key] = self.grid
        self.inputs[key] = InputObj(
            input_type=key,
            paths=out_path,
            yaml_file=yaml_path,
        )


    @register_input(name="initial_conditions", order=20, label="Generating initial conditions")
    def _generate_initial_conditions(self, key: str = "initial_conditions", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            ini_time=self.start_time,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        ic = rt.InitialConditions(grid=self.grid, **input_args)
        paths = ic.save(self._forcing_filename(key))
        ic.to_yaml(yaml_path)
        self.obj[key] = ic

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )


    @register_input(name="surface_forcing", order=30, label="Generating surface forcing (physics)")
    def _generate_surface_forcing(self, key: str = "surface_forcing", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        frc = rt.SurfaceForcing(grid=self.grid, **input_args)
        paths = frc.save(self._forcing_filename(key))
        frc.to_yaml(yaml_path)
        self.obj[key] = frc

        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="surface_forcing_bgc", order=40, label="Generating surface forcing (BGC)")
    def _generate_bgc_surface_forcing(self, key: str = "surface_forcing_bgc", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        frc_bgc = rt.SurfaceForcing(grid=self.grid, **input_args)
        paths = frc_bgc.save(self._forcing_filename(key))
        frc_bgc.to_yaml(yaml_path)
        self.obj[key] = frc_bgc
        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="boundary_forcing", order=50, label="Generating boundary forcing (physics)")
    def _generate_boundary_forcing(self, key: str = "boundary_forcing", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        bry = rt.BoundaryForcing(grid=self.grid, **input_args)
        paths = bry.save(self._forcing_filename(key))
        bry.to_yaml(yaml_path)
        self.obj[key] = bry
        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="boundary_forcing_bgc", order=60, label="Generating boundary forcing (BGC)")
    def _generate_bgc_boundary_forcing(self, key: str = "boundary_forcing_bgc", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)

        bry_bgc = rt.BoundaryForcing(grid=self.grid, **input_args)
        paths = bry_bgc.save(self._forcing_filename(key))
        bry_bgc.to_yaml(yaml_path)
        self.obj[key] = bry_bgc
        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="tidal_forcing", order=70, label="Generating tidal forcing")
    def _generate_tidal_forcing(self, key: str = "tidal_forcing", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            model_reference_date=self.start_time,
            use_dask=self.use_dask,
        )
        input_args = self._build_input_args(key, extra=extra)
        tidal = rt.TidalForcing(grid=self.grid, **input_args)
        paths = tidal.save(self._forcing_filename(key))
        tidal.to_yaml(yaml_path)
        self.obj[key] = tidal
        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="rivers", order=80, label="Generating river forcing")
    def _generate_river_forcing(self, key: str = "rivers", **kwargs):
        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
        )
        input_args = self._build_input_args(key, extra=extra)

        rivers = rt.RiverForcing(grid=self.grid, **input_args)
        paths = rivers.save(self._forcing_filename(key))
        rivers.to_yaml(yaml_path)
        self.obj[key] = rivers
        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    @register_input(name="cdr", order=80, label="Generating CDR forcing")
    def _generate_cdr_forcing(self, key: str = "cdr", cdr_list=None, **kwargs):
        cdr_list = [] if cdr_list is None else cdr_list
        if not cdr_list:
            return

        yaml_path = self._yaml_filename(key)
        extra = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=cdr_list,
        )
        input_args = self._build_input_args(key, extra=extra)

        cdr = rt.CDRForcing(grid=self.grid, **input_args)
        paths = cdr.save(self._forcing_filename(key))
        cdr.to_yaml(yaml_path)
        self.obj[key] = cdr
        self.inputs[key] = InputObj(
            input_type=key,
            paths=paths,
            yaml_file=yaml_path,
        )

    # ----------------------------
    # Partition step (not in registry)
    # ----------------------------

    def _partition_files(self, **kwargs):
        """
        Partition whole input files across tiles using roms_tools.partition_netcdf.

        Uses the paths stored in `inputs[...]` (for keys in input_list)
        to build the list of whole-field files, and records the partitioned
        paths on each InputObj.
        """
        input_args = dict(
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            output_dir=self.input_data_dir,
            include_coarse_dims=False,
        )

        for name in self.input_list:
            obj = self.inputs.get(name)
            if obj is None or obj.paths is None:
                continue
            obj.paths_partitioned = rt.partition_netcdf(obj.paths, **input_args)

    # ----------------------------
    # Blueprint writer
    # ----------------------------

    def _write_inputs_blueprint(self):
        """
        Serialize a summary of inputs state to a YAML blueprint:

            blueprints/{model_name}-{grid_name}/model-inputs.yml

        Contents include high-level configuration, model_spec, and a sanitized view of
        `inputs` (paths, arguments, etc.).
        """
        import xarray as xr

        XR_TYPES = (xr.Dataset, xr.DataArray)

        def _serialize(obj: Any) -> Any:
            from datetime import date, datetime
            from dataclasses import is_dataclass, asdict as dc_asdict

            if XR_TYPES and isinstance(obj, XR_TYPES):
                return None

            if is_dataclass(obj) and not isinstance(obj, type):
                return _serialize(dc_asdict(obj))

            if isinstance(obj, Path):
                # Convert path to template before serializing
                return _path_to_template(str(obj))

            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj

            if isinstance(obj, (date, datetime)):
                return obj.isoformat()

            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}

            if isinstance(obj, (list, tuple, set)):
                return [_serialize(v) for v in obj]

            if callable(obj):
                qualname = getattr(obj, "__qualname__", None)
                mod = getattr(obj, "__module__", None)
                if qualname and mod:
                    return f"{mod}.{qualname}"
                return repr(obj)

            return repr(obj)

        raw = dict(
            grid_name=self.grid_name,
            start_time=self.start_time,
            end_time=self.end_time,
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            boundaries=self.boundaries,
            input_data_dir=self.input_data_dir,
            model_spec=self.model_spec,
            inputs=self.inputs,
        )

        data = _serialize(raw)
        
        # Apply path templates to the serialized data (in case any paths were already strings)
        data = _apply_path_templates_to_value(data, template_to_path=False)

        with self.bp_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=True)

        print(f"📄  Wrote inputs blueprint to {self.bp_path}")


# =========================================================
# Build logic (from former model_build.py)
# =========================================================


def _generate_slurm_script(
    run_command: str,
    job_name: str,
    account: str,
    queue: str,
    wallclock_time: str,
    n_nodes: int,
    n_tasks_per_node: int,
    run_dir: Path,
    log_func: Callable[[str], None] = print,
) -> Path:
    """
    Generate a SLURM batch script for running the model, with system-specific options.

    Supported machines:
        - NERSC_perlmutter

    Raises NotImplementedError for all other systems.

    Parameters
    ----------
    run_command : str
        The command to execute (e.g., mpirun command).
    job_name : str
        Name for the SLURM job.
    account : str
        Account to charge the job to.
    queue : str
        Queue/partition to submit to.
    wallclock_time : str
        Wallclock time limit (format: HH:MM:SS).
    n_nodes : int
        Number of nodes to request.
    n_tasks_per_node : int
        Number of tasks to request.
    run_dir : Path
        Directory where the batch script and output files will be written.  
    log_func : callable, optional
        Logging function for messages.
    
    Returns
    -------
    Path
        Path to the generated batch script.
    """
    
    script_path = run_dir / f"{job_name}.run"
    stdout_path = run_dir / f"{job_name}.out"
    stderr_path = run_dir / f"{job_name}.err"

    if config.system_id == "NERSC_perlmutter":
        # Perlmutter (NERSC) SLURM script
        script_content = textwrap.dedent(f"""
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH --account={account}
            #SBATCH --qos={queue}
            #SBATCH --time={wallclock_time}
            #SBATCH --output={stdout_path}
            #SBATCH --error={stderr_path}
            #SBATCH --constraint=cpu
            #SBATCH --nodes={n_nodes}
            #SBATCH --ntasks-per-node={n_tasks_per_node}

            # Change to the run directory
            cd {run_dir}

            {run_command}
        """).strip()

    elif config.system_id == "RCAC_anvil":
        # Anvil (RCAC) SLURM script
        raise NotImplementedError(
            f"SLURM job script generation is not implemented for this system: {config.system_id}"
        )
    
    else:
        raise NotImplementedError(
            f"SLURM job script generation is not implemented for this system: {config.system_id}"
        )

    script_path.write_text(script_content)
    script_path.chmod(0o755)

    log_func(f"Generated SLURM batch script: {script_path}")
    log_func(f"  stdout: {stdout_path}")
    log_func(f"  stderr: {stderr_path}")

    return script_path


def _run_command(cmd: list[str], **kwargs: Any) -> str:
    """
    Convenience wrapper around subprocess.run that returns stdout.
    
    Parameters
    ----------
    cmd : list[str]
        Command and arguments to execute.
    **kwargs
        Additional keyword arguments forwarded to subprocess.run.
    
    Returns
    -------
    str
        Standard output from the command (stripped of trailing whitespace).
    """
    result = subprocess.run(
        cmd, check=True, text=True, capture_output=True, **kwargs
    )
    return result.stdout.strip()


def _get_conda_command() -> str:
    """Raise if a command is not found on PATH."""
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe is None:
        raise RuntimeError("Required command 'conda' not found on PATH.")
    return conda_exe

def _render_opt_base_dir_to_opt_dir(
    grid_name: str,
    parameters: Dict[str, Dict[str, Any]],
    opt_base_dir: Path,
    opt_dir: Path,
    overwrite: bool = False,
    log_func: Callable[[str], None] = print,
) -> None:
    """
    Stage and render model configuration templates using Jinja2.

    See original docstring in model_build.py for full details.
    """
    src = opt_base_dir.resolve()
    dst = opt_dir.resolve()

    if overwrite and dst.exists():
        log_func(f"[Render] Clearing existing opt_dir: {dst}")
        shutil.rmtree(dst)

    # Copy everything except an existing opt_<grid_name> directory
    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(f"opt_{grid_name}"),
    )

    env = Environment(
        loader=FileSystemLoader(str(dst)),
        undefined=StrictUndefined,
        autoescape=False,
    )

    for relpath, context in parameters.items():
        template_path = dst / relpath
        if not template_path.exists():
            raise FileNotFoundError(
                f"Template file '{relpath}' listed in parameters but not found in {dst}"
            )
        log_func(f"[Render] Rendering template: {relpath}")

        template = env.get_template(relpath)
        rendered = template.render(**context)

        st = template_path.stat()
        with template_path.open("w") as f:
            f.write(rendered)
        os.chmod(template_path, st.st_mode)


def _run_command_logged(
    label: str,
    logfile: Path,
    cmd: list[str],
    env: dict[str, str] | None = None,
    log_func: Callable[[str], None] = print,
) -> None:
    """
    Run a command, log stdout/stderr to a file, and fail loudly with context.

    All subprocess output is written only to logfile. High-level status
    messages go through log_func, which in build() is wired to write both
    to stdout and build.all.<token>.log.

    Parameters
    ----------
    label : str
        Human-readable label describing this build step.
    logfile : Path
        Path to the log file that will capture stdout/stderr.
    cmd : list[str]
        Command and arguments to execute.
    env : dict[str, str] or None, optional
        Environment variables to pass to subprocess.Popen. If None,
        the current process environment is used.
    log_func : callable, optional
        Logging function for high-level status messages.
    """
    log_func(f"[{label}] starting...")
    logfile.parent.mkdir(parents=True, exist_ok=True)

    with logfile.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        ret = proc.wait()

    if ret != 0:
        log_func(f"❌ {label} FAILED — see log: {logfile}")
        try:
            # Tail to stderr only; do not send to build.all log
            print(f"---- Last 50 lines of {logfile} ----", file=sys.stderr)
            with logfile.open() as f:
                lines = f.readlines()
            for line in lines[-50:]:
                sys.stderr.write(line)
            print("-------------------------------------", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"(could not read logfile: {e})", file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {ret}")

    log_func(f"[{label}] OK")


def _find_matching_build(
    builds_yaml: Path,
    fingerprint: dict,
    log_func: Callable[[str], None] = print,
) -> dict | None:
    """
    Look in builds.yaml for an entry whose configuration matches fingerprint.

    The comparison is done on a filtered view of each entry where the
    following keys are ignored:

      - token
      - timestamp_utc
      - exe   (we'll reuse whatever exe that entry points to)
      - clean
      - system

    Parameters
    ----------
    builds_yaml : Path
        Path to the builds.yaml file.
    fingerprint : dict
        Configuration fingerprint for the current build.
    log_func : callable, optional
        Logging function for informational messages.

    Returns
    -------
    dict or None
        The matching build entry dictionary if found and its recorded
        executable exists on disk; otherwise None.
    """
    if not builds_yaml.exists():
        return None

    with builds_yaml.open() as f:
        data = yaml.safe_load(f) or []

    if not isinstance(data, list):
        data = [data]

    ignore_keys = {"token", "timestamp_utc", "exe", "clean", "system"}

    def _filtered(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in ignore_keys}

    filtered_fingerprint = _filtered(fingerprint)

    log_func(f"Found {len(data)} existing build(s) in {builds_yaml}.")

    for entry in data:
        if not isinstance(entry, dict):
            continue

        entry_cfg = _filtered(entry)

        if entry_cfg == filtered_fingerprint:
            token = entry.get("token")
            exe_raw = entry.get("exe")
            log_func(f"Matching build found: token={token}")

            if not exe_raw:
                log_func("  -> exe field missing or empty in builds.yaml entry; skipping.")
                continue

            exe_path = Path(str(exe_raw)).expanduser()
            if exe_path.exists():
                log_func(f"  -> using existing executable at: {exe_path}")
                return entry
            else:
                log_func(
                    f"  -> recorded exe does not exist on filesystem: {exe_path}; skipping."
                )

    return None


def build(
    model_spec: ModelSpec,
    grid_name: str,
    input_data_path: Path,
    parameters: Dict[str, Dict[str, Any]],
    clean: bool = False,
    use_conda: bool = False,
    skip_inputs_check: bool = False,
) -> Optional[Path]:
    """
    Build the ocean model for a given grid and `model_name` (e.g., "roms-marbl").

    This is essentially the previous `build()` function from model_build.py,
    now using `ModelSpec` from this module.

    Parameters
    ----------
    model_spec : ModelSpec
        Model specification loaded from models.yml.
    grid_name : str
        Name of the grid configuration.
    input_data_path : Path
        Path to the directory containing the generated ROMS input files.
    parameters : Dict[str, Dict[str, Any]]
        Build parameters to pass to the model configuration.
    clean : bool, optional
        If True, clean the temporary build directory before building.
    use_conda : bool, optional
        If True, use conda to manage the build environment. If False (default),
        source a shell script from ROMS_ROOT/environments/{system}.sh.
    skip_inputs_check : bool, optional
        If True, skip the check for whether the input_data_path directory exists.
        Default is False.
    """
    # Unique build token and logging setup
    build_token = (
        datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )

    # Load model spec and derive directories
    opt_base_dir = config.paths.model_configs / model_spec.opt_base_dir
    build_root = config.paths.here / "builds" / f"{model_spec.name}_{grid_name}"
    build_root.mkdir(parents=True, exist_ok=True)

    opt_dir = build_root / "opt"
    opt_dir.mkdir(parents=True, exist_ok=True)

    # work in a temporary build directory in case clean=False
    build_dir_final = build_root / "bld"
    build_dir_tmp = build_root / "bld_tmp"
    if build_dir_tmp.exists() and clean:
        shutil.rmtree(build_dir_tmp)
    build_dir_tmp.mkdir(parents=True, exist_ok=True)

    roms_conda_env = model_spec.conda_env
    if "roms" not in model_spec.code or "marbl" not in model_spec.code:
        raise ValueError(f"Model spec {model_spec.name} must define at least 'roms' and 'marbl' in code.")

    logs_dir = build_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    build_all_log = logs_dir / f"build.{model_spec.name}.{build_token}.log"

    def log(msg: str = "") -> None:
        text = str(msg)
        print(text)
        build_all_log.parent.mkdir(parents=True, exist_ok=True)
        with build_all_log.open("a") as f:
            f.write(text + "\n")

    log(f"Build token: {build_token}")

    # Paths from config / sanity checks
    if not skip_inputs_check:
        if not input_data_path.is_dir():
            raise FileNotFoundError(
                f"Expected input data directory for grid '{grid_name}' at:\n"
                f"  {input_data_path}\n"
                "but it does not exist. Did you run the `gen_inputs` step?"
            )

    codes_root = config.paths.code_root
    # Use repo name as default directory name if not specified
    roms_root = codes_root / "ucla-roms"
    marbl_root = codes_root / "MARBL"

    log(f"Building {model_spec.name} for grid: {grid_name}")
    log(f"{model_spec.name} opt_base_dir : {opt_base_dir}")
    log(f"ROMS opt_dir      : {opt_dir}")
    log(f"ROMS build_dir    : {build_dir_final}")
    log(f"Input data path   : {input_data_path}")
    log(f"ROMS_ROOT         : {roms_root}")
    log(f"MARBL_ROOT        : {marbl_root}")
    log(f"Logs              : {logs_dir}")
    log(f"Build environment : {'conda' if use_conda else 'shell script'}")

    # Define build environment runner
    def _build_env_run(cmd: list[str]) -> list[str]:
        """
        Run a command in the build environment by sourcing a shell script.
        
        Sources {ROMS_ROOT}/environments/{system}.sh before running the command.
        """
        system_build_env_file = config.system
        roms_root_str = str(roms_root)
        
        # Convert command list to a single string for shell execution
        cmd_str = " ".join(shlex.quote(str(arg)) for arg in cmd)
        
        shell_cmd = textwrap.dedent(f"""
            pushd > /dev/null
            cd {shlex.quote(roms_root_str)}/environments
            source {shlex.quote(system_build_env_file)}.sh
            popd > /dev/null
            {cmd_str}
        """).strip()
        # Return a command that will be executed via bash -lc
        return ["bash", "-lc", shell_cmd]

    def _conda_run(cmd: list[str]) -> list[str]:
        """Run a command in the conda environment."""
        conda_exe = _get_conda_command()
        return [conda_exe, "run", "-n", roms_conda_env] + cmd

    # Choose the appropriate runner based on use_conda flag
    def _env_run(cmd: list[str]) -> list[str]:
        """Run a command in the build environment (conda or shell script)."""
        if use_conda:
            return _conda_run(cmd)
        else:
            return _build_env_run(cmd)

    # -----------------------------------------------------
    # Clone / update repos
    # -----------------------------------------------------
    if not (roms_root / ".git").is_dir():
        log(f"Cloning ROMS from {model_spec.code['roms'].location} into {roms_root}")
        _run_command(["git", "clone", model_spec.code["roms"].location, str(roms_root)])
    else:
        log(f"ROMS repo already present at {roms_root}")

    if model_spec.code["roms"].commit:
        log(f"Checking out ROMS {model_spec.code['roms'].commit}")
        _run_command(["git", "fetch", "--tags"], cwd=roms_root)
        _run_command(["git", "checkout", model_spec.code["roms"].commit], cwd=roms_root)

    if not (marbl_root / ".git").is_dir():
        log(f"Cloning MARBL from {model_spec.code['marbl'].location} into {marbl_root}")
        _run_command(["git", "clone", model_spec.code["marbl"].location, str(marbl_root)])
    else:
        log(f"MARBL repo already present at {marbl_root}")

    if model_spec.code["marbl"].commit:
        log(f"Checking out MARBL {model_spec.code['marbl'].commit}")
        _run_command(["git", "fetch", "--tags"], cwd=marbl_root)
        _run_command(["git", "checkout", model_spec.code["marbl"].commit], cwd=marbl_root)

    # -----------------------------------------------------
    # Sanity checks for directory trees
    # -----------------------------------------------------
    if not (roms_root / "src").is_dir():
        raise RuntimeError(f"ROMS_ROOT does not look correct: {roms_root}")
    
    if not (marbl_root / "src").is_dir():
        raise RuntimeError(f"MARBL_ROOT/src not found at {marbl_root}")

    # -----------------------------------------------------
    # Create conda env if needed (only when use_conda=True)
    # -----------------------------------------------------
    if use_conda:
        conda_exe = _get_conda_command()
        env_list = _run_command([conda_exe, "env", "list"])

        if roms_conda_env not in env_list:
            log(f"Creating conda env '{roms_conda_env}' from ROMS environment file...")
            env_yml = roms_root / "environments" / "conda_environment.yml"
            if not env_yml.exists():
                raise FileNotFoundError(f"Conda environment file not found: {env_yml}")
            _run_command(
                [
                    conda_exe,
                    "env",
                    "create",
                    "-f",
                    str(env_yml),
                    "--name",
                    roms_conda_env,
                ]
            )
        else:
            log(f"Conda env '{roms_conda_env}' already exists.")
    else:
        log(f"Using shell script environment: {config.system}.sh")
        env_script = roms_root / "environments" / f"{config.system}.sh"
        if not env_script.exists():
            raise FileNotFoundError(
                f"Build environment script not found: {env_script}\n"
                f"Expected at: {roms_root}/environments/{config.system}.sh"
            )

    # Toolchain checks
    try:
        _run_command(_env_run(["which", "gfortran"]))
        _run_command(_env_run(["which", "mpifort"]))
    except subprocess.CalledProcessError:
        env_name = roms_conda_env if use_conda else f"{config.system}.sh"
        raise RuntimeError(
            f"❌ gfortran or mpifort not found in build environment '{env_name}'. "
            "Check your build environment configuration."
        )

    compiler_kind = "gnu"
    try:
        mpifort_version = _run_command(_env_run(["mpifort", "--version"]))
        if any(token in mpifort_version.lower() for token in ["ifx", "ifort", "intel"]):
            compiler_kind = "intel"
    except Exception:
        pass

    log(f"Using compiler kind: {compiler_kind}")

    #-----------------------------------------------------
    # Build fingerprint & cache lookup
    # -----------------------------------------------------
    builds_yaml = config.paths.builds_yaml

    fingerprint = {
        "clean": bool(clean),
        "system": config.system,
        "compiler_kind": compiler_kind,
        "parameters": parameters,
        "grid_name": grid_name,
        "input_data_path": str(input_data_path),
        "logs_dir": str(logs_dir),
        "build_dir": str(build_dir_final),
        "marbl_root": str(marbl_root),
        "model_name": model_spec.name,
        "opt_base_dir": str(opt_base_dir),
        "opt_dir": str(opt_dir),
        "roms_conda_env": roms_conda_env,
        "roms_root": str(roms_root),
        "code": {
            name: {
                "location": spec.location,
                "commit": spec.commit,
            }
            for name, spec in model_spec.code.items()
        },
    }

    existing = _find_matching_build(builds_yaml, fingerprint, log_func=log)
    if existing is not None:
        exe_path = Path(existing.get("exe"))
        if not clean:
            log(
                "Found existing build matching current configuration; reusing executable."
            )
            log(f"  token : {existing.get('token')}")
            log(f"  exe   : {exe_path}")
            log("done.")
            return exe_path
        else:
            log(f"Clean build requested; attempting to remove existing executable: {exe_path}")
            try:
                if exe_path.exists() and exe_path.is_file():
                    try:
                        exe_path.chmod(0o755)
                    except OSError as e:
                        log(f"  ⚠️ chmod failed on exe before unlink: {e}")
                    exe_path.unlink()
                    log("  -> removed existing executable.")
                else:
                    log("  -> exe path missing or not a regular file; nothing to remove.")
            except OSError as e:
                log(f"⚠️ Failed to remove existing executable {exe_path}: {e}")
                log("Proceeding with clean rebuild; old exe may remain on disk.")


    # -----------------------------------------------------
    # Environment vars for builds
    # -----------------------------------------------------
    env = os.environ.copy()
    env["ROMS_ROOT"] = str(roms_root)
    env["MARBL_ROOT"] = str(marbl_root)
    env["GRID_NAME"] = grid_name
    env["BUILD_DIR"] = str(build_dir_tmp)

    if use_conda:
        try:
            conda_exe = _get_conda_command()
            conda_prefix = _run_command(
                [
                    conda_exe,
                    "run",
                    "-n",
                    roms_conda_env,
                    "python",
                    "-c",
                    "import os; print(os.environ['CONDA_PREFIX'])",
                ]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to determine CONDA_PREFIX for env '{roms_conda_env}'. "
                "Is the environment created correctly?"
            ) from exc

        env["MPIHOME"] = conda_prefix
        env["NETCDFHOME"] = conda_prefix
        env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + f":{conda_prefix}/lib"
    else:
        # For shell script environments, the environment variables should be set
        # by the sourced script. We'll query them from the environment after sourcing.
        # For now, we'll try to get them from the current environment or use defaults.
        # The sourced script should set MPIHOME, NETCDFHOME, etc.
        log("Using environment variables from sourced build script")
        # These will be set when commands are run via _build_env_run

    tools_path = str(roms_root / "Tools-Roms")
    env["PATH"] = tools_path + os.pathsep + env.get("PATH", "")

    # -----------------------------------------------------
    # Optional clean helper
    # -----------------------------------------------------
    def _maybe_clean(label: str, path: Path) -> None:
        if clean:
            log(f"[Clean] {label} ...")
            try:
                subprocess.run(
                    _env_run(["make", "-C", str(path), "clean"]),
                    check=False,
                    env=env,
                )
            except Exception as e:  # noqa: BLE001
                log(f"  ⚠️ clean failed for {label}: {e}")

    # -----------------------------------------------------
    # Builds (all via environment runner)
    # -----------------------------------------------------
    if use_conda:
        log(_run_command(_env_run(["conda", "list"])))

    log_marbl = logs_dir / f"build.MARBL.{build_token}.log"
    log_nhmg = logs_dir / f"build.NHMG.{build_token}.log"
    log_tools = logs_dir / f"build.Tools-Roms.{build_token}.log"
    log_roms = logs_dir / f"build.ROMS.{build_token}.log"

    # MARBL
    _maybe_clean("MARBL/src", marbl_root / "src")
    _run_command_logged(
        f"Build MARBL (compiler: {compiler_kind})",
        log_marbl,
        _env_run(
            ["make", "-C", str(marbl_root / "src"), compiler_kind, "USEMPI=TRUE"]
        ),
        env=env,
        log_func=log,
    )

    # NHMG (optional nonhydrostatic lib)
    _maybe_clean("NHMG/src", roms_root / "NHMG" / "src")
    _run_command_logged(
        "Build NHMG/src",
        log_nhmg,
        _env_run(["make", "-C", str(roms_root / "NHMG" / "src")]),
        env=env,
        log_func=log,
    )

    # Tools-Roms
    _maybe_clean("Tools-Roms", roms_root / "Tools-Roms")
    _run_command_logged(
        "Build Tools-Roms",
        log_tools,
        _env_run(["make", "-C", str(roms_root / "Tools-Roms")]),
        env=env,
        log_func=log,
    )

    # Render config files
    _render_opt_base_dir_to_opt_dir(
        grid_name=grid_name,
        parameters=parameters,
        opt_base_dir=opt_base_dir,
        opt_dir=opt_dir,
        overwrite=True,
        log_func=log,
    )

    _maybe_clean(f"ROMS ({opt_dir})", opt_dir)
    _run_command_logged(
        f"Build ROMS ({build_dir_tmp})",
        log_roms,
        _env_run(["make", "-C", str(opt_dir)]),
        env=env,
        log_func=log,
    )

    # Remove existing final directory if present
    if build_dir_final.exists():
        shutil.rmtree(build_dir_final)
    build_dir_tmp.rename(build_dir_final)

    # -----------------------------------------------------
    # Rename ROMS executable with token
    # -----------------------------------------------------
    exe_path = build_dir_final / "roms"
    exe_token_path = (
        build_root
        / "exe"
        / f"{model_spec.name}-{grid_name}-{build_token}"
    )
    exe_token_path.parent.mkdir(parents=True, exist_ok=True)

    if exe_path.exists():
        exe_path.rename(exe_token_path)
        log(f"{model_spec.name} executable -> {exe_token_path}")
    else:
        log(f"⚠️ {model_spec.name} executable not found at {exe_path}; not renamed.")


    # -----------------------------------------------------
    # Record build metadata in builds.yaml
    # -----------------------------------------------------
    if builds_yaml.exists():
        with builds_yaml.open() as f:
            builds_data = yaml.safe_load(f) or []
    else:
        builds_data = []

    if not isinstance(builds_data, list):
        builds_data = [builds_data]

    build_entry = {
        "token": build_token,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        **fingerprint,
        "exe": str(exe_token_path if exe_token_path.exists() else exe_path),
    }

    builds_data.append(build_entry)
    with builds_yaml.open("w") as f:
        yaml.safe_dump(builds_data, f)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    log("")
    log("✅ All builds completed.")
    log(f"• Build token:      {build_token}")
    log(f"• ROMS root:        {roms_root}")
    log(f"• MARBL root:       {marbl_root}")
    log(f"• App root:         {opt_base_dir}")
    log(f"• Logs:             {logs_dir}")
    log(
        f"• ROMS exe:         {exe_token_path if exe_token_path.exists() else exe_path}"
    )
    log(f"• builds.yaml:      {builds_yaml}")
    log("")

    return exe_token_path if exe_token_path.exists() else None



def run(
    model_spec: ModelSpec,
    grid_name: str,
    case: str,
    executable_path: Path,
    run_command: str,
    inputs: Dict[str, Any],
    cluster_type: Optional[str] = None,
    account: Optional[str] = None,
    queue: Optional[str] = None,
    wallclock_time: Optional[str] = None,
    n_nodes: Optional[int] = None,
    n_tasks_per_node: Optional[int] = None,
    log_func: Callable[[str], None] = print,
) -> None:
    """
    Run the model executable using the specified cluster type.
    
    Parameters
    ----------
    model_spec : ModelSpec
        Model specification.
    grid_name : str
        Name of the grid.
    case : str
        Case name for this run (used in job name and output directory).
    executable_path : Path
        Path to the model executable.
    run_command : str
        The command to execute (e.g., mpirun command).
    inputs : dict[str, InputObj]
        Dictionary of ROMS inputs (from inputs.inputs) used to populate
        template variables in the master_settings_file.
    cluster_type : str, optional
        Type of cluster/scheduler to use. Options: "LocalCluster", "SLURMCluster".
        Defaults based on config.system (MacOS → LocalCluster, others → SLURMCluster).
    account : str, optional
        Account for SLURM jobs (required for SLURMCluster).
    queue : str, optional
        Queue/partition for SLURM jobs (required for SLURMCluster).
    wallclock_time : str, optional
        Wallclock time limit for SLURM jobs in HH:MM:SS format (required for SLURMCluster).
    n_nodes : int, optional
        Number of nodes to request for SLURM jobs (required for SLURMCluster).
    n_tasks_per_node : int, optional
        Number of tasks per node to request for SLURM jobs (required for SLURMCluster).
    log_func : callable, optional
        Logging function for messages.
    
    Raises
    ------
    ValueError
        If required parameters are missing for the selected cluster type.
    RuntimeError
        If the executable doesn't exist or the run fails.
    """
    if not executable_path.exists():
        raise RuntimeError(f"Executable not found: {executable_path}")
    
    # Set default cluster type if not provided
    if cluster_type is None:
        from ._core import _default_cluster_type
        cluster_type = _default_cluster_type()
    
    # Set run directory internally with case
    run_dir = config.paths.run_dir / f"{model_spec.name}_{grid_name}" / case
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy settings input files from rendered opt directory to run directory
    build_root = config.paths.here / "builds" / f"{model_spec.name}_{grid_name}"
    opt_dir = build_root / "opt"
    
    if not opt_dir.exists():
        raise RuntimeError(
            f"Rendered opt directory not found: {opt_dir}. "
            f"Please run OcnModel.build() first."
        )
    
    # Get files to copy and render from run_time.filter.files
    files_to_process = list(model_spec.run_time.files)
    
    if not files_to_process:
        raise ValueError(
            f"Model '{model_spec.name}' has no files specified in 'run_time.filter.files'"
        )
    
    # Build context from inputs: map each input key to its path(s)
    # This context is used for templating all files
    context = {"CASENAME": case}
    for key, input_obj in inputs.items():
        if input_obj.paths is not None:
            # Convert Path or list[Path] to string or list of strings
            key_out = key.upper() + "_PATH"
            if isinstance(input_obj.paths, (list, tuple)):
                context[key_out] = "\n".join([str(p) for p in input_obj.paths])
            else:
                context[key_out] = str(input_obj.paths)
        else:
            raise ValueError(f"Input {key} has no paths")
    
    # Set up Jinja2 environment for templating
    env = Environment(
        loader=FileSystemLoader(str(opt_dir)),
        undefined=StrictUndefined,
        autoescape=False,
    )
    
    # Process each file: copy and render with template context
    log_func(f"Processing run-time files from {opt_dir} to {run_dir}:")
    for filename in files_to_process:
        src_file = opt_dir / filename
        dst_file = run_dir / filename
        
        if not src_file.exists():
            raise FileNotFoundError(
                f"Run-time file not found in opt directory: {src_file}"
            )
        
        # Render the template
        try:
            template = env.get_template(filename)
            rendered = template.render(**context)
            
            # Write rendered content to run directory
            dst_file.write_text(rendered)
            log_func(f"  Rendered {filename} -> {dst_file}")
        except Exception as e:
            # If templating fails, fall back to copying the file as-is
            log_func(f"  Warning: Template rendering failed for {filename}: {e}")
            log_func(f"  Copying file as-is: {filename} -> {dst_file}")
            shutil.copy2(src_file, dst_file)
    
    # Copy executable to run directory
    executable_name = executable_path.name
    run_executable = run_dir / executable_name
    log_func(f"Copying executable to run directory:")
    log_func(f"  {executable_path} -> {run_executable}")
    shutil.copy2(executable_path, run_executable)
    # Ensure executable has execute permissions
    run_executable.chmod(0o755)
    
    # Update run_command to use executable in run_dir
    # Replace the executable path in run_command with the run_dir executable
    run_command_updated = run_command.replace(str(executable_path), str(run_executable))
    
    # Create log file with case name and timestamp and append redirect to command
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    log_file = run_dir / f"{case}.{timestamp}.log"
    # Append shell redirect to send stdout and stderr to log file
    run_command_updated = f"{run_command_updated} > {log_file} 2>&1"
    
    from ._core import ClusterType
    
    if cluster_type == ClusterType.LOCAL:
        conda_env = model_spec.conda_env
        log_func(f"Running model locally in conda env '{conda_env}': {run_command_updated}")
        log_func(f"Working directory: {run_dir}")
        log_func(f"Log file: {log_file}")
        
        # Use conda run to execute in the correct environment
        # Parse the command (without the redirect) to get the base command
        import shlex
        # Remove the redirect from the command string
        if " > " in run_command_updated:
            cmd_part = run_command_updated.rsplit(" > ", 1)[0]
        else:
            cmd_part = run_command_updated
        
        # Build conda run command
        conda_cmd = ["conda", "run", "-n", conda_env, "--no-capture-output"] + shlex.split(cmd_part)
        
        # Run the command with redirection handled by subprocess
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as log_f:
            process = subprocess.Popen(
                conda_cmd,
                cwd=str(run_dir),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            
            # Wait for process to complete
            return_code = process.wait()
        
        if return_code != 0:
            raise RuntimeError(
                f"Model run failed with exit code {return_code}. "
                f"See log file for details: {log_file}"
            )
        
        log_func("Model run completed.")
        log_func(f"Log file: {log_file}")
        
        
    elif cluster_type == ClusterType.SLURM:
        # Validate required SLURM parameters
        if account is None:
            raise ValueError("'account' is required for SLURMCluster")
        if queue is None:
            raise ValueError("'queue' is required for SLURMCluster")
        if wallclock_time is None:
            raise ValueError("'wallclock_time' is required for SLURMCluster")
        
        job_name = f"{model_spec.name}_{grid_name}_{case}"
        
        # Generate batch script
        script_path = _generate_slurm_script(
            run_command=run_command_updated,
            job_name=job_name,
            account=account,
            queue=queue,
            wallclock_time=wallclock_time,
            n_nodes=n_nodes,
            n_tasks_per_node=n_tasks_per_node,
            run_dir=run_dir,
            log_func=log_func,
        )

        
        # Submit the job
        log_func(f"Submitting SLURM job: {job_name}")
        log_func(f"Log file: {log_file}")
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        log_func(result.stdout.strip())
        log_func(f"✅ Job submitted. Monitor with: squeue -u $USER")
        
    elif cluster_type == ClusterType.PBS:
        raise NotImplementedError("PBS cluster support not yet implemented")
        
    else:
        raise ValueError(
            f"Unknown cluster type: {cluster_type}. "
            f"Supported types: {ClusterType.LOCAL}, {ClusterType.SLURM}, {ClusterType.PBS}"
        )
