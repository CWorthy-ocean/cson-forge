from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import os
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import datetime
import uuid

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

import config
import roms_tools as rt
import source_data


# =========================================================
# Shared data structures (repos, model spec, inputs)
# =========================================================


@dataclass
class RepoSpec:
    """
    Specification for a code repository used in the build.

    Parameters
    ----------
    name : str
        Short name for the repository (e.g., "roms", "marbl").
    url : str
        Git URL for the repository.
    default_dirname : str
        Default directory name under the code root where this repo
        will be cloned.
    checkout : str, optional
        Optional tag, branch, or commit to check out after cloning.
    """
    name: str
    url: str
    default_dirname: str
    checkout: str | None = None


@dataclass
class ModelSpec:
    """
    Description of an ocean model configuration (e.g., ROMS/MARBL).

    Parameters
    ----------
    name : str
        Logical name of the model (e.g., "roms-marbl").
    opt_base_dir : str
        Relative path (under model-configs) to the base configuration
        directory.
    conda_env : str
        Name of the conda environment used to build/run this model.
    repos : dict[str, RepoSpec]
        Mapping from repo name to its specification.
    inputs : dict[str, dict]
        Per-input default arguments (from models.yml["<model>"]["inputs"]).
        These are merged with runtime arguments when constructing ROMS inputs.
    datasets : list[str]
        SourceData dataset keys required for this model (derived from inputs
        or explicitly listed in models.yml).
    settings_input_files : list[str]
        List of input files to copy from the rendered opt directory to the
        run directory before executing the model (e.g., ["roms.in", "marbl_in"]).
    master_settings_file : str
        Master settings file to append to the run command (e.g., "roms.in").
        This file should be in the run directory when the model executes.
    """
    name: str
    opt_base_dir: str
    conda_env: str
    repos: Dict[str, RepoSpec]
    inputs: Dict[str, Dict[str, Any]]
    datasets: List[str]
    settings_input_files: List[str]
    master_settings_file: str


def _extract_source_name(block: Union[str, Dict[str, Any], None]) -> Optional[str]:
    if block is None:
        return None
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return block.get("name")
    return None


def _dataset_keys_from_inputs(inputs: Dict[str, Dict[str, Any]]) -> set[str]:
    dataset_keys: set[str] = set()
    for cfg in inputs.values():
        if not isinstance(cfg, dict):
            continue
        for field_name in ("source", "bgc_source", "topography_source"):
            name = _extract_source_name(cfg.get(field_name))
            if not name:
                continue
            dataset_key = source_data.map_source_to_dataset_key(name)
            if dataset_key in source_data.DATASET_REGISTRY:
                dataset_keys.add(dataset_key)
    return dataset_keys


def _collect_datasets(block: Dict[str, Any], inputs: Dict[str, Dict[str, Any]]) -> List[str]:
    dataset_keys: set[str] = set()

    explicit = block.get("datasets") or []
    for item in explicit:
        if not item:
            continue
        dataset_keys.add(str(item).upper())

    dataset_keys.update(_dataset_keys_from_inputs(inputs))
    return sorted(dataset_keys)


def _load_models_yaml(path: Path, model: str) -> ModelSpec:
    """
    Load repository specifications, model metadata, and default input
    arguments from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the models.yaml file.
    model : str
        Name of the model block to load (e.g., "roms-marbl").

    Returns
    -------
    ModelSpec
        Parsed model specification including repository metadata and
        per-input defaults.

    Raises
    ------
    KeyError
        If the requested model is not present in the YAML file.
    """
    with path.open() as f:
        data = yaml.safe_load(f) or {}

    if model not in data:
        raise KeyError(f"Model '{model}' not found in models YAML file: {path}")

    block = data[model]

    repos: Dict[str, RepoSpec] = {}
    for key, val in block.get("repos", {}).items():
        repos[key] = RepoSpec(
            name=key,
            url=val["url"],
            default_dirname=val.get("default_dirname", key),
            checkout=val.get("checkout"),
        )

    inputs = block.get("inputs", {}) or {}
    datasets = _collect_datasets(block, inputs)
    settings_input_files = block.get("settings_input_files", []) or []
    
    if "master_settings_file" not in block:
        raise KeyError(
            f"Model '{model}' must specify 'master_settings_file' in models.yml"
        )
    master_settings_file = block["master_settings_file"]

    return ModelSpec(
        name=model,
        opt_base_dir=block["opt_base_dir"],
        conda_env=block["conda_env"],
        repos=repos,
        inputs=inputs,
        datasets=datasets,
        settings_input_files=settings_input_files,
        master_settings_file=master_settings_file,
    )


# =========================================================
# ROMS input generation (from former model_config.py)
# =========================================================


class InputStep:
    """Metadata for a single ROMS input generation step."""

    def __init__(self, name: str, order: int, label: str, handler: Callable):
        self.name = name  # canonical key used for filenames & paths
        self.order = order  # execution order
        self.label = label  # human-readable label
        self.handler = handler  # function expecting `self` (ROMSInputs instance)


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
class ROMSInputs:
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
        Serialize a summary of ROMSInputs state to a YAML blueprint:

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
                return str(obj)

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

        with self.bp_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=True)

        print(f"📄  Wrote ROMSInputs blueprint to {self.bp_path}")


# =========================================================
# Build logic (from former model_build.py)
# =========================================================


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
    if "roms" not in model_spec.repos or "marbl" not in model_spec.repos:
        raise ValueError(f"Model spec {model_spec.name} must define at least 'roms' and 'marbl' repos.")

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
    roms_root = codes_root / model_spec.repos["roms"].default_dirname
    marbl_root = codes_root / model_spec.repos["marbl"].default_dirname

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
        
        shell_cmd = f"""
pushd > /dev/null
cd {shlex.quote(roms_root_str)}/environments
source {shlex.quote(system_build_env_file)}.sh
popd > /dev/null
{cmd_str}
"""
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
        log(f"Cloning ROMS from {model_spec.repos['roms'].url} into {roms_root}")
        _run_command(["git", "clone", model_spec.repos["roms"].url, str(roms_root)])
    else:
        log(f"ROMS repo already present at {roms_root}")

    if model_spec.repos["roms"].checkout:
        log(f"Checking out ROMS {model_spec.repos['roms'].checkout}")
        _run_command(["git", "fetch", "--tags"], cwd=roms_root)
        _run_command(["git", "checkout", model_spec.repos["roms"].checkout], cwd=roms_root)

    if not (marbl_root / ".git").is_dir():
        log(f"Cloning MARBL from {model_spec.repos['marbl'].url} into {marbl_root}")
        _run_command(["git", "clone", model_spec.repos["marbl"].url, str(marbl_root)])
    else:
        log(f"MARBL repo already present at {marbl_root}")

    if model_spec.repos["marbl"].checkout:
        log(f"Checking out MARBL {model_spec.repos['marbl'].checkout}")
        _run_command(["git", "fetch", "--tags"], cwd=marbl_root)
        _run_command(["git", "checkout", model_spec.repos["marbl"].checkout], cwd=marbl_root)

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
        "repos": {
            name: {
                "url": spec.url,
                "default_dirname": spec.default_dirname,
                "checkout": spec.checkout,
            }
            for name, spec in model_spec.repos.items()
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


# =========================================================
# Model execution (run) functions
# =========================================================


class ClusterType:
    """Constants for cluster/scheduler types."""
    LOCAL = "LocalCluster"
    SLURM = "SLURMCluster"
    PBS = "PBSCluster"  # For future extensibility


def _default_cluster_type() -> str:
    """
    Return the default cluster type based on the current system.
    
    Returns
    -------
    str
        "LocalCluster" for MacOS, "SLURMCluster" for other systems.
    """
    if config.system == "MacOS":
        return ClusterType.LOCAL
    else:
        return ClusterType.SLURM


def _generate_slurm_script(
    run_command: str,
    job_name: str,
    account: str,
    queue: str,
    wallclock_time: str,
    run_dir: Path,
    conda_env: str,
    log_func: Callable[[str], None] = print,
) -> Path:
    """
    Generate a SLURM batch script for running the model.
    
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
    run_dir : Path
        Directory where the batch script and output files will be written.
    conda_env : str
        Name of the conda environment to run the model in.
    log_func : callable, optional
        Logging function for messages.
    
    Returns
    -------
    Path
        Path to the generated batch script.
    """
    script_path = run_dir / f"{job_name}.sh"
    stdout_path = run_dir / f"{job_name}.out"
    stderr_path = run_dir / f"{job_name}.err"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={queue}
#SBATCH --time={wallclock_time}
#SBATCH --output={stdout_path}
#SBATCH --error={stderr_path}

# Change to the run directory
cd {run_dir}

# Initialize conda (if not already initialized)
eval "$(conda shell.bash hook)"

# Run the model in the conda environment
conda run -n {conda_env} {run_command}
"""
    
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    
    log_func(f"Generated SLURM batch script: {script_path}")
    log_func(f"  stdout: {stdout_path}")
    log_func(f"  stderr: {stderr_path}")
    
    return script_path


def run(
    model_spec: ModelSpec,
    grid_name: str,
    case: str,
    input_data_path: Path,
    executable_path: Path,
    run_command: str,
    inputs: Dict[str, Any],
    cluster_type: Optional[str] = None,
    account: Optional[str] = None,
    queue: Optional[str] = None,
    wallclock_time: Optional[str] = None,
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
    input_data_path : Path
        Path to the input data directory.
    executable_path : Path
        Path to the model executable.
    run_command : str
        The command to execute (e.g., mpirun command).
    inputs : dict[str, InputObj]
        Dictionary of ROMS inputs (from ROMSInputs.inputs) used to populate
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
    
    if cluster_type is None:
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
    
    # Ensure master_settings_file is included in files to copy
    files_to_copy = list(model_spec.settings_input_files)
    if model_spec.master_settings_file not in files_to_copy:
        files_to_copy.append(model_spec.master_settings_file)
    
    if files_to_copy:
        log_func(f"Copying settings input files from {opt_dir} to {run_dir}:")
        for filename in files_to_copy:
            src_file = opt_dir / filename
            dst_file = run_dir / filename
            
            if not src_file.exists():
                raise FileNotFoundError(
                    f"Settings input file not found in opt directory: {src_file}"
                )
            
            shutil.copy2(src_file, dst_file)
            log_func(f"  {filename} -> {dst_file}")
    
    # Render master_settings_file with input paths
    master_settings_src = opt_dir / model_spec.master_settings_file
    master_settings_dst = run_dir / model_spec.master_settings_file
    
    if master_settings_src.exists():
        log_func(f"Rendering master settings file: {model_spec.master_settings_file}")
        
        # Build context from inputs: map each input key to its path(s)
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
        
        # Render the template
        env = Environment(
            loader=FileSystemLoader(str(opt_dir)),
            undefined=StrictUndefined,
            autoescape=False,
        )
        template = env.get_template(model_spec.master_settings_file)
        rendered = template.render(**context)
        
        # Write to run directory
        master_settings_dst.write_text(rendered)
        log_func(f"  Rendered {model_spec.master_settings_file} -> {master_settings_dst}")
    else:
        raise FileNotFoundError(
            f"Master settings file not found in opt directory: {master_settings_src}"
        )
    
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
        
        job_name = f"{model_spec.name}-{grid_name}-{case}"
        
        # Generate batch script
        script_path = _generate_slurm_script(
            run_command=run_command_updated,
            job_name=job_name,
            account=account,
            queue=queue,
            wallclock_time=wallclock_time,
            run_dir=run_dir,
            conda_env=model_spec.conda_env,
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


# =========================================================
# High-level OcnModel object
# =========================================================


@dataclass
class OcnModel:
    """
    High-level object:
      - model metadata from models.yml (ModelSpec),
      - source datasets (SourceData),
      - ROMS input generation (ROMSInputs),
      - model build (via `build()`).

    Typical usage
    -------------
    grid_kwargs = dict(
        nx=10,
        ny=10,
        size_x=4000,
        size_y=2000,
        center_lon=4.0,
        center_lat=-1.0,
        rot=0,
        N=5,
    )

    ocn = OcnModel(
        model_name="roms-marbl",
        grid_name=grid_name,
        grid_kwargs=grid_kwargs,
        boundaries=boundaries,
        start_time=start_time,
        end_time=end_time,
        np_eta=np_eta,
        np_xi=np_xi,
    )

    ocn.prepare_source_data()
    ocn.generate_inputs()
    ocn.build()
    """

    model_name: str
    grid_name: str
    grid_kwargs: Dict[str, Any]
    boundaries: dict
    start_time: object
    end_time: object
    np_eta: int
    np_xi: int
    grid: object = field(init=False)
    spec: ModelSpec = field(init=False)
    src_data: Optional[source_data.SourceData] = field(init=False, default=None)
    inputs: Optional[ROMSInputs] = field(init=False, default=None)
    executable: Optional[Path] = field(init=False, default=None)
    
    def __post_init__(self):
        self.grid = rt.Grid(**self.grid_kwargs)
        self.spec = _load_models_yaml(config.paths.models_yaml, self.model_name)
   
    @property
    def input_data_dir(self) -> Path:
        return config.paths.input_data / f"{self.model_name}_{self.grid_name}"

    @property
    def name(self) -> str:
        return f"{self.spec.name}_{self.grid_name}"

    @property
    def _run_command(self) -> str:
        """
        Return the mpirun command to execute the model.
        
        Returns
        -------
        str
            The mpirun command string with the number of processes
            (np_xi * np_eta), the executable path, and the master settings file.
        
        Raises
        ------
        RuntimeError
            If the executable has not been built yet.
        """
        if self.executable is None:
            raise RuntimeError(
                "Executable not built yet. Call OcnModel.build() first."
            )
        nprocs = self.np_xi * self.np_eta
        return f"mpirun -n {nprocs} {self.executable} {self.spec.master_settings_file}"

    def prepare_source_data(self, clobber: bool = False):
        self.src_data = source_data.SourceData(
            datasets=self.spec.datasets,
            clobber=clobber,
            grid=self.grid,
            grid_name=self.grid_name,
            start_time=self.start_time,
            end_time=self.end_time,
        ).prepare_all()
    
    def generate_inputs(
        self,
        clobber: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate ROMS input files for this model/grid.

        The list of inputs to generate is automatically derived from the
        keys in models.yml["<model_name>"]["inputs"].

        Parameters
        ----------
        clobber : bool, optional
            Passed through to ROMSInputs to allow overwriting existing
            NetCDF files.
        
        Returns
        -------
        dict
            Dictionary mapping input keys to their corresponding objects
            (e.g., grid, InitialConditions, SurfaceForcing, etc.).
        
        Raises
        ------
        RuntimeError
            If `prepare_source_data()` has not been called yet.
        """
        if self.src_data is None:
            raise RuntimeError(
                "You must call OcnModel.prepare_source_data() "
                "before generating inputs."
            )
        self.inputs = ROMSInputs(
            model_name=self.model_name,
            grid_name=self.grid_name,
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            boundaries=self.boundaries,
            source_data=self.src_data,
            model_spec=self.spec,
            clobber=clobber,
        ).generate_all()

        return self.inputs.obj

    def build(
        self, 
        parameters: Dict[str, Dict[str, Any]], 
        clean: bool = False, 
        skip_inputs_check: bool = False
    ) -> Path:
        """
        Build the model executable for this configuration, using the
        lower-level `build()` helper in this module.

        Parameters
        ----------
        parameters : dict
            Build-time parameter overrides for the build.
        clean : bool, optional
            If True, clean the existing build directory before building.
        skip_inputs_check : bool, optional
            If True, skip the check for whether inputs have been generated. Default is False.
        """
        if not skip_inputs_check and self.inputs is None:
            raise RuntimeError(
                "You must call OcnModel.generate_inputs() before building the model. "
                "If you wish to skip this check, pass skip_inputs_check=True to build()."
            )

        use_conda = config.system == "MacOS"
        
        exe_path = build(
            model_spec=self.spec,
            grid_name=self.grid_name,
            input_data_path=self.input_data_dir,
            parameters=parameters,
            clean=clean,
            use_conda=use_conda,
            skip_inputs_check=skip_inputs_check,
        )
        if exe_path is None:
            raise RuntimeError(
                "Build completed but executable was not found. "
                "Check the build logs for errors."
            )
        self.executable = exe_path
        return self.executable

    def run(
        self,
        case: str,
        cluster_type: Optional[str] = None,
        account: Optional[str] = None,
        queue: Optional[str] = None,
        wallclock_time: Optional[str] = None,
    ) -> None:
        """
        Run the model executable for this configuration.

    Parameters
    ----------
        case : str
            Case name for this run (used in job name and output directory).
        cluster_type : str, optional
            Type of cluster/scheduler to use. Options: "LocalCluster", "SLURMCluster".
            Defaults based on config.system (MacOS → LocalCluster, others → SLURMCluster).
        account : str, optional
            Account for SLURM jobs (required for SLURMCluster).
        queue : str, optional
            Queue/partition for SLURM jobs (required for SLURMCluster).
        wallclock_time : str, optional
            Wallclock time limit for SLURM jobs in HH:MM:SS format (required for SLURMCluster).
        
        Raises
        ------
        RuntimeError
            If inputs haven't been generated or executable hasn't been built.
        """
        if self.inputs is None:
            raise RuntimeError(
                "You must call OcnModel.generate_inputs() "
                "before running the model."
            )

        if self.executable is None:
            raise RuntimeError(
                "You must call OcnModel.build() "
                "before running the model."
            )
        
        run(
            model_spec=self.spec,
            grid_name=self.grid_name,
            case=case,
            input_data_path=self.inputs.input_data_dir,
            executable_path=self.executable,
            run_command=self._run_command,
            inputs=self.inputs.inputs,
            cluster_type=cluster_type,
            account=account,
            queue=queue,
            wallclock_time=wallclock_time,
        )

