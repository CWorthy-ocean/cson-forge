from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Any, List, Union, Optional

import yaml

import xarray as xr

import config
import roms_tools as rt


# ---------------------------------------------------
# Input registry (name -> metadata & handler)
# ---------------------------------------------------


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
        Key for this input (e.g., 'grd', 'ic', 'frc').
        This will be used in filenames, and to index `input_objs[name]`.
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


# ---------------------------------------------------
# InputObj dataclass (per-input metadata)
# ---------------------------------------------------


@dataclass
class InputObj:
    """
    Structured representation of a single ROMS input product.

    Attributes
    ----------
    paths : Path | list[Path] | None
        Path or list of paths to the generated NetCDF file(s), if applicable.
    yaml_file : Path | None
        Path to the YAML description written for this input, if any.
    obj : Any
        The underlying roms_tools (or grid / partition result) object used to
        generate the input.
    method : Callable | str | None
        The constructor or function used to create `obj` (e.g., rt.InitialConditions).
        For inputs that don't come from a constructor call (e.g., a pre-existing
        grid object), this may be None or a descriptive string.
    input_args : dict
        The keyword arguments passed into `method` to construct `obj`.
    """

    paths: Optional[Union[Path, List[Path]]] = None
    paths_partitioned: Optional[Union[Path, List[Path]]] = None
    yaml_file: Optional[Path] = None
    obj: Any = None
    method: Optional[Union[Callable[..., Any], str]] = None
    input_args: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------
# ROMSInputs dataclass
# ---------------------------------------------------


@dataclass
class ROMSInputs:
    """Generate and manage ROMS input files for a given grid."""

    # dataclass fields (constructor args)
    grid_name: str
    grid: object
    start_time: object
    end_time: object
    np_eta: int
    np_xi: int
    boundaries: dict
    source_data: object

    # Make roms_input_list configurable per instance
    roms_input_list: List[str] = field(
        default_factory=lambda: [
            "grd",
            "ic",
            "frc",
            "frc_bgc",
            "bry",
            "bry_bgc",
            "rivers",
            "cdr",
        ]
    )

    use_dask: bool = True
    clobber: bool = False

    # derived / internal fields (not passed by user)
    input_data_dir: Path = field(init=False)
    glorys_path: Path = field(init=False)
    bgc_forcing_path: Path = field(init=False)
    input_objs: Dict[str, InputObj] = field(init=False)
    bp_path: Path = field(init=False)
    
    def __post_init__(self):
        # Path to input directory
        self.input_data_dir = Path(config.paths.input_data) / self.grid_name
        self.input_data_dir.mkdir(exist_ok=True)

        self.bp_path = config.paths.blueprints / f"model-inputs_{self.grid_name}.yml"
        self.bp_path.parent.mkdir(parents=True, exist_ok=True)

        # Source data paths from SourceData
        self.glorys_path = self.source_data.paths["GLORYS_REGIONAL"]
        self.bgc_forcing_path = self.source_data.paths["UNIFIED_BGC"]

        # Storage for detailed per-input objects
        self.input_objs = {}

    # ----------------------------
    # Public API
    # ----------------------------

    def generate_all(self):
        """
        Generate all ROMS input files for this grid using the registered
        steps whose names appear in `roms_input_list`, then partition and
        write a blueprint.

        If any names in `roms_input_list` lack registered handlers,
        a ValueError is raised.
        """

        if not self._ensure_empty_or_clobber(self.clobber):
            return self

        # Sanity check
        registry_keys = set(INPUT_REGISTRY.keys())
        needed = set(self.roms_input_list)
        missing = sorted(needed - registry_keys)
        if missing:
            raise ValueError(
                "The following ROMS inputs are listed in `roms_input_list` but "
                f"have no registered handlers: {', '.join(missing)}"
            )

        # Use only the selected steps
        steps = [INPUT_REGISTRY[name] for name in self.roms_input_list]
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

        Returns
        -------
        bool
            True if it's safe to proceed with generation, False if we should
            skip (e.g., files exist and clobber=False).
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
        """
        Construct the NetCDF filename stem for a given input key.
        """
        return self.input_data_dir / f"roms_{key}.nc"

    def _yaml_filename(self, key: str) -> Path:
        """
        Construct the YAML blueprint filename for a given input key.
        """
        return config.paths.blueprints / f"{self.grid_name}.{key}.yaml"

    # ----------------------------
    # Registry-backed generation steps
    # ----------------------------

    @register_input(name="grd", order=10, label="Writing ROMS grid")
    def _generate_grid(self, key: str = "grd", **kwargs):
        out_path = self._forcing_filename(key)
        yaml_path = self._yaml_filename(key)

        self.grid.save(out_path)
        self.grid.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=out_path,
            yaml_file=yaml_path,
            obj=self.grid,
            method=None,  # no roms_tools constructor here
            input_args={},
        )

    @register_input(name="ic", order=20, label="Generating initial conditions")
    def _generate_initial_conditions(self, key: str = "ic", **kwargs):
        yaml_path = self._yaml_filename(key)
        input_args = dict(
            ini_time=self.start_time,
            source={
                "name": "GLORYS",
                "path": self.glorys_path,
            },
            bgc_source={
                "name": "UNIFIED",
                "path": self.bgc_forcing_path,
                "climatology": True,
            },
            use_dask=self.use_dask,
        )
        ic = rt.InitialConditions(grid=self.grid, **input_args)
        paths = ic.save(self._forcing_filename(key))
        ic.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=paths,
            yaml_file=yaml_path,
            obj=ic,
            method=rt.InitialConditions,
            input_args=input_args,
        )

    @register_input(name="frc", order=30, label="Generating surface forcing (physics)")
    def _generate_surface_forcing(self, key: str = "frc", **kwargs):
        yaml_path = self._yaml_filename(key)
        input_args = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            source={"name": "ERA5"},
            type="physics",
            use_dask=self.use_dask,
        )
        frc = rt.SurfaceForcing(grid=self.grid, **input_args)
        paths = frc.save(self._forcing_filename(key))
        frc.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=paths,
            yaml_file=yaml_path,
            obj=frc,
            method=rt.SurfaceForcing,
            input_args=input_args,
        )

    @register_input(name="frc_bgc", order=40, label="Generating surface forcing (BGC)")
    def _generate_bgc_surface_forcing(self, key: str = "frc_bgc", **kwargs):
        yaml_path = self._yaml_filename(key)
        input_args = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            source={
                "name": "UNIFIED",
                "path": self.bgc_forcing_path,
                "climatology": True,
            },
            type="bgc",
            use_dask=self.use_dask,
        )
        frc_bgc = rt.SurfaceForcing(grid=self.grid, **input_args)
        paths = frc_bgc.save(self._forcing_filename(key))
        frc_bgc.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=paths,
            yaml_file=yaml_path,
            obj=frc_bgc,
            method=rt.SurfaceForcing,
            input_args=input_args,
        )

    @register_input(name="bry", order=50, label="Generating boundary forcing (physics)")
    def _generate_boundary_forcing(self, key: str = "bry", **kwargs):
        yaml_path = self._yaml_filename(key)
        input_args = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            source={
                "name": "GLORYS",
                "path": self.glorys_path,
            },
            type="physics",
            use_dask=self.use_dask,
        )
        bry = rt.BoundaryForcing(grid=self.grid, **input_args)
        paths = bry.save(self._forcing_filename(key))
        bry.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=paths,
            yaml_file=yaml_path,
            obj=bry,
            method=rt.BoundaryForcing,
            input_args=input_args,
        )

    @register_input(name="bry_bgc", order=60, label="Generating boundary forcing (BGC)")
    def _generate_bgc_boundary_forcing(self, key: str = "bry_bgc", **kwargs):
        yaml_path = self._yaml_filename(key)
        input_args = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            boundaries=self.boundaries,
            source={
                "name": "UNIFIED",
                "path": self.bgc_forcing_path,
                "climatology": True,
            },
            type="bgc",
            use_dask=self.use_dask,
        )
        bry_bgc = rt.BoundaryForcing(grid=self.grid, **input_args)
        paths = bry_bgc.save(self._forcing_filename(key))
        bry_bgc.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=paths,
            yaml_file=yaml_path,
            obj=bry_bgc,
            method=rt.BoundaryForcing,
            input_args=input_args,
        )

    @register_input(name="rivers", order=70, label="Generating river forcing")
    def _generate_river_forcing(self, key: str = "rivers", **kwargs):
        yaml_path = self._yaml_filename(key)
        input_args = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            include_bgc=True,
        )
        rivers = rt.RiverForcing(grid=self.grid, **input_args)
        paths = rivers.save(self._forcing_filename(key))
        rivers.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=paths,
            yaml_file=yaml_path,
            obj=rivers,
            method=rt.RiverForcing,
            input_args=input_args,
        )

    @register_input(name="cdr", order=80, label="Generating CDR forcing")
    def _generate_cdr_forcing(self, key: str = "cdr", cdr_list=None, **kwargs):
        if cdr_list is None:
            cdr_list = []
        if not cdr_list:
            return

        yaml_path = self._yaml_filename(key)
        input_args = dict(
            start_time=self.start_time,
            end_time=self.end_time,
            releases=cdr_list,
        )
        cdr = rt.CDRForcing(grid=self.grid, **input_args)
        paths = cdr.save(self._forcing_filename(key))
        cdr.to_yaml(yaml_path)

        self.input_objs[key] = InputObj(
            paths=paths,
            yaml_file=yaml_path,
            obj=cdr,
            method=rt.CDRForcing,
            input_args=input_args,
        )

    # ----------------------------
    # Partition step (not in registry)
    # ----------------------------

    def _partition_files(self, **kwargs):
        """
        Partition whole input files across tiles using roms_tools.partition_netcdf.

        Uses the paths stored in `input_objs[...]` (for keys in roms_input_list)
        to build the list of whole-field files, then stores the partitioning
        result on `self.files_partitioned`. No YAML file is written for this
        step, and it is not part of INPUT_REGISTRY.
        """
        input_args = dict(
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            output_dir=self.input_data_dir,
            include_coarse_dims=False,
        )

        for name in self.roms_input_list:
            self.input_objs[name].paths_partitioned = rt.partition_netcdf(
                self.input_objs[name].paths, **input_args,
                )

    # ----------------------------
    # Blueprint writer
    # ----------------------------

    def _write_inputs_blueprint(self):
        """
        Serialize a summary of ROMSInputs state to a YAML blueprint:

            blueprints/model-inputs_{grid_name}.yml

        Contents include:
          - files_whole
          - files_partitioned
          - input_data_dir
          - roms_input_list
          - input_objs

        Paths are converted to strings, dataclasses are expanded to dicts,
        xarray Datasets/DataArrays are replaced with a short placeholder,
        and non-serializable objects (e.g., roms_tools callables) are
        represented via repr()/qualname.
        """

        XR_TYPES = (xr.Dataset, xr.DataArray)

        def _serialize(obj: Any) -> Any:
            from datetime import date, datetime
            from dataclasses import is_dataclass, asdict as dc_asdict

            # Drop/replace xarray objects explicitly
            if XR_TYPES and isinstance(obj, XR_TYPES):
                # You can also return None here if you prefer to omit them silently
                return None

            # Only treat *instances* of dataclasses this way, not classes/types.
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

            # For callables (e.g., rt.InitialConditions in InputObj.method),
            # store a readable identifier instead of the raw object.
            if callable(obj):
                qualname = getattr(obj, "__qualname__", None)
                mod = getattr(obj, "__module__", None)
                if qualname and mod:
                    return f"{mod}.{qualname}"
                return repr(obj)

            # Last resort: string representation
            return repr(obj)

        # Build a pared-down snapshot rather than dumping the entire dataclass
        raw = dict(
            grid_name=self.grid_name,
            start_time=self.start_time,
            end_time=self.end_time,
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            boundaries=self.boundaries,
            source_data=self.source_data,
            input_data_dir=self.input_data_dir,
            roms_input_list=self.roms_input_list,
            inputs=self.input_objs,
        )

        data = _serialize(raw)

        with self.bp_path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=True)

        print(f"📄  Wrote ROMSInputs blueprint to {self.bp_path}")


# ---------------------------------------------------
# convenience function that mirrors original API
# ---------------------------------------------------


def gen_inputs(
    grid_name,
    grid,
    roms_input_list,
    start_time,
    end_time,
    np_eta,
    np_xi,
    boundaries,
    source_data,
    clobber: bool = False,
):
    """
    Convenience wrapper to construct ROMSInputs and call generate_all().
    """
    roms_inputs = ROMSInputs(
        grid_name=grid_name,
        grid=grid,
        roms_input_list=roms_input_list,
        start_time=start_time,
        end_time=end_time,
        np_eta=np_eta,
        np_xi=np_xi,
        boundaries=boundaries,
        source_data=source_data,
        clobber=clobber,
    )
    roms_inputs.generate_all()
    return roms_inputs
