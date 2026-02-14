"""
CstarSpecBuilder - Pydantic-based builder for C-Star blueprints.

This class provides a Pydantic-based interface for building RomsMarblBlueprint objects.
"""
from __future__ import annotations

import copy
import time
import warnings
from dataclasses import asdict as dataclass_asdict
from datetime import datetime
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import xarray as xr
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

import cstar.orchestration.models as cstar_models
from cstar.orchestration.serialization import deserialize
from cstar.roms import ROMSSimulation
from cstar.execution.handler import ExecutionStatus
from . import config
from . import source_data
from . import models as cson_models
from . import input_data
from .settings import render_roms_settings
from .util import compute_timestep_from_cfl
import roms_tools as rt


class DatasetsDict(dict):
    """Dictionary-like class that supports method call with key parameter."""
    
    def __call__(self, key: Optional[str] = None):
        """
        Return a specific dataset by key, or the whole dictionary if key is None.
        
        Parameters
        ----------
        key : str, optional
            Key to retrieve. If None, returns self.
        
        Returns
        -------
        Union[xr.Dataset, List[xr.Dataset], dict]
            The dataset(s) for the key, or the whole dictionary if key is None.
        """
        if key is None:
            return self
        return self.get(key)


class BlueprintStage:
    """
    Blueprint stage constants and validation.
    
    Valid stages:
    - PRECONFIG: Blueprint before configuration
    - POSTCONFIG: Blueprint after configuration
    - BUILD: Blueprint after building/compiling the model
    - RUN: Blueprint for running the simulation
    """
    PRECONFIG: str = "preconfig"
    POSTCONFIG: str = "postconfig"
    BUILD: str = "build"
    RUN: str = "run"
    
    # Numerical values for stage comparison
    N_PRECONFIG: int = 0
    N_POSTCONFIG: int = 1
    N_BUILD: int = 2
    N_RUN: int = 3
    
    @classmethod
    def validate_stage(cls, stage: str) -> str:
        """Validate that stage is one of the valid values."""
        valid_stages = {cls.PRECONFIG, cls.POSTCONFIG, cls.BUILD, cls.RUN}
        if stage not in valid_stages:
            raise ValueError(f"stage must be one of {valid_stages}, got {stage}")
        return stage
    
    @classmethod
    def get_stage_value(cls, stage: str) -> int:
        """Get the numerical value of a stage for comparison."""
        stage_map = {
            cls.PRECONFIG: cls.N_PRECONFIG,
            cls.POSTCONFIG: cls.N_POSTCONFIG,
            cls.BUILD: cls.N_BUILD,
            cls.RUN: cls.N_RUN,
        }
        return stage_map.get(stage, -1)





class CstarSpecBuilder(BaseModel):
    """
    Builder for C-Star RomsMarblBlueprint specifications.
    
    This class provides a Pydantic-based interface for constructing
    and managing ROMS-MARBL blueprints through a staged workflow.
    
    **Workflow and Stage Progression:**
    
    The builder progresses through distinct stages, each representing a
    phase of the model configuration and execution pipeline:
    
    1. **PRECONFIG** (initialization):
       - Created during `model_post_init()` via `_initialize_blueprint()`
       - Blueprint structure initialized with placeholder data
       - Settings dictionaries initialized from model defaults
       - Blueprint persisted to disk
       
    2. **POSTCONFIG** (input generation):
       - Achieved by calling `generate_inputs()`
       - Source data prepared, input files generated (grid, initial conditions, forcing)
       - Blueprint updated with actual data file locations
       - Settings updated with input-specific values
       - Blueprint persisted to disk
       
    3. **BUILD** (configuration):
       - Achieved by calling `configure_build()`
       - Jinja2 templates rendered with current settings
       - Blueprint updated with rendered code locations
       - Blueprint persisted to disk
       - ROMSSimulation instance created
       
    4. **RUN** (execution):
       - Achieved by calling `run()` after `build()`
       - Blueprint persisted with runtime parameters
       - Model executable runs
       
    **Key Concepts:**
    
    - Settings are stored in sidecar YAML files (not in blueprint itself)
    - Blueprint state is persisted to disk at each stage transition
    - Grid object is created during initialization and reused throughout
    - Source data can be prepared independently via `ensure_source_data()`
    
    .. warning::
        This functionality is under active development and not yet fully implemented.
        Some methods (e.g., `build()` and `run()`) may raise `NotImplementedError`.
        Use with caution.
    """
    
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    
    # User inputs
    description: str = "Generated blueprint"
    model_name: str
    grid_name: str
    grid_kwargs: Dict[str, Any]
    grid_kwargs_parent: Optional[Dict[str, Any]] = Field(default=None, validate_default=False)
    grid_kwargs_child: Optional[Dict[str, Any]] = Field(default=None, validate_default=False)
    open_boundaries: cson_models.OpenBoundaries
    partitioning: cstar_models.PartitioningParameterSet
    start_date: datetime = Field(alias="start_time")
    end_date: datetime = Field(alias="end_time")
    cdr_forcing: Optional[cstar_models.Dataset] = Field(default=None, validate_default=False)
    ensemble_id: Optional[int] = Field(default=None, validate_default=False)
    # Internal attributes (computed/loaded)
    blueprint: Optional[cstar_models.RomsMarblBlueprint] = Field(
        default=None,
        init=False,
        validate_default=False,
        validate_assignment=False
    )
    src_data: Optional[source_data.SourceData] = Field(
        default=None,
        init=False,
        validate_default=False
    )
    grid: Optional[rt.Grid] = Field(
        default=None,
        init=False,
        validate_default=False,
        exclude=True
    )
    grid_parent: Optional[rt.Grid] = Field(
        default=None,
        init=False,
        validate_default=False,
        exclude=True
    )
    grid_child: Optional[rt.ChildGrid] = Field(
        default=None,
        init=False,
        validate_default=False,
        exclude=True
    )
    _model_spec: Optional[cson_models.ModelSpec] = PrivateAttr(default=None)
    _datasets: Optional[Dict[str, Union[xr.Dataset, List[xr.Dataset]]]] = PrivateAttr(default=None)
    _stage: Optional[str] = PrivateAttr(default=None)
    _cstar_simulation: Optional[Any] = PrivateAttr(default=None)
    _settings_compile_time: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _settings_run_time: Dict[str, Any] = PrivateAttr(default_factory=dict)
    
    @model_validator(mode="after")
    def _validate_dates(self) -> "CstarSpecBuilder":
        """Validate that start_date precedes end_date."""
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self
    
    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook called automatically after model validation.
        
        This method is called by Pydantic after the instance is validated and
        performs critical initialization:
        
        1. Creates the grid object from `grid_kwargs`
        2. Initializes the blueprint structure (calls `_initialize_blueprint()`)
        
        After this method completes, the blueprint is in the **PRECONFIG** stage
        and has been persisted to disk.
        """
        
        # Create grids
        # I am a child
        if self.grid_kwargs_parent is not None:
            # make parent, but if the parent is also a child, remove the metadata
            grid_kwargs_parent = {k: v for k, v in self.grid_kwargs_parent.items() if k != "metadata"}
            self.grid_parent = rt.Grid(**grid_kwargs_parent)
            self.grid = rt.ChildGrid(parent_grid=self.grid_parent, **self.grid_kwargs)
        else:
            grid_kwargs = {k: v for k, v in self.grid_kwargs.items() if k != "metadata"}
            self.grid = rt.Grid(**grid_kwargs)

        # I am a parent
        if self.grid_kwargs_child is not None:
            self.grid_child = rt.ChildGrid(parent_grid=self.grid, **self.grid_kwargs_child)
        else:
            self.grid_child = None

        # Initialize blueprint with basic structure
        self._initialize_blueprint()

    @property
    def name(self) -> str:
        """
        Return the name of this blueprint as '{model_spec.name}_{grid_name}'.
        
        This property sets blueprint.name when the blueprint is created.
        """
        ensemble_str = f"_{self.ensemble_id:03d}" if self.ensemble_id is not None else ""
        return f"{self._model_spec.name}_{self.grid_name}_{self.n_procs}procs{ensemble_str}"

    @property
    def n_procs(self) -> int:
        """Return the number of processors."""
        return self.partitioning.n_procs_x * self.partitioning.n_procs_y

    @property
    def datestr(self) -> str:
        """Return the date string."""
        return f"{self.start_date.strftime('%Y%m%d')}-{self.end_date.strftime('%Y%m%d')}"

    @property
    def casename(self) -> str:
        """Return the case name."""
        return f"{self.name}_{self.datestr}"

    @property
    def run_output_dir(self) -> Path:
        """Return the output directory path."""
        return config.paths.scratch / self.casename
    
    @property
    def default_runtime_params(self) -> cstar_models.RuntimeParameterSet:
        """
        Get default runtime parameters.
        
        Returns a RuntimeParameterSet with default values based on the builder's
        configuration (start_date, end_date, output_dir).
        """
        return cstar_models.RuntimeParameterSet(
            start_date=self.start_date,
            end_date=self.end_date,
            output_dir=self.run_output_dir,
        )

    @property
    def blueprint_dir(self) -> Path:
        """Return the blueprint directory path."""
        return config.paths.blueprints / config.system_id / self.name 

    @property
    def compile_time_code_dir(self) -> Path:
        """Return the compile-time code output directory path."""
        return config.paths.here / "builds" / self.name / "compile-time"
    
    @property
    def run_time_code_dir(self) -> Path:
        """Return the run-time code output directory path."""
        return config.paths.here / "builds" / self.name / "run-time"

    def persist(self) -> None:
        """
        Persist the current blueprint state to a YAML file.
        
        Saves the blueprint to disk at the file path determined by the current
        stage (PRECONFIG, POSTCONFIG, BUILD, or RUN). Also saves settings to
        a sidecar file.
        
        **File Structure:**
        
        - Blueprint: `B_{name}_{stage}.yml` (or with datestr for RUN stage)
        - Settings: `settings_B_{name}_{stage}.yml` (sidecar file)
        
        The settings are stored separately from the blueprint to avoid
        cluttering the blueprint with configuration details.
        
        **Notes:**
        
        - The directory is created if it doesn't exist
        - Serialization warnings are suppressed (expected for placeholder values)
        - Path objects are converted to strings for YAML compatibility
        
        Raises
        ------
        ValueError
            If blueprint is None, if _stage is None, if stage is "run" but
            runtime_params is not available, or if stage is not a valid
            blueprint stage.
        """
        if self.blueprint is None:
            raise ValueError("Cannot persist: blueprint is not initialized")
        
        if self._stage is None:
            raise ValueError("Cannot persist: _stage is not set")
        
        # Validate stage
        stage = BlueprintStage.validate_stage(self._stage)
        
        # Determine run_params for path_blueprint if stage is "run"
        run_params = None
        if stage == BlueprintStage.RUN:
            if self.blueprint.runtime_params is None:
                raise ValueError("Cannot persist run blueprint: runtime_params is not set")
            run_params = self.blueprint.runtime_params
        
        # Get the file path using path_blueprint
        bp_path = self.path_blueprint(stage=stage, run_params=run_params)
        
        # Ensure directory exists
        bp_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save blueprint to YAML file
        # Use mode='json' to ensure all values are JSON/YAML-serializable (no Python objects)
        # Use exclude_none=True to handle placeholder values gracefully
        # Suppress expected serialization warnings for placeholder values created with model_construct()
        with warnings.catch_warnings():
            # Filter all Pydantic serialization warnings
            # These occur because placeholder values (None) don't match expected types
            warnings.filterwarnings(
                'ignore',
                message='.*Pydantic.*',
                category=UserWarning
            )
            warnings.filterwarnings(
                'ignore',
                message='.*serialization.*',
                category=UserWarning
            )
            blueprint_dict = self.blueprint.model_dump(mode='json', exclude_none=True)
        
        with bp_path.open("w") as f:
            yaml.safe_dump(blueprint_dict, f, default_flow_style=False, sort_keys=False)
        
        # Write settings to sidecar file
        self._persist_settings(bp_path)
    
    def _ensure_empty_directory(self, directory: Union[str, Path]) -> None:
        """
        Ensure a directory exists and is empty.
        
        If the directory exists and is not empty, clears all contents silently.
        If the directory doesn't exist, creates it.
        
        Parameters
        ----------
        directory : Union[str, Path]
            Path to the directory to ensure is empty.
        """
        directory = Path(directory)
        if directory.exists():
            if any(directory.iterdir()):
                # Remove all contents silently (this is expected behavior)
                for item in directory.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            # Directory exists and is now empty (or was already empty)
        else:
            # Create output directory if it doesn't exist
            directory.mkdir(parents=True, exist_ok=True)
    
    def _clear_cstar_externals(self) -> None:
        """
        Clear C-Star externals directories before setup to avoid "dir not empty" errors.
        
        C-Star clones code repositories to externals directories. If these directories
        already exist and are not empty, the clone operation will fail. This method
        clears the externals directories for all codebases in the simulation.
        """
        try:
            from cstar.system.manager import cstar_sysmgr
                        
            # Get the package root (where C-Star is installed)
            package_root = Path(cstar_sysmgr.environment.package_root)
            externals_dir = package_root / "externals"
            
            # Clear externals directory if it exists
            if externals_dir.exists():
                for item in externals_dir.iterdir():
                    if item.is_dir():
                        # Remove the directory and all its contents
                        shutil.rmtree(item)
                    elif item.is_file():
                        # Remove individual files
                        item.unlink()
        except (ImportError, AttributeError) as e:
            # If we can't access C-Star internals, just warn and continue
            warnings.warn(
                f"Could not clear C-Star externals directory: {e}. "
                "If you encounter 'dir not empty' errors, manually clear "
                f"{externals_dir if 'externals_dir' in locals() else 'C-Star externals directory'}.",
                UserWarning,
                stacklevel=2
            )
    
    def _clear_simulation_directory(self) -> None:
        """
        Clear the entire C-Star simulation directory to avoid symlink FileExistsError.
        
        C-Star creates symlinks to input datasets in the run directory. If these
        symlinks already exist, C-Star will fail with FileExistsError. This method
        clears the entire simulation directory (including all subdirectories and files)
        before setup.
        """
        try:
            # Get the simulation directory
            if hasattr(self._cstar_simulation, 'directory') and self._cstar_simulation.directory:
                sim_dir = Path(self._cstar_simulation.directory)
            else:
                # Fallback to blueprint's runtime_params.output_dir
                sim_dir = Path(self.blueprint.runtime_params.output_dir)
            
            # Remove the entire simulation directory if it exists
            if sim_dir.exists():
                shutil.rmtree(sim_dir)
        except (AttributeError, TypeError) as e:
            # If we can't access the simulation or directory, just warn and continue
            warnings.warn(
                f"Could not clear simulation directory: {e}. "
                "If you encounter 'FileExistsError' for symlinks, manually clear "
                f"{sim_dir if 'sim_dir' in locals() else 'simulation directory'}.",
                UserWarning,
                stacklevel=2
            )
    
    def _path_settings_file(self, blueprint_path: Path) -> Path:
        """
        Return the path to the settings sidecar file for a given blueprint path.
        
        The settings file has the same name as the blueprint file, with "settings_" prepended.
        For example: "B_model_postconfig.yml" -> "settings_B_model_postconfig.yml"
        
        Parameters
        ----------
        blueprint_path : Path
            Path to the blueprint file.
        
        Returns
        -------
        Path
            Path to the settings sidecar file.
        """
        # Get the directory and filename
        directory = blueprint_path.parent
        filename = blueprint_path.name
        
        # Prepend "settings_" to the filename
        settings_filename = f"settings_{filename}"
        
        return directory / settings_filename
    
    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """
        Recursively convert Path objects to strings and replace non-serializable
        objects (e.g. xarray.Dataset) with placeholders for YAML serialization.
        
        Parameters
        ----------
        obj : Any
            Object to process (can be dict, list, Path, or other types).
        
        Returns
        -------
        Any
            Object with Path converted to strings and non-serializable types replaced.
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._convert_paths_to_strings(item) for item in obj)
        elif isinstance(obj, xr.Dataset):
            return "<xarray.Dataset>"
        elif isinstance(obj, xr.DataArray):
            return "<xarray.DataArray>"
        else:
            return obj
    
    def _persist_settings(self, blueprint_path: Path) -> None:
        """
        Persist settings dictionaries to a sidecar file.
        
        Writes compile_time and run_time settings to a YAML file with the same
        name as the blueprint file, prepended with "settings_".
        
        Parameters
        ----------
        blueprint_path : Path
            Path to the blueprint file (used to determine settings file path).
        """
        settings_path = self._path_settings_file(blueprint_path)
        
        # Prepare settings dictionary
        settings_dict = {
            "compile_time": self._settings_compile_time,
            "run_time": self._settings_run_time
        }
        
        # Convert all Path objects to strings for YAML serialization
        settings_dict = self._convert_paths_to_strings(settings_dict)
        
        # Ensure directory exists
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write settings to YAML file
        with settings_path.open("w") as f:
            yaml.safe_dump(settings_dict, f, default_flow_style=False, sort_keys=False)
    
    def _load_settings_from_file(self, blueprint_path: Path) -> None:
        """
        Load settings dictionaries from a sidecar file.
        
        Reads compile_time and run_time settings from a YAML file with the same
        name as the blueprint file, prepended with "settings_".
        
        If the settings file doesn't exist, leaves settings dictionaries unchanged.
        
        Parameters
        ----------
        blueprint_path : Path
            Path to the blueprint file (used to determine settings file path).
        """
        settings_path = self._path_settings_file(blueprint_path)
        
        if not settings_path.exists():
            # Settings file doesn't exist, leave settings unchanged
            return
        
        try:
            with settings_path.open("r") as f:
                settings_dict = yaml.safe_load(f)
            
            # Update settings dictionaries if they exist in the file
            if settings_dict:
                if "compile_time" in settings_dict:
                    self._settings_compile_time = settings_dict["compile_time"]
                if "run_time" in settings_dict:
                    self._settings_run_time = settings_dict["run_time"]
        except Exception as e:
            # If loading fails, issue a warning but don't fail
            warnings.warn(
                f"Failed to load settings from {settings_path}: {type(e).__name__}: {e}",
                UserWarning,
                stacklevel=2
            )
    
    def path_blueprint(self, stage: Optional[str] = None, run_params: Optional[cstar_models.RuntimeParameterSet] = None) -> Path:
        """
        Return the path to the blueprint file for a given stage.
        
        Parameters
        ----------
        stage : str, optional
            The blueprint stage. If not provided, uses the blueprint's current state.
        run_params : RuntimeParameterSet, optional
            Runtime parameters for the simulation. Required if stage="run", optional otherwise.
            Used to generate a unique filename for the run blueprint.
        
        Returns
        -------
        Path
            Path to the blueprint YAML file for the specified stage.
        
        Raises
        ------
        AssertionError
            If stage is not one of the valid values.
        ValueError
            If stage="run" and run_params is not provided, or if stage is None and blueprint is None.
        """
        if stage is None:
            if self.blueprint is None:
                raise ValueError("stage must be provided if blueprint is not initialized")
            stage = self.blueprint.state
        BlueprintStage.validate_stage(stage)
        
        if stage == BlueprintStage.RUN:
            if run_params is None:
                raise ValueError("run_params is required when stage='run'")
            # Generate a unique identifier from run_params for the filename
            # Using start_date and end_date to create a unique identifier

            return self.blueprint_dir / f"B_{self.name}_{stage}_{self.datestr}.yml"
        else:
            return self.blueprint_dir / f"B_{self.name}_{stage}.yml"

    @property
    def datasets(self) -> DatasetsDict:
        """
        Return a dictionary of xarray Datasets loaded from blueprint data files.
        
        This property lazily loads xarray Datasets from the NetCDF files referenced
        in the blueprint. The datasets are cached in `_datasets` for efficiency.
        
        **Supported Fields:**
        
        The dictionary includes datasets for all data fields in the blueprint:
        - "grid": Grid dataset
        - "initial_conditions": Initial conditions dataset
        - "forcing.boundary": Boundary forcing datasets
        - "forcing.surface": Surface forcing datasets
        - "forcing.tidal": Tidal forcing datasets
        - "forcing.rivers": River forcing datasets
        - "cdr_forcing": CDR forcing dataset
        
        **Usage:**
        
        Supports both dictionary-style and method-style access:
        - `datasets["grid"]` - dictionary indexing
        - `datasets(key="grid")` - method call with key parameter
        - `datasets()` or `datasets` - returns all datasets
        
        **Data Loading:**
        
        Datasets are loaded lazily from the blueprint's data file locations.
        If a field doesn't exist in the blueprint, it is skipped. Datasets are
        opened in read-only mode (lazy loading).
        
        Returns
        -------
        DatasetsDict
            Dictionary-like object mapping field names to xarray Datasets.
            Returns empty DatasetsDict if blueprint is not initialized.
            
        Warns
        -----
        UserWarning
            If blueprint is not initialized. Returns empty DatasetsDict.
        """
        
        if self.blueprint is None:
            warnings.warn(
                "Blueprint is not initialized. Cannot retrieve datasets.",
                UserWarning,
                stacklevel=2
            )
            return DatasetsDict()
        
        # Populate all datasets from blueprint if not already done
        if self._datasets is None:
            self._datasets = {}
        
        # Dynamically generate list of fields that contain data entries
        # Start with grid and initial_conditions
        data_fields = ["grid", "initial_conditions"]
        
        # Add forcing fields from model_spec.inputs.forcing
        
        if self._model_spec and self._model_spec.inputs and self._model_spec.inputs.forcing:
            # Loop over all fields in the forcing configuration
            for field_name in self._model_spec.inputs.forcing.model_fields.keys():
                data_fields.append(f"forcing.{field_name}")
        
        # Add cdr_forcing (not part of inputs, but a separate blueprint field)
        data_fields.append("cdr_forcing")
        
        # Loop over all data fields and call get_ds for each
        # Suppress Pydantic warnings when accessing datasets
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
            warnings.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*serializer.*", category=UserWarning)
            for field in data_fields:
                # Skip if already populated
                if field in self._datasets:
                    continue
                
                # Call get_ds to get the datasets (it will return None if field doesn't exist)
                ds_list = self.get_ds(field, from_file=False)
                if ds_list is not None and len(ds_list) > 0:
                    # Store single dataset or list
                    self._datasets[field] = ds_list[0] if len(ds_list) == 1 else ds_list
        
        # Return as DatasetsDict to support both dict access and method call
        return DatasetsDict(self._datasets)
    
    def _load_model_spec(self):
        """Load ModelSpec from models.yml."""
        self._model_spec = cson_models.load_models_yaml(
            config.paths.models_yaml,
            self.model_name
        )

    def _initialize_blueprint(self) -> None:
        """
        Initialize blueprint with basic structure and set stage to PRECONFIG.
        
        This method creates the initial blueprint structure with placeholder data.
        It is called automatically during initialization via `model_post_init()`.
        
        **Process:**
        
        1. Loads the model specification from models.yml
        2. Initializes compile-time and run-time settings from defaults
        3. Creates blueprint with:
           - Basic metadata (name, description, dates, partitioning)
           - Code repository specifications from model_spec
           - Placeholder Resource objects for grid, initial_conditions, forcing
        4. Sets `_stage` to PRECONFIG
        5. Persists blueprint to disk
        
        The blueprint at this stage has the correct structure but contains
        placeholder data (None locations). Actual data files are added during
        the POSTCONFIG stage via `generate_inputs()`.
        """

        # Load model spec
        self._load_model_spec()

        # Initialize settings from defaults
        self._init_settings_compile_time()
        self._init_settings_run_time()
                    
        # Create placeholder Resource objects to satisfy validation requirements
        placeholder_resource = cstar_models.Resource.model_construct(
            location=None,
            partitioned=False
        )
        forcing_config = cstar_models.ForcingConfiguration.model_construct(
            boundary=cstar_models.Dataset.model_construct(data=[placeholder_resource]),
            surface=cstar_models.Dataset.model_construct(data=[placeholder_resource]),
        )       
        empty_dataset = cstar_models.Dataset.model_construct(data=[placeholder_resource])
               
        # Use model_construct to bypass validation during initialization
        # The blueprint will be validated later when data is populated
        # Use placeholder datasets to satisfy structure requirements
        self.blueprint = cstar_models.RomsMarblBlueprint.model_construct(
            name=self.name,
            description=self.description,
            valid_start_date=self.start_date,
            valid_end_date=self.end_date,
            partitioning=self.partitioning,
            model_params=None,  # stored in sidecar files
            runtime_params=None,  # stored in sidecar files
            code=self._model_spec.code,
            grid=empty_dataset,
            initial_conditions=empty_dataset,
            forcing=forcing_config,
            cdr_forcing=None,
        )
        self._stage = BlueprintStage.PRECONFIG
        self.persist()
    
    def _compare_dicts_recursive(self, dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> bool:
        """
        Recursively compare two dictionaries, handling nested structures, lists, and datetime normalization.
        
        Parameters
        ----------
        dict1 : Dict[str, Any]
            First dictionary to compare.
        dict2 : Dict[str, Any]
            Second dictionary to compare.
        path : str, optional
            Current path in the dictionary structure (for error messages). Default is "".
        
        Returns
        -------
        bool
            True if dictionaries match, False otherwise.
        """
        # Handle None cases
        if dict1 is None and dict2 is None:
            return True
        if dict1 is None or dict2 is None:
            warnings.warn(f"One dict is None at path '{path}': dict1={dict1}, dict2={dict2}", UserWarning, stacklevel=2)
            return False
        
        # Check for missing or extra keys
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())
        
        if keys1 != keys2:
            missing = keys2 - keys1
            extra = keys1 - keys2
            msg_parts = []
            if missing:
                msg_parts.append(f"missing keys: {missing}")
            if extra:
                msg_parts.append(f"extra keys: {extra}")
            warnings.warn(f"Key mismatch at path '{path}': {', '.join(msg_parts)}", UserWarning, stacklevel=2)
            return False
        
        # Recursively compare all values
        for key in keys1:
            val1 = dict1[key]
            val2 = dict2[key]
            current_path = f"{path}.{key}" if path else key
            
            # Skip 'data' field when path is "grid"
            if path == "grid" and key == "data":
                continue
            
            # Handle nested dictionaries
            if isinstance(val1, dict) and isinstance(val2, dict):
                if not self._compare_dicts_recursive(val1, val2, current_path):
                    return False
                continue
            
            # Handle lists
            if isinstance(val1, list) and isinstance(val2, list):
                if len(val1) != len(val2):
                    warnings.warn(
                        f"different list lengths at path '{current_path}': {len(val1)} vs {len(val2)}",
                        UserWarning, stacklevel=2
                    )
                    return False
                for idx, (item1, item2) in enumerate(zip(val1, val2)):
                    item_path = f"{current_path}[{idx}]"
                    if isinstance(item1, dict) and isinstance(item2, dict):
                        if not self._compare_dicts_recursive(item1, item2, item_path):
                            return False
                    elif item1 != item2:
                        warnings.warn(
                            f"List item mismatch at path '{item_path}': {item1} != {item2}",
                            UserWarning, stacklevel=2
                        )
                        return False
                continue
            
            # Handle datetime normalization (datetime objects vs strings)
            if isinstance(val1, datetime) and isinstance(val2, str):
                if val1.isoformat() == val2 or val1.strftime("%Y-%m-%dT%H:%M:%S") == val2:
                    continue
            elif isinstance(val1, str) and isinstance(val2, datetime):
                if val1 == val2.isoformat() or val1 == val2.strftime("%Y-%m-%dT%H:%M:%S"):
                    continue
            
            # Direct comparison for other types
            if val1 != val2:
                warnings.warn(
                    f"Value mismatch at path '{current_path}': {val1} != {val2}",
                    UserWarning, stacklevel=2
                )
                return False
        
        return True

    def _file_blueprint_data_match(self, partition_files: bool = False) -> bool:
        """
        Check if the POSTCONFIG blueprint from file matches the current blueprint configuration.
        
        Compares specific blueprint fields, grid dataset, and partitioned flags to determine
        if the existing POSTCONFIG blueprint from file can be reused.
        
        Parameters
        ----------
        partition_files : bool, optional
            Expected value for all partitioned flags. Defaults to False.
        
        Returns
        -------
        bool
            True if the POSTCONFIG blueprint from file matches, False otherwise.
        """
        # Load POSTCONFIG blueprint from file (skip loading settings file)
        postconfig_blueprint = self._load_blueprint_file(stage=BlueprintStage.POSTCONFIG, load_settings=False)
        if postconfig_blueprint is None:
            return False
        
        # Convert both blueprints to dictionaries for comparison
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
            warnings.filterwarnings('ignore', message='.*Pydantic.*', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*serialization.*', category=UserWarning)
            current_dict = self.blueprint.model_dump(mode='json')
            file_dict = postconfig_blueprint.model_dump(mode='json')
        
        # Compare specific fields: name, description, valid_start_date, valid_end_date, partitioning, code
        fields_to_compare = ['name', 'description', 'valid_start_date', 'valid_end_date', 'partitioning', 'code']
        
        def compare_dict_field(current_val: Any, file_val: Any, field_name: str, path: str = "") -> Tuple[bool, Optional[str]]:
            """
            Compare two values, handling dictionaries recursively.
            
            Returns (is_match, error_message)
            """
            # Handle None cases
            if current_val is None and file_val is None:
                return True, None
            if current_val is None or file_val is None:
                return False, f"One value is None: current={current_val}, file={file_val}"
            
            # For dictionaries, compare recursively
            if isinstance(current_val, dict) and isinstance(file_val, dict):
                # Check for missing or extra keys
                current_keys = set(current_val.keys())
                file_keys = set(file_val.keys())
                
                if current_keys != file_keys:
                    missing = file_keys - current_keys
                    extra = current_keys - file_keys
                    msg_parts = []
                    if missing:
                        msg_parts.append(f"missing keys: {missing}")
                    if extra:
                        msg_parts.append(f"extra keys: {extra}")
                    return False, f"Key mismatch: {', '.join(msg_parts)}"
                
                # Recursively compare all values
                for key in current_keys:
                    current_item = current_val[key]
                    file_item = file_val[key]
                    item_path = f"{path}.{key}" if path else key
                    is_match, error_msg = compare_dict_field(current_item, file_item, field_name, item_path)
                    if not is_match:
                        return False, f"At {item_path}: {error_msg}"
                
                return True, None
            
            # For lists, compare element by element
            if isinstance(current_val, list) and isinstance(file_val, list):
                if len(current_val) != len(file_val):
                    return False, f"List length mismatch: current={len(current_val)}, file={len(file_val)}"
                for idx, (current_item, file_item) in enumerate(zip(current_val, file_val)):
                    item_path = f"{path}[{idx}]" if path else f"[{idx}]"
                    is_match, error_msg = compare_dict_field(current_item, file_item, field_name, item_path)
                    if not is_match:
                        return False, f"At {item_path}: {error_msg}"
                return True, None
            
            # For other types, direct comparison
            if current_val != file_val:
                return False, f"Value mismatch: current={current_val}, file={file_val}"
            
            return True, None
        
        for field in fields_to_compare:
            current_value = current_dict.get(field)
            file_value = file_dict.get(field)
            
            is_match, error_msg = compare_dict_field(current_value, file_value, field)
            if not is_match:
                warnings.warn(
                    f"Blueprint field '{field}' does not match POSTCONFIG blueprint from file. "
                    f"{error_msg}",
                    UserWarning,
                    stacklevel=2
                )
                return False
        
        # Compare grid datasets (drop xi_coarse dimension from self.grid.ds before comparison)
        # Extract grid dataset from POSTCONFIG blueprint
        # Handle both Pydantic model and dict cases (model_construct may leave nested objects as dicts)
        grid_obj = postconfig_blueprint.grid
        if grid_obj:
            # Get grid data - handle both Pydantic model and dict
            if isinstance(grid_obj, dict):
                grid_data = grid_obj.get("data")
            else:
                grid_data = grid_obj.data if hasattr(grid_obj, 'data') else None
            
            if grid_data:
                # Get the first resource location from the grid data
                grid_resource = grid_data[0] if isinstance(grid_data, list) and len(grid_data) > 0 else None
                # Handle both Pydantic model and dict for resource
                if grid_resource:
                    if isinstance(grid_resource, dict):
                        grid_location = grid_resource.get("location")
                    else:
                        grid_location = grid_resource.location if hasattr(grid_resource, 'location') else None
                    
                    if grid_location:
                        try:
                            # Load grid dataset from blueprint
                            grid_location_str = str(grid_location)
                            file_blueprint_grid_ds = xr.open_dataset(grid_location_str)
                            
                            # Prepare current grid dataset (drop xi_coarse if present)
                            # This is a hack to get around the fact that the grid file has a 
                            # xi_coarse dimension that is not supported by the patition_netcdf function.
                            # https://github.com/CWorthy-ocean/roms-tools/issues/518
                            current_grid_ds = self.grid.ds.copy()
                            if "xi_coarse" in current_grid_ds.dims:
                                current_grid_ds = current_grid_ds.drop_dims("xi_coarse")
                            
                            # Prepare blueprint grid dataset (drop xi_coarse if present)
                            if "xi_coarse" in file_blueprint_grid_ds.dims:
                                file_blueprint_grid_ds = file_blueprint_grid_ds.drop_dims("xi_coarse")
                            
                            # Compare datasets
                            if not current_grid_ds.equals(file_blueprint_grid_ds):
                                warnings.warn(
                                    "Grid dataset does not match POSTCONFIG blueprint grid dataset.",
                                    UserWarning,
                                    stacklevel=2
                                )
                                file_blueprint_grid_ds.close()
                                return False
                            
                            file_blueprint_grid_ds.close()
                        except Exception as e:
                            warnings.warn(
                                f"Failed to compare grid datasets: {e}",
                                UserWarning,
                                stacklevel=2
                            )
                            return False
        
        # Find all instances of "partitioned" in "data" fields and ensure they all match partition_files
        def extract_partitioned_flags(obj: Any) -> List[bool]:
            """Recursively extract all partitioned flags from 'data' fields."""
            partitioned_flags = []
            
            if obj is None:
                return partitioned_flags
            
            # If it's a dict, check for "data" key and recurse
            if isinstance(obj, dict):
                # If this dict has a "data" key, extract partitioned flags from it
                if "data" in obj:
                    data_value = obj["data"]
                    if isinstance(data_value, list):
                        for item in data_value:
                            if isinstance(item, dict) and "partitioned" in item:
                                partitioned_flags.append(item["partitioned"])
                    elif isinstance(data_value, dict) and "partitioned" in data_value:
                        partitioned_flags.append(data_value["partitioned"])
                # Recurse into all values to find nested "data" fields
                for value in obj.values():
                    partitioned_flags.extend(extract_partitioned_flags(value))
                return partitioned_flags
            
            # If it's a list, recurse into items
            if isinstance(obj, list):
                for item in obj:
                    partitioned_flags.extend(extract_partitioned_flags(item))
                return partitioned_flags
            
            return partitioned_flags
        
        # Extract all partitioned flags from the POSTCONFIG blueprint
        blueprint_partitioned_flags = extract_partitioned_flags(file_dict)
        
        # Check if all partitioned flags match partition_files
        if blueprint_partitioned_flags:
            mismatched_flags = [flag for flag in blueprint_partitioned_flags if flag != partition_files]
            if mismatched_flags:
                warnings.warn(
                    f"Partitioned flags in POSTCONFIG blueprint do not match partition_files={partition_files}. "
                    f"Found flags: {set(blueprint_partitioned_flags)}",
                    UserWarning,
                    stacklevel=2
                )
                return False
        
        return True
    
    def _load_blueprint_file(self, stage: Optional[str] = None, load_settings: bool = True) -> Optional[cstar_models.RomsMarblBlueprint]:
        """
        Load blueprint from file for the specified stage.
        
        Parameters
        ----------
        stage : Optional[str], optional
            Blueprint stage to load. If None, uses self._stage.
            If self._stage is also None, defaults to POSTCONFIG.
        load_settings : bool, optional
            If True, load settings from sidecar file. Defaults to True.
        
        Returns
        -------
        Optional[cstar_models.RomsMarblBlueprint]
            Loaded blueprint or None if file doesn't exist or loading fails.
        """
        # Determine which stage to use
        if stage is None:
            stage = self._stage if self._stage is not None else BlueprintStage.PRECONFIG
        
        # Get blueprint file path for this stage
        bp_path = self.path_blueprint(stage=stage, run_params=None)
        
        if not bp_path.exists():
            return None
        
        try:
            # Try to deserialize with full validation first
            # Suppress Pydantic serialization warnings (YAML may contain dicts where models expected)
            with warnings.catch_warnings():
                # Filter all UserWarnings from pydantic module and pydantic.main
                warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
                warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
                # Also filter warnings with "Pydantic" in the message
                warnings.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*serializer.*", category=UserWarning)
                blueprint = deserialize(bp_path, cstar_models.RomsMarblBlueprint)
        except Exception as e:
            # If validation fails (e.g., files don't exist), try lenient loading
            try:
                # Load YAML as dict and use model_construct to bypass validation
                with bp_path.open("r") as f:
                    blueprint_data = yaml.safe_load(f)
                # Use model_construct to bypass validation (files may not exist)
                blueprint = cstar_models.RomsMarblBlueprint.model_construct(**blueprint_data)
            except Exception as e2:
                # If lenient loading also fails, issue a warning and return None
                warnings.warn(
                    f"Failed to load blueprint from {bp_path}: "
                    f"{type(e).__name__}: {e}. "
                    f"Lenient loading also failed: {type(e2).__name__}: {e2}",
                    UserWarning,
                    stacklevel=2
                )
                return None
        
        # Load settings from sidecar file if blueprint was loaded and load_settings is True
        if blueprint is not None and load_settings:
            self._load_settings_from_file(bp_path)
        
        return blueprint
    
    @property
    def blueprint_from_file(self) -> Optional[cstar_models.RomsMarblBlueprint]:
        """
        Load and return blueprint from file based on current stage.
        
        Uses self._stage to determine which blueprint file to load.
        If self._stage is None, defaults to POSTCONFIG stage.
        
        Returns
        -------
        Optional[cstar_models.RomsMarblBlueprint]
            Loaded blueprint or None if file doesn't exist or loading fails.
        """
        # Suppress Pydantic warnings when loading blueprint
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
            warnings.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*serializer.*", category=UserWarning)
            return self._load_blueprint_file()
    
    def get_ds(self, field: str, from_file: bool = True) -> Optional[List[xr.Dataset]]:
        """
        Load xarray Datasets from NetCDF files referenced in a blueprint field.
        
        This method reads the file locations from a specific blueprint field and
        returns lazy-loaded xarray Datasets. Returns a list of datasets even for
        single files to maintain consistency and avoid alignment issues.
        
        **Field Paths:**
        
        Field paths can be simple (e.g., "grid") or nested (e.g., "forcing.surface").
        Nested paths are resolved by traversing the blueprint structure.
        
        **Data Source:**
        
        The `from_file` parameter determines which blueprint to use:
        - `True`: Uses blueprint loaded from disk (default, recommended)
        - `False`: Uses in-memory blueprint (may not reflect persisted state)
        
        Parameters
        ----------
        field : str
            Field path in blueprint. Examples:
            - "grid": Grid dataset
            - "initial_conditions": Initial conditions dataset
            - "forcing.surface": Surface forcing datasets
            - "forcing.tidal": Tidal forcing datasets
            - "cdr_forcing": CDR forcing dataset
        from_file : bool, optional
            If True, loads blueprint from disk first (recommended).
            If False, uses in-memory blueprint.
            Default is True.
        
        Returns
        -------
        Optional[List[xr.Dataset]]
            List of lazy-loaded xarray Datasets, one per file referenced in the field.
            Returns None if the field doesn't exist or has no file locations.
            Always returns a list, even for single files.
        """
        # Select which blueprint to use
        # Suppress Pydantic warnings when accessing blueprint
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
            warnings.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*serializer.*", category=UserWarning)
            if from_file:
                blueprint = self.blueprint_from_file
            else:
                blueprint = self.blueprint
        
        if blueprint is None:
            return None
        
        # Navigate to the field (handle nested fields like "forcing.surface")
        # Handle both model instances and dicts at each level
        # Suppress warnings during navigation as well
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")
            warnings.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*serializer.*", category=UserWarning)
            field_parts = field.split(".")
            data = blueprint
            for part in field_parts:
                # Convert model instances to dicts for easier navigation
                if hasattr(data, 'model_dump'):
                    data = data.model_dump()
                
                if isinstance(data, dict):
                    if part not in data:
                        return None
                    data = data[part]
                elif hasattr(data, part):
                    data = getattr(data, part)
                else:
                    return None
        
        # Convert Dataset to dict if it's a model instance
        if isinstance(data, cstar_models.Dataset):
            data = data.model_dump()
        
        # Extract locations from dict structure        
        if isinstance(data, dict) and "data" in data:
            location_list = [
                item.get("location") 
                for item in data["data"] 
                if isinstance(item, dict) and item.get("location")
            ]
        else:
            return None

        
        if not location_list:
            return None
        
        # Convert locations to strings (handle Path and HttpUrl objects)
        location_strs = []
        for location in location_list:
            if isinstance(location, Path):
                location_strs.append(str(location))
            elif hasattr(location, '__str__'):
                location_strs.append(str(location))
            else:
                location_strs.append(location)
        
        # Return a list of datasets (one per file) instead of combining them
        # This avoids alignment errors when datasets have incompatible dimensions
        # Suppress xarray FutureWarning about timedelta decoding
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, module='xarray')
            return [xr.open_dataset(location, decode_timedelta=False) for location in location_strs]
    
    def ensure_source_data(self, include_streamable: bool = False):
        """
        Ensure source data is prepared and ready for input file generation.
        
        This method prepares all required source datasets (grid, initial conditions,
        forcing data, etc.) using the model specification's dataset requirements.
        The prepared data is stored in `self.src_data` and used by `generate_inputs()`
        to create input files.
        
        **When to Call:**
        
        This method is called automatically by `generate_inputs()` if source data
        hasn't been prepared. It can also be called explicitly to prepare data
        before generating inputs, or to re-prepare data with different options.
        
        Parameters
        ----------
        include_streamable : bool, optional
            If True, include streamable datasets in preparation (datasets that
            can be accessed on-demand rather than pre-downloaded).
            Default is False.
        
        Raises
        ------
        RuntimeError
            If grid is not initialized (should be created during initialization).
        """
        if self.grid is None:
            raise RuntimeError(
                "Grid must be created before preparing source data. "
                "This should have been created during initialization."
            )
        
        if self._model_spec is None:
            self._load_model_spec()
        
        self.src_data = source_data.SourceData(
            datasets=self._model_spec.datasets,
            clobber=False,
            grid=self.grid,
            grid_name=self.grid_name,
            start_time=self.start_date,
            end_time=self.end_date,
        ).prepare_all(include_streamable=include_streamable)
    
    def generate_inputs(
        self,
        clobber: bool = False,
        use_dask: bool = True,
        partition_files: bool = False,
        test: bool = False,
    ) -> cstar_models.RomsMarblBlueprint:
        """
        Generate ROMS input files and advance blueprint to POSTCONFIG stage.
        
        This method generates all required input files (grid, initial conditions,
        forcing, etc.) and updates the blueprint with actual file locations.
        
        **Process:**
        
        1. Checks if existing POSTCONFIG blueprint matches current configuration
        2. If not matching or `clobber=True`:
           - Prepares source data if needed (calls `ensure_source_data()`)
           - Generates all input files via `RomsMarblInputData.generate_all()`
           - Updates settings dictionaries with input-specific values
           - Updates blueprint with actual data file locations
           - Sets `_stage` to POSTCONFIG
           - Persists blueprint to disk
        3. If matching blueprint exists:
           - Loads existing blueprint from disk (skips regeneration)
           
        **Stage Transition:**
        
        - **Input:** Blueprint in PRECONFIG stage (placeholder data)
        - **Output:** Blueprint in POSTCONFIG stage (actual data files)
        
        **Settings:**
        
        Settings dictionaries are updated with values generated during input
        file creation (e.g., grid dimensions, file paths). These updates are
        merged with existing settings to preserve user overrides.

        Parameters
        ----------
        clobber : bool, optional
            If True, overwrite existing input files even if blueprint matches.
            Default is False.
        use_dask : bool, optional
            If True, use dask for parallel computations. Default is True.
        partition_files : bool, optional
            If True, partition input files across tiles. Currently not implemented.
            Default is False.
        test : bool, optional
            If True, truncate the generation loop after 2 iterations for testing.
            Default is False.
            
        Returns
        -------
        cstar_models.RomsMarblBlueprint
            The blueprint updated with all input file locations (POSTCONFIG stage).
            
        Raises
        ------
        RuntimeError
            If blueprint is not initialized, or if settings are not initialized.
        NotImplementedError
            If partition_files is True (functionality not yet implemented).
        """
        # Raise error if partition_files is True (functionality under development)
        if partition_files:
            raise NotImplementedError(
                "File partitioning functionality is not yet fully implemented. "
                "Please set partition_files=False."
            )

        if self.blueprint is None:
            raise RuntimeError("Blueprint must be initialized before generating inputs")

        if not self._file_blueprint_data_match(partition_files=partition_files) or clobber:
            # Ensure settings are initialized before generating inputs
            # If settings are not present or empty, something has gone wrong
            if not hasattr(self, '_settings_compile_time') or not self._settings_compile_time:
                raise RuntimeError(
                    "_settings_compile_time is not initialized or is empty. "
                )
            if not hasattr(self, '_settings_run_time') or not self._settings_run_time:
                raise RuntimeError(
                    "_settings_run_time is not initialized or is empty. "
                )
            
            # Prepare source data if not already done
            if self.src_data is None:
                self.ensure_source_data(include_streamable=False)
            
            # Create inputs instance
            blueprint_elements, settings_compile_time, settings_run_time = input_data.RomsMarblInputData(
                domain_name=self.name,
                start_date=self.start_date,
                end_date=self.end_date,
                model_spec=self._model_spec,
                grid=self.grid,
                grid_child=self.grid_child,
                boundaries=self.open_boundaries,
                source_data=self.src_data,
                blueprint_dir=self.blueprint_dir,
                partitioning=self.partitioning,
                use_dask=use_dask,
            ).generate_all(partition_files=partition_files, clobber=clobber, test=test)
            
            if blueprint_elements is None:
                raise RuntimeError(
                    "Blueprint mismatch detected, but input files exist. "
                    "Set clobber=True to overwrite existing input files."
                )

           # Apply settings from input data generation (deep merge to preserve existing settings)
            self._update_settings_compile_time(settings_compile_time)
            self._update_settings_run_time(settings_run_time)

            if test:
               return

            # Map blueprint_elements to self.blueprint
            # Update the blueprint with the generated input data
            blueprint_dict = self.blueprint.model_dump()
            blueprint_dict["grid"] = blueprint_elements.grid.model_dump() if blueprint_elements.grid else None
            blueprint_dict["initial_conditions"] = blueprint_elements.initial_conditions.model_dump() if blueprint_elements.initial_conditions else None
            blueprint_dict["forcing"] = blueprint_elements.forcing.model_dump() if blueprint_elements.forcing else None
            blueprint_dict["cdr_forcing"] = blueprint_elements.cdr_forcing.model_dump() if blueprint_elements.cdr_forcing else None
                    
             
            # TODO: Uncomment this when settings are implemented in the blueprint
            # At present, we're using a sidecar file to store settings
            # blueprint_dict["model_params"] = settings_compile_time
            # blueprint_dict["runtime_params"] = settings_run_time
            # Set to None since they're stored in sidecar files
            blueprint_dict["model_params"] = None
            blueprint_dict["runtime_params"] = None

            self.blueprint = cstar_models.RomsMarblBlueprint.model_construct(**blueprint_dict)
            self._stage = BlueprintStage.POSTCONFIG
            
            # Persist blueprint to YAML file (skip in test mode)
            self.persist()
        else:            
            # Use existing blueprint from file
            print(f"ℹ️  Using existing blueprint from file: {self.path_blueprint(stage=BlueprintStage.POSTCONFIG).name}")
            self.blueprint = self._load_blueprint_file(stage=BlueprintStage.POSTCONFIG, load_settings=True)
            self._stage = BlueprintStage.POSTCONFIG
        
        return self.blueprint

    def _init_settings_compile_time(self) -> None:
        """
        Initialize compile-time settings dictionary from model defaults.
        
        Loads default compile-time settings from the model specification and
        stores them in `_settings_compile_time`. This dictionary is used as
        the basis for template rendering during `configure_build()`.
        
        Settings are deep-copied from the model spec to avoid modifying the
        original defaults. User overrides can be applied via `_update_settings_compile_time()`
        or by passing `compile_time_settings` to `configure_build()`.
        
        **Called by:** `_initialize_blueprint()` during initialization.
        """
        # Initialize from defaults (deep copy to avoid modifying the original)
        self._settings_compile_time = copy.deepcopy(self._model_spec.settings.compile_time.settings_dict)
        if self.grid_child is not None:
            if "metadata" not in self.grid_kwargs_child:
                raise ValueError(
                    "grid_kwargs_child must contain 'metadata' when grid_child is set (nested domain). "
                    f"Got keys: {sorted(self.grid_kwargs_child.keys())}"
                )
            if "period" not in self.grid_kwargs_child["metadata"]:
                raise ValueError(
                    "grid_kwargs_child['metadata'] must contain 'period' when grid_child is set. "
                    f"Got keys: {sorted(self.grid_kwargs_child['metadata'].keys())}"
                )
            self._settings_compile_time["extract_data"]["extract_period"] = self.grid_kwargs_child["metadata"]["period"]
    
    def _init_settings_run_time(self, dt: Optional[float] = None) -> None:
        """
        Initialize run-time settings dictionary from model defaults.
        
        Loads default run-time settings from the model specification and stores
        them in `_settings_run_time`. This dictionary is used as the basis for
        template rendering during `configure_build()`.
        
        **Dynamic Values:**
        
        Some settings are set dynamically based on instance properties:
        - `title.casename`: Set from `self.casename`
        - `output_root_name.output_root_name`: Set from `self.run_output_dir`
        - `time_stepping`: Calculated based on simulation dates and timestep
        
        Settings are deep-copied from the model spec to avoid modifying the
        original defaults. User overrides can be applied via `_update_settings_run_time()`
        or by passing `run_time_settings` to `configure_build()`.
        
        Parameters
        ----------
        dt : Optional[float], optional
            Timestep in seconds for time_stepping calculation. If None, computed
            from CFL criterion using grid properties. Default is None.
        
        **Called by:** `_initialize_blueprint()` during initialization.
        """
        # Initialize from defaults (deep copy to avoid modifying the original)
        self._settings_run_time = copy.deepcopy(self._model_spec.settings.run_time.settings_dict)
        
        # Set dynamic values that depend on instance properties
        self._settings_run_time["roms.in"]["title"] = dict( 
            casename = self.casename,   
        )
        self._settings_run_time["roms.in"]["output_root_name"] = dict( 
            output_root_name = str(self.run_output_dir / "output" / self.casename),
        )
        
        # Set timestepping defaults (will compute dt from CFL if dt is None)
        self._set_run_time_settings_timestepping_defaults(dt=dt)
    
    
    def _update_settings_compile_time(self, settings_compile_time: Dict[str, Any]) -> None:
        """
        Update compile-time settings with deep merge of nested dictionaries.
        
        Merges provided settings into existing compile-time settings using deep merge.
        For each top-level key, if it exists in both dictionaries, the nested dictionaries
        are merged (not replaced). This preserves existing settings while allowing updates.
        
        **Merging Behavior:**
        
        - If key exists in both: nested dicts are merged (preserves existing values)
        - If key exists only in new settings: raises ValueError (unknown key)
        - Non-dict values: replaced directly
        
        Parameters
        ----------
        settings_compile_time : Dict[str, Any]
            Dictionary of compile-time settings to merge into `_settings_compile_time`.
            Top-level keys must match existing keys in `_settings_compile_time`.
            
        Raises
        ------
        ValueError
            If a top-level key in `settings_compile_time` is not present in
            `_settings_compile_time` (unknown setting key).
        """
        if not settings_compile_time:
            return
        
        for key, value in settings_compile_time.items():
            if key in self._settings_compile_time:
                # Both exist - merge nested dictionaries
                if isinstance(self._settings_compile_time[key], dict) and isinstance(value, dict):
                    # Deep merge: update nested dict without replacing it
                    # Deep copy value to avoid modifying the original dict and ensure we merge correctly
                    value_copy = copy.deepcopy(value)
                    self._settings_compile_time[key].update(value_copy)
                else:
                    # One or both are not dicts - replace the value
                    self._settings_compile_time[key] = copy.deepcopy(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
            else:
                # Unknown key - raise error
                raise ValueError(
                    f"Unknown compile-time setting key: '{key}'. "
                    f"Valid keys are: {sorted(self._settings_compile_time.keys())}"
                )
    
    def _update_settings_run_time(self, settings_run_time: Dict[str, Any]) -> None:
        """
        Update run-time settings with deep merge of nested dictionaries.
        
        Merges provided settings into existing run-time settings using deep merge.
        For each top-level key, if it exists in both dictionaries, the nested dictionaries
        are merged (not replaced). This preserves existing settings while allowing updates.
        
        **Merging Behavior:**
        
        - If key exists in both: nested dicts are merged (preserves existing values)
        - If key exists only in new settings: raises ValueError (unknown key)
        - Non-dict values: replaced directly
        
        Parameters
        ----------
        settings_run_time : Dict[str, Any]
            Dictionary of run-time settings to merge into `_settings_run_time`.
            Top-level keys must match existing keys in `_settings_run_time`.
            
        Raises
        ------
        ValueError
            If a top-level key in `settings_run_time` is not present in
            `_settings_run_time` (unknown setting key).
        """
        if not settings_run_time:
            return
        
        for key, value in settings_run_time.items():
            if key in self._settings_run_time:
                # Both exist - merge nested dictionaries
                if isinstance(self._settings_run_time[key], dict) and isinstance(value, dict):
                    # Deep merge: update nested dict without replacing it
                    # Deep copy value to avoid modifying the original dict and ensure we merge correctly
                    value_copy = copy.deepcopy(value)
                    # Special handling for time_stepping: ensure ntimes is an integer
                    if key == "roms.in" and "time_stepping" in value_copy:
                        if "ntimes" in value_copy["time_stepping"]:
                            value_copy["time_stepping"]["ntimes"] = int(round(value_copy["time_stepping"]["ntimes"]))
                    self._settings_run_time[key].update(value_copy)
                else:
                    # One or both are not dicts - replace the value
                    self._settings_run_time[key] = copy.deepcopy(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
            else:
                # Unknown key - raise error
                raise ValueError(
                    f"Unknown run-time setting key: '{key}'. "
                    f"Valid keys are: {sorted(self._settings_run_time.keys())}"
                )
    
    def _set_run_time_settings_timestepping_defaults(self, dt: Optional[float] = None):
        """
        Update run-time timestepping settings in the settings dictionary.
        
        Sets the `time_stepping` section of `_settings_run_time["roms.in"]` with
        calculated values based on simulation dates and timestep.
        
        **Timestep Calculation:**
        
        If `dt` is not provided, it is computed from CFL criterion:
        1. Computes minimum grid spacing (dx, dy) from size_x/nx and size_y/ny
        2. Estimates fastest gravity wave speed: c = sqrt(g * H_max)
        3. Applies CFL condition: dt = CFL * dx_min / c
        
        **Values Set:**
        
        - `ntimes`: Number of timesteps (calculated from simulation duration / dt)
        - `dt`: Timestep in seconds (provided or computed)
        - `ndtfast`: Number of fast timesteps per baroclinic timestep (default: 60)
        - `ninfo`: Frequency of information output (default: 1)
        
        Parameters
        ----------
        dt : Optional[float]
            Timestep in seconds. If None, computed from CFL criterion using grid
            properties. Default is None.
        
        **Called by:** `_init_settings_run_time()` during initialization.
        """
        
        if dt is None:
            dt = compute_timestep_from_cfl(
                grid_size_x=self.grid.size_x,
                grid_size_y=self.grid.size_y,
                grid_nx=self.grid.nx,
                grid_ny=self.grid.ny,
                grid_ds=self.grid.ds,
            )
        
        ntimes = int(round((self.end_date - self.start_date).days * 24 * 3600 / dt))
        self._settings_run_time["roms.in"]["time_stepping"] = dict(
            ntimes = ntimes,
            dt = dt,
            ndtfast = 60,
            ninfo = 1,
        )
   
    def configure_build(
        self,
        compile_time_settings: Dict[str, Any] = None,
        run_time_settings: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Configure blueprint by rendering templates and advance to BUILD stage.
        
        This method renders Jinja2 templates with current settings to produce
        configuration files needed for model compilation and execution.
        
        **Process:**
        
        1. Validates blueprint is initialized and template configuration exists
        2. Merges user-provided settings overrides with existing settings
        3. Clears compile-time and run-time code output directories
        4. Renders Jinja2 templates:
           - Compile-time templates (e.g., bgc.opt, cppdefs.opt, param.opt)
           - Run-time templates (e.g., roms.in)
        5. Updates blueprint with rendered code locations and file lists
        6. Sets blueprint model_params and runtime_params
        7. Sets `_stage` to BUILD
        8. Persists blueprint to disk
        9. Creates ROMSSimulation instance from blueprint
        
        **Stage Transition:**
        
        - **Input:** Blueprint in POSTCONFIG stage (with input data files)
        - **Output:** Blueprint in BUILD stage (with rendered configuration files)
        
        **Settings:**
        
        Settings are merged using deep merge, preserving existing values while
        allowing user overrides. Run-time timestep (`dt`) can be provided
        explicitly or will be computed from CFL criterion.
        
        **Template Rendering:**
        
        Templates are rendered from the model specification's template locations
        using the current settings dictionaries. The rendered files are written
        to the code output directories and the blueprint is updated with their
        locations.
        
        Parameters
        ----------
        compile_time_settings : Dict[str, Any], optional
            Compile-time settings to override defaults. Merged with existing
            settings using deep merge. Defaults to empty dict.
        run_time_settings : Dict[str, Any], optional
            Run-time settings to override defaults. If a "time_stepping" dict
            with a "dt" key is provided, it will be used for timestep calculation;
            otherwise, the timestep is computed from CFL criterion.
            Defaults to empty dict.
        **kwargs
            Additional keyword arguments (currently unused, reserved for future use).
        
        Returns
        -------
        ROMSSimulation
            The C-Star simulation instance created from the configured blueprint.
        
        Raises
        ------
        RuntimeError
            If blueprint is not initialized (must call `generate_inputs()` first).
        ValueError
            If the model spec does not have required template configuration or
            properties (e.g., n_tracers).
        """
        # Initialize to empty dict if None
        if compile_time_settings is None:
            compile_time_settings = {}
        if run_time_settings is None:
            run_time_settings = {}

        # Validate that blueprint is initialized
        if self.blueprint is None:
            raise RuntimeError("Blueprint must be initialized before configuration. Call generate_inputs() first.")

        # Validate template configuration
        if (self._model_spec.templates is None or
            self._model_spec.templates.compile_time is None or
            self._model_spec.templates.compile_time.filter is None):
            raise ValueError("Model spec must have templates.compile_time.filter with files list")
        if (self._model_spec.templates.run_time is None or
            self._model_spec.templates.run_time.filter is None):
            raise ValueError("Model spec must have templates.run_time.filter with files list")

        # Initialize settings from defaults if not already initialized
        if not hasattr(self, '_settings_compile_time') or self._settings_compile_time is None:
            self._init_settings_compile_time()
        if not hasattr(self, '_settings_run_time') or self._settings_run_time is None:
            self._init_settings_run_time()

        # Update settings with user-provided overrides (deep merge to preserve existing settings)
        self._update_settings_compile_time(compile_time_settings)
        self._update_settings_run_time(run_time_settings)

        # Ensure ntimes is an integer (don't recalculate, just ensure type is correct)
        if "roms.in" in self._settings_run_time and "time_stepping" in self._settings_run_time["roms.in"]:
            if "ntimes" in self._settings_run_time["roms.in"]["time_stepping"]:
                ntimes = self._settings_run_time["roms.in"]["time_stepping"]["ntimes"]
                # Convert to integer if it's a float
                if isinstance(ntimes, float):
                    self._settings_run_time["roms.in"]["time_stepping"]["ntimes"] = int(round(ntimes))

        # Ensure compile-time code directory is empty
        self._ensure_empty_directory(self.compile_time_code_dir)
        self._ensure_empty_directory(self.run_time_code_dir)
            
        # Render templates and get location and file list
        # Get n_tracers from model_spec properties
        if self._model_spec.settings.properties is not None:
            n_tracers = self._model_spec.settings.properties.n_tracers
        else:
            raise ValueError("Model spec must have settings.properties.n_tracers")
        
        compile_time_code = render_roms_settings(
            template_files=self._model_spec.templates.compile_time.filter.files,
            template_dir=self._model_spec.templates.compile_time.location,
            settings_dict=self._settings_compile_time,
            code_output_dir=self.compile_time_code_dir,
            n_tracers=n_tracers,
        )
        run_time_code = render_roms_settings(
            template_files=self._model_spec.templates.run_time.filter.files,
            template_dir=self._model_spec.templates.run_time.location,
            settings_dict=self._settings_run_time,
            code_output_dir=self.run_time_code_dir,
            n_tracers=n_tracers,
        )

        # Suppress Pydantic serialization warnings when using model_dump(mode='json') and model_construct
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            
            blueprint_dict = self.blueprint.model_dump(mode='json')
            code_dict = blueprint_dict["code"]
            # Convert dicts from render_roms_settings to CodeRepository objects
            code_dict["compile_time"] = cstar_models.CodeRepository.model_construct(**compile_time_code)
            code_dict["run_time"] = cstar_models.CodeRepository.model_construct(**run_time_code)
            blueprint_dict["code"] = cstar_models.ROMSCompositeCodeRepository.model_construct(**code_dict)

            blueprint_dict["model_params"] = {
                "time_step": self._settings_run_time["roms.in"]["time_stepping"]["dt"],
            }
            blueprint_dict["runtime_params"] = {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "output_dir": self.run_output_dir,
            }

            self.blueprint = cstar_models.RomsMarblBlueprint.model_construct(**blueprint_dict)
            self._stage = BlueprintStage.BUILD
            self.persist()
        
        # Create and setup C-Star simulation from blueprint
        self._cstar_simulation = ROMSSimulation.from_blueprint(self.path_blueprint(stage=BlueprintStage.BUILD))
        
        return self._cstar_simulation
    
    def build(
        self,
        rebuild: bool = True,
        **kwargs
    ):
        """
        Build the model executable from the configured blueprint.
        
        This method compiles the ROMS model using the configuration files
        generated during `configure_build()`. It must be called after
        `configure_build()` has been executed.
        
        **Process:**
        
        1. Clears C-Star externals directories (to avoid "dir not empty" errors)
        2. Clears simulation directory (to avoid symlink FileExistsError)
        3. Sets up the C-Star simulation (calls `_cstar_simulation.setup()`)
        4. Builds the model executable (calls `_cstar_simulation.build()`)
        
        **Prerequisites:**
        
        - Blueprint must be in BUILD stage (call `configure_build()` first)
        - `_cstar_simulation` must be initialized (done by `configure_build()`)
        
        **Stage:**
        
        The blueprint remains in BUILD stage during and after this method.
        The model executable is built but not yet run.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments (reserved for future use).
        
        Returns
        -------
        ROMSSimulation
            The C-Star simulation instance (same as `_cstar_simulation`).
        
        Raises
        ------
        ValueError
            If `_cstar_simulation` is not initialized (call `configure_build()` first).
        """
       
        
        # ------------------------------------------------------------
        # TODO: These should be uncessary if C-Star can manage the directories itself,
        #       but these seem necessary for now.
        #
        # Clear externals directories before setup to avoid "dir not empty" errors
        self._clear_cstar_externals()
        #
        # Clear simulation directory to avoid symlink FileExistsError
        self._clear_simulation_directory()
        #
        # ------------------------------------------------------------
        self._cstar_simulation.setup()
        self._cstar_simulation.build(rebuild=rebuild)
        return self._cstar_simulation

    def run(
        self, 
        account_key: Optional[str] = None,
        queue_name: Optional[str] = None,
        walltime: str = "06:00:00",         
        run_time_settings: Optional[cstar_models.RuntimeParameterSet] = None, 
        **kwargs
    ):
        """
        Run the model executable and advance blueprint to RUN stage.
        
        This method executes the ROMS model simulation using the built executable.
        It must be called after `build()` has been executed.
        
        **Process:**
        
        1. Validates run-time settings are initialized
        2. Sets `_stage` to RUN
        3. Persists blueprint to disk with runtime parameters
        4. Executes the model simulation (calls `_cstar_simulation.run()`)
        
        **Stage Transition:**
        
        - **Input:** Blueprint in BUILD stage (executable built)
        - **Output:** Blueprint in RUN stage (simulation executed)
        
        **Prerequisites:**
        
        - Blueprint must be in BUILD stage (call `build()` first)
        - `_cstar_simulation` must be initialized (done by `configure_build()`)
        
        Parameters
        ----------
        account_key : str, optional
            The user's account key on the system (required if using a job scheduler).
            If not provided, defaults to the account from machine configuration.
            Default is None.
        queue_name : str, optional
            The name of the scheduler queue to submit the job to.
            If not provided, defaults to the default queue from machine configuration.
            Default is None.
        walltime : str, optional
            Maximum allowed execution time for a scheduler job in HH:MM:SS format.
            Default is "06:00:00".
        run_time_settings : RuntimeParameterSet, optional
            Runtime parameters for the simulation. **Not currently supported.**
            If provided, raises NotImplementedError. The C-Star simulation object
            is instantiated in `configure_build()` and built in `build()`, and
            cannot be updated with new run-time settings. To change run-time
            settings, call `configure_build()` again with the desired settings,
            then `build()` and `run()`.
            Default is None.
        **kwargs
            Additional keyword arguments (reserved for future use).
        
        Raises
        ------
        NotImplementedError
            If `run_time_settings` is provided (not supported).
        RuntimeError
            If run-time settings are not initialized, or if `_cstar_simulation`
            is not initialized (call `build()` first).
        """

        # Ensure runtime settings are initialized before configuring
        # If settings are not present or empty, something has gone wrong
        if not hasattr(self, '_settings_run_time') or not self._settings_run_time:
            raise RuntimeError(
                "_settings_run_time is not initialized or is empty. "
            )
        
        # Update with user-provided settings - NOT SUPPORTED
        # The C-Star simulation object is instantiated in configure_build() and
        # built in build(), and we don't know how to update it with new run-time settings
        if run_time_settings:
            raise NotImplementedError(
                "Changing run_time_settings in run() is not supported. "
                "The C-Star simulation object is instantiated in configure_build() and "
                "built in build(), and cannot be updated with new run-time settings. "
                "To change run-time settings, call configure_build() again with the "
                "desired settings, then build() and run()."
            )
               
        # Update blueprint with runtime_params before running
        #blueprint_dict = self.blueprint.model_dump()
        #blueprint_dict["runtime_params"] = final_runtime_params.model_dump()
        #self.blueprint = cstar_models.RomsMarblBlueprint(**blueprint_dict)
        
        # Persist blueprint to file
        self._stage = BlueprintStage.RUN
        self.persist()
        
        # Get account and queue from machine config if not provided
        machine_config = config.machine_config
        if account_key is None:
            account_key = machine_config.account
        if queue_name is None and machine_config.queues:
            queue_name = machine_config.queues.get("default")
        
        # Build run arguments
        run_kwargs = {
            "account_key": account_key,
            "queue_name": queue_name,
            "walltime": walltime,
            "job_name": self.casename,
        }
        
        
        # Run the simulation
        return self._cstar_simulation.run(**run_kwargs)
    
    def pre_run(self) -> None:
        """
        Execute pre-run operations.
        
        Calls the C-Star simulation's pre_run() method if the simulation is initialized.
        
        Raises
        ------
        ValueError
            If cstar_simulation is not initialized (build() must be called first).
        """
        if self._cstar_simulation is None:
            raise ValueError("cstar_simulation is not initialized. Call build() first.")        
        self._cstar_simulation.pre_run()
    
    def post_run(self) -> None:
        """
        Execute post-run operations.
        
        Calls the C-Star simulation's post_run() method if the simulation is initialized.
        
        Raises
        ------
        ValueError
            If cstar_simulation is not initialized (build() must be called first).
        """
        if self._cstar_simulation is None:
            raise ValueError("cstar_simulation is not initialized. Call build() first.")
        self._cstar_simulation.post_run()

    def set_blueprint_state(self, state: str) -> None:
        """
        Set the state of the blueprint.

        Parameters
        ----------
        state : str
            The new state for the blueprint. Must be a valid BlueprintState value from cstar.orchestration.models.
            Common values include "notset", "draft", "configured", "ready", etc.
            See cstar_models.BlueprintState for the complete list of valid values.
        
        Raises
        ------
        ValueError
            If blueprint is None or if state is not a valid BlueprintState value.
        """
        if self.blueprint is None:
            raise ValueError("Cannot set state: blueprint is not initialized")
        
        # Validate state if BlueprintState is available
        try:
            from cstar.orchestration.models import BlueprintState
            # Try to validate the state value
            if hasattr(BlueprintState, '__members__'):
                valid_states = set(BlueprintState.__members__.values())
                if state not in valid_states:
                    raise ValueError(
                        f"Invalid state '{state}'. Must be one of: {sorted(valid_states)}"
                    )
        except (ImportError, AttributeError):
            # BlueprintState might not be available or might not be an enum
            # In this case, we'll let Pydantic validation handle it
            pass
        
        # Update blueprint with new state
        # Use model_dump with exclude_none and mode='json' to handle placeholder values
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
            warnings.filterwarnings('ignore', message='.*Pydantic.*', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*serialization.*', category=UserWarning)
            blueprint_dict = self.blueprint.model_dump(mode='json', exclude_none=True)
        blueprint_dict["state"] = state
        # Use model_construct to bypass validation for placeholder values
        self.blueprint = cstar_models.RomsMarblBlueprint.model_construct(**blueprint_dict)
    
    def dump(self, file_path: Union[str, Path]) -> None:
        """
        Dump the exact state of CstarSpecBuilder to a YAML file.
        
        This method serializes all serializable fields including:
        - Regular Pydantic model fields (description, model_name, grid_name, etc.)
        - PrivateAttr fields (_model_spec, _stage, _settings_compile_time, _settings_run_time)
        - Complex nested objects (blueprint, src_data)
        
        Fields that cannot be serialized are excluded:
        - grid (excluded from model, but grid_kwargs is saved)
        - _datasets (xarray.Dataset objects - not directly YAML-serializable)
        - _cstar_simulation (runtime object - not serializable)
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the YAML file where the state will be saved.
        
        Notes
        -----
        - xarray.Dataset objects in _datasets are not serialized. They can be
          reconstructed from the blueprint's data entries after loading.
        - The grid object is not serialized, but grid_kwargs is saved, allowing
          the grid to be reconstructed using rt.Grid(**grid_kwargs).
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Start with Pydantic model dump (includes all regular fields)
        state_dict = self.model_dump(mode='json', exclude_none=True)
        
        # Add PrivateAttr fields that can be serialized
        private_attrs = {}
        
        # Serialize _model_spec if it exists (Pydantic model)
        if self._model_spec is not None:
            private_attrs["_model_spec"] = self._model_spec.model_dump(mode='json', exclude_none=True)
        
        # Serialize _stage (simple string)
        if self._stage is not None:
            private_attrs["_stage"] = self._stage
        
        # Serialize settings dictionaries
        if self._settings_compile_time:
            private_attrs["_settings_compile_time"] = self._convert_paths_to_strings(self._settings_compile_time)
        if self._settings_run_time:
            private_attrs["_settings_run_time"] = self._convert_paths_to_strings(self._settings_run_time)
        
        # Serialize src_data if it exists (dataclass)
        if self.src_data is not None:
            # Convert dataclass to dict, but exclude grid object
            src_data_dict = dataclass_asdict(self.src_data)
            # Remove grid object if present (not serializable)
            src_data_dict.pop("grid", None)
            # Convert Path objects to strings
            private_attrs["src_data"] = self._convert_paths_to_strings(src_data_dict)
        
        # Note: _datasets and _cstar_simulation are intentionally excluded
        # as they contain xarray.Dataset objects and runtime objects that
        # cannot be easily serialized to YAML.
        
        # Combine state with private attrs
        state_dict["_private_attrs"] = private_attrs
        
        # Convert all Path objects to strings for YAML serialization
        state_dict = self._convert_paths_to_strings(state_dict)
        
        # Write to YAML file
        with file_path.open("w") as f:
            yaml.safe_dump(state_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "CstarSpecBuilder":
        """
        Load CstarSpecBuilder state from a YAML file.
        
        This method deserializes a previously saved state and reconstructs
        the CstarSpecBuilder instance. After loading:
        - Regular Pydantic fields are restored
        - PrivateAttr fields are restored where possible
        - The grid object is reconstructed from grid_kwargs
        - The blueprint object is restored
        
        Fields that cannot be deserialized remain uninitialized:
        - _datasets: Will be populated when accessed (via datasets property)
        - _cstar_simulation: Will be initialized when build() is called
        
        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the YAML file containing the saved state.
        
        Returns
        -------
        CstarSpecBuilder
            A new CstarSpecBuilder instance with state restored from the file.
        
        Notes
        -----
        - The grid object is automatically reconstructed from grid_kwargs
          in model_post_init().
        - xarray.Dataset objects in _datasets can be loaded later from
          the blueprint's data entries if needed.
        - Model validation and post-init hooks are executed, so the instance
          will be fully initialized and validated.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CstarSpecBuilder state file not found: {file_path}")
        
        # Load YAML file
        with file_path.open("r") as f:
            state_dict = yaml.safe_load(f) or {}
        
        # Extract private attributes
        private_attrs = state_dict.pop("_private_attrs", {})
        
        # Restore _model_spec if present
        model_spec_dict = private_attrs.pop("_model_spec", None)
        
        # Handle blueprint separately - use model_construct to handle None values
        blueprint_dict = state_dict.pop("blueprint", None)
        
        # Create instance using Pydantic model_validate
        # This will trigger model_post_init which creates the grid
        instance = cls.model_validate(state_dict)
        
        # Restore blueprint using model_construct to handle None values
        if blueprint_dict is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
                warnings.filterwarnings('ignore', message='.*Pydantic.*', category=UserWarning)
                warnings.filterwarnings('ignore', message='.*serialization.*', category=UserWarning)
                instance.blueprint = cstar_models.RomsMarblBlueprint.model_construct(**blueprint_dict)
        
        # Restore PrivateAttr fields after instance creation
        if model_spec_dict is not None:
            # Use model_construct to handle None values and missing required fields
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
                warnings.filterwarnings('ignore', message='.*Pydantic.*', category=UserWarning)
                warnings.filterwarnings('ignore', message='.*serialization.*', category=UserWarning)
                instance._model_spec = cson_models.ModelSpec.model_construct(**model_spec_dict)
        
        # Restore _stage
        if "_stage" in private_attrs:
            instance._stage = private_attrs["_stage"]
        
        # Restore settings dictionaries
        if "_settings_compile_time" in private_attrs:
            instance._settings_compile_time = private_attrs["_settings_compile_time"]
        if "_settings_run_time" in private_attrs:
            instance._settings_run_time = private_attrs["_settings_run_time"]
        
        # Restore src_data if present
        if "src_data" in private_attrs:
            src_data_dict = private_attrs["src_data"]
            # Convert string paths back to Path objects where appropriate
            # Note: grid object cannot be restored from src_data_dict
            # as it was excluded during serialization
            instance.src_data = source_data.SourceData(**src_data_dict)
        
        # Note: _datasets and _cstar_simulation are not restored here.
        # They will be initialized when accessed or when build() is called.
        
        return instance


class CstarSpecEngine:
    """
    Engine for executing CstarSpecBuilder workflows from domain configurations.
    
    This class provides a convenient interface for loading domain configurations
    from a YAML file and executing the complete workflow:
    1. ensure_source_data()
    2. generate_inputs()
    3. configure_build()
    4. build()
    5. pre_run()
    
    **Usage:**
    
    ```python
    from cson_forge import CstarSpecEngine
    
    # Load and execute workflow for a domain
    engine = CstarSpecEngine(domains_file="domains.yml")
    builder = engine.generate_domain("test-tiny")
    ```
    
    **Domain Configuration:**
    
    Domain configurations are stored in a YAML file with the following structure:
    
    ```yaml
    grid_name:
      description: str
      model_name: str
      grid_name: str
      start_time: str  # ISO format: "YYYY-MM-DD"
      end_time: str    # ISO format: "YYYY-MM-DD"
      grid_kwargs: dict
      open_boundaries: dict
      partitioning: dict
    ```
    """
    
    def __init__(self, domains_file: Union[str, Path]):
        """
        Initialize CstarSpecEngine.
        
        Parameters
        ----------
        domains_file : Union[str, Path]
            Path to domains YAML file.
        """
        domains_file = Path(domains_file)
        
        self.domains_file = domains_file
        self._domains: Optional[Dict[str, Any]] = None
        self.builder: Optional[Dict[str, CstarSpecBuilder]] = None
        # Load domains on initialization
        self._load_domains()
    
    def _load_domains(self) -> Dict[str, Any]:
        """
        Load domain configurations from YAML file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping grid_name to domain configuration.
        """
        if self._domains is None:
            if not self.domains_file.exists():
                raise FileNotFoundError(
                    f"Domains file not found: {self.domains_file}"
                )
            with self.domains_file.open("r") as f:
                self._domains = yaml.safe_load(f) or {}
        return self._domains
    
    @property
    def domains(self) -> List[str]:
        """
        Return a list of available domain names (grid names).
        
        Returns
        -------
        List[str]
            Sorted list of domain names available in the domains file.
        """
        return sorted(self._load_domains().keys())
    
    def _get_domain_config(self, domain_name: str) -> Dict[str, Any]:
        """
        Get domain configuration for a specific grid name.
        
        Parameters
        ----------
        domain_name : str
            Name of the grid/domain to load.
        
        Returns
        -------
        Dict[str, Any]
            Domain configuration dictionary.
        
        Raises
        ------
        KeyError
            If domain_name is not found in domains file.
        """
        domains = self._load_domains()
        if domain_name not in domains:
            raise KeyError(
                f"Domain '{domain_name}' not found in {self.domains_file}. "
                f"Available domains: {sorted(domains.keys())}"
            )
        return domains[domain_name]
    
    def _create_builder(
        self,
        domain_name: str,
        overrides: Optional[Dict[str, Any]] = None,
        ensemble_id: Optional[int] = None
    ) -> CstarSpecBuilder:
        """
        Create a CstarSpecBuilder instance from domain configuration.
        
        Parameters
        ----------
        domain_name : str
            Name of the grid/domain to load.
        overrides : Optional[Dict[str, Any]], optional
            Dictionary of configuration overrides to apply. These will
            override values from the domain configuration. Default is None.
        ensemble_id : Optional[int], optional
            Ensemble ID to use for this builder. If None, uses default (1).
            Default is None.
        
        Returns
        -------
        CstarSpecBuilder
            Configured CstarSpecBuilder instance.
        """
        config_dict = self._get_domain_config(domain_name).copy()
        
        # Apply ensemble_id if provided
        if ensemble_id is not None:
            config_dict["ensemble_id"] = ensemble_id
        

        if "_parent_grid_name" in config_dict:
            parent_grid_config = self._get_domain_config(config_dict["_parent_grid_name"]).copy()
            config_dict["grid_kwargs_parent"] = parent_grid_config["grid_kwargs"]
            config_dict.pop("_parent_grid_name", None)

        if "_child_grid_name" in config_dict:
            child_grid_config = self._get_domain_config(config_dict["_child_grid_name"]).copy()
            config_dict["grid_kwargs_child"] = child_grid_config["grid_kwargs"]
            config_dict.pop("_child_grid_name", None)

        # Apply overrides if provided
        if overrides:
            config_dict.update(overrides)
        
        # Convert date strings to datetime objects
        if "start_time" in config_dict:
            if isinstance(config_dict["start_time"], str):
                config_dict["start_time"] = datetime.fromisoformat(config_dict["start_time"])
        if "end_time" in config_dict:
            if isinstance(config_dict["end_time"], str):
                config_dict["end_time"] = datetime.fromisoformat(config_dict["end_time"])
        
        # Convert open_boundaries dict to OpenBoundaries model
        if "open_boundaries" in config_dict:
            config_dict["open_boundaries"] = cson_models.OpenBoundaries(**config_dict["open_boundaries"])
        
        # Convert partitioning dict to PartitioningParameterSet
        if "partitioning" in config_dict:
            config_dict["partitioning"] = cstar_models.PartitioningParameterSet(**config_dict["partitioning"])


        # Create and return CstarSpecBuilder
        return CstarSpecBuilder(**config_dict)
    
    def generate_domain(
        self,
        domain_name: str,
        overrides: Optional[Dict[str, Any]] = None,
        clobber_inputs: bool = True,
        partition_files: bool = False,
        test: bool = False,
        compile_time_settings: Optional[Dict[str, Any]] = None,
        run_time_settings: Optional[Dict[str, Any]] = None,
        ensemble_id: Optional[int] = None,
    ) -> CstarSpecBuilder:
        """
        Execute the complete workflow for a domain.
        
        This method executes the full workflow:
        1. ensure_source_data()
        2. generate_inputs()
        3. configure_build()
        4. build()
        5. pre_run()
        
        Parameters
        ----------
        domain_name : str
            Name of the grid/domain to process.
        overrides : Optional[Dict[str, Any]], optional
            Dictionary of configuration overrides to apply. Default is None.
        clobber_inputs : bool, optional
            If True, overwrite existing input files. Default is True.
        partition_files : bool, optional
            If True, partition input files across tiles. Default is False.
        test : bool, optional
            If True, truncate generation loop after 2 iterations. Default is False.
        compile_time_settings : Optional[Dict[str, Any]], optional
            Compile-time settings overrides. Default is None.
        run_time_settings : Optional[Dict[str, Any]], optional
            Run-time settings overrides. Default is None.
        ensemble_id : Optional[int], optional
            Ensemble ID to use for this domain. If None, uses default (1).
            Default is None.
        
        Returns
        -------
        CstarSpecBuilder
            The CstarSpecBuilder instance after completing the workflow.
        """
        # Create builder from domain configuration
        builder = self._create_builder(domain_name, overrides=overrides, ensemble_id=ensemble_id)
        
        # Execute workflow
        builder.ensure_source_data()
        builder.generate_inputs(
            clobber=clobber_inputs,
            partition_files=partition_files,
            test=test
        )
        builder.configure_build(
            compile_time_settings=compile_time_settings or {},
            run_time_settings=run_time_settings or {}
        )
        builder.build()
        builder.pre_run()
        
        return builder
    
    def generate_all(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        clobber_inputs: bool = True,
        partition_files: bool = False,
        test: bool = False,
        compile_time_settings: Optional[Dict[str, Any]] = None,
        run_time_settings: Optional[Dict[str, Any]] = None,
        ensemble_id: Optional[int] = None,
        stop_on_failure: bool = False,
    ) -> Dict[str, CstarSpecBuilder]:
        """
        Execute the complete workflow for all domains (generation only).
        
        This method calls `generate_domain()` for each domain in the domains file.
        The generated builders are stored in `self.builder` attribute.
        To run the simulations, call `run_all()` after this method.
        
        Parameters
        ----------
        overrides : Optional[Dict[str, Any]], optional
            Dictionary of configuration overrides to apply to all domains. Default is None.
        clobber_inputs : bool, optional
            If True, overwrite existing input files. Default is True.
        partition_files : bool, optional
            If True, partition input files across tiles. Default is False.
        test : bool, optional
            If True, truncate generation loop after 2 iterations. Default is False.
        compile_time_settings : Optional[Dict[str, Any]], optional
            Compile-time settings overrides. Default is None.
        run_time_settings : Optional[Dict[str, Any]], optional
            Run-time settings overrides. Default is None.
        ensemble_id : Optional[int], optional
            Ensemble ID to use for all domains. If None, uses default (1).
            Default is None.
        
        Returns
        -------
        Dict[str, CstarSpecBuilder]
            Dictionary mapping grid_name to CstarSpecBuilder instance for each domain.
            Also stored in `self.builder` attribute.
        """
        builders = {}
        domain_list = self.domains
        total_domains = len(domain_list)
        
        print(f"\n{'='*80}")
        print(f"Starting generation for {total_domains} domain(s)")
        print(f"{'='*80}\n")
        
        
        failed_domains = []
        
        for idx, grid_name in enumerate(domain_list, start=1):
            print(f"\n{'-'*80}")
            print(f"[{idx}/{total_domains}] Processing domain: {grid_name}")
            print(f"{'-'*80}")
            
            # TEMPORARY: Remove stale cache between runs
            # TODO: Remove this once C-Star has a proper cache management system.
            # https://cworthy.atlassian.net/browse/CSD-538
            cache_dir = Path.home() / ".cache" / "cstar"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print("⚠ Removed ~/.cache/cstar to avoid stale C-Star cache between runs")
                warnings.warn(
                    "~/.cache/cstar was removed to avoid stale cache state",
                    UserWarning,
                    stacklevel=2,
                )
                        
            
            try:
                builders[grid_name] = self.generate_domain(
                    domain_name=grid_name,
                    overrides=overrides,
                    clobber_inputs=clobber_inputs,
                    partition_files=partition_files,
                    test=test,
                    compile_time_settings=compile_time_settings,
                    run_time_settings=run_time_settings,
                    ensemble_id=ensemble_id,
                )
                print(f"\n✓ Successfully completed domain: {grid_name}")
            except Exception as e:
                print(f"\n✗ Error processing domain {grid_name}: {e}")
                if stop_on_failure:
                    raise e
                warnings.warn(
                    f"Error processing domain {grid_name}: {e}",
                    UserWarning,
                    stacklevel=2
                )
                failed_domains.append((grid_name, str(e)))
        
        print(f"\n{'='*80}")
        print(f"Completed generation for all {total_domains} domain(s)")
        if failed_domains:
            print(f"⚠ {len(failed_domains)} domain(s) failed:")
            for grid_name, error in failed_domains:
                print(f"  - {grid_name}: {error}")
        print(f"{'='*80}\n")
        
        # Store builders as builder attribute (only successful ones)
        self.builder = builders
        
        # Warn if any domains failed
        if failed_domains:
            failed_names = [name for name, _ in failed_domains]
            warnings.warn(
                f"{len(failed_domains)} domain(s) failed during generation: {', '.join(failed_names)}",
                UserWarning,
                stacklevel=2
            )
        
        return builders
    
    def run_all(
        self,
        poll_interval: int = 30,
    ) -> Dict[str, Any]:
        """
        Run all simulations and wait for completion.
        
        This method runs each simulation from the `builder` attribute (set by `generate_all()`)
        and polls execution status until each simulation reaches a terminal state before
        moving to the next one.
        
        Parameters
        ----------
        poll_interval : int, optional
            Number of seconds between status checks. Default is 30.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "builders": Dict[str, CstarSpecBuilder] mapping grid_name to CstarSpecBuilder instances
            - "execution_handlers": Dict[str, ExecutionHandler] mapping grid_name to execution handler instances
        
        Raises
        ------
        RuntimeError
            If `builder` attribute is None (must call `generate_all()` first).
            If any simulation fails or is cancelled.
        """
        if self.builder is None:
            raise RuntimeError(
                "No builders available. Call generate_all() first to create builders."
            )
        
        builders = self.builder
        total_domains = len(builders)
        
        print(f"\n{'='*80}")
        print(f"Starting execution for {total_domains} domain(s)")
        print(f"{'='*80}\n")
        
        execution_results = {}
        failed_simulations = []
        
        for idx, (grid_name, builder) in enumerate(builders.items(), start=1):
            print(f"\n{'-'*80}")
            print(f"[{idx}/{total_domains}] Running simulation: {grid_name}")
            print(f"{'-'*80}")
            
            try:
                # Start the simulation
                execution_handler = builder.run()
                
                # Poll execution status until terminal
                print(f"Monitoring execution status for {grid_name}...")
                while True:
                    status = execution_handler.status
                    print(f"  Status: {status}")
                    
                    if ExecutionStatus.is_terminal(status):
                        if status == ExecutionStatus.COMPLETED:
                            print(f"\n✓ Simulation completed successfully: {grid_name}")
                        elif status == ExecutionStatus.FAILED:
                            print(f"\n✗ Simulation failed: {grid_name}")
                            warnings.warn(
                                f"Simulation {grid_name} failed with status {status}",
                                UserWarning,
                                stacklevel=2
                            )
                            failed_simulations.append((grid_name, status, "failed"))
                        elif status == ExecutionStatus.CANCELLED:
                            print(f"\n⚠ Simulation was cancelled: {grid_name}")
                            warnings.warn(
                                f"Simulation {grid_name} was cancelled",
                                UserWarning,
                                stacklevel=2
                            )
                            failed_simulations.append((grid_name, status, "cancelled"))
                        break
                    
                    # Wait before next status check
                    time.sleep(poll_interval)
                
                execution_results[grid_name] = execution_handler
                
            except Exception as e:
                print(f"\n✗ Error running simulation {grid_name}: {e}")
                warnings.warn(
                    f"Error running simulation {grid_name}: {e}",
                    UserWarning,
                    stacklevel=2
                )
                failed_simulations.append((grid_name, None, f"error: {e}"))
        
        print(f"\n{'='*80}")
        print(f"Completed execution for all {total_domains} domain(s)")
        if failed_simulations:
            print(f"⚠ {len(failed_simulations)} simulation(s) failed or were cancelled:")
            for grid_name, status, reason in failed_simulations:
                print(f"  - {grid_name}: {reason}")
        print(f"{'='*80}\n")
        
        # Warn if any simulations failed
        if failed_simulations:
            failed_names = [name for name, _, _ in failed_simulations]
            warnings.warn(
                f"{len(failed_simulations)} simulation(s) failed or were cancelled: {', '.join(failed_names)}",
                UserWarning,
                stacklevel=2
            )
        
        return execution_results