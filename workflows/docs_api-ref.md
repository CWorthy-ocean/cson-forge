# API Reference

### Core Classes

#### `ModelSpec`

```python
@dataclass
class ModelSpec:
```

Description of an ocean model configuration (e.g., ROMS/MARBL). 
Supported models are defined in [models.yml](models.yml).

**Parameters:**
- `name` (str): Logical name of the model (e.g., "roms-marbl").
- `opt_base_dir` (str): Relative path (under model-configs) to the base configuration directory.
- `conda_env` (str): Name of the conda environment used to build/run this model.
- `repos` (dict[str, RepoSpec]): Mapping from repo name to its specification.
- `inputs` (dict[str, dict]): Per-input default arguments (from models.yml["<model>"]["inputs"]). These are merged with runtime arguments when constructing ROMS inputs.
- `datasets` (list[str]): SourceData dataset keys required for this model (derived from inputs or explicitly listed in models.yml).
- `settings_input_files` (list[str]): List of input files to copy from the rendered opt directory to the run directory before executing the model (e.g., ["roms.in", "marbl_in"]).
- `master_settings_file` (str): Master settings file to append to the run command (e.g., "roms.in"). This file should be in the run directory when the model executes.

---

#### `ROMSInputs`

```python
@dataclass
class ROMSInputs:
```

Generate and manage ROMS input files for a given grid.

This object is driven by:
- model specification from `models.yml` (model_spec).

The list of inputs to generate (`input_list`) is automatically derived from the keys in `model_spec.inputs`.

The defaults from `model_spec.inputs[<key>]` are merged with runtime arguments (e.g., start_time, end_time, boundaries). Any "source" or "bgc_source" fields in the defaults are resolved through `SourceData`, which injects a "path" entry pointing at the prepared dataset file.

**Parameters:**
- `model_name` (str): Name of the model configuration.
- `grid_name` (str): Name of the grid.
- `grid` (object): Grid object from roms_tools.
- `start_time` (datetime): Start time for the simulation.
- `end_time` (datetime): End time for the simulation.
- `np_eta` (int): Number of processors in the eta direction.
- `np_xi` (int): Number of processors in the xi direction.
- `boundaries` (dict): Boundary configuration dictionary.
- `source_data` (source_data.SourceData): SourceData instance for dataset management.
- `model_spec` (ModelSpec): Model specification from models.yml.
- `use_dask` (bool, optional): Whether to use dask for computations. Default: True.
- `clobber` (bool, optional): Whether to overwrite existing files. Default: False.

**Attributes:**
- `input_list` (List[str]): List of input keys to generate (derived from model_spec.inputs).
- `input_data_dir` (Path): Directory where generated input files are stored.
- `blueprint_dir` (Path): Directory where blueprint YAML files are written.
- `inputs` (Dict[str, InputObj]): Dictionary mapping input keys to InputObj instances.
- `obj` (Dict[str, Any]): Dictionary mapping input keys to roms_tools objects (Grid, InitialConditions, SurfaceForcing, etc.).
- `bp_path` (Path): Path to the main blueprint YAML file.

**Methods:**

##### `generate_all()`

```python
def generate_all(self):
```

Generate all ROMS input files for this grid using the registered steps whose names appear in `input_list`, then partition and write a blueprint.

If any names in `input_list` lack registered handlers, a ValueError is raised.

**Returns:**
- `self` (ROMSInputs): Returns self for method chaining.

---

#### `OcnModel`

```python
@dataclass
class OcnModel:
```

High-level object:
- model metadata from models.yml (ModelSpec),
- source datasets (SourceData),
- ROMS input generation (ROMSInputs),
- model build (via `build()`).

**Parameters:**
- `model_name` (str): Name of the model configuration (e.g., "roms-marbl").
- `grid_name` (str): Name of the grid.
- `grid_kwargs` (Dict[str, Any]): Keyword arguments for creating the Grid object.
- `boundaries` (dict): Boundary configuration dictionary.
- `start_time` (datetime): Start time for the simulation.
- `end_time` (datetime): End time for the simulation.
- `np_eta` (int): Number of processors in the eta direction.
- `np_xi` (int): Number of processors in the xi direction.

**Attributes:**
- `grid` (object): Grid object created from grid_kwargs.
- `spec` (ModelSpec): Model specification loaded from models.yml.
- `src_data` (Optional[source_data.SourceData]): SourceData instance (set after prepare_source_data()).
- `inputs` (Optional[ROMSInputs]): ROMSInputs instance (set after generate_inputs()).
- `executable` (Optional[Path]): Path to the built executable (set after build()).

**Properties:**

##### `name`

```python
@property
def name(self) -> str:
```

Return the full model name as `{model_name}_{grid_name}`.

**Returns:**
- `str`: The full model name.


**Returns:**
- `str`: The mpirun command string with the number of processes (np_xi * np_eta), the executable path, and the master settings file.

**Raises:**
- `RuntimeError`: If the executable has not been built yet.

**Methods:**

##### `prepare_source_data()`

```python
def prepare_source_data(self, clobber: bool = False):
```

Prepare all source datasets required for this model configuration.

**Parameters:**
- `clobber` (bool, optional): Whether to overwrite existing source data files. Default: False.

##### `generate_inputs()`

```python
def generate_inputs(self, clobber: bool = False) -> Dict[str, Any]:
```

Generate ROMS input files for this model/grid.

The list of inputs to generate is automatically derived from the keys in models.yml["<model_name>"]["inputs"].

**Parameters:**
- `clobber` (bool, optional): Passed through to ROMSInputs to allow overwriting existing NetCDF files.

**Returns:**
- `dict`: Dictionary mapping input keys to their corresponding objects (e.g., grid, InitialConditions, SurfaceForcing, etc.).

**Raises:**
- `RuntimeError`: If `prepare_source_data()` has not been called yet.

##### `build()`

```python
def build(self, parameters: Dict[str, Dict[str, Any]], clean: bool = False) -> Path:
```

Build the model executable for this configuration, using the lower-level `build()` helper in this module.

**Parameters:**
- `parameters` (Dict[str, Dict[str, Any]]): Build parameters to pass to the model configuration.
- `clean` (bool, optional): If True, clean the temporary build directory before building. Default: False.

**Returns:**
- `Path`: Path to the built executable.

**Raises:**
- `RuntimeError`: If `generate_inputs()` has not been called yet.

##### `run()`

```python
def run(
    self,
    case: str,
    cluster_type: Optional[str] = None,
    account: Optional[str] = None,
    queue: Optional[str] = None,
    wallclock_time: Optional[str] = None,
) -> None:
```

Run the model executable for this configuration.

**Parameters:**
- `case` (str): Case name for this run (used in job name and output directory).
- `cluster_type` (str, optional): Type of cluster/scheduler to use. Options: "LocalCluster", "SLURMCluster". Defaults based on config.system (mac → LocalCluster, others → SLURMCluster).
- `account` (str, optional): Account for SLURM jobs (required for SLURMCluster).
- `queue` (str, optional): Queue/partition for SLURM jobs (required for SLURMCluster).
- `wallclock_time` (str, optional): Wallclock time limit for SLURM jobs in HH:MM:SS format (required for SLURMCluster).

**Raises:**
- `RuntimeError`: If inputs haven't been generated or executable hasn't been built.

---

### Supporting Data Structures

#### `RepoSpec`

```python
@dataclass
class RepoSpec:
```

Specification for a code repository used in the build.

**Parameters:**
- `name` (str): Short name for the repository (e.g., "roms", "marbl").
- `url` (str): Git URL for the repository.
- `default_dirname` (str): Default directory name under the code root where this repo will be cloned.
- `checkout` (str, optional): Optional tag, branch, or commit to check out after cloning.

---

#### `InputObj`

```python
@dataclass
class InputObj:
```

Structured representation of a single ROMS input product.

**Attributes:**
- `input_type` (str): The type/key of this input (e.g., "initial_conditions", "surface_forcing").
- `paths` (Path | list[Path] | None): Path or list of paths to the generated NetCDF file(s), if applicable.
- `paths_partitioned` (Path | list[Path] | None): Path(s) to the partitioned NetCDF file(s), if applicable.
- `yaml_file` (Path | None): Path to the YAML description written for this input, if any.
