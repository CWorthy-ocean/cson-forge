# Source data
The `source_data.py` module manages the acquisition, preparation, and caching of model input datasets required for ROMS/MARBL domain generation and simulation. 
These datasets are documented in ROMS Tools [here](https://roms-tools.readthedocs.io/en/latest/datasets.html).
It provides a registry-driven system for handling diverse data sources, allowing for flexible workflows whether datasets are streamed or locally cached.

## Key Concepts

### Dataset Handlers

Dataset-specific logic is registered via a decorator system:

- `@register_dataset(name, requires=[...])` registers a dataset preparation function with a logical identifier (`name`). 
    - Example: `@register_dataset("GLORYS_REGIONAL", requires=["grid", "start_time", "end_time"])`

The registry (`DATASET_REGISTRY`) associates logical names with their handler functions and dependency requirements.

### Main Class: `SourceData`

This dataclass organizes the process of preparing and managing source datasets.

**Constructor:**

```python
SourceData(
    datasets: List[str],
    clobber: bool = False,
    grid: Optional[GridType] = None,
    grid_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
)
```

- `datasets`: List of required source dataset names (e.g., `["GLORYS_REGIONAL", "UNIFIED_BGC", "SRTM15"]`).
- `clobber`: If `True`, will rebuild or re-download data even if local copies exist.
- Additional parameters (`grid`, `start_time`, etc.) are supplied as needed for specific dataset handlers.

### Logical Source Name Mapping

User-friendly names (as appear in `models.yml` or config) are mapped to dataset registry keys via `map_source_to_dataset_key`. For example:

| Alias    | Dataset Key         |
|----------|---------------------|
| GLORYS   | GLORYS_REGIONAL or GLORYS_GLOBAL (platform-dependent) |
| UNIFIED  | UNIFIED_BGC         |
| SRTM15   | SRTM15_V2.7         |
| ERA5     | ERA5                |
| TPXO     | TPXO (user-provided tides) |
| DAI      | DAI (river data, if available) |

### Streamable Sources

ROMS Tools supports streaming for some datasets, such as ERA5 (i.e., no local download/caching is needed).

### Example: Using SourceData

:::{important} Register for dataset access
GLORYS data is provided via the Copernicus Marine Service. 
Learn how to register for access [here](https://help.marine.copernicus.eu/en/articles/4220332-how-to-sign-up-for-copernicus-marine-service).
That process should result in a .copernicusmarine or .copernicusmarine-credentials file in your home directory
:::

To pre-stage model input data, you can use a call to `SourceData` like this.

```python
from datetime import datetime
from workflows.source_data import SourceData

start_time = datetime(2012, 1, 1)
end_time = datetime(2012, 1, 2)

src = SourceData(
    datasets=["GLORYS", "SRTM15", "UNIFIED_BGC"],
    clobber=True,
    grid=domain_grid,
    start_time=start_time,
    end_time=end_time,
)

# Prepares and caches the datasets needed
src.prepare()
# Paths to prepared files are available as: src.paths[<DATASET_KEY>]
```

## Dataset Preparation Logic

- **SRTM15**: Downloads topography from Scripps (version controlled, e.g. `SRTM15_V2.7`).
- **GLORYS**: Global or regional ocean initial conditions; subset and time-extract logic depends on system type.
- **UNIFIED**: [Unified biogeochemistry forcing & initial conditions from ROMS Tools](https://roms-tools.readthedocs.io/en/latest/initial_conditions.html#Adding-Biogeochemical-(BGC)-Initial-Conditions).
- **ERA5**: Atmospheric surface forcing (typically streamed).
- **TPXO**: Tidal harmonics file must be provided by user.

Each preparation routine ensures datasets are up to date and correctly subsetted for the target domain/grid.

---

For further reference, see:
- [`workflows/source_data.py`](https://github.com/smaticka/cson-forge/blob/main/workflows/source_data.py)
- Model input requirements in [`workflows/models.yml`](models.yml)
- Model configuration documentation in [docs_models.md](docs_models.md)

