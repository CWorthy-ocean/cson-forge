# Model specifications

Model specifications are defined in `workflows/models.yml`. Each model includes:

- Repository configurations (ROMS, MARBL)
- Input dataset defaults
- Settings files
- Required datasets

## `models.yml` Schema

The `models.yml` file uses YAML format to specify settings for each supported ocean model. The schema typically looks like:

```yaml
<model_name>:
  opt_base_dir: workflows/model-configs/<opt-dir> # model configuration source code, cppdefs, input files, etc.
  conda_env: <Conda environment for model build>
  repo:
    roms: <ROMS repository URL or path>
    marbl: <MARBL repository URL or path>
  settings_input_files:
    - <settings_file1.in>
    - <settings_file2.in>
  inputs:
    grid: 
      topography_source: SRTM15
    initial_conditions:
      source:
        name: GLORYS
    surface_forcing:
      source:
        name: ERA5
      correct_radiation: true
      ...            # other inputs
```

**Field Descriptions:**

- `repo`  
  Specifies the locations (URLs or local paths) of the ROMS and MARBL source code repositories.

- `opt_base_dir`  
  Source code files that are nominally "grid-invariant" defining a configuration of the system (e.g., with `cppdefs`, `*.opt`, `*.h`, etc.). Templating is applied to handle grid and simulation specific parameters.

- `settings_input_files`  
  Lists configuration files that the model reads at runtime.

- `inputs`  
  Describes default values for forcing datasets.


You can add new models by creating a new top-level key in the YAML file with the same schema as above.

For further reference, see:
- Model input requirements in [`workflows/models.yml`](models.yml)

