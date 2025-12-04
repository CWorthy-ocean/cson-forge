# Overview

C-SON Forge streamlines the creation of ROMS-MARBL domains by automating the generation of all required input files using [ROMS Tools](https://roms-tools.readthedocs.io/en/latest/index.html). 
The files include grids, initial conditions, boundary and surface forcing, rivers, and tidal forcing—from a variety of observational and reanalysis datasets. 


The tool produces **blueprint** YAML files that capture the complete configuration and file paths for each domain, enabling reproducible model setups that can be integrated into C-Star workflows.

## Key Features

- **Automated Input Generation**: Generate all ROMS input files (grid, initial conditions, forcing, boundaries, rivers, tidal forcing) from source datasets
- **Multi-Dataset Support**: Integrates with multiple data sources including:
  - GLORYS (ocean reanalysis)
  - ERA5 (atmospheric reanalysis)
  - UNIFIED_BGC (biogeochemical climatology)
  - SRTM15 (bathymetry)
  - DAI (river discharge)
  - TPXO (tidal forcing)
- **Blueprint System**: Automatically generates YAML blueprints that document:
  - Complete model specification (repositories, conda environments, input configurations)
  - All generated input file paths (both full and partitioned)
  - Domain configuration (grid name, time ranges, boundaries, processor layout)
  - Source data provenance
- **Reproducible Workflows**: Blueprints serve as complete descriptors that enable:
  - Exact reproduction of model configurations
  - Integration with C-Star workflow management
  - Version control and sharing of domain setups
- **Model Building**: Automated compilation of ROMS and MARBL executables with support for multiple compilers and MPI configurations
- **Execution Management**: Run models locally or submit to HPC clusters (SLURM, PBS) with automatic log file management

## Project Structure

```
cson-forge/
├── workflows/
│   ├── cson_forge.py           # Core orchestration (ModelSpec, ROMSInputs, OcnModel)
│   ├── source_data.py          # Dataset download and preparation
│   ├── config.py               # Path management and system detection
│   ├── models.yml              # Model configuration specifications
│   ├── catalog.py              # Blueprint catalog
│   ├── blueprints/             # Generated blueprint YAML files
│   │   └── {model}_{grid}/
│   │       ├── blueprint_{model}-{grid}.yml
│   │       └── _{input_type}.yml
│   ├── model-configs/          # Model source code & input file templates
│   └── builds/                 # Model compilation directories
└── README.md
```

## Blueprint System

Blueprints are YAML files that capture the complete state of a ROMS domain configuration. Each blueprint includes:

- **Domain Metadata**: Grid name, time ranges, processor layout, boundary configuration
- **Model Specification**: Code repositories, conda environments, input defaults, required datasets
- **Input File Inventory**: Paths to all generated NetCDF files (both full and partitioned for parallel execution)
- **Source Data Provenance**: Dataset sources and configuration used for each input type

These blueprints enable:
1. **Reproducibility**: Exact recreation of model setups from a single YAML file
2. **C-Star Integration**: Blueprints can be consumed by C-Star workflows to orchestrate model runs
3. **Documentation**: Self-documenting domain configurations with full provenance
4. **Version Control**: Track domain evolution and share configurations across teams

