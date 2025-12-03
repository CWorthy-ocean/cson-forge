# Overview

This section contains recipes for creating CSON domains.

The typical workflow for creating a new domain involves:

1. **Define the grid** - Specify grid parameters (size, resolution, location)
2. **Prepare source data** - Download and prepare required datasets
3. **Generate inputs** - Create ROMS input files (grid, initial conditions, forcing, etc.)
4. **Build the model** - Compile the ROMS executable
5. **Run the model** - Execute the model simulation

:::{important} Stamp out _Blueprints_
This workflow generates a `blueprint` a set of yaml files documenting the domain creation and providing a basis for reproducing the workflow. Blueprints are stored in `blueprints/{model_name}_{grid_name}`.
:::


## Available Recipes
- [California Current System](grid_california-current-system.ipynb) - High-resolution domain off the US West Coast
- [Hvalfjörður, Iceland](grid_hvalfjörður-iceland.ipynb) - Fjord grid in western Iceland
- [WIO Toy Domain](https://cworthy-ocean.github.io/cson-forge/grid-wio-toy/) - Western Indian Ocean toy domain
- [Gulf of Guinea Toy Domain](https://cworthy-ocean.github.io/cson-forge/grid-gulf-guinea-toy/) - Gulf of Guinea toy domain

