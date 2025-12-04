"""
Catalog module for API-driven access to blueprint information.

Provides utilities to discover, load, and query blueprints stored in the
blueprints directory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import pandas as pd

import config
import roms_tools as rt


class BlueprintCatalog:
    """
    API-driven access to blueprint information stored in the blueprints directory.
    
    Provides methods to discover and load blueprint data, including conversion
    to pandas DataFrames for easy querying and instantiation of OcnModel objects.
    """
    
    def __init__(self, blueprints_dir: Optional[Path] = None):
        """
        Initialize the blueprint catalog.
        
        Parameters
        ----------
        blueprints_dir : Path, optional
            Directory containing blueprint YAML files. Defaults to config.paths.blueprints.
        """
        if blueprints_dir is None:
            blueprints_dir = config.paths.blueprints
        self.blueprints_dir = Path(blueprints_dir)
    
    def find_blueprint_files(self) -> List[Path]:
        """
        Recursively find all blueprint_*.yml files in the blueprints directory.
        
        Returns
        -------
        List[Path]
            List of paths to blueprint YAML files.
        """
        blueprint_files = list(self.blueprints_dir.rglob("blueprint_*.yml"))
        # Filter out checkpoint directories
        blueprint_files = [
            f for f in blueprint_files 
            if ".ipynb_checkpoints" not in str(f)
        ]
        return sorted(blueprint_files)
    
    def load_blueprint(self, blueprint_path: Path) -> Dict[str, Any]:
        """
        Load a single blueprint YAML file.
        
        Parameters
        ----------
        blueprint_path : Path
            Path to the blueprint YAML file.
        
        Returns
        -------
        Dict[str, Any]
            Parsed blueprint data.
        
        Raises
        ------
        FileNotFoundError
            If the blueprint file does not exist.
        yaml.YAMLError
            If the YAML file cannot be parsed.
        """
        if not blueprint_path.exists():
            raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
        
        with blueprint_path.open("r") as f:
            return yaml.safe_load(f) or {}
    
    def load_grid_kwargs(self, grid_yaml_path: Path) -> Dict[str, Any]:
        """
        Load grid keyword arguments from a grid YAML file.
        
        Parameters
        ----------
        grid_yaml_path : Path
            Path to the grid YAML file (e.g., _grid.yml).
        
        Returns
        -------
        Dict[str, Any]
            Grid keyword arguments suitable for OcnModel initialization.
        
        Raises
        ------
        FileNotFoundError
            If the grid YAML file does not exist.
        KeyError
            If the Grid section is missing from the YAML file.
        """
        if not grid_yaml_path.exists():
            raise FileNotFoundError(f"Grid YAML file not found: {grid_yaml_path}")

        with grid_yaml_path.open("r") as f:
            docs = list(yaml.safe_load_all(f))            
        
        if len(docs) != 2:
            raise ValueError(f"Expected 2 documents in {grid_yaml_path}, but found {len(docs)}")
        grid_data = docs[1]
        
        if "Grid" not in grid_data:
            raise KeyError(f"Grid section not found in {grid_yaml_path}")
        
        return grid_data["Grid"]
    
    def load(self) -> pd.DataFrame:
        """
        Load all blueprints and return a pandas DataFrame with all data
        necessary to instantiate a cson_forge.OcnModel object.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - model_name: Model name from model_spec
            - grid_name: Grid name
            - grid_kwargs: Dictionary of grid parameters (from grid YAML)
            - boundaries: Dictionary of boundary configuration
            - start_time: Simulation start time
            - end_time: Simulation end time
            - np_eta: Number of processors in eta direction
            - np_xi: Number of processors in xi direction
            - blueprint_path: Path to the blueprint YAML file
            - grid_yaml_path: Path to the grid YAML file
            - input_data_dir: Path to input data directory
        
        Notes
        -----
        Blueprints that cannot be parsed or are missing required fields
        will be skipped with a warning message.
        """
        blueprint_files = self.find_blueprint_files()
        
        records = []
        for bp_file in blueprint_files:
            try:
                blueprint = self.load_blueprint(bp_file)
                
                # Extract required fields
                grid_name = blueprint.get("grid_name")
                model_spec = blueprint.get("model_spec", {})
                model_name = model_spec.get("name")
                
                if not grid_name or not model_name:
                    print(f"⚠️  Skipping {bp_file}: missing grid_name or model_spec.name")
                    continue
                
                # Get grid YAML path and load grid_kwargs
                grid_yaml_path = None
                grid_kwargs = None
                if "inputs" in blueprint and "grid" in blueprint["inputs"]:
                    grid_input = blueprint["inputs"]["grid"]
                    if "yaml_file" in grid_input:
                        grid_yaml_path = Path(grid_input["yaml_file"])
                        try:
                            grid_kwargs = self.load_grid_kwargs(grid_yaml_path)
                        except (FileNotFoundError, KeyError) as e:
                            print(f"⚠️  Skipping {bp_file}: could not load grid kwargs: {e}")
                            continue
                
                if grid_kwargs is None:
                    print(f"⚠️  Skipping {bp_file}: grid_kwargs not available")
                    continue
                
                # Extract other fields
                boundaries = blueprint.get("boundaries", {})
                start_time = blueprint.get("start_time")
                end_time = blueprint.get("end_time")
                np_eta = blueprint.get("np_eta")
                np_xi = blueprint.get("np_xi")
                input_data_dir = blueprint.get("input_data_dir")
                
                records.append({
                    "model_name": model_name,
                    "grid_name": grid_name,
                    "grid_kwargs": grid_kwargs,
                    "boundaries": boundaries,
                    "start_time": start_time,
                    "end_time": end_time,
                    "np_eta": np_eta,
                    "np_xi": np_xi,
                    "blueprint_path": bp_file,
                    "grid_yaml_path": grid_yaml_path,
                    "input_data_dir": input_data_dir,
                })
                
            except Exception as e:
                print(f"⚠️  Could not parse {bp_file}: {e}")
                continue
        
        if not records:
            return pd.DataFrame()
        
        return pd.DataFrame(records)


# Convenience instance
blueprint = BlueprintCatalog()
