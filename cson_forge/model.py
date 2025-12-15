from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Lazy import of source_data to avoid dependency issues
# source_data is only needed when _dataset_keys_from_inputs is called


@dataclass
class RepoSpec:
    """
    Specification for a code repository used in the build.

    Parameters
    ----------
    name : str
        Short name for the repository (e.g., "roms", "marbl").
    location : str
        Git URL for the repository.
    commit : str, optional
        Optional tag, branch, or commit to check out after cloning.
    """
    name: str
    location: str
    commit: str | None = None


@dataclass
class RunTimeFilter:
    """
    Filter specification for run-time files.
    
    Parameters
    ----------
    files : list[str]
        List of files to copy from the rendered opt directory to the
        run directory before executing the model (e.g., ["roms.in", "marbl_in"]).
        The first file is typically the master settings file.
    """
    files: List[str]


@dataclass
class CompileTimeFilter:
    """
    Filter specification for compile-time files.
    
    Parameters
    ----------
    files : list[str]
        List of files to copy from opt_base_dir to the build opt directory
        during compilation (e.g., ["bgc.opt", "cppdefs.opt", "Makefile"]).
    """
    files: List[str]


@dataclass
class ModelSpec:
    """
    Description of an ocean model configuration (e.g., ROMS/MARBL).

    Parameters
    ----------
    name : str
        Logical name of the model (e.g., "cson_roms-marbl_v0.1").
    opt_base_dir : str
        Relative path (under model-configs) to the base configuration
        directory.
    conda_env : str
        Name of the conda environment used to build/run this model.
    code : dict[str, RepoSpec]
        Mapping from repo name to its specification.
    inputs : dict[str, dict]
        Per-input default arguments (from models.yml["<model>"]["inputs"]).
        These are merged with runtime arguments when constructing ROMS inputs.
    datasets : list[str]
        SourceData dataset keys required for this model (derived from inputs
        or explicitly listed in models.yml).
    run_time : RunTimeFilter
        Run-time file filter specifying files to copy to run directory.
        The first file is the master settings file.
    compile_time : CompileTimeFilter, optional
        Compile-time file filter specifying files to copy during build.
    """
    name: str
    opt_base_dir: str
    conda_env: str
    code: Dict[str, RepoSpec]
    inputs: Dict[str, Dict[str, Any]]
    datasets: List[str]
    run_time: RunTimeFilter
    compile_time: Optional[CompileTimeFilter] = None
    
    
    @property
    def master_settings_file_name(self) -> str:
        """Get master settings file name (first file in run_time.filter.files)."""
        if not self.run_time.files:
            raise ValueError(
                f"Model '{self.name}' must specify at least one file in 'run_time.filter.files'"
            )
        return self.run_time.files[0]
    
    def validate_files_exist(self) -> None:
        """
        Validate that all run_time and compile_time files exist in the opt_base_dir,
        and that no extra files are present beyond those specified.
        
        Raises
        ------
        FileNotFoundError
            If any required files are missing from the opt_base_dir, or if extra
            files are present that are not in the run_time or compile_time lists.
        """
        # Lazy import to avoid circular dependency
        from . import config
        
        opt_base_path = config.paths.model_configs / self.opt_base_dir
        
        if not opt_base_path.exists():
            raise FileNotFoundError(
                f"opt_base_dir does not exist for model '{self.name}': {opt_base_path}"
            )
        
        missing_files = []
        extra_files = []
        
        # Build set of expected files
        expected_files = set(self.run_time.files)
        if self.compile_time is not None:
            expected_files.update(self.compile_time.files)
        
        # Check run_time files
        for filename in self.run_time.files:
            file_path = opt_base_path / filename
            if not file_path.exists():
                missing_files.append(f"run_time: {filename} (expected at {file_path})")
        
        # Check compile_time files (if specified)
        if self.compile_time is not None:
            for filename in self.compile_time.files:
                file_path = opt_base_path / filename
                if not file_path.exists():
                    missing_files.append(f"compile_time: {filename} (expected at {file_path})")
        
        # Check for extra files (files that exist but aren't in the expected list)
        if opt_base_path.is_dir():
            for item in opt_base_path.iterdir():
                # Only check files, not directories
                if item.is_file():
                    filename = item.name
                    if filename not in expected_files:
                        extra_files.append(f"{filename} (unexpected file at {item})")
        
        # Build error message
        error_parts = []
        if missing_files:
            error_parts.append(
                f"Model '{self.name}' is missing required files in {opt_base_path}:\n"
                + "\n".join(f"  - {f}" for f in missing_files)
            )
        if extra_files:
            error_parts.append(
                f"Model '{self.name}' has unexpected files in {opt_base_path}:\n"
                + "\n".join(f"  - {f}" for f in extra_files)
            )
        
        if error_parts:
            raise FileNotFoundError("\n".join(error_parts))
    
    @property
    def module(self):
        """
        Dynamically import and return the module corresponding to the code repositories.
        
        Currently supports:
        - {"roms", "marbl"} -> imports roms_marbl module
        
        Returns
        -------
        module
            The imported module for the model type.
            
        Raises
        ------
        NotImplementedError
            If the set of code keys is not supported.
        """
        code_keys = set(self.code.keys())
        
        if code_keys == {"roms", "marbl"}:
            # Dynamic import of roms_marbl module
            from . import roms_marbl
            return roms_marbl
        else:
            raise NotImplementedError(
                f"Model configuration with code keys {code_keys} is not supported. "
                f"Currently only {{'roms', 'marbl'}} is implemented."
            )


def _extract_source_name(block: Union[str, Dict[str, Any], None]) -> Optional[str]:
    if block is None:
        return None
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return block.get("name")
    return None


def _dataset_keys_from_inputs(inputs: Dict[str, Dict[str, Any]]) -> set[str]:
    """Extract dataset keys from input configurations.
    
    Note: This function requires source_data module to be available.
    If source_data is not available, returns empty set.
    """
    # Lazy import to avoid dependency issues during testing
    try:
        from . import source_data
    except ImportError:
        # If source_data is not available, return empty set
        return set()
    
    dataset_keys: set[str] = set()
    for cfg in inputs.values():
        if not isinstance(cfg, dict):
            continue
        for field_name in ("source", "bgc_source", "topography_source"):
            name = _extract_source_name(cfg.get(field_name))
            if not name:
                continue
            try:
                dataset_key = source_data.map_source_to_dataset_key(name)
                if dataset_key in source_data.DATASET_REGISTRY:
                    dataset_keys.add(dataset_key)
            except (AttributeError, ImportError):
                # If source_data functions aren't available, skip this dataset
                continue
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


def list_models(path: Path) -> List[str]:
    """
    List all model names (master keys) in a models YAML file.

    Parameters
    ----------
    path : Path
        Path to the models.yaml file.

    Returns
    -------
    List[str]
        Sorted list of model names found in the YAML file.
        Returns empty list if file doesn't exist or is empty.

    Examples
    --------
    >>> from pathlib import Path
    >>> from cson_forge.model import list_models
    >>> models = list_models(Path("workflows/models.yml"))
    >>> print(models)
    ['roms-marbl']
    """
    if not path.exists():
        return []
    
    try:
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        
        if not isinstance(data, dict):
            return []
        
        return sorted(data.keys())
    except Exception:
        return []


def load_models_yaml(path: Path, model: str) -> ModelSpec:
    """
    Load repository specifications, model metadata, and default input
    arguments from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the models.yaml file.
    model : str
        Name of the model block to load (e.g., "cson_roms-marbl_v0.1").

    Returns
    -------
    ModelSpec
        Parsed model specification including repository metadata and
        per-input defaults.

    Raises
    ------
    KeyError
        If the requested model is not present in the YAML file.
    ValueError
        If required fields are missing from the model specification.
    """
    with path.open() as f:
        data = yaml.safe_load(f) or {}

    if model not in data:
        raise KeyError(f"Model '{model}' not found in models YAML file: {path}")

    block = data[model]

    # Parse code
    if "code" not in block:
        raise ValueError(f"Model '{model}' must specify 'code' in models.yml")
    
    code: Dict[str, RepoSpec] = {}
    for key, val in block["code"].items():
        code[key] = RepoSpec(
            name=key,
            location=val["location"],
            commit=val.get("commit"),
        )

    inputs = block.get("inputs", {}) or {}
    datasets = _collect_datasets(block, inputs)
    
    # Parse run_time (required)
    if "run_time" not in block:
        raise ValueError(f"Model '{model}' must specify 'run_time' in models.yml")
    if "filter" not in block["run_time"]:
        raise ValueError(f"Model '{model}' must specify 'run_time.filter' in models.yml")
    
    run_time_files = block["run_time"]["filter"].get("files", [])
    if not run_time_files:
        raise ValueError(
            f"Model '{model}' must specify at least one file in 'run_time.filter.files'"
        )
    run_time = RunTimeFilter(files=run_time_files)
    
    # Parse compile_time (optional)
    compile_time = None
    if "compile_time" in block and "filter" in block["compile_time"]:
        compile_time_files = block["compile_time"]["filter"].get("files", [])
        if compile_time_files:
            compile_time = CompileTimeFilter(files=compile_time_files)

    return ModelSpec(
        name=model,
        opt_base_dir=block["opt_base_dir"],
        conda_env=block["conda_env"],
        code=code,
        inputs=inputs,
        datasets=datasets,
        run_time=run_time,
        compile_time=compile_time,
    )

