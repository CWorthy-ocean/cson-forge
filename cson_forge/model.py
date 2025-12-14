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

