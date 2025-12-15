from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import math

from . import config
from . import model
from .model import ModelSpec, RepoSpec
import roms_tools as rt
from . import source_data


# =========================================================
# Model execution (run) functions
# =========================================================


class ClusterType:
    """Constants for cluster/scheduler types."""
    LOCAL = "LocalCluster"
    SLURM = "SLURMCluster"
    PBS = "PBSCluster"  # For future extensibility


def _default_cluster_type() -> str:
    """
    Return the default cluster type based on the current system.
    
    Returns
    -------
    str
        "LocalCluster" for MacOS, "SLURMCluster" for other systems.
    """
    if config.system == "MacOS":
        return ClusterType.LOCAL
    else:
        return ClusterType.SLURM


# =========================================================
# High-level OcnModel object
# =========================================================


@dataclass
class OcnModel:
    """
    High-level object:
      - model metadata from models.yml (ModelSpec),
      - source datasets (SourceData),
      - ROMS input generation (inputs),
      - model build (via `build()`).

    Typical usage
    -------------
    grid_kwargs = dict(
        nx=10,
        ny=10,
        size_x=4000,
        size_y=2000,
        center_lon=4.0,
        center_lat=-1.0,
        rot=0,
        N=5,
    )

    ocn = OcnModel(
        model_name="roms-marbl",
        grid_name=grid_name,
        grid_kwargs=grid_kwargs,
        boundaries=boundaries,
        start_time=start_time,
        end_time=end_time,
        np_eta=np_eta,
        np_xi=np_xi,
    )

    ocn.prepare_source_data()
    ocn.generate_inputs()
    ocn.build()
    """

    model_name: str
    grid_name: str
    grid_kwargs: Dict[str, Any]
    boundaries: dict
    start_time: object
    end_time: object
    np_eta: int
    np_xi: int
    grid: object = field(init=False)
    spec: ModelSpec = field(init=False)
    src_data: Optional[source_data.SourceData] = field(init=False, default=None)
    inputs: Optional[Any] = field(init=False, default=None)
    executable: Optional[Path] = field(init=False, default=None)
    
    def __post_init__(self):
        self.grid = rt.Grid(**self.grid_kwargs)
        self.spec = model.load_models_yaml(config.paths.models_yaml, self.model_name)
        self.n_tasks = self.np_xi * self.np_eta
        self.cluster_type = _default_cluster_type()
        
        if self.cluster_type == ClusterType.SLURM:
            if config.machine.pes_per_node is None:
                raise RuntimeError(
                    f"pes_per_node not configured for system '{config.system}'. "
                    f"Please add it to machines.yml"
                )
            # Use ceiling division to ensure we have enough nodes
            self.n_nodes = math.ceil(self.n_tasks / config.machine.pes_per_node)
            self.n_tasks_per_node = config.machine.pes_per_node
        else:
            self.n_nodes = None
            self.n_tasks_per_node = None

    @property
    def input_data_dir(self) -> Path:
        return config.paths.input_data / f"{self.model_name}_{self.grid_name}"

    @property
    def name(self) -> str:
        return f"{self.spec.name}_{self.grid_name}"

    @property
    def _run_command(self) -> str:
        """
        Return the mpirun command to execute the model.
        
        Returns
        -------
        str
            The mpirun command string with the number of processes
            (np_xi * np_eta), the executable path, and the master settings file.
        
        Raises
        ------
        RuntimeError
            If the executable has not been built yet.
        """
        if self.executable is None:
            raise RuntimeError(
                "Executable not built yet. Call OcnModel.build() first."
            )
        master_settings_file = self.spec.master_settings_file_name
        if self.cluster_type == ClusterType.LOCAL:
            return f"mpirun -n {self.n_tasks} {self.executable} {master_settings_file}"
        elif self.cluster_type == ClusterType.SLURM:
            return f"srun -n {self.n_tasks} {self.executable} {master_settings_file}"
        else:
            raise NotImplementedError(
                f"Run command is not implemented for cluster type: {self.cluster_type}"
            )

    def prepare_source_data(self, clobber: bool = False):
        self.src_data = source_data.SourceData(
            datasets=self.spec.datasets,
            clobber=clobber,
            grid=self.grid,
            grid_name=self.grid_name,
            start_time=self.start_time,
            end_time=self.end_time,
        ).prepare_all()
    
    def generate_inputs(
        self,
        clobber: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate ROMS input files for this model/grid.

        The list of inputs to generate is automatically derived from the
        keys in models.yml["<model_name>"]["inputs"].

        Parameters
        ----------
        clobber : bool, optional
            Passed through to inputs to allow overwriting existing
            NetCDF files.
        
        Returns
        -------
        dict
            Dictionary mapping input keys to their corresponding objects
            (e.g., grid, InitialConditions, SurfaceForcing, etc.).
        
        Raises
        ------
        RuntimeError
            If `prepare_source_data()` has not been called yet.
        """
        if self.src_data is None:
            raise RuntimeError(
                "You must call OcnModel.prepare_source_data() "
                "before generating inputs."
            )
        inputs_class = self.spec.module.inputs
        self.inputs = inputs_class(
            model_name=self.model_name,
            grid_name=self.grid_name,
            grid=self.grid,
            start_time=self.start_time,
            end_time=self.end_time,
            np_eta=self.np_eta,
            np_xi=self.np_xi,
            boundaries=self.boundaries,
            source_data=self.src_data,
            model_spec=self.spec,
            clobber=clobber,
        ).generate_all()

        return self.inputs.obj

    def build(
        self, 
        parameters: Dict[str, Dict[str, Any]], 
        clean: bool = False, 
        skip_inputs_check: bool = False
    ) -> Path:
        """
        Build the model executable for this configuration, using the
        module-specific `build()` function.

        Parameters
        ----------
        parameters : dict
            Build-time parameter overrides for the build.
        clean : bool, optional
            If True, clean the existing build directory before building.
        skip_inputs_check : bool, optional
            If True, skip the check for whether inputs have been generated. Default is False.
        """
        if not skip_inputs_check and self.inputs is None:
            raise RuntimeError(
                "You must call OcnModel.generate_inputs() before building the model. "
                "If you wish to skip this check, pass skip_inputs_check=True to build()."
            )

        use_conda = config.system == "MacOS"
        
        exe_path = self.spec.module.build(
            model_spec=self.spec,
            grid_name=self.grid_name,
            input_data_path=self.input_data_dir,
            parameters=parameters,
            clean=clean,
            use_conda=use_conda,
            skip_inputs_check=skip_inputs_check,
        )
        if exe_path is None:
            raise RuntimeError(
                "Build completed but executable was not found. "
                "Check the build logs for errors."
            )
        self.executable = exe_path
        return self.executable

    def run(
        self,
        case: str,
        account: Optional[str] = None,
        queue: Optional[str] = None,
        wallclock_time: Optional[str] = None,
    ) -> None:
        """
        Run the model executable for this configuration.

        Parameters
        ----------
        case : str
            Case name for this run (used in job name and output directory).
        account : str, optional
            Account for SLURM jobs (required for SLURMCluster).
        queue : str, optional
            Queue/partition for SLURM jobs (required for SLURMCluster).
        wallclock_time : str, optional
            Wallclock time limit for SLURM jobs in HH:MM:SS format (required for SLURMCluster).
        
        Raises
        ------
        RuntimeError
            If inputs haven't been generated or executable hasn't been built.
        """
        if self.inputs is None:
            raise RuntimeError(
                "You must call OcnModel.generate_inputs() "
                "before running the model."
            )

        if self.executable is None:
            raise RuntimeError(
                "You must call OcnModel.build() "
                "before running the model."
            )
        
        self.spec.module.run(
            model_spec=self.spec,
            grid_name=self.grid_name,
            case=case,
            executable_path=self.executable,
            run_command=self._run_command,
            inputs=self.inputs.inputs,
            cluster_type=self.cluster_type,
            account=account,
            queue=queue,
            wallclock_time=wallclock_time,
            n_nodes=self.n_nodes,
            n_tasks_per_node=self.n_tasks_per_node,
        )

