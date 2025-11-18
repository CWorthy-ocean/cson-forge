from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import shutil
import tempfile
from urllib.request import urlopen

import copernicusmarine
import gdown
import roms_tools as rt
import config


# -----------------------------------------
# Dataset registry (name -> handler + metadata)
# -----------------------------------------


class DatasetHandler:
    """Container for a dataset handler and its required SourceData attributes."""

    def __init__(self, func: Callable[["SourceData"], Path], requires: List[str]):
        self.func = func
        self.requires = requires


DATASET_REGISTRY: Dict[str, DatasetHandler] = {}


def register_dataset(name: str, requires: Optional[List[str]] = None) -> Callable:
    """
    Decorator to register a dataset handler.

    Parameters
    ----------
    name : str
        Dataset name (e.g. "GLORYS_REGIONAL", "UNIFIED_BGC", "SRTM15").
        Stored in upper case.
    requires : list of str, optional
        Names of SourceData attributes that must be non-None for this
        dataset to be prepared (e.g. ["grid", "grid_name", "start_time", "end_time"]).

    Usage
    -----
        @register_dataset("GLORYS_REGIONAL", requires=["grid", "grid_name", "start_time", "end_time"])
        def _prepare_glorys_regional(self): ...
    """
    if requires is None:
        requires = []

    def decorator(func: Callable[["SourceData"], Path]) -> Callable:
        DATASET_REGISTRY[name.upper()] = DatasetHandler(func=func, requires=requires)
        return func

    return decorator


# -----------------------------------------
# Constants (SRTM15 versioning)
# -----------------------------------------

SRTM15_VERSION = "V2.7"
SRTM15_URL = f"https://topex.ucsd.edu/pub/srtm15_plus/SRTM15_{SRTM15_VERSION}.nc"


# -----------------------------------------
# SourceData
# -----------------------------------------


@dataclass
class SourceData:
    """
    Handles creation and caching of source data files
    (GLORYS_REGIONAL, UNIFIED_BGC, SRTM15, etc.) for ROMS preprocessing.

    Parameters
    ----------
    datasets : list of str
        Names of datasets to prepare, e.g. ["GLORYS_REGIONAL", "UNIFIED_BGC", "SRTM15"].
    clobber : bool, optional
        If True, re-download/rebuild datasets even if files exist.
    grid, grid_name, start_time, end_time : optional
        Only required for datasets whose handlers declare them via
        `requires=[...]` in the @register_dataset decorator.
        For example, GLORYS_REGIONAL needs all four.
    """

    datasets: List[str]
    clobber: bool = False

    # Optional attributes — only required if a dataset handler declares them
    grid: Optional[object] = None
    grid_name: Optional[str] = None
    start_time: Optional[object] = None
    end_time: Optional[object] = None

    def __post_init__(self):
        # Normalize dataset names
        self.datasets = [ds.upper() for ds in self.datasets]

        # Validate requested datasets
        known = set(DATASET_REGISTRY.keys())
        unknown = set(self.datasets) - known
        if unknown:
            raise ValueError(
                f"Unknown dataset(s) requested: {', '.join(sorted(unknown))}. "
                f"Known datasets: {', '.join(sorted(known))}"
            )

        # Per-dataset paths (generic) + convenience attrs
        self.paths: Dict[str, Path] = {}
        self.glorys_regional_path: Optional[Path] = None
        self.bgc_forcing_path: Optional[Path] = None
        self.srtm15_path: Optional[Path] = None

    # -----------------------------------------
    # Public API
    # -----------------------------------------

    def prepare_all(self):
        """Prepare all requested source datasets and populate `self.paths`."""
        for name in self.datasets:
            handler = DATASET_REGISTRY[name]

            # Make sure required attributes are provided
            missing = [attr for attr in handler.requires if getattr(self, attr) is None]
            if missing:
                raise ValueError(
                    f"Dataset '{name}' requires attributes {missing}, "
                    "but they were not provided to SourceData()."
                )

            path = handler.func(self)  # call handler with this instance
            self.paths[name] = path  # store generically

        return self

    # -----------------------------------------
    # Internals / helpers
    # -----------------------------------------

    def _construct_glorys_regional_path(self) -> Path:
        fn = (
            f"GLORYS_REGIONAL_{self.grid_name}_"
            f"{self.start_time.strftime('%Y-%m-%d')}-"
            f"{self.end_time.strftime('%Y-%m-%d')}.nc"
        )
        return config.paths.source_data / fn


# ---------------------------
# GLORYS_REGIONAL handler
# ---------------------------


@register_dataset(
    "GLORYS_REGIONAL",
    requires=["grid", "grid_name", "start_time", "end_time"],
)
def _prepare_glorys_regional(self: SourceData) -> Path:
    """
    Download or reuse a regional GLORYS subset for this grid and time range.
    """

    path = self._construct_glorys_regional_path()
    needs_download = self.clobber or (not path.exists())

    if needs_download:
        if path.exists():
            print(f"⚠️  Clobber=True: removing existing GLORYS_REGIONAL file {path.name}")
            path.unlink()

        print(f"⬇️  Downloading GLORYS_REGIONAL → {path}")
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["thetao", "so", "uo", "vo", "zos"],
            **rt.get_glorys_bounds(self.grid),
            start_datetime=self.start_time,
            end_datetime=self.end_time,
            coordinates_selection_method="outside",
            output_filename=path.name,
            output_directory=config.paths.source_data,
        )
    else:
        print(f"✔️  Using existing GLORYS_REGIONAL file: {path}")

    self.glorys_regional_path = path
    return path


# ---------------------------
# UNIFIED BGC handler
# ---------------------------


@register_dataset("UNIFIED_BGC")
def _prepare_unified_bgc_dataset(self: SourceData) -> Path:
    """
    Ensure the UNIFIED_BGC dataset exists locally.
    """
    url_bgc_forcing = (
        "https://drive.google.com/uc?id=1wUNwVeJsd6yM7o-5kCx-vM3wGwlnGSiq"
    )
    path = config.paths.source_data / "BGCdataset.nc"
    needs_download = self.clobber or (not path.exists())

    if needs_download:
        if path.exists():
            print(f"⚠️  Clobber=True: removing existing BGC file {path.name}")
            path.unlink()

        print(f"⬇️  Downloading BGC dataset → {path}")
        gdown.download(url_bgc_forcing, str(path), quiet=False)
    else:
        print(f"✔️  Using existing BGC dataset: {path}")

    self.bgc_forcing_path = path
    return path


# ---------------------------
# SRTM15+ handler
# ---------------------------


@register_dataset("SRTM15")
def _prepare_srtm15(self: SourceData) -> Path:
    """
    Ensure the SRTM15 bathymetry dataset exists locally.

    Download if:
      - the file does not exist, or
      - clobber=True.

    The file is stored under:
        config.paths.source_data / f"SRTM15_{SRTM15_VERSION}.nc"
    """
    path = config.paths.source_data / f"SRTM15_{SRTM15_VERSION}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)

    needs_download = self.clobber or (not path.exists())

    if needs_download:
        if path.exists():
            print(f"⚠️  Clobber=True: removing existing SRTM15 file {path.name}")
            path.unlink()

        print(f"⬇️  Downloading SRTM15+ {SRTM15_VERSION} bathymetry → {path}")

        # Atomic download: write to a temporary file, then move into place
        with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent)) as tmpfile:
            with urlopen(SRTM15_URL) as r:
                shutil.copyfileobj(r, tmpfile)
            tmp_path = Path(tmpfile.name)

        tmp_path.replace(path)
        print(f"✔️  SRTM15+ download complete: {path}")
    else:
        print(f"✔️  Using existing SRTM15+ dataset: {path}")

    self.srtm15_path = path
    return path
