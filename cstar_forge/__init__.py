"""
cstar_forge: A utility for generating regional oceanographic modeling domains
and spawning reproducible C-Star workflows.
"""

from . import models
from . import config
from . import source_data
from . import settings
from . import catalog
from . import diagnostics
from ._core import CstarSpecBuilder, CstarSpecEngine
from .nb_engine import save_notebook_copy, run_notebook

__all__ = ["source_data", "models", "config", "settings", "catalog", "diagnostics", "CstarSpecBuilder", "CstarSpecEngine", "save_notebook_copy", "run_notebook"]

