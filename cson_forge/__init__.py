"""
cson_forge: A utility for generating regional oceanographic modeling domains
and spawning reproducible C-Star workflows.
"""

from . import model
from . import config

# OcnModel and source_data may not be available if dependencies aren't installed
try:
    from .cson_forge import OcnModel
except ImportError:
    OcnModel = None

try:
    from . import source_data
except ImportError:
    source_data = None

__all__ = ["OcnModel", "source_data", "model", "config"]

