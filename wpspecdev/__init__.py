"""
wpspecdev
A python package for modeling light-matter interactions!
"""

# Add imports here
from .spectrum_driver import SpectrumDriver
from .em import TmmDriver
from .mie import MieDriver
from .factory import SpectrumFactory
from .materials import Materials
# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
