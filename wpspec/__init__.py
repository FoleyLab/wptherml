"""
wpspec
A python package for modeling light-matter interactions!
"""

# Add imports here
from .wpspec import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
