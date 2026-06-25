"""
wptherml
A python package for modeling light-matter interactions!
"""

# Add imports here
from .spectrum_driver import SpectrumDriver
from .em import TmmDriver
from .vec_tmm import VecTmmDriver
from .mie import MieDriver
from .therml import Therml
from .factory import SpectrumFactory
from .materials import Materials, build_refractive_index_array
from .optdriver import OptDriver
from .objectives import Objective, SelectiveMirrorObjective
from .solvers import GradientSolver, Solver, TMMSolver
from .spectra import OpticalSpectrum, OpticalSpectrumGradient
from .structures import MultilayerStructure
from .ensemble import EnsembleResult, ThicknessEnsemble

__all__ = [
    "SpectrumDriver",
    "TmmDriver",
    "VecTmmDriver",
    "MieDriver",
    "Therml",
    "SpectrumFactory",
    "Materials",
    "build_refractive_index_array",
    "OptDriver",
    "MultilayerStructure",
    "Solver",
    "GradientSolver",
    "TMMSolver",
    "OpticalSpectrum",
    "OpticalSpectrumGradient",
    "Objective",
    "SelectiveMirrorObjective",
    "ThicknessEnsemble",
    "EnsembleResult",
]

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
