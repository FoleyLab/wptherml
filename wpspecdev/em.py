from .spectrum_driver import SpectrumDriver
import numpy as np



class TmmDriver(SpectrumDriver):
    def __init__(self, thickness):
        self.thickness = thickness
        self.emissivity = 5 * self.thickness
        print('thickness of the layer is ',self.thickness)

    def compute_spectrum(self):
        return 5*self.thickness

