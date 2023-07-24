import numpy as np
from scipy.optimize import minimize


class OptDriver(SpectrumDriver):
    """Compute the absorption, scattering, and extinction spectra of a sphere using Mie theory

    Attributes
    ----------
    TBD



    Returns
    -------
    None

    Examples
    --------
    >>> fill_in_with_actual_example!
    """

    def __init__(self, args):

        self.parse_optimization_input(self, args):

    def parse_optimization_input(self, args):
        if "lower_bound" in args:
            self.lower_bound = args["lower_bound"]
        else:
            self.lower_bound = 1 

        if "upper_bound" in args:
            self.upper_bound = args["upper_bound"]
        else:
            self.upper_bound = 1000


    def super_func(x0):
        pass
        #opt_inst.t thickness_array[1:self.number_of_layers-1] = x0 * 1e-9
        

    