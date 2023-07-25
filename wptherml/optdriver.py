import numpy as np
from scipy.optimize import minimize
from .em import TmmDriver




class OptDriver(TmmDriver):
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
        
        # ordinary tmm driver input
        args = {k.lower(): v for k, v in args.items()}
        self.parse_input(args)
        self.parse_optimization_input(args)
        self.set_refractive_index_array()
        # compute reflectivity spectrum
        self.compute_spectrum()
        self.optimize()
        print("We started optimizing")

    def parse_optimization_input(self, args):
        """ Parse additional options related to the optimization, including:
            - the objective function to be optimized 
            - minimization or maximization of the objective
            - the optimization method to be used
            - lower- and upper-bounds on the layer thicknesses
            
        """
        # which objective function do we want to optimize
        if "objective_function" in args:
            self.objective_function = args["objective_function"]
        # default to selective mirror    
        else:
            self.objective_function = "selective_mirror"

        # do we want to minimze the objective?
        if "minimization" in args:
            self.mimimization = args["minimization"]
        # usually we want to mazimize, so default is false
        else: 
            self.minimization = False
        
        # set bounds on thickness of each layer
        if "lower_bound" in args:
            self.lower_bound = args["lower_bound"]
        else:
            self.lower_bound = 1 

        if "upper_bound" in args:
            self.upper_bound = args["upper_bound"]
        else:
            self.upper_bound = 1000

    def optimize(self):
        """"
        Method to wrap the optimizer
        """
        # initialize x array
        x_start = self.thickness_array[1:self.number_of_layers-1] * 1e9

        # set bounds
        bfgs_xmin = self.lower_bound * np.ones(self.number_of_layers-2)
        bfgs_xmax = self.upper_bound * np.ones(self.number_of_layers-2)
        # rewrite the bounds in the way required by L-BFGS-B
        bfgs_bounds = [(low, high) for low, high in zip(bfgs_xmin, bfgs_xmax)]

        fom_start, grad_start = self.super_func(x_start)
        print(F' Initial FOM is {fom_start}')
        print(F' Initial Gradient is {grad_start}')

        ret = minimize(self.super_func, x_start, method="L-BFGS-B", jac=True, bounds=bfgs_bounds)
        print(ret.x)


    def super_func(self, x0):
        """"
        Method to update the thickness array, the relevant spectra, objective function, and (if supported)
        the gradient
        """
        print(" Entering SuperFunc ")
        self.thickness_array[1:self.number_of_layers-1] = x0 * 1e-9
        self.compute_spectrum()
        self.compute_selective_mirror_fom()
        # compute_selective_mirror_fom_gradient calls compute_spectrum_gradient()
        self.compute_selective_mirror_fom_gradient()

        fom = self.reflection_efficiency
        grad = self.reflection_efficiency_gradient

        if self.minimization:
            fom *= 1
            grad *= 1e-9
        else:
            fom *= -1
            grad *= -1e-9

        print(F' Current FOM is {fom}')
        print(F' Current Gradient is {grad}')
        return fom, grad



        





        

    