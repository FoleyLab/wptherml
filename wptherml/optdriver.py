import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time
from .em import TmmDriver
#import torch
#import torch.nn as nn
#import torch.optim as optim
#import tqdm
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
import copy
#from pyqubo import Binary
#import neal
#import pandas as pd


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
        # store args
        self.args = args

        self.start_time = time.time()

        # ordinary tmm driver input
        args = {k.lower(): v for k, v in args.items()}
        self.parse_input(args)
        self.parse_optimization_input(args)
        self.set_refractive_index_array()
        # compute reflectivity spectrum
        self.compute_spectrum()
        # print("We started optimizing")
        # self.optimize()

    def parse_optimization_input(self, args):
        """Parse additional options related to the optimization, including:
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

        if "combinatorial_optimization" in args:
            self.combinatorial_optimization = True
            self.qubo_thickness_optimization = False
            self.optimization_dictionary = args["optimization_dictionary"]
        if "qubo_thickness_optimization" in args:
            self.qubo_thickness_optimization = True
            self.combinatorial_optimization = False
            self.optimization_dictionary = args["optimization_dictionary"]

        if "random_perturbation_scale" in args:
            self.random_perturbation_scale = args["random_perturbation_scale"]

        else:
            # default will be 10%, so we can randomly perturb a given layer by +/- 10% 
            # between basin hopping cycles
            self.random_perturbation_scale = 0.1

    def optimize_qubo(self, fom_function):
        if self.combinatorial_optimization == True:
            self.optimizer = self._qubo_combinatorial_structure_optimization(
                self, self.optimization_dictionary, fom_function
            )
            self.optimizer.learning_loop(num_to_train=25, num_iterations=10000)

        if self.qubo_thickness_optimization == True:
            self.optimizer = self._qubo_thickness_and_alloy_optimization(
                self,
                optimization_dict=self.optimization_dictionary,
                fom_func=fom_function,
            )
            self.optimizer.learning_loop(
                num_to_train=100,
                n_epochs=250,
                l2_lambda=0.001,
                num_iterations=400,
                reduction_factor=2,
                l1_lambda=0.000001,
                K=25,
                LR=0.01,
            )

    def optimize_bfgs(self):
        """ "
        Method to wrap the l-bfgs-b optimizer
        """
        # initialize x array
        x_start = self.thickness_array[1 : self.number_of_layers - 1] * 1e9

        # set bounds
        bfgs_xmin = self.lower_bound * np.ones(self.number_of_layers - 2)
        bfgs_xmax = self.upper_bound * np.ones(self.number_of_layers - 2)
        # rewrite the bounds in the way required by L-BFGS-B
        bfgs_bounds = [(low, high) for low, high in zip(bfgs_xmin, bfgs_xmax)]

        fom_start, grad_start = self.compute_fom_and_gradient_from_thickness_array(
            x_start
        )
        print(f" Initial FOM is {fom_start}")
        print(f" Initial Gradient is {grad_start}")

        ret = minimize(
            self.compute_fom_and_gradient_from_thickness_array,
            x_start,
            method="L-BFGS-B",
            jac=True,
            bounds=bfgs_bounds,
        )
        print(ret.x)

    def optimize_basin_hopping(self):
        """
        Method to wrap the l-bfgs-b optimizer
        """
        x_start = self.thickness_array[1 : self.number_of_layers - 1] * 1e9

        # set bounds for L-BFGS-B local optimizations
        bfgs_xmin = self.lower_bound * np.ones(self.number_of_layers - 2)
        bfgs_xmax = self.upper_bound * np.ones(self.number_of_layers - 2)

        # rewrite the bounds in the way required by L-BFGS-B
        bfgs_bounds = [(low, high) for low, high in zip(bfgs_xmin, bfgs_xmax)]

        fom_start, grad_start = self.compute_fom_and_gradient_from_thickness_array(
            x_start
        )

        # should update to determine if "jac" : True is consistent with 
        # selected FOM
        minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "bounds": bfgs_bounds}

        print(f" Initial FOM is {fom_start}")
        print(f" Initial Gradient is {grad_start}")
        ret = basinhopping(
            func = self.compute_fom_and_gradient_from_thickness_array,
            x0 = x_start,
            minimizer_kwargs=minimizer_kwargs,
            niter=50,
            take_step=self._take_step,
            callback=self._print_callback,
        )
        return ret

    def _take_step(self,x):
        """ method to apply random perturbations to 
            the current structure x in the basin hopping routine
        """

        _x_perturbed = np.copy(x)

        dim = len(_x_perturbed)
        for i in range(0,dim):
            _x_curr = _x_perturbed[i]
            if _x_curr * self.random_perturbation_scale > 1.0:
                _x_range = int(_x_curr * self.random_perturbation_scale)
            else:
                _x_range = 1
            _pert = np.random.randint(-_x_range, _x_range)
            # we don't want to perturb to less than the lower bound or greater than the upper bound
            _x_try = _x_curr + _pert 
            if _x_try > self.lower_bound and _x_try < self.upper_bound:
                _x_perturbed[i] = _x_try
            elif _x_try <= self.lower_bound: 
                _x_perturbed[i] = _x_curr + np.abs(_pert) #<== increase in thickness
            elif _x_try >= self.upper_bound:
                _x_perturbed[i] = _x_curr - np.abs(_pert) #<== decrease in thickness
            else:
                _x_perturbed[i] = _x_curr #<== do not change
            
        return _x_perturbed
    
    def _print_callback(self,x, f, accepted):
        c_time = time.time()
        print(f"\n Time elapsed is {c_time-self.start_time}")
        print(f" Current structure is {x}")
        print(f" Current FOM is {f}")

    def compute_fom_and_gradient_from_thickness_array(self, x0):
        """ "
        Method to update the thickness array, the relevant spectra, objective function, and (if supported)
        the gradient
        """
        print(" Computing figure of merit and graident from thickness array ", x0)

        # need to add options to use other figures of merit
        self.thickness_array[1 : self.number_of_layers - 1] = x0 * 1e-9
        self.compute_spectrum()
        self.compute_selective_mirror_fom()
        # compute_selective_mirror_fom_gradient calls compute_spectrum_gradient()
        self.compute_selective_mirror_fom_gradient()

        fom_1 = self.reflection_efficiency
        grad_1 = self.reflection_efficiency_gradient

        fom_2 = self.reflection_selectivity
        grad_2 = self.reflection_selectivity_gradient

        fom_3 = self.transmission_efficiency
        grad_3 = self.transmission_efficiency_gradient

        _expected_fom = self.reflection_efficiency_weight * fom_1 
        _expected_fom += self.reflection_selectivity_weight * fom_2
        _expected_fom += self.transmission_efficiency_weight * fom_3 

        fom = self.selective_mirror_fom
        grad = self.selective_mirror_fom_gradient

        assert np.isclose(_expected_fom, fom)

        if self.minimization:
            fom *= 1
            grad *= 1e-9
        else:
            fom *= -1
            grad *= -1e-9

        print(f" Current FOM is {fom}")
        print(f" Current Gradient is {grad}")
        return fom, grad
