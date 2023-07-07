import wptherml
from matplotlib import pyplot as plt
import numpy as np

test_args = {
    'Material_List': ['Air', 'SiO2', 'HfO2', 'SiO2', 'HfO2', 'SiO2', 'HfO2', 'SiO2', 'Ag', 'Air'],
    'Thickness_List': [0, 230e-9, 485e-9, 688e-9, 13e-9, 73e-9, 34e-9, 54e-9, 200e-9, 0],
    'Wavelength_List': [300e-9, 30000e-9, 2000], # note new name for this key
    "cooling": True # note use of boolean not integer now
}

# start the spectrum factory
sf = wptherml.SpectrumFactory()
# create an instance using the TMM with the structure defined as above
cool_ml = sf.spectrum_factory('Tmm', test_args)

from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time

test_args = {
    'Material_List': ['Air', 'SiO2', 'HfO2', 'SiO2', 'HfO2', 'SiO2', 'HfO2', 'SiO2', 'Ag', 'Air'],
    'Thickness_List': [0, 230e-9, 485e-9, 688e-9, 13e-9, 73e-9, 34e-9, 54e-9, 200e-9, 0],
    'Wavelength_List': [300e-9, 60000e-9, 5000], # note new name for this key
    "cooling": True # note use of boolean not integer now
}

# start the spectrum driver
sf = wptherml.SpectrumFactory()
# create an instance using the TMM with the structure defined as above
cool_ml = sf.spectrum_factory('Tmm', test_args)

def update_multilayer(x):
    """ function to update the thicknesses of each layer given an
        array of thicknesses stored in x"""
    
    cool_ml.thickness_array[1:cool_ml.number_of_layers-1] = x * 1e-9
    cool_ml.compute_cooling()

    ### return negative of cooling power - minimize functions want 
    ### to minimize, so trick them by passing negative of the objective you
    ### want to maximize
    return -cool_ml.net_cooling_power

### given an array of thicknesses of the coating, update
### the structure and compute the gradient vector of conversion efficiency wrt layer thicknesses
def analytic_grad(x0):
    cur = update_multilayer(x0)
    cool_ml.compute_cooling_gradient()

    g = cool_ml.net_cooling_power_gradient
    ### scale gradient to be in nm^-1 rather than over m^-1
    return -g*1e-9

### Function that gets the negative of the efficiency and the 
### negative of the gradient for use in the l-bfgs-b algorithm
### also prints out the time for timing purposes!
def SuperFunc(x0):
    en = update_multilayer(x0)
    c_time = time.time()
    if en<0:
        print(" This structure is net cooling with net power out being",-en)
    else:
        print(" This structure is net warming with net poer in being",-en)
    gr = analytic_grad(x0)
    return en, gr

# the bounds for L-BFGS-B updates!
# minimum layer thickness is 1 nm
bfgs_xmin = np.ones(cool_ml.number_of_layers-2)
# maximum layer thickness is 400 nm
bfgs_xmax = 30000.*np.ones(cool_ml.number_of_layers-2)

# rewrite the bounds in the way required by L-BFGS-B
bfgs_bounds = [(low, high) for low, high in zip(bfgs_xmin, bfgs_xmax)]

### initialize the solution vector xs to be the thicknesses from 
### Raman et al. paper
xs = np.array([230, 485, 688, 13, 73, 34, 54, 200])
### print out initial solution vector and initial efficiency
print("xs is ")
print(xs)
pflux = -update_multilayer(xs)
if pflux>0:
    print(" This structure is net cooling with net power out being",pflux)   
else:
    print(" This structure is net warming with net power in being",pflux)


### run l-bfgs-b algorithm!
#ret = minimize(SuperFunc, xs, method="L-BFGS-B", jac=True, bounds=bfgs_bounds)

### prints efficiency and time
def print_fun(x, f, accepted):
    c_time = time.time()
    print(f,",",c_time)

### called by the basin hopping algorithm to initiate new
### local optimizations
def my_take_step(x):
    xnew = np.copy(x)
    dim = len(xnew)
    for i in range(0,dim):
        _d = np.random.randint(1, 10000) 
        xnew[i] = _d
    return xnew


### bounds on basin hopping solutions
class MyBounds(object):
      ### note xmax and xmin need to have as many elements as there are thicknesses that are varied
    def __init__(self, xmax=30000.*np.ones(cool_ml.number_of_layers-2), xmin=1.0*np.ones(cool_ml.number_of_layers-2)):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

# the bounds for L-BFGS-B updates!
bfgs_xmin = 1.0*np.ones(cool_ml.number_of_layers-2)
bfgs_xmax = 30000*np.ones(cool_ml.number_of_layers-2)

# rewrite the bounds in the way required by L-BFGS-B
bfgs_bounds = [(low, high) for low, high in zip(bfgs_xmin, bfgs_xmax)]

### arguments for basin hopping algorithm
minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "bounds": bfgs_bounds}
mybounds = MyBounds()

### initial guess for AR layer thicknesses!
xs = np.array([230, 485, 688, 13, 73, 34, 54, 200])

### call basin hopping!
ret = basinhopping(SuperFunc, xs, minimizer_kwargs=minimizer_kwargs, niter=100, take_step=my_take_step, callback=print_fun, accept_test=mybounds)

### print optimimal result!
print(ret.x)
print(-update_multilayer(ret.x))

### print optimal solution and its efficiency!
#print(ret.x)
#print(-update_multilayer(ret.x))