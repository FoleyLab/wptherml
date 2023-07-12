import wptherml
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import time

# guess thickness for glass
d1 = 5000e-9 / (4 * 1.5)
# guess thickness for Al2O3
d2 = 5000e-9 / (4 * 1.85)


test_args = {
    "wavelength_list": [300e-9, 6000e-9, 1000],
    "Material_List": ["Air","SiO2", "Al2O3", "SiO2", "ZrO2","SiO2", "Al2O3", "SiO2", "Al2O3","SiO2", "Al2O3", "SiO2", "Al2O3","SiO2", "Al2O3","SiO2", "Al2O3", "SiO2", "Al2O3","SiO2", "Al2O3", "SiO2", "Al2O3" ,"SiO2", "Al2O3", "SiO2", "Al2O3" , "Air"],
    "Thickness_List": [0,d1, d2, d1, d2,d1, d2, d1, d2, d1, d2, d1, d2, d1, d2,d1, d2, d1, d2, d1, d2, d1, d2, d1, d2, d1, d2, 0],
    "reflective_window_wn" : [2000, 2400],
    "transmissive_window_nm" : [350, 700],
    "transmission_efficiency_weight" : 0.5,
    "reflection_efficiency_weight" : 0.5
 }


# start the spectrum factory
sf = wptherml.SpectrumFactory()
# create an instance using the TMM with the structure defined as above
test = sf.spectrum_factory('Tmm', test_args)




def SuperFunc(x0):
    """ function to update the thicknesses of each layer given an
        array of thicknesses stored in x, recompute FOM, and return"""
    test.thickness_array[1:test.number_of_layers-1] = x0 * 1e-9
    test.compute_spectrum()
    test.compute_selective_mirror_fom()
    test.compute_selective_mirror_fom_gradient()
    
    # We have three choices for what to define our FOM as 
    # choice 1: \eta_R
    fom = test.reflection_efficiency
    grad = test.reflection_efficiency_gradient
    
    #choice 2: \eta_T
    #fom = test.transmission_efficiency
    #grad = test.transmission_efficiency_gradient
    
    # choice 3: average of \eta_R + \eta_T
    #fom = test.selective_mirror_fom
    #grad = test.selective_mirror_fom_gradient
    
    # return negative of fom and grad, scale grad so that step size is reasonable
    return -1 * fom, -1 * grad * 1e-9

# the bounds for L-BFGS-B updates
# minimum layer thickness is 1 nm
bfgs_xmin = np.ones(test.number_of_layers-2)

# maximum layer thickness is 1000 nm
bfgs_xmax = 1000*np.ones(test.number_of_layers-2)

# rewrite the bounds in the way required by L-BFGS-B
bfgs_bounds = [(low, high) for low, high in zip(bfgs_xmin, bfgs_xmax)]

# define d1 and d2 in nanometers based on the trial values from first cell
d1nm = 835
d2nm = 675

# define trial x array - only specifiy the central layers that
# are being varied and specify their thickness in nanometers, not meters
xs = np.array([d1nm, d2nm, d1nm, d2nm, d1nm, d2nm, d1nm, 
               d2nm, d1nm, d2nm, d1nm, d2nm, d1nm, d2nm,
               d1nm, d2nm, d1nm, d2nm, d1nm, d2nm, d1nm, 
               d2nm, d1nm, d2nm, d1nm, d2nm])

print("xs is ")
print(xs)
fom = SuperFunc(xs)
print("initial FOM is ", fom)

### prints efficiency and time
def print_fun(x, f, accepted):
    c_time = time.time()
    print(f,",",c_time)
### called by the basin hopping algorithm to initiate new
### local optimizations
def my_take_step(x):
    dim = len(x)
    #xnew = np.copy(x)
    dim = len(xnew)
    # define xnew as a random scaling (between 0 and 2) of current x + 1 
    # so that nothing goes to zero accidentally
    xnew = 2 * np.random.rand(dim) * x + np.ones(dim) 

    #for i in range(0,dim):
    #    _d = np.random.randint(1, 10000) 
    #    
    #    xnew[i] = _d
    return xnew


### bounds on basin hopping solutions
class MyBounds(object):
      ### note xmax and xmin need to have as many elements as there are thicknesses that are varied
    def __init__(self, xmax=5000.*np.ones(test.number_of_layers-2), xmin=1.0*np.ones(test.number_of_layers-2)):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

# the bounds for L-BFGS-B updates!
bfgs_xmin = 1.0*np.ones(test.number_of_layers-2)
bfgs_xmax = 30000*np.ones(test.number_of_layers-2)

# rewrite the bounds in the way required by L-BFGS-B
bfgs_bounds = [(low, high) for low, high in zip(bfgs_xmin, bfgs_xmax)]

### arguments for basin hopping algorithm
minimizer_kwargs = {"method": "L-BFGS-B", "jac": True, "bounds": bfgs_bounds}
mybounds = MyBounds()

# define d1 and d2 in nanometers based on the trial values from first cell
d1nm = 835
d2nm = 675

# define trial x array - only specifiy the central layers that
# are being varied and specify their thickness in nanometers, not meters
xs = np.array([d1nm, d2nm, d1nm, d2nm, d1nm, d2nm, d1nm, 
               d2nm, d1nm, d2nm, d1nm, d2nm, d1nm, d2nm,
               d1nm, d2nm, d1nm, d2nm, d1nm, d2nm, d1nm, 
               d2nm, d1nm, d2nm, d1nm, d2nm])

print("xs is ")
print(xs)
fom = SuperFunc(xs)
print("initial FOM is ", fom)
### call basin hopping!
ret = basinhopping(SuperFunc, xs, minimizer_kwargs=minimizer_kwargs, niter=3, take_step=my_take_step, callback=print_fun, accept_test=mybounds)

### print optimimal result!
print(ret.x)
print(-SuperFunc(ret.x))

### print optimal solution and its efficiency!
#print(ret.x)
#print(-update_multilayer(ret.x))