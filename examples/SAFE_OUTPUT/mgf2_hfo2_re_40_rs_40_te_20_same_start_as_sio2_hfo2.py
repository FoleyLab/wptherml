import wptherml
from matplotlib import pyplot as plt
import numpy as np


d_start_1 = [0.00000000e+00, 4.4754E-08, 1.82725E-08, 2.3352E-08, 2.60723E-08, 
           8.154E-09, 3.56807E-07, 8.916E-09, 2.54991E-08, 2.4936E-08, 9.26829E-09, 
           6.06372E-07, 9.13304E-09, 2.514E-08, 2.54604E-08, 9.036E-09, 3.63376E-07, 
           7.884E-09, 2.7631E-08, 2.3676E-08, 1.0608E-08, 5.1972E-07, 1.04792E-08,
           2.376E-08, 2.74313E-08, 8.01E-09, 5.02658E-07, 9.024E-09, 2.54991E-08, 2.5272E-08, 9.12015E-09,
4.60194E-07, 8.81744E-09, 2.562E-08, 2.52092E-08, 9.24E-09, 4.96771E-07, 8.562E-09, 
2.60723E-08, 2.4672E-08, 9.37135E-09, 4.58406E-07, 8.7015E-09, 2.5302E-08, 2.5074E-08, 8.844E-09,
3.55853E-07, 8.916E-09, 2.47713E-08, 2.5212E-08, 8.57913E-09, 5.96106E-07, 9.28117E-09, 2.3982E-08, 
2.50289E-08, 8.496E-09, 4.86511E-07, 8.1E-09, 2.61947E-08, 2.3814E-08, 9.88661E-09,
4.53114E-07, 7.98658E-09, 2.6544E-08, 2.40499E-08, 9.816E-09, 3.54514E-07, 8.298E-09, 
2.56859E-08, 2.4828E-08, 8.83676E-09, 6.01578E-07, 1.15226E-08, 1.8816E-08, 9.98837E-08, 4.7928E-08,
0.00000000e+00]

mat_start = []
ctr = 0
for d in d_start_1:
    if ctr==0:
        mat_start.append("Air")
    elif ctr % 2 == 1:
        mat_start.append("MgF2")
    else:
        mat_start.append("HfO2_udm_no_loss")

    ctr += 1

mat_start[ctr-1] = 'Air'
        

print(len(mat_start))  


start_args = {
    "wavelength_list": [200e-9, 7000e-9, 5000],
    "Material_List": mat_start,
    "Thickness_List": d_start_1,
    "reflective_window_wn" : [2600, 3750],
    "transmissive_window_nm" : [300, 700],
    "transmission_efficiency_weight" : 0.2,
    "reflection_efficiency_weight" : 0.4,
    "reflection_selectivity_weight" : 0.4
}

sf = wptherml.SpectrumFactory() 
test = sf.spectrum_factory('Tmm', start_args)
ts = sf.spectrum_factory("Opt", start_args)
results = ts.optimize_basin_hopping()

print(results)



