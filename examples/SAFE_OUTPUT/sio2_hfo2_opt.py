import wptherml
from matplotlib import pyplot as plt
import numpy as np


d_start_1 = np.array([0.00000000e+00,
7.8961E-08, 2.837E-08, 4.12007E-08, 4.048E-08, 1.43864E-08, 5.5398E-07, 1.57308E-08, 3.959E-08, 4.39954E-08, 1.439E-08, 1.0551E-06,
1.418E-08, 4.43554E-08, 3.953E-08, 1.59425E-08, 5.6418E-07, 1.391E-08, 4.29E-08, 4.17724E-08, 1.647E-08, 9.04325E-07, 1.627E-08, 4.19206E-08, 4.259E-08,
1.41323E-08, 7.8043E-07, 1.59214E-08, 3.959E-08, 4.45883E-08, 1.416E-08, 8.00748E-07, 1.369E-08, 4.52023E-08, 3.914E-08, 1.63025E-08, 7.7129E-07,
1.51062E-08, 4.048E-08, 4.35297E-08, 1.455E-08, 7.97637E-07, 1.351E-08, 4.46412E-08, 3.893E-08, 1.56038E-08, 5.525E-07, 1.57308E-08, 3.846E-08,
 4.44824E-08, 1.332E-08, 1.03724E-06, 1.441E-08, 4.23123E-08, 3.886E-08, 1.49898E-08, 7.5536E-07, 1.42911E-08, 4.067E-08, 4.20159E-08, 1.535E-08, 7.88429E-07,
 1.24E-08, 4.68325E-08, 3.734E-08, 1.73187E-08, 5.5042E-07, 1.46404E-08, 3.988E-08, 4.38049E-08, 1.372E-08, 1.04676E-06, 1.789E-08, 3.31977E-08, 1.5508E-07, 8.4561E-08,
0.00000000e+00])

mat_start = []
ctr = 0
for d in d_start_1:
    if ctr==0:
        mat_start.append("Air")
    elif ctr % 2 == 1:
        mat_start.append("SiO2_udm")
    else:
        mat_start.append("HfO2_udm")

    ctr += 1

mat_start[ctr-1] = 'Air'
        

print(len(mat_start))  


start_args = {
    "wavelength_list": [200e-9, 7000e-9, 2000],
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
print(results.x)
for i in range(len(results.x)):
    print(F'{results.x[i]},')


