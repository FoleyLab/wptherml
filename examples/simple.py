import wpspecdev

mdf = wpspecdev.SpectrumFactory()
md = mdf.spectrum_factory('Tmm', 100e-9)
print(md.thickness)
print(md.number_of_wavelengths)
print(md._refractive_index_array[1,3])
print(md.compute_spectrum())



