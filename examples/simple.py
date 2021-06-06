import wpspecdev

mdf = wpspecdev.SpectrumFactory()
md = mdf.spectrum_factory('Mie', 100e-9)
print(md.compute_spectrum())
print(md.Q_ext)


