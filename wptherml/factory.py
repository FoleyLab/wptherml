from .spectrum_driver import SpectrumDriver

# child classes
from .mie import MieDriver
from .em import TmmDriver
from .therml import Therml
from .optdriver import OptDriver

class SpectrumFactory:
    def spectrum_factory(self, spectrum_toolkit, args):
        if spectrum_toolkit == "Mie":
            return MieDriver(args)
        elif spectrum_toolkit == "Tmm":
            return TmmDriver(args)
        elif spectrum_toolkit == "Opt":
            return OptDriver(args)
        else:
            raise TypeError("Toolkit not found")


"""class SpectrumFactory:
    _toolkits = {}

    def spectrum_factory(self, spectrum_toolkit, size):
        if spectrum_toolkit not in self._toolkits.keys():
            raise Exception('Toolkit not found.')
        cls = self._toolkits[spectrum_toolkit]
        return cls(size)

    def register(self, toolkit_name, toolkit_class):
        if not issubclass(toolkit_class, SpectrumAdapter):
            raise TypeError(f'{toolkit_class} is not a SpectrumAdapter')
        self._toolkits[toolkit_name] = toolkit_class
"""
