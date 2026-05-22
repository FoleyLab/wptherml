from .mie import MieDriver
from .em import TmmDriver
from .vec_tmm import VecTmmDriver
from .optdriver import OptDriver


class SpectrumFactory:
    _toolkits = {
        "Mie": MieDriver,
        "Tmm": TmmDriver,
        "VecTmm": VecTmmDriver,
        "Opt": OptDriver,
    }

    def spectrum_factory(self, spectrum_toolkit, args):
        try:
            driver_class = self._toolkits[spectrum_toolkit]
        except KeyError:
            supported = ", ".join(sorted(self._toolkits))
            raise TypeError(
                f"Toolkit '{spectrum_toolkit}' not found. Supported toolkits: {supported}"
            ) from None
        return driver_class(args)


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
