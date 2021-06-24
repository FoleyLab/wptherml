from abc import abstractmethod, ABC


class SpectrumDriver(ABC):
    @abstractmethod
    def compute_spectrum():
        pass
