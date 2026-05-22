"""Abstract objective interface."""

from abc import ABC, abstractmethod


class Objective(ABC):
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluate the objective."""
        raise NotImplementedError

    @abstractmethod
    def gradient(self, *args, **kwargs):
        """Evaluate the objective gradient."""
        raise NotImplementedError
