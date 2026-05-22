"""Abstract solver interfaces."""

from abc import ABC, abstractmethod


class Solver(ABC):
    @abstractmethod
    def solve(self, *args, **kwargs):
        """Compute a result object from a physical model."""
        raise NotImplementedError


class GradientSolver(ABC):
    @abstractmethod
    def solve_gradients(self, *args, **kwargs):
        """Compute gradients of a result object."""
        raise NotImplementedError
