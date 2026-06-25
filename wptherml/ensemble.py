"""Monte-Carlo ensembles over multilayer thickness uncertainty.

Fabricated multilayer stacks have manufacturing tolerances: a nominal design
``Air | d1 glass | d2 titania | d3 glass | Air`` really has thicknesses
``d1, d2, d3`` drawn from some distribution about the target values. This module
makes it easy to propagate that uncertainty through the optical response.

:class:`ThicknessEnsemble` takes a target :class:`MultilayerStructure`, samples a
swarm of replica structures whose internal-layer thicknesses are perturbed about
the target (a normal distribution by default), and evaluates reflectivity,
transmissivity and emissivity (and, optionally, their thickness gradients) for
the entire swarm in a single vectorized pass -- the replicas ride along as one
more batch axis of the fast transfer-matrix kernel, so there is no Python loop
over samples.

Example
-------
>>> structure = MultilayerStructure.from_spec(
...     materials=["Air", "SiO2", "TiO2", "SiO2", "Air"],
...     thicknesses=[0, 230e-9, 120e-9, 230e-9, 0],
...     wavelengths=np.linspace(400e-9, 800e-9, 200),
... )
>>> ensemble = ThicknessEnsemble(structure, relative_sigma=0.05, number_of_samples=512)
>>> result = ensemble.solve(polarizations="p")
>>> result.reflectivity.shape          # (512, 200)
>>> result.reflectivity_mean.shape     # (200,)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .structures import MultilayerStructure
from .tmm_fast import solve_rt, solve_rt_gradients


# Default 1-sigma thickness tolerance as a fraction of each layer's nominal
# thickness, used when neither an absolute nor relative sigma is supplied.
DEFAULT_RELATIVE_SIGMA = 0.05


@dataclass(frozen=True)
class EnsembleResult:
    """Per-replica spectra over a thickness ensemble, plus summary statistics.

    The leading axis of every spectrum array indexes the replica. When the solve
    used a single angle and a single polarization the angle/polarization axes are
    squeezed away, so ``reflectivity`` has shape ``(number_of_samples, N_wl)``;
    otherwise it is ``(number_of_samples, N_wl, N_angle, N_pol)``.
    """

    thicknesses: NDArray  # (number_of_samples, number_of_layers) sampled thicknesses
    wavelengths: NDArray
    reflectivity: NDArray
    transmissivity: NDArray
    emissivity: NDArray
    angles: NDArray | None = None
    polarizations: list[str] | None = None
    gradients: dict | None = None  # optional {"dR_dd","dT_dd","dE_dd","gradient_layers"}

    @property
    def number_of_samples(self) -> int:
        return self.thicknesses.shape[0]

    @property
    def reflectivity_mean(self) -> NDArray:
        return self.reflectivity.mean(axis=0)

    @property
    def reflectivity_std(self) -> NDArray:
        return self.reflectivity.std(axis=0)

    @property
    def transmissivity_mean(self) -> NDArray:
        return self.transmissivity.mean(axis=0)

    @property
    def transmissivity_std(self) -> NDArray:
        return self.transmissivity.std(axis=0)

    @property
    def emissivity_mean(self) -> NDArray:
        return self.emissivity.mean(axis=0)

    @property
    def emissivity_std(self) -> NDArray:
        return self.emissivity.std(axis=0)

    def quantile(self, q):
        """Return (R, T, E) quantiles across the ensemble at level(s) ``q``."""
        return (
            np.quantile(self.reflectivity, q, axis=0),
            np.quantile(self.transmissivity, q, axis=0),
            np.quantile(self.emissivity, q, axis=0),
        )


class ThicknessEnsemble:
    """Sample and solve a swarm of structures perturbed about a target.

    Parameters
    ----------
    structure : MultilayerStructure
        the nominal (target) structure; must carry refractive indices.
    sigma : float or array_like, optional
        absolute 1-sigma thickness uncertainty in meters. A scalar applies to
        every sampled layer; an array must match the sampled layers in order.
        Takes precedence over ``relative_sigma`` when both are given.
    relative_sigma : float, optional
        1-sigma uncertainty as a fraction of each sampled layer's nominal
        thickness. Used when ``sigma`` is not provided
        (default ``DEFAULT_RELATIVE_SIGMA`` = 0.05).
    number_of_samples : int, optional
        number of replicas to draw (default 256).
    sample_layers : sequence of int, optional
        indices of layers to perturb. Defaults to all internal (finite-thickness)
        layers ``1 .. number_of_layers - 2``; terminal semi-infinite layers are
        never perturbed.
    distribution : {"normal", "uniform"}, optional
        sampling distribution about the target thickness (default "normal").
        For "uniform" the half-width of the interval is ``sqrt(3) * sigma`` so the
        standard deviation still equals ``sigma``.
    minimum_thickness : float, optional
        sampled thicknesses are clipped to be at least this value in meters
        (default 0.0, i.e. non-negative).
    seed : int or numpy.random.Generator, optional
        seed or generator for reproducible sampling.
    """

    def __init__(
        self,
        structure: MultilayerStructure,
        sigma=None,
        relative_sigma: float | None = None,
        number_of_samples: int = 256,
        sample_layers=None,
        distribution: str = "normal",
        minimum_thickness: float = 0.0,
        seed=None,
    ) -> None:
        if structure.refractive_indices is None:
            raise ValueError(
                "ThicknessEnsemble requires structure.refractive_indices to be set"
            )
        distribution = distribution.lower()
        if distribution not in {"normal", "uniform"}:
            raise ValueError("distribution must be 'normal' or 'uniform'")

        self.structure = structure
        self.number_of_samples = int(number_of_samples)
        self.distribution = distribution
        self.minimum_thickness = float(minimum_thickness)
        self._rng = np.random.default_rng(seed)

        number_of_layers = structure.number_of_layers
        if sample_layers is None:
            sample_layers = list(range(1, number_of_layers - 1))
        self.sample_layers = np.atleast_1d(np.asarray(sample_layers, dtype=int))
        if np.any(
            (self.sample_layers < 1) | (self.sample_layers > number_of_layers - 2)
        ):
            raise ValueError(
                "sample_layers must be internal layers 1 .. number_of_layers - 2"
            )

        nominal = np.asarray(structure.thicknesses, dtype=float)[self.sample_layers]
        if sigma is not None:
            sigma_array = np.broadcast_to(
                np.asarray(sigma, dtype=float), nominal.shape
            ).copy()
        else:
            rel = (
                DEFAULT_RELATIVE_SIGMA if relative_sigma is None else float(relative_sigma)
            )
            sigma_array = rel * nominal
        self.sigma = sigma_array
        self._nominal_sampled = nominal

    def sample_thicknesses(self) -> NDArray:
        """Draw a ``(number_of_samples, number_of_layers)`` array of thicknesses."""
        base = np.broadcast_to(
            np.asarray(self.structure.thicknesses, dtype=float),
            (self.number_of_samples, self.structure.number_of_layers),
        ).copy()

        shape = (self.number_of_samples, self.sample_layers.size)
        if self.distribution == "normal":
            draws = self._rng.normal(
                loc=self._nominal_sampled, scale=self.sigma, size=shape
            )
        else:  # uniform with matching standard deviation
            half_width = np.sqrt(3.0) * self.sigma
            draws = self._rng.uniform(
                low=self._nominal_sampled - half_width,
                high=self._nominal_sampled + half_width,
                size=shape,
            )

        draws = np.clip(draws, self.minimum_thickness, None)
        base[:, self.sample_layers] = draws
        return base

    def solve(self, polarizations=None, gradients: bool = False) -> EnsembleResult:
        """Sample the ensemble and evaluate spectra (optionally gradients).

        Parameters
        ----------
        polarizations : str or sequence of str, optional
            polarization(s) to evaluate (default "p").
        gradients : bool, optional
            if True, also compute thickness gradients of R, T and E with respect
            to the sampled layers, stored in ``result.gradients``.
        """
        thicknesses = self.sample_thicknesses()
        structure = self.structure

        if gradients:
            (R, T, E), (dR, dT, dE) = solve_rt_gradients(
                structure.refractive_indices,
                structure.wavelengths,
                structure.angles,
                thicknesses,
                self.sample_layers,
                polarizations,
            )
            gradient_data = {
                "dR_dd": dR,
                "dT_dd": dT,
                "dE_dd": dE,
                "gradient_layers": np.array(self.sample_layers),
            }
        else:
            R, T, E = solve_rt(
                structure.refractive_indices,
                structure.wavelengths,
                structure.angles,
                thicknesses,
                polarizations,
            )
            gradient_data = None

        # Squeeze trailing angle/polarization axes when both are singletons, so a
        # simple normal-incidence, single-polarization ensemble is (N_replica, N_wl).
        angles = structure.angles
        n_angle = 1 if angles is None else np.atleast_1d(angles).shape[0]
        n_pol = R.shape[-1]
        if n_angle == 1 and n_pol == 1:
            R, T, E = R[:, :, 0, 0], T[:, :, 0, 0], E[:, :, 0, 0]
            if gradient_data is not None:
                for key in ("dR_dd", "dT_dd", "dE_dd"):
                    gradient_data[key] = gradient_data[key][:, :, 0, 0, :]

        return EnsembleResult(
            thicknesses=thicknesses,
            wavelengths=np.asarray(structure.wavelengths, dtype=float),
            reflectivity=R,
            transmissivity=T,
            emissivity=E,
            angles=None if angles is None else np.atleast_1d(angles),
            polarizations=None,
            gradients=gradient_data,
        )
