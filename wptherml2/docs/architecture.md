# wptherml2 Architecture

`wptherml2` is the narrow TMM core for the next-generation package direction.

## Module boundaries

- `src/wptherml2/types.py`
  Immutable request/result containers and validation.
- `src/wptherml2/api.py`
  Stable public entry points.
- `src/wptherml2/tmm/wavevectors.py`
  Wavevector and angle helpers.
- `src/wptherml2/tmm/matrices.py`
  Interface and propagation matrix assembly.
- `src/wptherml2/tmm/solve.py`
  Forward scalar/vectorized TMM solvers.
- `src/wptherml2/tmm/gradients.py`
  Thickness-gradient implementation behind a stable API.
- `src/wptherml2/materials/base.py`
  Minimal extension hook for future material providers.

## Current scope

- single-angle spectra,
- `s` and `p` polarization,
- scalar and vectorized NumPy execution paths,
- thickness gradients for spectral observables.

## Explicit non-goals for this scaffold

- legacy `wptherml` driver integration,
- angle integration,
- optimization drivers,
- autodiff backends,
- broad material-library migration.

## Near-term upgrade path

The main internal seam to improve next is `tmm/gradients.py`.
It currently uses centered finite differences to keep the scaffold runnable and testable.
That file can be upgraded to analytical thickness derivatives without changing the public API.
