# wptherml2

`wptherml2` is a clean-room transfer-matrix subpackage scaffold for the next phase of `wptherml`.

This first cut intentionally stays small:

- one-angle TMM observables,
- scalar and vectorized NumPy backends,
- typed request/result objects,
- a working thickness-gradient API,
- focused tests, docs, and a simple benchmark entry point.

The current gradient implementation uses centered finite differences as a correctness-first baseline. The public API is designed so an analytical gradient engine can replace the internals without breaking callers.

## Public API

```python
from wptherml2 import (
    LayerStack,
    SpectralGrid,
    TmmRequest,
    compute_gradients,
    compute_observables,
)
```

## Development

```bash
cd wptherml2
python -m pytest -q
```
