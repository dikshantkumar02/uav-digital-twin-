# Configuration reference

All engine, environment, sensor and fault parameters live in
[`config/`](../config/) as YAML files. They are loaded and validated by
`backend.config.load_config`.

## Files

| File                  | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `engine.yaml`         | Representative aero-piston engine geometry, maps, limits.  |
| `environment.yaml`    | Standard atmosphere, mission profile, turbulence, gusts.   |
| `sensors.yaml`        | Virtual sensor channels (12), each with noise/bias/etc.    |
| `faults.yaml`         | Fault classes and their parameterised effects.             |

## Provenance labels

Every numeric value carries a provenance label. The valid labels are:

| Label              | Meaning                                                 |
|--------------------|---------------------------------------------------------|
| `MEASURED`         | Real instrumented test data (must be supplied by user). |
| `PUBLIC_DATA`      | Open literature / textbooks.                            |
| `PHYSICS_DERIVED`  | Derived from first principles.                          |
| `INTERPOLATED`     | Interpolated from public / measured points.             |
| `SYNTHETIC`        | Synthesised for the prototype — not from a real engine. |
| `USER_CONFIGURED`  | Supplied at runtime by the operator.                    |

YAML comments document the provenance inline. The Pydantic schema also
enforces a default provenance on the engine block.

## Overriding at runtime

```python
from backend.config import load_config

cfg = load_config("/path/to/my/config/dir")
print(cfg.engine.geometry.rated_rpm)
```

Tests can patch the cache:

```python
from backend.config import reload_config
cfg = reload_config()  # bypass lru_cache
```
