"""
Model persistence (PHASE 9).

Pickle-based serialisation of a :class:`~backend.ml.trainer.TrainedModel`.
We use pickle (not joblib) because the artefact is a single
:class:`TrainedModel` dataclass, not a sklearn pipeline, and we
want the file to be self-describing.

A sidecar ``<path>.version`` file is written alongside the pickle
holding the model's :attr:`TrainedModel.model_version` string.
On load, if the sidecar is missing or holds a different version,
a :class:`ModelVersionWarning` is emitted so the operator can
investigate cross-version compatibility.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

if TYPE_CHECKING:
    from .trainer import MODEL_VERSION as _MV  # noqa: F401
    from .trainer import TrainedModel


class ModelVersionWarning(UserWarning):
    """Raised when a loaded model was serialised with a different
    ``model_version`` than the current code expects (or when the
    sidecar version file is missing)."""


_VERSION_SUFFIX = ".version"


def _sidecar_path(path: Path) -> Path:
    return Path(str(path) + _VERSION_SUFFIX)


def save_model(model: "TrainedModel", path: Path) -> None:
    """Serialise a :class:`TrainedModel` to disk.

    Writes the pickle at ``path`` and a sidecar at
    ``<path>.version`` containing ``model.model_version``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    # Sidecar version stamp. Best-effort; failure to write the
    # sidecar should not break save (the pickle already contains
    # the version field).
    try:
        with open(_sidecar_path(path), "w", encoding="utf-8") as f:
            f.write(str(getattr(model, "model_version", "unknown")))
    except OSError:
        pass


def _current_model_version() -> str:
    # Imported lazily to avoid a circular import.
    from .trainer import MODEL_VERSION
    return str(MODEL_VERSION)


def load_model(path: Path) -> "TrainedModel":
    """Load a :class:`TrainedModel` from disk.

    Emits :class:`ModelVersionWarning` if the sidecar version
    differs from :data:`~backend.ml.trainer.MODEL_VERSION`, or if
    the sidecar is absent (older artefact). The warning is
    informational; the returned model is the loaded artefact.
    """
    path = Path(path)
    with open(path, "rb") as f:
        model = pickle.load(f)  # noqa: S301 — internal file format

    current = _current_model_version()
    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        warn(
            f"model at {path} has no sidecar version file; "
            f"expected version {current}. The model will be loaded "
            f"but features/semantics may have changed.",
            ModelVersionWarning,
            stacklevel=2,
        )
    else:
        try:
            stamped = sidecar.read_text(encoding="utf-8").strip()
        except OSError:
            stamped = ""
        if stamped != current:
            warn(
                f"model at {path} was saved with version "
                f"{stamped!r} but current code expects {current!r}. "
                f"Verify feature schema / split policy are still "
                f"compatible before relying on this artefact.",
                ModelVersionWarning,
                stacklevel=2,
            )
    return model


__all__ = [
    "ModelVersionWarning",
    "_VERSION_SUFFIX",
    "load_model",
    "save_model",
]
