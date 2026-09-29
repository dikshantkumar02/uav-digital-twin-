#!/usr/bin/env python3
"""
Train the fault classifier (PHASE 9) and save the artefact.

Usage::

    python scripts/train_classifier.py \
        --config config \
        --output models/classifier.pkl \
        --n-estimators 100 \
        --seed 0

The output is a pickle file containing a
:class:`backend.ml.TrainedModel`. The dashboard (PHASE 13) loads
it at startup via :class:`backend.ml.FaultClassifier`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.ml import build_dataset, save_model, train_classifier  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config",
                   help="YAML config dir (default: ./config)")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "classifier.pkl",
                   help="Output pickle path")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-kind", choices=["random_forest", "gradient_boosting"],
                   default="random_forest")
    args = p.parse_args()

    print(f"Building dataset (config={args.config}, seed={args.seed})...")
    ds = build_dataset(config_dir=args.config, seed=args.seed)
    print(f"  rows: {ds.n_rows}, classes: {ds.n_per_class}")
    if ds.n_rows == 0:
        print("ERROR: empty dataset", file=sys.stderr)
        return 1

    print(f"Training {args.model_kind} (n_estimators={args.n_estimators})...")
    model = train_classifier(
        ds, model_kind=args.model_kind, n_estimators=args.n_estimators, seed=args.seed,
    )
    print(f"  train_accuracy: {model.train_accuracy:.3f}")
    if model.held_out_accuracy is not None:
        print(f"  held_out_accuracy: {model.held_out_accuracy:.3f}")

    save_model(model, args.output)
    print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
