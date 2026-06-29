from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def save_metrics(metrics, path="models/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cleaned = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            cleaned[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            cleaned[k] = int(v)
        elif isinstance(v, dict):
            cleaned[k] = {sk: float(sv) if isinstance(sv, (np.floating, float)) else sv for sk, sv in v.items()}
        else:
            cleaned[k] = v
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2)


def print_report(all_metrics):
    for model_name, metrics in all_metrics.items():
        if model_name == "cv":
            continue
        print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k:20s}: {v:.4f}")
            else:
                print(f"    {k:20s}: {v}")
