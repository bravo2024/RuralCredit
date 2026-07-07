from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from src.data import make_synthetic
from src.model import train_all_models, cross_validate
from src.evaluate import save_metrics, print_report
from src.persist import save_model


def main():
    parser = argparse.ArgumentParser(description="RuralCredit — Alternative Credit Scoring")
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    data = make_synthetic(n=args.n, seed=args.seed)
    print(f"{data['n_samples']:,} samples, default rate={data['positive_rate']:.2%}")
    bundle = train_all_models(data, seed=args.seed)
    print_report({n: r["metrics"] for n, r in bundle["results"].items()})
    cv = cross_validate(data, seed=args.seed, n_folds=args.cv)
    for n, s in cv.items():
        print(f"  {n:25s} AUC={s['roc_auc']['mean']:.4f} \u00b1{s['roc_auc']['std']:.4f}")
    best = max(bundle["results"], key=lambda n: bundle["results"][n]["metrics"].get("roc_auc", 0))
    print(f"Best: {best}")
    save_model({"models": bundle["models"], "scaler": bundle["scaler"], "features": bundle["features"], "best_model": best})
    save_metrics({"holdout": {n: bundle["results"][n]["metrics"] for n in bundle["results"]}, "cv": cv, "best_model": best})
    print("Saved models/model.pkl and models/metrics.json")


if __name__ == "__main__":
    main()
