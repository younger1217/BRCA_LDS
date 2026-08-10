"""
Raw-matrix leakage-free permutation test.

Each observed/permuted run starts from the original full omics matrices. Within
each CV training fold, zero substitution, ANOVA F-test feature selection, and
scaling are applied on training samples only; held-out samples are transformed
only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from raw_leakage_free_nested_cv import load_raw_data, metrics, train_eval


def json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def run_cv(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    sample_ids: list[str],
    feature_k: dict[str, int],
    n_splits: int,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model_params = {
        "omics_dims": {},
        "hidden_dim": 128,
        "num_classes": 2,
        "num_layers": 1,
        "dropout": 0.3,
        "num_heads": 8,
    }
    train_params = {"epochs": epochs, "lr": 1e-3, "weight_decay": 1e-4, "patience": patience}
    rows = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(labels)), labels), 1):
        result = train_eval(
            data,
            labels,
            train_idx,
            test_idx,
            feature_k,
            model_params,
            train_params,
            seed=seed * 100 + fold,
        )
        for local_i, sample_index in enumerate(test_idx):
            probs = result["probabilities"][local_i]
            rows.append(
                {
                    "fold": fold,
                    "sample_index": int(sample_index),
                    "sample_id": sample_ids[sample_index],
                    "true_label": int(result["true_labels"][local_i]),
                    "prediction": int(result["predictions"][local_i]),
                    "probability_class_0": float(probs[0]),
                    "probability_class_1": float(probs[1]),
                }
            )
    pred = pd.DataFrame(rows)
    score = metrics(pred.true_label.to_numpy(), pred.prediction.to_numpy(), pred.probability_class_0.to_numpy())
    return pred, score


def empirical_p(observed: float, null_values: pd.Series) -> float:
    values = null_values.dropna().to_numpy(dtype=float)
    return float((np.sum(values >= observed) + 1) / (len(values) + 1))


def plot_null(summary: pd.DataFrame, observed: dict[str, float], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["AUC", "PR_AUC", "Accuracy"]):
        ax.hist(summary[metric].dropna(), bins=20, color="#6C8EBF", alpha=0.85)
        ax.axvline(observed[metric], color="#C44E52", linewidth=2, label="Observed")
        ax.set_xlabel(metric)
        ax.set_ylabel("Permuted-label runs")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_permutation_null_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "raw_permutation_null_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Data-origin-matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw_leakage_free_permutation"))
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-transcriptome", type=int, default=1000)
    parser.add_argument("--k-proteomic", type=int, default=1000)
    parser.add_argument("--k-metabolomic", type=int, default=500)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, labels, sample_ids, _ = load_raw_data(args.data_dir)
    feature_k = {
        "transcriptome": args.k_transcriptome,
        "proteomic": args.k_proteomic,
        "metabolomic": args.k_metabolomic,
    }

    observed_predictions, observed = run_cv(
        data, labels, sample_ids, feature_k, args.n_splits, args.seed, args.epochs, args.patience
    )
    observed_predictions.to_csv(args.output_dir / "raw_observed_predictions.csv", index=False)

    rng = np.random.default_rng(args.seed)
    rows = []
    for permutation in range(args.n_permutations):
        permuted = rng.permutation(labels)
        _, score = run_cv(
            data,
            permuted,
            sample_ids,
            feature_k,
            args.n_splits,
            args.seed + permutation + 1,
            args.epochs,
            args.patience,
        )
        rows.append({"permutation": permutation, **score})
        pd.DataFrame(rows).to_csv(args.output_dir / "raw_permutation_summary_partial.csv", index=False)

    null_summary = pd.DataFrame(rows)
    null_summary.to_csv(args.output_dir / "raw_permutation_summary.csv", index=False)
    null_results = {
        metric: {
            "observed": observed[metric],
            "null_mean": float(null_summary[metric].mean()),
            "null_sd": float(null_summary[metric].std(ddof=0)),
            "null_95th_percentile": float(null_summary[metric].quantile(0.95)),
            "empirical_p": empirical_p(observed[metric], null_summary[metric]),
        }
        for metric in ["AUC", "PR_AUC", "Accuracy", "Sensitivity_BC_LDS", "Specificity_others", "F1_BC_LDS"]
    }
    report = {
        "n_samples": int(len(labels)),
        "class_distribution": {"BC-LDS_label_0": int((labels == 0).sum()), "other_label_1": int((labels == 1).sum())},
        "raw_feature_counts": {k: int(v.shape[1]) for k, v in data.items()},
        "fold_internal_selected_features": feature_k,
        "n_permutations": args.n_permutations,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "observed": observed,
        "null_results": null_results,
        "preprocessing": "For observed and every permuted-label run, each CV training fold applies zero substitution, ANOVA F-test feature selection, and MinMax scaling; each held-out fold is transformed only.",
    }
    (args.output_dir / "raw_permutation_test_summary.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_null(null_summary, observed, args.output_dir)
    print(pd.DataFrame(null_results).T.to_string())


if __name__ == "__main__":
    main()
