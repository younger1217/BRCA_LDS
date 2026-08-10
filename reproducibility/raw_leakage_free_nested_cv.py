"""
Leakage-free repeated nested CV from raw full omics matrices.

This experiment starts from the full feature matrices and
applies zero substitution, feature selection, and scaling inside each train
split only.
Labels are inferred from sample suffixes: A = BC-LDS, others = pooled control.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from modules.model_training import ModelTrainer, MultiOmicsGATModel  # noqa: E402


SAMPLE_RE = re.compile(r"(?:^|_)(\d{3})_([A-D])$")
OMICS_DEFAULTS = {
    "transcriptome": "gene_FPKM_matrix.csv",
    "proteomic": "ProteinTable_matrix.csv",
    "metabolomic": "metabolites_matrix.csv",
}


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_id(column: str) -> str | None:
    match = SAMPLE_RE.search(str(column))
    if not match:
        return None
    return f"{match.group(1)}_{match.group(2)}"


def feature_label(row: pd.Series, omics_type: str, idx: int) -> str:
    if omics_type == "transcriptome":
        return f"{row.iloc[0]}|{row.iloc[1]}"
    if omics_type == "proteomic":
        symbol = str(row.get("Symbol", "")).strip()
        accession = str(row.get("Accession_id", "")).strip()
        return symbol or accession or f"protein_{idx}"
    if omics_type == "metabolomic":
        return f"{row.iloc[0]}|{row.iloc[1]}"
    return f"{omics_type}_{idx}"


def read_omics_matrix(path: Path, omics_type: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    sample_cols = [c for c in df.columns if sample_id(c)]
    if not sample_cols:
        raise ValueError(f"No sample columns detected in {path}")
    ids = [sample_id(c) for c in sample_cols]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample IDs detected in {path}")

    values = df[sample_cols].apply(pd.to_numeric, errors="coerce").T
    values.index = ids
    names = [feature_label(row, omics_type, i) for i, (_, row) in enumerate(df.iterrows())]
    values.columns = names
    return values, names


def load_raw_data(data_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray, list[str], dict[str, list[str]]]:
    frames: dict[str, pd.DataFrame] = {}
    features: dict[str, list[str]] = {}
    for omics_type, filename in OMICS_DEFAULTS.items():
        frames[omics_type], features[omics_type] = read_omics_matrix(data_dir / filename, omics_type)

    common = sorted(set.intersection(*(set(frame.index) for frame in frames.values())), key=lambda x: int(x[:3]))
    if len(common) == 0:
        raise ValueError("No shared samples across omics matrices")

    labels = np.array([0 if sid.endswith("_A") else 1 for sid in common], dtype=int)
    data = {omics: frame.loc[common].to_numpy(dtype=float) for omics, frame in frames.items()}
    return data, labels, common, features


def preprocess_train_test(
    train_raw: dict[str, np.ndarray],
    test_raw: dict[str, np.ndarray],
    train_labels: np.ndarray,
    feature_k: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    train_out: dict[str, np.ndarray] = {}
    test_out: dict[str, np.ndarray] = {}
    selected_counts: dict[str, int] = {}

    for omics_type in train_raw:
        x_train = train_raw[omics_type].copy()
        x_test = test_raw[omics_type].copy()

        valid = ~np.all(np.isnan(x_train), axis=0)
        x_train = x_train[:, valid]
        x_test = x_test[:, valid]

        x_train = np.where(np.isnan(x_train), 0.0, x_train)
        x_test = np.where(np.isnan(x_test), 0.0, x_test)

        nonconstant = np.nanvar(x_train, axis=0) > 0
        x_train = x_train[:, nonconstant]
        x_test = x_test[:, nonconstant]

        k = min(feature_k[omics_type], x_train.shape[1])
        selector = SelectKBest(f_classif, k=k)
        x_train = selector.fit_transform(x_train, train_labels)
        x_test = selector.transform(x_test)

        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        train_out[omics_type] = x_train.astype(np.float32)
        test_out[omics_type] = x_test.astype(np.float32)
        selected_counts[omics_type] = int(k)

    return train_out, test_out, selected_counts


def subset(data: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[idx] for k, v in data.items()}


def train_eval(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_k: dict[str, int],
    model_params: dict,
    train_params: dict,
    seed: int,
) -> dict:
    train_data, test_data, selected_counts = preprocess_train_test(
        subset(data, train_idx), subset(data, test_idx), labels[train_idx], feature_k
    )
    params = model_params.copy()
    params["omics_dims"] = {k: v.shape[1] for k, v in train_data.items()}
    set_seed(seed)
    model = MultiOmicsGATModel(**params)
    trainer = ModelTrainer(model, random_state=seed)
    trainer.train_model(train_data, labels[train_idx], **train_params)
    result = trainer.evaluate_model(test_data, labels[test_idx])
    result["selected_counts"] = selected_counts
    return result


def metrics(y_raw: np.ndarray, pred_raw: np.ndarray, prob_class0: np.ndarray) -> dict[str, float]:
    y = (np.asarray(y_raw) == 0).astype(int)
    pred = (np.asarray(pred_raw) == 0).astype(int)
    prob = np.asarray(prob_class0, dtype=float)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "AUC": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "PR_AUC": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else np.nan,
        "Accuracy": float(accuracy_score(y, pred)),
        "Sensitivity_BC_LDS": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "Specificity_others": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "F1_BC_LDS": float(f1_score(y, pred, zero_division=0)),
        "Brier_score": float(np.mean((prob - y) ** 2)),
        "ECE_10_bins": ece(y, prob),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def ece(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    out = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob <= hi) if i == n_bins - 1 else (prob >= lo) & (prob < hi)
        if mask.any():
            out += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(out)


def bootstrap(rows: pd.DataFrame, n_bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    observed = metrics(rows.true_label.to_numpy(), rows.prediction.to_numpy(), rows.probability_class_0.to_numpy())
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(rows), len(rows))
        s = rows.iloc[idx]
        samples.append(metrics(s.true_label.to_numpy(), s.prediction.to_numpy(), s.probability_class_0.to_numpy()))
    boot = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "observed": observed[metric],
                "bootstrap_mean": float(boot[metric].mean()),
                "ci_lower_2.5": float(boot[metric].quantile(0.025)),
                "ci_upper_97.5": float(boot[metric].quantile(0.975)),
            }
            for metric in ["AUC", "PR_AUC", "Accuracy", "Sensitivity_BC_LDS", "Specificity_others", "F1_BC_LDS", "Brier_score", "ECE_10_bins"]
        ]
    )


def param_grid(grid_json: str) -> list[dict]:
    grid = json.loads(grid_json)
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Data-origin-matrix"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw_leakage_free_nested_cv"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-transcriptome", type=int, default=1000)
    parser.add_argument("--k-proteomic", type=int, default=1000)
    parser.add_argument("--k-metabolomic", type=int, default=500)
    parser.add_argument(
        "--grid",
        default='{"hidden_dim":[128],"num_layers":[1],"dropout":[0.3,0.5],"learning_rate":[0.001],"weight_decay":[0.0001],"num_heads":[8]}',
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, labels, sample_ids, features = load_raw_data(args.data_dir)
    feature_k = {
        "transcriptome": args.k_transcriptome,
        "proteomic": args.k_proteomic,
        "metabolomic": args.k_metabolomic,
    }
    grid = param_grid(args.grid)
    base_model = {"omics_dims": {}, "hidden_dim": 128, "num_classes": 2, "num_layers": 1, "dropout": 0.3, "num_heads": 8}
    base_train = {"epochs": args.epochs, "lr": 1e-3, "weight_decay": 1e-4, "patience": args.patience}

    rows = []
    selections = []
    for repeat in range(args.repeats):
        outer = StratifiedKFold(args.outer_folds, shuffle=True, random_state=args.seed + repeat)
        for outer_fold, (outer_train, outer_test) in enumerate(outer.split(np.zeros(len(labels)), labels), 1):
            inner = StratifiedKFold(args.inner_folds, shuffle=True, random_state=args.seed + 1000 + repeat * 100 + outer_fold)
            scored = []
            for grid_id, hp in enumerate(grid):
                aucs = []
                for inner_fold, (itr_rel, ival_rel) in enumerate(inner.split(np.zeros(len(outer_train)), labels[outer_train]), 1):
                    itr, ival = outer_train[itr_rel], outer_train[ival_rel]
                    mp = base_model | {k: hp[k] for k in ["hidden_dim", "num_layers", "dropout", "num_heads"] if k in hp}
                    tp = base_train | {"lr": hp.get("learning_rate", base_train["lr"]), "weight_decay": hp.get("weight_decay", base_train["weight_decay"])}
                    result = train_eval(data, labels, itr, ival, feature_k, mp, tp, args.seed + repeat * 10000 + outer_fold * 100 + inner_fold * 10 + grid_id)
                    aucs.append(metrics(result["true_labels"], result["predictions"], result["probabilities"][:, 0])["AUC"])
                scored.append({"grid_id": grid_id, "hyperparameters": hp, "mean_inner_auc": float(np.nanmean(aucs))})

            best = max(scored, key=lambda x: x["mean_inner_auc"])
            selections.append({"repeat": repeat + 1, "outer_fold": outer_fold, **best})
            hp = best["hyperparameters"]
            mp = base_model | {k: hp[k] for k in ["hidden_dim", "num_layers", "dropout", "num_heads"] if k in hp}
            tp = base_train | {"lr": hp.get("learning_rate", base_train["lr"]), "weight_decay": hp.get("weight_decay", base_train["weight_decay"])}
            result = train_eval(data, labels, outer_train, outer_test, feature_k, mp, tp, args.seed + repeat * 10000 + outer_fold)
            for local_i, sample_index in enumerate(outer_test):
                probs = result["probabilities"][local_i]
                rows.append(
                    {
                        "repeat": repeat + 1,
                        "outer_fold": outer_fold,
                        "sample_index": int(sample_index),
                        "sample_id": sample_ids[sample_index],
                        "true_label": int(result["true_labels"][local_i]),
                        "prediction": int(result["predictions"][local_i]),
                        "probability_class_0": float(probs[0]),
                        "probability_class_1": float(probs[1]),
                    }
                )
            pd.DataFrame(rows).to_csv(args.output_dir / "raw_nested_oof_predictions_partial.csv", index=False)
            pd.DataFrame(selections).to_csv(args.output_dir / "raw_nested_hyperparameter_selection_partial.csv", index=False)

    pred = pd.DataFrame(rows)
    pred.to_csv(args.output_dir / "raw_nested_oof_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(args.output_dir / "raw_nested_hyperparameter_selection.csv", index=False)
    summary = bootstrap(pred, args.bootstrap, args.seed)
    summary.to_csv(args.output_dir / "raw_nested_bootstrap_metric_ci.csv", index=False)
    observed_metrics = dict(zip(summary["metric"], summary["observed"]))
    report = {
        "n_samples_per_repeat": len(labels),
        "total_outer_predictions": len(pred),
        "class_distribution": {"BC-LDS_label_0": int((labels == 0).sum()), "other_label_1": int((labels == 1).sum())},
        "raw_feature_counts": {k: int(v.shape[1]) for k, v in data.items()},
        "fold_internal_selected_features": feature_k,
        "repeats": args.repeats,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "hyperparameter_grid": grid,
        "observed_metrics": {k: float(v) for k, v in observed_metrics.items()},
        "preprocessing": "Within each train split: drop all-missing/nonconstant train features, zero substitution, ANOVA F-test selection, MinMax scaling. Test split is transformed only.",
    }
    (args.output_dir / "raw_nested_cv_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
