from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from modules.model_training import MultiOmicsGATModel


class GradientSHAPInterpreter:
    def __init__(self, results_dir: str, random_seed: int = 42):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "gradient_shap_interpretability"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def run(self, top_n: int = 30, n_shap_samples: int = 200) -> None:
        omics_data, feature_names, labels = self.load_processed_data()
        models = self.load_models()
        fold_results = []
        for fold_id, model_info in models.items():
            np.random.seed(self.random_seed + fold_id)
            torch.manual_seed(self.random_seed + fold_id)
            model = self.rebuild_model(model_info)
            test_indices = model_info["metadata"]["test_indices"]
            fold_data = {omics_type: data[test_indices] for omics_type, data in omics_data.items()}
            tensors = {omics_type: torch.tensor(data, dtype=torch.float32) for omics_type, data in fold_data.items()}
            fold_results.append(self.compute_fold_shap(model, tensors, fold_data, test_indices, len(np.unique(labels)), n_shap_samples))
        shap_results = self.aggregate_folds(fold_results)
        self.save_arrays(shap_results)
        self.save_rankings(shap_results, feature_names, top_n)

    def load_processed_data(self) -> tuple[dict[str, np.ndarray], dict[str, list[str]], np.ndarray]:
        data_dir = self.results_dir / "data" / "processed"
        omics_types = ["transcriptome", "proteomic", "metabolomic"]
        data, names = {}, {}
        for omics_type in omics_types:
            data[omics_type] = pd.read_csv(data_dir / f"{omics_type}_processed.csv", header=None).values
            names[omics_type] = pd.read_csv(data_dir / f"{omics_type}_features.csv", header=None).iloc[:, 0].astype(str).tolist()
        labels = pd.read_csv(data_dir / "labels.csv", header=None).values.flatten().astype(int)
        return data, names, labels

    def load_models(self) -> dict[int, dict]:
        models = {}
        for model_path in sorted((self.results_dir / "cross_validation_results" / "models").glob("fold_*_model.pth")):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            fold_id = int(model_path.stem.split("_")[1])
            models[fold_id] = {"checkpoint": checkpoint, "metadata": checkpoint.get("metadata", {})}
        return models

    def rebuild_model(self, model_info: dict) -> MultiOmicsGATModel:
        model = MultiOmicsGATModel(**model_info["checkpoint"]["model_config"])
        model.load_state_dict(model_info["checkpoint"]["model_state_dict"])
        model.eval()
        return model

    def compute_fold_shap(self, model: MultiOmicsGATModel, tensors: dict[str, torch.Tensor], fold_data: dict[str, np.ndarray], test_indices: list[int], n_classes: int, n_samples: int) -> dict:
        class_shap = [self.compute_gradient_shap_per_sample(model, tensors, target_class, n_samples) for target_class in range(n_classes)]
        return {
            "sample_indices": test_indices,
            "omics_shap_values": {omics_type: np.mean([item[omics_type]["shap_values"] for item in class_shap], axis=0) for omics_type in fold_data},
            "omics_feature_values": fold_data,
        }

    def compute_gradient_shap_per_sample(self, model: MultiOmicsGATModel, data: dict[str, torch.Tensor], target_class: int, n_samples: int) -> dict:
        shap_results = {}
        model.eval()
        for omics_type, omics_data in data.items():
            sample_values = []
            for sample_idx in range(omics_data.shape[0]):
                sample_data = omics_data[sample_idx : sample_idx + 1]
                shap_samples = []
                for _ in range(n_samples):
                    baseline = torch.randn_like(sample_data) * omics_data.std()
                    alpha = torch.rand(1).item()
                    interpolated = baseline + alpha * (sample_data - baseline)
                    interpolated.requires_grad_(True)
                    full_input = {key: (interpolated if key == omics_type else value[sample_idx : sample_idx + 1]) for key, value in data.items()}
                    target_score = model(full_input)[0, target_class]
                    grad = torch.autograd.grad(target_score, interpolated, retain_graph=False, create_graph=False)[0]
                    shap_samples.append(((sample_data - baseline).cpu().numpy() * grad.cpu().numpy())[0])
                sample_values.append(np.mean(shap_samples, axis=0))
            shap_results[omics_type] = {"shap_values": np.array(sample_values), "feature_values": omics_data.cpu().numpy()}
        return shap_results

    def aggregate_folds(self, fold_results: list[dict]) -> dict:
        omics_types = list(fold_results[0]["omics_shap_values"])
        return {
            "omics_shap_values": {omics_type: np.vstack([fold["omics_shap_values"][omics_type] for fold in fold_results]) for omics_type in omics_types},
            "omics_feature_values": {omics_type: np.vstack([fold["omics_feature_values"][omics_type] for fold in fold_results]) for omics_type in omics_types},
        }

    def save_arrays(self, shap_results: dict) -> None:
        for omics_type in shap_results["omics_shap_values"]:
            np.save(self.output_dir / f"{omics_type}_shap_values.npy", shap_results["omics_shap_values"][omics_type])
            np.save(self.output_dir / f"{omics_type}_feature_values.npy", shap_results["omics_feature_values"][omics_type])

    def save_rankings(self, shap_results: dict, feature_names: dict[str, list[str]], top_n: int) -> None:
        rows = []
        for omics_type, shap_values in shap_results["omics_shap_values"].items():
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            for idx, importance in enumerate(mean_abs):
                rows.append({"feature_name": feature_names[omics_type][idx], "omics_type": omics_type, "mean_abs_shap": importance})
        rows.sort(key=lambda row: row["mean_abs_shap"], reverse=True)
        total = sum(row["mean_abs_shap"] for row in rows)
        ranked = pd.DataFrame([{**row, "rank": rank, "percentage": row["mean_abs_shap"] / total * 100} for rank, row in enumerate(rows, 1)])
        ranked = ranked[["rank", "feature_name", "omics_type", "mean_abs_shap", "percentage"]]
        ranked.to_csv(self.output_dir / "complete_feature_ranking.csv", index=False, encoding="utf-8-sig")
        ranked.head(top_n).to_csv(self.output_dir / f"top{top_n}_features_shap.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gradient SHAP interpretability analysis.")
    parser.add_argument("--results-dir", "--results_dir", dest="results_dir", default="results/mogat_internal_cv")
    parser.add_argument("--top-n", "--top_n", dest="top_n", type=int, default=30)
    parser.add_argument("--n-shap-samples", "--n_shap_samples", dest="n_shap_samples", type=int, default=200)
    parser.add_argument("--random-seed", "--random_seed", dest="random_seed", type=int, default=42)
    args = parser.parse_args()
    GradientSHAPInterpreter(args.results_dir, args.random_seed).run(args.top_n, args.n_shap_samples)


if __name__ == "__main__":
    main()
