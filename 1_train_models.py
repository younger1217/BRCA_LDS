import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))

from modules.data_preprocessing import MultiOmicsDataProcessor
from modules.model_training import CrossValidator, MultiOmicsGATModel


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiOmicsModelTrainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config["output_dir"]
        set_random_seeds(config.get("random_state", 42))
        for subdir in ["data/processed", "cross_validation_results", "reports"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def run(self):
        processor = MultiOmicsDataProcessor(
            scaler_type=self.config.get("scaler_type", "none"),
            feature_selection_k=self.config.get("feature_selection_k", None),
            imbalance_method=self.config.get("imbalance_method", "class_weight"),
        )
        omics_data, labels = processor.load_data(self.config["data_path"])
        omics_data = processor.preprocess_data(omics_data, labels)
        cv_splits = processor.create_cross_validation_splits(
            labels,
            n_splits=self.config.get("cv_folds", 5),
            random_state=self.config.get("random_state", 42),
        )
        processor.save_preprocessed_data(omics_data, labels, os.path.join(self.output_dir, "data", "processed"))

        model_config = {
            "omics_dims": {omics_type: data.shape[1] for omics_type, data in omics_data.items()},
            "hidden_dim": self.config.get("hidden_dim", 128),
            "num_classes": self.config.get("num_classes", 2),
            "num_layers": self.config.get("num_layers", 1),
            "dropout": self.config.get("dropout", 0.47),
            "num_heads": self.config.get("num_heads", 8),
        }
        train_params = {
            "epochs": self.config.get("epochs", 100),
            "lr": self.config.get("learning_rate", 1e-3),
            "weight_decay": self.config.get("weight_decay", 1e-4),
            "patience": self.config.get("patience", 50),
        }
        cv = CrossValidator(MultiOmicsGATModel, model_config, save_dir=os.path.join(self.output_dir, "cross_validation_results"))
        cv_results = cv.run_cross_validation(omics_data, labels, cv_splits, train_params, data_processor=processor)
        self.save_summary(cv_results, labels, omics_data, processor)
        return cv_results

    def save_summary(self, cv_results, labels, omics_data, processor):
        report_config = dict(self.config)
        report_config["output_dir"] = "<output_dir>"
        summary = {
            "experiment_info": {
                "timestamp": "reproducible_run",
                "config": report_config,
                "random_seed": self.config.get("random_state", 42),
            },
            "data_info": {
                "n_samples": len(labels),
                "n_classes": len(np.unique(labels)),
                "class_distribution": np.bincount(labels).tolist(),
                "omics_types": list(omics_data.keys()),
                "feature_counts": {omics_type: data.shape[1] for omics_type, data in omics_data.items()},
                "imbalance_info": processor.get_class_imbalance_info(),
            },
            "model_performance": {
                "cross_validation": {
                    "mean_accuracy": float(cv_results["mean_accuracy"]),
                    "std_accuracy": float(cv_results["std_accuracy"]),
                    "mean_auc": float(cv_results["mean_auc"]) if cv_results["mean_auc"] else None,
                    "std_auc": float(cv_results["std_auc"]) if cv_results["std_auc"] else None,
                }
            },
            "fold_details": [
                {
                    "fold": result["fold"],
                    "accuracy": float(result["accuracy"]),
                    "auc_score": float(result["auc_score"]) if result["auc_score"] else None,
                }
                for result in cv_results["fold_results"]
            ],
        }
        with open(os.path.join(self.output_dir, "reports", "training_summary.json"), "w", encoding="utf-8") as f:
            json.dump(self._json_ready(summary), f, indent=2)

    def _json_ready(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {key: self._json_ready(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._json_ready(value) for value in obj]
        return obj


def load_config(config_file):
    spec = importlib.util.spec_from_file_location("config", config_file)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def main():
    parser = argparse.ArgumentParser(description="Train the MOGAT multi-omics classifier.")
    parser.add_argument("--config", default="example_config.py")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["epochs"] = args.epochs

    results = MultiOmicsModelTrainer(config).run()
    print(json.dumps({
        "mean_accuracy": results["mean_accuracy"],
        "std_accuracy": results["std_accuracy"],
        "mean_auc": results["mean_auc"],
        "std_auc": results["std_auc"],
    }, indent=2))


if __name__ == "__main__":
    main()
