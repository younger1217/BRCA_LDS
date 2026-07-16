import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold


class MultiOmicsDataProcessor:
    def __init__(self, scaler_type="none", feature_selection_k=None, imbalance_method="class_weight"):
        self.scaler_type = scaler_type
        self.feature_selection_k = feature_selection_k
        self.imbalance_method = imbalance_method
        self.feature_names = {}
        self.omics_types = ["transcriptome", "proteomic", "metabolomic"]
        self.class_distribution_original = None

    def load_data(self, data_path):
        data_dir = Path(data_path)
        omics_data = {}
        for i, omics_type in enumerate(self.omics_types, 1):
            data_file = data_dir / f"{i}_tr.csv"
            feature_file = data_dir / f"{i}_featname.csv"
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")
            data_matrix = pd.read_csv(data_file, header=None).values
            omics_data[omics_type] = data_matrix
            if feature_file.exists():
                self.feature_names[omics_type] = pd.read_csv(feature_file, header=None).iloc[:, 0].astype(str).tolist()
            else:
                self.feature_names[omics_type] = [f"{omics_type}_feature_{j}" for j in range(data_matrix.shape[1])]

        label_file = data_dir / "labels_tr.csv"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        labels = pd.read_csv(label_file, header=None).values.flatten().astype(int)
        self.class_distribution_original = np.bincount(labels)
        return omics_data, labels

    def preprocess_data(self, omics_data, labels):
        return {omics_type: self._quality_control(data, omics_type) for omics_type, data in omics_data.items()}

    def preprocess_fold_data(self, train_data, test_data, train_labels, fold_id=None):
        processed_train_data = {}
        processed_test_data = {}
        for omics_type in train_data:
            train_omics = train_data[omics_type].copy()
            test_omics = test_data[omics_type].copy()
            if self.feature_selection_k is not None:
                selector = SelectKBest(f_classif, k=min(self.feature_selection_k, train_omics.shape[1]))
                train_omics = selector.fit_transform(train_omics, train_labels)
                test_omics = selector.transform(test_omics)
            processed_train_data[omics_type] = train_omics
            processed_test_data[omics_type] = test_omics
        return processed_train_data, processed_test_data

    def _quality_control(self, data, omics_type):
        if np.isnan(data).any():
            column_means = np.nanmean(data, axis=0)
            column_means = np.where(np.isnan(column_means), 0, column_means)
            data = np.where(np.isnan(data), column_means, data)
        zero_var_mask = np.var(data, axis=0) == 0
        if zero_var_mask.any():
            data = data[:, ~zero_var_mask]
            self.feature_names[omics_type] = [name for i, name in enumerate(self.feature_names[omics_type]) if not zero_var_mask[i]]
        return data

    def get_class_imbalance_info(self):
        if self.class_distribution_original is None:
            return {}
        total = int(np.sum(self.class_distribution_original))
        imbalance_ratio = float(np.max(self.class_distribution_original) / np.min(self.class_distribution_original))
        return {
            "original_distribution": self.class_distribution_original.tolist(),
            "class_proportions": (self.class_distribution_original / total).tolist(),
            "imbalance_ratio": imbalance_ratio,
            "total_samples": total,
            "is_imbalanced": bool(imbalance_ratio > 1.5),
            "imbalance_method_used": self.imbalance_method,
        }

    def create_cross_validation_splits(self, labels, n_splits=5, random_state=42):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(len(labels)), labels))

    def get_data_statistics(self, omics_data, labels):
        return {
            "n_samples": len(labels),
            "n_classes": len(np.unique(labels)),
            "class_distribution": np.bincount(labels).tolist(),
            "omics_info": {
                omics_type: {
                    "n_features": data.shape[1],
                    "mean_value": float(np.mean(data)),
                    "std_value": float(np.std(data)),
                    "min_value": float(np.min(data)),
                    "max_value": float(np.max(data)),
                }
                for omics_type, data in omics_data.items()
            },
            "imbalance_info": self.get_class_imbalance_info(),
        }

    def save_preprocessed_data(self, omics_data, labels, output_path):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for omics_type, data in omics_data.items():
            pd.DataFrame(data).to_csv(output_dir / f"{omics_type}_processed.csv", index=False, header=False)
            pd.DataFrame(self.feature_names[omics_type]).to_csv(output_dir / f"{omics_type}_features.csv", index=False, header=False)
        pd.DataFrame(labels).to_csv(output_dir / "labels.csv", index=False, header=False)
        (output_dir / "data_statistics.json").write_text(json.dumps(self.get_data_statistics(omics_data, labels), indent=2), encoding="utf-8")
