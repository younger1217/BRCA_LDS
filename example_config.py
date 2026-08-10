"""Configuration for MOGAT model training and Gradient SHAP analysis."""

config = {
    "data_path": "Data/Data_preprocessed",
    "output_dir": "results/mogat_internal_cv",
    "num_classes": 2,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.47,
    "num_heads": 8,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patience": 50,
    "scaler_type": "none",
    "feature_selection_k": None,
    "imbalance_method": "class_weight",
    "cv_folds": 5,
    "random_state": 42,
    "interpretability_methods": ["gradient_shap"],
}
