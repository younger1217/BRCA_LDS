# Reproducibility Release Manifest

This manifest lists the minimum files required for a versioned public release
supporting the manuscript analyses and reviewer-requested validation.

## Code

```text
1_train_models.py
2_gradient_shap_interpretability.py
modules/data_preprocessing.py
modules/model_training.py
reproducibility/raw_leakage_free_nested_cv.py
reproducibility/raw_leakage_free_permutation_test.py
reproducibility/README.md
```

## Data or Access Instructions

```text
Data/Data-origin-matrix/gene_FPKM_matrix.csv
Data/Data-origin-matrix/ProteinTable_matrix.csv
Data/Data-origin-matrix/metabolites_matrix.csv
Data/Data_preprocessed/1_tr.csv
Data/Data_preprocessed/1_featname.csv
Data/Data_preprocessed/2_tr.csv
Data/Data_preprocessed/2_featname.csv
Data/Data_preprocessed/3_tr.csv
Data/Data_preprocessed/3_featname.csv
Data/Data_preprocessed/labels_tr.csv
Data/Data_preprocessed/labels_tr_4class.csv
```


## Result Files to Include

```text
results/raw_leakage_free_nested_cv/raw_nested_cv_report.json
results/raw_leakage_free_nested_cv/raw_nested_oof_predictions.csv
results/raw_leakage_free_nested_cv/raw_nested_bootstrap_metric_ci.csv
results/raw_leakage_free_nested_cv/raw_nested_hyperparameter_selection.csv
results/raw_leakage_free_permutation/raw_permutation_test_summary.json
results/raw_leakage_free_permutation/raw_observed_predictions.csv
results/raw_leakage_free_permutation/raw_permutation_summary.csv
```


## Documentation

```text
README.md
requirements.txt
example_config.py
RELEASE_MANIFEST.md
```
