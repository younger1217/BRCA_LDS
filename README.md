# MOGAT for BC-LDS Multi-Omics Analysis

This repository provides the code used for the internal MOGAT classification and Gradient SHAP interpretability analyses in the revised manuscript on breast cancer-associated Liver depression syndrome (BC-LDS).

The code is intended for reproducible exploratory modelling, not for clinical diagnosis or externally validated prediction.

## Repository Structure

```text
.
├── 1_train_models.py
├── 2_gradient_shap_interpretability.py
├── example_config.py
├── requirements.txt
└── modules/
    ├── data_preprocessing.py
    └── model_training.py
```

## Input Files

Place the processed multi-omics input files in `data/processed_multiomics/` or provide another path with `--data-path`.

```text
1_tr.csv          transcriptomic matrix
1_featname.csv    transcriptomic feature names
2_tr.csv          proteomic matrix
2_featname.csv    proteomic feature names
3_tr.csv          metabolomic matrix
3_featname.csv    metabolomic feature names
labels_tr.csv     binary labels
```

Label coding:

```text
0 = BC-LDS
1 = comparator participants
```

Rows must be matched across all omics matrices and the label file.

## Installation

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python 1_train_models.py --config example_config.py
```

With explicit paths:

```bash
python 1_train_models.py --config example_config.py --data-path /path/to/processed_multiomics --output-dir results/mogat_internal_cv
```

Main outputs:

```text
results/mogat_internal_cv/cross_validation_results/
results/mogat_internal_cv/models/
results/mogat_internal_cv/reports/training_summary.json
```

## Run Gradient SHAP

Run this step after model training:

```bash
python 2_gradient_shap_interpretability.py --results-dir results/mogat_internal_cv --top_n 30 --n_shap_samples 200
```

Main outputs:

```text
results/mogat_internal_cv/gradient_shap_interpretability/complete_feature_ranking.csv
results/mogat_internal_cv/gradient_shap_interpretability/top30_features_shap.csv
results/mogat_internal_cv/gradient_shap_interpretability/shap_beeswarm_top30.png
results/mogat_internal_cv/gradient_shap_interpretability/shap_beeswarm_top30_hq.pdf
```

## Notes

This public release contains only the MOGAT internal classification and Gradient SHAP interpretation workflow corresponding to the manuscript. Raw clinical or omics data are not included; users should place the processed matrices in the expected input format before running the scripts.
