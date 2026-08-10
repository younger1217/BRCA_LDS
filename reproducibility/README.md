# Reproducibility Package

This folder contains scripts for leakage-free validation, preprocessing
reproducibility, permutation testing, and model-interpretation transparency.

## Purpose

The scripts start from the original full feature matrices rather than from the
previously preselected 2500-feature matrices. In each outer training split, the
following operations are fitted using training data only and then applied to the
held-out test split:

- missing-value zero substitution
- nonconstant feature filtering
- ANOVA feature selection
- scaling/normalization
- model training and prediction

This design prevents supervised feature selection or preprocessing parameters
from being estimated using held-out test samples.

## Included Scripts

```text
raw_leakage_free_nested_cv.py
raw_leakage_free_permutation_test.py
```

`raw_leakage_free_nested_cv.py` performs repeated nested cross-validation from
the raw matrices.

`raw_leakage_free_permutation_test.py` repeats the full fold-internal
preprocessing, ANOVA selection, model training, and testing workflow after label
permutation.

## Expected Raw Input Files

Raw matrices are included in `Data/Data-origin-matrix/`. You can provide another
path with `--data-dir`.

```text
gene_FPKM_matrix.csv
ProteinTable_matrix.csv
metabolites_matrix.csv
```

The scripts align participants across omics layers using the shared sample
identifier suffix in the matrix column names.

## Main Validation Command

```bash
python raw_leakage_free_nested_cv.py \
  --data-dir Data/Data-origin-matrix \
  --output-dir results/raw_leakage_free_nested_cv \
  --repeats 3 \
  --outer-folds 5 \
  --inner-folds 3 \
  --epochs 50 \
  --patience 20 \
  --bootstrap 2000 \
  --k-transcriptome 1000 \
  --k-proteomic 1000 \
  --k-metabolomic 500
```

Main outputs:

```text
raw_nested_cv_report.json
raw_nested_oof_predictions.csv
raw_nested_bootstrap_metric_ci.csv
raw_nested_cluster_bootstrap_metric_ci.csv
```

## Main Permutation-Test Command

```bash
python raw_leakage_free_permutation_test.py \
  --data-dir Data/Data-origin-matrix \
  --output-dir results/raw_leakage_free_permutation \
  --n-permutations 100 \
  --n-splits 5 \
  --epochs 50 \
  --patience 20 \
  --k-transcriptome 1000 \
  --k-proteomic 1000 \
  --k-metabolomic 500
```
