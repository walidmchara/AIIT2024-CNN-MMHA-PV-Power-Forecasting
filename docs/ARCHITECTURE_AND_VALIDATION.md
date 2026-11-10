# Architecture and scientific alignment

## Source

**W. Mchara et al.**, “A Photovoltaic Power Prediction Model for Solar Electric Vehicles Using Convolutional Neural Networks and Masked Multi-Head Attention Mechanism,” *2025 International Conference on Innovation in Artificial Intelligence and Internet of Things (AIIT)*. DOI: [10.1109/AIIT63112.2025.11082930](https://doi.org/10.1109/AIIT63112.2025.11082930).

The repository name retains `AIIT2024` from the previously agreed GitHub naming convention, but the proceedings and DOI explicitly identify the **2025** conference. This discrepancy is confined to the repository slug.

## Implemented pipeline

1. Nine meteorological predictors at each historical hour pass through a learnable **9→49** affine projection, folded into a **7×7** spatial map.
2. A time-distributed CNN applies **128, 256, 10** output channels, with **3×3, 3×3, 1×1** kernels and stride 2 in the first two layers.
3. Each spatial output is flattened and projected to a **100-dimensional** temporal embedding.
4. Sinusoidal positional encoding is added, followed by **20-head causal masked self-attention**.
5. Add-and-normalize and **100→50→100** feed-forward blocks generate temporal representations. The final historical token predicts one next-step PV-power sample.
6. **FCM with four clusters** is implemented for weather-regime assignment and regime-specific diagnostic metrics. Cluster centers are trained on historical training data only.

The architecture dimensions, training batch size 32, learning rate 0.001, MSE and 1000 epochs are reported in Table I of the paper. Values configurable without being claimed as published include **lookback 24**, early-stopping patience 60 and a train/validation/test chronology of **64/16/20**. The paper describes an 80/20 partition within each weather category, not the stricter held-out chronological test chosen here; this difference must be reconciled for exact numerical comparisons.

## Scientific limitations

The article's original PV output time series and complete preprocessing code are not provided. The project is a technical implementation consistent with the disclosed component sequence and published hyperparameters, not an exact reproduction of the authors' experimental files or reported accuracy. Specific uncertainties include the exact look-back, any CNN pooling details, the treatment of FCM during training, the source of the weather-category labels, feature-selection criteria and long-gap imputation policy.

The correlation analysis is calculated on training records only. Min–max parameters are learned only on training records, preventing distribution look-ahead. MAPE is reported above a configurable near-zero power threshold, with the threshold and count explicit in `metrics.json`. Daylight selection and zero-power handling should match the paper before any benchmark claims.

## Output files

An experiment produces `checkpoint.pt`, `training_history.csv`, `test_predictions.csv` and `metrics.json`. Each metric belongs to the current run; reported article results are not copied or invented.
