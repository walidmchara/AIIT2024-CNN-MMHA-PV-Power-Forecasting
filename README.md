# CNN–MMHA for Photovoltaic Power Forecasting

Research implementation accompanying **“A Photovoltaic Power Prediction Model for Solar Electric Vehicles Using Convolutional Neural Networks and Masked Multi-Head Attention Mechanism.”** 2025 IEEE AIIT Conference. [DOI: 10.1109/AIIT63112.2025.11082930](https://doi.org/10.1109/AIIT63112.2025.11082930).

## Method

The predictive architecture integrates a learnable spatial transformation, three convolutional layers and causal masked multi-head self-attention. Nine meteorological inputs are projected to a 7×7 spatial representation for each historical hour. Convolutional processing extracts local feature interactions, while a 20-head attention block models temporal dependencies before one-step-ahead PV-power regression. Fuzzy c-means identifies four atmospheric regimes for diagnostic evaluation.

The implementation retains Table I's 9×49 spatial mapping, 128/256/10 convolutional channels, 100-dimensional embeddings, 20 heads, 100→50→100 feed-forward layers, batch size 32 and learning rate 0.001. [Architecture and protocol notes](docs/ARCHITECTURE_AND_VALIDATION.md) distinguish these published settings from the implementation choices requiring further validation.

## Dataset

The study uses hourly weather observations from Borj Cedria, Tunisia (2015–2022) and photovoltaic output from a 1 kWc monocrystalline module. The data interface and physical units are in [data/SCHEMA.md](data/SCHEMA.md). The paper's original measurements are not redistributed in this repository.

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
python -m pytest -q
```

For an end-to-end software check on explicitly synthetic signals:

```bash
python train.py --smoke --epochs 1 --out outputs/smoke
```

To fit and evaluate the model using measured hourly photovoltaic data:

```bash
python train.py --csv path/to/hourly_pv.csv --epochs 1000 --out outputs/borj_cedria
```

Training outputs include a model checkpoint, epoch history, timestamp-aligned test predictions, overall metrics and four weather-regime diagnostics. The five comparison families (MLP, LSTM, BiLSTM, CNN–LSTM, CNN–BiLSTM) are available through `--architecture`, while `cnn_mmha` is the default. Baseline hyperparameters are reference choices and require a separate fairness audit against the conference experiments. Visualize predictions with `python scripts/plot_forecasts.py outputs/borj_cedria`.

## Reproducibility

Normalization and fuzzy clustering are fitted on training observations only. A chronological 80/20 development/test split is used, with a validation subset taken from development. The article's reported split is by weather regime, and this difference is explicitly documented. One-step forecasting is constructed so that inputs at hour `t` predict the target at `t+1`, without using future weather measurements.

Software tests and synthetic demo runs confirm implementation behavior; they do not establish agreement with the paper's reported experimental results. Full benchmark reproduction requires the source observations and the missing preprocessing/split details.
