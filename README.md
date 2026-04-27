# Ames House Prices — House Price Prediction

> A low-level machine learning project built from scratch using **NumPy** and **Pandas**.
The goal is to predict residential house sale prices in Ames, Iowa, by implementing
the full data science pipeline from raw data to cross-validated regression models without relying on high-level ML libraries such as scikit-learn for modelling or preprocessing.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-ffffff?style=flat-square&logo=matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-2.4-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458?style=flat-square&logo=pandas&logoColor=white)

---

## Project Motivation

Most introductory data science projects use high level libraries built around pre-built estimators.
This project takes the opposite approach: every statistical concept is implemented manually,
from chi-squared MCAR tests and one-hot encoding to Lasso regression and stratified
cross-validation. The objective is not to maximise predictive performance, but to build
a deep understanding of each step in the pipeline.

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [Ames Housing Dataset — Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| **Original features** | 82 (numeric + categorical) |
| **Observations (after cleaning)** | 2,183 |
| **Target variable** | `price` — residential sale price in USD |
| **Geographic scope** | Ames, Iowa, USA |

---

## Repository Structure

```
ames_house_prices/
│
├── data/
│   ├── processed/
│   └── raw/
│       ├── ames_housing_test.csv
│       ├── ames_housing_train.csv    # The only one used in this project.
│       └── ames_housing.csv
│
├── ames_house_prics.ipynb    # Main analysis notebook.
├── ames_house_prics.html     # For easy vidualization.
│
├── data/
│   ├── processing.py     # Data cleaning & encoding functions.
│   ├── analysis.py       # EDA, visualization & statistical functions.
│   ├── model.py          # CV engine & regression models implementations.
|   └── config.py         # Constants and paths
│
├── pyproject.toml    # UV config file.
├── uv.lock           # Libraries specifications.
├── .python_version
|
├── LICENSE
└── README.md
```

---

## Pipeline

The notebook follows a sequential, fully documented pipeline across 8 sections:

```
1. Import & Layout Cleaning
        ↓
2. Visual Analysis
        ↓
3. Function Definitions (moved to modules)
        ↓
3. Data Cleaning
   ├── Quick dataset pruning
   ├── MCAR chi-squared assessment
   ├── MICE imputation
   └── Feasibility analysis
        ↓
4. Pre-processing
   ├── Feature engineering
   ├── Outlier handling
   └── Correlation pruning
        ↓
5. EDA — Exploratory Data Analysis
   └── Custom scatter_2d with linear & quadratic fit lines
        ↓
6. Encoding
   ├── One-hot encoding
   └── Ordinal encoding
        ↓
7. Modeling & Evaluation
   └── Stratified K-Fold CV × 4 dataframes × 4 models
```

---

## Models

All models are implemented from scratch in NumPy and follow a unified contract
accepted by the CV engine:

```python
model_fn(X_train, y_train, X_test) → (y_pred, logs)
```

| Model | Method | Key Parameter |
|---|---|---|
| **OLS** | Normal equation: `β = (XᵀX)⁻¹Xᵀy` |
| **Ridge** | Regularized normal equation: `β = (XᵀX + λI)⁻¹Xᵀy` | `lambda_` |
| **Gradient Descent Linreg** | Iterative weight updates with early stopping | `alpha`, `max_iter`, `tolerance` |
| **Lasso Regression** | Coordinate Descent | `lambda_`, `max_iter` |

---

## Evaluation Strategy

Given the small dataset size (2,183 rows), a **stratified 10-fold cross-validation**
is used instead of a fixed train/test split, providing a more reliable performance
estimate than a single hold-out set.

**Stratification** is performed by binning `price` into quantile buckets before
splitting, ensuring every fold contains the same proportion of low, mid, and
high-value properties. Metrics are reported per fold with mean and standard deviation.

| Metric | What it measures |
|---|---|
| **RMSE** | Typical error magnitude, penalises large errors heavily |
| **MAE** | Robust error magnitude in USD terms |
| **R²** | Proportion of price variance explained by the model |
| **MAPE** | Scale-free percentage error, comparable across dataframe variants |

---

## Dataframe Variants

To assess the impact of outlier handling strategy on model performance,
all three models are evaluated on four versions of the same dataset:

| Variant | Transformation applied |
|---|---|
| `original` | No outlier treatment |
| `log` | Log1p applied to skewed variables (skewness > 2.5) |
| `winsor` | Numerical variables capped at the 97th percentile |
| `iqr` | Numerical variables capped at Q3 + 1.5 × IQR |

---

## Custom Implementations

The following components are built from scratch with raw math an linear algebra without ML library wrappers:

- **MCAR test** — chi-squared independence test for missing data assessment
- **One-hot encoding** — binary dummy variable generation with reference category control
- **Ordinal encoding** — hand-defined integer mappings for quality/condition scales
- **Normalization** — min-max scaling to arbitrary output range
- **Stratified K-Fold CV** — quantile-binned fold generation with shuffle
- **OLS regression** — closed-form normal equation
- **Ridge regression** — L2-regularized closed-form solution
- **Gradient Descent Linear regression** — iterative solver with early stopping and loss logging
- **Lasso regression** — L1-regularized open-for solution
- **`scatter_2d()`** — fully custom scatter plot with color, size, alpha, shape aesthetics and stacked legends

---

## Known Limitations

- **Preprocessing leakage:** cleaning and encoding decisions (column dropping thresholds, imputation, encoding maps) were derived from the full 2,183-row dataset rather than from training folds only. In a production system, each preprocessing step would be fit on the training fold and applied to the test fold at each CV iteration.

- **No external test set:** the original Kaggle test CSV withholds sale prices and cannot be used for local evaluation. The stratified 10-fold CV is used as a substitute.

- **MICE imputation uses scikit-learn:** `sklearn.impute.IterativeImputer` is the only external ML tool used in the pipeline. All other statistical and modelling logic is implemented in raw NumPy/Pandas. In future a custom low level MICE imputer will be developed from scratch and used in the project.

---

## Requirements

```
python >= 3.10
numpy
pandas
matplotlib
scipy
scikit-learn
```

---

## Installation

The project is built with python 3.12 using `uv environment manager` for speed and reproduceability. The package requirements are established in the `pyproject.toml` file. If using another environment manager please make sure to check out the file and use the right package versions.

The following instructions are for `Ubuntu 24.04.4`, if you are using Windows or Mac, please consult the official `uv` documentation.

Install instructions:

- Navigate to the project folder:
```Bash
cd /your-path/ames_house_prices
```

- Create the `uv.lock` file to tell the system which packages to install:
```Bash
uv lock
```

- Synchronize the environment (here `uv` will actually download the packages):
```Bash
uv sync
```

Download the [Ames House Prices dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), place the train dataset in `/data/raw` and call it exactly `ames_housing_train.csv`.

---

## Author

Built as a self-directed learning project with the explicit goal of understanding
the mathematical and statistical foundations of the data science pipeline.
Every implementation decision is documented inline in the notebook.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Dataset: [Ames House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) by Anna Montoya on Kaggle.