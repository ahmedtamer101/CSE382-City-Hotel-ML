# CSE382 — City Hotel Booking Demand

A modular repository for the **CSE382 — Introduction to Machine Learning** course project. The project will study City Hotel booking demand through regression, classification, and clustering while keeping data preparation, modeling, evaluation, and inference responsibilities separated.

> **Current phase:** Phase 0 repository initialization is complete. No data analysis, preprocessing experiment, model training, tuning, evaluation, clustering, or prediction workflow has been implemented yet.

## Project objectives

The planned work is to:

- understand and validate the City Hotel booking data;
- build reproducible preprocessing and feature-engineering pipelines;
- predict average daily rate (`ADR`) with regression;
- predict booking cancellation (`IsCanceled`) with classification;
- segment bookings with unsupervised clustering; and
- evaluate, compare, document, and reproduce the resulting workflows.

## Dataset

The project uses **City Hotel H2.csv**, which contains 79,330 records and 31 source variables. The configured local path is `data/H2.csv`.

Official source: [Hotel Booking Demand — H2.csv](https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv)

The dataset is deliberately not included or downloaded during repository initialization. Local CSV files are ignored by Git. Each contributor should place an authorized copy at `data/H2.csv` when Milestone 1 work begins.

## Planned machine-learning tasks

| Task | Target or purpose | Status |
| --- | --- | --- |
| Regression | Predict `ADR` | Planned |
| Classification | Predict `IsCanceled` | Planned |
| Clustering | Discover booking segments | Planned |

## Repository structure

```text
CSE382-City-Hotel-ML/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── .gitignore
├── data/                    # Local datasets (not committed)
├── src/
│   ├── data/                # Future data-loading and validation code
│   ├── preprocessing/       # Future reusable preprocessing pipelines
│   ├── features/            # Future feature engineering and selection
│   ├── regression/          # Future regression workflows
│   ├── classification/      # Future classification workflows
│   ├── clustering/          # Future clustering workflows
│   ├── tuning/              # Future hyperparameter tuning
│   ├── evaluation/          # Future metrics and reporting utilities
│   └── inference/           # Future prediction interfaces
├── artifacts/               # Local trained/serialized objects
├── outputs/                 # Intended result tables and figures
├── reports/                 # Project reports
├── docs/                    # Supporting documentation
└── tests/                   # Automated tests
```

The `.gitkeep` files retain currently empty directories. Package `__init__.py` files establish importable Python modules without implementing project logic.

## Environment setup (Windows PowerShell)

Python 3.11 or newer is required. From the repository root:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks local activation scripts, use a process-scoped policy for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Runnable commands

The Phase 0 CLI supports help and version output only:

```powershell
python main.py --help
python main.py --version
```

Future commands are named in the help output, but are explicitly marked as pending. Invoking one exits without running ML work:

```powershell
python main.py analyze
```

Run the automated checks with:

```powershell
python -m pytest
```

Before installing development dependencies, the same tests can also run with Python's standard library:

```powershell
python -m unittest -v
```

## Team responsibilities

### Member 1 — Data Understanding and EDA

- Dataset inspection
- Data quality checks
- Exploratory analysis
- Plots and summary statistics

### Member 2 — Preprocessing and Feature Engineering

- Reusable preprocessing pipelines
- Encoding and scaling
- Feature selection support
- Train/validation/test splitting strategy
- Preprocessing leakage prevention

### Member 3 — Regression

- ADR prediction baseline
- Regression metrics and evaluation
- Regression-specific feature and leakage analysis

### Member 4 — Classification

- `IsCanceled` prediction baseline
- Classification metrics and evaluation
- Classification-specific feature and leakage analysis

### Member 5 — Clustering

- Unsupervised booking segmentation
- Initial clustering algorithm
- Clustering metrics
- Cluster visualization and interpretation

### Shared responsibilities

- GitHub collaboration
- Integration and code reviews
- Testing
- Report writing
- Reproducibility

## Git workflow

1. Update local `main` before starting work.
2. Create a short-lived branch named for the task, such as `feature/eda-summary` or `fix/data-validation`.
3. Keep commits focused and use descriptive imperative messages.
4. Do not commit datasets, environments, caches, trained models, or secrets.
5. Open a pull request into `main` with a concise summary and verification notes.
6. Request at least one teammate review; resolve discussions and automated test failures before merging.
7. Prefer small pull requests and coordinate before changing shared interfaces or configuration.

## Milestones

### Milestone 1 — due November 5, 2026

Milestone 1 will establish the first reproducible project workflows: dataset understanding and quality checks, exploratory analysis, leakage-aware preprocessing and data splitting, initial engineered features, and baseline regression, classification, and clustering work owned by the respective team members. Phase 0 provides only the repository foundation for that work.

### Milestone 2 — planned

Milestone 2 is expected to integrate and refine the three ML tracks through tuning, stronger evaluation and comparison, finalized visualizations and interpretations, saved artifacts where appropriate, prediction/inference support, tests, reproducibility checks, and final reporting. Exact deliverables should be aligned with the course specification before implementation.

## Configuration

Project-wide paths, logging defaults, and the random seed live in `config.yaml`. Paths are relative to the repository root so the project remains portable across contributors' machines.

## Data Understanding and Exploratory Data Analysis (EDA)

This module performs Data Understanding and Exploratory Data Analysis (EDA) on the original City Hotel dataset (`H2.csv`).

The analysis identifies missing values, duplicate records, unusual values, outliers, feature distributions, and correlations without modifying the original dataset.

### 1. Requirements

Install the required Python libraries from the project root:

```bash
pip install -r requirements.txt
```

For automated testing, ensure pytest is installed:

```bash
pip install pytest
```

### 2. Dataset Setup

Download the original City Hotel dataset (`H2.csv`) and place it inside the `data` directory.

Expected location:

```text
data/H2.csv
```

Dataset source:

https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv

**Important:**
- The dataset must contain the expected 31 columns.
- The original dataset contains 79,330 rows.
- The EDA script validates the dataset before running the analysis.
- The dataset is not downloaded automatically.
- The original dataset is not modified during analysis.

### 3. Run EDA

From the repository root, execute:

```bash
python -m src.data.eda --data data/H2.csv --out outputs/eda
```

Alternatively, use the default paths:

```bash
python -m src.data.eda
```

If `data/H2.csv` is missing, the script displays an error explaining where to place the dataset.

### 4. Generated EDA Outputs

The analysis generates CSV tables and figures inside:

```text
outputs/eda/
```

#### CSV Tables

- `column_profile.csv` — Data types, missing values, and unique values.
- `text_missing.csv` — Missing values represented as text.
- `quality_checks.csv` — Duplicate records and unusual values.
- `iqr_screening.csv` — Outlier detection using the IQR method.
- `pearson_correlations.csv` — Correlations between numerical features.
- `cancel_by_MarketSegment.csv` — Cancellation statistics by market segment.
- `cancel_by_DepositType.csv` — Cancellation statistics by deposit type.
- `cancel_by_CustomerType.csv` — Cancellation statistics by customer type.
- `cancel_by_ArrivalDateMonth.csv` — Monthly cancellation statistics.

#### Figures

- `numerical_distributions.png` — Cancellation, lead time, and ADR distributions.
- `categorical_distributions.png` — Distributions of important categorical variables.
- `correlation_heatmap.png` — Correlation heatmap of numerical features.

### 5. Automated Tests

Automated tests are implemented in:

```text
tests/test_eda.py
```

Run the tests from the repository root:

```bash
python -m pytest tests/test_eda.py -v
```

The tests verify:

1. Expected dataset column count.
2. Missing-column detection.
3. Unexpected-column detection.
4. Incorrect-column-order detection.
5. Missing-value calculations.
6. Duplicate-record counting.
7. Data-quality calculations.

### 6. Test Results

The EDA tests were executed using Google Colab.

Environment:
- Python 3.13.18
- pytest 8.4.2

Results:

```text
7 tests collected
7 tests passed
Execution time: 1.86 seconds
```

All seven automated tests passed successfully.

### 7. EDA Report

Detailed EDA findings and interpretations are documented in:

```text
reports/eda.md
```

The report covers:

- Dataset structure and feature types.
- Missing values and duplicate records.
- Unusual values and outliers.
- Lead time and ADR distributions.
- Booking cancellation patterns.
- Correlations between numerical features.
- Recommendations for subsequent preprocessing and modeling.

### 8. Reproducibility

To reproduce the EDA results:

1. Clone the project repository.
2. Install the required dependencies.
3. Place the original `H2.csv` inside `data/`.
4. Run the EDA script.
5. Review the generated results in `outputs/eda/`.
6. Run the automated tests to verify the implementation.

The analysis is reproducible and does not modify the original dataset.