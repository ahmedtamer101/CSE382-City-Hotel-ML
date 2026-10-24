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

