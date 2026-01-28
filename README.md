# Practical Application III — Bank Marketing (prompt_III.ipynb)

This repository contains a Jupyter Notebook `prompt_III.ipynb` that performs exploratory data analysis and classification experiments on the UCI Bank Marketing dataset. The notebook guides you from data loading and EDA through basic model training and hyperparameter search.

## Contents
- `prompt_III.ipynb` — main notebook (EDA, feature engineering, baseline models, GridSearchCV tuning)
- `data/` — CSV files required by the notebook (e.g., `bank-additional-full.csv`)

## Dependencies
Install the required Python packages (recommended inside a virtual environment):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyterlab nbconvert
```

Alternatively, using `conda`:

```bash
conda create -n bankenv python=3.11 -y
conda activate bankenv
pip install pandas numpy matplotlib seaborn plotly scikit-learn jupyterlab nbconvert
```

## How to run the notebook
1. Activate your environment (virtualenv or conda environment above).
2. Launch Jupyter and open the notebook:

```bash
jupyter notebook prompt_III.ipynb
# or
jupyter lab prompt_III.ipynb
```

3. Recommended execution order:
   - Run the imports/setup cell.
   - Run the data loading cell: `df = pd.read_csv('data/bank-additional-full.csv', sep=';')`.
   - Execute EDA and plotting cells to inspect the data.
   - Run model training and hyperparameter search cells when ready. Note: the GridSearch cell can be time-consuming.

## Quick verification (headless)
To execute the notebook non-interactively and save outputs in-place:

```bash
jupyter nbconvert --to notebook --execute prompt_III.ipynb --ExecutePreprocessor.timeout=1200 --inplace
```

Increase `--ExecutePreprocessor.timeout` if long-running cells (e.g., GridSearch) need more time.

## Tips
- To speed verification of the GridSearch cell, reduce parameter grid sizes or `cv` folds.
- If you run into environment/kernal issues with `nbconvert`, open the notebook interactively and ensure the kernel matches the Python environment where dependencies are installed.

## Findings
- **Dataset / target handling:** The notebook uses the UCI Bank Marketing dataset. The target column has different names in some sources (`deposit`, `Deposit`, or `y`). The notebook now detects the name automatically and creates a safe `deposit='unknown'` fallback if none is present to avoid runtime errors.
- **EDA highlights:** Visualizations summarize common `job` and `marital` categories; counts were computed using `age` as a consistent counting column in a few plots to maintain behavior across datasets. Plotly-based charts were standardized (explicit small DataFrames + `fig.show()`) to prevent mismatched column errors.
- **Model / GridSearch summary:** GridSearchCV was used for four classifiers (Logistic Regression, Decision Tree, KNN, SVC). Representative best cross-validation accuracies observed during a run:
   - Logistic Regression: **0.8969** (best params: `{'clf__C': 0.01, 'clf__penalty': 'l2', 'clf__solver': 'liblinear'}`)
   - Decision Tree: **0.8972** (best params: `{'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}`)
   - KNN: **0.8965** (best params: `{'clf__metric': 'minkowski', 'clf__n_neighbors': 11, 'clf__weights': 'uniform'}`)
   - SVC: **0.8970** (best params: `{'clf__C': 1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}`)
   - Best overall (observed): **Decision Tree — 0.8972**
- **Operational note:** GridSearch in the notebook is configured to run with `n_jobs=1` to avoid platform-specific `ChildProcessError` issues; if you have a stable multiprocessing environment, you can increase `n_jobs` for speed.
  
## Next steps and recommendations
- Identify best features using Random forester
- If we have information about customer account balance, it would help to predict subscription based on balance held (assumption: customer with more balance tend to subsribe term deposits)
- Deep Learning (ANN)

## Next suggestions
- Run the notebook top-to-bottom to confirm no remaining issues.
- Optionally, add evaluation cells to persist the best model and plot confusion matrices or ROC curves.

Generated on: 2026-01-27
Notebook path: `prompt_III.ipynb`
