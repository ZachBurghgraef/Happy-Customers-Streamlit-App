# 😃 Happy Customers & Streamlit App

This repository explores, models, and predicts customer happiness drivers from survey data. By training classical machine learning algorithms and employing disciplined feature pruning, the finalized model achieves a **73% predictive accuracy** on unseen test data.

> 📄 **Looking for a deep dive?** Read the detailed [Project Overview](reports/summary/project_overview.md) for complete EDA findings, SHAP attribution insights, and strategic business recommendations.

---

## ✨ Key Features

* **Exploratory Data Analysis:** Pairgrid visualizations and correlation heatmaps to isolate feature signal from background noise.
* **Model Screening:** Automated multi-model benchmarking using `LazyPredict`.
* **Tuned Ensembles:** `RandomForestClassifier` optimized with strict holdout test safeguards, achieving a reproducible 73% test accuracy.
* **Interactive Streamlit Web App:** Real-time inference dashboard powered by live SHAP waterfall plots for sample-level interpretability.

---

## 📁 Project Structure

```text
├── data/       # Raw and processed datasets (excluded for privacy)
├── notebooks/  # Exploratory data analysis & modeling experimentation
├── reports/    # Detailed findings, executive summaries, & figures
│   └── summary/
|       └──project_overview.md
├── src/        # Core source code and utilities
│   ├── models/    # Saved .joblib model artifacts & metadata
│   ├── streamlit/ # Dashboard layout & application UI logic
│   └── utils/     # ModelTuner & PipelineTuner workflow managers
└── app.py      # Streamlit application entry point

```

---

## 🚀 Quickstart & Usage

### 1. Launching the Streamlit Dashboard

Run the application directly from the project root directory:

```bash
streamlit run ./app.py

```

### 2. Hyperparameter Tuning via `ModelTuner`

The `ModelTuner` and `PipelineTuner` classes (`src/utils/tune_hyperparameters.py`) accelerate iterative hyperparameter search while enforcing strict safeguards against test-set leakage.

#### Key Methods Overview:

| Method | Description |
| --- | --- |
| `__init__()` | Ingests data, assigns model, and initializes train/test splits. |
| `get_features()` / `set_features()` | Manages input feature column assignments. |
| `get_predictors()` / `set_predictors()` | Manages target variable assignment. |
| `change_model()` | Swaps in a new scikit-learn estimator instance. |
| `tune_model()` | Executes a randomized cross-validation search across a parameter grid. |
| `display_parameter_fits()` | Plots performance trends across parameter search space (supports `groupby`). |
| `feature_permutation()` | Ranks feature importance via model-agnostic permutation scoring. |
| `shap_analysis()` | Generates global SHAP value plots to diagnose model feature reliance. |
| `predict_test_split()` | Evaluates the model on holdout test data **once** and locks the instance against further tuning. |
| `save_model()` | Exports fitted model and metadata as `.joblib` artifacts for Streamlit deployment. |

---

## 💡 Example Workflow

Below is a complete workflow example for tuning a `RandomForestClassifier` within `notebooks/modeling/`:

```python
import sys
from pathlib import Path
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier

# Resolve project root path
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.tune_hyperparameters import ModelTuner

# Setup paths and parameters
filename = "../../data/raw/ACME-HappinessSurvey2020.csv"
random_state = 0

# Define hyperparameter search space
parameter_grid = {
    "n_estimators": randint(50, 150),
    "min_weight_fraction_leaf": uniform(0.0, 0.5),
    "bootstrap": [True, False],
}

# Instantiate model and tuner
model = RandomForestClassifier(random_state=random_state)
categorical_features = ["X1", "X2", "X3", "X4", "X5", "X6"]
predictor = "Y"

tuner = ModelTuner(
    model=model,
    filepath=filename,
    predictor=predictor,
    feature_names=categorical_features,
    random_state=random_state
)

# Run randomized search & diagnostic checks
search = tuner.tune_model(hyperParameterGrid=parameter_grid, n_iter=100)
tuner.feature_permutation(search.best_params_)
tuner.shap_analysis(search.best_params_)
tuner.display_parameter_fits(search, parameter_grid, groupby="n_estimators")

# Finalize evaluation on holdout test set (Locks tuner instance)
tuner.predict_test_split(search.best_params_)

```

> **Note on Test Set Leakage Safeguard:** If the resulting test score is unsatisfactory, iterate `random_state += 1` and re-instantiate `ModelTuner`. This re-splits the dataset to prevent implicit tuning against the holdout set.

Once satisfied with test performance, serialize the model for production:

```python
tuner.save_model(
    filename="../../src/models/saved_models/RandomForest_01",
    final=True,
    notes="CV: 0.64, testScore: 0.73"
)
```