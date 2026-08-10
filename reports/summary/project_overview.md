# Project Overview

* **Project Objective**: To explore customer survey data, isolate key happiness drivers, and build a reproducible predictive model.
* **Core Achievement**: Developed a `RandomForestClassifier` achieving **73.08% accuracy** on the unseen test set, backed by a Streamlit interactive tool using SHAP interpretability.
* **Business Value**: Provides a clear, data-driven roadmap identifying exactly which survey questions dictate customer retention and happiness.

---

# Exploratory Data Analysis (EDA)

## Explaining the Data

* **Dataset Size**: 126 total customer response samples.
* **Rating Metric**: Statements rated on an ordinal scale from 1 to 5 (higher numbers indicate stronger customer alignment).
* **Feature Schema**:
  * **$Y$**: Target attribute — `0` (unhappy), `1` (happy)
  * **$X_1$**: "My order was delivered on time"
  * **$X_2$**: "Contents of my order was as I expected"
  * **$X_3$**: "I ordered everything I wanted to order"
  * **$X_4$**: "I paid a good price for my order"
  * **$X_5$**: "I am satisfied with my courier"
  * **$X_6$**: "The app makes ordering easy for me"

### Visualization & Feature Insights

* **Noise & Overlap ([Figure 1 - Pairgrid Analysis](../figures/pairgrid.png))**:
  * Heavy visual overlap across feature distributions indicates high data noise.
  * Small sample size ($N = 126$) creates a high risk of overfitting.
  * *Mitigation*: Employed classical machine learning models alongside careful hyperparameter regularization to maximize generalizability.
* **Correlation Dynamics ([Figure 2 - Heatmap Analysis](../figures/featureCorrelationHeatmap.png))**:
  * Target predictor $Y$ exhibits the weakest correlation with $X_2$ and $X_4$, and the strongest correlation with $X_1$.
  * Removing low-correlation features ($X_2$, $X_4$) consistently increased overall model performance by eliminating uninformative noise.
  * Strongest feature $X_1$ shares a $\sim 0.4$ correlation with $X_5$ and $X_6$.
  * Threat of model instability due to multicollinearity was mitigated through algorithm selection.
* **Screening Search ([Table 1 - LazyPredict](../figures/LazyPredictResults.png))**:
  * Top-performing models out of 26 tested were consistently tree-based ensembles, which naturally resist multicollinearity.
  * Selected top candidates were passed downstream for cross-validated hyperparameter tuning.

---

# Modeling Methodology

## Use of the `ModelTuner` Object

* **Source File**: `src/utils/tune_hyperparameters.py`
* **Instantiation & Splitting**:
  * Accepts `feature_names`, `predictor_names`, and `filepath`.
  * Ingests dataset and splits into training set (default: 80%) and holdout test set (default: 20%).
* **Optimization & Visualization**:
  * `tune_model(search_space)`: Runs a randomized search with cross-validation on the specified feature set; returns the search estimator object.
  * `display_parameter_fits(search_object, search_space, groupby=None)`: Plots performance across parameter ranges, with optional `groupby` support to analyze hyperparameter interaction effects.
* **Feature Importance & Interpretability**:
  * `shap_analysis(search_space, model_type)`: Conducts global SHAP analysis to plot overall feature importance and sample-level force contributions.
  * `feature_permutation()`: Ranks features using a model-agnostic algorithm based on performance degradation following data permutation.
* **Overfitting Safeguard**:
  * `predict_test_split()`: Scores tuned hyperparameters against the holdout test set **exactly once**.
  * **Strict Lockout**: Once executed, the tuner instance prevents further search or test-scoring calls to eliminate test-set leakage. Iterating requires instantiating a new tuner instance with an updated random seed.
* **Model Persistence**:
  * `save_model(final=False)`: Exports fitted models to `src/models/saved_models/` in `.joblib` format.
  * Passing `final=True` retrains the model on 100% of the dataset (train + test) prior to saving for production deployment.
* **Pipeline Support**:
  * `PipelineTuner`: Shares identical method signatures adapted specifically for scikit-learn `Pipeline` objects with preprocessing steps.

---

# Model Performance and Interpretability

## Tuned Model Results

| Model | Final Test Set Accuracy | Cross-Validation (CV) Score | Features Used |
|---|---|---|---|
| **Random Forest** | **0.7307692307692307** | $0.64 \pm 0.05831$ | $X_1, X_3, X_5$ |
| **XGBoost** | **0.7307692307692307** | $0.62 \pm 0.074833$ | $X_1, X_5$ |
| **ExtraTree** | **0.6923076923076923** | $0.70 \pm 0.094868$ | $X_1, X_3, X_5$ |

## Key Analytical Findings

* **Feature Pruning**:
  * $X_2$ and $X_4$ consistently scored at the bottom of permutation tests across all full-feature candidate models.
  * Pruning $X_2$ and $X_4$ boosted test performance by reducing noise.
* **Multicollinearity Management**:
  * Removing either $X_5$ or $X_6$ yielded the most robust validation scores due to their moderate mutual correlation and collinearity with $X_1$.
  * Dropping *both* $X_5$ and $X_6$ resulted in insufficient information for effective learning.
* **SHAP Behavior Shift**:
  * *Unpruned Models*: SHAP analysis showed predictions over-indexed on only 1–2 dominant features.
  * *Pruned Models ($X_1, X_3, X_5$ remaining)*: SHAP feature importance distributed evenly across all active features, reflecting balanced decision logic.

---

# Streamlit App

* **Automated Model Discovery**: Automatically detects `.joblib` files created by `save_model()` in `src/models/saved_models/`.
* **Metadata Extraction**: Displays stored model metadata directly beneath the user's choice, including the active feature set used during inference.
* **Interactive Inference**: Provides 1–5 range sliders for survey response inputs.
* **Real-time SHAP Explainability**: Dynamic predictions update automatically alongside a SHAP waterfall plot showing exact feature contributions to the outcome (on model-dependent marginal scale).

---

# Strategic Recommendations

* **Survey Streamlining (Remove $X_2, X_4, X_6$)**:
  * Statements regarding order expectations ($X_2$), price perception ($X_4$), and app usability ($X_6$) are poor predictors of happiness among existing active customers.
  * *Action*: Remove $X_2$, $X_4$, and $X_6$ from future survey iterations to reduce user friction without compromising model predictive capability.
* **Operational Focus (Prioritize $X_1, X_3, X_5$)**:
  * Customer happiness is driven primarily by on-time delivery ($X_1$ — largest single predictor), order completeness ($X_3$), and courier satisfaction ($X_5$).
  * *Action*: Focus operational resources directly on logistically reliable fulfillment and courier quality.