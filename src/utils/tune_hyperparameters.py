import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sklearn
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# saving models
import joblib
import datetime

# imports for type hints
from scipy.stats._distn_infrastructure import rv_discrete_frozen, rv_continuous_frozen
from scipy.stats import uniform, randint
from typing import TypeAlias

# Type Aliases for clarity
HyperParameterValue: TypeAlias = list | tuple | rv_discrete_frozen | rv_continuous_frozen

class ModelTuner():
    def __init__(self, model, filename: str, predictor: str, featureNames: list[str], **kwargs):
        """
        model: the instance of the sklearn api model that will be tuned
        filename: literal string of relative path
        predictor: name of feature in the dataframe that will be predicted
        featureNames: list of feature names in the dataframe that will be predicted
        """

        self.last_search = None
        self.saved_params = {}
        self.saved_score = 0

        # initialization of random seed to be consitent
        self.random_state = kwargs.get("random_state", 0)
        np.random.seed(self.random_state)

        self._predicting = False
        
        self._max_visable_features = kwargs["max_visable_features"] if "max_viable_features" in kwargs else 6
        self._cv = kwargs["cv"] if "cv" in kwargs else None


        self._model = model

        test_size = kwargs["test_size"] if "test_size" in kwargs else 0.2
        self.df = pd.read_csv(filename)
        self._predictor = predictor
        self._featureNames = featureNames
        
        # format dataframe
        X = self.df.copy()
        self.Y = X[self._predictor]
        self.X = X.drop([self._predictor], axis=1)

        # Split into train and hold-out test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=self.random_state, stratify=self.Y)

    def get_predictors(self):
        return self._predictor
    def set_predictors(self, newPredictor: str):
        if self._predicting:
            raise Exception("Tuner can no longer change features once the test set has been looked at\nPlease make new tuner with a new random state")
        self._predictor = newPredictor
        
    def get_features(self):
        return self._featureNames
    def set_features(self, newFeatureList: list[str]):
        if self._predicting:
            raise Exception("Tuner can no longer change features once the test set has been looked at\nPlease make new tuner with a new random state")
        self._featureNames = newFeatureList

    def change_model(self, newModel):
        """
        pass in a new sklearn api model to tune
        """
        if self._predicting:
            raise Exception("Tuner can no longer change models once the test set has been looked at\nPlease make new tuner with a new random state")
        self._model = newModel
            
        
    def tune_model(self, hyperParameterGrid:dict[str, HyperParameterValue], **kwargs):

        if self._predicting:
            raise Exception("Tuner can no longer tune once the test set has been looked at\nPlease make new tuner with a new random state")

        verbose = kwargs["verbose"] if "verbose" in kwargs else 10

        n_iter = kwargs["n_iter"] if "n_iter" in kwargs else 500
        scoring = kwargs["scoring"] if "scoring" in kwargs else None
        n_jobs = kwargs["n_jobs"] if "n_jobs" in kwargs else 1


        search = RandomizedSearchCV(self._model, hyperParameterGrid, 
                                    scoring= scoring, n_iter= n_iter, 
                                    n_jobs= n_jobs, verbose=verbose, 
                                    random_state=self.random_state, cv = self._cv, 
                                    return_train_score=True,
                                    )

        search.fit(self.X_train[self._featureNames], self.y_train)

        print("|I|Randomized Search Best Score: ",search.best_score_)
        print("|I|Randomized Search Best Estimator: ",search.best_estimator_)

        self.display_parameter_fits(search, hyperParameterGrid)

        self.last_search = search

        return search

    def display_parameter_fits(self, search, hyperParameterGrid: dict[str, HyperParameterValue], groupby: str | None = None, **kwargs):
        
        searchResults = pd.DataFrame(search.cv_results_)

        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(1+len(hyperParameterGrid)//3,3, figure=fig)

        # graphing

        if groupby is None:
            for i, t in enumerate(hyperParameterGrid.items()):
                parameter, value = t
                ax = fig.add_subplot(gs[i//3, i%3])
                if type(value) in (list, tuple):
                    sns.violinplot(data=searchResults, y = "mean_test_score", x = "param_" + parameter, ax=ax)
                    ax.set_title(parameter)

                if type(value) == type(uniform(loc=1,scale=1,)) or type(value) == type(randint(1,1000)):
                    sns.lineplot(data=searchResults, y = "mean_test_score", x = "param_" + parameter, ax=ax)
                    ax.set_title(parameter)
        else:
            if not groupby in hyperParameterGrid:
                raise ValueError(f"Failed to display all parameters in: {groupby}\nSome parameters were not in the hyperParameterGrid")
            
            # Read logarithmic flag from kwargs (defaults to False if not provided)
            is_log_scale = kwargs.get('log_scale', False)

            for i, t in enumerate(hyperParameterGrid.items()):
                parameter, value = t
                ax = fig.add_subplot(gs[i//3, i%3])
                
                group_param = "param_" + groupby

                legend_title = groupby
                
                try:
                    unique_count = searchResults[group_param].nunique()
                    num_bins = min(3, unique_count) if unique_count > 0 else 3
                    
                    if is_log_scale and searchResults[group_param].min() > 0:
                        # Convert data to log10 space to get equal geometric intervals
                        log_data = np.log10(searchResults[group_param])
                        # Cut the log data without string labels to retain exact mathematical boundaries
                        log_bins = pd.cut(log_data, bins=num_bins)
                        
                        # Reconstruct the original scale boundaries for clean legend readability
                        searchResults[legend_title] = log_bins.apply(
                            lambda interval: f"({10**interval.left:.2e}, {10**interval.right:.2e}]"
                        )
                    else:
                        # Standard linear spacing. Omitting 'labels' prints exact mathematical ranges
                        # Formatting to 2 decimal places / engineering notation keeps legends clean
                        raw_bins = pd.cut(searchResults[group_param], bins=num_bins)
                        searchResults[legend_title] = raw_bins.apply(
                            lambda interval: f"({interval.left:.2e}, {interval.right:.2e}]"
                        )
                        
                except (ValueError, TypeError) as e:
                    print(f"Encountered error: {e}\nTrying non-numeric method")
                    # Fallback for string-based or non-numeric categories
                    searchResults[legend_title] = searchResults[group_param]

                is_scipy_dist = hasattr(value, 'dist') and hasattr(value, 'rvs')

                is_categorical_list = isinstance(value, (list, tuple)) and any(isinstance(x, str) for x in value)

                if is_categorical_list or isinstance(value, str):
                    sns.violinplot(
                        data=searchResults, 
                        y="mean_test_score", 
                        x="param_" + parameter, 
                        hue=legend_title, 
                        ax=ax
                    )
                    ax.set_title(parameter)

                elif is_scipy_dist or isinstance(value, (list, tuple)): 
                    sns.lineplot(
                        data=searchResults, 
                        y="mean_test_score", 
                        x="param_" + parameter, 
                        hue=legend_title, 
                        ax=ax
                    )
                    ax.set_title(parameter)
            


        # title making
        if len(self._featureNames) <= self._max_visable_features:
            features_str = ", ".join(self._featureNames)
        else:
            features_str = ", ".join(self._featureNames[:self._max_visable_features]) + f" (+{len(self._featureNames) - self._max_visable_features} more)"

        # Extract performance metrics dynamically
        best_score = search.best_score_


        best_index = search.best_index_
        best_train_score = search.cv_results_['mean_train_score'][best_index]

        title_text = (
            f"Fits with Features: [{features_str}]\n"
            f"Best CV Score: {best_score:.4f} | Best Average Train Split Score: {best_train_score:.4f}"
        )

        fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)  # Prevents subplots from overlapping the super title
        plt.title(f"Fits with Features:\n[{features_str}]")
        plt.show()

    def feature_permutation(self, params):
        
        params = params.copy()
    
        # update model
        instantiatedModel = self._model.set_params(**params)
        instantiatedModel.fit(self.X_train[self._featureNames], self.y_train)

        result = permutation_importance(
            self._model, self.X_test[self._featureNames], self.y_test, n_repeats=10, scoring='accuracy', random_state=self.random_state, n_jobs=1
        )
        
        print("\n\nFeature Permutation Importance\n______________________")
        for i in result.importances_mean.argsort()[::-1]:
            print(f"{self.X[self._featureNames].columns[i]}: {result.importances_mean[i]:.3f} +/- {result.importances_std[i]:.3f}")
        print("\n______________________")

    def shap_analysis(self, params: dict, model_type: str = "tree"):
    
        # update model
        instantiatedModel = self._model.set_params(**params)
        instantiatedModel.fit(self.X_train[self._featureNames], self.y_train)

        # Subset testing features
        X_test_subset = self.X_test[self._featureNames]

        match model_type.lower().strip():
            case "tree":
                # For Tree-based models (RandomForest, XGBoost, Decision Trees, etc.)
                explainer = shap.TreeExplainer(self._model)
            case "linear":
                # For Linear models (LogisticRegression, Ridge, LinearRegression, etc.)
                # Requires a background dataset matrix to compute feature correlations
                explainer = shap.LinearExplainer(self._model, X_test_subset)
            case "kernel" | "generic":
                # Universal fallback (SVM, Neural Networks, K-NN, etc.)
                # Summarizes background matrix to 10 clusters to maintain quick compute runtime
                background_summary = shap.kmeans(X_test_subset, 10)
                # Use predict_proba for classifiers; if handling regression, change to self._model.predict
                prediction_func = (
                    self._model.predict_proba 
                    if hasattr(self._model, "predict_proba") 
                    else self._model.predict
                )
                explainer = shap.KernelExplainer(prediction_func, background_summary)
            case _:
                raise ValueError(
                    f"Unsupported type: '{model_type}'. Choose from 'tree', 'linear', or 'kernel'."
                )


        shap_values = explainer(X_test_subset)

        if len(self._featureNames) <= self._max_visable_features:
            features_str = ", ".join(self._featureNames)
        else:
            features_str = ", ".join(self._featureNames[:self._max_visable_features]) + f" (+{len(self._featureNames) - self._max_visable_features} more)"

        # plotting the shap values
        # Modern SHAP yields a 3D shape (samples, features, classes) for multi-class
        if len(shap_values.shape) == 3:
            num_classes = shap_values.shape[2]
            for i in range(num_classes):
                shap.summary_plot(shap_values[:, :, i], X_test_subset, show=False)
                plt.gca().set_title(f"SHAP Class {i} Analysis\nFeatures: [{features_str}]", fontsize=12, pad=20)
                plt.gcf().tight_layout()
                plt.show()
        else:
            shap.summary_plot(shap_values, X_test_subset, show=False)
            plt.gca().set_title(f"SHAP Global Analysis\nFeatures: [{features_str}]", fontsize=12, pad=20)
            plt.gcf().tight_layout()
            plt.show()

    def predict_test_split(self, test_Params):
        if self._predicting:
            raise Exception("Tuner has already predicted on the test set")

        self._predicting = True
        
        instantiatedModel = self._model.set_params(**test_Params)
        instantiatedModel.fit(self.X_train[self._featureNames], self.y_train)
        score = instantiatedModel.score(self.X_test[self._featureNames], self.y_test)

        self.saved_params = test_Params
        self.saved_score = score

        return score

    def save_model(self, filename:str, hyperparams: dict, final:bool = False, notes:str=""):
        """
        filename: the base filename without an extension
        final: If it is the final save for a model (will be trained with all data)
        """

        if final:
            trainingData = self.X[self._featureNames]
            predictors = self.Y
            try:
                score = self.predict_test_split(hyperparams)
            except:
                score = self.saved_score
                
        else:
            trainingData = self.X_train[self._featureNames]
            predictors = self.y_train
            score = None

        self._model.set_params(**hyperparams)
        self._model.fit(trainingData, predictors)


        metadataDict = {
            "features": self._featureNames, 
            "predictors":self._predictor,
            "type": type(self._model),
            "notes": notes,
            "datatime": datetime.datetime.now(),
            "CV_SD": pd.DataFrame(self.last_search.cv_results_).loc[self.last_search.best_index_]["std_test_score"],
            "score": score,
            }
        
        joblib.dump(metadataDict, filename + "_metadata.joblib")
        joblib.dump(self._model, filename + ".joblib")


class PipelineTuner(ModelTuner):
    def __init__(self, pipeline: Pipeline, filename: str, predictor: str, featureNames: list[str], **kwargs):
        super().__init__(pipeline, filename, predictor, featureNames, **kwargs)

    def shap_analysis(self, params: dict[str, HyperParameterValue], model_type: str = "tree"):
        
        self._model.set_params(**params)

        self._model.fit(self.X_train[self._featureNames], self.y_train)

        X_test_transformed = self._model[:-1].transform(self.X_test[self._featureNames])
        transformed_names = self._model[:-1].get_feature_names_out()

        # Convert back to a DataFrame to keep your feature names in the plots
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_names)
        
        if len(transformed_names) <= self._max_visable_features:
            features_str = ", ".join(transformed_names)
        else:
            features_str = ", ".join(transformed_names[:self._max_visable_features]) + f" (+{len(transformed_names) - self._max_visable_features} more)"

        # 5. Extract only the fitted classifier step
        classifier = self._model.named_steps['classifier']

        # 6. Match case to determine the SHAP explainer type
        match model_type.lower().strip():
            case "tree":
                # For Tree-based models (RandomForest, XGBoost, LightGBM, etc.)
                explainer = shap.TreeExplainer(classifier)
            case "linear":
                # For Linear models (LogisticRegression, Ridge, Lasso, etc.)
                # Needs training data background to estimate feature correlations
                explainer = shap.LinearExplainer(classifier, X_test_transformed_df)
            case "kernel" | "generic":
                # Model-agnostic fallback (SVC, Neural Networks, K-NN, etc.)
                # Warning: Can be slow. Using a background sample dataset optimizes speed.
                background_summary = shap.kmeans(X_test_transformed_df, 10)
                explainer = shap.KernelExplainer(classifier.predict_proba, background_summary)
            case _:
                raise ValueError(
                    f"Unsupported model_type: '{model_type}'. Choose from 'tree', 'linear', or 'kernel'."
                )

        # 7. Generate SHAP values
        shap_values = explainer(X_test_transformed_df)

        # 8. Dynamically plot based on classification shapes
        if len(shap_values.shape) == 3:
            num_classes = shap_values.shape[2]  # <-- FIX 1: Get the integer count of classes
            for i in range(num_classes):
                # FIX 2: Use .values for correct slicing of modern SHAP Explanation objects
                shap.summary_plot(shap_values.values[:, :, i], X_test_transformed_df, show=False)
                plt.gca().set_title(f"SHAP Class {i} Analysis\nFeatures: [{features_str}]", fontsize=12, pad=20)
                plt.gcf().tight_layout()
                plt.show()
        else:
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
            plt.gca().set_title(f"SHAP Global Analysis\nFeatures: [{features_str}]", fontsize=12, pad=20)
            plt.gcf().tight_layout()
            plt.show()
