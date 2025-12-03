"""
MLflow integration for the Spam NLP pipeline.

This module:
- Runs one full experiment for a given scenario
- Logs parameters & metrics
- Logs useful artifacts:
  * trained model
  * confusion matrix plot
  * feature importance / vocabulary CSV

It also logs:
- Text preprocessing configuration
- Model hyperparameters (C, solver, etc.)
- Dataset configuration and evaluation strategy
- Cross-validation statistics (mean/std of F1 on training set)
- Training time and feature dimensionality
"""

import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

from pipeline.data_processor import DataProcessor
from pipeline.text_preprocessor import TextPreprocessor
from pipeline.model_trainer import ModelTrainer
from utils.config import (
    DEFAULT_TOKEN_REGEX,
    NB_FEATURES_DEFAULT,
    NB_ITERATIONS_DEFAULT,
)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _get_train_test_for_scenario(dp: DataProcessor, scenario: str):
    """Return train/test splits depending on the scenario."""
    if scenario == "sms_only":
        return dp.train_eval_sms()
    elif scenario == "email_only":
        return dp.train_eval_email()
    elif scenario == "sms_to_email":
        return dp.train_sms_eval_email()
    elif scenario == "combined":
        return dp.train_eval_combined()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def _plot_and_save_confusion_matrix(cm: np.ndarray, out_path: str):
    """
    Create a simple confusion matrix plot and save to disk.

    cm is a 2x2 numpy array with:
        [[tn, fp],
         [fn, tp]]
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ham", "Spam"])
    ax.set_yticklabels(["Ham", "Spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # Add values inside cells
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main MLflow runner
# ----------------------------------------------------------------------
def run_experiment_with_mlflow(
    scenario: str,
    token_pattern: str = DEFAULT_TOKEN_REGEX,
    max_features: int = NB_FEATURES_DEFAULT,
    lowercase: bool = True,
    stop_words=None,
    max_iter: int = NB_ITERATIONS_DEFAULT,
    class_balancing: bool = False,
    experiment_name: str = "SpamNLP_SpamDetection",
    run_name: str | None = None,
):
    """
    Run a single experiment wrapped in an MLflow run and log:
    - parameters
    - metrics (incl. cross-validation F1 mean/std)
    - training time and feature dimensionality
    - model + confusion matrix + feature importance CSV as artifacts

    Parameters
    ----------
    scenario : str
        One of 'sms_only', 'email_only', 'sms_to_email', 'combined'.
    token_pattern : str
        Regex pattern passed to CountVectorizer.
    max_features : int
        Vocabulary size.
    lowercase : bool
        Whether to lowercase text.
    stop_words : any
        Stopword configuration (None or 'english').
    max_iter : int
        Max iterations for Logistic Regression.
    class_balancing : bool
        Whether class balancing was applied (for logging only).
    experiment_name : str
        Name of the MLflow experiment.
    run_name : str | None
        Optional human-readable run name.

    Returns
    -------
    dict
        Dictionary with metrics and confusion matrix counts.
    """
    # Choose or create experiment
    mlflow.set_experiment(experiment_name)

    if run_name is None:
        run_name = (
            f"{scenario}_feat={max_features}_lc={lowercase}"
            f"_sw={stop_words or 'None'}"
        )

    dp = DataProcessor()

    with mlflow.start_run(run_name=run_name):
        # ------------------------------------------------------------------
        # 1) Data: get train/test for the chosen scenario
        # ------------------------------------------------------------------
        train_messages, test_messages, train_labels, test_labels = (
            _get_train_test_for_scenario(dp, scenario)
        )

        # ------------------------------------------------------------------
        # 2) Vectorizer / text preprocessing
        # ------------------------------------------------------------------
        preprocessor = TextPreprocessor(
            token_pattern=token_pattern,
            max_features=max_features,
            lowercase=lowercase,
            stop_words=stop_words,
        )
        X_train = preprocessor.fit_vectorizer(train_messages)
        X_test = preprocessor.transform_messages(test_messages)

        feature_count = int(X_train.shape[1])

        # ------------------------------------------------------------------
        # 3) Model training (with timing)
        # ------------------------------------------------------------------
        trainer = ModelTrainer(max_iter=max_iter)
        trainer.build_model()
        model = trainer.model  # scikit-learn LogisticRegression

        # We can inspect model hyperparameters
        model_params = model.get_params()
        model_C = model_params.get("C", None)
        model_penalty = model_params.get("penalty", None)
        model_solver = model_params.get("solver", None)
        model_random_state = model_params.get("random_state", None)
        model_class_weight = (
            model_params.get("class_weight", None)
            if model_params.get("class_weight", None) is not None
            else "None"
        )

        # Time the training
        t0 = time.time()
        trainer.train(X_train, train_labels)
        training_duration = float(time.time() - t0)

        # ------------------------------------------------------------------
        # 4) Cross-validation statistics (on training set)
        # ------------------------------------------------------------------
        # Simple 3-fold CV on F1 score
        try:
            cv_scores = cross_val_score(
                model,
                X_train,
                train_labels,
                cv=3,
                scoring="f1",
            )
            cv_f1_mean = float(cv_scores.mean())
            cv_f1_std = float(cv_scores.std(ddof=1))
        except Exception:
            # In case something goes wrong, don't crash the run
            cv_f1_mean = None
            cv_f1_std = None

        # ------------------------------------------------------------------
        # 5) Evaluation on test set
        # ------------------------------------------------------------------
        y_true = np.array(test_labels)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, pos_label=1))
        rec = float(recall_score(y_true, y_pred, pos_label=1))
        f1 = float(f1_score(y_true, y_pred, pos_label=1))

        # confusion_matrix returns [[tn, fp], [fn, tp]]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "n_train": int(len(train_labels)),
            "n_test": int(len(test_labels)),
            "training_time_seconds": training_duration,
            "feature_dimensionality": feature_count,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
        }

        # ------------------------------------------------------------------
        # 6) Log parameters (preprocessing + model + dataset + strategy)
        # ------------------------------------------------------------------
        mlflow.log_params(
            {
                # Dataset / scenario
                "dataset_combination": scenario,
                "train_test_split": 0.2,
                "class_balancing": class_balancing,
                # Text preprocessing
                "tokenization_pattern": token_pattern,
                "max_features": max_features,
                "lowercase_conversion": lowercase,
                "stop_words_removal": stop_words if stop_words is not None else "None",
                # Model hyperparameters
                "model_type": "logistic_regression",
                "model_max_iter": max_iter,
                "model_C": model_C,
                "model_penalty": model_penalty,
                "model_solver": model_solver,
                "model_random_state": model_random_state,
                "model_class_weight": model_class_weight,
            }
        )

        # ------------------------------------------------------------------
        # 7) Log metrics
        # ------------------------------------------------------------------
        metrics_to_log = {
            # Core performance metrics
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            # Confusion matrix
            "confusion_matrix_tp": int(tp),
            "confusion_matrix_tn": int(tn),
            "confusion_matrix_fp": int(fp),
            "confusion_matrix_fn": int(fn),
            # Dataset sizes
            "n_train": int(len(train_labels)),
            "n_test": int(len(test_labels)),
            # Training metadata
            "training_time_seconds": training_duration,
            "feature_dimensionality": feature_count,
        }

        # Cross-validation statistics (if available)
        if cv_f1_mean is not None:
            metrics_to_log["cv_f1_mean"] = cv_f1_mean
        if cv_f1_std is not None:
            metrics_to_log["cv_f1_std"] = cv_f1_std

        mlflow.log_metrics(metrics_to_log)

        # ------------------------------------------------------------------
        # 8) Log artifacts (plots, CSV, model)
        # ------------------------------------------------------------------
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # 8.1 – Confusion matrix plot
        cm_path = artifacts_dir / "confusion_matrix.png"
        _plot_and_save_confusion_matrix(cm, str(cm_path))
        mlflow.log_artifact(str(cm_path))

        # 8.2 – Feature importance / vocabulary CSV
        feature_names = preprocessor.vectorizer.get_feature_names_out()
        coefs = model.coef_[0]  # shape (n_features,)

        fi_df = pd.DataFrame(
            {
                "token": feature_names,
                "coefficient": coefs,
                "abs_coefficient": np.abs(coefs),
            }
        ).sort_values("abs_coefficient", ascending=False)

        fi_path = artifacts_dir / "feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))

        # 8.3 – Log the trained model itself
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="spam_classifier",
        )

    return results
