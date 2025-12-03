"""
Run repeated NLP spam experiments to estimate statistical significance.

Goal:
- For SMS and Email, compare:
    * a baseline preprocessing configuration
    * a candidate "best" configuration
- Repeat each configuration across several random train/test splits
- Report mean ± std for accuracy, precision, recall, and F1

Usage:
    uv run python scripts/run_significance_experiments.py
"""

from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from pipeline.data_processor import DataProcessor
from pipeline.text_preprocessor import TextPreprocessor
from pipeline.model_trainer import ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import TRAIN_TEST_SPLIT_SIZE, MESSAGE_COL, LABEL_COL


# ----------------------------------------------------------------------
# Core repeated experiment logic (generic)
# ----------------------------------------------------------------------
def _run_repeated_experiment_generic(
    messages,
    labels,
    config: Dict,
    seeds: List[int],
    label: str,
):
    """
    Generic repeated experiment runner for one dataset (SMS or Email).

    Parameters
    ----------
    messages : pandas.Series
        All messages from the dataset.
    labels : pandas.Series
        Corresponding labels.
    config : dict
        Contains token_pattern, max_features, lowercase, stop_words, max_iter.
    seeds : list of int
        Random seeds used for train/test split.
    label : str
        Name of the dataset ("SMS" or "EMAIL") for printing.

    Returns
    -------
    dict
        Mean and std for each metric (accuracy, precision, recall, f1).
    """
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    print(f"\n=== {label} repeated experiment ===")
    print("Config:", config)
    print("Seeds:", seeds)

    for seed in seeds:
        print(f"\n--- {label} | Seed = {seed} ---")

        # 1) New random split for each seed
        train_m, test_m, train_y, test_y = train_test_split(
            messages,
            labels,
            test_size=TRAIN_TEST_SPLIT_SIZE,
            random_state=seed,
            stratify=labels,
        )

        # 2) Text preprocessing
        preprocessor = TextPreprocessor(
            token_pattern=config["token_pattern"],
            max_features=config["max_features"],
            lowercase=config["lowercase"],
            stop_words=config["stop_words"],
        )
        X_train = preprocessor.fit_vectorizer(train_m)
        X_test = preprocessor.transform_messages(test_m)

        # 3) Model training
        trainer = ModelTrainer(max_iter=config["max_iter"])
        trainer.build_model()
        trainer.train(X_train, train_y)

        # 4) Evaluation
        evaluator = Evaluator(pos_label=1)
        pred = trainer.predict(X_test)

        acc = evaluator.accuracy(test_y, pred)
        prec = evaluator.precision(test_y, pred)
        rec = evaluator.recall(test_y, pred)
        f1 = evaluator.f1_score(test_y, pred)

        print(
            f"accuracy={acc:.4f}, precision={prec:.4f}, "
            f"recall={rec:.4f}, f1={f1:.4f}"
        )

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # 5) Aggregate statistics
    stats = {
        "acc_mean": float(np.mean(accuracies)),
        "acc_std": float(np.std(accuracies, ddof=1)),
        "prec_mean": float(np.mean(precisions)),
        "prec_std": float(np.std(precisions, ddof=1)),
        "rec_mean": float(np.mean(recalls)),
        "rec_std": float(np.std(recalls, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)),
        "n_runs": len(seeds),
    }

    print(f"\n>>> {label} configuration statistics over {len(seeds)} runs")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return stats


# ----------------------------------------------------------------------
# SMS and Email specific helpers
# ----------------------------------------------------------------------
def run_repeated_experiment_sms(config: Dict, seeds: List[int]) -> Dict[str, float]:
    """Run the same configuration several times on the SMS dataset."""
    dp = DataProcessor()
    sms_df = dp.load_sms()
    messages = sms_df[MESSAGE_COL]
    labels = sms_df[LABEL_COL]
    return _run_repeated_experiment_generic(messages, labels, config, seeds, "SMS")


def run_repeated_experiment_email(config: Dict, seeds: List[int]) -> Dict[str, float]:
    """Run the same configuration several times on the Email dataset."""
    dp = DataProcessor()
    email_df = dp.load_email()
    messages = email_df[MESSAGE_COL]
    labels = email_df[LABEL_COL]
    return _run_repeated_experiment_generic(messages, labels, config, seeds, "EMAIL")


# ----------------------------------------------------------------------
# Main: define configs and compare baseline vs candidate
# ----------------------------------------------------------------------
def main():
    # Number of repetitions (seeds)
    seeds = [1, 2, 3, 4, 5]  # You can increase to 10 if you have time

    # --------------------------------------------------
    # 1) SMS configurations
    # --------------------------------------------------
    # TODO: adjust these based on your MLflow results
    config_sms_baseline = {
        "token_pattern": r"(\S+)",             # simple tokenization
        "max_features": 2500,
        "lowercase": True,
        "stop_words": None,
        "max_iter": 1000,
    }

    config_sms_candidate = {
    "token_pattern": r"([A-Za-z0-9]{2,})",
    "max_features": 2500,
    "lowercase": True,
    "stop_words": None,
    "max_iter": 1000,
    }


    print("\n############################")
    print("# SMS: Baseline configuration")
    print("############################")
    stats_sms_baseline = run_repeated_experiment_sms(config_sms_baseline, seeds)

    print("\n############################")
    print("# SMS: Candidate configuration")
    print("############################")
    stats_sms_candidate = run_repeated_experiment_sms(config_sms_candidate, seeds)

    print("\n===== SMS COMPARISON (mean F1) =====")
    print(
        f"Baseline  F1: {stats_sms_baseline['f1_mean']:.4f} "
        f"± {stats_sms_baseline['f1_std']:.4f}"
    )
    print(
        f"Candidate F1: {stats_sms_candidate['f1_mean']:.4f} "
        f"± {stats_sms_candidate['f1_std']:.4f}"
    )

    # --------------------------------------------------
    # 2) Email configurations
    # --------------------------------------------------
    # TODO: adjust these based on your MLflow results for email
    config_email_baseline = {
        "token_pattern": r"(\S+)",
        "max_features": 2500,
        "lowercase": True,
        "stop_words": None,
        "max_iter": 1000,
    }

    config_email_candidate = {
    "token_pattern": r"(\S+)",               # non_space
    "max_features": 10000,                   # EMAIL_tok=non_space_feat=10000 (your best)
    "lowercase": True,
    "stop_words": None,
    "max_iter": 1000,
    }

    print("\n############################")
    print("# EMAIL: Baseline configuration")
    print("############################")
    stats_email_baseline = run_repeated_experiment_email(config_email_baseline, seeds)

    print("\n############################")
    print("# EMAIL: Candidate configuration")
    print("############################")
    stats_email_candidate = run_repeated_experiment_email(config_email_candidate, seeds)

    print("\n===== EMAIL COMPARISON (mean F1) =====")
    print(
        f"Baseline  F1: {stats_email_baseline['f1_mean']:.4f} "
        f"± {stats_email_baseline['f1_std']:.4f}"
    )
    print(
        f"Candidate F1: {stats_email_candidate['f1_mean']:.4f} "
        f"± {stats_email_candidate['f1_std']:.4f}"
    )


if __name__ == "__main__":
    main()
