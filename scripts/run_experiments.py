"""
Run grid experiments for the Spam NLP pipeline and log everything to MLflow.

We explore:
- Different tokenization patterns
- Different vocabulary sizes (max_features)
- Effect of lowercase normalization (extra runs with lowercase=False)
- For both SMS-only and Email-only scenarios
"""

from pipeline.mlflow_runner import run_experiment_with_mlflow


def run_sms_experiments():
    print("===== Running SMS tokenization experiments (sms_only) =====\n")

    # Grid of token patterns and vocabulary sizes (lowercase=True)
    feature_grid = [1000, 2500, 5000, 10000]

    tokenizers = [
        ("SMS_tok=non_space_feat", r"(\S+)"),
        ("SMS_tok=word_min2_feat", r"([A-Za-z0-9]{2,})"),
        # You can adjust this regex if your "word_with_punct" is slightly different
        ("SMS_tok=word_with_punct_feat", r"([A-Za-z0-9_!?.,:;]{2,})"),
    ]

    for label, pattern in tokenizers:
        for nb_feat in feature_grid:
            run_name = f"{label}={nb_feat}"
            print(f">>> {run_name}")
            run_experiment_with_mlflow(
                scenario="sms_only",
                token_pattern=pattern,
                max_features=nb_feat,
                lowercase=True,
                stop_words=None,
                max_iter=1000,
                class_balancing=False,
                run_name=run_name,
            )

    # ------------------------------------------------------------------
    # Extra: lowercase=False experiments for SMS
    # ------------------------------------------------------------------
    print("\n===== Running SMS lowercase=False experiments (sms_only) =====\n")

    # We pick a couple of representative configurations and flip only lowercase
    sms_lc_false_configs = [
        {
            "run_name": "SMS_lcFalse_non_space_feat=2500",
            "token_pattern": r"(\S+)",
            "max_features": 2500,
        },
        {
            "run_name": "SMS_lcFalse_word_min2_feat=2500",
            "token_pattern": r"([A-Za-z0-9]{2,})",
            "max_features": 2500,
        },
    ]

    for cfg in sms_lc_false_configs:
        print(f">>> {cfg['run_name']}")
        run_experiment_with_mlflow(
            scenario="sms_only",
            token_pattern=cfg["token_pattern"],
            max_features=cfg["max_features"],
            lowercase=False,   
            stop_words=None,
            max_iter=1000,
            class_balancing=False,
            run_name=cfg["run_name"],
        )


def run_email_experiments():
    print("\n===== Running Email tokenization experiments (email_only) =====\n")

    feature_grid = [1000, 2500, 5000, 10000]

    tokenizers = [
        ("EMAIL_tok=non_space_feat", r"(\S+)"),
        ("EMAIL_tok=word_min2_feat", r"([A-Za-z0-9]{2,})"),
        ("EMAIL_tok=word_with_punct_feat", r"([A-Za-z0-9_!?.,:;]{2,})"),
    ]

    # Main grid with lowercase=True
    for label, pattern in tokenizers:
        for nb_feat in feature_grid:
            run_name = f"{label}={nb_feat}"
            print(f">>> {run_name}")
            run_experiment_with_mlflow(
                scenario="email_only",
                token_pattern=pattern,
                max_features=nb_feat,
                lowercase=True,
                stop_words=None,
                max_iter=1000,
                class_balancing=False,
                run_name=run_name,
            )

    # ------------------------------------------------------------------
    # Extra: lowercase=False experiments for Email
    # ------------------------------------------------------------------
    print("\n===== Running Email lowercase=False experiments (email_only) =====\n")

    email_lc_false_configs = [
        {
            # Compare with a strong non_space config
            "run_name": "EMAIL_lcFalse_non_space_feat=10000",
            "token_pattern": r"(\S+)",
            "max_features": 10000,
        },
        {
            # Compare with a strong word_min2 config
            "run_name": "EMAIL_lcFalse_word_min2_feat=5000",
            "token_pattern": r"([A-Za-z0-9]{2,})",
            "max_features": 5000,
        },
    ]

    for cfg in email_lc_false_configs:
        print(f">>> {cfg['run_name']}")
        run_experiment_with_mlflow(
            scenario="email_only",
            token_pattern=cfg["token_pattern"],
            max_features=cfg["max_features"],
            lowercase=False, 
            stop_words=None,
            max_iter=1000,
            class_balancing=False,
            run_name=cfg["run_name"],
        )


def main():
    run_sms_experiments()
    run_email_experiments()
    print("\n All configured experiments have finished.")


if __name__ == "__main__":
    main()
